/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- QLinearConv.cpp - Lowering QLinearConv Op -----------------===//
//
// This file lowers the ONNX QLinearConv Operator to Krnl dialect by
// decomposing it into an im2col unfold (Slice/Concat/Reshape/Transpose) plus
// the same int8x8->int32 accumulate + rescale/saturate pipeline already used
// by ONNXQLinearMatMulOpLowering (QLinearMatMul.cpp), so the actual quantized
// arithmetic reuses that already-correct kernel instead of a new one.
//
// v1 scope (see project plan): 4D NCHW tensors, auto_pad == NOTSET, and all
// shape-affecting attributes/dims statically known. group >= 1 is supported
// (including depthwise, group == C == CO): each group is unfolded and
// matmul'd independently (slicing X's input channels and W/bias/per-channel
// scale-zero-point's output channels to that group's range), then the
// per-group results are concatenated back along the output-channel axis.
// This means `group` separate im2col+matmul sequences get unrolled into the
// IR at compile time -- fine for this research/tuning tool's purposes, but
// something to be aware of for models with many groups (e.g. a depthwise
// layer with a few hundred channels).
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Returns std::nullopt if the (optional) attribute is absent.
std::optional<SmallVector<int64_t, 4>> getOptionalIntArray(
    std::optional<ArrayAttr> attr) {
  if (!attr.has_value())
    return std::nullopt;
  SmallVector<int64_t, 4> vals;
  ArrayAttrIntVals(*attr, vals);
  return vals;
}

// Reshape a rank-0 or rank-1-size-1 tensor/memref Value to a true rank-0
// (scalar) tensor, as required by ONNXPadOp's constant_value operand. Uses
// ShapedType (not RankedTensorType) because at this point in ONNXToKrnl,
// adaptor-remapped operands like this one are already memref-typed (see the
// comment on the xType/wType extraction below). Returns a null Value if the
// input isn't one of those shapes.
Value toScalarValue(OnnxBuilder &create, Value v) {
  auto ty = mlir::dyn_cast<ShapedType>(v.getType());
  if (!ty || !ty.hasRank())
    return nullptr;
  if (ty.getRank() == 0)
    return v;
  if (ty.getRank() == 1 && ty.getShape()[0] == 1) {
    auto scalarTy = RankedTensorType::get({}, ty.getElementType());
    return create.reshape(scalarTy, v, create.constantInt64({}));
  }
  return nullptr;
}

// Slice a per-output-channel Value (rank-1, shape [CO]) down to the range
// [start, start+count); a rank-0 (scalar, i.e. per-tensor) Value is
// returned unchanged, since a single shared value applies identically to
// every group. Used for w_scale/w_zero_point/y_scale/y_zero_point (which
// the spec allows to be either per-tensor or per-output-channel) and for
// bias (always per-output-channel when present).
Value sliceChannelRangeOrScalar(
    OnnxBuilder &create, Value v, int64_t start, int64_t count) {
  auto ty = mlir::dyn_cast<ShapedType>(v.getType());
  if (!ty || !ty.hasRank())
    return v;
  if (ty.getRank() == 0)
    return v;
  auto outTy = RankedTensorType::get({count}, ty.getElementType());
  return create.slice(outTy, v, create.constantInt64({start}),
      create.constantInt64({start + count}), create.constantInt64({0}),
      create.constantInt64({1}));
}

} // namespace

struct ONNXQLinearConvOpLowering : public OpConversionPattern<ONNXQLinearConvOp> {
  ONNXQLinearConvOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXQLinearConvOp qlcOp,
      ONNXQLinearConvOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = qlcOp.getOperation();
    Location loc = ONNXLoc<ONNXQLinearConvOp>(op);
    OnnxBuilder create(rewriter, loc);

    Value X = adaptor.getX();
    Value xScale = adaptor.getXScale();
    Value xZeroPoint = adaptor.getXZeroPoint();
    Value W = adaptor.getW();
    Value wScale = adaptor.getWScale();
    Value wZeroPoint = adaptor.getWZeroPoint();
    Value yScale = adaptor.getYScale();
    Value yZeroPoint = adaptor.getYZeroPoint();
    Value Bias = adaptor.getB();
    bool hasBias = !mlir::isa<NoneType>(Bias.getType());

    if (qlcOp.getAutoPad() != "NOTSET")
      return rewriter.notifyMatchFailure(
          op, "QLinearConv->Krnl lowering only supports auto_pad=NOTSET");

    // Use the original (pre-type-conversion) op's operand types for static
    // shape queries: by this point in ONNXToKrnl, adaptor.getX()/getW() are
    // already memref-typed (an intermediate mixed memref/tensor IR state
    // that OnnxBuilder tolerates, matching ONNXQLinearMatMulOpLowering's use
    // of adaptor operands directly), but qlcOp.getX()/getW() still carry the
    // op's original, always-ranked-tensor ODS types.
    auto xType = mlir::dyn_cast<RankedTensorType>(qlcOp.getX().getType());
    auto wType = mlir::dyn_cast<RankedTensorType>(qlcOp.getW().getType());
    if (!xType || !wType || xType.getRank() != 4 || wType.getRank() != 4)
      return rewriter.notifyMatchFailure(op,
          "QLinearConv->Krnl lowering only supports 4D NCHW X/W tensors");
    if (!xType.hasStaticShape() || !wType.hasStaticShape())
      return rewriter.notifyMatchFailure(op,
          "QLinearConv->Krnl lowering requires statically shaped X/W");

    Type xElemType = xType.getElementType();
    ArrayRef<int64_t> xShape = xType.getShape(); // [N, C, H, Win]
    ArrayRef<int64_t> wShape = wType.getShape(); // [CO, C/group, KH, KW]
    int64_t N = xShape[0], C = xShape[1], H = xShape[2], Win = xShape[3];
    int64_t CO = wShape[0], KH = wShape[2], KW = wShape[3];

    int64_t group = qlcOp.getGroup();
    if (group < 1 || C % group != 0 || CO % group != 0)
      return rewriter.notifyMatchFailure(op,
          "QLinearConv->Krnl lowering requires group>=1 evenly dividing "
          "both the input and output channel counts");
    int64_t CPerGroup = C / group;
    int64_t COPerGroup = CO / group;
    if (wShape[1] != CPerGroup)
      return rewriter.notifyMatchFailure(op,
          "W's channel-in dim must equal X's channel dim divided by group");

    SmallVector<int64_t, 4> kernelShape{KH, KW};
    if (auto ks = getOptionalIntArray(qlcOp.getKernelShape())) {
      if (ks->size() != 2)
        return rewriter.notifyMatchFailure(
            op, "QLinearConv->Krnl lowering only supports 2D kernel_shape");
      kernelShape = *ks;
    }
    SmallVector<int64_t, 4> strides{1, 1};
    if (auto s = getOptionalIntArray(qlcOp.getStrides()))
      strides = *s;
    SmallVector<int64_t, 4> dilations{1, 1};
    if (auto d = getOptionalIntArray(qlcOp.getDilations()))
      dilations = *d;
    SmallVector<int64_t, 4> pads{0, 0, 0, 0};
    if (auto p = getOptionalIntArray(qlcOp.getPads()))
      pads = *p;

    int64_t padHBegin = pads[0], padWBegin = pads[1];
    int64_t padHEnd = pads[2], padWEnd = pads[3];
    int64_t HPad = H + padHBegin + padHEnd;
    int64_t WPad = Win + padWBegin + padWEnd;
    int64_t HO = (HPad - dilations[0] * (kernelShape[0] - 1) - 1) / strides[0] + 1;
    int64_t WO = (WPad - dilations[1] * (kernelShape[1] - 1) - 1) / strides[1] + 1;
    int64_t M = N * HO * WO;

    // --- Pad X once (if needed), using x_zero_point as the fill value: in
    // the quantized domain, x_zero_point *is* real zero. Padding is spatial
    // only, so it's shared across all groups below. ---
    Value paddedX = X;
    if (padHBegin || padWBegin || padHEnd || padWEnd) {
      Value fill = toScalarValue(create, xZeroPoint);
      if (!fill)
        return rewriter.notifyMatchFailure(op,
            "QLinearConv->Krnl lowering requires a scalar x_zero_point when "
            "padding is needed");
      Value padsVal = create.constantInt64(
          {0, 0, padHBegin, padWBegin, 0, 0, padHEnd, padWEnd});
      paddedX = create.pad(X, padsVal, fill, "constant");
    }

    // --- Common types / result element type. ---
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();
    auto resMemRefType = mlir::dyn_cast<MemRefType>(
        typeConverter->convertType(qlcOp.getResult().getType()));
    Type resElementType = resMemRefType.getElementType();

    Value cst128;
    if (resElementType.isUnsignedInteger(8)) {
      auto cst128Attr = DenseElementsAttr::get(
          RankedTensorType::get({}, i32Ty), static_cast<int32_t>(128));
      cst128 = create.constant(cst128Attr);
    }

    // --- Per-group im2col + quantized matmul. Each group only ever sees
    // its own slice of input channels (from paddedX) and output channels
    // (from W / w_scale / w_zero_point / bias); group==1 (the common case)
    // takes exactly the same path as before, just with a trivial
    // single-iteration loop around it. ---
    SmallVector<Value, 4> groupResults;
    for (int64_t g = 0; g < group; ++g) {
      Value paddedXg = paddedX;
      if (group > 1) {
        auto xgTy = RankedTensorType::get({N, CPerGroup, HPad, WPad}, xElemType);
        paddedXg = create.slice(xgTy, paddedX,
            create.constantInt64({g * CPerGroup}),
            create.constantInt64({(g + 1) * CPerGroup}),
            create.constantInt64({1}), create.constantInt64({1}));
      }

      // im2col: for each kernel tap (kh, kw), slice out its contribution
      // across all output positions, then concat taps (kh outer, kw inner)
      // so the flattened K-dimension order matches W's [KH,KW,C,CO]
      // reshape below.
      auto tapType = RankedTensorType::get({N, CPerGroup, HO, WO}, xElemType);
      SmallVector<Value, 16> taps;
      for (int64_t kh = 0; kh < kernelShape[0]; ++kh) {
        int64_t hStart = kh * dilations[0];
        int64_t hEnd = hStart + (HO - 1) * strides[0] + 1;
        for (int64_t kw = 0; kw < kernelShape[1]; ++kw) {
          int64_t wStart = kw * dilations[1];
          int64_t wEnd = wStart + (WO - 1) * strides[1] + 1;
          Value tap = create.slice(tapType, paddedXg,
              create.constantInt64({hStart, wStart}),
              create.constantInt64({hEnd, wEnd}), create.constantInt64({2, 3}),
              create.constantInt64({strides[0], strides[1]}));
          // [N,CPerGroup,HO,WO] -> [N,HO,WO,CPerGroup], so concatenating
          // taps along the last axis below yields flattened order
          // (kh,kw,c) matching W.
          taps.push_back(create.transposeInt64(tap, {0, 2, 3, 1}));
        }
      }
      int64_t K = kernelShape[0] * kernelShape[1] * CPerGroup;
      Value unfolded =
          taps.size() == 1 ? taps[0]
                            : create.concat(RankedTensorType::get({N, HO, WO, K},
                                                xElemType),
                                  taps, /*axis=*/3);
      Value AMat = create.reshape(RankedTensorType::get({M, K}, xElemType),
          unfolded, create.constantInt64({M, K}));

      // Slice W to this group's output channels, then reshape [COPerGroup,
      // CPerGroup,KH,KW] -> [KH,KW,CPerGroup,COPerGroup] -> [K,COPerGroup],
      // matching the A side's (kh,kw,c) flattening order, with COPerGroup
      // kept as the last dim so per-output-channel w_scale/w_zero_point
      // broadcast correctly (same assumption QLinearMatMul.cpp's own
      // B-operand handling relies on).
      Value Wg = W;
      if (group > 1) {
        auto wgTy =
            RankedTensorType::get({COPerGroup, CPerGroup, KH, KW}, wType.getElementType());
        Wg = create.slice(wgTy, W, create.constantInt64({g * COPerGroup}),
            create.constantInt64({(g + 1) * COPerGroup}),
            create.constantInt64({0}), create.constantInt64({1}));
      }
      Value wPermuted = create.transposeInt64(Wg, {2, 3, 1, 0});
      Value BMat = create.reshape(
          RankedTensorType::get({K, COPerGroup}, wType.getElementType()),
          wPermuted, create.constantInt64({K, COPerGroup}));

      Value wScaleG = sliceChannelRangeOrScalar(create, wScale, g * COPerGroup, COPerGroup);
      Value wZeroPointG =
          sliceChannelRangeOrScalar(create, wZeroPoint, g * COPerGroup, COPerGroup);
      Value yZeroPointG =
          sliceChannelRangeOrScalar(create, yZeroPoint, g * COPerGroup, COPerGroup);
      Value yScaleG = sliceChannelRangeOrScalar(create, yScale, g * COPerGroup, COPerGroup);

      // --- int8x8 -> int32 accumulate, + bias, + rescale/saturate. Mirrors
      // ONNXQLinearMatMulOpLowering (QLinearMatMul.cpp), with a bias-add
      // inserted in the i32 domain before rescaling (QLinearMatMul has no
      // bias operand to hang this off of, which is why QLinearConv can't
      // just delegate to OnnxBuilder::qlinearMatMul() as a black box). ---
      Value AI8 = create.getOrCastToI8(AMat);
      Value AI32 = create.cast(AI8, i32Ty);
      Value xZeroPointI32 = create.cast(create.getOrCastToI8(xZeroPoint), i32Ty);
      AI32 = create.sub(AI32, xZeroPointI32);

      Value BI8 = create.getOrCastToI8(BMat);
      Value BI32 = create.cast(BI8, i32Ty);
      Value wZeroPointI32 = create.cast(create.getOrCastToI8(wZeroPointG), i32Ty);
      BI32 = create.sub(BI32, wZeroPointI32);

      Value yZeroPointI32 = create.cast(create.getOrCastToI8(yZeroPointG), i32Ty);

      Value resI32 = create.matmul(
          RankedTensorType::get({M, COPerGroup}, i32Ty), AI32, BI32);

      if (hasBias) {
        Value biasG = sliceChannelRangeOrScalar(create, Bias, g * COPerGroup, COPerGroup);
        // Bias is required by spec to already be quantized with
        // scale = x_scale * w_scale, zero_point = 0, so it adds directly
        // into the pre-rescale i32 accumulator. Shape [COPerGroup]
        // broadcasts against [M,COPerGroup] on the trailing dim.
        resI32 = create.add(resI32, biasG);
      }

      Value resF32 = create.cast(resI32, f32Ty);
      Value scale = create.div(create.mul(xScale, wScaleG), yScaleG);
      resF32 = create.mul(resF32, scale);

      Value roundToEven = create.round(resF32);
      resI32 = create.cast(roundToEven, i32Ty);
      resI32 = create.add(resI32, yZeroPointI32);
      if (resElementType.isUnsignedInteger(8)) {
        resI32 = create.add(resI32, cst128);
        resI32 = create.cast(resI32, i8Ty);
      }
      Value resG = create.cast(resI32, resElementType); // [M, COPerGroup]
      groupResults.push_back(resG);
    }

    Value res = groupResults.size() == 1
                    ? groupResults[0]
                    : create.concat(RankedTensorType::get({M, CO}, resElementType),
                          groupResults, /*axis=*/1);

    // [M, CO] = [N*HO*WO, CO] -> [N,HO,WO,CO] -> [N,CO,HO,WO] (NCHW).
    Value res4d = create.reshape(
        RankedTensorType::get({N, HO, WO, CO}, resElementType), res,
        create.constantInt64({N, HO, WO, CO}));
    Value resNCHW = create.transposeInt64(res4d, {0, 3, 1, 2});

    rewriter.replaceOp(op, {create.toMemref(resNCHW)});
    return success();
  }
};

void populateLoweringONNXQLinearConvOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXQLinearConvOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
