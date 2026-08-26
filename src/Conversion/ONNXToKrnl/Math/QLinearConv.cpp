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
// v1 scope (see project plan): group == 1, 4D NCHW tensors, auto_pad ==
// NOTSET, and all shape-affecting attributes/dims statically known. Grouped /
// depthwise convolution (e.g. MobileNet) is intentionally not yet supported
// and the pattern fails to match (leaving the op unconverted) rather than
// emitting incorrect results.
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

    if (qlcOp.getGroup() != 1)
      return rewriter.notifyMatchFailure(op,
          "QLinearConv->Krnl lowering only supports group==1 for now "
          "(depthwise/grouped conv, e.g. MobileNet, is a follow-up phase)");
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
    ArrayRef<int64_t> wShape = wType.getShape(); // [CO, C, KH, KW]
    int64_t N = xShape[0], C = xShape[1], H = xShape[2], Win = xShape[3];
    int64_t CO = wShape[0], KH = wShape[2], KW = wShape[3];
    if (wShape[1] != C)
      return rewriter.notifyMatchFailure(
          op, "W's channel-in dim must equal X's channel dim under group==1");

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

    // --- Pad X (if needed), using x_zero_point as the fill value: in the
    // quantized domain, x_zero_point *is* real zero. ---
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

    // --- im2col: for each kernel tap (kh, kw), slice out its contribution
    // across all output positions, then concat taps (kh outer, kw inner) so
    // the flattened K-dimension order matches W's [KH,KW,C,CO] reshape
    // below. ---
    auto tapType = RankedTensorType::get({N, C, HO, WO}, xElemType);
    SmallVector<Value, 16> taps;
    for (int64_t kh = 0; kh < kernelShape[0]; ++kh) {
      int64_t hStart = kh * dilations[0];
      int64_t hEnd = hStart + (HO - 1) * strides[0] + 1;
      for (int64_t kw = 0; kw < kernelShape[1]; ++kw) {
        int64_t wStart = kw * dilations[1];
        int64_t wEnd = wStart + (WO - 1) * strides[1] + 1;
        Value tap = create.slice(tapType, paddedX,
            create.constantInt64({hStart, wStart}),
            create.constantInt64({hEnd, wEnd}), create.constantInt64({2, 3}),
            create.constantInt64({strides[0], strides[1]}));
        // [N,C,HO,WO] -> [N,HO,WO,C], so concatenating taps along the last
        // axis below yields flattened order (kh,kw,c) matching W.
        taps.push_back(create.transposeInt64(tap, {0, 2, 3, 1}));
      }
    }
    int64_t K = kernelShape[0] * kernelShape[1] * C;
    Value unfolded = taps.size() == 1 ? taps[0]
                                      : create.concat(RankedTensorType::get(
                                            {N, HO, WO, K}, xElemType),
                                            taps, /*axis=*/3);
    int64_t M = N * HO * WO;
    Value AMat = create.reshape(RankedTensorType::get({M, K}, xElemType),
        unfolded, create.constantInt64({M, K}));

    // --- Reshape W [CO,C,KH,KW] -> [KH,KW,C,CO] -> [K,CO], matching the A
    // side's (kh,kw,c) flattening order, with CO kept as the last dim so
    // per-output-channel w_scale/w_zero_point ([CO]) broadcast correctly
    // against the [K,CO] shape below (same assumption QLinearMatMul.cpp's
    // own B-operand handling already relies on). ---
    Value wPermuted = create.transposeInt64(W, {2, 3, 1, 0});
    Value BMat = create.reshape(
        RankedTensorType::get({K, CO}, wType.getElementType()), wPermuted,
        create.constantInt64({K, CO}));

    // --- int8x8 -> int32 accumulate, + bias, + rescale/saturate. Mirrors
    // ONNXQLinearMatMulOpLowering (QLinearMatMul.cpp), with a bias-add
    // inserted in the i32 domain before rescaling (QLinearMatMul has no
    // bias operand to hang this off of, which is why QLinearConv can't just
    // delegate to OnnxBuilder::qlinearMatMul() as a black box). ---
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

    Value AI8 = create.getOrCastToI8(AMat);
    Value AI32 = create.cast(AI8, i32Ty);
    Value xZeroPointI32 = create.cast(create.getOrCastToI8(xZeroPoint), i32Ty);
    AI32 = create.sub(AI32, xZeroPointI32);

    Value BI8 = create.getOrCastToI8(BMat);
    Value BI32 = create.cast(BI8, i32Ty);
    Value wZeroPointI32 = create.cast(create.getOrCastToI8(wZeroPoint), i32Ty);
    BI32 = create.sub(BI32, wZeroPointI32);

    Value yZeroPointI32 = create.cast(create.getOrCastToI8(yZeroPoint), i32Ty);

    Value resI32 = create.matmul(
        RankedTensorType::get({M, CO}, i32Ty), AI32, BI32);

    if (hasBias)
      // Bias is required by spec to already be quantized with
      // scale = x_scale * w_scale, zero_point = 0, so it adds directly into
      // the pre-rescale i32 accumulator. Shape [CO] broadcasts against
      // [M,CO] on the trailing dim.
      resI32 = create.add(resI32, Bias);

    Value resF32 = create.cast(resI32, f32Ty);
    Value scale = create.div(create.mul(xScale, wScale), yScale);
    resF32 = create.mul(resF32, scale);

    Value roundToEven = create.round(resF32);
    resI32 = create.cast(roundToEven, i32Ty);
    resI32 = create.add(resI32, yZeroPointI32);
    if (resElementType.isUnsignedInteger(8)) {
      resI32 = create.add(resI32, cst128);
      resI32 = create.cast(resI32, i8Ty);
    }
    Value res = create.cast(resI32, resElementType); // [M, CO]

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
