/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- QLinearMatMul.cpp - Lowering QLinearMatMul Op --------------===//
//
// Copyright 2019-2025 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX QLinearMatMul Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXQLinearMatMulOpLowering
    : public OpConversionPattern<ONNXQLinearMatMulOp> {
public:
  ONNXQLinearMatMulOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXQLinearMatMulOp qlmmOp,
      ONNXQLinearMatMulOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    using LocalDialectBuilder =
        MultiDialectBuilder<IndexExprBuilderForKrnl, OnnxBuilder>;
    Operation *op = qlmmOp.getOperation();
    Location loc = ONNXLoc<ONNXQLinearMatMulOp>(op);
    LocalDialectBuilder create(rewriter, loc);

    ValueRange operands = adaptor.getOperands();
    Value A = adaptor.getA();
    Value aScale = adaptor.getAScale();
    Value aZeroPoint = adaptor.getAZeroPoint();
    Value B = adaptor.getB();
    Value bScale = adaptor.getBScale();
    Value bZeroPoint = adaptor.getBZeroPoint();
    Value yScale = adaptor.getYScale();
    Value yZeroPoint = adaptor.getYZeroPoint();

    // Common types.
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();
    Type f32Ty = rewriter.getF32Type();
    auto resMemRefType = dyn_cast<MemRefType>(
        typeConverter->convertType(qlmmOp.getResult().getType()));
    Type resElementType = resMemRefType.getElementType();

    // Get shape.
    ONNXQLinearMatMulOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Opset 21 float8 (E4M3FN/E5M2) form: A/B/Y (and their zero points, per
    // the T1/T2/T3 type constraint) are float8, not int8/uint8. There is no
    // native int32-style accumulator domain for float8 -- unlike int8, the
    // matmul itself just happens in real f32 (decode -> matmul -> rescale ->
    // encode), relying on MathBuilder::cast's F8<->F32 conversion support
    // (decode/encode via bit manipulation, since F8 has no working
    // arith-to-LLVM float-op lowering at all; see that function's own
    // comment block for why). This mirrors the int8 path's overall
    // structure (dequant subtract, matmul, rescale, add zero point) but
    // entirely in f32 instead of the int8 path's i32 accumulate domain.
    if (mlir::isa<Float8E4M3FNType, Float8E5M2Type>(getElementType(A.getType()))) {
      if (!mlir::isa<Float8E4M3FNType, Float8E5M2Type>(getElementType(B.getType())))
        return failure();
      if (!getElementType(aScale.getType()).isF32())
        return failure();
      if (!getElementType(bScale.getType()).isF32())
        return failure();
      if (!getElementType(yScale.getType()).isF32())
        return failure();

      Value AF32 = create.onnx.cast(A, f32Ty);
      Value aZeroPointF32 = create.onnx.cast(aZeroPoint, f32Ty);
      auto aZeroPointType = mlir::cast<ShapedType>(aZeroPoint.getType());
      int64_t aZeroPointRank = aZeroPointType.getRank();
      // Same per-row broadcast handling as the int8 path below.
      if ((aZeroPointRank == 1) && (aZeroPointType.getShape()[0] != 1)) {
        SmallVector<int64_t, 4> unsqueezeShape(aZeroPointType.getShape());
        unsqueezeShape.emplace_back(1);
        aZeroPointF32 =
            create.onnx.unsqueeze(RankedTensorType::get(unsqueezeShape, f32Ty),
                aZeroPointF32, create.onnx.constantInt64({aZeroPointRank}));
      }
      AF32 = create.onnx.sub(AF32, aZeroPointF32);

      Value BF32 = create.onnx.cast(B, f32Ty);
      Value bZeroPointF32 = create.onnx.cast(bZeroPoint, f32Ty);
      BF32 = create.onnx.sub(BF32, bZeroPointF32);

      Value yZeroPointF32 = create.onnx.cast(yZeroPoint, f32Ty);

      Value resF32 = create.onnx.matmul(
          RankedTensorType::get(resMemRefType.getShape(), f32Ty), AF32, BF32);
      Value scale = create.onnx.div(create.onnx.mul(aScale, bScale), yScale);
      resF32 = create.onnx.mul(resF32, scale);
      // y_zero_point is added in f32 before the single final encode-to-F8
      // rounding step (rather than in a separate post-round F8 domain add,
      // which would need its own decode/encode round trip): per the ONNX
      // spec's own doc, zero-point is "usually not used" (i.e. F8's own
      // encoding of 0.0) for float8 quantization, so this ordering choice
      // is immaterial in the realistic case and avoids an extra round trip.
      resF32 = create.onnx.add(resF32, yZeroPointF32);
      Value res = create.onnx.cast(resF32, resElementType);

      rewriter.replaceOp(op, {create.onnx.toMemref(res)});
      return success();
    }

    // Now only support integer8 for inputs and zeropoints, and support float32
    // for scale.
    if (!getElementType(A.getType()).isInteger(8))
      return failure();
    if (!getElementType(B.getType()).isInteger(8))
      return failure();
    if (!getElementType(aScale.getType()).isF32())
      return failure();
    if (!getElementType(bScale.getType()).isF32())
      return failure();
    if (!getElementType(yScale.getType()).isF32())
      return failure();
    if (!getElementType(aZeroPoint.getType()).isInteger(8))
      return failure();
    if (!getElementType(bZeroPoint.getType()).isInteger(8))
      return failure();
    if (!getElementType(yZeroPoint.getType()).isInteger(8))
      return failure();

    Value cst128;
    if (resElementType.isUnsignedInteger(8)) {
      auto cst128Attr = DenseElementsAttr::get(
          RankedTensorType::get({}, i32Ty), static_cast<int32_t>(128));
      cst128 = create.onnx.constant(cst128Attr);
    }

    // Prepare input A.
    Value AI8 = create.onnx.getOrCastToI8(A);
    Value AI32 = create.onnx.cast(AI8, i32Ty);
    auto aZeroPointType = mlir::cast<ShapedType>(aZeroPoint.getType());
    int64_t aZeroPointRank = aZeroPointType.getRank();
    Value aZeroPointI8 = create.onnx.getOrCastToI8(aZeroPoint);
    Value aZeroPointI32 = create.onnx.cast(aZeroPointI8, i32Ty);
    // If broadcasting, e.g. A is [MxK], zeroPoint is [M], M != 1.
    // Unsqueeze zeroPoint to [Mx1] to make shapes compatible.
    // There is no need to handle scalar zeroPoint (e.g. tensor<dtype> or
    // tensor<1xdtype>), which is always true for broadcasting.
    if ((aZeroPointRank == 1) && (aZeroPointType.getShape()[0] != 1)) {
      SmallVector<int64_t, 4> unsqueezeShape(aZeroPointType.getShape());
      unsqueezeShape.emplace_back(1);
      aZeroPointI32 =
          create.onnx.unsqueeze(RankedTensorType::get(unsqueezeShape, i32Ty),
              aZeroPointI32, create.onnx.constantInt64({aZeroPointRank}));
    }
    AI32 = create.onnx.sub(AI32, aZeroPointI32);

    // Prepare input B.
    Value BI8 = create.onnx.getOrCastToI8(B);
    Value BI32 = create.onnx.cast(BI8, i32Ty);
    // K is the broadcating dim: [KxN] - [N] = [KxN] - [1xN]
    Value bZeroPointI8 = create.onnx.getOrCastToI8(bZeroPoint);
    Value bZeroPointI32 = create.onnx.cast(bZeroPointI8, i32Ty);
    BI32 = create.onnx.sub(BI32, bZeroPointI32);

    // Prepare output Y
    Value yZeroPointI8 = create.onnx.getOrCastToI8(yZeroPoint);
    Value yZeroPointI32 = create.onnx.cast(yZeroPointI8, i32Ty);

    // Emit MatMul.
    Value resI32 = create.onnx.matmul(
        RankedTensorType::get(resMemRefType.getShape(), i32Ty), AI32, BI32);

    // Scale the output.
    Value resF32 = create.onnx.cast(resI32, f32Ty);
    Value scale = create.onnx.div(create.onnx.mul(aScale, bScale), yScale);
    resF32 = create.onnx.mul(resF32, scale);

    // Saturate and add zero point.
    Value roundToEven = create.onnx.round(resF32);
    resI32 = create.onnx.cast(roundToEven, i32Ty);
    resI32 = create.onnx.add(resI32, yZeroPointI32);
    if (resElementType.isUnsignedInteger(8)) {
      resI32 = create.onnx.add(resI32, cst128);
      resI32 = create.onnx.cast(resI32, i8Ty);
    }
    Value res = create.onnx.cast(resI32, resElementType);

    rewriter.replaceOp(op, {create.onnx.toMemref(res)});
    return success();
  }
};

void populateLoweringONNXQLinearMatMulOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXQLinearMatMulOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
