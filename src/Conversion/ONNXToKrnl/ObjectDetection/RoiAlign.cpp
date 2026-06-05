/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- RoiAlign.cpp - Lowering RoiAlign Op ---------------------===//
//
// Copyright 2026
//
// =============================================================================
//
// This file lowers the ONNX RoiAlign Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXRoiAlignOpLowering : public OpConversionPattern<ONNXRoiAlignOp> {
  ONNXRoiAlignOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXRoiAlignOp roiAlignOp,
      ONNXRoiAlignOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = roiAlignOp.getOperation();
    Location loc = ONNXLoc<ONNXRoiAlignOp>(op);
    ValueRange operands = adaptor.getOperands();

    if (roiAlignOp.getMode() != "avg")
      return emitError(loc, "RoiAlign lowering currently supports only mode=avg");
    if (roiAlignOp.getCoordinateTransformationMode() != "half_pixel")
      return emitError(loc,
          "RoiAlign lowering currently supports only "
          "coordinate_transformation_mode=half_pixel");

    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputType = cast<MemRefType>(convertedType);
    if (!outputType.getElementType().isF32())
      return emitError(loc, "RoiAlign lowering currently supports f32 output");

    MultiDialectBuilder<IndexExprBuilderForKrnl, MemRefBuilder> create(
        rewriter, loc);
    ONNXRoiAlignOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    Value alloc = create.mem.alignedAlloc(outputType, shapeHelper.getOutputDims());

    std::vector<std::string> attributeNames = {"coordinate_transformation_mode",
        "mode", "output_height", "output_width", "sampling_ratio",
        "spatial_scale"};
    rewriter.create<KrnlCallOp>(
        loc, "RoiAlign", alloc, op, operands, attributeNames);
    rewriter.replaceOp(op, alloc);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXRoiAlignOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXRoiAlignOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
