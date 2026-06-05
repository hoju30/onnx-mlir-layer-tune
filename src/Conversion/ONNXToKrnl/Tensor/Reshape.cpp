/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Reshape.cpp - Lowering Reshape Op -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX Reshape Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "reshape_onnx_to_krnl"

using namespace mlir;

namespace onnx_mlir {

static bool getOriginalShapeLiterals(
    ONNXReshapeOp reshapeOp, SmallVectorImpl<std::optional<int64_t>> &dims) {
  Value shape = reshapeOp.getShape();

  SmallVector<int64_t, 4> constShape;
  if (getI64ValuesFromONNXConstantOp(shape, constShape)) {
    dims.reserve(constShape.size());
    for (int64_t dim : constShape)
      dims.emplace_back(dim);
    return true;
  }

  if (!areDimsFromConcat(shape))
    return false;

  SmallVector<Value, 4> shapeDims;
  getDims(shape, shapeDims);
  dims.reserve(shapeDims.size());
  for (Value dimVal : shapeDims) {
    SmallVector<int64_t, 1> constDim;
    if (getI64ValuesFromONNXConstantOp(dimVal, constDim) && constDim.size() == 1)
      dims.emplace_back(constDim[0]);
    else
      dims.emplace_back(std::nullopt);
  }
  return true;
}

struct ONNXReshapeOpLowering : public OpConversionPattern<ONNXReshapeOp> {
  DimAnalysis *dimAnalysis;

  ONNXReshapeOpLowering(
      TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis)
      : OpConversionPattern(typeConverter, ctx), dimAnalysis(dimAnalysis) {}

  LogicalResult matchAndRewrite(ONNXReshapeOp reshapeOp,
      ONNXReshapeOpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = reshapeOp.getOperation();
    Location loc = ONNXLoc<ONNXReshapeOp>(op);
    ValueRange operands = adaptor.getOperands();
    Value data = adaptor.getData();

    // If reshape does not change dimensions or it is an identity, just replace
    // the output with the input.
    if (isIdentityReshape(reshapeOp, dimAnalysis)) {
      LLVM_DEBUG(llvm::dbgs() << "Lowering reshape to identity\n");
      rewriter.replaceOp(op, data);
      return success();
    }

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType memRefType = mlir::cast<MemRefType>(convertedType);
    LLVM_DEBUG(llvm::dbgs() << "memRefType: " << memRefType << "\n");

    MultiDialectBuilder<KrnlBuilder, IndexExprBuilderForKrnl, MathBuilder>
        create(rewriter, loc);

    // Get shape.
    ONNXReshapeOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    DimsExpr outputDims = shapeHelper.getOutputDims();
    SmallVector<std::optional<int64_t>, 4> originalShapeLits;
    bool hasOriginalShapeLits =
        getOriginalShapeLiterals(reshapeOp, originalShapeLits);

    // Some reshape patterns still carry literal -1/0 shape markers into the
    // lowering path even though the input rank is known. Normalize them here
    // before creating the memref view so runtime descriptors never see
    // negative extents such as [-1, 1, 768].
    if (create.krnlIE.hasShapeAndRank(data)) {
      int64_t dataRank = create.krnlIE.getShapedTypeRank(data);
      int64_t outputRank = outputDims.size();
      IndexExpr totalInputElements = LitIE(1);
      for (int64_t i = 0; i < dataRank; ++i)
        totalInputElements =
            totalInputElements * create.krnlIE.getShapeAsDim(data, i);

      int64_t inferDimPos = -1;
      IndexExpr knownShapeProduct = LitIE(1);
      for (int64_t i = 0; i < outputRank; ++i) {
        IndexExpr dim = outputDims[i];
        std::optional<int64_t> originalLit =
            (hasOriginalShapeLits && i < static_cast<int64_t>(originalShapeLits.size()))
                ? originalShapeLits[i]
                : std::nullopt;
        if (originalLit.has_value()) {
          int64_t lit = *originalLit;
          if (lit == 0 && i < dataRank)
            dim = create.krnlIE.getShapeAsDim(data, i);
          else if (lit == -1) {
            inferDimPos = i;
            dim = LitIE(1);
          }
        } else {
          IndexExpr dimShape =
              create.krnlIE.getIntFromArrayAsSymbol(adaptor.getShape(), i);
          if (dimShape.isDefined() && dimShape.isLiteral()) {
            int64_t lit = dimShape.getLiteral();
            if (lit == 0 && i < dataRank)
              dim = create.krnlIE.getShapeAsDim(data, i);
            else if (lit == -1) {
              inferDimPos = i;
              dim = LitIE(1);
            }
          }
        }
        outputDims[i] = dim;
        knownShapeProduct = knownShapeProduct * dim;
      }
      if (inferDimPos >= 0)
        outputDims[inferDimPos] =
            totalInputElements.floorDiv(knownShapeProduct);
    }

    // Lower to ReinterpretCastOp so that the data is never copied or modified.
    Value newView = emitMemRefReinterpretCastOp(
        rewriter, loc, data, outputDims, convertedType);
    LLVM_DEBUG(llvm::dbgs() << "newView: " << newView << "\n");

    rewriter.replaceOp(op, newView);
    onnxToKrnlSimdReport(op);
    return success();
  }
};

void populateLoweringONNXReshapeOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis) {
  patterns.insert<ONNXReshapeOpLowering>(typeConverter, ctx, dimAnalysis);
}

} // namespace onnx_mlir
