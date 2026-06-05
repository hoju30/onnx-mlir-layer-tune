/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------- Custom.cpp - Lowering Custom Op--------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNXCustomOp to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

struct ONNXCustomOpLowering : public OpConversionPattern<ONNXCustomOp> {
  ONNXCustomOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx) {}

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
      ONNXCustomOpAdaptor operandAdaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = customOp.getOperation();
    Location loc = op->getLoc();
    ValueRange operands = operandAdaptor.getOperands();

    // Helper builders.
    MultiDialectBuilder<AffineBuilder, IndexExprBuilderForKrnl, KrnlBuilder,
        MemRefBuilder>
        create(rewriter, loc);
    IndexExprScope scope(create.krnlIE);

    // Get shape when custom op exposes a supported shape inference pattern.
    ONNXCustomOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    const bool hasShapeHelper = shapeHelper.isImplemented();
    if (hasShapeHelper)
      shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare outputs for krnl.call
    SmallVector<Type, 4> outputMemRefTypes;
    SmallVector<Value, 4> outputAllocs;
    for (size_t idx = 0; idx < op->getResultTypes().size(); idx++) {
      Type ty = op->getResultTypes()[idx];
      Type convertedType = typeConverter->convertType(ty);
      MemRefType outputMemRefType;
      DimsExpr allocDims;

      if (auto rankedTy = mlir::dyn_cast<MemRefType>(convertedType)) {
        outputMemRefType = rankedTy;
      } else if (auto unrankedTy =
                     mlir::dyn_cast<UnrankedMemRefType>(convertedType)) {
        // Materialize a ranked memref for unranked outputs.
        if (hasShapeHelper) {
          DimsExpr outputDims = shapeHelper.getOutputDims(idx);
          SmallVector<int64_t, 4> shape;
          shape.reserve(outputDims.size());
          for (const IndexExpr &dim : outputDims) {
            shape.emplace_back(
                dim.isLiteral() ? dim.getLiteral() : ShapedType::kDynamic);
          }
          outputMemRefType =
              MemRefType::get(shape, unrankedTy.getElementType());
        } else {
          // Fallback for custom ops without shape helper support:
          // infer output rank/shape from the first ranked memref input.
          Value shapeLikeOperand;
          MemRefType shapeLikeType;
          for (Value operand : operands) {
            if (auto operandTy = mlir::dyn_cast<MemRefType>(operand.getType())) {
              shapeLikeOperand = operand;
              shapeLikeType = operandTy;
              break;
            }
          }
          if (!shapeLikeOperand || !shapeLikeType) {
            return rewriter.notifyMatchFailure(customOp,
                "cannot infer ranked output shape for unranked custom op");
          }

          SmallVector<int64_t, 4> shape(shapeLikeType.getShape().begin(),
              shapeLikeType.getShape().end());
          outputMemRefType =
              MemRefType::get(shape, unrankedTy.getElementType());
          for (int64_t i = 0, e = shapeLikeType.getRank(); i < e; ++i) {
            if (shapeLikeType.isDynamicDim(i)) {
              Value dynDim = create.mem.dim(shapeLikeOperand, i);
              allocDims.emplace_back(DimIndexExpr(dynDim));
            } else {
              allocDims.emplace_back(LiteralIndexExpr(shapeLikeType.getDimSize(i)));
            }
          }
        }
      } else {
        return rewriter.notifyMatchFailure(
            customOp, "custom op result is not a memref type after conversion");
      }

      outputMemRefTypes.emplace_back(outputMemRefType);
      if (allocDims.empty()) {
        if (hasShapeHelper) {
          allocDims = shapeHelper.getOutputDims(idx);
        } else {
          for (int64_t i = 0, e = outputMemRefType.getRank(); i < e; ++i) {
            if (outputMemRefType.isDynamicDim(i)) {
              return rewriter.notifyMatchFailure(customOp,
                  "dynamic custom op output needs shape helper or shape-like input");
            }
            allocDims.emplace_back(
                LiteralIndexExpr(outputMemRefType.getDimSize(i)));
          }
        }
      }

      Value alloc = create.mem.alignedAlloc(outputMemRefType, allocDims);
      outputAllocs.emplace_back(alloc);
    }

    // Lower to Krnl for special CustomOp
    // Create Krnl.Call

    // Handle the attributes: exclude the attributes used for analysis
    // function_name is passed explicitly. Others may include shape inference
    std::vector<std::string> excludeStrings = {"function_name",
        "shape_infer_pattern", "inputs_for_infer", "output_element_type"};
    std::vector<std::string> attributeNames;
    for (NamedAttribute namedAttr : customOp->getAttrs()) {
      std::string attrName = namedAttr.getName().getValue().str();
      if (std::find(excludeStrings.begin(), excludeStrings.end(), attrName) ==
          excludeStrings.end())
        attributeNames.push_back(attrName);
    }
    rewriter.create<KrnlCallOp>(loc, customOp.getFunctionName().str(),
        outputAllocs, op, operands, attributeNames);

    rewriter.replaceOp(op, outputAllocs);
    return success();
  }
};

void populateLoweringONNXCustomOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCustomOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
