#include "TypeConverters.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h" 
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Casting.h" 

#include "src/Conversion/ONNXToPosit/TypeConverters.hpp"

// 讓mlir::posit可以被讀懂
#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositPasses.h"
#include "src/Dialect/Posit/PositOps.h"

// ONNX
#include "src/Dialect/ONNX/ONNXOps.hpp"
//new
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

namespace onnx_mlir {




// Container 型別遞迴改元素
static Type convertElementTypeToPosit(Type ety,
                                      unsigned nbits, unsigned es,
                                      MLIRContext *ctx) {
  // 只轉 float / int，其它直接放過
  if (!llvm::isa<FloatType>(ety) && !llvm::isa<IntegerType>(ety))
    return ety;

  // 參數防呆：nbits>0 且 es<nbits
  if (nbits == 0 || es >= nbits)
    return ety;

  // 回傳 !posit.type<nbits, es>
  return posit::PositType::get(ctx, nbits, es);
}

/// 把 container type（tensor/unranked tensor）裡的 element 換掉
static Type convertContainer(Type t,
                             unsigned nbits, unsigned es,
                             MLIRContext *ctx) {
  if (auto shaped = llvm::dyn_cast<ShapedType>(t)) {
    Type oldElem = shaped.getElementType();
    Type newElem = convertElementTypeToPosit(oldElem, nbits, es, ctx);

    if (auto ranked = llvm::dyn_cast<RankedTensorType>(t))
      return RankedTensorType::get(ranked.getShape(), newElem);
    if (auto unr = llvm::dyn_cast<UnrankedTensorType>(t))
      return UnrankedTensorType::get(newElem);

    // 其它 ShapedType（例如 memref）暫時先不處理
    return shaped;
  }

  // 非 ShapedType 就當 scalar，直接轉 element type
  return convertElementTypeToPosit(t, nbits, es, ctx);
}

PositTypeConverter::PositTypeConverter(unsigned nbits, unsigned es,
                                       MLIRContext *ctx) {
  // 關鍵
  addConversion([nbits, es, ctx](Type t) { return convertContainer(t, nbits, es, ctx); });

  // Source/Target materialization：用 UnrealizedConversionCastOp
  auto makeCast = [](OpBuilder &b, Type dst, ValueRange inputs,
                     Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    if (inputs.front().getType() == dst)
      return inputs.front();
    auto cast =
        b.create<UnrealizedConversionCastOp>(loc, dst, inputs);
    return cast.getResult(0);
  };

  addSourceMaterialization(makeCast);
  addTargetMaterialization(makeCast);
}


} // namespace onnx_mlir
