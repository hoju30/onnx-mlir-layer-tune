#include "src/Conversion/PositToKrnl/TypeConverters.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Casting.h"


#include "src/Dialect/Posit/PositDialect.h"

using namespace mlir;

namespace onnx_mlir {

static Type convertPositScalarToIType(Type t, unsigned nbits, MLIRContext *ctx) {
  if (llvm::isa<posit::PositType>(t))
    return IntegerType::get(ctx, nbits);
  return t;
}

static Type convertPositTensorToMemref(Type t, unsigned nbits, MLIRContext *ctx) {
  // tensor<...x!posit.type>  -> memref<...xi8>
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(t)) {
    Type elem = ranked.getElementType();
    if (llvm::isa<posit::PositType>(elem)) {
      Type iElem = IntegerType::get(ctx, nbits);
      return MemRefType::get(ranked.getShape(), iElem);
    }
    return t; // 不是 posit tensor 就先不動
  }

  if (auto unranked = llvm::dyn_cast<UnrankedTensorType>(t)) {
    Type elem = unranked.getElementType();
    if (llvm::isa<posit::PositType>(elem)) {
      Type iElem = IntegerType::get(ctx, nbits);
      return UnrankedMemRefType::get(iElem, /*memorySpace=*/0);
    }
    return t;
  }

  // scalar !posit.type -> iN
  return convertPositScalarToIType(t, nbits, ctx);
}

PositToKrnlTypeConverter::PositToKrnlTypeConverter(unsigned nbits, unsigned es,
                                                   MLIRContext *ctx)
    : nbits_(nbits), es_(es) {
  // 先放一個 identity conversion，避免其它 type 全部 conversion fail
  addConversion([](Type t) { return t; });

  // 我們要的轉換（capture by value，避免 dangling reference）
  addConversion([nbits, ctx](Type t) {
    return convertPositTensorToMemref(t, nbits, ctx);
  });

  // materialization：用 unrealized_conversion_cast 先橋接型別落差
  auto makeCast = [](OpBuilder &b, Type dst, ValueRange inputs,
                     Location loc) -> Value {
    if (inputs.size() != 1)
      return {};
    if (inputs.front().getType() == dst)
      return inputs.front();
    auto cast = b.create<UnrealizedConversionCastOp>(loc, dst, inputs);
    return cast.getResult(0);
  };

  addSourceMaterialization(makeCast);
  addTargetMaterialization(makeCast);
}

} // namespace onnx_mlir