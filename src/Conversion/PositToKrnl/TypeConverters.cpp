#include "src/Conversion/PositToKrnl/TypeConverters.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/Casting.h"

#include "src/Dialect/Posit/PositDialect.h"

using namespace mlir;

namespace onnx_mlir {

// [MOD][2026-01-14] scalar !posit.type -> iN
static Type convertPositScalarToIType(Type t, unsigned nbits, MLIRContext *ctx) {
  const unsigned storageBits = getPositStorageBitWidth(nbits);
  if (llvm::isa<posit::PositType>(t))
    return IntegerType::get(ctx, storageBits);
  return t;
}

// [MOD][2026-01-14] tensor<T> -> memref<T> (keep element type), for f32/i64 tensors
static Type tensorToMemrefKeepElem(Type t, MLIRContext *ctx) {
  (void)ctx;
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(t)) {
    return MemRefType::get(ranked.getShape(), ranked.getElementType());
  }
  if (auto unranked = llvm::dyn_cast<UnrankedTensorType>(t)) {
    return UnrankedMemRefType::get(unranked.getElementType(), /*memorySpace=*/0);
  }
  return t;
}

static Type convertPositTensorToMemref(Type t, unsigned nbits, MLIRContext *ctx) {
  const unsigned storageBits = getPositStorageBitWidth(nbits);
  // tensor<...x!posit.type>  -> memref<...xi8>
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(t)) {
    Type elem = ranked.getElementType();
    if (llvm::isa<posit::PositType>(elem)) {
      Type iElem = IntegerType::get(ctx, storageBits);
      return MemRefType::get(ranked.getShape(), iElem);
    }
    // [MOD][2026-01-14] f32/i64 tensor -> memref<...xf32>/<...xi64>
    return tensorToMemrefKeepElem(t, ctx);
  }

  if (auto unranked = llvm::dyn_cast<UnrankedTensorType>(t)) {
    Type elem = unranked.getElementType();
    if (llvm::isa<posit::PositType>(elem)) {
      Type iElem = IntegerType::get(ctx, storageBits);
      return UnrankedMemRefType::get(iElem, /*memorySpace=*/0);
    }
    // [MOD]
    return tensorToMemrefKeepElem(t, ctx);
  }

  // scalar !posit.type -> iN
  return convertPositScalarToIType(t, nbits, ctx);
}

PositToKrnlTypeConverter::PositToKrnlTypeConverter(unsigned nbits, unsigned es,
                                                   MLIRContext *ctx)
    : nbits_(nbits), es_(es) {
  addConversion([](Type t) { return t; });

  // Read the actual nbits from the PositType itself so mixed-format models
  // (e.g. Gemm_3:p8e1 + Gemm_5:p16e2) produce the right memref element type.
  addConversion([ctx](Type t) -> std::optional<Type> {
    auto getStorageBits = [](Type elem) -> unsigned {
      if (auto pt = llvm::dyn_cast<posit::PositType>(elem))
        return getPositStorageBitWidth(pt.getNbits());
      return 0;
    };
    if (auto ranked = llvm::dyn_cast<RankedTensorType>(t)) {
      unsigned sb = getStorageBits(ranked.getElementType());
      if (sb)
        return MemRefType::get(ranked.getShape(), IntegerType::get(ctx, sb));
      return tensorToMemrefKeepElem(t, ctx);
    }
    if (auto unranked = llvm::dyn_cast<UnrankedTensorType>(t)) {
      unsigned sb = getStorageBits(unranked.getElementType());
      if (sb)
        return UnrankedMemRefType::get(IntegerType::get(ctx, sb), 0);
      return tensorToMemrefKeepElem(t, ctx);
    }
    if (auto pt = llvm::dyn_cast<posit::PositType>(t))
      return IntegerType::get(ctx, getPositStorageBitWidth(pt.getNbits()));
    return t;
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
