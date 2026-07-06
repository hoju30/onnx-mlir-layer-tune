#include "src/Conversion/PositToKrnl/PositToKrnl.hpp"
#include "src/Conversion/PositToKrnl/TypeConverters.hpp"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/Builders.h"  // [FIX] for OpBuilder::InsertionGuard
#include "mlir/IR/BuiltinTypes.h"  // [FIX] for UnrankedMemRefType
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositOps.h"


#include "src/Dialect/ONNX/ONNXOps.hpp"
// Krnl dialect
#include "src/Dialect/Krnl/KrnlOps.hpp"

#include "src/Conversion/PositToKrnl/Pattern/Math.cpp"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLFunctionalExtras.h"

using namespace mlir;

namespace onnx_mlir {
namespace {

// [EXTEND] Keep cast-chain lowering aligned with pass-selected posit format.
static unsigned gCastChainPositNbits = 8;
static unsigned gCastChainPositEs = 0;
static unsigned gCastChainStorageBits = 8;
static int64_t gTensorConstGlobalId = 0;

static std::string getPositRuntimeSuffix() {
  return "_p" + std::to_string(gCastChainPositNbits) + "e" +
         std::to_string(gCastChainPositEs);
}

// [FIX] Lower the placeholder cast chains that still carry !posit.type into real runtime calls.
// `builtin.unrealized_conversion_cast` is meant as a temporary bridge during type conversion and
// has NO execution semantics by itself; leaving it to the LLVM translation stage will break.  (See MLIR Builtin dialect docs.)
//
// We rewrite the specific chains produced in our pipeline:
//   memref<f32> -> tensor<f32> -> tensor<!posit.type> -> memref<i8>   ==> call _mlir_ciface_posit_from_f32_p8e0
//   memref<i8>  -> tensor<!posit.type> -> tensor<f32> -> memref<f32> ==> call _mlir_ciface_posit_to_f32_p8e0
//
// This removes all occurrences of !posit.type from the Krnl/LLVM stages.

static FlatSymbolRefAttr getOrInsertPositFromF32(ModuleOp module, OpBuilder &rewriter) {
  MLIRContext *ctx = module.getContext();
  std::string runtimeName = "_mlir_ciface_posit_from_f32" + getPositRuntimeSuffix();
  StringRef name(runtimeName);
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    auto f32 = rewriter.getF32Type();
    auto iN = rewriter.getIntegerType(gCastChainStorageBits);
    auto inTy = UnrankedMemRefType::get(f32, /*memorySpace=*/0);
    auto outTy = UnrankedMemRefType::get(iN, /*memorySpace=*/0);
    auto fnTy = rewriter.getFunctionType({inTy, outTy, rewriter.getI64Type()}, {});
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto fn = rewriter.create<func::FuncOp>(module.getLoc(), name, fnTy);
    fn.setPrivate();
  }
  return SymbolRefAttr::get(ctx, name);
}

static FlatSymbolRefAttr getOrInsertPositToF32(ModuleOp module, OpBuilder &rewriter) {
  MLIRContext *ctx = module.getContext();
  std::string runtimeName = "_mlir_ciface_posit_to_f32" + getPositRuntimeSuffix();
  StringRef name(runtimeName);
  if (!module.lookupSymbol<func::FuncOp>(name)) {
    auto f32 = rewriter.getF32Type();
    auto iN = rewriter.getIntegerType(gCastChainStorageBits);
    auto inTy = UnrankedMemRefType::get(iN, /*memorySpace=*/0);
    auto outTy = UnrankedMemRefType::get(f32, /*memorySpace=*/0);
    auto fnTy = rewriter.getFunctionType({inTy, outTy}, {});
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto fn = rewriter.create<func::FuncOp>(module.getLoc(), name, fnTy);
    fn.setPrivate();
  }
  return SymbolRefAttr::get(ctx, name);
}

static FailureOr<SmallVector<Value, 4>> buildAllocDynamicDimsFromLike(
    OpBuilder &rewriter, Location loc, MemRefType allocTy, Value like) {
  SmallVector<Value, 4> dynDims;
  if (!allocTy.hasRank())
    return failure();
  if (allocTy.hasStaticShape())
    return dynDims;

  auto likeTy = dyn_cast<MemRefType>(like.getType());
  if (!likeTy || !likeTy.hasRank() || likeTy.getRank() != allocTy.getRank())
    return failure();

  for (int64_t i = 0, e = allocTy.getRank(); i < e; ++i) {
    if (!allocTy.isDynamicDim(i))
      continue;
    dynDims.push_back(rewriter.create<memref::DimOp>(loc, like, i));
  }
  return dynDims;
}

static FailureOr<SmallVector<Value, 4>> buildAllocDynamicDimsBestEffort(
    OpBuilder &rewriter, Location loc, MemRefType allocTy, Value like) {
  FailureOr<SmallVector<Value, 4>> fromLike =
      buildAllocDynamicDimsFromLike(rewriter, loc, allocTy, like);
  return fromLike;
}

static FailureOr<SmallVector<Value, 4>> buildAllocDynamicDimsForScalarBroadcast(
    OpBuilder &rewriter, Location loc, MemRefType allocTy) {
  if (!allocTy.hasRank())
    return failure();
  SmallVector<Value, 4> dynDims;
  if (allocTy.hasStaticShape())
    return dynDims;
  for (int64_t i = 0, e = allocTy.getRank(); i < e; ++i) {
    if (!allocTy.isDynamicDim(i))
      continue;
    dynDims.push_back(rewriter.create<arith::ConstantIndexOp>(loc, 1));
  }
  return dynDims;
}

static MemRefType getIdentityMemRefType(MemRefType ty) {
  return MemRefType::get(ty.getShape(), ty.getElementType());
}

static bool hasIdentityMemRefLayout(MemRefType ty) {
  return ty.getLayout().isIdentity();
}

// Ensure runtime-call source descriptors are canonical contiguous memrefs.
// For strided/subview-like memrefs, materialize a contiguous copy first.
static FailureOr<Value> materializeContiguousMemRefForRead(
    PatternRewriter &rewriter, Location loc, Value src) {
  auto srcTy = dyn_cast<MemRefType>(src.getType());
  if (!srcTy || !srcTy.hasRank())
    return failure();
  if (hasIdentityMemRefLayout(srcTy))
    return src;

  MemRefType idTy = getIdentityMemRefType(srcTy);
  FailureOr<SmallVector<Value, 4>> maybeDynDims =
      buildAllocDynamicDimsFromLike(rewriter, loc, idTy, src);
  if (failed(maybeDynDims))
    return failure();
  Value packed = rewriter.create<memref::AllocOp>(loc, idTy, *maybeDynDims);
  rewriter.create<memref::CopyOp>(loc, src, packed);
  return packed;
}

static Value findBackingMemrefFromTensorViaUnrealized(Value tensorLike,
    function_ref<bool(Type)> elemTypePred) {
  Value cursor = tensorLike;
  SmallPtrSet<Operation *, 16> visited;
  while (auto cast = cursor.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (!visited.insert(cast).second)
      break;
    if (cast.getNumOperands() != 1)
      break;
    Value in = cast.getOperand(0);
    if (auto memTy = dyn_cast<MemRefType>(in.getType())) {
      if (elemTypePred(memTy.getElementType()))
        return in;
    }
    if (auto tTy = dyn_cast<ShapedType>(in.getType())) {
      if (!elemTypePred(tTy.getElementType()))
        break;
    } else {
      break;
    }
    cursor = in;
  }
  return Value();
}

static bool isPositShapedType(Type t) {
  auto st = dyn_cast<ShapedType>(t);
  return st && isa<posit::PositType>(st.getElementType());
}

struct LowerF32ToPositBitsCastChain final
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    // We match the LAST op in the chain: tensor<!posit> -> memref<i8>
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    auto dstMemref = dyn_cast<MemRefType>(op.getResult(0).getType());
    if (!dstMemref || !dstMemref.getElementType().isInteger(gCastChainStorageBits))
      return failure();

    auto srcTensorMid = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
    if (!srcTensorMid)
      return failure();
    if (srcTensorMid.getElementType().isF32())
      return failure();

    // Mid: tensor<f32> -> tensor<!posit>
    auto mid = op.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
    if (!mid || mid.getNumOperands() != 1 || mid.getNumResults() != 1)
      return failure();
    auto midSrc = dyn_cast<RankedTensorType>(mid.getOperand(0).getType());
    if (!midSrc || !midSrc.getElementType().isF32())
      return failure();

    // Pre: memref<f32> -> tensor<f32>
    auto pre = mid.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
    if (!pre || pre.getNumOperands() != 1 || pre.getNumResults() != 1)
      return failure();
    auto srcMemref = dyn_cast<MemRefType>(pre.getOperand(0).getType());
    if (!srcMemref || !srcMemref.getElementType().isF32())
      return failure();

    Location loc = op.getLoc();
    Value src = pre.getOperand(0); // ranked memref<f32>
    FailureOr<Value> maybePackedSrc =
        materializeContiguousMemRefForRead(rewriter, loc, src);
    if (failed(maybePackedSrc))
      return failure();
    src = *maybePackedSrc;

    // Allocate ranked memref<i8> for posit bits.
    MemRefType runtimeDstMemref = dstMemref;
    if (!hasIdentityMemRefLayout(runtimeDstMemref))
      runtimeDstMemref = getIdentityMemRefType(dstMemref);
    FailureOr<SmallVector<Value, 4>> maybeDynDims =
        buildAllocDynamicDimsFromLike(rewriter, loc, runtimeDstMemref, src);
    if (failed(maybeDynDims))
      return failure();
    Value dst = rewriter.create<memref::AllocOp>(
        loc, runtimeDstMemref, *maybeDynDims);

    // Cast to unranked for CRunner-style runtime call.
    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    auto unrankedI8 = UnrankedMemRefType::get(
        rewriter.getIntegerType(gCastChainStorageBits), 0);
    Value srcU = rewriter.create<memref::CastOp>(loc, unrankedF32, src);
    Value dstU = rewriter.create<memref::CastOp>(loc, unrankedI8, dst);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto callee = getOrInsertPositFromF32(module, rewriter);
    Value qalignKey = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    rewriter.create<func::CallOp>(
        loc, callee, TypeRange{}, ValueRange{srcU, dstU, qalignKey});

    // Replace last cast result with the real converted buffer.
    Value dstForReplace = dst;
    if (runtimeDstMemref != dstMemref)
      dstForReplace =
          rewriter.create<memref::CastOp>(loc, dstMemref, dst).getResult();
    rewriter.replaceOp(op, dstForReplace);

    // Clean up the now-dead casts.
    if (mid->use_empty())
      rewriter.eraseOp(mid);
    if (pre->use_empty())
      rewriter.eraseOp(pre);
    return success();
  }
};

struct LowerPositBitsToF32CastChain final
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    // Match the LAST op in the chain: tensor<f32> -> memref<f32>
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    auto dstMemref = dyn_cast<MemRefType>(op.getResult(0).getType());
    if (!dstMemref || !dstMemref.getElementType().isF32())
      return failure();

    auto srcTensorF32 = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
    if (!srcTensorF32 || !srcTensorF32.getElementType().isF32())
      return failure();

    // Mid: tensor<!posit> -> tensor<f32>
    auto mid = op.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
    if (!mid || mid.getNumOperands() != 1 || mid.getNumResults() != 1)
      return failure();
    auto midSrc = dyn_cast<RankedTensorType>(mid.getOperand(0).getType());
    if (!midSrc || midSrc.getElementType().isF32())
      return failure();

    // Pre: memref<i8> -> tensor<!posit>
    auto pre = mid.getOperand(0).getDefiningOp<UnrealizedConversionCastOp>();
    if (!pre || pre.getNumOperands() != 1 || pre.getNumResults() != 1)
      return failure();
    auto srcMemref = dyn_cast<MemRefType>(pre.getOperand(0).getType());
    if (!srcMemref || !srcMemref.getElementType().isInteger(gCastChainStorageBits))
      return failure();

    Location loc = op.getLoc();
    Value src = pre.getOperand(0); // ranked memref<i8>
    FailureOr<Value> maybePackedSrc =
        materializeContiguousMemRefForRead(rewriter, loc, src);
    if (failed(maybePackedSrc))
      return failure();
    src = *maybePackedSrc;

    // Allocate ranked memref<f32> for output floats.
    MemRefType runtimeDstMemref = dstMemref;
    if (!hasIdentityMemRefLayout(runtimeDstMemref))
      runtimeDstMemref = getIdentityMemRefType(dstMemref);
    FailureOr<SmallVector<Value, 4>> maybeDynDims =
        buildAllocDynamicDimsFromLike(rewriter, loc, runtimeDstMemref, src);
    if (failed(maybeDynDims))
      return failure();
    Value dst = rewriter.create<memref::AllocOp>(
        loc, runtimeDstMemref, *maybeDynDims);

    // Cast to unranked for runtime call.
    auto unrankedI8 = UnrankedMemRefType::get(
        rewriter.getIntegerType(gCastChainStorageBits), 0);
    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    Value srcU = rewriter.create<memref::CastOp>(loc, unrankedI8, src);
    Value dstU = rewriter.create<memref::CastOp>(loc, unrankedF32, dst);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto callee = getOrInsertPositToF32(module, rewriter);
    rewriter.create<func::CallOp>(loc, callee, TypeRange{}, ValueRange{srcU, dstU});

    Value dstForReplace = dst;
    if (runtimeDstMemref != dstMemref)
      dstForReplace =
          rewriter.create<memref::CastOp>(loc, dstMemref, dst).getResult();
    rewriter.replaceOp(op, dstForReplace);

    if (mid->use_empty())
      rewriter.eraseOp(mid);
    if (pre->use_empty())
      rewriter.eraseOp(pre);
    return success();
  }
};

// [FIX] Generic variant of f32->posit->...->memref cast-chain lowering.
// This handles extra placeholder tensor rank-casts between posit tensors, e.g.:
//   tensor<f32> -> tensor<!posit> -> tensor<?x!posit> -> memref<?xi8>
struct LowerF32ToPositBitsGenericCastChain final
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    auto dstMemref = dyn_cast<MemRefType>(op.getResult(0).getType());
    if (!dstMemref || !dstMemref.getElementType().isInteger(gCastChainStorageBits))
      return failure();
    if (!isPositShapedType(op.getOperand(0).getType()))
      return failure();

    SmallVector<UnrealizedConversionCastOp, 8> chain;
    UnrealizedConversionCastOp bridge;
    Value cursor = op.getOperand(0);
    while (true) {
      auto cast = cursor.getDefiningOp<UnrealizedConversionCastOp>();
      if (!cast || cast.getNumOperands() != 1 || cast.getNumResults() != 1)
        return failure();
      chain.push_back(cast);

      auto srcShaped = dyn_cast<ShapedType>(cast.getOperand(0).getType());
      auto dstShaped = dyn_cast<ShapedType>(cast.getResult(0).getType());
      if (!srcShaped || !dstShaped || !isa<posit::PositType>(dstShaped.getElementType()))
        return failure();

      Type srcElem = srcShaped.getElementType();
      if (srcElem.isF32()) {
        bridge = cast;
        break;
      }
      if (!isa<posit::PositType>(srcElem))
        return failure();
      cursor = cast.getOperand(0);
    }

    Value srcMemref = findBackingMemrefFromTensorViaUnrealized(
        bridge.getOperand(0), [](Type t) { return t.isF32(); });
    if (!srcMemref)
      return failure();
    FailureOr<Value> maybePackedSrc =
        materializeContiguousMemRefForRead(rewriter, op.getLoc(), srcMemref);
    if (failed(maybePackedSrc))
      return failure();
    srcMemref = *maybePackedSrc;

    Location loc = op.getLoc();
    auto srcMemrefTy = dyn_cast<MemRefType>(srcMemref.getType());
    MemRefType runtimeDstMemref = dstMemref;
    if (!hasIdentityMemRefLayout(runtimeDstMemref))
      runtimeDstMemref = getIdentityMemRefType(dstMemref);
    FailureOr<SmallVector<Value, 4>> maybeDynDims = failure();
    if (srcMemrefTy && srcMemrefTy.hasRank() && srcMemrefTy.getRank() == 0 &&
        runtimeDstMemref.hasRank() && runtimeDstMemref.getRank() > 0) {
      maybeDynDims =
          buildAllocDynamicDimsForScalarBroadcast(rewriter, loc, runtimeDstMemref);
    } else {
      maybeDynDims =
          buildAllocDynamicDimsBestEffort(
              rewriter, loc, runtimeDstMemref, srcMemref);
    }
    if (failed(maybeDynDims))
      return failure();
    Value dst = rewriter.create<memref::AllocOp>(
        loc, runtimeDstMemref, *maybeDynDims);

    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    auto unrankedI = UnrankedMemRefType::get(
        rewriter.getIntegerType(gCastChainStorageBits), 0);
    Value srcU = rewriter.create<memref::CastOp>(loc, unrankedF32, srcMemref);
    Value dstU = rewriter.create<memref::CastOp>(loc, unrankedI, dst);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto callee = getOrInsertPositFromF32(module, rewriter);
    Value qalignKey = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
    rewriter.create<func::CallOp>(
        loc, callee, TypeRange{}, ValueRange{srcU, dstU, qalignKey});

    Value dstForReplace = dst;
    if (runtimeDstMemref != dstMemref)
      dstForReplace =
          rewriter.create<memref::CastOp>(loc, dstMemref, dst).getResult();
    rewriter.replaceOp(op, dstForReplace);

    for (auto c : chain)
      if (c->use_empty())
        rewriter.eraseOp(c);
    return success();
  }
};

// [FIX] Generic variant of posit->...->f32->memref cast-chain lowering.
// Handles rank-only placeholder casts around the core posit->f32 bridge.
struct LowerPositBitsToF32GenericCastChain final
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    auto dstMemref = dyn_cast<MemRefType>(op.getResult(0).getType());
    if (!dstMemref || !dstMemref.getElementType().isF32())
      return failure();

    auto srcShapedTop = dyn_cast<ShapedType>(op.getOperand(0).getType());
    if (!srcShapedTop || !srcShapedTop.getElementType().isF32())
      return failure();

    SmallVector<UnrealizedConversionCastOp, 8> chain;
    UnrealizedConversionCastOp bridge;
    Value cursor = op.getOperand(0);
    while (true) {
      auto cast = cursor.getDefiningOp<UnrealizedConversionCastOp>();
      if (!cast || cast.getNumOperands() != 1 || cast.getNumResults() != 1)
        return failure();
      chain.push_back(cast);

      auto srcShaped = dyn_cast<ShapedType>(cast.getOperand(0).getType());
      auto dstShaped = dyn_cast<ShapedType>(cast.getResult(0).getType());
      if (!srcShaped || !dstShaped)
        return failure();

      Type srcElem = srcShaped.getElementType();
      Type dstElem = dstShaped.getElementType();
      if (isa<posit::PositType>(srcElem) && dstElem.isF32()) {
        bridge = cast;
        break;
      }

      if (srcElem == dstElem && (srcElem.isF32() || isa<posit::PositType>(srcElem))) {
        cursor = cast.getOperand(0);
        continue;
      }
      return failure();
    }

    Value srcMemref = findBackingMemrefFromTensorViaUnrealized(
        bridge.getOperand(0), [](Type t) {
          return t.isInteger(gCastChainStorageBits);
        });
    if (!srcMemref)
      return failure();
    FailureOr<Value> maybePackedSrc =
        materializeContiguousMemRefForRead(rewriter, op.getLoc(), srcMemref);
    if (failed(maybePackedSrc))
      return failure();
    srcMemref = *maybePackedSrc;

    Location loc = op.getLoc();
    auto srcMemrefTy = dyn_cast<MemRefType>(srcMemref.getType());
    MemRefType runtimeDstMemref = dstMemref;
    if (!hasIdentityMemRefLayout(runtimeDstMemref))
      runtimeDstMemref = getIdentityMemRefType(dstMemref);
    FailureOr<SmallVector<Value, 4>> maybeDynDims = failure();
    if (srcMemrefTy && srcMemrefTy.hasRank() && srcMemrefTy.getRank() == 0 &&
        runtimeDstMemref.hasRank() && runtimeDstMemref.getRank() > 0) {
      maybeDynDims =
          buildAllocDynamicDimsForScalarBroadcast(rewriter, loc, runtimeDstMemref);
    } else {
      maybeDynDims =
          buildAllocDynamicDimsBestEffort(
              rewriter, loc, runtimeDstMemref, srcMemref);
    }
    if (failed(maybeDynDims))
      return failure();
    Value dst = rewriter.create<memref::AllocOp>(
        loc, runtimeDstMemref, *maybeDynDims);

    auto unrankedI = UnrankedMemRefType::get(
        rewriter.getIntegerType(gCastChainStorageBits), 0);
    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    Value srcU = rewriter.create<memref::CastOp>(loc, unrankedI, srcMemref);
    Value dstU = rewriter.create<memref::CastOp>(loc, unrankedF32, dst);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto callee = getOrInsertPositToF32(module, rewriter);
    rewriter.create<func::CallOp>(loc, callee, TypeRange{}, ValueRange{srcU, dstU});

    Value dstForReplace = dst;
    if (runtimeDstMemref != dstMemref)
      dstForReplace =
          rewriter.create<memref::CastOp>(loc, dstMemref, dst).getResult();
    rewriter.replaceOp(op, dstForReplace);

    for (auto c : chain)
      if (c->use_empty())
        rewriter.eraseOp(c);
    return success();
  }
};

// [FIX] Materialize tensor arith.constant feeding an unrealized cast to memref
// as krnl.global so krnl-to-llvm won't see illegal tensor constants.
struct LowerTensorConstCastToKrnlGlobal final
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    auto dstMemRefTy = dyn_cast<MemRefType>(op.getResult(0).getType());
    if (!dstMemRefTy || !dstMemRefTy.hasStaticShape())
      return failure();

    auto cst = op.getOperand(0).getDefiningOp<arith::ConstantOp>();
    if (!cst)
      return failure();

    auto srcTensorTy = dyn_cast<TensorType>(cst.getType());
    if (!srcTensorTy || !srcTensorTy.hasStaticShape())
      return failure();

    Attribute valueAttr = cst.getValue();
    if (!isa<ElementsAttr>(valueAttr))
      return failure();

    if (srcTensorTy.getElementType() != dstMemRefTy.getElementType())
      return failure();
    if (srcTensorTy.getShape() != dstMemRefTy.getShape())
      return failure();

    OperationState st(op.getLoc(), "krnl.global");
    st.addTypes(dstMemRefTy);
    st.addAttribute(
        "name",
        rewriter.getStringAttr("posit_fconst_" +
                               std::to_string(gTensorConstGlobalId++)));
    st.addAttribute("shape", rewriter.getI64ArrayAttr(dstMemRefTy.getShape()));
    st.addAttribute("value", valueAttr);

    Operation *global = rewriter.create(st);
    rewriter.replaceOp(op, global->getResult(0));
    if (cst->use_empty())
      rewriter.eraseOp(cst);
    return success();
  }
};

//0201 新增onnx相關 op 未處理的優化

// [FIX] onnx.Return -> func.return
struct LowerONNXReturnToFuncReturn final
    : public mlir::OpConversionPattern<mlir::ONNXReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ONNXReturnOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Use adaptor operands so type conversion can feed converted values when
    // the pass is running after ONNX/Krnl lowering. Falling back to the original
    // operands can leave tensor-typed onnx.Return behind and later krnl->llvm
    // reports an illegal onnx.Return.
    rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, adaptor.getOperands());
    return mlir::success();
  }
};

// [FIX] erase onnx.NoValue (only when dead)
struct EraseONNXNoValueIfDead final
    : public mlir::OpConversionPattern<mlir::ONNXNoneOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ONNXNoneOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (!op.getResult().use_empty())
      return mlir::failure(); // still used -> you must lower its users first
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// 0203 erase onnx.EntryPoint
struct EraseONNXEntryPoint final
    : public mlir::OpConversionPattern<mlir::ONNXEntryPointOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::ONNXEntryPointOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// 0204  [FIX] Lower `builtin.unrealized_conversion_cast` between f32 tensor <-> posit tensor
/*struct LowerUnrealizedCastToPositConvert final
    : public mlir::OpConversionPattern<mlir::UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::UnrealizedConversionCastOp op, OpAdaptor adaptor,
      mlir::ConversionPatternRewriter &rewriter) const override {
    // Only handle 1 -> 1 casts.
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "only supports 1->1 casts");

    Type srcTy = op.getOperand(0).getType();
    Type dstTy = op.getResult(0).getType();

    auto srcRtt = llvm::dyn_cast<RankedTensorType>(srcTy);
    auto dstRtt = llvm::dyn_cast<RankedTensorType>(dstTy);
    if (!srcRtt || !dstRtt)
      return rewriter.notifyMatchFailure(op, "expect ranked tensor cast");

    Type srcElem = srcRtt.getElementType();
    Type dstElem = dstRtt.getElementType();

    // f32 tensor -> posit tensor  ==>  posit.from_f32
    if (llvm::isa<Float32Type>(srcElem) && llvm::isa<posit::PositType>(dstElem)) {
      auto conv = rewriter.create<posit::FromF32Op>(op.getLoc(), dstTy, op.getOperand(0));
      rewriter.replaceOp(op, conv.getResult());
      return mlir::success();
    }

    // posit tensor -> f32 tensor  ==>  posit.to_f32
    if (llvm::isa<posit::PositType>(srcElem) && llvm::isa<Float32Type>(dstElem)) {
      auto conv = rewriter.create<posit::ToF32Op>(op.getLoc(), dstTy, op.getOperand(0));
      rewriter.replaceOp(op, conv.getResult());
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unhandled unrealized cast types");
  }
};*/

struct LowerUnrealizedCastToPositConvert final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "only supports 1->1 casts");

    Type srcTy = op.getOperand(0).getType();
    Type dstTy = op.getResult(0).getType();

    auto srcShaped = llvm::dyn_cast<ShapedType>(srcTy);
    auto dstShaped = llvm::dyn_cast<ShapedType>(dstTy);
    if (!srcShaped || !dstShaped)
      return rewriter.notifyMatchFailure(op, "expect shaped tensor types");

    Type srcElem = srcShaped.getElementType();
    Type dstElem = dstShaped.getElementType();

    // For rank-only casts (e.g. tensor<1x10x!posit> <-> tensor<*x!posit>), convert
    // directly in converted memref domain.
    if (srcElem == dstElem) {
      if (adaptor.getOperands().empty())
        return rewriter.notifyMatchFailure(op, "missing converted operand");
      Value convertedSrc = adaptor.getOperands()[0];
      Type convertedDstTy = getTypeConverter()->convertType(dstTy);
      if (!convertedDstTy)
        return rewriter.notifyMatchFailure(op, "failed to convert destination type");

      if (convertedSrc.getType() == convertedDstTy) {
        rewriter.replaceOp(op, convertedSrc);
        return success();
      }

      bool srcIsMemRef =
          isa<MemRefType, UnrankedMemRefType>(convertedSrc.getType());
      bool dstIsMemRef = isa<MemRefType, UnrankedMemRefType>(convertedDstTy);
      if (srcIsMemRef && dstIsMemRef) {
        auto cast =
            rewriter.create<memref::CastOp>(op.getLoc(), convertedDstTy, convertedSrc);
        rewriter.replaceOp(op, cast.getResult());
        return success();
      }

      return rewriter.notifyMatchFailure(
          op, "same-element cast is not memref-castable");
    }

    // tensor<f32> -> tensor<!posit>
    if (llvm::isa<Float32Type>(srcElem) && llvm::isa<posit::PositType>(dstElem)) {
      if (!llvm::isa<RankedTensorType>(srcTy) || !llvm::isa<RankedTensorType>(dstTy))
        return rewriter.notifyMatchFailure(
            op, "f32->posit cast requires ranked tensors");
      auto conv = rewriter.create<posit::FromF32Op>(op.getLoc(), dstTy, op.getOperand(0));
      rewriter.replaceOp(op, conv.getResult());
      return success();
    }

    // tensor<!posit> -> tensor<f32>
    if (llvm::isa<posit::PositType>(srcElem) && llvm::isa<Float32Type>(dstElem)) {
      if (!llvm::isa<RankedTensorType>(srcTy) || !llvm::isa<RankedTensorType>(dstTy))
        return rewriter.notifyMatchFailure(
            op, "posit->f32 cast requires ranked tensors");
      auto conv = rewriter.create<posit::ToF32Op>(op.getLoc(), dstTy, op.getOperand(0));
      rewriter.replaceOp(op, conv.getResult());
      return success();
    }

    return rewriter.notifyMatchFailure(op, "unhandled unrealized cast types");
  }
};

struct EraseDeadUnrealizedCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    if (!op.getResults().empty() && !op.getResult(0).use_empty())
      return rewriter.notifyMatchFailure(op, "cast is still used");
    rewriter.eraseOp(op);
    return success();
  }
};

// [FIX] Lower tensor.from_elements + unrealized_cast(memref) to concrete
// memref allocation/stores so tensor ops do not leak into krnl->llvm.
struct LowerTensorFromElementsCastToMemref final
    : public OpRewritePattern<UnrealizedConversionCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(UnrealizedConversionCastOp op,
      PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 1 || op.getNumResults() != 1)
      return failure();

    auto dstMemrefTy = dyn_cast<MemRefType>(op.getResult(0).getType());
    if (!dstMemrefTy || !dstMemrefTy.hasRank() || dstMemrefTy.getRank() != 1 ||
        !dstMemrefTy.hasStaticShape())
      return failure();

    auto fromElems = op.getOperand(0).getDefiningOp<tensor::FromElementsOp>();
    if (!fromElems)
      return failure();
    auto srcTensorTy = dyn_cast<RankedTensorType>(fromElems.getType());
    if (!srcTensorTy || srcTensorTy.getRank() != 1 || !srcTensorTy.hasStaticShape())
      return failure();

    int64_t len = srcTensorTy.getShape()[0];
    if (len != dstMemrefTy.getShape()[0])
      return failure();
    if (srcTensorTy.getElementType() != dstMemrefTy.getElementType())
      return failure();
    if (static_cast<int64_t>(fromElems.getElements().size()) != len)
      return failure();

    Value alloc = rewriter.create<memref::AllocOp>(op.getLoc(), dstMemrefTy);
    for (int64_t i = 0; i < len; ++i) {
      Value idx = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), i);
      rewriter.create<memref::StoreOp>(
          op.getLoc(), fromElems.getElements()[i], alloc, ValueRange{idx});
    }

    rewriter.replaceOp(op, alloc);
    if (fromElems->use_empty())
      rewriter.eraseOp(fromElems);
    return success();
  }
};

// [FIX] Prevent tensor.dim on placeholder posit tensors from leaking to
// krnl->llvm by redirecting dim queries to the underlying memref source.
struct LowerTensorDimOnCastedMemref final
    : public OpRewritePattern<tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(
      tensor::DimOp op, PatternRewriter &rewriter) const override {
    Value memrefSrc = op.getSource();
    bool changed = true;
    while (changed) {
      changed = false;
      if (auto cast = memrefSrc.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (cast.getNumOperands() != 1)
          break;
        memrefSrc = cast.getOperand(0);
        changed = true;
        continue;
      }
      if (auto memrefCast = memrefSrc.getDefiningOp<memref::CastOp>()) {
        memrefSrc = memrefCast.getSource();
        changed = true;
        continue;
      }
    }

    if (!isa<MemRefType, UnrankedMemRefType>(memrefSrc.getType()))
      return failure();

    Value dim = rewriter.create<memref::DimOp>(
        op.getLoc(), memrefSrc, op.getIndex());
    rewriter.replaceOp(op, dim);
    return success();
  }
};

static bool isPositF32BridgeRuntime(StringRef callee) {
  return callee.starts_with("_mlir_ciface_posit_from_f32_") ||
         callee.starts_with("_mlir_ciface_posit_to_f32_");
}

static LogicalResult verifyPositRuntimeDescriptorCalls(ModuleOp module) {
  bool ok = true;
  module.walk([&](func::CallOp callOp) {
    StringRef callee = callOp.getCallee();
    if (!isPositF32BridgeRuntime(callee))
      return;

    const bool isFrom =
        callee.starts_with("_mlir_ciface_posit_from_f32_");
    const unsigned expectedOperands = isFrom ? 3u : 2u;
    if (callOp.getNumOperands() != expectedOperands) {
      callOp.emitError("posit runtime bridge has unexpected operand count");
      ok = false;
      return;
    }

    for (unsigned i = 0; i < 2; ++i) {
      Value arg = callOp.getOperand(i);
      Value ranked = arg;
      if (auto cast = ranked.getDefiningOp<memref::CastOp>())
        ranked = cast.getSource();

      auto memTy = dyn_cast<MemRefType>(ranked.getType());
      if (!memTy || !memTy.hasRank()) {
        callOp.emitError("runtime bridge arg ")
            << i << " must come from ranked memref source";
        ok = false;
        continue;
      }

      if (!hasIdentityMemRefLayout(memTy)) {
        callOp.emitError("runtime bridge arg ")
            << i
            << " has non-identity memref layout; descriptor may be unsafe";
        ok = false;
      }

      if (i == 1 && !ranked.getDefiningOp<memref::AllocOp>()) {
        callOp.emitError("runtime bridge destination arg must be memref.alloc");
        ok = false;
      }

      Type elemTy = memTy.getElementType();
      if (isFrom) {
        if (i == 0 && !elemTy.isF32()) {
          callOp.emitError("posit_from_f32 source must be memref<...xf32>");
          ok = false;
        }
        if (i == 1 && !elemTy.isInteger(gCastChainStorageBits)) {
          callOp.emitError("posit_from_f32 destination must match posit storage bits i")
              << gCastChainStorageBits;
          ok = false;
        }
      } else {
        if (i == 0 && !elemTy.isInteger(gCastChainStorageBits)) {
          callOp.emitError("posit_to_f32 source must match posit storage bits i")
              << gCastChainStorageBits;
          ok = false;
        }
        if (i == 1 && !elemTy.isF32()) {
          callOp.emitError("posit_to_f32 destination must be memref<...xf32>");
          ok = false;
        }
      }
    }

    if (isFrom) {
      Type keyTy = callOp.getOperand(2).getType();
      if (!llvm::isa<IntegerType, IndexType>(keyTy)) {
        callOp.emitError("posit_from_f32 qalign_key must be integer/index");
        ok = false;
      }
    }
  });
  return success(ok);
}


struct ConvertPositToKrnlPass
    : public PassWrapper<ConvertPositToKrnlPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertPositToKrnlPass)

  ConvertPositToKrnlPass() = default;
  ConvertPositToKrnlPass(unsigned nbits, unsigned es) : nbits(nbits), es(es) {}

  StringRef getArgument() const final { return "convert-posit-to-krnl"; }
  StringRef getDescription() const final {
    return "Lower Posit ops/types to Krnl level (memref + runtime calls).";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<posit::PositDialect,
                    func::FuncDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    tensor::TensorDialect,
                    mlir::KrnlDialect>();
  }

  unsigned nbits = 8;
  unsigned es = 0;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    bool hasPositOp = false;
    bool hasPositCastChain = false;
    bool hasResidualONNXOp = false;
    module.walk([&](Operation *op) {
      if (hasPositOp && hasResidualONNXOp && hasPositCastChain)
        return;
      StringRef ns = op->getName().getDialectNamespace();
      if (ns == "posit")
        hasPositOp = true;
      if (ns == "onnx" && !llvm::isa<ONNXEntryPointOp, ONNXNoneOp, ONNXReturnOp>(op))
        hasResidualONNXOp = true;
      if (auto ucast = llvm::dyn_cast<UnrealizedConversionCastOp>(op)) {
        auto hasPositElem = [](Type t) {
          if (auto st = llvm::dyn_cast<ShapedType>(t))
            return llvm::isa<posit::PositType>(st.getElementType());
          return llvm::isa<posit::PositType>(t);
        };
        if ((ucast.getNumOperands() == 1 && hasPositElem(ucast.getOperand(0).getType())) ||
            (ucast.getNumResults() == 1 && hasPositElem(ucast.getResult(0).getType())))
          hasPositCastChain = true;
      }
    });
    MLIRContext &ctx = getContext();

    if (!hasPositOp && !hasPositCastChain) {
      // This pass is invoked twice in the build pipeline. In the second
      // invocation, after convert-onnx-to-krnl, some models may have no Posit
      // ops left but still retain ONNX terminator/metadata ops. Clean those up
      // here so krnl->llvm never sees onnx.Return.
      if (!hasResidualONNXOp) {
        RewritePatternSet cleanupPatterns(&ctx);
        cleanupPatterns.add<LowerONNXReturnToFuncReturn, EraseONNXEntryPoint,
                            EraseONNXNoValueIfDead>(&ctx);
        if (failed(applyPatternsGreedily(module, std::move(cleanupPatterns))))
          signalPassFailure();
      }
      return;
    }
    const bool castOnlyMode = !hasPositOp && hasPositCastChain;
    const bool mixedWithONNX = hasResidualONNXOp;

    unsigned localNbits = nbits;
    unsigned localEs = es;
    auto maybeAdoptPositTypeConfig = [&](Type ty) -> bool {
      if (auto shapedTy = dyn_cast<ShapedType>(ty))
        ty = shapedTy.getElementType();
      if (auto positTy = dyn_cast<posit::PositType>(ty)) {
        localNbits = positTy.getNbits();
        localEs = positTy.getEs();
        return true;
      }
      return false;
    };
    if (localNbits == 8 && localEs == 0) {
      bool foundPositType = false;
      module.walk([&](Operation *op) {
        if (foundPositType)
          return WalkResult::interrupt();
        for (Value operand : op->getOperands()) {
          if (maybeAdoptPositTypeConfig(operand.getType())) {
            foundPositType = true;
            return WalkResult::interrupt();
          }
        }
        for (Type resultTy : op->getResultTypes()) {
          if (maybeAdoptPositTypeConfig(resultTy)) {
            foundPositType = true;
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
    }
    PositToKrnlTypeConverter typeConverter(localNbits, localEs, &ctx);
    gCastChainPositNbits = localNbits;
    gCastChainPositEs = localEs;
    gCastChainStorageBits = getPositStorageBitWidth(localNbits);

    if (castOnlyMode) {
      RewritePatternSet castChainPatterns(&ctx);
      castChainPatterns.add<LowerF32ToPositBitsCastChain,
                            LowerPositBitsToF32CastChain,
                            LowerF32ToPositBitsGenericCastChain,
                            LowerPositBitsToF32GenericCastChain,
                            LowerTensorConstCastToKrnlGlobal,
                            LowerTensorFromElementsCastToMemref,
                            LowerTensorDimOnCastedMemref>(&ctx);
      if (failed(applyPatternsGreedily(module, std::move(castChainPatterns))))
        signalPassFailure();
      if (failed(verifyPositRuntimeDescriptorCalls(module)))
        signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&ctx);
    if (!mixedWithONNX) {
      populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                     typeConverter);
      populateCallOpTypeConversionPattern(patterns, typeConverter);
      populateReturnOpTypeConversionPattern(patterns, typeConverter);
      populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);
    }

    populatePositToKrnlConversionPattern(
        typeConverter, patterns, &ctx, localNbits, localEs);

    // In mixed ONNX+Posit mode, keep ONNXReturn alive so the later
    // ONNX->Krnl pipeline can convert the function signature and return types
    // consistently. Replacing it here with func.return(memref) while the
    // enclosing func still returns tensor leads to tensor/memref mismatches.
    if (!mixedWithONNX)
      patterns.add<LowerONNXReturnToFuncReturn>(typeConverter, &ctx);
    patterns.add<EraseONNXNoValueIfDead, EraseONNXEntryPoint>(typeConverter,
                                                              &ctx);
    patterns.add<LowerTensorConstCastToKrnlGlobal>(&ctx);
    patterns.add<LowerF32ToPositBitsCastChain, LowerPositBitsToF32CastChain,
                 LowerF32ToPositBitsGenericCastChain,
                 LowerPositBitsToF32GenericCastChain>(&ctx);
    patterns.add<LowerUnrealizedCastToPositConvert>(typeConverter, &ctx);
    patterns.add<EraseDeadUnrealizedCast>(typeConverter, &ctx);

    ConversionTarget target(ctx);
    target.addLegalDialect<memref::MemRefDialect,
                           arith::ArithDialect, tensor::TensorDialect,
                           mlir::KrnlDialect,
                           func::FuncDialect>();

    // dynamic legality
    if (!mixedWithONNX) {
      target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
        return typeConverter.isSignatureLegal(op.getFunctionType()) &&
               typeConverter.isLegal(&op.getBody());
      });
      target.addDynamicallyLegalOp<func::CallOp>(
          [&](func::CallOp op) { return typeConverter.isLegal(op); });
      target.addDynamicallyLegalOp<func::ReturnOp>(
          [&](func::ReturnOp op) { return typeConverter.isLegal(op); });
    } else {
      target.addLegalOp<func::FuncOp, func::CallOp, func::ReturnOp>();
    }

    // [FIX] Don't allow UnrealizedConversionCastOp to leak past this pass.
    // We provide explicit rewrite patterns for the remaining cast chains.
    if (!mixedWithONNX)
      target.addIllegalOp<UnrealizedConversionCastOp>();
    else
      target.addLegalOp<UnrealizedConversionCastOp>();
    if (mixedWithONNX) {
      // Keep ONNXReturn/ONNXNone alive for later ONNX->Krnl lowering when the
      // mixed graph still contains tensor-typed ONNX ops. This avoids
      // func.return(memref) mismatching a tensor-returning function signature.
      target.addIllegalOp<ONNXEntryPointOp>();
      target.addLegalOp<ONNXReturnOp, ONNXNoneOp>();
    } else {
      target.addIllegalOp<ONNXReturnOp, ONNXNoneOp, ONNXEntryPointOp>();
    }

    // [MOD][2026-01-14] 把 Posit ops 變 illegal，逼它們一定要被轉掉
    target.addIllegalOp<posit::ConstantOp, posit::AddOp, posit::SubOp,
                        posit::MulOp, posit::DivOp,
                        posit::FromF32Op, posit::ToF32Op,
                        posit::DequantizeLinearOp, posit::ReluOp,
                        posit::ClipOp,
                        posit::Conv2DOp, posit::MaxPool2DOp, posit::ReshapeOp,
                        posit::FlattenOp, posit::GemmOp,
                        posit::ReduceMeanOp>();

    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (mixedWithONNX && op->getName().getDialectNamespace() == "onnx")
        return true;
      return typeConverter.isLegal(op);
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // In mixed ONNX+Posit mode, unrealized_conversion_cast is legal in the
    // conversion target, so apply these chain-lowering rewrites greedily here.
    RewritePatternSet castChainPatterns(&ctx);
    castChainPatterns.add<LowerF32ToPositBitsCastChain,
        LowerPositBitsToF32CastChain, LowerF32ToPositBitsGenericCastChain,
        LowerPositBitsToF32GenericCastChain, LowerTensorFromElementsCastToMemref,
        LowerTensorDimOnCastedMemref>(&ctx);
    if (failed(applyPatternsGreedily(module, std::move(castChainPatterns))))
      signalPassFailure();

    // [FIX] Finalize tensor constant encoding path:
    // arith.constant(tensor) -> unrealized_conversion_cast(memref) must become
    // krnl.global, otherwise krnl-to-llvm sees illegal tensor constants.
    bool rewrittenTensorConstCasts = true;
    while (rewrittenTensorConstCasts) {
      rewrittenTensorConstCasts = false;
      SmallVector<UnrealizedConversionCastOp, 8> castsToRewrite;
      module.walk([&](UnrealizedConversionCastOp castOp) {
        if (castOp.getNumOperands() != 1 || castOp.getNumResults() != 1)
          return;
        auto dstMemRefTy = dyn_cast<MemRefType>(castOp.getResult(0).getType());
        if (!dstMemRefTy || !dstMemRefTy.hasStaticShape())
          return;
        auto cst = castOp.getOperand(0).getDefiningOp<arith::ConstantOp>();
        if (!cst)
          return;
        auto srcTensorTy = dyn_cast<TensorType>(cst.getType());
        if (!srcTensorTy || !srcTensorTy.hasStaticShape())
          return;
        if (!isa<ElementsAttr>(cst.getValue()))
          return;
        if (srcTensorTy.getElementType() != dstMemRefTy.getElementType())
          return;
        if (srcTensorTy.getShape() != dstMemRefTy.getShape())
          return;
        castsToRewrite.push_back(castOp);
      });

      for (UnrealizedConversionCastOp castOp : castsToRewrite) {
        auto cst = castOp.getOperand(0).getDefiningOp<arith::ConstantOp>();
        OpBuilder b(castOp);
        OperationState st(castOp.getLoc(), "krnl.global");
        auto dstMemRefTy = cast<MemRefType>(castOp.getResult(0).getType());
        st.addTypes(dstMemRefTy);
        st.addAttribute(
            "name",
            b.getStringAttr("posit_fconst_" +
                            std::to_string(gTensorConstGlobalId++)));
        st.addAttribute("shape", b.getI64ArrayAttr(dstMemRefTy.getShape()));
        st.addAttribute("value", cst.getValue());
        Operation *global = b.create(st);

        castOp.getResult(0).replaceAllUsesWith(global->getResult(0));
        castOp.erase();
        if (cst && cst->use_empty())
          cst.erase();
        rewrittenTensorConstCasts = true;
      }
    }

    // Clean up dead tensor shape constants left behind by reshape lowering.
    // krnl-to-llvm does not legalize tensor arith.constant.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<arith::ConstantOp, 8> deadTensorConsts;
      module.walk([&](arith::ConstantOp cst) {
        if (!cst.getResult().use_empty())
          return;
        if (!isa<TensorType>(cst.getType()))
          return;
        deadTensorConsts.push_back(cst);
      });
      for (arith::ConstantOp cst : deadTensorConsts) {
        cst.erase();
        changed = true;
      }
    }

    if (failed(verifyPositRuntimeDescriptorCalls(module)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertPositToKrnlPass(unsigned nbits,
                                                         unsigned es) {
  return std::make_unique<ConvertPositToKrnlPass>(nbits, es);
}

std::unique_ptr<mlir::Pass> createConvertPositToKrnlPass() {
  return createConvertPositToKrnlPass(/*nbits=*/8, /*es=*/0);
}

} // namespace onnx_mlir
