#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

// [FIX][signed-int-attr] Avoid IntegerAttr::getInt() assertion on signed integer types (si*)
// Use APInt accessors directly so it works for signless/signed/unsigned integers.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "src/Dialect/Posit/PositOps.h"
#include "src/Conversion/PositToKrnl/TypeConverters.hpp"


#include "mlir/IR/BuiltinOps.h"

// krnl.global
#include "src/Dialect/Krnl/KrnlOps.hpp"

#include <string>

using namespace mlir;

namespace onnx_mlir {
namespace {

// [EXTEND] Current pass-configured posit format. Set in populatePositToKrnlConversionPattern().
static unsigned gPositNbits = 8;
static unsigned gPositEs = 0;
static unsigned gPositStorageBits = 8;

static IntegerType getPositStorageIntType(OpBuilder &b) {
  return b.getIntegerType(gPositStorageBits);
}

static UnrankedMemRefType getUnrankedPositBitsType(OpBuilder &b) {
  return UnrankedMemRefType::get(getPositStorageIntType(b), 0);
}

static std::string getPositRuntimeCompandRegisterName() {
  return "posit_register_tensor_compand_p" +
         std::to_string(gPositNbits) + "e" + std::to_string(gPositEs);
}

static std::string getPositRuntimeConstMetaRegisterName() {
  return "posit_register_tensor_constmeta_p" +
         std::to_string(gPositNbits) + "e" + std::to_string(gPositEs);
}

static std::string getPositRuntimeConstMetaChannelRegisterName() {
  return "posit_register_tensor_constmeta_channel_p" +
         std::to_string(gPositNbits) + "e" + std::to_string(gPositEs);
}

static std::string getPositRuntimeName(StringRef stem) {
  return ("posit_" + stem + "_p" + std::to_string(gPositNbits) +
          "e" + std::to_string(gPositEs))
      .str();
}

// Per-op format helpers for mixed-format lowering.
static std::pair<unsigned, unsigned> getFormatFromPositType(Type ty) {
  if (auto shaped = llvm::dyn_cast<ShapedType>(ty))
    ty = shaped.getElementType();
  if (auto pt = llvm::dyn_cast<posit::PositType>(ty))
    return {pt.getNbits(), pt.getEs()};
  return {gPositNbits, gPositEs};
}

static std::string getPositRuntimeNameFor(StringRef stem,
                                          unsigned nbits, unsigned es) {
  return ("posit_" + stem + "_p" + std::to_string(nbits) +
          "e" + std::to_string(es)).str();
}

static UnrankedMemRefType getUnrankedITypeFor(OpBuilder &b, unsigned storageBits) {
  return UnrankedMemRefType::get(b.getIntegerType(storageBits), 0);
}


// 0202
// [FIX] --- helper to read IntegerAttr safely (si64/signless/index) ---
static int64_t getInt64Safe(mlir::Attribute a) {
  auto ia = llvm::dyn_cast<mlir::IntegerAttr>(a);
  if (!ia)
    llvm_unreachable("expected IntegerAttr");
  // Avoid IntegerAttr::getInt() which asserts on signed/unsigned integer types.
  return ia.getValue().getSExtValue();
}

static int64_t getInt64Safe(mlir::IntegerAttr ia) {
  return ia.getValue().getSExtValue();
}

static func::FuncOp getOrCreateFunc(ModuleOp module, StringRef name,
                                   FunctionType type) {
  if (auto f = module.lookupSymbol<func::FuncOp>(name))
    return f;
  OpBuilder b(module.getBodyRegion());
  auto fn = b.create<func::FuncOp>(module.getLoc(), name, type);
  fn.setPrivate();
  return fn;
}

static Value cstI64(OpBuilder &b, Location loc, int64_t v) {
  return b.create<arith::ConstantIntOp>(loc, v, 64);
}

static Value cstF32(OpBuilder &b, Location loc, float v) {
  return b.create<arith::ConstantFloatOp>(loc,  b.getF32Type(),APFloat(v));
}

static Value cstF64(OpBuilder &b, Location loc, double v) {
  return b.create<arith::ConstantFloatOp>(loc, b.getF64Type(), APFloat(v));
}

static Value castToUnranked(OpBuilder &b, Location loc, Value ranked,
                            Type unrankedTy) {
  if (ranked.getType() == unrankedTy)
    return ranked;
  return b.create<memref::CastOp>(loc, unrankedTy, ranked);
}

static Value createKrnlGlobalFromDense(ConversionPatternRewriter &rewriter,
                                       Location loc, DenseElementsAttr dense,
                                       StringRef namePrefix) {
  auto rtt = llvm::dyn_cast<RankedTensorType>(dense.getType());
  if (!rtt || !rtt.hasStaticShape())
    return Value();
  auto memrefTy = MemRefType::get(rtt.getShape(), rtt.getElementType());
  static int64_t gid = 0;
  std::string gname = (namePrefix.str() + "_" + std::to_string(gid++));

  OperationState st(loc, "krnl.global");
  st.addTypes(memrefTy);
  st.addAttribute("name", rewriter.getStringAttr(gname));
  st.addAttribute("shape", rewriter.getI64ArrayAttr(memrefTy.getShape()));
  st.addAttribute("value", dense);
  Operation *global = rewriter.create(st);
  return global->getResult(0);
}

static DenseIntElementsAttr makeI64Dense1D(OpBuilder &b, ArrayRef<int64_t> vals) {
  auto ty = RankedTensorType::get({static_cast<int64_t>(vals.size())}, b.getI64Type());
  SmallVector<APInt, 4> ap;
  ap.reserve(vals.size());
  for (int64_t v : vals)
    ap.push_back(APInt(64, static_cast<uint64_t>(v), true));
  return DenseIntElementsAttr::get(ty, ap);
}

static Value stripUnrealizedMemrefCast(Value v, Type expectedElemTy) {
  Value cur = v;
  while (auto cast = cur.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1)
      break;
    cur = cast.getOperand(0);
  }
  auto memTy = llvm::dyn_cast<MemRefType>(cur.getType());
  if (!memTy || memTy.getElementType() != expectedElemTy)
    return v;
  return cur;
}

static Value stripUnrealizedCast(Value v) {
  Value cur = v;
  while (auto cast = cur.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getNumOperands() != 1)
      break;
    cur = cast.getOperand(0);
  }
  return cur;
}

static Value stripMemrefAndUnrealizedCast(Value v) {
  Value cur = v;
  while (true) {
    if (auto mcast = cur.getDefiningOp<memref::CastOp>()) {
      cur = mcast.getSource();
      continue;
    }
    if (auto ucast = cur.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (ucast.getNumOperands() != 1)
        break;
      cur = ucast.getOperand(0);
      continue;
    }
    break;
  }
  return cur;
}

static Value allocLikeValue(ConversionPatternRewriter &rewriter, Location loc,
                            MemRefType outTy, Value like,
                            ArrayRef<int64_t> outDimToLikeDim = {}) {
  if (!outTy)
    return Value();
  if (!outTy.hasRank())
    return Value();
  if (!outTy.hasStaticShape()) {
    auto likeTy = llvm::dyn_cast<MemRefType>(like.getType());
    if (!likeTy || !likeTy.hasRank())
      return Value();
    SmallVector<Value, 4> dynDims;
    for (int64_t i = 0, e = outTy.getRank(); i < e; ++i) {
      if (!outTy.isDynamicDim(i))
        continue;
      int64_t likeDim = i;
      if (!outDimToLikeDim.empty()) {
        if (i >= static_cast<int64_t>(outDimToLikeDim.size()))
          return Value();
        likeDim = outDimToLikeDim[i];
      }
      if (likeDim < 0 || likeDim >= likeTy.getRank())
        return Value();
      dynDims.push_back(rewriter.create<memref::DimOp>(loc, like, likeDim));
    }
    return rewriter.create<memref::AllocOp>(loc, outTy, dynDims);
  }
  return rewriter.create<memref::AllocOp>(loc, outTy);
}

// In mixed ONNX+Posit mode, some boundaries can remain as:
//   memref<f32> -> tensor<f32> -> tensor<!posit> -> memref<iN>
// If consumed directly, memref<iN> may just reinterpret raw f32 bytes.
// Materialize a real posit_from_f32 conversion when this cast chain is detected.
// nbits=0 → fall back to global format. Pass explicit values for mixed-format ops.
static Value materializePositFromF32CastChainIfNeeded(
    ConversionPatternRewriter &rewriter, Location loc, Value v,
    unsigned nbits = 0, unsigned es = 0) {
  if (!nbits) { nbits = gPositNbits; es = gPositEs; }
  unsigned storageBits = getPositStorageBitWidth(nbits);

  Value cur = v;
  while (auto mcast = cur.getDefiningOp<memref::CastOp>())
    cur = mcast.getSource();

  auto dstTy = llvm::dyn_cast<MemRefType>(cur.getType());
  if (!dstTy || !dstTy.hasRank() ||
      dstTy.getElementType() != rewriter.getIntegerType(storageBits))
    return v;

  auto castToMem = cur.getDefiningOp<UnrealizedConversionCastOp>();
  if (!castToMem || castToMem.getNumOperands() != 1)
    return v;
  Value srcF32 = castToMem.getOperand(0);
  for (int hop = 0; hop < 8; ++hop) {
    while (auto mcast = srcF32.getDefiningOp<memref::CastOp>())
      srcF32 = mcast.getSource();
    auto ty = srcF32.getType();
    if (auto memTy = llvm::dyn_cast<MemRefType>(ty)) {
      if (memTy.hasRank() && llvm::isa<Float32Type>(memTy.getElementType()))
        break;
    }
    auto ucast = srcF32.getDefiningOp<UnrealizedConversionCastOp>();
    if (!ucast || ucast.getNumOperands() != 1)
      return v;
    srcF32 = ucast.getOperand(0);
  }
  auto srcF32Ty = llvm::dyn_cast<MemRefType>(srcF32.getType());
  if (!srcF32Ty || !srcF32Ty.hasRank() || !llvm::isa<Float32Type>(srcF32Ty.getElementType()))
    return v;

  Value out = allocLikeValue(rewriter, loc, dstTy, srcF32);
  if (!out)
    return v;

  auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
  auto unrankedI = getUnrankedITypeFor(rewriter, storageBits);
  Value inU = castToUnranked(rewriter, loc, srcF32, unrankedF32);
  Value outU = castToUnranked(rewriter, loc, out, unrankedI);

  ModuleOp module = rewriter.getBlock()->getParentOp()->getParentOfType<ModuleOp>();
  auto fnTy = rewriter.getFunctionType(
      {unrankedF32, unrankedI, rewriter.getI64Type()}, {});
  auto callee = getOrCreateFunc(module, getPositRuntimeNameFor("from_f32", nbits, es), fnTy);
  Value qalignKey = cstI64(rewriter, loc, 0);
  rewriter.create<func::CallOp>(loc, callee, ValueRange{inU, outU, qalignKey});
  return out;
}

static Value loadShapeAsI64(ConversionPatternRewriter &rewriter, Location loc,
                            Value shapeMemref, int64_t idx) {
  auto shapeTy = llvm::dyn_cast<MemRefType>(shapeMemref.getType());
  if (!shapeTy || !shapeTy.hasRank() || shapeTy.getRank() != 1)
    return Value();
  Value pos = rewriter.create<arith::ConstantIndexOp>(loc, idx);
  Value dim = rewriter.create<memref::LoadOp>(loc, shapeMemref, ValueRange{pos});
  auto i64Ty = rewriter.getI64Type();
  if (dim.getType() == i64Ty)
    return dim;
  if (llvm::isa<IndexType>(dim.getType()))
    return rewriter.create<arith::IndexCastOp>(loc, i64Ty, dim);
  auto intTy = llvm::dyn_cast<IntegerType>(dim.getType());
  if (!intTy)
    return Value();
  if (intTy.getWidth() < 64)
    return rewriter.create<arith::ExtSIOp>(loc, i64Ty, dim);
  if (intTy.getWidth() > 64)
    return rewriter.create<arith::TruncIOp>(loc, i64Ty, dim);
  return dim;
}

static func::FuncOp getOrCreateRegisterTensorCompandFunc(ModuleOp module,
    ConversionPatternRewriter &rewriter) {
  auto fnTy = rewriter.getFunctionType(
      {getUnrankedPositBitsType(rewriter), rewriter.getI64Type(),
          rewriter.getF64Type(), rewriter.getF64Type()},
      {});
  return getOrCreateFunc(
      module, getPositRuntimeCompandRegisterName(), fnTy);
}

static func::FuncOp getOrCreateRegisterTensorConstMetaFunc(
    ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto fnTy = rewriter.getFunctionType(
      {getUnrankedPositBitsType(rewriter), rewriter.getI64Type(),
          rewriter.getF64Type(), rewriter.getF64Type(), rewriter.getI64Type(),
          rewriter.getI64Type(), rewriter.getI64Type()},
      {});
  return getOrCreateFunc(
      module, getPositRuntimeConstMetaRegisterName(), fnTy);
}

static func::FuncOp getOrCreateRegisterTensorConstMetaChannelFunc(
    ModuleOp module, ConversionPatternRewriter &rewriter) {
  auto fnTy = rewriter.getFunctionType(
      {getUnrankedPositBitsType(rewriter), rewriter.getI64Type(),
       rewriter.getI64Type(), rewriter.getF64Type(), rewriter.getF64Type(),
       rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI64Type()},
      {});
  return getOrCreateFunc(
      module, getPositRuntimeConstMetaChannelRegisterName(), fnTy);
}

static bool getDenseI64AttrVector(Attribute attr, SmallVectorImpl<int64_t> &out) {
  auto dense = llvm::dyn_cast_or_null<ElementsAttr>(attr);
  if (!dense)
    return false;
  auto rtt = llvm::dyn_cast<RankedTensorType>(dense.getType());
  if (!rtt || !rtt.hasStaticShape() || rtt.getRank() != 1)
    return false;
  auto intTy = llvm::dyn_cast<IntegerType>(rtt.getElementType());
  if (!intTy)
    return false;
  out.clear();
  out.reserve(rtt.getNumElements());
  for (APInt v : dense.getValues<APInt>())
    out.push_back(v.getSExtValue());
  return true;
}

static bool getDenseF64AttrVector(Attribute attr, SmallVectorImpl<double> &out) {
  auto dense = llvm::dyn_cast_or_null<ElementsAttr>(attr);
  if (!dense)
    return false;
  auto rtt = llvm::dyn_cast<RankedTensorType>(dense.getType());
  if (!rtt || !rtt.hasStaticShape() || rtt.getRank() != 1)
    return false;
  if (!llvm::isa<FloatType>(rtt.getElementType()))
    return false;
  out.clear();
  out.reserve(rtt.getNumElements());
  for (APFloat v : dense.getValues<APFloat>())
    out.push_back(v.convertToDouble());
  return true;
}

//===----------------------------------------------------------------------===//
// posit.constant -> krnl.global
//===----------------------------------------------------------------------===//

struct PositConstantOpLowering : public OpConversionPattern<posit::ConstantOp> {
  using OpConversionPattern<posit::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    Location loc = op.getLoc();

    Attribute v = op->getAttr("value");
    auto dense = llvm::dyn_cast_or_null<DenseElementsAttr>(v);
    if (!dense)
      return rewriter.notifyMatchFailure(op, "posit.constant expects DenseElementsAttr 'value'");

    Type converted = getTypeConverter()->convertType(op.getResult().getType());
    auto memrefTy = llvm::dyn_cast_or_null<MemRefType>(converted);
    if (!memrefTy)
      return rewriter.notifyMatchFailure(op, "expected converted type to be MemRefType");

    static int64_t id = 0;
    std::string gname = "posit_const_" + std::to_string(id++);

    OperationState st(loc, "krnl.global");
    st.addTypes(memrefTy);
    st.addAttribute("name", rewriter.getStringAttr(gname));
    st.addAttribute("shape", rewriter.getI64ArrayAttr(memrefTy.getShape()));
    st.addAttribute("value", dense);

    Operation *global = rewriter.create(st);
    auto modeAttr = op->getAttrOfType<IntegerAttr>("compand_mode");
    auto thetaAttr = op->getAttrOfType<FloatAttr>("compand_theta");
    auto gammaAttr = op->getAttrOfType<FloatAttr>("compand_gamma");
    auto gpEnabledAttr = op->getAttrOfType<IntegerAttr>("gp_enabled");
    auto gpRsAttr = op->getAttrOfType<IntegerAttr>("gp_rs");
    auto gpScAttr = op->getAttrOfType<IntegerAttr>("gp_sc");
    auto axisAttr = op->getAttrOfType<IntegerAttr>("constmeta_axis");
    const bool hasCompand =
        modeAttr && thetaAttr && gammaAttr && getInt64Safe(modeAttr) != 0;
    const bool hasGp = gpEnabledAttr && getInt64Safe(gpEnabledAttr) != 0 &&
                       gpRsAttr && gpScAttr;
    SmallVector<int64_t, 8> modesAxis;
    SmallVector<double, 8> thetasAxis;
    SmallVector<double, 8> gammasAxis;
    SmallVector<int64_t, 8> gpEnabledAxis;
    SmallVector<int64_t, 8> gpRsAxis;
    SmallVector<int64_t, 8> gpScAxis;
    const bool hasPerAxis =
        axisAttr && getInt64Safe(axisAttr) == 0 &&
        getDenseI64AttrVector(op->getAttr("compand_mode_axis0"), modesAxis) &&
        getDenseF64AttrVector(op->getAttr("compand_theta_axis0"), thetasAxis) &&
        getDenseF64AttrVector(op->getAttr("compand_gamma_axis0"), gammasAxis) &&
        getDenseI64AttrVector(op->getAttr("gp_enabled_axis0"), gpEnabledAxis) &&
        getDenseI64AttrVector(op->getAttr("gp_rs_axis0"), gpRsAxis) &&
        getDenseI64AttrVector(op->getAttr("gp_sc_axis0"), gpScAxis) &&
        modesAxis.size() == thetasAxis.size() &&
        modesAxis.size() == gammasAxis.size() &&
        modesAxis.size() == gpEnabledAxis.size() &&
        modesAxis.size() == gpRsAxis.size() &&
        modesAxis.size() == gpScAxis.size();
    if (hasPerAxis) {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      auto fn = getOrCreateRegisterTensorConstMetaChannelFunc(module, rewriter);
      auto callee = SymbolRefAttr::get(module.getContext(), fn.getName());
      Value globalU = castToUnranked(
          rewriter, loc, global->getResult(0), getUnrankedPositBitsType(rewriter));
      for (size_t i = 0; i < modesAxis.size(); ++i) {
        rewriter.create<func::CallOp>(
            loc, callee, TypeRange{},
            ValueRange{globalU,
                       cstI64(rewriter, loc, static_cast<int64_t>(i)),
                       cstI64(rewriter, loc, modesAxis[i]),
                       cstF64(rewriter, loc, thetasAxis[i]),
                       cstF64(rewriter, loc, gammasAxis[i]),
                       cstI64(rewriter, loc, gpEnabledAxis[i]),
                       cstI64(rewriter, loc, gpRsAxis[i]),
                       cstI64(rewriter, loc, gpScAxis[i])});
      }
    } else if (hasCompand || hasGp) {
      ModuleOp module = op->getParentOfType<ModuleOp>();
      auto fn = getOrCreateRegisterTensorConstMetaFunc(module, rewriter);
      auto callee = SymbolRefAttr::get(module.getContext(), fn.getName());
      Value globalU = castToUnranked(
          rewriter, loc, global->getResult(0), getUnrankedPositBitsType(rewriter));
      Value modeV = cstI64(rewriter, loc, hasCompand ? getInt64Safe(modeAttr) : 0);
      Value thetaV = cstF64(rewriter, loc,
                            hasCompand ? thetaAttr.getValueAsDouble() : 1.0);
      Value gammaV = cstF64(rewriter, loc,
                            hasCompand ? gammaAttr.getValueAsDouble() : 0.0);
      Value gpEnabledV = cstI64(rewriter, loc, hasGp ? 1 : 0);
      Value gpRsV = cstI64(rewriter, loc, hasGp ? getInt64Safe(gpRsAttr) : 7);
      Value gpScV = cstI64(rewriter, loc, hasGp ? getInt64Safe(gpScAttr) : 0);
      rewriter.create<func::CallOp>(
          loc, callee, TypeRange{},
          ValueRange{globalU, modeV, thetaV, gammaV, gpEnabledV, gpRsV,
                     gpScV});
    }
    rewriter.replaceOp(op, global->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// posit.from_f32 / posit.to_f32
//===----------------------------------------------------------------------===//

struct PositFromF32OpLowering : public OpConversionPattern<posit::FromF32Op> {
  using OpConversionPattern<posit::FromF32Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::FromF32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    int64_t qalignKey = 0;
    if (auto a = op->template getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);

    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "from_f32 expects memref result");

    // Fold from_f32(to_f32(x)) -> x only when qalign is not active.
    if (qalignKey == 0) {
      Value rawInput = stripMemrefAndUnrealizedCast(op.getInput());
      if (auto toF32 = rawInput.getDefiningOp<posit::ToF32Op>()) {
        Value positSrc = stripMemrefAndUnrealizedCast(toF32.getInput());
        Value remapped = rewriter.getRemappedValue(positSrc);
        if (!remapped)
          remapped = positSrc;
        remapped = stripMemrefAndUnrealizedCast(remapped);
        if (llvm::isa<MemRefType>(remapped.getType())) {
          Value folded = remapped;
          if (folded.getType() != outTy)
            folded = rewriter
                         .create<UnrealizedConversionCastOp>(loc, outTy, folded)
                         .getResult(0);
          rewriter.replaceOp(op, folded);
          return success();
        }
      }
    }

    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getInput());
    if (!out)
      return rewriter.notifyMatchFailure(op, "failed to allocate dynamic from_f32 result");

    auto [opNbits, opEs] = getFormatFromPositType(op.getResult().getType());
    unsigned opStorageBits = getPositStorageBitWidth(opNbits);
    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    auto unrankedI  = getUnrankedITypeFor(rewriter, opStorageBits);

    Value inU  = castToUnranked(rewriter, loc, adaptor.getInput(), unrankedF32);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fnTy = rewriter.getFunctionType(
        {unrankedF32, unrankedI, rewriter.getI64Type()}, {});
    auto callee = getOrCreateFunc(module, getPositRuntimeNameFor("from_f32", opNbits, opEs), fnTy);
    Value qalignKeyV = cstI64(rewriter, loc, qalignKey);

    rewriter.create<func::CallOp>(loc, callee, ValueRange{inU, outU, qalignKeyV});
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct PositToF32OpLowering : public OpConversionPattern<posit::ToF32Op> {
  using OpConversionPattern<posit::ToF32Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::ToF32Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "to_f32 expects memref result");

    // Fold to_f32(from_f32(x)) -> x only when qalign is not active.
    Value rawInput = stripMemrefAndUnrealizedCast(op.getInput());
    if (auto fromF32 = rawInput.getDefiningOp<posit::FromF32Op>()) {
      int64_t qalignKey = 0;
      if (auto a = fromF32->getAttrOfType<IntegerAttr>("qalign_key"))
        qalignKey = getInt64Safe(a);
      if (qalignKey == 0) {
        Value f32Src = stripMemrefAndUnrealizedCast(fromF32.getInput());
        Value remapped = rewriter.getRemappedValue(f32Src);
        if (!remapped)
          remapped = f32Src;
        remapped = stripMemrefAndUnrealizedCast(remapped);
        if (llvm::isa<MemRefType>(remapped.getType())) {
          Value folded = remapped;
          if (folded.getType() != outTy)
            folded = rewriter
                         .create<UnrealizedConversionCastOp>(loc, outTy, folded)
                         .getResult(0);
          rewriter.replaceOp(op, folded);
          return success();
        }
      }
    }

    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getInput());
    if (!out)
      return rewriter.notifyMatchFailure(op, "failed to allocate dynamic to_f32 result");

    auto [opNbits, opEs] = getFormatFromPositType(op.getInput().getType());
    unsigned opStorageBits = getPositStorageBitWidth(opNbits);
    auto unrankedF32 = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    auto unrankedI  = getUnrankedITypeFor(rewriter, opStorageBits);

    Value inU  = castToUnranked(rewriter, loc, adaptor.getInput(), unrankedI);
    Value outU = castToUnranked(rewriter, loc, out, unrankedF32);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fnTy = rewriter.getFunctionType({unrankedI, unrankedF32}, {});
    auto callee = getOrCreateFunc(module, getPositRuntimeNameFor("to_f32", opNbits, opEs), fnTy);

    rewriter.create<func::CallOp>(loc, callee, ValueRange{inU, outU});
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct PositDequantizeLinearOpLowering
    : public OpConversionPattern<posit::DequantizeLinearOp> {
  using OpConversionPattern<posit::DequantizeLinearOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::DequantizeLinearOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto inTy = llvm::dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto inUnrankedTy =
        llvm::dyn_cast<UnrankedMemRefType>(adaptor.getInput().getType());
    if (!outTy || (!inTy && !inUnrankedTy))
      return rewriter.notifyMatchFailure(
          op, "dequantize_linear expects memref/unranked_memref input and ranked memref output");

    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getInput());
    if (!out)
      return rewriter.notifyMatchFailure(
          op, "failed to allocate dynamic dequantize_linear result");

    float scale = 1.0f;
    int64_t zeroPoint = 0;
    int64_t hasZeroPoint = 0;
    int64_t axis = 1;
    int64_t inputSigned = 1;
    int64_t qalignKey = 0;
    DenseElementsAttr scaleValues;
    DenseElementsAttr zeroPointValues;
    if (auto a = op->getAttrOfType<FloatAttr>("scale"))
      scale = a.getValueAsDouble();
    if (auto a = op->getAttrOfType<IntegerAttr>("zero_point"))
      zeroPoint = getInt64Safe(a);
    if (auto a = op->getAttrOfType<BoolAttr>("has_zero_point"))
      hasZeroPoint = a.getValue() ? 1 : 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("axis"))
      axis = getInt64Safe(a);
    if (auto a = op->getAttrOfType<BoolAttr>("input_signed"))
      inputSigned = a.getValue() ? 1 : 0;
    if (auto a = op->template getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);
    if (auto a = op->getAttrOfType<DenseElementsAttr>("scale_values"))
      scaleValues = a;
    if (auto a = op->getAttrOfType<DenseElementsAttr>("zero_point_values"))
      zeroPointValues = a;

    auto getDenseStaticNumElems = [](DenseElementsAttr dense) -> int64_t {
      if (!dense)
        return 0;
      auto st = llvm::dyn_cast<ShapedType>(dense.getType());
      if (!st || !st.hasRank() || !st.hasStaticShape())
        return -1;
      return st.getNumElements();
    };

    const bool hasScaleValues = static_cast<bool>(scaleValues);
    const bool hasZeroPointValues = static_cast<bool>(zeroPointValues);
    const bool useAxisVector = hasScaleValues || hasZeroPointValues;

    int64_t normalizedAxis = axis;
    int64_t inputRank = -1;
    if (inTy && inTy.hasRank())
      inputRank = inTy.getRank();
    else if (inUnrankedTy)
      inputRank = inUnrankedTy.getRank();
    if (inputRank >= 0) {
      if (normalizedAxis < 0)
        normalizedAxis += inputRank;
      if (normalizedAxis < 0 || normalizedAxis >= inputRank)
        return rewriter.notifyMatchFailure(
            op, "inconsistent q-params: axis is out of range for input rank");
      axis = normalizedAxis;
    } else if (normalizedAxis < 0) {
      return rewriter.notifyMatchFailure(
          op, "inconsistent q-params: negative axis requires ranked input");
    }

    if (useAxisVector) {
      if (!hasScaleValues)
        return rewriter.notifyMatchFailure(op,
            "inconsistent q-params: zero_point_values exists but scale_values is missing");
      if (!hasZeroPoint && hasZeroPointValues)
        return rewriter.notifyMatchFailure(op,
            "inconsistent q-params: has_zero_point=0 but zero_point_values exists");
      if (hasZeroPoint && !hasZeroPointValues)
        return rewriter.notifyMatchFailure(op,
            "inconsistent q-params: per-axis mode requires zero_point_values when has_zero_point=1");

      int64_t scaleLen = getDenseStaticNumElems(scaleValues);
      if (scaleLen <= 0)
        return rewriter.notifyMatchFailure(op,
            "inconsistent q-params: scale_values must be ranked/static with positive length");

      if (hasZeroPointValues) {
        int64_t zpLen = getDenseStaticNumElems(zeroPointValues);
        if (zpLen <= 0)
          return rewriter.notifyMatchFailure(op,
              "inconsistent q-params: zero_point_values must be ranked/static with positive length");
        if (zpLen != scaleLen)
          return rewriter.notifyMatchFailure(op,
              "inconsistent q-params: zero_point_values length must equal scale_values length");
      }

      if (scaleLen > 1) {
        if (!inTy || !inTy.hasRank())
          return rewriter.notifyMatchFailure(op,
              "inconsistent q-params: per-axis scale_values (>1) requires ranked input");
        int64_t axisDim = inTy.getShape()[axis];
        if (axisDim != ShapedType::kDynamic && axisDim != scaleLen)
          return rewriter.notifyMatchFailure(op,
              "inconsistent q-params: per-axis scale_values length must match input dim at axis");
      }
    }

    const bool dqOutputsF32 = outTy.getElementType().isF32();
    Type inElemTy = inTy ? inTy.getElementType() : inUnrankedTy.getElementType();
    auto unrankedIn = UnrankedMemRefType::get(inElemTy, 0);
    auto unrankedOut = dqOutputsF32 ? UnrankedMemRefType::get(rewriter.getF32Type(), 0)
                                    : getUnrankedPositBitsType(rewriter);
    Value inputForCall = adaptor.getInput();
    if (auto iTy = llvm::dyn_cast<IntegerType>(inElemTy)) {
      // Runtime dequantize kernels take int8 payload. Some upstream paths may
      // still present i32/signless ints here; normalize to i8 for a stable
      // callee signature.
      if (iTy.getWidth() != 8 || iTy.isUnsignedInteger()) {
        Type i8InTy;
        if (inTy)
          i8InTy = MemRefType::get(inTy.getShape(), rewriter.getI8Type());
        else
          i8InTy = UnrankedMemRefType::get(rewriter.getI8Type(), 0);
        inputForCall = rewriter
                           .create<UnrealizedConversionCastOp>(
                               loc, i8InTy, inputForCall)
                           .getResult(0);
        inElemTy = rewriter.getI8Type();
      }
    }
    unrankedIn = UnrankedMemRefType::get(inElemTy, 0);
    Value inU = castToUnranked(rewriter, loc, inputForCall, unrankedIn);
    Value outU = castToUnranked(rewriter, loc, out, unrankedOut);

    // Optional second operand is the pre-QuantizeLinear f32 reference tensor.
    // It is passed only to the *_ref runtime variants so runtime can collect
    // full-reference rows: orig_x, int8_x_dq, runtime_x_dq, runtime_final.
    Value origRefU;
    bool hasOrigRefOperand = false;
    if (adaptor.getOperands().size() >= 2) {
      // IMPORTANT: use the converted adaptor operand here.  The original
      // op operand is a tensor<...xf32>; after type conversion it becomes
      // memref<...xf32>.  Do not strip UnrealizedConversionCastOp here,
      // otherwise we fall back to the original tensor operand and this
      // pattern fails to legalize posit.dequantize_linear(..., orig_ref).
      Value origRef = adaptor.getOperands()[1];
      Type origTy = origRef.getType();
      bool isF32Memref = false;
      if (auto mt = llvm::dyn_cast<MemRefType>(origTy))
        isF32Memref = mt.getElementType().isF32();
      else if (auto ut = llvm::dyn_cast<UnrankedMemRefType>(origTy))
        isF32Memref = ut.getElementType().isF32();
      if (!isF32Memref)
        return rewriter.notifyMatchFailure(
            op, "optional dequantize full-reference operand must be converted to f32 memref");
      origRefU = castToUnranked(
          rewriter, loc, origRef, UnrankedMemRefType::get(rewriter.getF32Type(), 0));
      hasOrigRefOperand = true;
    }

    ModuleOp module = op->getParentOfType<ModuleOp>();
    // Keep ONNX axis-broadcast semantics whenever vector constants are present.
    if (useAxisVector) {
      Value scaleGlobal = createKrnlGlobalFromDense(
          rewriter, loc, scaleValues, "posit_dq_scale");
      if (!scaleGlobal)
        return rewriter.notifyMatchFailure(
            op, "failed to materialize per-axis scale_values global");

      Value zpGlobal;
      if (hasZeroPoint) {
        zpGlobal = createKrnlGlobalFromDense(
            rewriter, loc, zeroPointValues, "posit_dq_zp");
      } else {
        auto zpDense = makeI64Dense1D(rewriter, ArrayRef<int64_t>{zeroPoint});
        zpGlobal = createKrnlGlobalFromDense(
            rewriter, loc, zpDense, "posit_dq_zp_scalar");
      }
      if (!zpGlobal)
        return rewriter.notifyMatchFailure(
            op, "failed to materialize zero_point_values global");

      auto unrankedScale = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
      auto unrankedZp = UnrankedMemRefType::get(rewriter.getI64Type(), 0);
      Value scaleU = castToUnranked(rewriter, loc, scaleGlobal, unrankedScale);
      Value zpU = castToUnranked(rewriter, loc, zpGlobal, unrankedZp);

      SmallVector<Type, 9> argTys{unrankedIn, unrankedOut};
      if (hasOrigRefOperand)
        argTys.push_back(UnrankedMemRefType::get(rewriter.getF32Type(), 0));
      argTys.append({unrankedScale, unrankedZp, rewriter.getI64Type(),
          rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI64Type()});
      auto fnTy = rewriter.getFunctionType(argTys, {});
      std::string calleeName = getPositRuntimeName(
          dqOutputsF32 ? (hasOrigRefOperand ? "dequantize_linear_axis_f32_ref"
                                            : "dequantize_linear_axis_f32")
                       : (hasOrigRefOperand ? "dequantize_linear_axis_ref"
                                            : "dequantize_linear_axis"));
      auto callee = getOrCreateFunc(module, calleeName, fnTy);
      SmallVector<Value, 9> args{inU, outU};
      if (hasOrigRefOperand)
        args.push_back(origRefU);
      args.append({scaleU, zpU, cstI64(rewriter, loc, hasZeroPoint),
          cstI64(rewriter, loc, axis), cstI64(rewriter, loc, inputSigned),
          cstI64(rewriter, loc, qalignKey)});
      rewriter.create<func::CallOp>(loc, callee, args);
    } else {
      SmallVector<Type, 9> argTys{unrankedIn, unrankedOut};
      if (hasOrigRefOperand)
        argTys.push_back(UnrankedMemRefType::get(rewriter.getF32Type(), 0));
      argTys.append({rewriter.getF32Type(), rewriter.getI64Type(),
          rewriter.getI64Type(), rewriter.getI64Type(),
          rewriter.getI64Type(), rewriter.getI64Type()});
      auto fnTy = rewriter.getFunctionType(argTys, {});
      std::string calleeName = getPositRuntimeName(
          dqOutputsF32 ? (hasOrigRefOperand ? "dequantize_linear_f32_ref"
                                            : "dequantize_linear_f32")
                       : (hasOrigRefOperand ? "dequantize_linear_ref"
                                            : "dequantize_linear"));
      auto callee = getOrCreateFunc(module, calleeName, fnTy);
      SmallVector<Value, 9> args{inU, outU};
      if (hasOrigRefOperand)
        args.push_back(origRefU);
      args.append({cstF32(rewriter, loc, scale), cstI64(rewriter, loc, zeroPoint),
          cstI64(rewriter, loc, hasZeroPoint), cstI64(rewriter, loc, axis),
          cstI64(rewriter, loc, inputSigned), cstI64(rewriter, loc, qalignKey)});
      rewriter.create<func::CallOp>(loc, callee, args);
    }
    rewriter.replaceOp(op, out);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// elementwise ops: add/sub/mul/div
//===----------------------------------------------------------------------===//

template <typename OpT>
struct PositBinaryOpLowering : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  explicit PositBinaryOpLowering(TypeConverter &tc, MLIRContext *ctx,
                                 StringRef calleeName)
      : OpConversionPattern<OpT>(tc, ctx), calleeName(calleeName.str()) {}

  LogicalResult matchAndRewrite(OpT op, typename OpConversionPattern<OpT>::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto outTy = llvm::dyn_cast<MemRefType>(
        this->getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "binary op expects memref result");

    Type positBitsTy = getPositStorageIntType(rewriter);
    Value lhs = stripUnrealizedMemrefCast(adaptor.getLhs(), positBitsTy);
    Value rhs = stripUnrealizedMemrefCast(adaptor.getRhs(), positBitsTy);

    // For broadcasting (e.g. GPT-2 attention mask: scalar/lower-rank operand vs a
    // full-rank tensor), the output shape must be derived from the operand whose
    // REAL rank matches the result rank. An unrealized_conversion_cast can fake a
    // rank-changing operand (rank-0 -> rank-N); using it for alloc emits memref.dim
    // on a 0-ranked memref. Pick the genuinely rank-matching operand (lhs first).
    // The runtime elementwise kernels handle the actual broadcast via
    // offset_with_broadcast (incl. rank-0 scalars).
    auto lhsRealTy = llvm::dyn_cast<MemRefType>(lhs.getType());
    auto rhsRealTy = llvm::dyn_cast<MemRefType>(rhs.getType());

    // Allocate the (possibly dynamic) broadcast result. For each dynamic output
    // dim, pull its size from whichever operand actually covers that dim (NumPy
    // trailing alignment) and is NOT broadcast (size != 1). A scalar / lower-rank
    // operand (e.g. GPT-2 attention-mask constant) is skipped automatically, so we
    // never emit memref.dim on a rank-0 memref<i8>.
    Value out;
    if (outTy.hasStaticShape()) {
      out = rewriter.create<memref::AllocOp>(loc, outTy);
    } else {
      int64_t oRank = outTy.getRank();
      auto dimFrom = [&](Value v, MemRefType ty, int64_t od) -> Value {
        if (!ty || !ty.hasRank())
          return Value();
        int64_t md = od - (oRank - ty.getRank());
        if (md < 0)
          return Value();
        if (ty.isDynamicDim(md))
          return rewriter.create<memref::DimOp>(loc, v, md);
        if (ty.getDimSize(md) == 1)
          return Value();
        return rewriter.create<arith::ConstantIndexOp>(loc, ty.getDimSize(md));
      };
      SmallVector<Value, 4> dynDims;
      for (int64_t od = 0; od < oRank; ++od) {
        if (!outTy.isDynamicDim(od))
          continue;
        Value d = dimFrom(lhs, lhsRealTy, od);
        if (!d)
          d = dimFrom(rhs, rhsRealTy, od);
        if (!d)
          return rewriter.notifyMatchFailure(
              op, "failed to infer dynamic broadcast output dimension");
        dynDims.push_back(d);
      }
      out = rewriter.create<memref::AllocOp>(loc, outTy, dynDims);
    }

    auto unrankedI8 = getUnrankedPositBitsType(rewriter);
    int64_t qalignKey = 0;
    if (auto a = op->template getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);
    Value aU   = castToUnranked(rewriter, loc, lhs, unrankedI8);
    Value bU   = castToUnranked(rewriter, loc, rhs, unrankedI8);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI8);

    ModuleOp module = op->template getParentOfType<ModuleOp>();
    auto fnTy = rewriter.getFunctionType(
        {unrankedI8, unrankedI8, unrankedI8, rewriter.getI64Type()}, {});
    auto callee = getOrCreateFunc(module, calleeName, fnTy);

    rewriter.create<func::CallOp>(
        loc, callee, ValueRange{aU, bU, outU, cstI64(rewriter, loc, qalignKey)});
    rewriter.replaceOp(op, out);
    return success();
  }

  std::string calleeName;
};

//===----------------------------------------------------------------------===//
// relu
//===----------------------------------------------------------------------===//

struct PositReluOpLowering : public OpConversionPattern<posit::ReluOp> {
  using OpConversionPattern<posit::ReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::ReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "relu expects memref result");

    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getInput());
    if (!out)
      return rewriter.notifyMatchFailure(op, "failed to allocate dynamic relu result");

    auto unrankedI8 = getUnrankedPositBitsType(rewriter);
    int64_t qalignKey = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);

    Value inU  = castToUnranked(rewriter, loc, adaptor.getInput(), unrankedI8);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI8);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fnTy = rewriter.getFunctionType(
        {unrankedI8, unrankedI8, rewriter.getI64Type()}, {});
    auto callee = getOrCreateFunc(module, getPositRuntimeName("relu"), fnTy);

    rewriter.create<func::CallOp>(
        loc, callee, ValueRange{inU, outU, cstI64(rewriter, loc, qalignKey)});
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct PositClipOpLowering : public OpConversionPattern<posit::ClipOp> {
  using OpConversionPattern<posit::ClipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::ClipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "clip expects memref result");

    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getInput());
    if (!out)
      return rewriter.notifyMatchFailure(op, "failed to allocate dynamic clip result");

    bool hasMin = false, hasMax = false;
    float minVal = 0.0f, maxVal = 0.0f;
    int64_t qalignKey = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);
    if (auto a = op->getAttrOfType<BoolAttr>("has_min"))
      hasMin = a.getValue();
    if (auto a = op->getAttrOfType<BoolAttr>("has_max"))
      hasMax = a.getValue();
    if (auto a = op->getAttrOfType<FloatAttr>("min_val"))
      minVal = a.getValueAsDouble();
    if (auto a = op->getAttrOfType<FloatAttr>("max_val"))
      maxVal = a.getValueAsDouble();

    auto unrankedI = getUnrankedPositBitsType(rewriter);
    Value inU = castToUnranked(rewriter, loc, adaptor.getInput(), unrankedI);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fnTy = rewriter.getFunctionType(
        {unrankedI, unrankedI, rewriter.getF32Type(), rewriter.getF32Type(),
            rewriter.getI64Type(), rewriter.getI64Type(), rewriter.getI64Type()},
        {});
    auto callee = getOrCreateFunc(module, getPositRuntimeName("clip"), fnTy);
    rewriter.create<func::CallOp>(
        loc, callee,
        ValueRange{inU, outU, cstF32(rewriter, loc, minVal),
            cstF32(rewriter, loc, maxVal), cstI64(rewriter, loc, hasMin ? 1 : 0),
            cstI64(rewriter, loc, hasMax ? 1 : 0),
            cstI64(rewriter, loc, qalignKey)});
    rewriter.replaceOp(op, out);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// conv2d / maxpool2d / gemm
//===----------------------------------------------------------------------===//

static void readI64ArrayAttr(ArrayAttr a, int64_t &x0, int64_t &x1) {
  x0 = 0; x1 = 0;
  if (!a || a.size() < 2) return;
  x0 = llvm::cast<IntegerAttr>(a[0]).getValue().getSExtValue();
  x1 = llvm::cast<IntegerAttr>(a[1]).getValue().getSExtValue();
}

static void readPadsAttr(ArrayAttr a, int64_t &pt, int64_t &pl, int64_t &pb, int64_t &pr) {
  pt=pl=pb=pr=0;
  if (!a || a.size() < 4) return;
  pt = llvm::cast<IntegerAttr>(a[0]).getValue().getSExtValue();
  pl = llvm::cast<IntegerAttr>(a[1]).getValue().getSExtValue();
  pb = llvm::cast<IntegerAttr>(a[2]).getValue().getSExtValue();
  pr = llvm::cast<IntegerAttr>(a[3]).getValue().getSExtValue();
}

static void computeSamePads(StringRef mode,
                            int64_t inH, int64_t inW,
                            int64_t outH, int64_t outW,
                            int64_t kH, int64_t kW,
                            int64_t sH, int64_t sW,
                            int64_t dH, int64_t dW,
                            int64_t &pt, int64_t &pl, int64_t &pb, int64_t &pr) {
  // effective kernel
  int64_t effKH = (kH - 1) * dH + 1;
  int64_t effKW = (kW - 1) * dW + 1;

  int64_t padH = (outH - 1) * sH + effKH - inH;
  int64_t padW = (outW - 1) * sW + effKW - inW;
  if (padH < 0) padH = 0;
  if (padW < 0) padW = 0;

  if (mode == "SAME_LOWER") {
    // extra goes to beginning
    pt = (padH + 1) / 2;
    pb = padH - pt;
    pl = (padW + 1) / 2;
    pr = padW - pl;
  } else {
    // SAME_UPPER (default): extra goes to end
    pt = padH / 2;
    pb = padH - pt;
    pl = padW / 2;
    pr = padW - pl;
  }
}

struct PositConv2DOpLowering : public OpConversionPattern<posit::Conv2DOp> {
  using OpConversionPattern<posit::Conv2DOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::Conv2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "conv2d expects memref result");
    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getX());
    if (!out)
      return rewriter.notifyMatchFailure(op, "failed to allocate dynamic conv2d result");

    // attrs
    int64_t sH=1,sW=1,dH=1,dW=1,pt=0,pl=0,pb=0,pr=0,group=1;
    readI64ArrayAttr(op->getAttrOfType<ArrayAttr>("strides"), sH, sW);
    readI64ArrayAttr(op->getAttrOfType<ArrayAttr>("dilations"), dH, dW);
    group = op->getAttrOfType<IntegerAttr>("group") ? op->getAttrOfType<IntegerAttr>("group").getValue().getSExtValue() : 1;

    auto padsAttr = op->getAttrOfType<ArrayAttr>("pads");
    if (padsAttr) {
      readPadsAttr(padsAttr, pt, pl, pb, pr);
    } else {
      // [MOD][2026-01-14] auto_pad SAME_*：依 output shape 反推 pads（照 ONNX）
      auto autoPad = op->getAttrOfType<StringAttr>("auto_pad");
      StringRef mode = autoPad ? autoPad.getValue() : "NOTSET";
      if (mode == "SAME_UPPER" || mode == "SAME_LOWER") {
        auto xTy = llvm::dyn_cast<MemRefType>(adaptor.getX().getType());
        if (!xTy || !xTy.hasStaticShape() || !outTy.hasStaticShape())
          return rewriter.notifyMatchFailure(op, "SAME_* needs static input/output shapes");

        int64_t inH = xTy.getShape()[2], inW = xTy.getShape()[3];
        int64_t outH = outTy.getShape()[2], outW = outTy.getShape()[3];
        int64_t kH = 0, kW = 0;
        // Prefer ONNX kernel_shape attr so this also works when W is unranked.
        readI64ArrayAttr(op->getAttrOfType<ArrayAttr>("kernel_shape"), kH, kW);
        if (kH <= 0 || kW <= 0) {
          auto wTy = llvm::dyn_cast<MemRefType>(adaptor.getW().getType());
          if (!wTy || !wTy.hasStaticShape() || wTy.getRank() < 4)
            return rewriter.notifyMatchFailure(
                op, "SAME_* needs kernel_shape attr or static ranked W");
          kH = wTy.getShape()[2];
          kW = wTy.getShape()[3];
        }
        computeSamePads(mode, inH, inW, outH, outW, kH, kW, sH, sW, dH, dW, pt, pl, pb, pr);
      }
    }

    auto unrankedI8 = getUnrankedPositBitsType(rewriter);
    int64_t qalignKey = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);

    Value xPrepared = materializePositFromF32CastChainIfNeeded(
        rewriter, loc, adaptor.getX());
    Value xU   = castToUnranked(rewriter, loc, xPrepared, unrankedI8);
    Value wU   = castToUnranked(rewriter, loc, adaptor.getW(), unrankedI8);
    Value bU   = castToUnranked(rewriter, loc, adaptor.getB(), unrankedI8);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI8);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    SmallVector<Type> inTys = {unrankedI8, unrankedI8, unrankedI8, unrankedI8,
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type()};
    auto fnTy = rewriter.getFunctionType(inTys, {});
    auto callee = getOrCreateFunc(module, getPositRuntimeName("conv2d_nchw"), fnTy);

    rewriter.create<func::CallOp>(
        loc, callee,
        ValueRange{xU, wU, bU, outU,
                   cstI64(rewriter, loc, sH), cstI64(rewriter, loc, sW),
                   cstI64(rewriter, loc, dH), cstI64(rewriter, loc, dW),
                   cstI64(rewriter, loc, pt), cstI64(rewriter, loc, pl),
                   cstI64(rewriter, loc, pb), cstI64(rewriter, loc, pr),
                   cstI64(rewriter, loc, group),
                   cstI64(rewriter, loc, qalignKey)});

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct PositMaxPool2DOpLowering : public OpConversionPattern<posit::MaxPool2DOp> {
  using OpConversionPattern<posit::MaxPool2DOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::MaxPool2DOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "maxpool2d expects memref result");
    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getX());
    if (!out)
      return rewriter.notifyMatchFailure(op, "failed to allocate dynamic maxpool result");

    int64_t kH=1,kW=1,sH=1,sW=1,pt=0,pl=0,pb=0,pr=0,ceilMode=0;
    readI64ArrayAttr(op->getAttrOfType<ArrayAttr>("kernel_shape"), kH, kW);
    readI64ArrayAttr(op->getAttrOfType<ArrayAttr>("strides"), sH, sW);
    readPadsAttr(op->getAttrOfType<ArrayAttr>("pads"), pt, pl, pb, pr);
    if (auto cm = op->getAttrOfType<IntegerAttr>("ceil_mode"))
      ceilMode = cm.getValue().getSExtValue();

    auto unrankedI8 = getUnrankedPositBitsType(rewriter);
    int64_t qalignKey = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);

    Value xU   = castToUnranked(rewriter, loc, adaptor.getX(), unrankedI8);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI8);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    SmallVector<Type> inTys = {unrankedI8, unrankedI8,
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type(), rewriter.getI64Type()};
    auto fnTy = rewriter.getFunctionType(inTys, {});
    auto callee = getOrCreateFunc(module, getPositRuntimeName("maxpool2d_nchw"), fnTy);

    rewriter.create<func::CallOp>(
        loc, callee,
        ValueRange{xU, outU,
                   cstI64(rewriter, loc, kH), cstI64(rewriter, loc, kW),
                   cstI64(rewriter, loc, sH), cstI64(rewriter, loc, sW),
                   cstI64(rewriter, loc, pt), cstI64(rewriter, loc, pl),
                   cstI64(rewriter, loc, pb), cstI64(rewriter, loc, pr),
                   cstI64(rewriter, loc, ceilMode),
                   cstI64(rewriter, loc, qalignKey)});

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct PositGemmOpLowering : public OpConversionPattern<posit::GemmOp> {
  using OpConversionPattern<posit::GemmOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::GemmOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "gemm expects memref result");

    float alpha = 1.0f, beta = 1.0f;
    int64_t transA = 0, transB = 0;
    int64_t qalignKey = 0;
    if (auto a = op->getAttrOfType<FloatAttr>("alpha")) alpha = a.getValueAsDouble();
    if (auto b = op->getAttrOfType<FloatAttr>("beta"))  beta  = b.getValueAsDouble();
    if (auto ta = op->getAttrOfType<IntegerAttr>("transA")) transA = ta.getValue().getSExtValue();
    if (auto tb = op->getAttrOfType<IntegerAttr>("transB")) transB = tb.getValue().getSExtValue();
    if (auto a = op->getAttrOfType<IntegerAttr>("qalign_key")) qalignKey = getInt64Safe(a);

    auto [opNbits, opEs] = getFormatFromPositType(op.getResult().getType());
    unsigned opStorageBits = getPositStorageBitWidth(opNbits);
    Value aPrepared = materializePositFromF32CastChainIfNeeded(
        rewriter, loc, adaptor.getA(), opNbits, opEs);
    Value bPrepared = materializePositFromF32CastChainIfNeeded(
        rewriter, loc, adaptor.getB(), opNbits, opEs);
    Value cPrepared = materializePositFromF32CastChainIfNeeded(
        rewriter, loc, adaptor.getC(), opNbits, opEs);

    auto aTy = llvm::dyn_cast<MemRefType>(aPrepared.getType());
    auto bTy = llvm::dyn_cast<MemRefType>(bPrepared.getType());
    auto getDimOrConst = [&](Value v, MemRefType ty, int64_t d) -> Value {
      if (!ty || !ty.hasRank() || d < 0 || d >= ty.getRank())
        return Value();
      if (ty.isDynamicDim(d))
        return rewriter.create<memref::DimOp>(loc, v, d);
      return rewriter.create<arith::ConstantIndexOp>(loc, ty.getDimSize(d));
    };

    // Gemm output dims:
    // 1) Standard 2D GEMM: out[M,N], M from A, N from B (respecting transA/transB).
    // 2) Conv-im2col style GEMM used by MobileNet path:
    //    A[M,K], B[N,K,S], out[N,M,S] (batch N comes from B dim0).
    SmallVector<Value, 4> dynDims;
    if (!outTy.hasStaticShape()) {
      for (int64_t od = 0, oe = outTy.getRank(); od < oe; ++od) {
        if (!outTy.isDynamicDim(od))
          continue;

        Value dimVal;
        if (outTy.getRank() == 3 && aTy && bTy && aTy.hasRank() && bTy.hasRank() &&
            aTy.getRank() == 2 && bTy.getRank() == 3) {
          if (od == 0) {
            dimVal = getDimOrConst(bPrepared, bTy, 0);
          } else if (od == 1) {
            dimVal = getDimOrConst(aPrepared, aTy, transA ? 1 : 0);
          } else if (od == 2) {
            dimVal = getDimOrConst(bPrepared, bTy, transB ? 1 : 2);
          }
        } else if (outTy.getRank() == 2 && aTy && bTy && aTy.hasRank() &&
                   bTy.hasRank() && aTy.getRank() == 2 && bTy.getRank() == 2) {
          if (od == 0) {
            dimVal = getDimOrConst(aPrepared, aTy, transA ? 1 : 0);
          } else if (od == 1) {
            dimVal = getDimOrConst(bPrepared, bTy, transB ? 0 : 1);
          }
        } else if (outTy.getRank() >= 3 && aTy && bTy && aTy.hasRank() &&
                   bTy.hasRank() && aTy.getRank() == outTy.getRank() &&
                   bTy.getRank() == outTy.getRank()) {
          // Batched matmul: out[batch..., M, N]. Leading (batch) dims from A;
          // M from A's second-last dim (transA-aware), N from B's last dim
          // (transB-aware). Without this, rank>=3 hit the fallback which copied
          // A's dim -> wrong N (GPT-2 attention QK^T got [.,.,1,64] instead of
          // [.,.,1,1]); the runtime gemm then saw an inconsistent output shape
          // and zeroY'd -> attention all zeros -> logits collapse.
          int64_t R = outTy.getRank();
          if (od < R - 2)
            dimVal = getDimOrConst(aPrepared, aTy, od);
          else if (od == R - 2)
            dimVal = getDimOrConst(aPrepared, aTy, transA ? R - 1 : R - 2);
          else
            dimVal = getDimOrConst(bPrepared, bTy, transB ? R - 2 : R - 1);
        }

        if (!dimVal) {
          int64_t fallbackDim = std::min<int64_t>(
              od, aTy && aTy.hasRank() ? aTy.getRank() - 1 : 0);
          dimVal = getDimOrConst(aPrepared, aTy, fallbackDim);
        }
        if (!dimVal)
          return rewriter.notifyMatchFailure(
              op, "failed to infer dynamic gemm output dimension");
        dynDims.push_back(dimVal);
      }
    }

    Value out = outTy.hasStaticShape()
                    ? rewriter.create<memref::AllocOp>(loc, outTy)
                    : rewriter.create<memref::AllocOp>(loc, outTy, dynDims);

    auto unrankedI = getUnrankedITypeFor(rewriter, opStorageBits);

    Value aU   = castToUnranked(rewriter, loc, aPrepared, unrankedI);
    Value bU   = castToUnranked(rewriter, loc, bPrepared, unrankedI);
    Value cU   = castToUnranked(rewriter, loc, cPrepared, unrankedI);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    SmallVector<Type> inTys = {unrankedI, unrankedI, unrankedI, unrankedI,
                               rewriter.getF32Type(), rewriter.getF32Type(),
                               rewriter.getI64Type(), rewriter.getI64Type(),
                               rewriter.getI64Type()};
    auto fnTy = rewriter.getFunctionType(inTys, {});
    auto callee = getOrCreateFunc(module, getPositRuntimeNameFor("gemm", opNbits, opEs), fnTy);

    rewriter.create<func::CallOp>(
        loc, callee,
        ValueRange{aU, bU, cU, outU,
                   cstF32(rewriter, loc, alpha),
                   cstF32(rewriter, loc, beta),
                   cstI64(rewriter, loc, transA),
                   cstI64(rewriter, loc, transB),
                   cstI64(rewriter, loc, qalignKey)});

    rewriter.replaceOp(op, out);
    return success();
  }
};

struct PositReduceMeanOpLowering : public OpConversionPattern<posit::ReduceMeanOp> {
  using OpConversionPattern<posit::ReduceMeanOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::ReduceMeanOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "reduce_mean expects memref result");

    Value out = allocLikeValue(rewriter, loc, outTy, adaptor.getInput());
    if (!out)
      return rewriter.notifyMatchFailure(
          op, "failed to allocate dynamic reduce_mean result");

    SmallVector<int64_t, 4> axes;
    if (auto aa = op->getAttrOfType<ArrayAttr>("axes")) {
      for (Attribute a : aa)
        axes.push_back(getInt64Safe(a));
    }
    int64_t keepdims = 1;
    int64_t qalignKey = 0;
    if (auto a = op->getAttrOfType<IntegerAttr>("keepdims"))
      keepdims = getInt64Safe(a);
    if (auto a = op->getAttrOfType<IntegerAttr>("qalign_key"))
      qalignKey = getInt64Safe(a);

    int64_t axis0 = axes.size() >= 1 ? axes[0] : 2;
    int64_t axis1 = axes.size() >= 2 ? axes[1] : 3;

    auto unrankedI = getUnrankedPositBitsType(rewriter);
    Value inU = castToUnranked(rewriter, loc, adaptor.getInput(), unrankedI);
    Value outU = castToUnranked(rewriter, loc, out, unrankedI);

    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto fnTy = rewriter.getFunctionType(
        {unrankedI, unrankedI, rewriter.getI64Type(), rewriter.getI64Type(),
            rewriter.getI64Type(), rewriter.getI64Type()},
        {});
    auto callee = getOrCreateFunc(module, getPositRuntimeName("reduce_mean"), fnTy);
    rewriter.create<func::CallOp>(
        loc, callee,
        ValueRange{inU, outU, cstI64(rewriter, loc, axis0),
            cstI64(rewriter, loc, axis1), cstI64(rewriter, loc, keepdims),
            cstI64(rewriter, loc, qalignKey)});
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct PositFlattenOpLowering : public OpConversionPattern<posit::FlattenOp> {
  using OpConversionPattern<posit::FlattenOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::FlattenOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    auto inTy = llvm::dyn_cast<MemRefType>(adaptor.getData().getType());
    if (!outTy || !inTy || !inTy.hasRank() || outTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "flatten expects ranked memrefs");

    int64_t axis = 1;
    if (auto a = op->getAttrOfType<IntegerAttr>("axis"))
      axis = getInt64Safe(a);
    if (axis < 0)
      axis += inTy.getRank();
    if (axis < 0 || axis > inTy.getRank())
      return rewriter.notifyMatchFailure(op, "invalid flatten axis");

    auto cstIndex = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto dimOrConst = [&](int64_t d) -> Value {
      if (inTy.isDynamicDim(d))
        return rewriter.create<memref::DimOp>(loc, adaptor.getData(), d);
      return cstIndex(inTy.getDimSize(d));
    };

    Value dim0 = cstIndex(1);
    for (int64_t i = 0; i < axis; ++i)
      dim0 = rewriter.create<arith::MulIOp>(loc, dim0, dimOrConst(i));
    Value dim1 = cstIndex(1);
    for (int64_t i = axis; i < inTy.getRank(); ++i)
      dim1 = rewriter.create<arith::MulIOp>(loc, dim1, dimOrConst(i));

    SmallVector<OpFoldResult, 2> sizes;
    if (outTy.isDynamicDim(0))
      sizes.push_back(dim0);
    else
      sizes.push_back(rewriter.getIndexAttr(outTy.getDimSize(0)));
    if (outTy.isDynamicDim(1))
      sizes.push_back(dim1);
    else
      sizes.push_back(rewriter.getIndexAttr(outTy.getDimSize(1)));
    SmallVector<OpFoldResult, 2> strides;
    if (outTy.isDynamicDim(1))
      strides.push_back(dim1);
    else
      strides.push_back(rewriter.getIndexAttr(outTy.getDimSize(1)));
    strides.push_back(rewriter.getIndexAttr(1));
    OpFoldResult offset = rewriter.getIndexAttr(0);
    auto rc = rewriter.create<memref::ReinterpretCastOp>(
        loc, outTy, adaptor.getData(), offset, sizes, strides);
    rewriter.replaceOp(op, rc.getResult());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// reshape: memref.reinterpret_cast (MNIST static + contiguous)
//===----------------------------------------------------------------------===//

struct PositReshapeOpLowering : public OpConversionPattern<posit::ReshapeOp> {
  using OpConversionPattern<posit::ReshapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::ReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto outTy = llvm::dyn_cast<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "reshape expects memref result");
    auto inTy = llvm::dyn_cast<MemRefType>(adaptor.getData().getType());
    if (!inTy || !inTy.hasRank())
      return rewriter.notifyMatchFailure(op, "reshape expects ranked input memref");

    auto i64Ty = rewriter.getI64Type();
    auto cstIndex = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantIndexOp>(loc, v);
    };
    auto cstI64 = [&](int64_t v) -> Value {
      return rewriter.create<arith::ConstantOp>(
          loc, i64Ty, rewriter.getI64IntegerAttr(v));
    };
    auto toI64 = [&](Value idxOrI64) -> Value {
      if (idxOrI64.getType() == i64Ty)
        return idxOrI64;
      return rewriter.create<arith::IndexCastOp>(loc, i64Ty, idxOrI64);
    };
    auto inDimAsI64 = [&](int64_t d) -> Value {
      if (inTy.isDynamicDim(d))
        return toI64(rewriter.create<memref::DimOp>(loc, adaptor.getData(), d));
      return cstI64(inTy.getDimSize(d));
    };

    // ONNX reshape semantics:
    // - dim == 0 (allowzero=0 default): copy input dim at same axis.
    // - dim == -1: infer from total_input_elems / product(other output dims).
    Value totalInputElems = cstI64(1);
    for (int64_t d = 0, e = inTy.getRank(); d < e; ++d)
      totalInputElems =
          rewriter.create<arith::MulIOp>(loc, totalInputElems, inDimAsI64(d));

    SmallVector<OpFoldResult> sizes(outTy.getRank());
    SmallVector<OpFoldResult> strides;
    SmallVector<Value, 4> sizeVals(outTy.getRank(), Value());
    SmallVector<Value, 4> dynDimI64(outTy.getRank(), Value());
    SmallVector<Value, 4> dynIsInfer(outTy.getRank(), Value());
    Value knownProd = cstI64(1);
    Value c0I64 = cstI64(0);
    Value c1I64 = cstI64(1);
    Value cNeg1I64 = cstI64(-1);
    for (int64_t i = 0, e = outTy.getRank(); i < e; ++i) {
      if (outTy.isDynamicDim(i)) {
        Value rawI64 = loadShapeAsI64(rewriter, loc, adaptor.getShape(), i);
        if (!rawI64)
          return rewriter.notifyMatchFailure(
              op, "dynamic reshape requires rank-1 i64 shape memref");
        Value resolvedI64 = rawI64;
        if (i < inTy.getRank()) {
          Value isZero =
              rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                             rawI64, c0I64);
          Value inDimI64 = inDimAsI64(i);
          resolvedI64 =
              rewriter.create<arith::SelectOp>(loc, isZero, inDimI64, rawI64);
        }
        Value isInfer = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::eq, resolvedI64, cNeg1I64);
        Value mulDim =
            rewriter.create<arith::SelectOp>(loc, isInfer, c1I64, resolvedI64);
        knownProd = rewriter.create<arith::MulIOp>(loc, knownProd, mulDim);
        dynDimI64[i] = resolvedI64;
        dynIsInfer[i] = isInfer;
      } else {
        int64_t s = outTy.getDimSize(i);
        sizeVals[i] = cstIndex(s);
        sizes[i] = rewriter.getIndexAttr(s);
        // BUGFIX: static output dims must also divide out of the inferred (-1)
        // dim. Previously knownProd only accumulated dynamic dims, so e.g.
        // reshape [1,1,768] -> [-1,768] computed inferredDim = total/1 = 768
        // instead of total/768 = 1, making the seq dim = hidden dim and feeding
        // every transformer gemm an [hidden,hidden] operand (nan/garbage output,
        // mapped_range guard rejections). Include static dims in knownProd.
        knownProd = rewriter.create<arith::MulIOp>(loc, knownProd, cstI64(s));
      }
    }
    Value inferredDim = rewriter.create<arith::DivSIOp>(loc, totalInputElems, knownProd);
    for (int64_t i = 0, e = outTy.getRank(); i < e; ++i) {
      if (!outTy.isDynamicDim(i))
        continue;
      Value finalI64 = rewriter.create<arith::SelectOp>(
          loc, dynIsInfer[i], inferredDim, dynDimI64[i]);
      Value finalIdx =
          rewriter.create<arith::IndexCastOp>(loc, rewriter.getIndexType(), finalI64);
      sizeVals[i] = finalIdx;
      sizes[i] = finalIdx;
    }

    SmallVector<Value, 4> strideVals(outTy.getRank());
    Value one = cstIndex(1);
    if (outTy.getRank() > 0)
      strideVals[outTy.getRank() - 1] = one;
    for (int64_t i = outTy.getRank() - 2; i >= 0; --i)
      strideVals[i] =
          rewriter.create<arith::MulIOp>(loc, strideVals[i + 1], sizeVals[i + 1]);

    for (int64_t i = 0, e = outTy.getRank(); i < e; ++i) {
      bool allTailStatic = true;
      int64_t tailProd = 1;
      for (int64_t j = i + 1; j < e; ++j) {
        if (outTy.isDynamicDim(j)) {
          allTailStatic = false;
          break;
        }
        tailProd *= outTy.getDimSize(j);
      }
      if (allTailStatic)
        strides.push_back(rewriter.getIndexAttr(tailProd));
      else
        strides.push_back(strideVals[i]);
    }

    OpFoldResult offset = rewriter.getIndexAttr(0);

    auto rc = rewriter.create<memref::ReinterpretCastOp>(
        loc, outTy, adaptor.getData(), offset, sizes, strides);

    rewriter.replaceOp(op, rc.getResult());
    return success();
  }
};

} // namespace

void populatePositToKrnlConversionPattern(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         MLIRContext *ctx,
                                         unsigned nbits,
                                         unsigned es) {
  gPositNbits = nbits;
  gPositEs = es;
  gPositStorageBits = getPositStorageBitWidth(nbits);
  // [MOD][2026-01-14] 加上 MNIST 需要的 lowering
  patterns.add<PositConstantOpLowering,
               PositFromF32OpLowering, PositToF32OpLowering,
               PositDequantizeLinearOpLowering,
               PositReluOpLowering, PositClipOpLowering,
               PositConv2DOpLowering, PositMaxPool2DOpLowering,
               PositReshapeOpLowering, PositFlattenOpLowering,
               PositGemmOpLowering, PositReduceMeanOpLowering>(
      typeConverter, ctx);

  patterns.add<PositBinaryOpLowering<posit::AddOp>>(typeConverter, ctx, getPositRuntimeName("add"));
  // [MOD] 上面 template ctor 只吃一個 name；所以我們把其它三個再各加一次
  patterns.add<PositBinaryOpLowering<posit::SubOp>>(typeConverter, ctx, getPositRuntimeName("sub"));
  patterns.add<PositBinaryOpLowering<posit::MulOp>>(typeConverter, ctx, getPositRuntimeName("mul"));
  patterns.add<PositBinaryOpLowering<posit::DivOp>>(typeConverter, ctx, getPositRuntimeName("div"));
}

} // namespace onnx_mlir
