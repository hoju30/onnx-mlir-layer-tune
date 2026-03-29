#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/APFloat.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Posit/PositOps.h"
#include "src/Conversion/ONNXToPosit/TypeConverters.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>


using namespace mlir;
namespace onnx_mlir {
namespace { // anonymous namespace for internal implementation

static unsigned getPositStorageBitWidth(unsigned nbits) {
  if (nbits <= 8)
    return 8;
  if (nbits <= 16)
    return 16;
  if (nbits <= 32)
    return 32;
  if (nbits <= 64)
    return 64;
  return nbits;
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

static ElementsAttr getElementsAttrFromValue(Value v) {
  Value base = stripUnrealizedCast(v);
  if (auto cst = base.getDefiningOp<mlir::ONNXConstantOp>())
    return llvm::dyn_cast_or_null<ElementsAttr>(cst->getAttr("value"));
  if (auto cst = base.getDefiningOp<mlir::arith::ConstantOp>())
    return llvm::dyn_cast<ElementsAttr>(cst.getValue());
  if (auto cst = base.getDefiningOp<mlir::posit::ConstantOp>())
    return llvm::dyn_cast_or_null<ElementsAttr>(cst->getAttr("value"));
  return nullptr;
}

static LogicalResult collectFPValuesAsDouble(ElementsAttr elements,
                                             SmallVectorImpl<double> &out) {
  if (auto denseFP = llvm::dyn_cast<DenseFPElementsAttr>(elements)) {
    out.reserve(denseFP.getNumElements());
    for (APFloat apf : denseFP.getValues<APFloat>())
      out.push_back(apf.convertToDouble());
    return success();
  }

  if (auto resF32 = llvm::dyn_cast<mlir::detail::DenseResourceElementsAttrBase<float>>(elements)) {
    auto arr = resF32.tryGetAsArrayRef();
    if (!arr)
      return failure();
    out.reserve(arr->size());
    for (float v : *arr)
      out.push_back(static_cast<double>(v));
    return success();
  }

  if (auto resF64 = llvm::dyn_cast<mlir::detail::DenseResourceElementsAttrBase<double>>(elements)) {
    auto arr = resF64.tryGetAsArrayRef();
    if (!arr)
      return failure();
    out.reserve(arr->size());
    for (double v : *arr)
      out.push_back(v);
    return success();
  }

  return failure();
}

static bool getDenseIntegerTensorFromValue(Value v, RankedTensorType &rtt,
                                           SmallVectorImpl<int64_t> &vals);

static bool getDenseFloatTensorFromValue(Value v, RankedTensorType &rtt,
                                         SmallVectorImpl<double> &vals) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  auto ty = ea ? llvm::dyn_cast<RankedTensorType>(ea.getType()) : RankedTensorType();
  if (!ty)
    return false;
  if (!llvm::isa<FloatType>(ty.getElementType()))
    return false;

  vals.clear();
  if (failed(collectFPValuesAsDouble(ea, vals)))
    return false;
  if (static_cast<int64_t>(vals.size()) != ty.getNumElements())
    return false;
  rtt = ty;
  return true;
}

static bool getScalarFloatFromValue(Value v, double &out) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  if (!ea || ea.getNumElements() != 1)
    return false;

  SmallVector<double, 1> fpVals;
  if (succeeded(collectFPValuesAsDouble(ea, fpVals)) && fpVals.size() == 1) {
    out = fpVals[0];
    return true;
  }

  if (auto fp = llvm::dyn_cast<DenseFPElementsAttr>(ea)) {
    out = (*fp.getValues<APFloat>().begin()).convertToDouble();
    return true;
  }
  if (auto ints = llvm::dyn_cast<DenseIntElementsAttr>(ea)) {
    APInt ap = *ints.getValues<APInt>().begin();
    auto intTy = llvm::dyn_cast<IntegerType>(ints.getElementType());
    out = (intTy && intTy.isUnsigned()) ? static_cast<double>(ap.getZExtValue())
                                        : static_cast<double>(ap.getSExtValue());
    return true;
  }
  return false;
}

static bool getScalarIntFromValue(Value v, int64_t &out) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  if (!ea || ea.getNumElements() != 1)
    return false;
  if (auto ints = llvm::dyn_cast<DenseIntElementsAttr>(ea)) {
    APInt ap = *ints.getValues<APInt>().begin();
    auto intTy = llvm::dyn_cast<IntegerType>(ints.getElementType());
    out = (intTy && intTy.isUnsigned()) ? static_cast<int64_t>(ap.getZExtValue())
                                        : static_cast<int64_t>(ap.getSExtValue());
    return true;
  }
  RankedTensorType rtt;
  SmallVector<int64_t, 1> vals;
  if (getDenseIntegerTensorFromValue(v, rtt, vals) && vals.size() == 1) {
    out = vals.front();
    return true;
  }
  return false;
}

static bool getDenseIntegerTensorFromValue(Value v, RankedTensorType &rtt,
                                           SmallVectorImpl<int64_t> &vals) {
  ElementsAttr ea = getElementsAttrFromValue(v);
  auto ty = ea ? llvm::dyn_cast<RankedTensorType>(ea.getType()) : RankedTensorType();
  if (!ty)
    return false;

  auto intTy = llvm::dyn_cast<IntegerType>(ty.getElementType());
  if (!intTy)
    return false;
  rtt = ty;
  vals.clear();

  if (auto ints = llvm::dyn_cast_or_null<DenseIntElementsAttr>(ea)) {
    const bool isUnsigned = intTy.isUnsigned();
    vals.reserve(static_cast<size_t>(ints.getNumElements()));
    for (APInt ap : ints.getValues<APInt>())
      vals.push_back(isUnsigned ? static_cast<int64_t>(ap.getZExtValue())
                                : static_cast<int64_t>(ap.getSExtValue()));
    return true;
  }

  const bool isUnsigned = intTy.isUnsigned();
  auto pushValsFromResource = [&](auto dummy) -> bool {
    using ElemTy = decltype(dummy);
    auto res = llvm::dyn_cast_or_null<mlir::detail::DenseResourceElementsAttrBase<ElemTy>>(ea);
    if (!res)
      return false;
    auto arr = res.tryGetAsArrayRef();
    if (!arr)
      return false;
    vals.reserve(arr->size());
    for (ElemTy x : *arr)
      vals.push_back(static_cast<int64_t>(x));
    return true;
  };

  switch (intTy.getWidth()) {
  case 8:
    return isUnsigned ? pushValsFromResource(uint8_t{}) : pushValsFromResource(int8_t{});
  case 16:
    return isUnsigned ? pushValsFromResource(uint16_t{}) : pushValsFromResource(int16_t{});
  case 32:
    return isUnsigned ? pushValsFromResource(uint32_t{}) : pushValsFromResource(int32_t{});
  case 64:
    return isUnsigned ? pushValsFromResource(uint64_t{}) : pushValsFromResource(int64_t{});
  default:
    return false;
  }
}

static bool isTensorOfF32(Type t) {
  auto st = llvm::dyn_cast<ShapedType>(t);
  return st && st.getElementType().isF32();
}

static bool isTensorOfPosit(Type t) {
  auto st = llvm::dyn_cast<ShapedType>(t);
  return st && llvm::isa<mlir::posit::PositType>(st.getElementType());
}

// ------------- ONNXAddOp -> posit.add -------------

// [FIX] Forward declaration: ONNXAddOpLowering uses this helper before its definition.
static Value broadcastPositConstantIfNeeded(ConversionPatternRewriter &rewriter,
                                           Location loc, Value rhs,
                                           RankedTensorType outRtt,
                                           unsigned storageBits);
static Value buildZeroPositTensorConst(ConversionPatternRewriter &rewriter,
                                       Location loc, Type positTensorTy,
                                       unsigned storageBits);


struct ONNXAddOpLowering : public OpConversionPattern<mlir::ONNXAddOp> {
  using OpConversionPattern<mlir::ONNXAddOp>::OpConversionPattern;
  ONNXAddOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXAddOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXAddOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 把 result type 用 TypeConverter 轉成 posit 型別 (tensor<...x!posit.type<8,0>>)
    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    // adaptor 的 operands 已經是「轉換後」的型別
    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];

    // posit.add requires identical operand/result types. When ONNX output type is
    // unranked, pick a ranked operand type (if available) for stable lowering.
    Type addTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(addTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        addTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        addTy = rhsRtt;
    }

    // [FIX] Handle MNIST bias broadcasting for RHS posit constants.
    auto addRtt = llvm::dyn_cast<RankedTensorType>(addTy);
    auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType());
    // If ranks differ, first expand RHS rank by prefixing ones (e.g., 1000 -> 1x1000).
    // This avoids unresolved rank-changing tensor casts later in Krnl/LLVM lowering.
    if (addRtt && rhsRtt && rhsRtt.hasRank() &&
        rhsRtt.getRank() < addRtt.getRank() && rhsRtt.hasStaticShape()) {
      SmallVector<int64_t, 4> expanded(addRtt.getRank(), 1);
      for (int64_t i = 0, e = rhsRtt.getRank(); i < e; ++i)
        expanded[addRtt.getRank() - rhsRtt.getRank() + i] = rhsRtt.getDimSize(i);
      auto expandedTy =
          RankedTensorType::get(expanded, rhsRtt.getElementType());
      auto shapeTy = RankedTensorType::get(
          {static_cast<int64_t>(expanded.size())}, rewriter.getI64Type());
      SmallVector<Attribute, 4> shapeAttrs;
      shapeAttrs.reserve(expanded.size());
      for (int64_t d : expanded)
        shapeAttrs.push_back(rewriter.getI64IntegerAttr(d));
      Value shapeCst = rewriter.create<arith::ConstantOp>(
          loc, DenseIntElementsAttr::get(shapeTy, shapeAttrs));
      rhs = rewriter
                .create<mlir::posit::ReshapeOp>(loc, expandedTy, rhs, shapeCst)
                .getResult();
      rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType());
    }

    if (addRtt && rhs && rhs.getType() != addTy) {
      if (Value bc =
              broadcastPositConstantIfNeeded(rewriter, loc, rhs, addRtt, storageBits))
        rhs = bc;
    }

    if (lhs.getType() != addTy)
      lhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, addTy, lhs).getResult(0);
    if (rhs.getType() != addTy)
      rhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, addTy, rhs).getResult(0);

    Value addRes = rewriter.create<mlir::posit::AddOp>(loc, addTy, lhs, rhs).getResult();
    if (addRes.getType() != convertedType) {
      addRes = rewriter
                   .create<UnrealizedConversionCastOp>(loc, convertedType, addRes)
                   .getResult(0);
    }
    rewriter.replaceOp(op, addRes);
    return success();
  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

// sub mul div 新增
struct ONNXSubOpLowering : public OpConversionPattern<mlir::ONNXSubOp> {
  using OpConversionPattern<mlir::ONNXSubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXSubOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Type binTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(binTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        binTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        binTy = rhsRtt;
    }
    if (lhs.getType() != binTy)
      lhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, binTy, lhs).getResult(0);
    if (rhs.getType() != binTy)
      rhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, binTy, rhs).getResult(0);

    Value out = rewriter.create<mlir::posit::SubOp>(loc, binTy, lhs, rhs).getResult();
    if (out.getType() != convertedType)
      out = rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedType, out)
                .getResult(0);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ONNXMulOpLowering : public OpConversionPattern<mlir::ONNXMulOp> {
  using OpConversionPattern<mlir::ONNXMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXMulOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Type binTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(binTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        binTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        binTy = rhsRtt;
    }
    if (lhs.getType() != binTy)
      lhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, binTy, lhs).getResult(0);
    if (rhs.getType() != binTy)
      rhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, binTy, rhs).getResult(0);

    Value out = rewriter.create<mlir::posit::MulOp>(loc, binTy, lhs, rhs).getResult();
    if (out.getType() != convertedType)
      out = rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedType, out)
                .getResult(0);
    rewriter.replaceOp(op, out);
    return success();
  }
};

struct ONNXDivOpLowering : public OpConversionPattern<mlir::ONNXDivOp> {
  using OpConversionPattern<mlir::ONNXDivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXDivOp op,
                                typename OpConversionPattern::OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type origOutType = op.getResult().getType();
    Type convertedType = getTypeConverter()->convertType(origOutType);
    if (!convertedType)
      return rewriter.notifyMatchFailure(op, "failed to convert result type");

    if (adaptor.getOperands().size() != 2)
      return rewriter.notifyMatchFailure(op, "expected 2 operands");

    Value lhs = adaptor.getOperands()[0];
    Value rhs = adaptor.getOperands()[1];
    Type binTy = convertedType;
    if (llvm::isa<UnrankedTensorType>(binTy)) {
      if (auto lhsRtt = llvm::dyn_cast<RankedTensorType>(lhs.getType()))
        binTy = lhsRtt;
      else if (auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhs.getType()))
        binTy = rhsRtt;
    }
    if (lhs.getType() != binTy)
      lhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, binTy, lhs).getResult(0);
    if (rhs.getType() != binTy)
      rhs =
          rewriter.create<UnrealizedConversionCastOp>(loc, binTy, rhs).getResult(0);

    Value out = rewriter.create<mlir::posit::DivOp>(loc, binTy, lhs, rhs).getResult();
    if (out.getType() != convertedType)
      out = rewriter
                .create<UnrealizedConversionCastOp>(loc, convertedType, out)
                .getResult(0);
    rewriter.replaceOp(op, out);
    return success();
  }
};

// 需要新增i64 i32
 
// ONNXConstantOp -> posit.constant

//===----------------------------------------------------------------------===//
// [FIX] MNIST 需要的非 elementwise op：Conv / Relu / MaxPool / Reshape / MatMul
//===----------------------------------------------------------------------===//

// 產生一個全 0 的 posit tensor constant（用於 MatMul -> Gemm 的 C，beta=0 時可忽略）。
static Value buildZeroPositTensorConst(ConversionPatternRewriter &rewriter, Location loc,
                                      Type positTensorTy, unsigned storageBits) {
  auto rtt = llvm::dyn_cast<RankedTensorType>(positTensorTy);
  if (!rtt)
    return Value();
  // 目前 pass 固定 nbits=8, es=0。
  auto intElemTy = rewriter.getIntegerType(storageBits);
  auto intTensorTy = RankedTensorType::get(rtt.getShape(), intElemTy);

  // [FIX] Use the Attribute-based DenseElementsAttr::get for compatibility across MLIR versions.
  SmallVector<Attribute, 1> splat{IntegerAttr::get(intElemTy, 0)};
  DenseElementsAttr zeros = DenseElementsAttr::get(intTensorTy, splat);

  auto cst = rewriter.create<mlir::posit::ConstantOp>(loc, positTensorTy, zeros);
  return cst.getResult();
}


// [FIX][0127] Broadcast RHS posit.constant to match output tensor type when posit.add
// requires identical operand/result types. MNIST uses channel-bias constants shaped like
// (C,1,1) added to (N,C,H,W). We support general numpy-style broadcast for RHS constants.
static Value broadcastPositConstantIfNeeded(ConversionPatternRewriter &rewriter,
                                           Location loc, Value rhs,
                                           RankedTensorType outRtt,
                                           unsigned storageBits) {
  Value rhsBase = stripUnrealizedCast(rhs);
  if (!rhsBase || !outRtt)
    return Value();

  auto rhsRtt = llvm::dyn_cast<RankedTensorType>(rhsBase.getType());
  if (!rhsRtt || !rhsRtt.hasStaticShape() || !outRtt.hasStaticShape())
    return Value();

  if (rhsRtt == outRtt)
    return rhsBase;

  SmallVector<int64_t, 4> outShape(outRtt.getShape().begin(), outRtt.getShape().end());
  SmallVector<int64_t, 4> rhsShape(rhsRtt.getShape().begin(), rhsRtt.getShape().end());
  const int outRank = (int)outShape.size();
  const int rhsRank = (int)rhsShape.size();

  // Align ranks by prefixing ones.
  SmallVector<int64_t, 4> rhsAligned(outRank, 1);
  for (int i = 0; i < rhsRank; ++i)
    rhsAligned[outRank - rhsRank + i] = rhsShape[i];

  // Validate broadcast.
  for (int i = 0; i < outRank; ++i) {
    if (rhsAligned[i] != 1 && rhsAligned[i] != outShape[i])
      return Value();
  }

  auto computeStrides = [](ArrayRef<int64_t> shape) {
    SmallVector<int64_t, 4> strides(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i)
      strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
  };

  SmallVector<int64_t, 4> outStrides = computeStrides(outShape);
  SmallVector<int64_t, 4> rhsStrides = computeStrides(rhsAligned);

  // Case 1: rhs is posit.constant(i<bits>) -> broadcast in bit-domain.
  if (auto rhsCst = rhsBase.getDefiningOp<mlir::posit::ConstantOp>()) {
    Attribute a = rhsCst->getAttr("value");
    if (!a)
      a = rhsCst->getAttr("valueAttr");
    auto bitsEA = llvm::dyn_cast_or_null<ElementsAttr>(a);
    auto dense = llvm::dyn_cast_or_null<DenseElementsAttr>(bitsEA);
    auto intElemTy = rewriter.getIntegerType(storageBits);
    if (!dense || dense.getElementType() != intElemTy)
      return Value();

    SmallVector<APInt, 64> rhsBits;
    rhsBits.reserve((size_t)rhsRtt.getNumElements());
    if (dense.isSplat()) {
      APInt ap = dense.getSplatValue<APInt>();
      rhsBits.assign((size_t)rhsRtt.getNumElements(),
          APInt(storageBits, ap.getZExtValue()));
    } else {
      for (APInt ap : dense.getValues<APInt>())
        rhsBits.push_back(APInt(storageBits, ap.getZExtValue()));
    }

    const int64_t outNumElts = outRtt.getNumElements();
    SmallVector<APInt, 64> outBits(outNumElts);
    for (int64_t lin = 0; lin < outNumElts; ++lin) {
      int64_t rem = lin;
      int64_t rhsLin = 0;
      for (int d = 0; d < outRank; ++d) {
        int64_t idx = rem / outStrides[d];
        rem = rem % outStrides[d];
        int64_t rIdx = (rhsAligned[d] == 1) ? 0 : idx;
        rhsLin += rIdx * rhsStrides[d];
      }
      outBits[(size_t)lin] = rhsBits[(size_t)rhsLin];
    }

    auto outBitsTy = RankedTensorType::get(outShape, intElemTy);
    SmallVector<Attribute, 8> outAttrs;
    outAttrs.reserve((size_t)outNumElts);
    for (const APInt &v : outBits)
      outAttrs.push_back(IntegerAttr::get(intElemTy, v));

    DenseElementsAttr outBitsAttr = DenseElementsAttr::get(outBitsTy, outAttrs);
    auto newCst = rewriter.create<mlir::posit::ConstantOp>(loc, outRtt, outBitsAttr);
    return newCst.getResult();
  }

  // Case 2: rhs is posit.from_f32(arith.constant) -> broadcast in f32 domain,
  // then convert back with posit.from_f32 so add operands keep identical shape.
  if (auto fromF32 = rhsBase.getDefiningOp<mlir::posit::FromF32Op>()) {
    Value srcBase = stripUnrealizedCast(fromF32.getInput());
    auto srcCst = srcBase.getDefiningOp<mlir::arith::ConstantOp>();
    if (!srcCst)
      return Value();
    auto srcDense = llvm::dyn_cast<DenseElementsAttr>(srcCst.getValue());
    auto srcRtt = srcDense ? llvm::dyn_cast<RankedTensorType>(srcDense.getType())
                           : RankedTensorType();
    auto srcFp = srcDense ? llvm::dyn_cast<DenseFPElementsAttr>(srcDense)
                          : DenseFPElementsAttr();
    if (!srcDense || !srcRtt || !srcRtt.hasStaticShape() || !srcFp ||
        srcRtt.getShape() != rhsRtt.getShape())
      return Value();

    SmallVector<float, 64> rhsVals;
    rhsVals.reserve((size_t)rhsRtt.getNumElements());
    if (srcFp.isSplat()) {
      float v = srcFp.getSplatValue<APFloat>().convertToFloat();
      rhsVals.assign((size_t)rhsRtt.getNumElements(), v);
    } else {
      for (APFloat ap : srcFp.getValues<APFloat>())
        rhsVals.push_back(ap.convertToFloat());
    }

    const int64_t outNumElts = outRtt.getNumElements();
    SmallVector<float, 64> outVals(outNumElts);
    for (int64_t lin = 0; lin < outNumElts; ++lin) {
      int64_t rem = lin;
      int64_t rhsLin = 0;
      for (int d = 0; d < outRank; ++d) {
        int64_t idx = rem / outStrides[d];
        rem = rem % outStrides[d];
        int64_t rIdx = (rhsAligned[d] == 1) ? 0 : idx;
        rhsLin += rIdx * rhsStrides[d];
      }
      outVals[(size_t)lin] = rhsVals[(size_t)rhsLin];
    }

    auto f32Ty = rewriter.getF32Type();
    auto outF32Ty = RankedTensorType::get(outShape, f32Ty);
    SmallVector<Attribute, 8> outAttrs;
    outAttrs.reserve((size_t)outNumElts);
    for (float v : outVals)
      outAttrs.push_back(FloatAttr::get(f32Ty, v));
    auto outF32 = DenseElementsAttr::get(outF32Ty, outAttrs);
    Value outF32Cst = rewriter.create<mlir::arith::ConstantOp>(loc, outF32);
    return rewriter.create<mlir::posit::FromF32Op>(loc, outRtt, outF32Cst)
        .getResult();
  }

  return Value();
}

// QDQ lowering strategy:
// 1) Quantize->Dequantize pair: bypass int8 path and keep data in posit domain.
// 2) Dequantize(const int tensor, scale, zp): fold at compile-time to posit.constant.
struct ONNXDequantizeLinearOpLowering
    : public OpConversionPattern<mlir::ONNXDequantizeLinearOp> {
  using OpConversionPattern<mlir::ONNXDequantizeLinearOp>::OpConversionPattern;
  ONNXDequantizeLinearOpLowering(TypeConverter &tc, MLIRContext *ctx,
                                 unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXDequantizeLinearOp>(tc, ctx), nbits(nbits),
        es(es), storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXDequantizeLinearOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type origOutType = op.getResult().getType();
    auto positElemTy = mlir::posit::PositType::get(rewriter.getContext(), nbits, es);

    auto toPositShaped = [&](Type srcTy) -> Type {
      if (auto rtt = llvm::dyn_cast<RankedTensorType>(srcTy))
        return RankedTensorType::get(rtt.getShape(), positElemTy);
      if (auto utt = llvm::dyn_cast<UnrankedTensorType>(srcTy))
        return UnrankedTensorType::get(positElemTy);
      return Type();
    };
    auto toF32Shaped = [&](Type srcTy) -> Type {
      auto f32Ty = rewriter.getF32Type();
      if (auto rtt = llvm::dyn_cast<RankedTensorType>(srcTy))
        return RankedTensorType::get(rtt.getShape(), f32Ty);
      if (auto utt = llvm::dyn_cast<UnrankedTensorType>(srcTy))
        return UnrankedTensorType::get(f32Ty);
      return Type();
    };

    Type positOutType = toPositShaped(origOutType);
    if (!positOutType)
      return rewriter.notifyMatchFailure(
          op, "dequantize result must be a ranked/unranked tensor");

    auto buildRuntimeDQPosit = [&](Value x, Type outPosTy, double scale,
                                   int64_t zeroPoint) -> Value {
      bool inputSigned = true;
      if (auto stTy = llvm::dyn_cast<ShapedType>(x.getType())) {
        if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
          inputSigned = !it.isUnsignedInteger();
      }
      OperationState st(loc, mlir::posit::DequantizeLinearOp::getOperationName());
      st.addOperands({x});
      st.addTypes({outPosTy});
      st.addAttribute(
          "scale", FloatAttr::get(rewriter.getF32Type(), static_cast<float>(scale)));
      st.addAttribute("zero_point", rewriter.getI64IntegerAttr(zeroPoint));
      st.addAttribute("has_zero_point", rewriter.getBoolAttr(true));
      st.addAttribute("axis", rewriter.getI64IntegerAttr(1));
      st.addAttribute("input_signed", rewriter.getBoolAttr(inputSigned));
      return rewriter.create(st)->getResult(0);
    };

    auto buildRuntimeDQPositFromQParams = [&](Value qTensor, Value yScale,
                                              Value yZeroPoint, int64_t axisAttr,
                                              Type outPosTy) -> Value {
      RankedTensorType scaleRtt;
      SmallVector<double, 16> scaleVals;
      if (!getDenseFloatTensorFromValue(yScale, scaleRtt, scaleVals) ||
          scaleVals.empty())
        return Value();
      double scale = scaleVals.front();
      bool perAxis = scaleVals.size() > 1;

      int64_t zeroPoint = 0;
      bool hasZeroPoint = false;
      RankedTensorType zpRtt;
      SmallVector<int64_t, 16> zpVals;
      if (!llvm::isa<NoneType>(yZeroPoint.getType())) {
        hasZeroPoint = true;
        if (getDenseIntegerTensorFromValue(yZeroPoint, zpRtt, zpVals)) {
          if (zpVals.empty())
            return Value();
          zeroPoint = zpVals.front();
          if (zpVals.size() > 1)
            perAxis = true;
        } else if (getScalarIntFromValue(yZeroPoint, zeroPoint)) {
          zpVals.push_back(zeroPoint);
        } else {
          return Value();
        }
      }

      int64_t axis = axisAttr;
      if (auto qTy = llvm::dyn_cast<ShapedType>(qTensor.getType())) {
        if (qTy.hasRank()) {
          int64_t rank = qTy.getRank();
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank)
            return Value();
          if (perAxis && !qTy.isDynamicDim(axis)) {
            int64_t axisDim = qTy.getDimSize(axis);
            if (axisDim > 0 && static_cast<int64_t>(scaleVals.size()) != axisDim &&
                static_cast<int64_t>(scaleVals.size()) != 1)
              return Value();
            if (hasZeroPoint && !zpVals.empty() &&
                static_cast<int64_t>(zpVals.size()) != axisDim &&
                static_cast<int64_t>(zpVals.size()) != 1)
              return Value();
          }
        }
      }

      bool inputSigned = true;
      if (auto stTy = llvm::dyn_cast<ShapedType>(qTensor.getType())) {
        if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
          inputSigned = !it.isUnsignedInteger();
      }

      OperationState st(loc, mlir::posit::DequantizeLinearOp::getOperationName());
      st.addOperands({qTensor});
      st.addTypes({outPosTy});
      st.addAttribute(
          "scale", FloatAttr::get(rewriter.getF32Type(), static_cast<float>(scale)));
      st.addAttribute("zero_point", rewriter.getI64IntegerAttr(zeroPoint));
      st.addAttribute("has_zero_point", rewriter.getBoolAttr(hasZeroPoint));
      st.addAttribute("axis", rewriter.getI64IntegerAttr(axis));
      st.addAttribute("input_signed", rewriter.getBoolAttr(inputSigned));

      if (perAxis) {
        auto f32Ty = rewriter.getF32Type();
        auto scaleTy =
            RankedTensorType::get({static_cast<int64_t>(scaleVals.size())}, f32Ty);
        SmallVector<Attribute, 16> sAttrs;
        sAttrs.reserve(scaleVals.size());
        for (double v : scaleVals)
          sAttrs.push_back(FloatAttr::get(f32Ty, static_cast<float>(v)));
        st.addAttribute("scale_values", DenseElementsAttr::get(scaleTy, sAttrs));

        if (hasZeroPoint) {
          if (zpVals.empty())
            zpVals.push_back(zeroPoint);
          auto i64Ty = rewriter.getI64Type();
          auto zpTy =
              RankedTensorType::get({static_cast<int64_t>(zpVals.size())}, i64Ty);
          SmallVector<APInt, 16> zpAp;
          zpAp.reserve(zpVals.size());
          for (int64_t v : zpVals)
            zpAp.push_back(APInt(64, static_cast<uint64_t>(v), true));
          st.addAttribute(
              "zero_point_values", DenseIntElementsAttr::get(zpTy, zpAp));
        }
      }

      return rewriter.create(st)->getResult(0);
    };

    Value xBase = stripUnrealizedCast(op.getX());

    // Case A0: DQ(QLinearMatMul(...)).
    if (auto qmm = xBase.getDefiningOp<mlir::ONNXQLinearMatMulOp>()) {
      Type aPosTy = toPositShaped(qmm.getA().getType());
      Type bPosTy = toPositShaped(qmm.getB().getType());
      if (!aPosTy || !bPosTy)
        return rewriter.notifyMatchFailure(
            op, "QLinearMatMul supports only tensor A/B types");

      double aScale = 0.0, bScale = 0.0;
      int64_t aZeroPoint = 0, bZeroPoint = 0;
      if (!getScalarFloatFromValue(qmm.getAScale(), aScale) ||
          !getScalarFloatFromValue(qmm.getBScale(), bScale))
        return rewriter.notifyMatchFailure(
            op, "QLinearMatMul path requires scalar a/b scales");
      if (!getScalarIntFromValue(qmm.getAZeroPoint(), aZeroPoint) ||
          !getScalarIntFromValue(qmm.getBZeroPoint(), bZeroPoint))
        return rewriter.notifyMatchFailure(
            op, "QLinearMatMul path requires scalar a/b zero points");

      auto buildPositFromQorInt = [&](Value in, Type outPosTy, double s,
                                      int64_t zp) -> Value {
        Value inBase = stripUnrealizedCast(in);
        return buildRuntimeDQPosit(inBase, outPosTy, s, zp);
      };

      Value aPos = buildPositFromQorInt(qmm.getA(), aPosTy, aScale, aZeroPoint);
      Value bPos = buildPositFromQorInt(qmm.getB(), bPosTy, bScale, bZeroPoint);
      if (!aPos || !bPos)
        return rewriter.notifyMatchFailure(
            op, "failed to build posit operands for QLinearMatMul");
      Value c = buildZeroPositTensorConst(
          rewriter, loc, RankedTensorType::get({1}, positElemTy), storageBits);
      if (!c)
        return rewriter.notifyMatchFailure(
            op, "failed to build GEMM C=0 constant for QLinearMatMul");

      OperationState gst(loc, mlir::posit::GemmOp::getOperationName());
      gst.addOperands({aPos, bPos, c});
      gst.addTypes({positOutType});
      gst.addAttribute("alpha", FloatAttr::get(rewriter.getF32Type(), 1.0));
      gst.addAttribute("beta", FloatAttr::get(rewriter.getF32Type(), 0.0));
      gst.addAttribute("transA", rewriter.getI64IntegerAttr(0));
      gst.addAttribute("transB", rewriter.getI64IntegerAttr(0));
      Value gemmPos = rewriter.create(gst)->getResult(0);

      rewriter.replaceOp(op, gemmPos);
      if (qmm.getResult().use_empty())
        rewriter.eraseOp(qmm);
      return success();
    }

    // Case A: DQ(Q(...)).
    if (auto qop = xBase.getDefiningOp<mlir::ONNXQuantizeLinearOp>()) {
      Value qTensor = rewriter.getRemappedValue(qop.getResult());
      if (!qTensor && !adaptor.getOperands().empty())
        qTensor = adaptor.getOperands()[0];
      if (!qTensor)
        qTensor = op.getX();
      qTensor = stripUnrealizedCast(qTensor);

      // Keep fractional quant-domain information by default:
      // do not force early int round/clamp in DQ(Q(x)); this avoids cases like
      // 7.2 being collapsed to 7 before posit conversion.
      constexpr bool preferFractionalQDomain = true;
      if (!preferFractionalQDomain) {
        // Strict ONNX QDQ path (round/clamp + scale/zp) if needed.
        if (Value pos = buildRuntimeDQPositFromQParams(
                qTensor, qop.getYScale(), qop.getYZeroPoint(), qop.getAxis(),
                positOutType)) {
          rewriter.replaceOp(op, pos);
          if (qop.getResult().use_empty())
            rewriter.eraseOp(qop);
          return success();
        }
      }

      // Default path: direct f32->posit from Q input.
      Value src = stripUnrealizedCast(qop.getX());
      Value srcRemapped = rewriter.getRemappedValue(src);
      if (srcRemapped)
        src = stripUnrealizedCast(srcRemapped);
      if (!isTensorOfF32(src.getType()) && isTensorOfPosit(src.getType())) {
        Type srcF32Ty = qop.getX().getType();
        if (!isTensorOfF32(srcF32Ty))
          srcF32Ty = toF32Shaped(src.getType());
        if (!srcF32Ty)
          return rewriter.notifyMatchFailure(
              op, "Q->DQ source posit type cannot be converted to f32");
        src = rewriter.create<mlir::posit::ToF32Op>(loc, srcF32Ty, src).getResult();
      }
      if (!isTensorOfF32(src.getType()))
        return rewriter.notifyMatchFailure(op, "Q->DQ source must be f32/posit tensor");
      Type posTy = toPositShaped(src.getType());
      if (!posTy)
        return rewriter.notifyMatchFailure(
            op, "failed to build posit type for fake QDQ tensor");
      Value pos =
          rewriter.create<mlir::posit::FromF32Op>(loc, posTy, src).getResult();
      rewriter.replaceOp(op, pos);
      if (qop.getResult().use_empty())
        rewriter.eraseOp(qop);
      return success();
    }

    // Case B: compile-time dequantization for int constants.
    RankedTensorType xRtt;
    SmallVector<int64_t, 16> xVals;
    if (getDenseIntegerTensorFromValue(op.getX(), xRtt, xVals)) {
      double scale = 0.0;
      if (!getScalarFloatFromValue(op.getXScale(), scale))
        return rewriter.notifyMatchFailure(op, "x_scale must be scalar constant for now");

      int64_t zeroPoint = 0;
      if (!llvm::isa<NoneType>(op.getXZeroPoint().getType()) &&
          !getScalarIntFromValue(op.getXZeroPoint(), zeroPoint)) {
        return rewriter.notifyMatchFailure(
            op, "x_zero_point must be scalar integer constant or none");
      }

      SmallVector<Attribute, 16> dequantAttrs;
      dequantAttrs.reserve(xVals.size());
      auto f32Ty = rewriter.getF32Type();
      for (int64_t x : xVals) {
        const double dequantized =
            (static_cast<double>(x) - static_cast<double>(zeroPoint)) * scale;
        dequantAttrs.push_back(
            FloatAttr::get(f32Ty, static_cast<float>(dequantized)));
      }

      auto f32TensorTy = RankedTensorType::get(xRtt.getShape(), f32Ty);
      DenseElementsAttr dequantAttr = DenseElementsAttr::get(f32TensorTy, dequantAttrs);
      Value f32Const = rewriter.create<arith::ConstantOp>(loc, dequantAttr).getResult();

      auto rankedPositTy = RankedTensorType::get(xRtt.getShape(), positElemTy);
      Value rankedPosit =
          rewriter.create<mlir::posit::FromF32Op>(loc, rankedPositTy, f32Const)
              .getResult();

      rewriter.replaceOp(op, rankedPosit);
      return success();
    }

    // Case C: runtime dequantization for non-constant integer tensors with
    // scalar or per-axis constant scale / zero_point.
    double scale = 0.0;
    int64_t zeroPoint = 0;
    bool hasZeroPoint = false;
    RankedTensorType scaleRtt;
    SmallVector<double, 16> scaleVals;
    if (getDenseFloatTensorFromValue(op.getXScale(), scaleRtt, scaleVals)) {
      if (scaleVals.empty())
        return rewriter.notifyMatchFailure(op, "x_scale cannot be empty");
      scale = scaleVals.front();
      bool perAxis = scaleVals.size() > 1;

      RankedTensorType zpRtt;
      SmallVector<int64_t, 16> zpVals;
      if (!llvm::isa<NoneType>(op.getXZeroPoint().getType())) {
        hasZeroPoint = true;
        if (getDenseIntegerTensorFromValue(op.getXZeroPoint(), zpRtt, zpVals)) {
          if (zpVals.empty())
            return rewriter.notifyMatchFailure(op, "x_zero_point cannot be empty");
          zeroPoint = zpVals.front();
          if (zpVals.size() > 1)
            perAxis = true;
        } else if (getScalarIntFromValue(op.getXZeroPoint(), zeroPoint)) {
          zpVals.push_back(zeroPoint);
        } else {
          return rewriter.notifyMatchFailure(
              op, "x_zero_point must be scalar/1D integer constant or none");
        }
      }

      int64_t axis = op.getAxis();
      if (auto xTy = llvm::dyn_cast<ShapedType>(op.getX().getType())) {
        if (xTy.hasRank()) {
          int64_t rank = xTy.getRank();
          if (axis < 0)
            axis += rank;
          if (axis < 0 || axis >= rank)
            return rewriter.notifyMatchFailure(op, "invalid dequantize axis");
          if (perAxis && xTy.isDynamicDim(axis) == false) {
            int64_t axisDim = xTy.getDimSize(axis);
            if (axisDim > 0 && static_cast<int64_t>(scaleVals.size()) != axisDim &&
                static_cast<int64_t>(scaleVals.size()) != 1)
              return rewriter.notifyMatchFailure(
                  op, "x_scale length must match axis dimension or be 1");
            if (hasZeroPoint && !zpVals.empty() &&
                static_cast<int64_t>(zpVals.size()) != axisDim &&
                static_cast<int64_t>(zpVals.size()) != 1)
              return rewriter.notifyMatchFailure(
                  op, "x_zero_point length must match axis dimension or be 1");
          }
        }
      }

      OperationState st(loc, mlir::posit::DequantizeLinearOp::getOperationName());
      st.addOperands({adaptor.getX()});
      st.addTypes({positOutType});
      st.addAttribute(
          "scale", FloatAttr::get(rewriter.getF32Type(), static_cast<float>(scale)));
      st.addAttribute("zero_point", rewriter.getI64IntegerAttr(zeroPoint));
      st.addAttribute("has_zero_point", rewriter.getBoolAttr(hasZeroPoint));
      st.addAttribute("axis", rewriter.getI64IntegerAttr(axis));
      bool inputSigned = true;
      if (auto stTy = llvm::dyn_cast<ShapedType>(op.getX().getType())) {
        if (auto it = llvm::dyn_cast<IntegerType>(stTy.getElementType()))
          inputSigned = !it.isUnsignedInteger();
      }
      st.addAttribute("input_signed", rewriter.getBoolAttr(inputSigned));

      if (perAxis) {
        auto f32Ty = rewriter.getF32Type();
        auto scaleTy =
            RankedTensorType::get({static_cast<int64_t>(scaleVals.size())}, f32Ty);
        SmallVector<Attribute, 16> sAttrs;
        sAttrs.reserve(scaleVals.size());
        for (double v : scaleVals)
          sAttrs.push_back(FloatAttr::get(f32Ty, static_cast<float>(v)));
        st.addAttribute("scale_values", DenseElementsAttr::get(scaleTy, sAttrs));

        if (hasZeroPoint) {
          if (zpVals.empty())
            zpVals.push_back(zeroPoint);
          auto i64Ty = rewriter.getI64Type();
          auto zpTy =
              RankedTensorType::get({static_cast<int64_t>(zpVals.size())}, i64Ty);
          SmallVector<APInt, 16> zpAp;
          zpAp.reserve(zpVals.size());
          for (int64_t v : zpVals)
            zpAp.push_back(APInt(64, static_cast<uint64_t>(v), true));
          st.addAttribute(
              "zero_point_values", DenseIntElementsAttr::get(zpTy, zpAp));
        }
      }

      Operation *newOp = rewriter.create(st);
      rewriter.replaceOp(op, newOp->getResult(0));
      return success();
    }

    // Case D: passthrough when source was already converted/remapped.
    Value remappedX = rewriter.getRemappedValue(op.getX());
    if (!remappedX && !adaptor.getOperands().empty())
      remappedX = adaptor.getOperands()[0];
    if (!remappedX)
      remappedX = op.getX();

    if (isTensorOfF32(remappedX.getType())) {
      Type remappedPosTy = toPositShaped(remappedX.getType());
      if (!remappedPosTy)
        return rewriter.notifyMatchFailure(
            op, "failed to build posit type for fallback dequantize source");
      Value remappedPos =
          rewriter.create<mlir::posit::FromF32Op>(loc, remappedPosTy, remappedX)
              .getResult();
      rewriter.replaceOp(op, remappedPos);
      return success();
    }
    if (isTensorOfPosit(remappedX.getType())) {
      rewriter.replaceOp(op, remappedX);
      return success();
    }

    return rewriter.notifyMatchFailure(op,
        "dequantize not matched (expected Q->DQ, const fold, or runtime dequant)");
  }

  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

struct ONNXReshapeOpLowering : public OpConversionPattern<mlir::ONNXReshapeOp> {
  using OpConversionPattern<mlir::ONNXReshapeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXReshapeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert reshape result type");

    // ONNXReshape operands: data, shape (shape tensor stays i64 by TypeConverter policy)
    Value data = adaptor.getOperands()[0];
    Value shape = adaptor.getOperands()[1];

    OperationState st(loc, mlir::posit::ReshapeOp::getOperationName());
    st.addOperands({data, shape});
    st.addTypes({outTy});
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXUnsqueezeOpLowering : public OpConversionPattern<mlir::ONNXUnsqueezeOp> {
  using OpConversionPattern<mlir::ONNXUnsqueezeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXUnsqueezeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    auto outRtt = llvm::dyn_cast_or_null<RankedTensorType>(outTy);
    if (!outRtt)
      return rewriter.notifyMatchFailure(op, "expected ranked result type");

    Value data = adaptor.getData();
    auto inRtt = llvm::dyn_cast<RankedTensorType>(data.getType());
    if (!inRtt)
      return rewriter.notifyMatchFailure(op, "expected ranked input type");

    RankedTensorType axesTy;
    SmallVector<int64_t, 8> axesVals;
    if (!getDenseIntegerTensorFromValue(op.getAxes(), axesTy, axesVals))
      return rewriter.notifyMatchFailure(op, "axes must be constant integer tensor");

    const int64_t outRank = outRtt.getRank();
    SmallVector<char, 8> isUnsqueezeAxis(static_cast<size_t>(outRank), 0);
    for (int64_t axis : axesVals) {
      int64_t a = axis < 0 ? axis + outRank : axis;
      if (a < 0 || a >= outRank)
        return rewriter.notifyMatchFailure(op, "axis out of range");
      isUnsqueezeAxis[static_cast<size_t>(a)] = 1;
    }

    SmallVector<Value, 8> shapeElems;
    shapeElems.reserve(static_cast<size_t>(outRank));
    int64_t inPos = 0;
    IntegerType i64Ty = rewriter.getI64Type();
    for (int64_t outPos = 0; outPos < outRank; ++outPos) {
      if (isUnsqueezeAxis[static_cast<size_t>(outPos)]) {
        shapeElems.push_back(rewriter.create<arith::ConstantOp>(
            loc, i64Ty, rewriter.getI64IntegerAttr(1)));
        continue;
      }

      if (inPos >= inRtt.getRank())
        return rewriter.notifyMatchFailure(op, "rank mismatch in unsqueeze lowering");

      if (!inRtt.isDynamicDim(inPos)) {
        shapeElems.push_back(rewriter.create<arith::ConstantOp>(loc, i64Ty,
            rewriter.getI64IntegerAttr(inRtt.getDimSize(inPos))));
      } else {
        Value inAxis = rewriter.create<arith::ConstantIndexOp>(loc, inPos);
        Value dimIdx = rewriter.create<tensor::DimOp>(loc, data, inAxis);
        Value dimI64 = rewriter.create<arith::IndexCastOp>(loc, i64Ty, dimIdx);
        shapeElems.push_back(dimI64);
      }
      ++inPos;
    }

    auto shapeTy = RankedTensorType::get({outRank}, i64Ty);
    Value shapeTensor =
        rewriter.create<tensor::FromElementsOp>(loc, shapeTy, shapeElems);

    OperationState st(loc, mlir::posit::ReshapeOp::getOperationName());
    st.addOperands({data, shapeTensor});
    st.addTypes({outTy});
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXReluOpLowering : public OpConversionPattern<mlir::ONNXReluOp> {
  using OpConversionPattern<mlir::ONNXReluOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXReluOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert relu result type");

    Value x = adaptor.getOperands()[0];
    OperationState st(loc, mlir::posit::ReluOp::getOperationName());
    st.addOperands({x});
    st.addTypes({outTy});
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXClipOpLowering : public OpConversionPattern<mlir::ONNXClipOp> {
  using OpConversionPattern<mlir::ONNXClipOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXClipOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert clip result type");

    Value x = adaptor.getInput();
    OperationState st(loc, mlir::posit::ClipOp::getOperationName());
    st.addOperands({x});
    st.addTypes({outTy});

    double minVal = 0.0;
    double maxVal = 0.0;
    bool hasMin = false;
    bool hasMax = false;
    Value minInput = op.getMin();
    Value maxInput = op.getMax();
    if (minInput && !llvm::isa<NoneType>(minInput.getType())) {
      hasMin = getScalarFloatFromValue(minInput, minVal);
      if (!hasMin)
        return rewriter.notifyMatchFailure(op, "clip min must be scalar constant");
    }
    if (maxInput && !llvm::isa<NoneType>(maxInput.getType())) {
      hasMax = getScalarFloatFromValue(maxInput, maxVal);
      if (!hasMax)
        return rewriter.notifyMatchFailure(op, "clip max must be scalar constant");
    }

    st.addAttribute("has_min", rewriter.getBoolAttr(hasMin));
    st.addAttribute("has_max", rewriter.getBoolAttr(hasMax));
    st.addAttribute("min_val", FloatAttr::get(rewriter.getF32Type(),
        static_cast<float>(minVal)));
    st.addAttribute("max_val", FloatAttr::get(rewriter.getF32Type(),
        static_cast<float>(maxVal)));

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXMaxPoolSingleOutOpLowering
    : public OpConversionPattern<mlir::ONNXMaxPoolSingleOutOp> {
  using OpConversionPattern<mlir::ONNXMaxPoolSingleOutOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXMaxPoolSingleOutOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert maxpool result type");

    Value x = adaptor.getOperands()[0];

    OperationState st(loc, mlir::posit::MaxPool2DOp::getOperationName());
    st.addOperands({x});
    st.addTypes({outTy});

    // attrs: kernel_shape / strides / pads / ceil_mode (照 ONNX attr)
    if (auto a = op->getAttr("kernel_shape")) st.addAttribute("kernel_shape", a);
    if (auto a = op->getAttr("strides"))      st.addAttribute("strides", a);
    if (auto a = op->getAttr("pads"))         st.addAttribute("pads", a);
    if (auto a = op->getAttr("ceil_mode"))    st.addAttribute("ceil_mode", a);
    // optional: auto_pad (不一定用得到，但保留)
    if (auto a = op->getAttr("auto_pad"))     st.addAttribute("auto_pad", a);

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXFlattenOpLowering : public OpConversionPattern<mlir::ONNXFlattenOp> {
  using OpConversionPattern<mlir::ONNXFlattenOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXFlattenOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert flatten result type");

    OperationState st(loc, mlir::posit::FlattenOp::getOperationName());
    st.addOperands({adaptor.getInput()});
    st.addTypes({outTy});
    st.addAttribute("axis", rewriter.getI64IntegerAttr(op.getAxis()));
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXConvOpLowering : public OpConversionPattern<mlir::ONNXConvOp> {
  using OpConversionPattern<mlir::ONNXConvOp>::OpConversionPattern;
  ONNXConvOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXConvOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXConvOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert conv result type");

    auto ops = adaptor.getOperands();
    if (ops.size() < 2)
      return rewriter.notifyMatchFailure(op, "conv expects at least X and W");
    Value x = ops[0];
    Value w = ops[1];
    Value b;
    if (ops.size() >= 3) {
      b = ops[2];
      if (b && llvm::isa<NoneType>(b.getType()))
        b = Value();
      // [FIX] ONNX bias may be represented as `none` (from onnx.NoValue / ONNXNoneOp).
      // If so, materialize a zero bias tensor.
    }
    if (!b) {
      int64_t c = ShapedType::kDynamic;
      Type elemTy;
      if (auto wTy = llvm::dyn_cast<RankedTensorType>(w.getType())) {
        if (wTy.getRank() >= 1) {
          c = wTy.getShape()[0];
          elemTy = wTy.getElementType();
        }
      }
      if ((c == ShapedType::kDynamic || !elemTy) &&
          llvm::isa<RankedTensorType>(outTy)) {
        auto outRtt = llvm::cast<RankedTensorType>(outTy);
        if (outRtt.getRank() >= 2) {
          c = outRtt.getShape()[1];
          elemTy = outRtt.getElementType();
        }
      }
      if (c == ShapedType::kDynamic || !elemTy)
        return rewriter.notifyMatchFailure(
            op, "cannot materialize default conv bias without known output channels");
      auto biasTy = RankedTensorType::get({c}, elemTy);
      b = buildZeroPositTensorConst(rewriter, loc, biasTy, storageBits);
      if (!b)
        return rewriter.notifyMatchFailure(op, "failed to build default bias constant");
    }

    OperationState st(loc, mlir::posit::Conv2DOp::getOperationName());
    st.addOperands({x, w, b});
    st.addTypes({outTy});

    // attrs: strides / pads / dilations / group / auto_pad (照 ONNX attr)
    if (auto a = op->getAttr("strides"))   st.addAttribute("strides", a);
    if (auto a = op->getAttr("pads"))      st.addAttribute("pads", a);
    if (auto a = op->getAttr("dilations")) st.addAttribute("dilations", a);
    if (auto a = op->getAttr("kernel_shape")) st.addAttribute("kernel_shape", a);
    if (auto a = op->getAttr("group"))     st.addAttribute("group", a);
    if (auto a = op->getAttr("auto_pad"))  st.addAttribute("auto_pad", a);

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

struct ONNXMatMulOpLowering : public OpConversionPattern<mlir::ONNXMatMulOp> {
  using OpConversionPattern<mlir::ONNXMatMulOp>::OpConversionPattern;
  ONNXMatMulOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXMatMulOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXMatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert matmul result type");

    Value a = adaptor.getOperands()[0];
    Value b = adaptor.getOperands()[1];

    Type elemTy;
    if (auto outRtt = llvm::dyn_cast<RankedTensorType>(outTy))
      elemTy = outRtt.getElementType();
    else if (auto outUtt = llvm::dyn_cast<UnrankedTensorType>(outTy))
      elemTy = outUtt.getElementType();
    if (!elemTy)
      return rewriter.notifyMatchFailure(op, "cannot infer matmul output element type");
    auto cTy = RankedTensorType::get({1}, elemTy);
    Value c = buildZeroPositTensorConst(rewriter, loc, cTy, storageBits);
    if (!c)
      return rewriter.notifyMatchFailure(op, "failed to build GEMM C=0 constant");

    OperationState st(loc, mlir::posit::GemmOp::getOperationName());
    st.addOperands({a, b, c});
    st.addTypes({outTy});

    st.addAttribute("alpha", FloatAttr::get(rewriter.getF32Type(), 1.0));
    st.addAttribute("beta",  FloatAttr::get(rewriter.getF32Type(), 0.0));
    st.addAttribute("transA", rewriter.getI64IntegerAttr(0));
    st.addAttribute("transB", rewriter.getI64IntegerAttr(0));

    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

struct ONNXGemmOpLowering : public OpConversionPattern<mlir::ONNXGemmOp> {
  using OpConversionPattern<mlir::ONNXGemmOp>::OpConversionPattern;
  ONNXGemmOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXGemmOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXGemmOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert gemm result type");

    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!c || llvm::isa<NoneType>(c.getType())) {
      Type elemTy;
      if (auto outRtt = llvm::dyn_cast<RankedTensorType>(outTy))
        elemTy = outRtt.getElementType();
      else if (auto outUtt = llvm::dyn_cast<UnrankedTensorType>(outTy))
        elemTy = outUtt.getElementType();
      if (!elemTy)
        return rewriter.notifyMatchFailure(op, "cannot infer gemm output element type");
      auto cTy = RankedTensorType::get({1}, elemTy);
      c = buildZeroPositTensorConst(rewriter, loc, cTy, storageBits);
      if (!c)
        return rewriter.notifyMatchFailure(op, "failed to build GEMM default C");
    }

    OperationState st(loc, mlir::posit::GemmOp::getOperationName());
    st.addOperands({a, b, c});
    st.addTypes({outTy});
    st.addAttribute(
        "alpha", FloatAttr::get(rewriter.getF32Type(), op.getAlpha().convertToFloat()));
    st.addAttribute(
        "beta", FloatAttr::get(rewriter.getF32Type(), op.getBeta().convertToFloat()));
    st.addAttribute("transA", rewriter.getI64IntegerAttr(op.getTransA()));
    st.addAttribute("transB", rewriter.getI64IntegerAttr(op.getTransB()));
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }

  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

struct ONNXReduceMeanV13OpLowering
    : public OpConversionPattern<mlir::ONNXReduceMeanV13Op> {
  using OpConversionPattern<mlir::ONNXReduceMeanV13Op>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXReduceMeanV13Op op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type outTy = getTypeConverter()->convertType(op.getResult().getType());
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "failed to convert reduce_mean result type");

    SmallVector<int64_t, 4> axesVals;
    if (std::optional<ArrayAttr> axesAttr = op.getAxes()) {
      for (Attribute a : *axesAttr)
        axesVals.push_back(llvm::cast<IntegerAttr>(a).getValue().getSExtValue());
    }

    OperationState st(loc, mlir::posit::ReduceMeanOp::getOperationName());
    st.addOperands({adaptor.getData()});
    st.addTypes({outTy});
    st.addAttribute("axes", rewriter.getI64ArrayAttr(axesVals));
    st.addAttribute("keepdims", rewriter.getI64IntegerAttr(op.getKeepdims()));
    st.addAttribute("noop_with_empty_axes", rewriter.getI64IntegerAttr(0));
    Operation *newOp = rewriter.create(st);
    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct ONNXConstantOpLowering
    : public OpConversionPattern<mlir::ONNXConstantOp> {
  using OpConversionPattern<mlir::ONNXConstantOp>::OpConversionPattern;
  ONNXConstantOpLowering(TypeConverter &tc, MLIRContext *ctx, unsigned nbits, unsigned es)
      : OpConversionPattern<mlir::ONNXConstantOp>(tc, ctx), nbits(nbits), es(es),
        storageBits(getPositStorageBitWidth(nbits)) {}

  LogicalResult matchAndRewrite(mlir::ONNXConstantOp op,
                              typename OpConversionPattern::OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Location loc = op.getLoc();

	  // 1. 抓出 "value" attribute。
	  // [FIX] ONNX Constant 的 value 可能是 dense<...> 或 dense_resource<...>。
	  // dense_resource 不是 DenseFPElementsAttr，必須用 ElementsAttr +
	  // DenseResourceElementsAttrBase<T>::tryGetAsArrayRef() 來取資料。
	  Attribute valueAttr = op->getAttr("value");
  if (!valueAttr)
    return rewriter.notifyMatchFailure(
        op, "ONNXConstantOp has no 'value' attribute (other forms not handled yet)");

	  // [FIX] 接受任何 ElementsAttr（DenseElementsAttr / DenseResourceElementsAttr / splat...）。
	  auto elements = llvm::dyn_cast<ElementsAttr>(valueAttr);
	  if (!elements)
	    return rewriter.notifyMatchFailure(
	        op, "only ElementsAttr (dense/dense_resource) is handled for ONNXConstantOp");

		  // 2. 只接受浮點 element（整數走 shape-constant 路徑）
	  auto elemType = elements.getElementType();
  // Integer constants (typically shape tensors) are kept in ONNX dialect and
  // lowered by the later ONNX->Krnl pass, so this pattern only handles floats.
  if (llvm::isa<IntegerType>(elemType))
    return rewriter.notifyMatchFailure(op, "integer ONNXConstant is kept as ONNX");

		  if (!llvm::isa<FloatType>(elemType))
	    return rewriter.notifyMatchFailure(
	      op, "only floating-point element type is handled");

  // Scalar float constants are kept as ONNXConstant (see conversion target
  // dynamic legality) and lowered later by ONNX->Krnl.
  if (elements.getNumElements() == 1)
    return rewriter.notifyMatchFailure(op, "scalar float ONNXConstant is kept as ONNX");

	  // 3. 目前先只處理 RankedTensor，且 ONNXConstant 的 result 也應該是 tensor
	  auto outType = op.getResult().getType();
	  auto origOutType = llvm::dyn_cast<RankedTensorType>(outType);
	  if (!origOutType)
    return rewriter.notifyMatchFailure(
          op, "only ranked tensor results are handled");

	  // 4. 用 TypeConverter 把 result type 轉成 tensor<...x!posit.type<nbits,es>>
	  Type convertedType = getTypeConverter()->convertType(origOutType);
	  if (!convertedType)
	    return rewriter.notifyMatchFailure(
	        op, "failed to convert constant result type to posit tensor");

	  // 5. 將 ONNX float tensor constant 保持為 arith.constant，接著用
	  //    posit.from_f32 轉成 posit tensor。
	  auto f32ElemTy = rewriter.getF32Type();
	  auto f32TensorTy = RankedTensorType::get(origOutType.getShape(), f32ElemTy);
      Value f32Const;
	  if (elemType.isF64()) {
	    SmallVector<double, 4> fpVals;
	    if (failed(collectFPValuesAsDouble(elements, fpVals)))
	      return rewriter.notifyMatchFailure(
	          op, "failed to read f64 constant payload from dense/dense_resource");
	    SmallVector<Attribute, 8> f32Attrs;
	    f32Attrs.reserve(fpVals.size());
	    for (double d : fpVals)
	      f32Attrs.push_back(FloatAttr::get(f32ElemTy, static_cast<float>(d)));
	    DenseElementsAttr f32Elements = DenseElementsAttr::get(f32TensorTy, f32Attrs);
        f32Const = rewriter.create<arith::ConstantOp>(loc, f32Elements).getResult();
	  } else if (!elemType.isF32()) {
	    return rewriter.notifyMatchFailure(op, "only f32/f64 constant tensors are handled");
	  } else {
        f32Const = rewriter.create<arith::ConstantOp>(loc, elements).getResult();
	  }

	  Value out = rewriter.create<mlir::posit::FromF32Op>(loc, convertedType, f32Const).getResult();
	  rewriter.replaceOp(op, out);
	  return success();                           
	    
	  }
  unsigned nbits;
  unsigned es;
  unsigned storageBits;
};

} // end anonymous namespace

// 對外提供一個 helper，讓 ONNXToPosit.cpp 呼叫來註冊 patterns。  ONNXAddOpLowering, ONNXConstantOpLowering
void populateONNXToPositConversionPattern(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          MLIRContext *ctx,
                                          unsigned nbits,
                                          unsigned es) {
  patterns.add<ONNXDequantizeLinearOpLowering>(typeConverter, ctx, nbits, es);
  patterns.add<ONNXAddOpLowering>(typeConverter, ctx, nbits, es);
  patterns.add<ONNXSubOpLowering, ONNXMulOpLowering, ONNXDivOpLowering>(
      typeConverter, ctx);
  patterns.add<ONNXReshapeOpLowering, ONNXUnsqueezeOpLowering,
      ONNXReluOpLowering, ONNXClipOpLowering, ONNXMaxPoolSingleOutOpLowering,
      ONNXFlattenOpLowering, ONNXReduceMeanV13OpLowering>(typeConverter, ctx);
  patterns.add<ONNXConvOpLowering, ONNXMatMulOpLowering, ONNXGemmOpLowering>(
      typeConverter, ctx, nbits, es);
}

} // namespace onnx_mlir
