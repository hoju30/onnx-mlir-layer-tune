#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Casting.h"
#include "llvm/ADT/APFloat.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/Posit/PositOps.h"
#include "src/Conversion/ONNXToPosit/TypeConverters.hpp"

//12/28

#include <cstdint>

extern "C" {
#include "softposit.h"
}


using namespace mlir;
namespace onnx_mlir {
namespace { // anonymous namespace for internal implementation

// Conversion pattern for ONNXAddOp -> posit.add
static uint64_t dummyEncodeDoubleToPositBits(double v, unsigned nbits) {
  llvm::APFloat apf(v);
  llvm::APInt bits = apf.bitcastToAPInt(); // 例如 64-bit
  if (nbits >= bits.getBitWidth())
    return bits.getZExtValue();
  unsigned shift = bits.getBitWidth() - nbits;
  llvm::APInt shr = bits.lshr(shift);
  return shr.getZExtValue() & ((1ULL << nbits) - 1);
}

static bool encodeDoubleToPositBitsSoftPosit(double v, unsigned nbits,
                                             unsigned es, uint64_t &outBits) {
    if (nbits == 8 && es == 0) {
        posit8_t p = convertDoubleToP8(v);      // double -> posit8_t 
        uint8_t bits = (uint8_t)castUI(p);      // posit8_t -> raw bits (uint8) 
        outBits = (uint64_t)bits;
        return true;
    }
    return false;
                                             }


// ------------- ONNXAddOp -> posit.add -------------

struct ONNXAddOpLowering : public OpConversionPattern<mlir::ONNXAddOp> {
  using OpConversionPattern<mlir::ONNXAddOp>::OpConversionPattern;

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

    auto newAdd =
        rewriter.create<mlir::posit::AddOp>(loc, convertedType, lhs, rhs);
    rewriter.replaceOp(op, newAdd.getResult());
    return success();
  }
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

    auto newOp = rewriter.create<mlir::posit::SubOp>(loc, convertedType, lhs, rhs);
    rewriter.replaceOp(op, newOp.getResult());
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

    auto newOp = rewriter.create<mlir::posit::MulOp>(loc, convertedType, lhs, rhs);
    rewriter.replaceOp(op, newOp.getResult());
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

    auto newOp = rewriter.create<mlir::posit::DivOp>(loc, convertedType, lhs, rhs);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

// 需要新增i64 i32
 
// 1/14 [FIX] DenseFPElementsAttr 與 DenseResourceElementsAttr 讀值方式不同。
// DenseResourceElementsAttrBase<T>::tryGetAsArrayRef() 可能回傳 nullopt（resource 不在 IR/未載入）。
static LogicalResult collectFPValuesAsDouble(ElementsAttr elements,
                                             SmallVectorImpl<double> &out) {
  // 1) 普通 dense<...>
  if (auto denseFP = llvm::dyn_cast<DenseFPElementsAttr>(elements)) {
    out.reserve(denseFP.getNumElements());
    for (APFloat apf : denseFP.getValues<APFloat>())
      out.push_back(apf.convertToDouble());
    return success();
  }

  // 2) dense_resource<...> : tensor<...xf32>
  if (auto resF32 = llvm::dyn_cast<mlir::detail::DenseResourceElementsAttrBase<float>>(elements)) {
    auto arr = resF32.tryGetAsArrayRef();
    if (!arr)
      return failure();
    out.reserve(arr->size());
    for (float v : *arr)
      out.push_back(static_cast<double>(v));
    return success();
  }

  // 3) dense_resource<...> : tensor<...xf64>
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


// ONNXConstantOp -> posit.constant

struct ONNXConstantOpLowering
    : public OpConversionPattern<mlir::ONNXConstantOp> {
  using OpConversionPattern<mlir::ONNXConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(mlir::ONNXConstantOp op,
                              typename OpConversionPattern::OpAdaptor adaptor,
                              ConversionPatternRewriter &rewriter) const override {
  Location loc = op.getLoc();
  MLIRContext *ctx = rewriter.getContext();

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

	  // 2. 只接受浮點 element（f32/f64）
	  auto elemType = elements.getElementType();
	  if (!llvm::isa<FloatType>(elemType))
    return rewriter.notifyMatchFailure(
      op, "only floating-point element type is handled");

  // 3. 目前先只處理 RankedTensor，且 ONNXConstant 的 result 也應該是 tensor
  auto outType = op.getResult().getType();
  auto origOutType = llvm::dyn_cast<RankedTensorType>(outType);
  if (!origOutType)
    return rewriter.notifyMatchFailure(
          op, "only ranked tensor results are handled");

	  ArrayRef<int64_t> shape = origOutType.getShape();
	  int64_t numElems = origOutType.getNumElements();

	  // [FIX] 先把 ElementsAttr 裡的浮點值抽出成 double（同時支援 dense_resource）。
	  SmallVector<double, 4> fpVals;
	  if (failed(collectFPValuesAsDouble(elements, fpVals)))
	    return rewriter.notifyMatchFailure(
	        op, "failed to read float data from constant (dense/dense_resource)");
	  if (numElems != static_cast<int64_t>(fpVals.size()))
	    return rewriter.notifyMatchFailure(
	        op, "tensor shape does not match number of elements in 'value'");

  // 4. 決定 posit 的參數（暫時寫死跟 pass 的預設一樣）
  unsigned nbits = 8;
  unsigned es = 0;

  // DenseElementsAttr 內部要用 integer 元素來存 raw bits（例如 i8）
  auto intElemTy = IntegerType::get(ctx, nbits);
  auto intTensorTy = RankedTensorType::get(shape, intElemTy);

  SmallVector<APInt, 4> positBits;
  positBits.reserve(numElems);

  /*for (APFloat apf : denseFP.getValues<APFloat>()) {
    double d = apf.convertToDouble();
    uint64_t bits = dummyEncodeDoubleToPositBits(d, nbits);
    positBits.emplace_back(nbits, bits);
  }*/
	  //12/28測試使用正式的softposit
	  // [FIX] 改用 fpVals（由 ElementsAttr 抽出），避免 dense_resource 走 DenseFPElementsAttr 造成 crash。
	  for (double d : fpVals) {
	    uint64_t bits = 0;
	    if (!encodeDoubleToPositBitsSoftPosit(d, nbits, es, bits)) {
	      return rewriter.notifyMatchFailure(
	          op, "SoftPosit encoder: unsupported (nbits, es) combination for FP->Posit");
	    }
	    positBits.emplace_back(nbits, bits);
	  }

  DenseElementsAttr positDenseAttr =
      DenseElementsAttr::get(intTensorTy, positBits);

  // 5. 用 TypeConverter 把 result type 轉成 tensor<...x!posit.type<8,0>>
  Type convertedType = getTypeConverter()->convertType(origOutType);
  if (!convertedType)
    return rewriter.notifyMatchFailure(
        op, "failed to convert constant result type to posit tensor");

  // 6. 建立 posit.constant：result type = tensor<...x!posit.type<8,0>>
  //                         value attr = tensor<...xi8> 的 DenseElementsAttr
  auto newConst =
      rewriter.create<mlir::posit::ConstantOp>(loc, convertedType,
                                               positDenseAttr);

  rewriter.replaceOp(op, newConst.getResult());
  return success();                           
    
  }
};

} // end anonymous namespace

// 對外提供一個 helper，讓 ONNXToPosit.cpp 呼叫來註冊 patterns。  ONNXAddOpLowering, ONNXConstantOpLowering
void populateONNXToPositConversionPattern(TypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<ONNXAddOpLowering, ONNXSubOpLowering, ONNXMulOpLowering,
               ONNXDivOpLowering, ONNXConstantOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir
