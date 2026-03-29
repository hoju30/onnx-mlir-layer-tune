#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Casting.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "src/Dialect/Posit/PositOps.h"
#include "src/Conversion/PositToKrnl/TypeConverters.hpp"

// krnl.global
#include "src/Dialect/Krnl/KrnlOps.hpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

    
//  func.func @posit_add_p8e0_4xi8(memref<4xi8>, memref<4xi8>, memref<4xi8>) -> ()


static func::FuncOp getOrCreatePositAddFunc(ModuleOp module, MemRefType memrefTy,
                                            unsigned nbits, unsigned es) {
  OpBuilder b(module.getBodyRegion());
  std::string name = "posit_add_p" + std::to_string(nbits) + "e" +
                     std::to_string(es) + "_";
  // 把 shape 也編進名字，避免不同 shape 卻共用同名造成 signature 不一致
  for (int64_t d : memrefTy.getShape()) {
    name += std::to_string(d) + "x";
  }
  name += "i" + std::to_string(nbits);

  if (auto f = module.lookupSymbol<func::FuncOp>(name))
    return f;

  auto fnType = b.getFunctionType({memrefTy, memrefTy, memrefTy}, {});
  auto funcOp = b.create<func::FuncOp>(module.getLoc(), name, fnType);
  funcOp.setPrivate(); // runtime function
  return funcOp;
}

// 1/6 new submul div 有關相關的posit計算呼叫softposit
static func::FuncOp getOrCreatePositSubFunc(ModuleOp module, MemRefType memrefTy,
                                            unsigned nbits, unsigned es) {
  OpBuilder b(module.getBodyRegion());
  std::string name = "posit_sub_p" + std::to_string(nbits) + "e" +
                     std::to_string(es) + "_";
  for (int64_t d : memrefTy.getShape()) {
    name += std::to_string(d) + "x";
  }
  name += "i" + std::to_string(nbits);

  if (auto f = module.lookupSymbol<func::FuncOp>(name))
    return f;

  auto fnType = b.getFunctionType({memrefTy, memrefTy, memrefTy}, {});
  auto funcOp = b.create<func::FuncOp>(module.getLoc(), name, fnType);
  funcOp.setPrivate(); // runtime function
  return funcOp;
}

static func::FuncOp getOrCreatePositMulFunc(ModuleOp module, MemRefType memrefTy,
                                            unsigned nbits, unsigned es) {
  OpBuilder b(module.getBodyRegion());
  std::string name = "posit_mul_p" + std::to_string(nbits) + "e" +
                     std::to_string(es) + "_";
  for (int64_t d : memrefTy.getShape()) {
    name += std::to_string(d) + "x";
  }
  name += "i" + std::to_string(nbits);

  if (auto f = module.lookupSymbol<func::FuncOp>(name))
    return f;

  auto fnType = b.getFunctionType({memrefTy, memrefTy, memrefTy}, {});
  auto funcOp = b.create<func::FuncOp>(module.getLoc(), name, fnType);
  funcOp.setPrivate(); // runtime function
  return funcOp;
}

static func::FuncOp getOrCreatePositDivFunc(ModuleOp module, MemRefType memrefTy,
                                            unsigned nbits, unsigned es) {
  OpBuilder b(module.getBodyRegion());
  std::string name = "posit_div_p" + std::to_string(nbits) + "e" +
                     std::to_string(es) + "_";
  for (int64_t d : memrefTy.getShape()) {
    name += std::to_string(d) + "x";
  }
  name += "i" + std::to_string(nbits);

  if (auto f = module.lookupSymbol<func::FuncOp>(name))
    return f;

  auto fnType = b.getFunctionType({memrefTy, memrefTy, memrefTy}, {});
  auto funcOp = b.create<func::FuncOp>(module.getLoc(), name, fnType);
  funcOp.setPrivate(); // runtime function
  return funcOp;
}
// ------------------- 1/6 //


// posit.constant -> krnl.global


struct PositConstantOpLowering : public OpConversionPattern<posit::ConstantOp> {
  using OpConversionPattern<posit::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *ctx = rewriter.getContext();

    // 取出 value attribute（你目前 posit.constant 就是 DenseElementsAttr）
    Attribute v = op->getAttr("value");
    auto dense = llvm::dyn_cast_or_null<DenseElementsAttr>(v);
    if (!dense)
      return rewriter.notifyMatchFailure(op, "posit.constant expects DenseElementsAttr 'value'");

    // 轉換 result type：tensor<...x!posit.type> -> memref<...xi8>
    Type converted = getTypeConverter()->convertType(op.getResult().getType());
    auto memrefTy = llvm::dyn_cast_or_null<MemRefType>(converted);
    if (!memrefTy)
      return rewriter.notifyMatchFailure(op, "expected converted type to be MemRefType");

    // 用 krnl.global 存常數（onnx-mlir 也會產生 krnl.global 存常數）
    static int64_t id = 0;
    std::string gname = "posit_const_" + std::to_string(id++);

    OperationState st(loc, "krnl.global");
    st.addTypes(memrefTy);
    st.addAttribute("name", rewriter.getStringAttr(gname));
    st.addAttribute("shape", rewriter.getI64ArrayAttr(memrefTy.getShape()));
    st.addAttribute("value", dense);

    //OperationState st(loc, "krnl.global");
    Operation *global = rewriter.create(st);
    rewriter.replaceOp(op, global->getResult(0));
    return success();
  }
};

// posit.add -> func.call @posit_add_...(lhs, rhs, out)


struct PositAddOpLowering : public OpConversionPattern<posit::AddOp> {
  using OpConversionPattern<posit::AddOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type converted = getTypeConverter()->convertType(op.getType());
    auto outTy = llvm::dyn_cast_or_null<MemRefType>(converted);
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "expected result type to convert to MemRefType");

    // 目前先只支援 static shape
    if (!outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static-shape memref supported for now");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();

    // allocate output
    Value out = rewriter.create<memref::AllocOp>(loc, outTy);

    // 取 converter 參數（nbits/es）
    auto *tc = static_cast<const PositToKrnlTypeConverter *>(getTypeConverter());
    unsigned nbits = tc->getNbits();
    unsigned es = tc->getEs();

    // declare/get runtime function and call it
    ModuleOp module = op->getParentOfType<ModuleOp>();
    func::FuncOp callee = getOrCreatePositAddFunc(module, outTy, nbits, es);

    rewriter.create<func::CallOp>(loc, callee, ValueRange{lhs, rhs, out});

    // replace posit.add result with out memref
    rewriter.replaceOp(op, out);
    return success();
  }
};

// ------ 1/6 //
// posit.sub
struct PositSubOpLowering : public OpConversionPattern<posit::SubOp> {
  using OpConversionPattern<posit::SubOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::SubOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type converted = getTypeConverter()->convertType(op.getType());
    auto outTy = llvm::dyn_cast_or_null<MemRefType>(converted);
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "expected result type to convert to MemRefType");
    if (!outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static-shape memref supported for now");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = rewriter.create<memref::AllocOp>(loc, outTy);

    auto *tc = static_cast<const PositToKrnlTypeConverter *>(getTypeConverter());
    unsigned nbits = tc->getNbits();
    unsigned es = tc->getEs();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    func::FuncOp callee = getOrCreatePositSubFunc(module, outTy, nbits, es);
    rewriter.create<func::CallOp>(loc, callee, ValueRange{lhs, rhs, out});

    rewriter.replaceOp(op, out);
    return success();
  }
};

// posit.mul
struct PositMulOpLowering : public OpConversionPattern<posit::MulOp> {
  using OpConversionPattern<posit::MulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::MulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type converted = getTypeConverter()->convertType(op.getType());
    auto outTy = llvm::dyn_cast_or_null<MemRefType>(converted);
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "expected result type to convert to MemRefType");
    if (!outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static-shape memref supported for now");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = rewriter.create<memref::AllocOp>(loc, outTy);

    auto *tc = static_cast<const PositToKrnlTypeConverter *>(getTypeConverter());
    unsigned nbits = tc->getNbits();
    unsigned es = tc->getEs();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    func::FuncOp callee = getOrCreatePositMulFunc(module, outTy, nbits, es);
    rewriter.create<func::CallOp>(loc, callee, ValueRange{lhs, rhs, out});

    rewriter.replaceOp(op, out);
    return success();
  }
};

//  posit.div
struct PositDivOpLowering : public OpConversionPattern<posit::DivOp> {
  using OpConversionPattern<posit::DivOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(posit::DivOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Type converted = getTypeConverter()->convertType(op.getType());
    auto outTy = llvm::dyn_cast_or_null<MemRefType>(converted);
    if (!outTy)
      return rewriter.notifyMatchFailure(op, "expected result type to convert to MemRefType");
    if (!outTy.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "only static-shape memref supported for now");

    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    Value out = rewriter.create<memref::AllocOp>(loc, outTy);

    auto *tc = static_cast<const PositToKrnlTypeConverter *>(getTypeConverter());
    unsigned nbits = tc->getNbits();
    unsigned es = tc->getEs();

    ModuleOp module = op->getParentOfType<ModuleOp>();
    func::FuncOp callee = getOrCreatePositDivFunc(module, outTy, nbits, es);
    rewriter.create<func::CallOp>(loc, callee, ValueRange{lhs, rhs, out});

    rewriter.replaceOp(op, out);
    return success();
  }
};
// ------------ //

} // namespace

void populatePositToKrnlConversionPattern(TypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  // 新增sub mul div
  patterns.add<PositConstantOpLowering, PositAddOpLowering, PositSubOpLowering,
               PositMulOpLowering, PositDivOpLowering>(typeConverter, ctx);
}

} // namespace onnx_mlir