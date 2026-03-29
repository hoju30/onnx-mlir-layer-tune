#include "onnx-mlir/Conversion/ONNXToPosit/ONNXToPosit.hpp"
#include "TypeConverters.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

// 11/26 pipeline
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

// 改成TypeConverters.hpp
#include "src/Conversion/ONNXToPosit/TypeConverters.hpp"

// 讓mlir::posit可以被讀懂
#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositPasses.h"
#include "src/Dialect/Posit/PositOps.h"

// ONNX
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "src/Conversion/ONNXToPosit/Pattern/Math.cpp"
#include "src/Pass/Passes.hpp"
// pass需修改 Pass/Passes.hpp 但盡量別動到 要重編  + Tools/onnx-mlir-opt/RegisterPasses.cpp


using namespace onnx_mlir;
using namespace mlir;
using namespace posit;



namespace onnx_mlir {

namespace {

struct ConvertONNXToPositPass
    : public PassWrapper<ConvertONNXToPositPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToPositPass)

  ConvertONNXToPositPass() = default;
  ConvertONNXToPositPass(unsigned nbits, unsigned es)
      : nbits(nbits), es(es) {}

  StringRef getArgument() const final { return "convert-onnx-to-posit"; }

  StringRef getDescription() const final {
    return "Lower ONNX ops and types to the Posit dialect.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<posit::PositDialect, arith::ArithDialect,
                    tensor::TensorDialect, func::FuncDialect>();
  }

  unsigned nbits = 8;
  unsigned es = 0;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &ctx = getContext();

    // 建立 PositTypeConverter 這裡固定 8,0，之後可以改
    PositTypeConverter typeConverter(nbits, es, &ctx);

    // 建 pattern set，加上 func/call/return/branch 的 type conversion
    RewritePatternSet patterns(&ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    // 加上ONNX -> Posit patterns
    populateONNXToPositConversionPattern(typeConverter, patterns, &ctx);

    // 設定 ConversionTarget
    ConversionTarget target(ctx);
    // 最終允許存在的 dialect
    target.addLegalDialect<posit::PositDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect>();
    // UnrealizedConversionCast 是 legal（用來 bridge type conversion）
    target.addLegalOp<UnrealizedConversionCastOp>();

    // 把已覆蓋的 ONNX ops 設為 illegal，確保一定要被轉換
    // 12/3 先測constant就好  ONNXAddOp, ONNXConstantOp
    // 1/6 新增 sub div mul 測試
    target.addIllegalOp<ONNXAddOp, ONNXSubOp, ONNXMulOp, ONNXDivOp,
                        ONNXConstantOp>();

    // 其他 op 若其型別對 TypeConverter 來說是 legal，就暫時視為 legal
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    // 套用轉換
    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  
};

} // namespace

// 對外函式 如果想在別處 new 這個 pass
std::unique_ptr<Pass> createConvertONNXToPositPass(unsigned nbits,
                                                   unsigned es) {
  return std::make_unique<ConvertONNXToPositPass>(nbits, es);
}
// 第二種宣告 會自動輸入（8,0）
std::unique_ptr<mlir::Pass> createConvertONNXToPositPass() {
  return createConvertONNXToPositPass(8, 0);
}

} // namespace onnx_mlir
// 靜態註冊
//static PassRegistration<ConvertONNXToPositPass> pass;
