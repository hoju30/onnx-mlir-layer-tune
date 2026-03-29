#include "src/Conversion/PositToKrnl/PositToKrnl.hpp"
#include "src/Conversion/PositToKrnl/TypeConverters.hpp"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositOps.h"

// Krnl dialect
#include "src/Dialect/Krnl/KrnlOps.hpp"


#include "src/Conversion/PositToKrnl/Pattern/Math.cpp"

using namespace mlir;

namespace onnx_mlir {
namespace {

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
                    mlir::KrnlDialect>();
  }

  unsigned nbits = 8;
  unsigned es = 0;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &ctx = getContext();

    PositToKrnlTypeConverter typeConverter(nbits, es, &ctx);

    RewritePatternSet patterns(&ctx);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);
    populateBranchOpInterfaceTypeConversionPattern(patterns, typeConverter);

    populatePositToKrnlConversionPattern(typeConverter, patterns, &ctx);

    ConversionTarget target(ctx);
    target.addLegalDialect<func::FuncDialect, memref::MemRefDialect,
                           arith::ArithDialect, mlir::KrnlDialect>();

    // 型別橋接
    target.addLegalOp<UnrealizedConversionCastOp>();

    // 把 Posit ops 變 illegal，逼它們一定要被轉掉
    // 1/6 新增sub mul div
    target.addIllegalOp<posit::ConstantOp, posit::AddOp, posit::SubOp,
                       posit::MulOp, posit::DivOp>();

    // 其他未知 op
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
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