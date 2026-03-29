#include "onnx-mlir/Conversion/ONNXToPosit/ONNXToPosit.hpp"
#include "TypeConverters.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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

#include "mlir/IR/BuiltinAttributes.h"

#include "src/Conversion/ONNXToPosit/Pattern/Math.cpp"
#include "src/Pass/Passes.hpp"

using namespace onnx_mlir;
using namespace mlir;
using namespace posit;

namespace onnx_mlir {
namespace {

struct ConvertONNXToPositPass
    : public PassWrapper<ConvertONNXToPositPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToPositPass)

  ConvertONNXToPositPass() = default;
  ConvertONNXToPositPass(unsigned nbits, unsigned es) : nbits(nbits), es(es) {}

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

    const unsigned localNbits = nbits;
    const unsigned localEs = es;
    PositTypeConverter typeConverter(localNbits, localEs, &ctx);

    RewritePatternSet patterns(&ctx);

    // [MOD][2026-01-14] 不做 func signature type-conversion
    // 讓 entry function 仍然是 f32 I/O，靠 posit.from_f32 / posit.to_f32 bridging
    // [EXTEND] Pass (nbits, es) through so pattern code no longer hardcodes p8e0.
    populateONNXToPositConversionPattern(
        typeConverter, patterns, &ctx, localNbits, localEs);

    ConversionTarget target(ctx);

    target.addLegalDialect<posit::PositDialect, arith::ArithDialect,
                           tensor::TensorDialect, func::FuncDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    // Keep ONNX graph terminators/metadata ops legal.
    target.addLegalOp<ONNXReturnOp, ONNXEntryPointOp, ONNXNoneOp,
                      ONNXQuantizeLinearOp, ONNXConstantOp>();

    // Force all DequantizeLinear ops through ONNX->Posit rewrite so we can
    // avoid the ONNX QDQ int8 lowering path entirely.
    target.addIllegalOp<ONNXDequantizeLinearOp>();
    // Keep the quantized subgraph fully on posit path to avoid repeated
    // f32<->posit bouncing at mixed ONNX/posit boundaries.
    target.addIllegalOp<ONNXAddOp, ONNXSubOp, ONNXMulOp, ONNXDivOp,
                        ONNXReshapeOp, ONNXUnsqueezeOp, ONNXReluOp,
                        ONNXClipOp, ONNXMaxPoolSingleOutOp, ONNXFlattenOp,
                        ONNXConvOp, ONNXMatMulOp, ONNXGemmOp,
                        ONNXReduceMeanV13Op>();

    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return true; });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // QDQ rewrite removes DequantizeLinear; QuantizeLinear becomes dead in the
    // common Q->DQ pattern. Erase dead Quantize ops so they do not leak to
    // later lowering stages.
    bool changed = true;
    while (changed) {
      changed = false;
      SmallVector<ONNXQuantizeLinearOp, 8> deadQuantOps;
      module.walk([&](ONNXQuantizeLinearOp qop) {
        if (qop.getResult().use_empty())
          deadQuantOps.push_back(qop);
      });
      for (ONNXQuantizeLinearOp qop : deadQuantOps) {
        qop.erase();
        changed = true;
      }

      SmallVector<ONNXQLinearMatMulOp, 4> deadQLinearMatMulOps;
      module.walk([&](ONNXQLinearMatMulOp qmm) {
        if (qmm.getResult().use_empty())
          deadQLinearMatMulOps.push_back(qmm);
      });
      for (ONNXQLinearMatMulOp qmm : deadQLinearMatMulOps) {
        qmm.erase();
        changed = true;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> createConvertONNXToPositPass(unsigned nbits, unsigned es) {
  return std::make_unique<ConvertONNXToPositPass>(nbits, es);
}
std::unique_ptr<mlir::Pass> createConvertONNXToPositPass() {
  return createConvertONNXToPositPass(8, 0);
}

} // namespace onnx_mlir
