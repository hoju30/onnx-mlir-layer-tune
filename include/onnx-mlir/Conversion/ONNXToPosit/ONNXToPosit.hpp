#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
class RewritePatternSet;
class MLIRContext;
class TypeConverter;
} // namespace mlir

namespace onnx_mlir {
/// 建立 pass 
std::unique_ptr<mlir::Pass> createConvertONNXToPositPass(
    unsigned positES, unsigned positFS);

/// 供外部測試或其他 pass 呼叫時，把轉換規則注入到 patterns。
void populateONNXToPositPatterns(mlir::RewritePatternSet &patterns,
                                 mlir::TypeConverter &typeConverter,
                                 mlir::MLIRContext &ctx);
} // namespace onnx_mlir
