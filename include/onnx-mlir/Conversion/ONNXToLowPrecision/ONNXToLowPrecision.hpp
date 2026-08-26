#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace onnx_mlir {
/// Retype selected ONNX nodes to a native low-precision float type (BF16,
/// F16) via onnx.Cast boundary casts. Node selection and target format come
/// from the LOWP_NODE_FORMATS env var (node_name:format_tag[,...], e.g.
/// "Conv_1:bf16,Gemm_3:f16"). Nodes not named in LOWP_NODE_FORMATS, and ops
/// whose type is not in the supported set, are left untouched.
std::unique_ptr<mlir::Pass> createConvertONNXToLowPrecisionPass();
} // namespace onnx_mlir
