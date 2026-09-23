#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace onnx_mlir {
/// Retype selected ONNX nodes so that weight and activation independently
/// simulate storage in different low-precision formats via QDQ
/// (quantize-then-immediately-dequantize) round trips -- the real
/// arithmetic always stays in f32; only the *storage* of the weight and/or
/// activation values is simulated as if they had been rounded through the
/// target format. Node selection and the per-operand-role target formats
/// come from the MIXEDP_NODE_FORMATS env var
/// (node_name:w=<weight_fmt>:a=<activation_fmt>[:<params>],..., e.g.
/// "Conv_1:w=int4:a=p8e1,Gemm_3:w=p6e1:a=int8:0.05:0:0.1:0"). This pass is
/// independent of, and does not modify the behavior of,
/// --convert-onnx-to-lowprecision (LOWP_NODE_FORMATS) or
/// --convert-onnx-to-posit (POSIT_NODE_FORMATS); a single model may mix all
/// three on different nodes, each pass only touching the nodes named in its
/// own env var. See docs/MixedPrecisionFormats.md.
std::unique_ptr<mlir::Pass> createConvertONNXToMixedPrecisionPass();
} // namespace onnx_mlir
