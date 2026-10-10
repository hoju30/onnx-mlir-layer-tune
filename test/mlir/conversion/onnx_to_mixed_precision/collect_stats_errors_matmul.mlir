// RUN: echo '{"M": {"x": [-1.0, 1.0], "y": [-2.0, 2.0]}}' > %t.ranges.json
// RUN: env MIXEDP_STATS_MODE=errors MIXEDP_STATS_FORMATS=int4,int8 MIXEDP_STATS_RANGES=%t.ranges.json MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats --canonicalize %s | FileCheck %s

// Regression: the int QDQ of a MatMul result divides by its scale; with a
// SCALAR scale, canonicalization (FuseScalarDivMatMulPattern) folds that Div
// into a second MatMul with a scaled copy of the constant weight, once per int
// format -- BERT's statistics graph grew from 0.9 to 4.3 GB that way. The
// statistics graph divides by a splat of the last axis' length instead, which
// the pattern does not match: one MatMul, one weight.
func.func @main_graph(%arg0: tensor<1x4x8xf32>) -> tensor<1x4x6xf32> {
  %w = onnx.Constant dense<0.5> : tensor<8x6xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "M"} : (tensor<1x4x8xf32>, tensor<8x6xf32>) -> tensor<1x4x6xf32>
  %1 = "onnx.Relu"(%0) : (tensor<1x4x6xf32>) -> tensor<1x4x6xf32>
  return %1 : tensor<1x4x6xf32>
}
// CHECK-LABEL: func.func @main_graph
// CHECK-COUNT-1: "onnx.MatMul"
// CHECK-NOT: "onnx.MatMul"
// CHECK: return
