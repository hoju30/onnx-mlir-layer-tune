// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s -split-input-file | FileCheck %s

// ReduceMean (opset 18) with no axes -- here an empty constant -- is the
// identity when noop_with_empty_axes = 1: its backward passes dY through
// unchanged (it used to broadcast and divide by the element count, halving
// the gradient here).
func.func @noop(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %ax = onnx.Constant dense<> : tensor<0xi64>
  %r = "onnx.Relu"(%arg0) {onnx_node_name = "R"} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %m = "onnx.ReduceMean"(%r, %ax) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<1x2xf32>, tensor<0xi64>) -> tensor<1x2xf32>
  %w = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%m, %w, %b) {onnx_node_name = "G", transB = 1 : si64} : (tensor<1x2xf32>, tensor<2x2xf32>, none) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @noop
// CHECK-NOT: "onnx.Expand"
// CHECK-NOT: dense<5.000000e-01>
// CHECK: return

// -----

// Without noop_with_empty_axes, no axes means every axis: dY is broadcast
// back and divided by the 2 elements.
func.func @reduce_all(%arg0: tensor<1x2xf32>) -> tensor<1x1xf32> {
  %ax = onnx.Constant dense<> : tensor<0xi64>
  %r = "onnx.Relu"(%arg0) {onnx_node_name = "R"} : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %m = "onnx.ReduceMean"(%r, %ax) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2xf32>, tensor<0xi64>) -> tensor<1x1xf32>
  %w = onnx.Constant dense<1.0> : tensor<1x1xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%m, %w, %b) {onnx_node_name = "G", transB = 1 : si64} : (tensor<1x1xf32>, tensor<1x1xf32>, none) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @reduce_all
// CHECK-DAG: "onnx.Expand"
// CHECK-DAG: dense<5.000000e-01>
// CHECK: return
