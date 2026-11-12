// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_NODES=G MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s -split-input-file | FileCheck %s

// Backward rules added for Qwen3.5's Gated DeltaNet (decay exp(.), softplus,
// l2-norm sums, causal depthwise Conv1D). Numerics: qwen35_tiny_check.py
// (native vs PyTorch FP64 autograd on a tiny random Qwen3.5).

// Exp: dx = dY * y (the op's own result).
func.func @exp(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %e = "onnx.Exp"(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %w = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%e, %w, %b) {onnx_node_name = "G", transB = 1 : si64} : (tensor<1x2xf32>, tensor<2x2xf32>, none) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @exp
// CHECK: [[Y:%.+]] = "onnx.Exp"(%arg0)
// CHECK: "onnx.Mul"(%{{.+}}, [[Y]])
// CHECK: return

// -----

// Softplus: dx = dY * sigmoid(x).
func.func @softplus(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %s = "onnx.Softplus"(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %w = onnx.Constant dense<1.0> : tensor<2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%s, %w, %b) {onnx_node_name = "G", transB = 1 : si64} : (tensor<1x2xf32>, tensor<2x2xf32>, none) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @softplus
// CHECK: "onnx.Sigmoid"(%arg0)
// CHECK: return

// -----

// ReduceSum over the last axis (keepdims): dY broadcast back, not divided by
// the 2 summed elements (ReduceMean would multiply by 0.5).
func.func @reduce_sum(%arg0: tensor<1x2xf32>) -> tensor<1x1xf32> {
  %ax = onnx.Constant dense<-1> : tensor<1xi64>
  %r = "onnx.Relu"(%arg0) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %s = "onnx.ReduceSum"(%r, %ax) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<1x2xf32>, tensor<1xi64>) -> tensor<1x1xf32>
  %w = onnx.Constant dense<1.0> : tensor<1x1xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%s, %w, %b) {onnx_node_name = "G", transB = 1 : si64} : (tensor<1x1xf32>, tensor<1x1xf32>, none) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @reduce_sum
// CHECK-NOT: dense<5.000000e-01>
// CHECK: "onnx.Expand"
// CHECK-NOT: dense<5.000000e-01>
// CHECK: return

// -----

// Causal depthwise Conv1D ([1, C, T], group C, pads [K-1, K-1]): its input
// gradient is the 2-D rule on [1, C, 1, T].
func.func @conv1d(%arg0: tensor<1x2x4xf32>) -> tensor<1x1xf32> {
  %r = "onnx.Relu"(%arg0) : (tensor<1x2x4xf32>) -> tensor<1x2x4xf32>
  %k = onnx.Constant dense<[[[1.0, 2.0]], [[3.0, 4.0]]]> : tensor<2x1x2xf32>
  %nb = "onnx.NoValue"() {value} : () -> none
  %c = "onnx.Conv"(%r, %k, %nb) {group = 2 : si64, kernel_shape = [2], pads = [1, 1], strides = [1], dilations = [1]} : (tensor<1x2x4xf32>, tensor<2x1x2xf32>, none) -> tensor<1x2x5xf32>
  %sh = onnx.Constant dense<[1, 10]> : tensor<2xi64>
  %f = "onnx.Reshape"(%c, %sh) : (tensor<1x2x5xf32>, tensor<2xi64>) -> tensor<1x10xf32>
  %w = onnx.Constant dense<1.0> : tensor<1x10xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%f, %w, %b) {onnx_node_name = "G", transB = 1 : si64} : (tensor<1x10xf32>, tensor<1x10xf32>, none) -> tensor<1x1xf32>
  return %0 : tensor<1x1xf32>
}
// CHECK-LABEL: func.func @conv1d
// CHECK: "onnx.Conv"({{.*}}) {{.*}} -> tensor<1x2x1x4xf32>
// CHECK: return
