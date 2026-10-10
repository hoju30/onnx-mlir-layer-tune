// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.json

// A bias computed at run time gets its gradient: Conv dB = dY summed over
// batch and space, Gemm dC = beta * dY summed back over the broadcast axes.
// Both biases here are Relu outputs, so the Relu points must have gradients
// (they used to be reported without one, i.e. as zero sensitivity).
func.func @main_graph(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<3xf32>) -> tensor<1x3xf32> {
  %w = onnx.Constant dense<0.5> : tensor<2x1x2x2xf32>
  %b = "onnx.Relu"(%arg1) {onnx_node_name = "RB"} : (tensor<2xf32>) -> tensor<2xf32>
  %0 = "onnx.Conv"(%arg0, %w, %b) {onnx_node_name = "C", auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x1x3x3xf32>, tensor<2x1x2x2xf32>, tensor<2xf32>) -> tensor<1x2x2x2xf32>
  %s = onnx.Constant dense<[1, 8]> : tensor<2xi64>
  %1 = "onnx.Reshape"(%0, %s) {allowzero = 0 : si64} : (tensor<1x2x2x2xf32>, tensor<2xi64>) -> tensor<1x8xf32>
  %fw = onnx.Constant dense<0.25> : tensor<3x8xf32>
  %c = "onnx.Relu"(%arg2) {onnx_node_name = "RC"} : (tensor<3xf32>) -> tensor<3xf32>
  %2 = "onnx.Gemm"(%1, %fw, %c) {onnx_node_name = "G", alpha = 1.0 : f32, beta = 2.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<1x8xf32>, tensor<3x8xf32>, tensor<3xf32>) -> tensor<1x3xf32>
  return %2 : tensor<1x3xf32>
}
// Gemm C [3] from dY [1,3]: beta * dY, summed over the broadcast axis 0.
// CHECK: %[[BETA:[0-9]+]] = onnx.Constant dense<2.000000e+00> : tensor<f32>
// CHECK: %[[GC:[0-9]+]] = "onnx.Mul"({{%[0-9]+}}, %[[BETA]])
// CHECK: "onnx.ReduceSum"(%[[GC]], {{%[0-9]+}}) <{keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x3xf32>
// Conv bias [2] from dY [1,2,2,2]: summed over axes 0, 2, 3.
// CHECK: "onnx.ReduceSum"({{%[0-9]+}}, {{%[0-9]+}}) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x2x2x2xf32>, tensor<3xi64>) -> tensor<2xf32>

// LAYOUT: "points": [
// LAYOUT: "node": "RB",
// LAYOUT: "taps": [
// LAYOUT-NOT: "has_grad": false
// LAYOUT: "weights": [
