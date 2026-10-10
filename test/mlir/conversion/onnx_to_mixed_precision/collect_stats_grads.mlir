// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.json

// Grads mode: the function gets an "mixedp_onehot" argument shaped like
// result 0, its own backward pass for softmax cross-entropy (seed
// (softmax(z) - onehot) times each row's one-hot sum, so a row without a
// target gets no gradient), and one "mixedp_stats" result holding, per tap,
// sum(g^2) and sum(g) per channel (axis 1 of an NCHW tensor, else the last
// axis), and per weight sum(dW^2) per output channel. Here: a strided, padded
// 3x3 conv (input gradient through zero insertion + a conv with the flipped
// weight; weight gradient as a conv of X with dY), a ReLU6 Clip (strict mask),
// and a Gemm with transB = 1 (output channel axis 0).
func.func @main_graph(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x3xf32> {
  %w = onnx.Constant dense<0.5> : tensor<2x2x3x3xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %w, %b) {onnx_node_name = "C", auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x2x4x4xf32>, tensor<2x2x3x3xf32>, none) -> tensor<1x2x2x2xf32>
  %lo = onnx.Constant dense<0.0> : tensor<f32>
  %hi = onnx.Constant dense<6.0> : tensor<f32>
  %1 = "onnx.Clip"(%0, %lo, %hi) : (tensor<1x2x2x2xf32>, tensor<f32>, tensor<f32>) -> tensor<1x2x2x2xf32>
  %s = onnx.Constant dense<[1, 8]> : tensor<2xi64>
  %2 = "onnx.Reshape"(%1, %s) {allowzero = 0 : si64} : (tensor<1x2x2x2xf32>, tensor<2xi64>) -> tensor<1x8xf32>
  %fw = onnx.Constant dense<0.25> : tensor<3x8xf32>
  %3 = "onnx.Gemm"(%2, %fw, %b) {onnx_node_name = "G", alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<1x8xf32>, tensor<3x8xf32>, none) -> tensor<1x3xf32>
  return %3 : tensor<1x3xf32>
}
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: (%arg0: tensor<1x2x4x4xf32>, %[[OH:[a-z0-9]+]]: tensor<1x3xf32> {onnx.name = "mixedp_onehot"}) -> (tensor<1x3xf32>, tensor<35xf32> {onnx.name = "mixedp_stats"})
// CHECK: %[[Z:[0-9]+]] = "onnx.Gemm"
// CHECK: %[[SM:[0-9]+]] = "onnx.Softmax"(%[[Z]]) <{axis = -1 : si64}>
// CHECK: %[[ROW:[0-9]+]] = "onnx.ReduceSum"(%[[OH]], {{%[0-9]+}}) <{keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x3xf32>, tensor<1xi64>) -> tensor<1x1xf32>
// CHECK: %[[DIFF:[0-9]+]] = "onnx.Sub"(%[[SM]], %[[OH]])
// CHECK: %[[SEED:[0-9]+]] = "onnx.Mul"(%[[DIFF]], %[[ROW]])
// CHECK: "onnx.MatMul"(%[[SEED]]
// CHECK: "onnx.Greater"
// CHECK: "onnx.Less"
// CHECK: "onnx.And"
// CHECK: "onnx.Where"
// CHECK: "onnx.Pad"
// CHECK: "onnx.Conv"
// CHECK: "onnx.Transpose"
// CHECK: "onnx.Conv"
// CHECK: "onnx.ReduceSum"
// CHECK: %[[ST:[0-9]+]] = "onnx.Concat"({{.*}}) <{axis = 0 : si64}> : ({{.*}}) -> tensor<35xf32>
// CHECK: return %[[Z]], %[[ST]] : tensor<1x3xf32>, tensor<35xf32>

// Taps: C.x (2 channels), C.y (2), G.x ([1,8], 8 features), G.y (3) ->
// 2*(2+2+8+3) = 30 values; weights: C (2 output channels), G (3) -> 35.
// LAYOUT: "input_index": 1,
// LAYOUT: "input_name": "mixedp_onehot",
// LAYOUT: "mode": "grads",
// LAYOUT: "num_values": 35,
// LAYOUT: "taps": [
// LAYOUT: "channel_axis": 1,
// LAYOUT-NEXT: "channels": 2,
// LAYOUT-NEXT: "has_grad": true,
// LAYOUT-NEXT: "offset": 0,
// LAYOUT-NEXT: "positions": 16,
// LAYOUT-NEXT: "tap": 0
// LAYOUT: "channels": 8,
// LAYOUT-NEXT: "has_grad": true,
// LAYOUT-NEXT: "offset": 8,
// LAYOUT-NEXT: "positions": 1,
// LAYOUT-NEXT: "tap": 2
// LAYOUT: "weights": [
// LAYOUT: "channel_axis": 0,
// LAYOUT-NEXT: "channels": 2,
// LAYOUT-NEXT: "elements_per_channel": 18,
// LAYOUT-NEXT: "node": "C",
// LAYOUT-NEXT: "offset": 30,
// LAYOUT: "channel_axis": 0,
// LAYOUT-NEXT: "channels": 3,
// LAYOUT-NEXT: "elements_per_channel": 8,
// LAYOUT-NEXT: "node": "G",
// LAYOUT-NEXT: "offset": 32,
