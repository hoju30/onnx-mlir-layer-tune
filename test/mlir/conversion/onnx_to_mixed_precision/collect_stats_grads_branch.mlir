// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_INPUT_GRAD=branch MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.json
// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_LAYOUT=%t.total.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s --check-prefix=TOTAL
// RUN: FileCheck %s --check-prefix=TOTAL-LAYOUT < %t.total.json

// Two weight nodes sharing their input (as Qwen3's gate and up projections).
// The tap of %arg0 holds the total gradient (both branches); with
// MIXEDP_STATS_INPUT_GRAD=branch each weight node's entry also holds sum(g^2)
// and sum(g) per channel of its own input-edge gradient (dY W^T), the
// sensitivity of the QDQ the mixed-precision pass inserts on that edge only.
// Taps: x (4 channels), A.y (3), B.y (3) -> 20 values; weights A, B: 3 output
// channels each -> 26; branch: 2 x 4 per node -> 42. Without the variable the
// layout is unchanged (26 values, no branch fields).
func.func @main_graph(%arg0: tensor<1x4xf32>) -> tensor<1x3xf32> {
  %wa = onnx.Constant dense<0.5> : tensor<4x3xf32>
  %wb = onnx.Constant dense<0.25> : tensor<4x3xf32>
  %a = "onnx.MatMul"(%arg0, %wa) {onnx_node_name = "A"} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
  %b = "onnx.MatMul"(%arg0, %wb) {onnx_node_name = "B"} : (tensor<1x4xf32>, tensor<4x3xf32>) -> tensor<1x3xf32>
  %0 = "onnx.Add"(%a, %b) : (tensor<1x3xf32>, tensor<1x3xf32>) -> tensor<1x3xf32>
  return %0 : tensor<1x3xf32>
}
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: -> (tensor<1x3xf32>, tensor<42xf32> {onnx.name = "mixedp_stats"})
// TOTAL-LABEL: func.func @main_graph
// TOTAL-SAME: -> (tensor<1x3xf32>, tensor<26xf32> {onnx.name = "mixedp_stats"})

// LAYOUT: "num_values": 42,
// LAYOUT: "weights": [
// LAYOUT: "channels": 3,
// LAYOUT-NEXT: "elements_per_channel": 4,
// LAYOUT-NEXT: "node": "A",
// LAYOUT-NEXT: "offset": 20,
// LAYOUT-NEXT: "op": "MatMul",
// LAYOUT-NEXT: "x_branch_offset": 23,
// LAYOUT-NEXT: "x_channel_axis": 1,
// LAYOUT-NEXT: "x_channels": 4,
// LAYOUT-NEXT: "x_has_grad": true
// LAYOUT: "node": "B",
// LAYOUT-NEXT: "offset": 31,
// LAYOUT-NEXT: "op": "MatMul",
// LAYOUT-NEXT: "x_branch_offset": 34,

// TOTAL-LAYOUT: "num_values": 26,
// TOTAL-LAYOUT-NOT: x_branch_offset
