// RUN: env MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s 2> %t.err | FileCheck %s
// RUN: FileCheck %s --check-prefix=WARN < %t.err
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.json

// A supported node inside an If branch is not instrumented: its result
// cannot reach the function's return without being yielded out of the If
// (and a Loop body's value exists once per iteration). It is reported, and
// the IR stays valid (onnx-mlir-opt verifies it after the pass; tapping the
// branch-local value used to fail with "operand #2 does not dominate this
// use"). The top-level node is tapped as usual.
func.func @main_graph(%arg0: tensor<i1>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg1) {onnx_node_name = "TOP"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %1 = "onnx.If"(%arg0) ({
    %2 = "onnx.Relu"(%0) {onnx_node_name = "IN_THEN"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
    onnx.Yield %2 : tensor<1x4xf32>
  }, {
    onnx.Yield %0 : tensor<1x4xf32>
  }) : (tensor<i1>) -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}
// WARN: warning: --mixedp-collect-stats: 1 supported node(s) inside nested regions (If/Loop/Scan bodies) are not instrumented
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: -> (tensor<1x4xf32>, tensor<4xf32> {onnx.name = "mixedp_stats"})
// CHECK: "onnx.Relu"(%arg1) {onnx_node_name = "TOP"}
// CHECK: "onnx.If"
// CHECK: "onnx.Relu"({{%[0-9]+}}) {onnx_node_name = "IN_THEN"}
// CHECK-NOT: onnx.ReduceMin
// CHECK: onnx.Yield
// CHECK: return {{%[0-9]+}}, {{%[0-9]+}} : tensor<1x4xf32>, tensor<4xf32>

// LAYOUT: "num_taps": 2
// LAYOUT: "points": [
// LAYOUT: "node": "TOP",
// LAYOUT-NEXT: "op": "Relu",
// LAYOUT-NEXT: "slot": "x",
// LAYOUT-NEXT: "tap": 0
// LAYOUT: "node": "TOP",
// LAYOUT-NEXT: "op": "Relu",
// LAYOUT-NEXT: "slot": "y",
// LAYOUT-NEXT: "tap": 1
// LAYOUT-NOT: IN_THEN
// LAYOUT: "skipped_nested": [
// LAYOUT-NEXT: "IN_THEN"
// LAYOUT-NEXT: ]
