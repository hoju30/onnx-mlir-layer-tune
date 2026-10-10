// RUN: env MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.json
// RUN: not env MIXEDP_STATS_LAYOUT=%t.bad.json MIXEDP_STATS_MODE=bogus onnx-mlir-opt --mixedp-collect-stats %s 2>&1 | FileCheck %s --check-prefix=BADMODE
// RUN: env -u MIXEDP_STATS_LAYOUT not onnx-mlir-opt --mixedp-collect-stats %s 2>&1 | FileCheck %s --check-prefix=NOLAYOUT

// Every named supported node gets a tap on each activation operand (the
// weight excluded) and on its result; a value shared by several points
// (%arg0: C.x and A.b, %0: C.y and R.x, %1: R.y and A.a) is tapped once,
// right after it is defined. The unnamed Relu is not a quantization point.
// The taps are concatenated into one extra result, 2 values (min, max) per tap.
func.func @main_graph(%arg0: tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32> {
  %w = onnx.Constant dense<[[[[1.0]], [[2.0]]], [[[3.0]], [[4.0]]]]> : tensor<2x2x1x1xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %w, %b) {onnx_node_name = "C", auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1]} : (tensor<1x2x2x2xf32>, tensor<2x2x1x1xf32>, none) -> tensor<1x2x2x2xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "R"} : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %2 = "onnx.Add"(%1, %arg0) {onnx_node_name = "A"} : (tensor<1x2x2x2xf32>, tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  %3 = "onnx.Relu"(%2) : (tensor<1x2x2x2xf32>) -> tensor<1x2x2x2xf32>
  return %3 : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: -> (tensor<1x2x2x2xf32>, tensor<8xf32> {onnx.name = "mixedp_stats"})
// CHECK: %[[F0:[0-9]+]] = "onnx.Reshape"(%arg0,
// CHECK: %[[MIN0:[0-9]+]] = "onnx.ReduceMin"(%[[F0]],
// CHECK: %[[MAX0:[0-9]+]] = "onnx.ReduceMax"(%[[F0]],
// CHECK: %[[T0:[0-9]+]] = "onnx.Concat"(%[[MIN0]], %[[MAX0]]) <{axis = 0 : si64}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<2xf32>
// CHECK: %[[CONV:[0-9]+]] = "onnx.Conv"(%arg0,
// CHECK: %[[F1:[0-9]+]] = "onnx.Reshape"(%[[CONV]],
// CHECK: %[[T1:[0-9]+]] = "onnx.Concat"
// CHECK: %[[RELU:[0-9]+]] = "onnx.Relu"(%[[CONV]]) {onnx_node_name = "R"}
// CHECK: %[[F2:[0-9]+]] = "onnx.Reshape"(%[[RELU]],
// CHECK: %[[T2:[0-9]+]] = "onnx.Concat"
// CHECK: %[[ADD:[0-9]+]] = "onnx.Add"(%[[RELU]], %arg0) {onnx_node_name = "A"}
// CHECK: %[[F3:[0-9]+]] = "onnx.Reshape"(%[[ADD]],
// CHECK: %[[T3:[0-9]+]] = "onnx.Concat"
// CHECK: %[[OUT:[0-9]+]] = "onnx.Relu"(%[[ADD]]) :
// CHECK: %[[STATS:[0-9]+]] = "onnx.Concat"(%[[T0]], %[[T1]], %[[T2]], %[[T3]]) <{axis = 0 : si64}> : (tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<8xf32>
// CHECK: return %[[OUT]], %[[STATS]] : tensor<1x2x2x2xf32>, tensor<8xf32>

// LAYOUT: "mode": "ranges"
// LAYOUT: "num_taps": 4
// LAYOUT: "points": [
// LAYOUT: "node": "C",
// LAYOUT-NEXT: "op": "Conv",
// LAYOUT-NEXT: "slot": "x",
// LAYOUT-NEXT: "tap": 0
// LAYOUT: "node": "C",
// LAYOUT-NEXT: "op": "Conv",
// LAYOUT-NEXT: "slot": "y",
// LAYOUT-NEXT: "tap": 1
// LAYOUT: "node": "R",
// LAYOUT-NEXT: "op": "Relu",
// LAYOUT-NEXT: "slot": "x",
// LAYOUT-NEXT: "tap": 1
// LAYOUT: "node": "R",
// LAYOUT-NEXT: "op": "Relu",
// LAYOUT-NEXT: "slot": "y",
// LAYOUT-NEXT: "tap": 2
// LAYOUT: "node": "A",
// LAYOUT-NEXT: "op": "Add",
// LAYOUT-NEXT: "slot": "a",
// LAYOUT-NEXT: "tap": 2
// LAYOUT: "node": "A",
// LAYOUT-NEXT: "op": "Add",
// LAYOUT-NEXT: "slot": "b",
// LAYOUT-NEXT: "tap": 0
// LAYOUT: "node": "A",
// LAYOUT-NEXT: "op": "Add",
// LAYOUT-NEXT: "slot": "y",
// LAYOUT-NEXT: "tap": 3
// LAYOUT: "result_name": "mixedp_stats"
// LAYOUT: "values_per_tap": 2

// BADMODE: MIXEDP_STATS_MODE='bogus' is not supported
// NOLAYOUT: --mixedp-collect-stats needs MIXEDP_STATS_LAYOUT
