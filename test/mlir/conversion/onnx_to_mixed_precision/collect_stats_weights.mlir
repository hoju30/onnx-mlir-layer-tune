// RUN: env MIXEDP_STATS_MODE=weights MIXEDP_WEIGHT_PER_CHANNEL=1 MIXEDP_STATS_FORMATS=int4,bf16 MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s --check-prefix=IR
// RUN: FileCheck %s < %t.json
// RUN: env -u MIXEDP_WEIGHT_PER_CHANNEL MIXEDP_STATS_MODE=weights MIXEDP_STATS_FORMATS=int4 MIXEDP_STATS_LAYOUT=%t.pt.json onnx-mlir-opt --mixedp-collect-stats %s > /dev/null
// RUN: FileCheck %s --check-prefix=PERTENSOR < %t.pt.json
// RUN: not env MIXEDP_STATS_MODE=weights MIXEDP_STATS_FORMATS=int4,fp16 MIXEDP_STATS_LAYOUT=%t.bad.json onnx-mlir-opt --mixedp-collect-stats %s 2>&1 | FileCheck %s --check-prefix=BADFMT

// Weights mode: compile time only, the IR is left untouched. Per named
// weight-bearing node with a constant f32 weight, the squared error of the
// pass's own weight fake-quant summed per output channel (Gemm transB=1 ->
// axis 0). Expected values computed independently in numpy with the same
// float32 arithmetic: per-channel int4 scales 1/7 and 2/7 give
// [(0.3 - 2/7)^2, 0]; one per-tensor int4 scale 2/7 also rounds 1.0 to 6/7;
// bf16 only changes 0.3 (-> 0.30078125). The MatMul's right operand is a
// function argument, not a compile-time weight, so it is listed as skipped.
func.func @main_graph(%arg0: tensor<1x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<1x3xf32> {
  %w = onnx.Constant dense<[[1.0, 0.3], [2.0, -2.0]]> : tensor<2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %w, %b) {onnx_node_name = "G", alpha = 1.0 : f32, beta = 1.0 : f32, transA = 0 : si64, transB = 1 : si64} : (tensor<1x2xf32>, tensor<2x2xf32>, none) -> tensor<1x2xf32>
  %1 = "onnx.MatMul"(%0, %arg1) {onnx_node_name = "M"} : (tensor<1x2xf32>, tensor<2x3xf32>) -> tensor<1x3xf32>
  return %1 : tensor<1x3xf32>
}
// IR-LABEL: func.func @main_graph
// IR-SAME: -> tensor<1x3xf32>
// IR: onnx.Constant dense<{{\[\[}}1.000000e+00, 3.000000e-01], [2.000000e+00, -2.000000e+00]]>
// IR-NOT: mixedp_stats
// IR: return

// CHECK: "formats": [
// CHECK-NEXT: "int4",
// CHECK-NEXT: "bf16"
// CHECK: "mode": "weights"
// CHECK: "nodes": [
// CHECK: "channel_axis": 0,
// CHECK-NEXT: "channels": 2,
// CHECK-NEXT: "elements": 4,
// CHECK-NEXT: "err2": {
// CHECK-NEXT: "bf16": [
// CHECK-NEXT: 6.1033293619{{[0-9]*}}e-07,
// CHECK-NEXT: 0
// CHECK-NEXT: ],
// CHECK-NEXT: "int4": [
// CHECK-NEXT: 0.00020408160832{{[0-9]*}},
// CHECK-NEXT: 0
// CHECK-NEXT: ]
// CHECK-NEXT: },
// CHECK-NEXT: "node": "G",
// CHECK-NEXT: "op": "Gemm"
// CHECK: "skipped": [
// CHECK-NEXT: "M"
// CHECK-NEXT: ],
// CHECK-NEXT: "weight_per_channel": true

// PERTENSOR: "int4": [
// PERTENSOR-NEXT: 0.0206122254108{{[0-9]*}},
// PERTENSOR-NEXT: 0
// PERTENSOR: "weight_per_channel": false

// BADFMT: MIXEDP_STATS_FORMATS has an unrecognized format tag 'fp16'
