// RUN: echo '{"R": {"x": [-1.0, 3.0], "y": [0.0, 3.0]}}' > %t.ranges.json
// RUN: env MIXEDP_STATS_MODE=errors MIXEDP_STATS_FORMATS=int8,p8e1,f16 MIXEDP_STATS_RANGES=%t.ranges.json MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.json
// RUN: not env MIXEDP_STATS_MODE=errors MIXEDP_STATS_FORMATS=int8 MIXEDP_STATS_LAYOUT=%t.bad.json onnx-mlir-opt --mixedp-collect-stats %s 2>&1 | FileCheck %s --check-prefix=NORANGES

// Errors mode: per point and format, d = Q(t) - t with Q the pass's own
// activation QDQ, inserted right after t is defined, reduced to sum(d) and
// sum(d^2) over every axis but the channel axis (the last one for this rank-3
// tensor: 3 channels). int8 takes (scale, zero point) from the range in
// MIXEDP_STATS_RANGES exactly as eval_random_mixed_precision.int_qparams
// computes them: [-1, 3] -> (4/255, -64), [0, 3] -> (3/255, -128); the scale is
// a splat along the last axis (see collect_stats_errors_matmul.mlir). posit is one
// posit.round_f32, f16 a cast pair. 2 points x 3 formats x 2 x 3 = 36 values.
func.func @main_graph(%arg0: tensor<1x2x3xf32>) -> tensor<1x2x3xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R"} : (tensor<1x2x3xf32>) -> tensor<1x2x3xf32>
  return %0 : tensor<1x2x3xf32>
}
// CHECK-LABEL: func.func @main_graph
// CHECK-SAME: -> (tensor<1x2x3xf32>, tensor<36xf32> {onnx.name = "mixedp_stats"})
// CHECK-DAG: %[[S:[0-9]+]] = onnx.Constant dense<{{0\.0156862[0-9]*|1\.568627[0-9]*e-02}}> : tensor<3xf32>
// CHECK-DAG: %[[Z:[0-9]+]] = onnx.Constant dense<-6.400000e+01> : tensor<f32>
// CHECK: %[[DIV:[0-9]+]] = "onnx.Div"(%arg0, %[[S]])
// CHECK: "onnx.Add"(%[[DIV]], %[[Z]])
// CHECK: "onnx.Round"
// CHECK: "onnx.Clip"
// CHECK: %[[Q:[0-9]+]] = "onnx.Mul"
// CHECK: %[[D:[0-9]+]] = "onnx.Sub"(%[[Q]], %arg0)
// CHECK: "onnx.ReduceSum"(%[[D]],
// CHECK: %[[D2:[0-9]+]] = "onnx.Mul"(%[[D]], %[[D]])
// CHECK: "onnx.ReduceSum"(%[[D2]],
// CHECK: %[[P:[0-9]+]] = posit.round_f32 %arg0 {es = 1 : i64, nbits = 8 : i64}
// CHECK: "onnx.Sub"(%[[P]], %arg0)
// CHECK: %[[H:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = f16}>
// CHECK: %[[HF:[0-9]+]] = "onnx.Cast"(%[[H]]) <{saturate = 1 : si64, to = f32}>
// CHECK: "onnx.Sub"(%[[HF]], %arg0)
// CHECK: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "R"}
// CHECK-DAG: onnx.Constant dense<{{0\.011764[0-9]*|1\.176470[0-9]*e-02}}> : tensor<3xf32>
// CHECK-DAG: onnx.Constant dense<-1.280000e+02> : tensor<f32>
// CHECK: "onnx.Div"(%[[R]],
// CHECK: posit.round_f32 %[[R]]
// CHECK: "onnx.Cast"(%[[R]])
// CHECK: %[[ST:[0-9]+]] = "onnx.Concat"({{.*}}) -> tensor<36xf32>
// CHECK: return %[[R]], %[[ST]] : tensor<1x2x3xf32>, tensor<36xf32>

// LAYOUT: "error_fields": [
// LAYOUT-NEXT: "dsum",
// LAYOUT-NEXT: "d2"
// LAYOUT: "formats": [
// LAYOUT-NEXT: "int8",
// LAYOUT-NEXT: "p8e1",
// LAYOUT-NEXT: "f16"
// LAYOUT: "mode": "errors",
// LAYOUT: "num_values": 36,
// LAYOUT: "taps": [
// LAYOUT: "channel_axis": 2,
// LAYOUT-NEXT: "channels": 3,
// LAYOUT-NEXT: "errors": {
// LAYOUT-NEXT: "f16": 12,
// LAYOUT-NEXT: "int8": 0,
// LAYOUT-NEXT: "p8e1": 6
// LAYOUT-NEXT: },
// LAYOUT-NEXT: "int_params": {
// LAYOUT-NEXT: "int8": [
// LAYOUT-NEXT: 0.01568627450980{{[0-9]*}},
// LAYOUT-NEXT: -64
// LAYOUT: "positions": 2,
// LAYOUT-NEXT: "tap": 0
// LAYOUT: "errors": {
// LAYOUT-NEXT: "f16": 30,
// LAYOUT-NEXT: "int8": 18,
// LAYOUT-NEXT: "p8e1": 24
// LAYOUT: "int8": [
// LAYOUT-NEXT: 0.01176470588235{{[0-9]*}},
// LAYOUT-NEXT: -128

// NORANGES: int activation formats need MIXEDP_STATS_RANGES
