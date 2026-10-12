// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_LAYOUT=%t.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s
// RUN: FileCheck %s --check-prefix=LAYOUT < %t.json

// RoPE's rotate_half (two Slices, Neg, Concat) and GQA's repeat_kv (Expand),
// as in Qwen3: a Slice's dY is padded with zeros back to its input (pads =
// start and dim - end on the sliced axis; an INT64_MAX end means the full
// axis), an Expand's dY is summed over the expanded axis.
func.func @main_graph(%arg0: tensor<1x4xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<0.5> : tensor<4x4xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "M"} : (tensor<1x4xf32>, tensor<4x4xf32>) -> tensor<1x4xf32>
  %s0 = onnx.Constant dense<0> : tensor<1xi64>
  %s2 = onnx.Constant dense<2> : tensor<1xi64>
  %end = onnx.Constant dense<9223372036854775807> : tensor<1xi64>
  %ax = onnx.Constant dense<1> : tensor<1xi64>
  %one = onnx.Constant dense<1> : tensor<1xi64>
  %lo = "onnx.Slice"(%0, %s0, %s2, %ax, %one) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2xf32>
  %hi = "onnx.Slice"(%0, %s2, %end, %ax, %one) : (tensor<1x4xf32>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1x2xf32>
  %n = "onnx.Neg"(%hi) : (tensor<1x2xf32>) -> tensor<1x2xf32>
  %r = "onnx.Concat"(%n, %lo) <{axis = 1 : si64}> : (tensor<1x2xf32>, tensor<1x2xf32>) -> tensor<1x4xf32>
  %u = onnx.Constant dense<[1, 1, 4]> : tensor<3xi64>
  %1 = "onnx.Reshape"(%r, %u) {allowzero = 0 : si64} : (tensor<1x4xf32>, tensor<3xi64>) -> tensor<1x1x4xf32>
  %e = onnx.Constant dense<[1, 3, 4]> : tensor<3xi64>
  %2 = "onnx.Expand"(%1, %e) : (tensor<1x1x4xf32>, tensor<3xi64>) -> tensor<1x3x4xf32>
  %f = onnx.Constant dense<[1, 12]> : tensor<2xi64>
  %3 = "onnx.Reshape"(%2, %f) {allowzero = 0 : si64} : (tensor<1x3x4xf32>, tensor<2xi64>) -> tensor<1x12xf32>
  %w2 = onnx.Constant dense<0.25> : tensor<12x2xf32>
  %4 = "onnx.MatMul"(%3, %w2) {onnx_node_name = "N"} : (tensor<1x12xf32>, tensor<12x2xf32>) -> tensor<1x2xf32>
  return %4 : tensor<1x2xf32>
}
// Expand [1,1,4] -> [1,3,4]: dY summed over axis 1.
// CHECK: "onnx.ReduceSum"({{%[0-9]+}}, {{%[0-9]+}}) <{keepdims = 1 : si64, noop_with_empty_axes = 0 : si64}> : (tensor<1x3x4xf32>, tensor<1xi64>) -> tensor<1x1x4xf32>
// Slice [:, 2:end] of [1,4]: zeros before it on axis 1.
// CHECK: %[[PH:[0-9]+]] = onnx.Constant dense<[0, 2, 0, 0]> : tensor<4xi64>
// CHECK: "onnx.Pad"({{%[0-9]+}}, %[[PH]], {{%[0-9]+}}, {{%[0-9]+}}) {{.*}} : (tensor<1x2xf32>, tensor<4xi64>, {{.*}}) -> tensor<1x4xf32>
// Slice [:, 0:2]: zeros after it.
// CHECK: %[[PL:[0-9]+]] = onnx.Constant dense<[0, 0, 0, 2]> : tensor<4xi64>
// CHECK: "onnx.Pad"({{%[0-9]+}}, %[[PL]], {{%[0-9]+}}, {{%[0-9]+}}) {{.*}} : (tensor<1x2xf32>, tensor<4xi64>, {{.*}}) -> tensor<1x4xf32>

// LAYOUT: "taps": [
// LAYOUT-NOT: "has_grad": false
// LAYOUT: "weights": [
