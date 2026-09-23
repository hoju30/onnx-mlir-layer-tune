// RUN: env MIXEDP_NODE_FORMATS="SHARED_INT4:w=int4:a=bf16,SHARED_INT7:w=int7,BRANCH_BF16:a=bf16,CONV_BIAS:w=int4:a=bf16,GEMM_BIAS:w=int4:a=f16" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file 2>&1 | FileCheck %s --implicit-check-not=warning

// A shared weight must be copied for each selected consumer, not rewritten
// in place. The unselected branch must retain the original activation and
// original weight even though two other nodes use different weight formats.
func.func @shared_weight_and_activation(%arg0: tensor<1x2xf32>) -> (tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>) {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %a = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "SHARED_INT4"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  %b = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "SHARED_INT7"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  %c = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "SHARED_UNSELECTED"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %a, %b, %c : tensor<1x2xf32>, tensor<1x2xf32>, tensor<1x2xf32>
}
// CHECK-LABEL: func.func @shared_weight_and_activation
// CHECK: %[[ORIG:[0-9]+]] = onnx.Constant dense<{{.*}}1.000000e-01, -2.000000e-01{{.*}}4.000000e-01, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: %[[W4:[0-9]+]] = onnx.Constant dense<{{.*}}0.114285715, -0.22857143{{.*}}0.45714286, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: %[[D:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = bf16}>
// CHECK: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}>
// CHECK: %[[A:[0-9]+]] = "onnx.MatMul"(%[[U]], %[[W4]]) {onnx_node_name = "SHARED_INT4"}
// CHECK: %[[YD:[0-9]+]] = "onnx.Cast"(%[[A]]) <{saturate = 1 : si64, to = bf16}>
// CHECK: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// INT7 scale = float32(0.8 / 63). Rounded integer weights are 8,-16,32,-63.
// CHECK: %[[W7:[0-9]+]] = onnx.Constant dense<{{.*}}0.101587303, -0.203174606{{.*}}0.406349212, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: %[[B:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W7]]) {onnx_node_name = "SHARED_INT7"}
// CHECK: %[[C:[0-9]+]] = "onnx.MatMul"(%arg0, %[[ORIG]]) {onnx_node_name = "SHARED_UNSELECTED"}
// CHECK: return %[[YU]], %[[B]], %[[C]]

// -----

// Selecting one consumer of an intermediate value must not redirect the
// unselected consumer to that consumer's private QDQ input.
func.func @shared_intermediate(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4xf32>) {
  %source = "onnx.Relu"(%arg0) {onnx_node_name = "BRANCH_SOURCE"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %a = "onnx.Relu"(%source) {onnx_node_name = "BRANCH_BF16"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  %b = "onnx.Relu"(%source) {onnx_node_name = "BRANCH_UNSELECTED"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %a, %b : tensor<1x4xf32>, tensor<1x4xf32>
}
// CHECK-LABEL: func.func @shared_intermediate
// CHECK-NEXT: %[[SOURCE:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "BRANCH_SOURCE"}
// CHECK-NEXT: %[[D:[0-9]+]] = "onnx.Cast"(%[[SOURCE]]) <{saturate = 1 : si64, to = bf16}>
// CHECK-NEXT: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}>
// CHECK-NEXT: %[[A:[0-9]+]] = "onnx.Relu"(%[[U]]) {onnx_node_name = "BRANCH_BF16"}
// CHECK-NEXT: %[[YD:[0-9]+]] = "onnx.Cast"(%[[A]]) <{saturate = 1 : si64, to = bf16}>
// CHECK-NEXT: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// CHECK-NEXT: %[[B:[0-9]+]] = "onnx.Relu"(%[[SOURCE]]) {onnx_node_name = "BRANCH_UNSELECTED"}
// CHECK-NEXT: return %[[YU]], %[[B]]

// -----

// Conv bias has values that would change under either int4 or bf16. It must
// remain the exact original constant while both weight and activation change.
func.func @conv_nonzero_bias(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x1x1xf32> {
  %w = onnx.Constant dense<[[[[0.1, -0.2], [0.4, -0.8]]]]> : tensor<1x1x2x2xf32>
  %bias = onnx.Constant dense<[0.12345]> : tensor<1xf32>
  %r = "onnx.Conv"(%arg0, %w, %bias) {onnx_node_name = "CONV_BIAS", auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2]} : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, tensor<1xf32>) -> tensor<1x1x1x1xf32>
  return %r : tensor<1x1x1x1xf32>
}
// CHECK-LABEL: func.func @conv_nonzero_bias
// CHECK: %[[BIAS:[0-9]+]] = onnx.Constant dense<1.234500e-01> : tensor<1xf32>
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.114285715, -0.22857143{{.*}}0.45714286, -8.000000e-01{{.*}}> : tensor<1x1x2x2xf32>
// CHECK-NEXT: %[[D:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = bf16}>
// CHECK-NEXT: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}>
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.Conv"(%[[U]], %[[W]], %[[BIAS]])
// CHECK-NEXT: %[[YD:[0-9]+]] = "onnx.Cast"(%[[R]]) <{saturate = 1 : si64, to = bf16}>
// CHECK-NEXT: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// CHECK-NEXT: return %[[YU]]

// -----

// Gemm's C (including broadcast bias) is likewise never passed through QDQ.
// Non-default arithmetic/transpose attributes must survive rebuilding the op.
func.func @gemm_nonzero_bias(%arg0: tensor<2x1xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %bias = onnx.Constant dense<[0.12345, -0.23456]> : tensor<2xf32>
  %r = "onnx.Gemm"(%arg0, %w, %bias) {onnx_node_name = "GEMM_BIAS", alpha = 0.5 : f32, beta = 1.5 : f32, transA = 1 : si64, transB = 1 : si64} : (tensor<2x1xf32>, tensor<2x2xf32>, tensor<2xf32>) -> tensor<1x2xf32>
  return %r : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @gemm_nonzero_bias
// CHECK: %[[BIAS:[0-9]+]] = onnx.Constant dense<[1.234500e-01, -2.345600e-01]> : tensor<2xf32>
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.114285715, -0.22857143{{.*}}0.45714286, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK-NEXT: %[[D:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = f16}>
// CHECK-NEXT: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}>
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.Gemm"(%[[U]], %[[W]], %[[BIAS]]) <{alpha = 5.000000e-01 : f32, beta = 1.500000e+00 : f32, transA = 1 : si64, transB = 1 : si64}>
// CHECK-NEXT: %[[YD:[0-9]+]] = "onnx.Cast"(%[[R]]) <{saturate = 1 : si64, to = f16}>
// CHECK-NEXT: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// CHECK-NEXT: return %[[YU]]
