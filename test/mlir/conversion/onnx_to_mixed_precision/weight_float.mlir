// RUN: env MIXEDP_NODE_FORMATS="BF:w=bf16,HF:w=f16,HF_EDGE:w=f16,F32:w=fp32:a=fp32" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file 2>&1 | FileCheck %s --implicit-check-not=warning

// bf16 / f16 weight fake-quant (compile time): round-to-nearest-even through
// the target format and back to f32, folded into the constant. Expected values
// come from numpy float16 and an independent bit-level bf16 rounding.

func.func @bf16(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.3, -0.4]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "BF"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @bf16
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.100097656, -0.200195313{{.*}}0.30078125, -0.400390625{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

func.func @f16(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.3, -0.4]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "HF"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @f16
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.0999755859, -0.199951172{{.*}}0.300048828, -0.399902344{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// f16 range edges: 70000 overflows to +Inf, the smallest f16 subnormal
// (5.96046448e-8) survives, 1e-8 underflows to 0, 65504 (f16 max) is exact.
func.func @f16_edges(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[70000.0, 5.96046448e-08], [1.0e-8, 65504.0]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "HF_EDGE"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @f16_edges
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0x7F800000, 5.96046448E-8{{.*}}0.000000e+00, 6.550400e+04{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// fp32 is the explicit "leave alone" format: no replacement constant is
// created, the op keeps consuming the original weight.
func.func @fp32_is_identity(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.3, -0.4]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "F32"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @fp32_is_identity
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}1.000000e-01, -2.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK-NEXT: "onnx.MatMul"(%arg0, %[[W]])
