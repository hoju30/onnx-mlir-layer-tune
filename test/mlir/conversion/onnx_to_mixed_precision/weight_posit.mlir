// RUN: env MIXEDP_NODE_FORMATS="P4_A:w=p4e1,P4_B:w=p4e1,P4_C:w=p4e1,P6:w=p6e1,P8:w=p8e1" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file 2>&1 | FileCheck %s --implicit-check-not=warning

// Posit weight fake-quant (compile time, LUT-accelerated for N<=8). The
// expected values below were produced by sw::universal::posit<N,ES> itself
// (float -> posit -> float), NOT by the pass, and the LUT is required to match
// it exactly. Posit rounding is on the encoded bit string, so it is *not*
// "nearest representable value": a nonzero input never underflows to 0 (it
// lands on minpos), values past maxpos saturate, and ties/regime boundaries
// follow the encoding (e.g. p4e1 0.75 -> 1, 5 -> 4).

// Regression: -16 is an ordinary, exactly-representable p4e1 value. An earlier
// version put posit's NaR encoding (which decodes to NaN) into the sorted LUT,
// which corrupted the binary search and turned -16 into NaN.
func.func @p4e1_nar_regression(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[-16.0, -4.0], [0.001, 0.75]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "P4_A"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @p4e1_nar_regression
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}-1.600000e+01, -4.000000e+00{{.*}}6.250000e-02, 1.000000e+00{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// Saturation to +-maxpos (16 for p4e1) and bit-string rounding (5 -> 4).
func.func @p4e1_saturation(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[3.0e38, -3.0e38], [1.0, 5.0]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "P4_B"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @p4e1_saturation
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}1.600000e+01, -1.600000e+01{{.*}}1.000000e+00, 4.000000e+00{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// Posit has no infinity: +-Inf and NaN all become NaR, which decodes to NaN
// (0x7FC00000). This differs on purpose from the INT weight path, which passes
// non-finite elements through unchanged. Zero stays zero.
func.func @p4e1_non_finite(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0x7F800000, 0xFF800000], [0x7FC00000, 0.0]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "P4_C"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @p4e1_non_finite
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0x7FC00000, 0x7FC00000{{.*}}0x7FC00000, 0.000000e+00{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// p6e1: 0.001 -> minpos (2^-8), 0.75 exactly representable, +-maxpos = 256.
func.func @p6e1(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.001, 0.75], [-256.0, 3.0e38]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "P6"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @p6e1
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}3.906250e-03, 7.500000e-01{{.*}}-2.560000e+02, 2.560000e+02{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// p8e1: 0.001 -> 2^-10, +-maxpos = 4096.
func.func @p8e1(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.001, 0.75], [-4096.0, 3.0e38]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "P8"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @p8e1
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}9.765625E-4, 7.500000e-01{{.*}}-4.096000e+03, 4.096000e+03{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])
