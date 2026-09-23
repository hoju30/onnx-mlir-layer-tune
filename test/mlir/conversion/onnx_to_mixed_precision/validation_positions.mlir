// RUN: env MIXEDP_NODE_FORMATS="SECOND_SCALE:a=int4:0.25:0:0:0:0.25:0,SECOND_ZP:a=int5:0.25:0:0.5:nan:0.25:0,OUTPUT_ZP:a=int7:0.25:0:0.5:0:0.25:inf,OUTPUT_SCALE:a=int6:0.25:0:0.5:0:1e-40:0,SUBNORMAL_SCALE:a=int6:1e-40:0:0.25:0,INF_SCALE:a=int8:+Inf:0:0.25:0,KEEP_WEIGHT:w=int7:a=int4:0.25:0:0.25:inf,KEEP_WEIGHT_BAD_TAG:w=int4:a=fp16" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file > %t 2> %t.err
// RUN: FileCheck %s --check-prefix=WARN < %t.err
// RUN: FileCheck %s --check-prefix=IR --implicit-check-not=onnx.Round --implicit-check-not=onnx.Cast --implicit-check-not=posit.from_f32 < %t

// Validation must examine every parameter position and every supported INT
// width. A bad second input or result parameter disables the entire activation
// bracket, leaving the original operands and return value connected directly.
func.func @second_input_scale(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %r = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "SECOND_SCALE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS entry for 'SECOND_SCALE' (onnx.Add) has an invalid activation scale/zero-point
// IR-LABEL: func.func @second_input_scale
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "SECOND_SCALE"}
// IR-NEXT: return %[[R]]

// -----

func.func @second_input_zero_point(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %r = "onnx.Sub"(%arg0, %arg1) {onnx_node_name = "SECOND_ZP"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS entry for 'SECOND_ZP' (onnx.Sub) has an invalid activation scale/zero-point
// IR-LABEL: func.func @second_input_zero_point
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Sub"(%arg0, %arg1) {onnx_node_name = "SECOND_ZP"}
// IR-NEXT: return %[[R]]

// -----

func.func @binary_output_zero_point(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %r = "onnx.Mul"(%arg0, %arg1) {onnx_node_name = "OUTPUT_ZP"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS entry for 'OUTPUT_ZP' (onnx.Mul) has an invalid activation scale/zero-point
// IR-LABEL: func.func @binary_output_zero_point
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Mul"(%arg0, %arg1) {onnx_node_name = "OUTPUT_ZP"}
// IR-NEXT: return %[[R]]

// -----

func.func @binary_output_scale(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
  %r = "onnx.Div"(%arg0, %arg1) {onnx_node_name = "OUTPUT_SCALE"} : (tensor<4xf32>, tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS entry for 'OUTPUT_SCALE' (onnx.Div) has an invalid activation scale/zero-point
// IR-LABEL: func.func @binary_output_scale
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Div"(%arg0, %arg1) {onnx_node_name = "OUTPUT_SCALE"}
// IR-NEXT: return %[[R]]

// -----

// Unlike 1e-100, this scale is nonzero after narrowing to f32. It must still
// be rejected because it is subnormal and targets may flush it to zero.
func.func @positive_subnormal_scale(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %r = "onnx.Relu"(%arg0) {onnx_node_name = "SUBNORMAL_SCALE"} : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS entry for 'SUBNORMAL_SCALE' (onnx.Relu) has an invalid activation scale/zero-point
// IR-LABEL: func.func @positive_subnormal_scale
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "SUBNORMAL_SCALE"}
// IR-NEXT: return %[[R]]

// -----

func.func @literal_positive_infinity_scale(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %r = "onnx.Relu"(%arg0) {onnx_node_name = "INF_SCALE"} : (tensor<4xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS entry for 'INF_SCALE' (onnx.Relu) has an invalid activation scale/zero-point
// IR-LABEL: func.func @literal_positive_infinity_scale
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "INF_SCALE"}
// IR-NEXT: return %[[R]]

// -----

// Invalid activation params must not cancel a valid weight request. Confirm
// actual replacement values and direct f32 activation/result connections.
func.func @valid_weight_bad_activation_param(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %r = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "KEEP_WEIGHT"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %r : tensor<1x2xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS entry for 'KEEP_WEIGHT' (onnx.MatMul) has an invalid activation scale/zero-point
// IR-LABEL: func.func @valid_weight_bad_activation_param
// IR-NEXT: onnx.Constant dense<{{.*}}1.000000e-01, -2.000000e-01{{.*}}4.000000e-01, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// IR-NEXT: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.101587303, -0.203174606{{.*}}0.406349212, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// IR-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "KEEP_WEIGHT"}
// IR-NEXT: return %[[R]]

// -----

func.func @valid_weight_bad_activation_tag(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %r = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "KEEP_WEIGHT_BAD_TAG"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %r : tensor<1x2xf32>
}
// WARN: warning: MIXEDP_NODE_FORMATS 'a=' entry for 'KEEP_WEIGHT_BAD_TAG' has an unrecognized format tag
// IR-LABEL: func.func @valid_weight_bad_activation_tag
// IR-NEXT: onnx.Constant dense<{{.*}}1.000000e-01, -2.000000e-01{{.*}}4.000000e-01, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// IR-NEXT: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.114285715, -0.22857143{{.*}}0.45714286, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// IR-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "KEEP_WEIGHT_BAD_TAG"}
// IR-NEXT: return %[[R]]
