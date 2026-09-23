// RUN: env MIXEDP_NODE_FORMATS="MM_INT4:w=int4,MM_INT8:w=int8,CONV_INT6:w=int6,GEMM_INT5:w=int5,EXT_INT8:w=int8,INF_INT8:w=int8,ZERO_INT4:w=int4,NAN_INT7:w=int7,EXT_INT6:w=int6" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file 2>&1 | FileCheck %s --implicit-check-not=warning

// INT weight fake-quant (compile time): symmetric, scale = max|w| / (2^(bits-1)-1),
// round half away from zero, clamp, rescale back to f32. The rewritten constant
// must replace the weight operand of the (otherwise unchanged) op. Expected
// values were computed independently in float32 (not copied from the pass);
// max|w| = 0.8, so e.g. int4: scale = 0.8/7, 0.1 -> 1*scale, 0.4 (=3.5*scale) -> 4*scale.

func.func @matmul_int4(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "MM_INT4"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @matmul_int4
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.114285715, -0.22857143{{.*}}0.45714286, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

func.func @matmul_int8(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "MM_INT8"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @matmul_int8
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.100787401, -0.201574802{{.*}}0.403149605, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// The rewrite is op-generic: Conv (4-D weight) and Gemm use the same path.
func.func @conv_int6(%arg0: tensor<1x1x2x2xf32>) -> tensor<1x1x1x1xf32> {
  %w = onnx.Constant dense<[[[[0.1, -0.2], [0.4, -0.8]]]]> : tensor<1x1x2x2xf32>
  %b = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %w, %b) {onnx_node_name = "CONV_INT6", auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [2, 2]} : (tensor<1x1x2x2xf32>, tensor<1x1x2x2xf32>, none) -> tensor<1x1x1x1xf32>
  return %0 : tensor<1x1x1x1xf32>
}
// CHECK-LABEL: func.func @conv_int6
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.103225805, -0.20645161{{.*}}0.412903219, -8.000000e-01{{.*}}> : tensor<1x1x2x2xf32>
// CHECK: "onnx.Conv"(%arg0, %[[W]], %{{[0-9]+}})

// -----

func.func @gemm_int5(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %c = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Gemm"(%arg0, %w, %c) {onnx_node_name = "GEMM_INT5"} : (tensor<1x2xf32>, tensor<2x2xf32>, none) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @gemm_int5
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.106666669, -0.213333338{{.*}}0.426666677, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.Gemm"(%arg0, %[[W]], %{{[0-9]+}})

// -----

// Extreme values (regression). With max|w| = FLT_MAX the int8 rescale
// 127 * (FLT_MAX/127) overflows float32 to Inf; that element must fall back to
// its original finite value instead of becoming Inf. 1.0 is far below one
// quantization step at this dynamic range, so it correctly quantizes to 0.
func.func @int8_flt_max(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[3.402823466e38, -3.402823466e38], [1.0, -1.0]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "EXT_INT8"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @int8_flt_max
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}3.40282347E+38, -3.40282347E+38{{.*}}0.000000e+00, 0.000000e+00{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// Non-finite weight elements are passed through unchanged (regression: they
// used to poison the whole tensor with NaN). The scale is derived from the
// finite elements only (max|w| = 1), so 1.0 and -1.0 round-trip exactly.
func.func @int8_non_finite(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[1.0, -1.0], [0x7F800000, 0xFF800000]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "INF_INT8"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @int8_non_finite
// The rewritten constant is textually identical to the original here, so match
// the original first and then the replacement that the MatMul actually uses.
// CHECK: onnx.Constant dense<{{.*}}0x7F800000, 0xFF800000{{.*}}>
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}1.000000e+00, -1.000000e+00{{.*}}0x7F800000, 0xFF800000{{.*}}> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"(%arg0, %[[W]])

// -----

// All-zero weights have no nonzero maxAbs. They must remain zero without
// producing a zero divisor or non-finite replacement values.
func.func @int4_all_zero(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<0.0> : tensor<2x2xf32>
  %r = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "ZERO_INT4"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %r : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @int4_all_zero
// CHECK-NEXT: onnx.Constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT: %[[W:[0-9]+]] = onnx.Constant dense<0.000000e+00> : tensor<2x2xf32>
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "ZERO_INT4"}
// CHECK-NEXT: return %[[R]]

// -----

// NaN must neither poison the finite-element scale nor be converted to a
// saturated finite value. INT7 also exercises this policy away from INT8.
func.func @int7_nan_and_finite(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.8], [0x7FC00000, 0.0]]> : tensor<2x2xf32>
  %r = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "NAN_INT7"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %r : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @int7_nan_and_finite
// CHECK-NEXT: onnx.Constant dense<{{.*}}1.000000e-01, -8.000000e-01{{.*}}0x7FC00000, 0.000000e+00{{.*}}> : tensor<2x2xf32>
// CHECK-NEXT: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.101587303, -8.000000e-01{{.*}}0x7FC00000, 0.000000e+00{{.*}}> : tensor<2x2xf32>
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "NAN_INT7"}
// CHECK-NEXT: return %[[R]]

// -----

// INT6 is another width where qmax * float32(FLT_MAX/qmax) overflows.
// Preserve the finite extremes, while smaller values legitimately become 0.
func.func @int6_flt_max(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[3.402823466e38, -3.402823466e38], [1.0, -1.0]]> : tensor<2x2xf32>
  %r = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "EXT_INT6"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %r : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @int6_flt_max
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}3.40282347E+38, -3.40282347E+38{{.*}}0.000000e+00, 0.000000e+00{{.*}}> : tensor<2x2xf32>
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "EXT_INT6"}
// CHECK-NEXT: return %[[R]]
