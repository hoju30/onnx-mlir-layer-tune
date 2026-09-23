// RUN: env MIXEDP_NODE_FORMATS="V_SCALE0:a=int8:0:0:0.1:0,V_SCALE_TINY:a=int8:1e-100:0:0.1:0,V_SCALE_HUGE:a=int8:1e100:0:0.1:0,V_SCALE_NAN:a=int8:nan:0:0.1:0,V_SCALE_NEG:a=int8:-1:0:0.1:0,V_ZP_NAN:a=int8:0.25:nan:0.1:0,V_ZP_HUGE:a=int8:0.25:1e100:0.1:0,V_YSCALE0:a=int8:0.25:0:0:0,V_NOTNUM:a=int8:oops:0:0.1:0,V_OVERFLOWNUM:a=int8:1e9999:0:0.1:0,V_COUNT:a=int8:0.1:0,V_TYPO_A:a=fp16,V_ES_HUGE:a=p8e999,V_ES_MAX:a=p8e4294967295,V_ES_GE_N:a=p4e4,V_W_NOWEIGHT:w=int8,V_TYPO_W:w=int9,V_W_NONCONST:w=int8,V_W_SUBNORMAL:w=int8,V_W_F16CONST:w=int4,V_MAXPOOL:a=int8:0.05:0:0.05:0,V_UNRANKED:a=p8e1" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file -o /dev/null 2>&1 | FileCheck %s
// RUN: env MIXEDP_NODE_FORMATS="V_SCALE0:a=int8:0:0:0.1:0,V_SCALE_TINY:a=int8:1e-100:0:0.1:0,V_SCALE_HUGE:a=int8:1e100:0:0.1:0,V_SCALE_NAN:a=int8:nan:0:0.1:0,V_SCALE_NEG:a=int8:-1:0:0.1:0,V_ZP_NAN:a=int8:0.25:nan:0.1:0,V_ZP_HUGE:a=int8:0.25:1e100:0.1:0,V_YSCALE0:a=int8:0.25:0:0:0,V_NOTNUM:a=int8:oops:0:0.1:0,V_OVERFLOWNUM:a=int8:1e9999:0:0.1:0,V_COUNT:a=int8:0.1:0,V_TYPO_A:a=fp16,V_ES_HUGE:a=p8e999,V_ES_MAX:a=p8e4294967295,V_ES_GE_N:a=p4e4,V_W_NOWEIGHT:w=int8,V_TYPO_W:w=int9,V_W_NONCONST:w=int8,V_W_SUBNORMAL:w=int8,V_W_F16CONST:w=int4,V_MAXPOOL:a=int8:0.05:0:0.05:0,V_UNRANKED:a=p8e1" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file 2>/dev/null | FileCheck %s --check-prefix=IR --implicit-check-not=onnx.Round --implicit-check-not=onnx.Cast --implicit-check-not=posit.from_f32

// Invalid configuration must be REJECTED WITH A WARNING and the node left
// exactly as it was -- never silently turned into "no quantization", never
// built into broken IR (divide by zero / NaN / Inf constants), and never able
// to abort or crash the compiler. Each case below guards one specific problem
// that was found and fixed; the second RUN line asserts that none of these
// nodes ended up with any quantization ops inserted.
//
// The pass exits 0 for all of them: RUN pipelines fail on a nonzero/crash exit
// of onnx-mlir-opt (lit runs pipelines with pipefail).

// scale = 0 (would build a divide-by-zero).
func.func @v_scale0(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE0"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_scale0
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE0"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_SCALE0' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// scale = 1e-100: a fine positive double, but 0.0f once narrowed to the FP32 constant that is baked into the IR.
func.func @v_scale_tiny(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_TINY"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_scale_tiny
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_TINY"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_SCALE_TINY' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// scale = 1e100: a fine double, but +Inf once narrowed to FP32.
func.func @v_scale_huge(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_HUGE"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_scale_huge
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_HUGE"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_SCALE_HUGE' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// scale = NaN.
func.func @v_scale_nan(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_NAN"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_scale_nan
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_NAN"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_SCALE_NAN' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// scale < 0.
func.func @v_scale_neg(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_NEG"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_scale_neg
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_SCALE_NEG"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_SCALE_NEG' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// zero-point = NaN. (Only scales used to be validated; zero-points slipped through.)
func.func @v_zp_nan(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_ZP_NAN"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_zp_nan
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_ZP_NAN"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_ZP_NAN' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// zero-point = 1e100: finite double, +Inf as FP32.
func.func @v_zp_huge(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_ZP_HUGE"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_zp_huge
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_ZP_HUGE"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_ZP_HUGE' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// the RESULT's scale (3rd param) is validated too, not just the operand's.
func.func @v_yscale0(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_YSCALE0"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_yscale0
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_YSCALE0"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_YSCALE0' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// non-numeric field: std::stod used to throw and abort the whole compiler; it is now caught and rejected here.
func.func @v_notnum(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_NOTNUM"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_notnum
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_NOTNUM"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_NOTNUM' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// 1e9999 overflows double: std::stod's out_of_range used to abort the compiler.
func.func @v_overflownum(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_OVERFLOWNUM"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_overflownum
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_OVERFLOWNUM"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_OVERFLOWNUM' (onnx.Relu) has an invalid activation scale/zero-point

// -----

// wrong parameter count (int8 on a unary op needs 4).
func.func @v_count(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_COUNT"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_count
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_COUNT"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS entry for 'V_COUNT' (onnx.Relu) needs exactly 4 activation params

// -----

// typo in the format tag: used to silently mean 'no quantization'.
func.func @v_typo_a(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_TYPO_A"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_typo_a
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_TYPO_A"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS 'a=' entry for 'V_TYPO_A' has an unrecognized format tag

// -----

// illegal posit ES (es >= nbits): PositType's verifier used to assert-abort the compiler.
func.func @v_es_huge(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_ES_HUGE"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_es_huge
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_ES_HUGE"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS 'a=' entry for 'V_ES_HUGE' has an unrecognized format tag

// -----

// illegal posit ES = UINT_MAX.
func.func @v_es_max(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_ES_MAX"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_es_max
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_ES_MAX"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS 'a=' entry for 'V_ES_MAX' has an unrecognized format tag

// -----

// es == nbits is also illegal.
func.func @v_es_ge_n(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_ES_GE_N"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_es_ge_n
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_ES_GE_N"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS 'a=' entry for 'V_ES_GE_N' has an unrecognized format tag

// -----

// weight format on an op that has no weight operand.
func.func @v_w_noweight(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_W_NOWEIGHT"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// IR-LABEL: func.func @v_w_noweight
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_W_NOWEIGHT"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS gives 'V_W_NOWEIGHT' a weight format but its op type ('onnx.Relu') has no weight operand

// -----

// typo in the WEIGHT format tag (int9 does not exist).
func.func @v_typo_w(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "V_TYPO_W"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// IR-LABEL: func.func @v_typo_w
// IR-NEXT: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}1.000000e-01, -2.000000e-01{{.*}}4.000000e-01, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// IR-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "V_TYPO_W"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS 'w=' entry for 'V_TYPO_W' has an unrecognized format tag

// -----

// weight is not a compile-time constant, so it cannot be folded.
func.func @v_w_nonconst(%arg0: tensor<1x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<1x2xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "V_W_NONCONST"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// IR-LABEL: func.func @v_w_nonconst
// IR-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %arg1) {onnx_node_name = "V_W_NONCONST"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS names 'V_W_NONCONST' with a weight format but its weight operand is not a compile-time onnx.Constant

// -----

// all-subnormal weight: maxAbs/qmax underflows to a 0.0f scale; the weight is left untouched instead of dividing by zero.
func.func @v_w_subnormal(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[1.40129846e-45, -1.40129846e-45], [0.0, 1.40129846e-45]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "V_W_SUBNORMAL"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// IR-LABEL: func.func @v_w_subnormal
// IR-NEXT: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}1.401300e-45, -1.401300e-45{{.*}}0.000000e+00, 1.401300e-45{{.*}}> : tensor<2x2xf32>
// IR-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "V_W_SUBNORMAL"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS names 'V_W_SUBNORMAL' with an unsupported weight format; leaving weight untouched

// -----

// weight constant stored as f16: getValues<float>() used to assert-abort; now warned and left untouched.
func.func @v_w_f16const(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf16>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "V_W_F16CONST"} : (tensor<1x2xf32>, tensor<2x2xf16>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// IR-LABEL: func.func @v_w_f16const
// IR-NEXT: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}1.000000e+00, 2.000000e+00{{.*}}3.000000e+00, 4.000000e+00{{.*}}> : tensor<2x2xf16>
// IR-NEXT: %[[R:[0-9]+]] = "onnx.MatMul"(%arg0, %[[W]]) {onnx_node_name = "V_W_F16CONST"}
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS names 'V_W_F16CONST' with an unsupported weight format; leaving weight untouched

// -----

// The TWO-result onnx.MaxPool (Y plus an Indices result) is unsupported: onnx-mlir's own
// --convert-onnx-to-krnl never lowers it, with or without any quantization pass, so
// rewriting it would gain nothing. The single-result onnx.MaxPoolSingleOut that real
// models contain IS supported (see maxpool.mlir).
func.func @v_maxpool(%arg0: tensor<1x1x4x4xf32>) -> tensor<1x1x2x2xf32> {
  %0, %1 = "onnx.MaxPool"(%arg0) {onnx_node_name = "V_MAXPOOL", kernel_shape = [2, 2]} : (tensor<1x1x4x4xf32>) -> (tensor<1x1x2x2xf32>, none)
  return %0 : tensor<1x1x2x2xf32>
}
// IR-LABEL: func.func @v_maxpool
// IR-NEXT: %[[R:[a-zA-Z0-9_]+]], %{{[a-zA-Z0-9_]+}} = "onnx.MaxPool"(%arg0)
// IR-NEXT: return %[[R]]


// CHECK: warning: MIXEDP_NODE_FORMATS names node 'V_MAXPOOL' but its op type ('onnx.MaxPool') is not supported; skipping

// -----

// Crash regression: an UNRANKED posit activation used to SIGSEGV. The value is a function argument, which has no
// defining op, and the warning path dereferenced it. It now warns twice (once for the block argument's location,
// once for the op) and exits normally.
func.func @v_unranked(%arg0: tensor<*xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "V_UNRANKED"} : (tensor<*xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
// IR-LABEL: func.func @v_unranked
// IR-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "V_UNRANKED"}
// IR-NEXT: return %[[R]]


// CHECK: warning: posit activation QDQ needs a ranked tensor; leaving untouched.
// CHECK: warning: posit activation QDQ needs a ranked tensor; leaving untouched.

