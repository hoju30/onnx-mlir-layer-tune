// RUN: env MIXEDP_NODE_FORMATS="R_INT4:a=int4:0.05:0:0.1:0,ADD_INT6:a=int6:0.05:0:0.06:1:0.1:0,R_BF16:a=bf16,R_F16:a=f16,R_P8:a=p8e1,R_F32:a=fp32,MM_W4_ABF16:w=int4:a=bf16,MM_ABF16_ONLY:a=bf16" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file > %t 2>&1
// RUN: FileCheck %s --implicit-check-not=warning < %t
// RUN: FileCheck %s --check-prefix=UNIQUE < %t

// Activation QDQ (runtime): the pass inserts a quantize->dequantize round trip
// in front of every activation operand and after the result, and the ORIGINAL
// op is rebuilt unchanged on the round-tripped f32 values. The checks below
// follow the SSA dataflow (each step must consume the previous step's result),
// not just the presence of ops.

// INT4: x -> Div(scale) -> Add(zp) -> Round -> Clip(qmin=-8, qmax=7) -> Sub(zp)
// -> Mul(scale), then Relu on that, then the same chain again with the result's
// own (y_scale, y_zp).
func.func @int4_unary(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_INT4"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @int4_unary
// CHECK: %[[S:[0-9]+]] = onnx.Constant dense<5.000000e-02> : tensor<f32>
// CHECK: %[[ZP:[0-9]+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[QMIN:[0-9]+]] = onnx.Constant dense<-8.000000e+00> : tensor<f32>
// CHECK: %[[QMAX:[0-9]+]] = onnx.Constant dense<7.000000e+00> : tensor<f32>
// CHECK: %[[DIV:[0-9]+]] = "onnx.Div"(%arg0, %[[S]])
// CHECK: %[[ADD:[0-9]+]] = "onnx.Add"(%[[DIV]], %[[ZP]])
// CHECK: %[[RND:[0-9]+]] = "onnx.Round"(%[[ADD]])
// CHECK: %[[CLP:[0-9]+]] = "onnx.Clip"(%[[RND]], %[[QMIN]], %[[QMAX]])
// CHECK: %[[SUB:[0-9]+]] = "onnx.Sub"(%[[CLP]], %[[ZP]])
// CHECK: %[[MUL:[0-9]+]] = "onnx.Mul"(%[[SUB]], %[[S]])
// CHECK: %[[RELU:[0-9]+]] = "onnx.Relu"(%[[MUL]]) {onnx_node_name = "R_INT4"}
// CHECK: %[[YS:[0-9]+]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[YZP:[0-9]+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[YQMIN:[0-9]+]] = onnx.Constant dense<-8.000000e+00> : tensor<f32>
// CHECK: %[[YQMAX:[0-9]+]] = onnx.Constant dense<7.000000e+00> : tensor<f32>
// CHECK: %[[YDIV:[0-9]+]] = "onnx.Div"(%[[RELU]], %[[YS]])
// CHECK: %[[YADD:[0-9]+]] = "onnx.Add"(%[[YDIV]], %[[YZP]])
// CHECK: %[[YRND:[0-9]+]] = "onnx.Round"(%[[YADD]])
// CHECK: %[[YCLP:[0-9]+]] = "onnx.Clip"(%[[YRND]], %[[YQMIN]], %[[YQMAX]])
// CHECK: %[[YSUB:[0-9]+]] = "onnx.Sub"(%[[YCLP]], %[[YZP]])
// CHECK: %[[OUT:[0-9]+]] = "onnx.Mul"(%[[YSUB]], %[[YS]])
// CHECK: return %[[OUT]]

// -----

// Two-input elementwise op: each operand is round-tripped with ITS OWN
// (scale, zero_point) -- 0.05/0 for %arg0, 0.06/1 for %arg1 -- and the result
// with the third pair (0.1/0). INT6 => clip range [-32, 31].
func.func @int6_binary(%arg0: tensor<1x4xf32>, %arg1: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "ADD_INT6"} : (tensor<1x4xf32>, tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @int6_binary
// CHECK: %[[AS:[0-9]+]] = onnx.Constant dense<5.000000e-02> : tensor<f32>
// CHECK: %[[AZP:[0-9]+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[AMIN:[0-9]+]] = onnx.Constant dense<-3.200000e+01> : tensor<f32>
// CHECK: %[[AMAX:[0-9]+]] = onnx.Constant dense<3.100000e+01> : tensor<f32>
// CHECK: %[[ADIV:[0-9]+]] = "onnx.Div"(%arg0, %[[AS]])
// CHECK: %[[AADD:[0-9]+]] = "onnx.Add"(%[[ADIV]], %[[AZP]])
// CHECK: %[[ARND:[0-9]+]] = "onnx.Round"(%[[AADD]])
// CHECK: %[[ACLP:[0-9]+]] = "onnx.Clip"(%[[ARND]], %[[AMIN]], %[[AMAX]])
// CHECK: %[[ASUB:[0-9]+]] = "onnx.Sub"(%[[ACLP]], %[[AZP]])
// CHECK: %[[AQ:[0-9]+]] = "onnx.Mul"(%[[ASUB]], %[[AS]])
// CHECK: %[[BS:[0-9]+]] = onnx.Constant dense<6.000000e-02> : tensor<f32>
// CHECK: %[[BZP:[0-9]+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[BMIN:[0-9]+]] = onnx.Constant dense<-3.200000e+01> : tensor<f32>
// CHECK: %[[BMAX:[0-9]+]] = onnx.Constant dense<3.100000e+01> : tensor<f32>
// CHECK: %[[BDIV:[0-9]+]] = "onnx.Div"(%arg1, %[[BS]])
// CHECK: %[[BADD:[0-9]+]] = "onnx.Add"(%[[BDIV]], %[[BZP]])
// CHECK: %[[BRND:[0-9]+]] = "onnx.Round"(%[[BADD]])
// CHECK: %[[BCLP:[0-9]+]] = "onnx.Clip"(%[[BRND]], %[[BMIN]], %[[BMAX]])
// CHECK: %[[BSUB:[0-9]+]] = "onnx.Sub"(%[[BCLP]], %[[BZP]])
// CHECK: %[[BQ:[0-9]+]] = "onnx.Mul"(%[[BSUB]], %[[BS]])
// CHECK: %[[SUM:[0-9]+]] = "onnx.Add"(%[[AQ]], %[[BQ]]) {onnx_node_name = "ADD_INT6"}
// CHECK: %[[YS:[0-9]+]] = onnx.Constant dense<1.000000e-01> : tensor<f32>
// CHECK: %[[YZP:[0-9]+]] = onnx.Constant dense<0.000000e+00> : tensor<f32>
// CHECK: %[[YMIN:[0-9]+]] = onnx.Constant dense<-3.200000e+01> : tensor<f32>
// CHECK: %[[YMAX:[0-9]+]] = onnx.Constant dense<3.100000e+01> : tensor<f32>
// CHECK: %[[YDIV:[0-9]+]] = "onnx.Div"(%[[SUM]], %[[YS]])
// CHECK: %[[YADD:[0-9]+]] = "onnx.Add"(%[[YDIV]], %[[YZP]])
// CHECK: %[[YRND:[0-9]+]] = "onnx.Round"(%[[YADD]])
// CHECK: %[[YCLP:[0-9]+]] = "onnx.Clip"(%[[YRND]], %[[YMIN]], %[[YMAX]])
// CHECK: %[[YSUB:[0-9]+]] = "onnx.Sub"(%[[YCLP]], %[[YZP]])
// CHECK: %[[OUT:[0-9]+]] = "onnx.Mul"(%[[YSUB]], %[[YS]])
// CHECK: return %[[OUT]]

// -----

// bf16 / f16: cast down and immediately back up to f32 (a bracket, NOT a
// sustained retype -- the original op still runs on f32 operands).
func.func @bf16_bracket(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_BF16"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @bf16_bracket
// CHECK: %[[D:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = bf16}> : (tensor<1x4xf32>) -> tensor<1x4xbf16>
// CHECK: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}> : (tensor<1x4xbf16>) -> tensor<1x4xf32>
// CHECK: %[[R:[0-9]+]] = "onnx.Relu"(%[[U]]) {onnx_node_name = "R_BF16"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
// CHECK: %[[YD:[0-9]+]] = "onnx.Cast"(%[[R]]) <{saturate = 1 : si64, to = bf16}>
// CHECK: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// CHECK: return %[[YU]]

// -----

func.func @f16_bracket(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_F16"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @f16_bracket
// CHECK: %[[D:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = f16}> : (tensor<1x4xf32>) -> tensor<1x4xf16>
// CHECK: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}> : (tensor<1x4xf16>) -> tensor<1x4xf32>
// CHECK: %[[R:[0-9]+]] = "onnx.Relu"(%[[U]]) {onnx_node_name = "R_F16"}
// CHECK: %[[YD:[0-9]+]] = "onnx.Cast"(%[[R]]) <{saturate = 1 : si64, to = f16}>
// CHECK: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// CHECK: return %[[YU]]

// -----

// posit: from_f32 -> to_f32 pairs around the op. Regression: every from_f32
// MUST carry a nonzero, per-site-unique "qalign_key". Without it,
// --convert-posit-to-krnl folds a directly-adjacent to_f32(from_f32(x)) to x,
// silently turning the whole simulated round trip into a no-op (see
// posit_lowering.mlir for the lowered-IR check of the same regression).
func.func @posit_p8e1(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_P8"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @posit_p8e1
// CHECK: %[[P:[0-9]+]] = posit.from_f32 %arg0 {qalign_key = [[K1:[1-9][0-9]*]] : i64} : tensor<1x4xf32> -> tensor<1x4x!posit.type<8, 1>>
// CHECK: %[[F:[0-9]+]] = posit.to_f32 %[[P]] : tensor<1x4x!posit.type<8, 1>> -> tensor<1x4xf32>
// CHECK: %[[R:[0-9]+]] = "onnx.Relu"(%[[F]]) {onnx_node_name = "R_P8"}
// CHECK: %[[YP:[0-9]+]] = posit.from_f32 %[[R]] {qalign_key = [[K2:[1-9][0-9]*]] : i64}
// CHECK: %[[YF:[0-9]+]] = posit.to_f32 %[[YP]] : tensor<1x4x!posit.type<8, 1>> -> tensor<1x4xf32>
// CHECK: return %[[YF]]

// Use an independent scan: the forbidden region must include the ENTIRE
// second from_f32 line. A CHECK-NOT followed by CHECK on that same line would
// stop before the positive match and could miss a repeated key inside it.
// UNIQUE-LABEL: func.func @posit_p8e1
// UNIQUE: posit.from_f32 %arg0 {qalign_key = [[KEY:[1-9][0-9]*]] : i64}
// UNIQUE-NOT: qalign_key = [[KEY]] :
// UNIQUE: return

// -----

// fp32 is the explicit "leave alone" format: nothing is inserted.
func.func @fp32_is_noop(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_F32"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @fp32_is_noop
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "R_F32"}
// CHECK-NEXT: return %[[R]]

// -----

// A node that isn't named in MIXEDP_NODE_FORMATS is never touched.
func.func @unselected_node_untouched(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_NOT_LISTED"} : (tensor<1x4xf32>) -> tensor<1x4xf32>
  return %0 : tensor<1x4xf32>
}
// CHECK-LABEL: func.func @unselected_node_untouched
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.Relu"(%arg0) {onnx_node_name = "R_NOT_LISTED"}
// CHECK-NEXT: return %[[R]]

// -----

// The point of the feature: ONE node, weight and activation in DIFFERENT
// formats. Weight = int4 (compile-time constant, replaced), activation = bf16
// (cast bracket on the input and on the result). The MatMul consumes the
// bracketed activation and the replaced weight.
func.func @independent_weight_and_activation(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "MM_W4_ABF16"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @independent_weight_and_activation
// CHECK: onnx.Constant dense<{{.*}}1.000000e-01, -2.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}0.114285715, -0.22857143{{.*}}0.45714286, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK: %[[D:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = bf16}>
// CHECK: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}>
// CHECK: %[[MM:[0-9]+]] = "onnx.MatMul"(%[[U]], %[[W]]) {onnx_node_name = "MM_W4_ABF16"}
// CHECK: %[[YD:[0-9]+]] = "onnx.Cast"(%[[MM]]) <{saturate = 1 : si64, to = bf16}>
// CHECK: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// CHECK: return %[[YU]]

// -----

// Omitting w= leaves the weight alone: only the activation is touched.
func.func @activation_only_keeps_weight(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "MM_ABF16_ONLY"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @activation_only_keeps_weight
// CHECK: %[[W:[0-9]+]] = onnx.Constant dense<{{.*}}1.000000e-01, -2.000000e-01{{.*}}4.000000e-01, -8.000000e-01{{.*}}> : tensor<2x2xf32>
// CHECK-NOT: onnx.Constant
// CHECK: "onnx.MatMul"(%{{[0-9]+}}, %[[W]]) {onnx_node_name = "MM_ABF16_ONLY"}
