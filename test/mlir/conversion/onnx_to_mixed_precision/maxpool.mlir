// RUN: env MIXEDP_NODE_FORMATS="MP_INT8:a=int8:0.05:1:0.02:-2,MP_ATTRS:a=bf16" onnx-mlir-opt --convert-onnx-to-mixed-precision %s -split-input-file 2>&1 | FileCheck %s --implicit-check-not=warning
// RUN: env MIXEDP_NODE_FORMATS="MP_INT8:a=int8:0.05:1:0.02:-2,MP_ATTRS:a=bf16" onnx-mlir-opt --convert-onnx-to-mixed-precision --canonicalize --shape-inference --convert-onnx-to-krnl --canonicalize --convert-krnl-to-affine --convert-krnl-to-llvm --reconcile-unrealized-casts %s -split-input-file -o /dev/null

// MaxPool is supported in the form real models contain: the single-result
// onnx.MaxPoolSingleOut. (The two-result onnx.MaxPool is not; see validation.mlir.)
//
// The second RUN line pushes the rewritten module through the ENTIRE lowering
// pipeline in ONE onnx-mlir-opt invocation and only requires it to succeed. It
// guards the failure ONNXToLowPrecision.cpp blames on a rebuilt MaxPool
// ("failed to legalize operation 'onnx.MaxPool'"): that failure belongs to the
// two-result form, which onnx-mlir never lowers even without any quantization
// pass, and does not occur for the op supported here.

// INT8 activation: QDQ chain in front of the pooling and another (with the
// result's own scale/zero-point) behind it. The pooling op itself is rebuilt
// unchanged: every attribute must survive, and it must consume the quantized
// input, not the original one.
func.func @maxpool_int8(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {onnx_node_name = "MP_INT8", kernel_shape = [2, 2], strides = [2, 2]} : (tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: func.func @maxpool_int8
// CHECK: %[[S:[0-9]+]] = onnx.Constant dense<5.000000e-02> : tensor<f32>
// CHECK: %[[ZP:[0-9]+]] = onnx.Constant dense<1.000000e+00> : tensor<f32>
// CHECK: %[[QMIN:[0-9]+]] = onnx.Constant dense<-1.280000e+02> : tensor<f32>
// CHECK: %[[QMAX:[0-9]+]] = onnx.Constant dense<1.270000e+02> : tensor<f32>
// CHECK: %[[DIV:[0-9]+]] = "onnx.Div"(%arg0, %[[S]])
// CHECK: %[[ADD:[0-9]+]] = "onnx.Add"(%[[DIV]], %[[ZP]])
// CHECK: %[[RND:[0-9]+]] = "onnx.Round"(%[[ADD]])
// CHECK: %[[CLP:[0-9]+]] = "onnx.Clip"(%[[RND]], %[[QMIN]], %[[QMAX]])
// CHECK: %[[SUB:[0-9]+]] = "onnx.Sub"(%[[CLP]], %[[ZP]])
// CHECK: %[[MUL:[0-9]+]] = "onnx.Mul"(%[[SUB]], %[[S]])
// CHECK: %[[MP:[0-9]+]] = "onnx.MaxPoolSingleOut"(%[[MUL]]) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], storage_order = 0 : si64, strides = [2, 2]}> {onnx_node_name = "MP_INT8"} : (tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32>
// CHECK: %[[YS:[0-9]+]] = onnx.Constant dense<2.000000e-02> : tensor<f32>
// CHECK: %[[YZP:[0-9]+]] = onnx.Constant dense<-2.000000e+00> : tensor<f32>
// CHECK: %[[YQMIN:[0-9]+]] = onnx.Constant dense<-1.280000e+02> : tensor<f32>
// CHECK: %[[YQMAX:[0-9]+]] = onnx.Constant dense<1.270000e+02> : tensor<f32>
// CHECK: %[[YDIV:[0-9]+]] = "onnx.Div"(%[[MP]], %[[YS]])
// CHECK: %[[YADD:[0-9]+]] = "onnx.Add"(%[[YDIV]], %[[YZP]])
// CHECK: %[[YRND:[0-9]+]] = "onnx.Round"(%[[YADD]])
// CHECK: %[[YCLP:[0-9]+]] = "onnx.Clip"(%[[YRND]], %[[YQMIN]], %[[YQMAX]])
// CHECK: %[[YSUB:[0-9]+]] = "onnx.Sub"(%[[YCLP]], %[[YZP]])
// CHECK: %[[OUT:[0-9]+]] = "onnx.Mul"(%[[YSUB]], %[[YS]])
// CHECK: return %[[OUT]]

// -----

// Non-default geometry: padding, a larger kernel, an overlapping stride and
// ceil_mode must all be carried over verbatim (bf16 shown: a cast bracket).
func.func @maxpool_attrs_preserved(%arg0: tensor<1x1x7x7xf32>) -> tensor<1x1x4x4xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {onnx_node_name = "MP_ATTRS", kernel_shape = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1], ceil_mode = 1 : si64} : (tensor<1x1x7x7xf32>) -> tensor<1x1x4x4xf32>
  return %0 : tensor<1x1x4x4xf32>
}
// CHECK-LABEL: func.func @maxpool_attrs_preserved
// CHECK: %[[D:[0-9]+]] = "onnx.Cast"(%arg0) <{saturate = 1 : si64, to = bf16}> : (tensor<1x1x7x7xf32>) -> tensor<1x1x7x7xbf16>
// CHECK: %[[U:[0-9]+]] = "onnx.Cast"(%[[D]]) <{saturate = 1 : si64, to = f32}> : (tensor<1x1x7x7xbf16>) -> tensor<1x1x7x7xf32>
// CHECK: %[[MP:[0-9]+]] = "onnx.MaxPoolSingleOut"(%[[U]]) <{auto_pad = "NOTSET", ceil_mode = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> {onnx_node_name = "MP_ATTRS"}
// CHECK: %[[YD:[0-9]+]] = "onnx.Cast"(%[[MP]]) <{saturate = 1 : si64, to = bf16}>
// CHECK: %[[YU:[0-9]+]] = "onnx.Cast"(%[[YD]]) <{saturate = 1 : si64, to = f32}>
// CHECK: return %[[YU]]

// -----

// Not named in MIXEDP_NODE_FORMATS: left exactly as it was.
func.func @maxpool_unselected(%arg0: tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {onnx_node_name = "MP_NOT_LISTED", kernel_shape = [2, 2], strides = [2, 2]} : (tensor<1x2x4x4xf32>) -> tensor<1x2x2x2xf32>
  return %0 : tensor<1x2x2x2xf32>
}
// CHECK-LABEL: func.func @maxpool_unselected
// CHECK-NEXT: %[[R:[0-9]+]] = "onnx.MaxPoolSingleOut"(%arg0)
// CHECK-NEXT: return %[[R]]
