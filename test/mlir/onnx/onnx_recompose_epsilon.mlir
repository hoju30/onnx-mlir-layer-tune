// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// The recomposed (RMS) LayerNormalization keeps the epsilon of the pattern's
// constant, which exporters write as a scalar dense tensor. Epsilons other
// than the op's default (1e-5): Qwen3's RMSNorm (1e-6, x * (1 / sqrt) * w) and
// BERT's LayerNorm (1e-12).

// -----

func.func @rms_norm_eps_1e6(%x: tensor<1x128x1024xf32>, %scale: tensor<1024xf32>) -> (tensor<1x128x1024xf32>) {
  %eps = onnx.Constant dense<9.99999997E-7> : tensor<f32>
  %one = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %dd = "onnx.Mul"(%x, %x) : (tensor<1x128x1024xf32>, tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64} : (tensor<1x128x1024xf32>) -> tensor<1x128x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x128x1xf32>, tensor<f32>) -> tensor<1x128x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %InvStdDev = "onnx.Div"(%one, %StdDev) : (tensor<f32>, tensor<1x128x1xf32>) -> tensor<1x128x1xf32>
  %Norm = "onnx.Mul"(%x, %InvStdDev) : (tensor<1x128x1024xf32>, tensor<1x128x1xf32>) -> tensor<1x128x1024xf32>
  %Y = "onnx.Mul"(%scale, %Norm) : (tensor<1024xf32>, tensor<1x128x1024xf32>) -> tensor<1x128x1024xf32>
  return %Y : tensor<1x128x1024xf32>
// CHECK-LABEL:  func.func @rms_norm_eps_1e6
// CHECK:          "onnx.RMSLayerNormalization"({{.*}}epsilon = 9.99999997E-7 : f32
}

// -----

func.func @layer_norm_eps_1e12(%x: tensor<1x64x768xf32>, %scale: tensor<768xf32>, %bias: tensor<768xf32>) -> (tensor<1x64x768xf32>) {
  %eps = onnx.Constant dense<9.99999996E-13> : tensor<f32>
  %mean = "onnx.ReduceMeanV13"(%x) {axes = [-1], keepdims = 1 : si64} : (tensor<1x64x768xf32>) -> tensor<1x64x1xf32>
  %d = "onnx.Sub"(%x, %mean) : (tensor<1x64x768xf32>, tensor<1x64x1xf32>) -> tensor<1x64x768xf32>
  %dd = "onnx.Mul"(%d, %d) : (tensor<1x64x768xf32>, tensor<1x64x768xf32>) -> tensor<1x64x768xf32>
  %var = "onnx.ReduceMeanV13"(%dd) {axes = [-1], keepdims = 1 : si64} : (tensor<1x64x768xf32>) -> tensor<1x64x1xf32>
  %varEps = "onnx.Add"(%var, %eps) : (tensor<1x64x1xf32>, tensor<f32>) -> tensor<1x64x1xf32>
  %StdDev = "onnx.Sqrt"(%varEps) : (tensor<1x64x1xf32>) -> tensor<1x64x1xf32>
  %Norm = "onnx.Div"(%d, %StdDev) : (tensor<1x64x768xf32>, tensor<1x64x1xf32>) -> tensor<1x64x768xf32>
  %NormScaled = "onnx.Mul"(%Norm, %scale) : (tensor<1x64x768xf32>, tensor<768xf32>) -> tensor<1x64x768xf32>
  %Y = "onnx.Add"(%NormScaled, %bias) : (tensor<1x64x768xf32>, tensor<768xf32>) -> tensor<1x64x768xf32>
  return %Y : tensor<1x64x768xf32>
// CHECK-LABEL:  func.func @layer_norm_eps_1e12
// CHECK:          "onnx.LayerNormalization"({{.*}}epsilon = 9.99999996E-13 : f32
}
