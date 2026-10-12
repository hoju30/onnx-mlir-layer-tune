// RUN: onnx-mlir-opt --emit-bytecode %s -o %t.mlirbc
// RUN: head -c 4 %t.mlirbc | od -c | FileCheck %s --check-prefix=MAGIC
// RUN: onnx-mlir-opt %t.mlirbc | FileCheck %s
// RUN: onnx-mlir-opt %s -o %t.text.mlir
// RUN: onnx-mlir-opt %t.mlirbc -o %t.bytecode.mlir
// RUN: diff %t.text.mlir %t.bytecode.mlir

// onnx-mlir-opt --emit-bytecode writes MLIR bytecode (magic "ML\xefR"), which
// onnx-mlir-opt reads back without a flag. A weight held as a dense_resource
// blob keeps its bytes, and the bytecode printed as text is the text path's
// output exactly.
func.func @main_graph(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense_resource<w> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "M"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  onnx.Return %0 : tensor<1x2xf32>
}
{-#
  dialect_resources: {
    builtin: {
      w: "0x040000000000803F000000400000404000008040"
    }
  }
#-}

// MAGIC: M L 357 R
// CHECK: onnx.Constant dense_resource<w> : tensor<2x2xf32>
// CHECK: "onnx.MatMul"
// CHECK: w: "0x040000000000803F000000400000404000008040"
