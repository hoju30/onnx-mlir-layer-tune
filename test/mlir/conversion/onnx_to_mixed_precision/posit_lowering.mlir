// RUN: env MIXEDP_NODE_FORMATS="R_P4:a=p4e1,R_P6:a=p6e1,R_P8:a=p8e1,MM_P4W:w=p4e1" onnx-mlir-opt --convert-onnx-to-mixed-precision --canonicalize --convert-onnx-to-krnl --convert-posit-to-krnl --canonicalize %s -split-input-file 2>&1 | FileCheck %s --implicit-check-not=warning --implicit-check-not="call @posit_"
// RUN: env MIXEDP_POSIT_ACTIVATION=runtime MIXEDP_NODE_FORMATS="R_P4:a=p4e1,R_P6:a=p6e1,R_P8:a=p8e1,MM_P4W:w=p4e1" onnx-mlir-opt --convert-onnx-to-mixed-precision --canonicalize --convert-onnx-to-krnl --convert-posit-to-krnl --canonicalize %s -split-input-file 2>&1 | FileCheck %s --check-prefix=RUNTIME --implicit-check-not=warning

// Regression test for the most serious bug found in this pass: after the full
// lowering pipeline (documented in docs/MixedPrecisionFormats.md), the posit
// activation round trip must still be there.
//
// Default path (CHECK): each posit activation is one posit.round_f32, lowered
// to an inlined table lookup -- two krnl.global tables (thresholds padded to
// 2^k - 1 entries, values) and a loop that reads the INPUT and writes its own
// buffer; the Relu loop reads that buffer, never the raw input, and no
// posit_* runtime function is called anywhere (--implicit-check-not).
//
// MIXEDP_POSIT_ACTIVATION=runtime (RUNTIME): from_f32/to_f32 runtime calls.
// --convert-posit-to-krnl folds a directly-adjacent to_f32(from_f32(x)) to x
// unless the from_f32 carries a nonzero qalign_key; before the key was set,
// the compiled model was bit-identical to plain FP32 (the quantization was
// silently a no-op). What must hold after lowering, for input %x and the
// original Relu:
//   1. posit_from_f32 then posit_to_f32 are called on the INPUT,
//   2. the Relu loop reads the to_f32 OUTPUT buffer, never the raw input,
//   3. the result gets its own from_f32/to_f32 pair.

func.func @posit_activation_survives_lowering_p8e1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_P8"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @posit_activation_survives_lowering_p8e1
// CHECK-SAME: (%[[X:[a-z0-9_]+]]: memref<4xf32>)
// CHECK: %[[DQ:[a-z_0-9]+]] = memref.alloc() : memref<4xf32>
// CHECK: "krnl.global"() <{name = "posit_round_thresholds_{{[0-9]+}}", shape = [255]
// CHECK: "krnl.global"() <{name = "posit_round_values_{{[0-9]+}}", shape = [255]
// CHECK: scf.for
// CHECK: memref.load %[[X]]
// CHECK: memref.store %{{[0-9]+}}, %[[DQ]]
// CHECK: krnl.iterate
// CHECK-NOT: krnl.load %[[X]]
// CHECK: krnl.load %[[DQ]]
// CHECK: "krnl.global"() <{name = "posit_round_thresholds_{{[0-9]+}}", shape = [255]
// CHECK: scf.for
// CHECK: return

// RUNTIME-LABEL: func.func @posit_activation_survives_lowering_p8e1
// RUNTIME-SAME: (%[[X:[a-z0-9_]+]]: memref<4xf32>)
// RUNTIME: %[[XC:[a-z_0-9]+]] = memref.cast %[[X]] : memref<4xf32> to memref<*xf32>
// RUNTIME: call @posit_from_f32_p8e1(%[[XC]], %{{[a-z_0-9]+}}, %{{[a-z_0-9]+}})
// RUNTIME: %[[DQ:[a-z_0-9]+]] = memref.alloc() : memref<4xf32>
// RUNTIME: %[[DQC:[a-z_0-9]+]] = memref.cast %[[DQ]] : memref<4xf32> to memref<*xf32>
// RUNTIME: call @posit_to_f32_p8e1(%{{[a-z_0-9]+}}, %[[DQC]])
// RUNTIME: krnl.iterate
// RUNTIME-NOT: krnl.load %[[X]]
// RUNTIME: krnl.load %[[DQ]]
// RUNTIME: call @posit_from_f32_p8e1
// RUNTIME: call @posit_to_f32_p8e1
// RUNTIME: return

// -----

// Same property for the other two supported formats (runtime symbol names are
// derived from (nbits, es): posit_{from,to}_f32_p<N>e<ES>).
func.func @posit_activation_survives_lowering_p4e1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_P4"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @posit_activation_survives_lowering_p4e1
// CHECK-SAME: (%[[X:[a-z0-9_]+]]: memref<4xf32>)
// CHECK: %[[DQ:[a-z_0-9]+]] = memref.alloc() : memref<4xf32>
// CHECK: "krnl.global"() <{name = "posit_round_thresholds_{{[0-9]+}}", shape = [15]
// CHECK: "krnl.global"() <{name = "posit_round_values_{{[0-9]+}}", shape = [15]
// CHECK: scf.for
// CHECK: memref.load %[[X]]
// CHECK: memref.store %{{[0-9]+}}, %[[DQ]]
// CHECK: krnl.iterate
// CHECK-NOT: krnl.load %[[X]]
// CHECK: krnl.load %[[DQ]]
// CHECK: "krnl.global"() <{name = "posit_round_thresholds_{{[0-9]+}}", shape = [15]
// CHECK: scf.for
// CHECK: return

// RUNTIME-LABEL: func.func @posit_activation_survives_lowering_p4e1
// RUNTIME-SAME: (%[[X:[a-z0-9_]+]]: memref<4xf32>)
// RUNTIME: call @posit_from_f32_p4e1
// RUNTIME: %[[DQ:[a-z_0-9]+]] = memref.alloc() : memref<4xf32>
// RUNTIME: call @posit_to_f32_p4e1
// RUNTIME-NOT: krnl.load %[[X]]
// RUNTIME: krnl.load %[[DQ]]
// RUNTIME: call @posit_from_f32_p4e1
// RUNTIME: call @posit_to_f32_p4e1

// -----

func.func @posit_activation_survives_lowering_p6e1(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R_P6"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// CHECK-LABEL: func.func @posit_activation_survives_lowering_p6e1
// CHECK-SAME: (%[[X:[a-z0-9_]+]]: memref<4xf32>)
// CHECK: %[[DQ:[a-z_0-9]+]] = memref.alloc() : memref<4xf32>
// CHECK: "krnl.global"() <{name = "posit_round_thresholds_{{[0-9]+}}", shape = [63]
// CHECK: "krnl.global"() <{name = "posit_round_values_{{[0-9]+}}", shape = [63]
// CHECK: scf.for
// CHECK: memref.load %[[X]]
// CHECK: memref.store %{{[0-9]+}}, %[[DQ]]
// CHECK: krnl.iterate
// CHECK-NOT: krnl.load %[[X]]
// CHECK: krnl.load %[[DQ]]
// CHECK: "krnl.global"() <{name = "posit_round_thresholds_{{[0-9]+}}", shape = [63]
// CHECK: scf.for
// CHECK: return

// RUNTIME-LABEL: func.func @posit_activation_survives_lowering_p6e1
// RUNTIME-SAME: (%[[X:[a-z0-9_]+]]: memref<4xf32>)
// RUNTIME: call @posit_from_f32_p6e1
// RUNTIME: %[[DQ:[a-z_0-9]+]] = memref.alloc() : memref<4xf32>
// RUNTIME: call @posit_to_f32_p6e1
// RUNTIME-NOT: krnl.load %[[X]]
// RUNTIME: krnl.load %[[DQ]]
// RUNTIME: call @posit_from_f32_p6e1
// RUNTIME: call @posit_to_f32_p6e1

// -----

// A posit WEIGHT is folded into the constant at compile time, so it must not
// introduce any runtime dependency on the posit library: no posit_* call
// anywhere in the lowered function.
func.func @posit_weight_needs_no_runtime(%arg0: tensor<1x2xf32>) -> tensor<1x2xf32> {
  %w = onnx.Constant dense<[[0.1, -0.2], [0.4, -0.8]]> : tensor<2x2xf32>
  %0 = "onnx.MatMul"(%arg0, %w) {onnx_node_name = "MM_P4W"} : (tensor<1x2xf32>, tensor<2x2xf32>) -> tensor<1x2xf32>
  return %0 : tensor<1x2xf32>
}
// CHECK-LABEL: func.func @posit_weight_needs_no_runtime
// CHECK-NOT: posit_
// CHECK: return
// RUNTIME-LABEL: func.func @posit_weight_needs_no_runtime
// RUNTIME-NOT: posit_
// RUNTIME: return
