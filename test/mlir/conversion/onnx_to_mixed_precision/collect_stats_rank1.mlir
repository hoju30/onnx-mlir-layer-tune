// RUN: echo '{"R": {"x": [-1.0, 1.0], "y": [0.0, 1.0]}}' > %t.ranges.json
// RUN: env MIXEDP_STATS_MODE=grads MIXEDP_STATS_LAYOUT=%t.g.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s --check-prefix=GRADS
// RUN: env MIXEDP_STATS_MODE=errors MIXEDP_STATS_FORMATS=int8,p8e1 MIXEDP_STATS_RANGES=%t.ranges.json MIXEDP_STATS_LAYOUT=%t.e.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s --check-prefix=ERRORS
// RUN: env MIXEDP_STATS_MODE=criterion MIXEDP_STATS_FORMATS=int8,p8e1 MIXEDP_STATS_RANGES=%t.ranges.json MIXEDP_STATS_LAYOUT=%t.c.json onnx-mlir-opt --mixedp-collect-stats %s | FileCheck %s --check-prefix=CRIT

// Rank-1 points: the channel axis is the only axis, so the per-channel sums
// are the tensors themselves (no ReduceSum, which with an empty axes list
// would reduce everything and used to crash shape inference). 2 points of 4
// channels: grads 2 x 2 x 4, errors 2 formats x 2 x 2 x 4, criterion both.
func.func @main_graph(%arg0: tensor<4xf32>) -> tensor<4xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "R"} : (tensor<4xf32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
// GRADS: -> (tensor<4xf32>, tensor<16xf32> {onnx.name = "mixedp_stats"})
// The only ReduceSum is the seed's row mask over the one-hot.
// GRADS: "onnx.ReduceSum"(%arg1,
// GRADS-NOT: "onnx.ReduceSum"
// ERRORS: -> (tensor<4xf32>, tensor<32xf32> {onnx.name = "mixedp_stats"})
// ERRORS-NOT: "onnx.ReduceSum"
// CRIT: -> (tensor<4xf32>, tensor<48xf32> {onnx.name = "mixedp_stats"})
