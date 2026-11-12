/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- MixedPrecisionBackward.hpp - reverse-mode ONNX gradients -----===//
//
// Reverse-mode differentiation of an FP32 ONNX-dialect function body, emitted
// as ONNX ops in the same function: the gradients a gradient-based
// mixed-precision criterion needs are then computed by the compiled model
// itself, with no external autodiff framework. See MixedPrecisionBackward.cpp
// for the supported ops and their vector-Jacobian products.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

#include <utility>

namespace onnx_mlir {

struct BackwardGradients {
  // d(loss)/d(v) for every active value v the loss depends on, summed over all
  // of v's consumers. A value missing here has no gradient path to the loss.
  llvm::DenseMap<mlir::Value, mlir::Value> values;
  // d(loss)/d(operand) of one op, for the ops in `weightGradOps` (Conv, Gemm,
  // MatMul): operand 1 is the weight (a weight shared by several ops gets one
  // entry per op); operand 0, when active, is the input as seen by this op
  // alone -- its own contribution to the input's gradient, i.e. the gradient
  // of a perturbation of the input applied on this consumer's edge only (the
  // mixed-precision pass quantizes each consumer's copy of a shared input).
  llvm::DenseMap<std::pair<mlir::Operation *, unsigned>, mlir::Value> operands;
};

// Emits, at the builder's insertion point (which must be after every op of
// `func`'s body except the terminator), the ops computing the gradients of a
// loss with d(loss)/d(root) = seed. Active values are those derived from a
// float function argument; constants and shape computations are not. For each
// op in `weightGradOps` (Conv/Gemm/MatMul with a compile-time weight operand 1)
// the gradient of that weight operand is computed too. Emits an error and
// fails on an op on the gradient path that has no rule, or on a value whose
// shape is not static.
mlir::FailureOr<BackwardGradients> buildBackward(mlir::OpBuilder &builder,
    mlir::func::FuncOp func, mlir::Value root, mlir::Value seed,
    const llvm::DenseSet<mlir::Operation *> &weightGradOps);

} // namespace onnx_mlir
