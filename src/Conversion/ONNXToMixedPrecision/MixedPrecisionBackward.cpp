/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- MixedPrecisionBackward.cpp - reverse-mode ONNX gradients -----===//
//
// Reverse-mode differentiation of an FP32 ONNX-dialect function body, as ONNX
// ops (so the result is lowered and compiled like any other ONNX graph). The
// body is one block in topological order; the sweep visits it backwards,
// applies each op's vector-Jacobian product to the gradient of its result and
// accumulates the operand gradients (a value with several consumers gets the
// sum, i.e. its total gradient).
//
// Supported ops (all shapes static; Conv batch 1, 2-D, group 1 or depthwise
// for the weight gradient):
//   Conv      dX = Conv(zero-inserted dY, spatially flipped and regrouped W)
//             dW = Conv of X with dY as the kernel (strides <-> dilations
//                  swapped; MatMul for an unpadded 1x1 conv)
//             dB = dY summed over batch and space, when the bias is computed
//                  at run time
//   Gemm (2-D), MatMul (rank >= 2, batched and broadcasting)   the usual
//                        transposed products, summed back over broadcast batch
//                        axes (and beta * dY for a Gemm C computed at run time)
//   Add, Sub, Mul        with broadcasting (gradients summed back)
//   Relu, Clip           dY masked by x > 0 / min < x < max (strict, as
//                        PyTorch's threshold/hardtanh backward)
//   MaxPool (2-D)        dY to the first maximum of each window (row-major,
//                        the PyTorch / ONNX Runtime argmax convention)
//   Sigmoid, Tanh, Sqrt, Erf, Neg, Pow (constant exponent)   elementwise
//   Div, Where (on its two data operands), Concat, Split (multi-result),
//   Gather of an activation at a constant index or contiguous index range
//   Softmax              y (dY - sum(dY y)) along the axis
//   Transpose            dY transposed back
//   ReduceMean, GlobalAveragePool   dY broadcast back, divided by the count
//             (ReduceMean with no axes and noop_with_empty_axes = 1: dY)
//   Reshape, Flatten, Identity      dY reshaped to the input shape
// Weights are compile-time constants: their gradients are only computed where
// requested (weightGradOps), and the conv input gradient uses the constant W.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToMixedPrecision/MixedPrecisionBackward.hpp"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;

namespace onnx_mlir {
namespace {

using Shape = SmallVector<int64_t, 4>;

static std::optional<Shape> staticShape(Value v) {
  auto t = llvm::dyn_cast<RankedTensorType>(v.getType());
  if (!t || !t.hasStaticShape())
    return std::nullopt;
  return Shape(t.getShape());
}

static int64_t numElems(ArrayRef<int64_t> s) {
  int64_t n = 1;
  for (int64_t d : s)
    n *= d;
  return n;
}

static bool isF32Tensor(Value v) {
  auto t = llvm::dyn_cast<ShapedType>(v.getType());
  return t && t.getElementType().isF32();
}

static ElementsAttr constantAttr(Value v) {
  Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return nullptr;
  return llvm::dyn_cast_or_null<ElementsAttr>(def->getAttr("value"));
}

static int64_t intAttr(Operation *op, StringRef name, int64_t dflt) {
  if (auto a = op->getAttrOfType<IntegerAttr>(name))
    return a.getValue().getSExtValue();
  return dflt;
}

static double floatAttr(Operation *op, StringRef name, double dflt) {
  if (auto a = op->getAttrOfType<FloatAttr>(name))
    return a.getValueAsDouble();
  return dflt;
}

static Shape intsAttr(Operation *op, StringRef name, Shape dflt) {
  auto a = op->getAttrOfType<ArrayAttr>(name);
  if (!a)
    return dflt;
  Shape out;
  for (Attribute e : a)
    out.push_back(llvm::cast<IntegerAttr>(e).getValue().getSExtValue());
  return out;
}

// Thin layer over OnnxBuilder that keeps every result type static.
struct Emitter {
  Emitter(OpBuilder &b, Location loc) : b(b), loc(loc), create(b, loc) {}
  OpBuilder &b;
  Location loc;
  OnnxBuilder create;

  RankedTensorType f32(ArrayRef<int64_t> s) {
    return RankedTensorType::get(s, b.getF32Type());
  }
  Shape shape(Value v) { return *staticShape(v); }
  Value i64(ArrayRef<int64_t> v) { return create.constantInt64(v); }
  Value scalar(float v) {
    return create.constant(
        DenseElementsAttr::get(RankedTensorType::get({}, b.getF32Type()), v));
  }
  Value constant(ArrayRef<int64_t> s, ArrayRef<float> values) {
    return create.constant(DenseElementsAttr::get(f32(s), values));
  }
  Value reshape(Value v, ArrayRef<int64_t> s) {
    return shape(v) == Shape(s) ? v : create.reshape(f32(s), v, i64(s));
  }
  Value transpose(Value v, ArrayRef<int64_t> perm) {
    Shape in = shape(v), out;
    for (int64_t p : perm)
      out.push_back(in[p]);
    return create.transpose(f32(out), v, b.getI64ArrayAttr(perm));
  }
  Value add(Value x, Value y) { return create.add(x, y); }
  Value sub(Value x, Value y) { return create.sub(x, y); }
  Value mul(Value x, Value y) { return create.mul(x, y); }
  // ONNX MatMul of operands of rank >= 2: batch dims broadcast.
  Value matmul(Value x, Value y) {
    Shape a = shape(x), c = shape(y), out;
    int64_t ba = a.size() - 2, bc = c.size() - 2, nb = std::max(ba, bc);
    for (int64_t d = 0; d < nb; ++d) {
      int64_t da = d - (nb - ba) >= 0 ? a[d - (nb - ba)] : 1;
      int64_t dc = d - (nb - bc) >= 0 ? c[d - (nb - bc)] : 1;
      out.push_back(da == 1 ? dc : da);
    }
    out.push_back(a[a.size() - 2]);
    out.push_back(c.back());
    return create.matmul(f32(out), x, y);
  }
  // x @ w^T for a 2-D w, as a Gemm with transB = 1 on x flattened to 2-D: no
  // transposed copy of w is materialized (for a constant w, a Transpose would
  // be folded into a second copy of the weight -- GPT-2's would then no longer
  // fit the 2 GiB a single .ll file may have).
  Value matmulBT(Value x, Value w) {
    Shape xs = shape(x), ws = shape(w);
    if (ws.size() != 2)
      return matmul(x, swapLast2(w));
    int64_t rows = 1;
    for (size_t d = 0; d + 1 < xs.size(); ++d)
      rows *= xs[d];
    Value x2 = reshape(x, {rows, xs.back()});
    IntegerType si64 = b.getIntegerType(64, /*isSigned=*/true);
    Value y2 = create
                   .createTypedOpAndInferShapes<ONNXGemmOp>(f32({rows, ws[0]}),
                       x2, w, create.none(), b.getF32FloatAttr(1.0f),
                       b.getF32FloatAttr(1.0f), b.getIntegerAttr(si64, 0),
                       b.getIntegerAttr(si64, 1))
                   .getResult();
    Shape out(xs);
    out.back() = ws[0];
    return reshape(y2, out);
  }
  Value swapLast2(Value v) {
    SmallVector<int64_t, 4> perm;
    int64_t r = shape(v).size();
    for (int64_t d = 0; d < r; ++d)
      perm.push_back(d);
    std::swap(perm[r - 2], perm[r - 1]);
    return transpose(v, perm);
  }
  Value div(Value x, Value y) {
    return create.div(x, y);
  }
  template <typename UnaryOp>
  Value unary(Value x) {
    return create.createTypedOpAndInferShapes<UnaryOp>(f32(shape(x)), x)
        .getResult();
  }
  Value pow(Value x, float p) {
    return create
        .createTypedOpAndInferShapes<ONNXPowOp>(f32(shape(x)), x, scalar(p))
        .getResult();
  }
  Value reduceSum(Value v, ArrayRef<int64_t> axes, bool keep) {
    Shape in = shape(v), out;
    for (int64_t d = 0; d < static_cast<int64_t>(in.size()); ++d)
      if (!llvm::is_contained(axes, d))
        out.push_back(in[d]);
      else if (keep)
        out.push_back(1);
    return create.reduceSum(f32(out), v, i64(axes), keep);
  }
  Value expand(Value v, ArrayRef<int64_t> s) {
    return create.expand(f32(s), v, i64(s));
  }
  Value mask(Value cond, Value v) { // v where cond, else 0
    return create.where(f32(shape(v)), cond, v, scalar(0.0f));
  }
  Value maskNot(Value cond, Value v) { // 0 where cond, else v
    return create.where(f32(shape(v)), cond, scalar(0.0f), v);
  }
  Value concat(ArrayRef<Value> parts, int64_t axis) {
    Shape out = shape(parts.front());
    out[axis] = 0;
    for (Value v : parts)
      out[axis] += shape(v)[axis];
    return create.concat(f32(out), parts, axis);
  }
  template <typename CmpOp>
  Value compare(Value x, Value y) {
    return create
        .createTypedOpAndInferShapes<CmpOp>(
            RankedTensorType::get(shape(x), b.getI1Type()), x, y)
        .getResult();
  }
  Value andOp(Value x, Value y) {
    return create
        .createTypedOpAndInferShapes<ONNXAndOp>(
            RankedTensorType::get(shape(x), b.getI1Type()), x, y)
        .getResult();
  }
  Value orOp(Value x, Value y) {
    return create
        .createTypedOpAndInferShapes<ONNXOrOp>(
            RankedTensorType::get(shape(x), b.getI1Type()), x, y)
        .getResult();
  }
  Value notOp(Value x) {
    return create
        .createTypedOpAndInferShapes<ONNXNotOp>(
            RankedTensorType::get(shape(x), b.getI1Type()), x)
        .getResult();
  }
  Value padZero(Value v, ArrayRef<int64_t> pads) { return padValue(v, pads, 0.0f); }
  Value padValue(Value v, ArrayRef<int64_t> pads, float value) {
    Shape in = shape(v), out(in);
    int64_t r = in.size();
    for (int64_t d = 0; d < r; ++d)
      out[d] += pads[d] + pads[r + d];
    Value p = value == 0.0f ? create.padZero(v, i64(pads))
                            : create.pad(v, i64(pads), scalar(value));
    return p.getType() == Type(f32(out)) ? p : create.reshape(f32(out), p, i64(out));
  }
  // v[starts[d] : ends[d] : steps[d]] on every axis (steps default 1).
  Value slice(Value v, ArrayRef<int64_t> starts, ArrayRef<int64_t> ends,
      ArrayRef<int64_t> steps = {}) {
    Shape in = shape(v), out, axes, st;
    for (size_t d = 0; d < starts.size(); ++d) {
      int64_t step = steps.empty() ? 1 : steps[d];
      out.push_back((ends[d] - starts[d] + step - 1) / step);
      axes.push_back(d);
      st.push_back(step);
    }
    if (out == in)
      return v;
    return create.slice(f32(out), v, i64(starts), i64(ends), i64(axes), i64(st));
  }
  Value conv(Value x, Value w, int64_t group, ArrayRef<int64_t> strides,
      ArrayRef<int64_t> dilations, ArrayRef<int64_t> pads) {
    Shape xs = shape(x), ws = shape(w), out{xs[0], ws[0]};
    for (int i = 0; i < 2; ++i)
      out.push_back((xs[2 + i] + pads[i] + pads[2 + i] -
                        dilations[i] * (ws[2 + i] - 1) - 1) /
                        strides[i] +
                    1);
    return create.conv(f32(out), x, w, create.none(), "NOTSET", dilations,
        group, ArrayRef<int64_t>(ws).drop_front(2), pads, strides);
  }
  // Sum `g` (shaped like a broadcast result) back to `target`.
  Value unbroadcast(Value g, ArrayRef<int64_t> target) {
    Shape gs = shape(g);
    if (gs == Shape(target))
      return g;
    int64_t lead = gs.size() - target.size();
    Shape axes;
    for (int64_t d = 0; d < static_cast<int64_t>(gs.size()); ++d)
      if (d < lead || (target[d - lead] == 1 && gs[d] != 1))
        axes.push_back(d);
    return reshape(reduceSum(g, axes, /*keep=*/true), target);
  }
};

struct Sweep {
  Emitter &e;
  const llvm::DenseSet<Operation *> &weightGradOps;
  llvm::DenseSet<Value> active;
  BackwardGradients out;

  bool isActive(Value v) { return v && active.contains(v); }
  void accumulate(Value v, Value g) {
    auto [it, inserted] = out.values.try_emplace(v, g);
    if (!inserted)
      it->second = e.add(it->second, g);
  }

  LogicalResult visit(Operation *op, Value gy);
  LogicalResult conv(Operation *op, Value gy);
  LogicalResult gemm(Operation *op, Value gy);
  LogicalResult matmul(Operation *op, Value gy);
  LogicalResult maxPool(Operation *op, Value gy);
  LogicalResult split(Operation *op, ArrayRef<Value> gys);
  LogicalResult reduceMean(Operation *op, Value gy);
};

// [N, C, H, W] -> [N, C, (H-1)*sh+1, (W-1)*sw+1] with strides-1 zeros between
// the elements: dY placed at the input positions a strided window starts at.
static Value insertStrideZeros(Emitter &e, Value g, ArrayRef<int64_t> strides) {
  if (strides[0] == 1 && strides[1] == 1)
    return g;
  Shape s = e.shape(g);
  Value r = e.reshape(g, {s[0], s[1], s[2], 1, s[3], 1});
  r = e.padZero(r, {0, 0, 0, 0, 0, 0, 0, 0, 0, strides[0] - 1, 0, strides[1] - 1});
  r = e.reshape(r, {s[0], s[1], s[2] * strides[0], s[3] * strides[1]});
  return e.slice(r, {0, 0, 0, 0},
      {s[0], s[1], (s[2] - 1) * strides[0] + 1, (s[3] - 1) * strides[1] + 1});
}

// dX of Conv: correlate the stride-dilated dY with W flipped in space and with
// input/output channels swapped inside each group, padded so the result has
// X's spatial size.
static Value convInputGrad(Emitter &e, Value gy, ElementsAttr wAttr,
    ArrayRef<int64_t> xs, int64_t group, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> dilations, ArrayRef<int64_t> pads) {
  Shape ys = e.shape(gy);
  auto wTy = llvm::cast<ShapedType>(wAttr.getType());
  int64_t M = wTy.getDimSize(0), Cg = wTy.getDimSize(1), KH = wTy.getDimSize(2),
          KW = wTy.getDimSize(3), Mg = M / group, C = Cg * group;

  // W'[g*Cg + c][m][KH-1-kh][KW-1-kw] = W[g*Mg + m][c][kh][kw].
  SmallVector<float> w(wAttr.getValues<float>());
  SmallVector<float> wf(w.size());
  for (int64_t g = 0; g < group; ++g)
    for (int64_t m = 0; m < Mg; ++m)
      for (int64_t c = 0; c < Cg; ++c)
        for (int64_t kh = 0; kh < KH; ++kh)
          for (int64_t kw = 0; kw < KW; ++kw)
            wf[(((g * Cg + c) * Mg + m) * KH + (KH - 1 - kh)) * KW +
                (KW - 1 - kw)] = w[(((g * Mg + m) * Cg + c) * KH + kh) * KW + kw];
  Value wFlip = e.constant({C, Mg, KH, KW}, wf);

  // Insert strides-1 zeros between the dY elements.
  Value up = insertStrideZeros(e, gy, strides);
  int64_t uh = (ys[2] - 1) * strides[0] + 1, uw = (ys[3] - 1) * strides[1] + 1;
  // Full-correlation padding back to X's size: begin (K-1)d - p_begin, end
  // H - Hu + p_begin; a negative amount crops instead.
  int64_t K[2] = {KH, KW}, U[2] = {uh, uw};
  int64_t pb[2], pe[2];
  for (int i = 0; i < 2; ++i) {
    pb[i] = (K[i] - 1) * dilations[i] - pads[i];
    pe[i] = xs[2 + i] - U[i] + pads[i];
  }
  if (pb[0] < 0 || pb[1] < 0 || pe[0] < 0 || pe[1] < 0) {
    up = e.padZero(up, {0, 0, std::max<int64_t>(pb[0], 0),
                           std::max<int64_t>(pb[1], 0), 0, 0,
                           std::max<int64_t>(pe[0], 0),
                           std::max<int64_t>(pe[1], 0)});
    Shape us = e.shape(up);
    up = e.slice(up, {0, 0, std::max<int64_t>(-pb[0], 0), std::max<int64_t>(-pb[1], 0)},
        {us[0], us[1], us[2] - std::max<int64_t>(-pe[0], 0),
            us[3] - std::max<int64_t>(-pe[1], 0)});
    return e.conv(up, wFlip, group, {1, 1}, dilations, {0, 0, 0, 0});
  }
  return e.conv(up, wFlip, group, {1, 1}, dilations, {pb[0], pb[1], pe[0], pe[1]});
}

LogicalResult Sweep::conv(Operation *op, Value gy) {
  Value X = op->getOperand(0), W = op->getOperand(1);
  std::optional<Shape> xs = staticShape(X), ws = staticShape(W);
  StringRef autoPad = "NOTSET";
  if (auto a = op->getAttrOfType<StringAttr>("auto_pad"))
    autoPad = a.getValue();
  if (!xs || !ws || xs->size() != 4 || (*xs)[0] != 1 ||
      (autoPad != "NOTSET" && autoPad != "VALID"))
    return op->emitError("backward: Conv needs a static 2-D batch-1 input "
                         "and auto_pad NOTSET/VALID");
  int64_t group = intAttr(op, "group", 1);
  Shape strides = intsAttr(op, "strides", {1, 1});
  Shape dilations = intsAttr(op, "dilations", {1, 1});
  Shape pads = autoPad == "VALID" ? Shape{0, 0, 0, 0}
                                  : intsAttr(op, "pads", {0, 0, 0, 0});
  int64_t C = (*xs)[1], M = (*ws)[0];

  if (isActive(X)) {
    ElementsAttr wAttr = constantAttr(W);
    if (!wAttr || !isF32Tensor(W))
      return op->emitError("backward: Conv input gradient needs a constant "
                           "f32 weight");
    accumulate(X, convInputGrad(e, gy, wAttr, *xs, group, strides, dilations, pads));
  }
  // A bias computed at run time (not a constant) still needs its gradient:
  // dB[m] = sum of dY[:, m] over batch and space.
  if (op->getNumOperands() > 2 && isActive(op->getOperand(2))) {
    Value bias = op->getOperand(2);
    std::optional<Shape> bs = staticShape(bias);
    if (!bs)
      return op->emitError("backward: Conv bias needs a static shape");
    accumulate(bias, e.reshape(e.reduceSum(gy, {0, 2, 3}, /*keep=*/false), *bs));
  }
  if (!weightGradOps.contains(op))
    return success();

  Shape ys = e.shape(gy);
  int64_t KH = (*ws)[2], KW = (*ws)[3];
  Value dW;
  bool unit = KH == 1 && KW == 1 && strides == Shape{1, 1} &&
              pads == Shape{0, 0, 0, 0};
  if (group == 1 && unit) {
    // dW[m, c] = sum_p dY[m, p] X[c, p]
    Value dy2 = e.reshape(gy, {M, ys[2] * ys[3]});
    Value x2 = e.reshape(X, {C, (*xs)[2] * (*xs)[3]});
    dW = e.reshape(e.matmul(dy2, e.transpose(x2, {1, 0})), {M, C, 1, 1});
  } else if (group == 1) {
    // Batch <-> channel swapped: X as C images of 1 channel, dY as M kernels.
    Value r = e.conv(e.transpose(X, {1, 0, 2, 3}), e.transpose(gy, {1, 0, 2, 3}),
        1, dilations, strides, pads);
    dW = e.transpose(e.slice(r, {0, 0, 0, 0}, {C, M, KH, KW}), {1, 0, 2, 3});
  } else if (group == C && M == C) {
    // Depthwise: channel c of X correlated with channel c of dY.
    Value r = e.conv(X, e.reshape(gy, {M, 1, ys[2], ys[3]}), C, dilations,
        strides, pads);
    dW = e.reshape(e.slice(r, {0, 0, 0, 0}, {1, C, KH, KW}), {M, 1, KH, KW});
  } else {
    return op->emitError("backward: Conv weight gradient supports group 1 "
                         "and depthwise only");
  }
  out.operands[{op, 1}] = dW;
  return success();
}

// dX of MaxPool (2-D): every kernel offset in row-major order is compared
// with Y on the -inf padded input, and dY goes to the FIRST offset of each
// window that holds the maximum -- PyTorch's (and ONNX Runtime's) argmax
// convention, which matters after ReLU, where windows of equal zeros are
// common. Each offset's share is zero-inserted back to input positions and
// the shares are summed; the padding is cropped at the end.
LogicalResult Sweep::maxPool(Operation *op, Value gy) {
  Value X = op->getOperand(0), Y = op->getResult(0);
  std::optional<Shape> xs = staticShape(X), ys = staticShape(Y);
  StringRef autoPad = "NOTSET";
  if (auto a = op->getAttrOfType<StringAttr>("auto_pad"))
    autoPad = a.getValue();
  Shape kernel = intsAttr(op, "kernel_shape", {});
  if (!xs || !ys || xs->size() != 4 || kernel.size() != 2 ||
      intAttr(op, "ceil_mode", 0) != 0 ||
      (autoPad != "NOTSET" && autoPad != "VALID"))
    return op->emitError("backward: MaxPool needs a static 2-D input, "
                         "ceil_mode 0 and auto_pad NOTSET/VALID");
  Shape strides = intsAttr(op, "strides", {1, 1});
  Shape dilations = intsAttr(op, "dilations", {1, 1});
  Shape pads = autoPad == "VALID" ? Shape{0, 0, 0, 0}
                                  : intsAttr(op, "pads", {0, 0, 0, 0});
  float ninf = -std::numeric_limits<float>::infinity();
  Value xp = e.padValue(X, {0, 0, pads[0], pads[1], 0, 0, pads[2], pads[3]}, ninf);
  Shape ps = e.shape(xp);
  int64_t N = (*ys)[0], C = (*ys)[1], OH = (*ys)[2], OW = (*ys)[3];
  Value taken, dxp;
  for (int64_t kh = 0; kh < kernel[0]; ++kh)
    for (int64_t kw = 0; kw < kernel[1]; ++kw) {
      int64_t h0 = kh * dilations[0], w0 = kw * dilations[1];
      Value window = e.slice(xp, {0, 0, h0, w0},
          {N, C, h0 + (OH - 1) * strides[0] + 1, w0 + (OW - 1) * strides[1] + 1},
          {1, 1, strides[0], strides[1]});
      Value hit = e.compare<ONNXEqualOp>(window, Y);
      Value first = taken ? e.andOp(hit, e.notOp(taken)) : hit;
      taken = taken ? e.orOp(taken, hit) : hit;
      Value share = insertStrideZeros(e, e.mask(first, gy), strides);
      Shape us = e.shape(share);
      share = e.padZero(share,
          {0, 0, h0, w0, 0, 0, ps[2] - h0 - us[2], ps[3] - w0 - us[3]});
      dxp = dxp ? e.add(dxp, share) : share;
    }
  accumulate(X, e.slice(dxp, {0, 0, pads[0], pads[1]},
                    {N, C, pads[0] + (*xs)[2], pads[1] + (*xs)[3]}));
  return success();
}

LogicalResult Sweep::gemm(Operation *op, Value gy) {
  Value A = op->getOperand(0), B = op->getOperand(1);
  std::optional<Shape> as = staticShape(A), bs = staticShape(B);
  if (!as || !bs || as->size() != 2 || bs->size() != 2)
    return op->emitError("backward: Gemm needs static 2-D operands");
  bool tA = intAttr(op, "transA", 0) != 0;
  bool tB = intAttr(op, "transB", 0) != 0;
  double alpha = floatAttr(op, "alpha", 1.0);
  Value g = alpha == 1.0 ? gy : e.mul(gy, e.scalar(static_cast<float>(alpha)));
  if (isActive(A)) {
    // op(A) = g op(B)^T; op(B)^T is B when transB, else B^T.
    Value dA = tB ? e.matmul(g, B) : e.matmulBT(g, B);
    accumulate(A, tA ? e.transpose(dA, {1, 0}) : dA);
  }
  if (isActive(B) || weightGradOps.contains(op)) {
    Value dB = e.matmul(tA ? A : e.transpose(A, {1, 0}), g);
    dB = tB ? e.transpose(dB, {1, 0}) : dB;
    if (isActive(B))
      accumulate(B, dB);
    if (weightGradOps.contains(op))
      out.operands[{op, 1}] = dB;
  }
  // Gemm's C (bias), when computed at run time: beta * dY, summed back over
  // the axes C was broadcast along.
  if (op->getNumOperands() > 2 && isActive(op->getOperand(2))) {
    Value bias = op->getOperand(2);
    std::optional<Shape> bs = staticShape(bias);
    if (!bs)
      return op->emitError("backward: Gemm C needs a static shape");
    double beta = floatAttr(op, "beta", 1.0);
    Value gc = beta == 1.0 ? gy : e.mul(gy, e.scalar(static_cast<float>(beta)));
    accumulate(bias, e.unbroadcast(gc, *bs));
  }
  return success();
}

// MatMul of operands of rank >= 2 (batched, broadcasting): dA = dY B^T,
// dB = A^T dY over the last two axes, each summed back over the batch axes the
// operand was broadcast along.
LogicalResult Sweep::matmul(Operation *op, Value gy) {
  Value A = op->getOperand(0), B = op->getOperand(1);
  std::optional<Shape> as = staticShape(A), bs = staticShape(B);
  if (!as || !bs || as->size() < 2 || bs->size() < 2)
    return op->emitError("backward: MatMul needs static operands of rank >= 2");
  if (isActive(A))
    accumulate(A, e.unbroadcast(e.matmulBT(gy, B), *as));
  if (isActive(B) || weightGradOps.contains(op)) {
    Value dB = e.unbroadcast(e.matmul(e.swapLast2(A), gy), *bs);
    if (isActive(B))
      accumulate(B, dB);
    if (weightGradOps.contains(op))
      out.operands[{op, 1}] = dB;
  }
  return success();
}

// Split: the input gradient is the results' gradients concatenated back (zero
// for a result the loss does not depend on). The result shapes give the
// sizes, whichever way the split was specified.
LogicalResult Sweep::split(Operation *op, ArrayRef<Value> gys) {
  Value X = op->getOperand(0);
  std::optional<Shape> xs = staticShape(X);
  if (!xs)
    return op->emitError("backward: Split needs a static input");
  int64_t axis = intAttr(op, "axis", 0);
  axis = axis < 0 ? axis + static_cast<int64_t>(xs->size()) : axis;
  SmallVector<Value, 4> parts;
  for (auto [r, g] : llvm::zip(op->getResults(), gys)) {
    std::optional<Shape> rs = staticShape(r);
    if (!rs)
      return op->emitError("backward: Split needs static results");
    parts.push_back(g ? g : e.constant(*rs, SmallVector<float>(
                                                  numElems(*rs), 0.0f)));
  }
  accumulate(X, e.concat(parts, axis));
  return success();
}

LogicalResult Sweep::reduceMean(Operation *op, Value gy) {
  Value X = op->getOperand(0);
  std::optional<Shape> xs = staticShape(X);
  if (!xs)
    return op->emitError("backward: ReduceMean needs a static input");
  int64_t rank = xs->size();
  Shape axes;
  bool keep = intAttr(op, "keepdims", 1) != 0;
  StringRef name = op->getName().getStringRef();
  if (name == "onnx.GlobalAveragePool") {
    for (int64_t d = 2; d < rank; ++d)
      axes.push_back(d);
    keep = true;
  } else if (name == "onnx.ReduceMeanV13") {
    axes = intsAttr(op, "axes", {});
  } else if (op->getNumOperands() > 1 &&
             !llvm::isa<NoneType>(op->getOperand(1).getType())) {
    ElementsAttr a = constantAttr(op->getOperand(1));
    if (!a)
      return op->emitError("backward: ReduceMean axes must be a constant");
    for (int64_t v : a.getValues<int64_t>())
      axes.push_back(v);
  }
  if (axes.empty()) {
    // ONNX ReduceMean (opset 18): no axes -- an absent operand or an empty
    // constant -- reduces every axis, unless noop_with_empty_axes = 1, where
    // the op is the identity. ReduceMeanV13 and GlobalAveragePool have no
    // such attribute (and GlobalAveragePool always has axes).
    if (name == "onnx.ReduceMean" && intAttr(op, "noop_with_empty_axes", 0) != 0) {
      accumulate(X, gy);
      return success();
    }
    for (int64_t d = 0; d < rank; ++d)
      axes.push_back(d);
  }
  Shape keepShape(*xs);
  int64_t count = 1;
  for (int64_t &a : axes) {
    a = a < 0 ? a + rank : a;
    count *= (*xs)[a];
    keepShape[a] = 1;
  }
  Value g = keep ? gy : e.reshape(gy, keepShape);
  accumulate(X, e.mul(e.expand(g, *xs), e.scalar(1.0f / static_cast<float>(count))));
  return success();
}

LogicalResult Sweep::visit(Operation *op, Value gy) {
  StringRef name = op->getName().getStringRef();
  auto operand = [&](unsigned i) { return op->getOperand(i); };
  if (name == "onnx.Conv")
    return conv(op, gy);
  if (name == "onnx.Gemm")
    return gemm(op, gy);
  if (name == "onnx.MatMul")
    return matmul(op, gy);
  if (name == "onnx.MaxPoolSingleOut")
    return maxPool(op, gy);
  if (name == "onnx.ReduceMean" || name == "onnx.ReduceMeanV13" ||
      name == "onnx.GlobalAveragePool")
    return reduceMean(op, gy);
  if (name == "onnx.Add" || name == "onnx.Sub" || name == "onnx.Mul" ||
      name == "onnx.Div") {
    Value A = operand(0), B = operand(1);
    for (int i = 0; i < 2; ++i) {
      Value v = i == 0 ? A : B;
      if (!isActive(v))
        continue;
      std::optional<Shape> s = staticShape(v);
      if (!s)
        return op->emitError("backward: ") << name << " needs static operands";
      Value g = gy;
      if (name == "onnx.Mul")
        g = e.mul(gy, i == 0 ? B : A);
      else if (name == "onnx.Sub" && i == 1)
        g = e.sub(e.scalar(0.0f), gy);
      else if (name == "onnx.Div") // y = a / b: da = g / b, db = -g y / b
        g = i == 0 ? e.div(gy, B)
                   : e.sub(e.scalar(0.0f), e.div(e.mul(gy, op->getResult(0)), B));
      accumulate(v, e.unbroadcast(g, *s));
    }
    return success();
  }
  if (name == "onnx.Where") { // where(cond, a, b): cond is not differentiable
    for (unsigned i = 1; i < 3; ++i) {
      Value v = operand(i);
      if (!isActive(v))
        continue;
      std::optional<Shape> s = staticShape(v);
      if (!s)
        return op->emitError("backward: Where needs static operands");
      Value g = i == 1 ? e.mask(operand(0), gy) : e.maskNot(operand(0), gy);
      accumulate(v, e.unbroadcast(g, *s));
    }
    return success();
  }
  if (name == "onnx.Gather") {
    // Gather of an activation (e.g. a BERT pooler taking token 0) at constant
    // indices that are one scalar or a contiguous ascending range: dY placed
    // back at those positions, zero elsewhere. Embedding lookups (constant
    // data) carry no gradient and never get here.
    Value X = operand(0);
    std::optional<Shape> xs = staticShape(X);
    ElementsAttr idx = constantAttr(operand(1));
    if (!isActive(X))
      return success();
    if (!xs || !idx)
      return op->emitError("backward: Gather of an activation needs a static "
                           "input and constant indices");
    int64_t axis = intAttr(op, "axis", 0), rank = xs->size();
    axis = axis < 0 ? axis + rank : axis;
    int64_t n = (*xs)[axis];
    SmallVector<int64_t, 8> ids;
    for (APInt v : idx.getValues<APInt>())
      ids.push_back(v.getSExtValue() < 0 ? v.getSExtValue() + n : v.getSExtValue());
    for (size_t i = 1; i < ids.size(); ++i)
      if (ids[i] != ids[0] + static_cast<int64_t>(i))
        return op->emitError("backward: Gather of an activation needs a "
                             "contiguous ascending index range");
    if (ids.empty() || mlir::cast<ShapedType>(idx.getType()).getRank() > 1)
      return op->emitError("backward: Gather of an activation needs a scalar "
                           "or 1-D index");
    Shape placed(*xs);
    placed[axis] = static_cast<int64_t>(ids.size());
    SmallVector<int64_t, 8> pads(2 * rank, 0);
    pads[axis] = ids[0];
    pads[rank + axis] = n - ids[0] - static_cast<int64_t>(ids.size());
    accumulate(X, e.padZero(e.reshape(gy, placed), pads));
    return success();
  }
  if (name == "onnx.Concat") {
    int64_t axis = intAttr(op, "axis", 0);
    std::optional<Shape> ys = staticShape(op->getResult(0));
    if (!ys)
      return op->emitError("backward: Concat needs a static result");
    axis = axis < 0 ? axis + static_cast<int64_t>(ys->size()) : axis;
    int64_t offset = 0;
    for (Value v : op->getOperands()) {
      std::optional<Shape> s = staticShape(v);
      if (!s)
        return op->emitError("backward: Concat needs static operands");
      if (isActive(v)) {
        Shape starts(ys->size(), 0), ends(*ys);
        starts[axis] = offset;
        ends[axis] = offset + (*s)[axis];
        accumulate(v, e.slice(gy, starts, ends));
      }
      offset += (*s)[axis];
    }
    return success();
  }
  Value X = operand(0);
  if (!isActive(X))
    return success();
  if (name == "onnx.Relu") {
    accumulate(X, e.mask(e.compare<ONNXGreaterOp>(X, e.scalar(0.0f)), gy));
    return success();
  }
  if (name == "onnx.Clip") {
    Value cond;
    for (unsigned i = 1; i < 3 && i < op->getNumOperands(); ++i) {
      Value bound = operand(i);
      if (llvm::isa<NoneType>(bound.getType()))
        continue;
      Value c = i == 1 ? e.compare<ONNXGreaterOp>(X, bound)
                       : e.compare<ONNXLessOp>(X, bound);
      cond = cond ? e.andOp(cond, c) : c;
    }
    accumulate(X, cond ? e.mask(cond, gy) : gy);
    return success();
  }
  if (name == "onnx.Sigmoid") {
    Value y = op->getResult(0);
    accumulate(X, e.mul(gy, e.mul(y, e.sub(e.scalar(1.0f), y))));
    return success();
  }
  if (name == "onnx.Tanh") {
    Value y = op->getResult(0);
    accumulate(X, e.mul(gy, e.sub(e.scalar(1.0f), e.mul(y, y))));
    return success();
  }
  if (name == "onnx.Sqrt") {
    accumulate(X, e.div(e.mul(gy, e.scalar(0.5f)), op->getResult(0)));
    return success();
  }
  if (name == "onnx.Erf") { // d erf(x) = 2 / sqrt(pi) exp(-x^2)
    Value g = e.unary<ONNXExpOp>(e.sub(e.scalar(0.0f), e.mul(X, X)));
    accumulate(X, e.mul(gy, e.mul(g, e.scalar(1.1283791670955126f))));
    return success();
  }
  if (name == "onnx.Neg") {
    accumulate(X, e.sub(e.scalar(0.0f), gy));
    return success();
  }
  if (name == "onnx.Pow") { // constant scalar exponent p: p x^(p-1)
    ElementsAttr pa = constantAttr(operand(1));
    if (!pa || pa.getNumElements() != 1 || isActive(operand(1)))
      return op->emitError("backward: Pow needs a constant scalar exponent");
    float p = *pa.getValues<float>().begin();
    Value d = p == 2.0f ? e.mul(X, e.scalar(2.0f))
                        : e.mul(e.pow(X, p - 1.0f), e.scalar(p));
    accumulate(X, e.mul(gy, d));
    return success();
  }
  if (name == "onnx.Softmax") { // dx = y (g - sum(g y))
    std::optional<Shape> xs = staticShape(X);
    if (!xs)
      return op->emitError("backward: Softmax needs a static input");
    int64_t axis = intAttr(op, "axis", -1);
    axis = axis < 0 ? axis + static_cast<int64_t>(xs->size()) : axis;
    Value y = op->getResult(0);
    Value dot = e.reduceSum(e.mul(gy, y), {axis}, /*keep=*/true);
    accumulate(X, e.mul(y, e.sub(gy, dot)));
    return success();
  }
  if (name == "onnx.Transpose") {
    std::optional<Shape> xs = staticShape(X);
    if (!xs)
      return op->emitError("backward: Transpose needs a static input");
    int64_t r = xs->size();
    Shape perm = intsAttr(op, "perm", {});
    if (perm.empty())
      for (int64_t d = r - 1; d >= 0; --d)
        perm.push_back(d);
    Shape inv(r);
    for (int64_t d = 0; d < r; ++d)
      inv[perm[d]] = d;
    accumulate(X, e.transpose(gy, inv));
    return success();
  }
  if (name == "onnx.Slice") {
    // Constant starts / ends / axes and unit steps (e.g. RoPE's rotate_half):
    // dY padded with zeros back to X's shape.
    std::optional<Shape> xs = staticShape(X);
    ElementsAttr starts = constantAttr(operand(1)), ends = constantAttr(operand(2));
    if (!xs || !starts || !ends)
      return op->emitError("backward: Slice needs a static input and constant "
                           "starts / ends");
    auto ints = [](ElementsAttr a) {
      SmallVector<int64_t, 8> v;
      for (APInt x : a.getValues<APInt>())
        v.push_back(x.getSExtValue());
      return v;
    };
    auto optional = [&](unsigned i) -> FailureOr<SmallVector<int64_t, 8>> {
      if (op->getNumOperands() <= i || llvm::isa<NoneType>(operand(i).getType()))
        return SmallVector<int64_t, 8>();
      ElementsAttr a = constantAttr(operand(i));
      if (!a)
        return failure();
      return ints(a);
    };
    SmallVector<int64_t, 8> lo = ints(starts), hi = ints(ends);
    FailureOr<SmallVector<int64_t, 8>> axes = optional(3), steps = optional(4);
    if (failed(axes) || failed(steps))
      return op->emitError("backward: Slice axes / steps must be constants");
    int64_t rank = xs->size();
    SmallVector<int64_t, 8> pads(2 * rank, 0);
    for (size_t i = 0; i < lo.size(); ++i) {
      int64_t d = axes->empty() ? static_cast<int64_t>(i) : (*axes)[i];
      d = d < 0 ? d + rank : d;
      if (!steps->empty() && (*steps)[i] != 1)
        return op->emitError("backward: Slice with a step other than 1");
      int64_t n = (*xs)[d];
      int64_t b = std::clamp(lo[i] < 0 ? lo[i] + n : lo[i], int64_t(0), n);
      int64_t f = std::clamp(hi[i] < 0 ? hi[i] + n : hi[i], int64_t(0), n);
      pads[d] = b;
      pads[rank + d] = n - std::max(b, f);
    }
    accumulate(X, e.padZero(gy, pads));
    return success();
  }
  if (name == "onnx.Expand") { // broadcast: dX sums dY over the expanded axes
    std::optional<Shape> xs = staticShape(X);
    if (!xs || !staticShape(op->getResult(0)))
      return op->emitError("backward: Expand needs static shapes");
    accumulate(X, e.unbroadcast(gy, *xs));
    return success();
  }
  if (name == "onnx.Cast") { // f32 -> f32 only: other inputs are not active
    if (!isF32Tensor(op->getResult(0)))
      return op->emitError("backward: Cast of an activation to a non-f32 type");
    accumulate(X, gy);
    return success();
  }
  if (name == "onnx.Reshape" || name == "onnx.Flatten" ||
      name == "onnx.Identity" || name == "onnx.Squeeze" ||
      name == "onnx.Unsqueeze") {
    std::optional<Shape> xs = staticShape(X);
    if (!xs)
      return op->emitError("backward: ") << name << " needs a static input";
    accumulate(X, e.reshape(gy, *xs));
    return success();
  }
  return op->emitError("backward: no gradient rule for ") << name;
}

} // namespace

FailureOr<BackwardGradients> buildBackward(OpBuilder &builder,
    func::FuncOp func, Value root, Value seed,
    const llvm::DenseSet<Operation *> &weightGradOps) {
  Block &body = func.getBody().front();
  Emitter e(builder, root.getLoc());
  Sweep sweep{e, weightGradOps, {}, {}};

  // Forward activity: float tensors derived from a float function argument,
  // and the outputs of a Gather whose indices come from a function argument
  // (embedding lookups of input token ids -- the data starts there; the
  // gradient stops at them). Integer and boolean values (shapes, masks) never
  // carry a gradient.
  llvm::DenseSet<Value> fromInput;
  for (BlockArgument a : body.getArguments()) {
    fromInput.insert(a);
    if (isF32Tensor(a))
      sweep.active.insert(a);
  }
  SmallVector<Operation *, 256> ops;
  for (Operation &op : body.without_terminator()) {
    ops.push_back(&op);
    bool input = llvm::any_of(
        op.getOperands(), [&](Value v) { return fromInput.contains(v); });
    if (input)
      for (Value r : op.getResults())
        fromInput.insert(r);
    bool embedding = input && op.getName().getStringRef() == "onnx.Gather" &&
                     fromInput.contains(op.getOperand(1));
    if (!embedding && !llvm::any_of(op.getOperands(),
                          [&](Value v) { return sweep.isActive(v); }))
      continue;
    for (Value r : op.getResults())
      if (isF32Tensor(r))
        sweep.active.insert(r);
  }

  if (!staticShape(root) || !staticShape(seed) ||
      *staticShape(root) != *staticShape(seed))
    return func.emitError("backward: loss seed must have the root's static shape");
  sweep.out.values[root] = seed;
  for (Operation *op : llvm::reverse(ops)) {
    SmallVector<Value, 4> gys;
    for (Value r : op->getResults())
      gys.push_back(sweep.out.values.lookup(r));
    if (llvm::none_of(gys, [](Value g) { return static_cast<bool>(g); }))
      continue;
    bool anyActive = llvm::any_of(op->getOperands(),
        [&](Value v) { return sweep.isActive(v); });
    if (!anyActive && !weightGradOps.contains(op))
      continue;
    LogicalResult r =
        op->getNumResults() == 1 ? sweep.visit(op, gys.front())
        : op->getName().getStringRef().starts_with("onnx.Split")
            ? sweep.split(op, gys)
            : op->emitError("backward: no gradient rule for multi-result ")
                  << op->getName();
    if (failed(r))
      return failure();
  }
  for (Operation *op : weightGradOps)
    if (!sweep.out.operands.count({op, 1}))
      return op->emitError("backward: the loss does not depend on this node, "
                           "so it has no weight gradient");
  return std::move(sweep.out);
}

} // namespace onnx_mlir
