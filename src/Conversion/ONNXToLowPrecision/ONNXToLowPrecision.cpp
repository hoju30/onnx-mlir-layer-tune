#include "onnx-mlir/Conversion/ONNXToLowPrecision/ONNXToLowPrecision.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using namespace mlir;

namespace onnx_mlir {
namespace {

// Formats handled by this pass.
enum class LowPrecisionFormat { BF16, F16, INT8, FP8E4M3, FP8E5M2 };

static bool isQuantFormat(LowPrecisionFormat fmt) {
  return fmt == LowPrecisionFormat::INT8 || fmt == LowPrecisionFormat::FP8E4M3 ||
         fmt == LowPrecisionFormat::FP8E5M2;
}

struct NodeFormatEntry {
  LowPrecisionFormat format;
  // INT8/FP8E4M3/FP8E5M2 only, supplied via LOWP_NODE_FORMATS as
  // "Conv_1:int8:<x_scale>:<x_zero_point>:<y_scale>:<y_zero_point>". For
  // int8, the zero-points are integer values; for the fp8 formats they're
  // real values (almost always 0.0 in practice -- see the fp8
  // quantize/dequantize helpers below). Weight/bias quantization params are
  // derived from the weight/bias constants themselves (see the
  // quantizeConstant* helpers below), not supplied here, since they're
  // static and don't need calibration.
  //
  // Conv/Gemm/MatMul/Relu (single-input ops) take 4 params as above.
  // Add/Sub/Mul/Div (2 independently-scaled inputs) take 6:
  // <a_scale>:<a_zero_point>:<b_scale>:<b_zero_point>:<y_scale>:<y_zero_point>
  // -- see retypeElementwiseToQuant below.
  SmallVector<double, 6> quantParams;
};

static bool parseLowPrecisionFormatTag(
    StringRef tag, LowPrecisionFormat &fmt) {
  if (tag == "bf16") {
    fmt = LowPrecisionFormat::BF16;
    return true;
  }
  if (tag == "f16") {
    fmt = LowPrecisionFormat::F16;
    return true;
  }
  if (tag == "int8") {
    fmt = LowPrecisionFormat::INT8;
    return true;
  }
  if (tag == "fp8e4m3") {
    fmt = LowPrecisionFormat::FP8E4M3;
    return true;
  }
  if (tag == "fp8e5m2") {
    fmt = LowPrecisionFormat::FP8E5M2;
    return true;
  }
  return false;
}

// Parse LOWP_NODE_FORMATS=Conv_1:bf16,Gemm_3:f16,Conv_5:int8:0.05:0:0.1:0
// into node_name -> format(+params). Entries with an unrecognized format tag
// are skipped rather than erroring, so future tags don't break a
// mixed-format env var.
static std::map<std::string, NodeFormatEntry> parseNodeFormats() {
  std::map<std::string, NodeFormatEntry> result;
  const char *e = std::getenv("LOWP_NODE_FORMATS");
  if (!e || !*e)
    return result;
  std::string s(e);
  size_t pos = 0;
  while (pos < s.size()) {
    size_t comma = s.find(',', pos);
    if (comma == std::string::npos)
      comma = s.size();
    std::string entry = s.substr(pos, comma - pos);
    size_t c1 = entry.find(':');
    if (c1 != std::string::npos) {
      std::string name = entry.substr(0, c1);
      std::string rest = entry.substr(c1 + 1);
      size_t c2 = rest.find(':');
      std::string tagStr = c2 == std::string::npos ? rest : rest.substr(0, c2);
      LowPrecisionFormat fmt;
      if (parseLowPrecisionFormatTag(tagStr, fmt)) {
        NodeFormatEntry nfe;
        nfe.format = fmt;
        if (isQuantFormat(fmt) && c2 != std::string::npos) {
          std::string paramsStr = rest.substr(c2 + 1);
          size_t ppos = 0;
          while (ppos < paramsStr.size()) {
            size_t pc = paramsStr.find(':', ppos);
            if (pc == std::string::npos)
              pc = paramsStr.size();
            nfe.quantParams.push_back(
                std::stod(paramsStr.substr(ppos, pc - ppos)));
            ppos = pc + 1;
          }
        }
        result[name] = nfe;
      }
    }
    pos = comma + 1;
  }
  return result;
}

static Type getTargetElementType(LowPrecisionFormat fmt, Builder &b) {
  switch (fmt) {
  case LowPrecisionFormat::BF16:
    return b.getBF16Type();
  case LowPrecisionFormat::F16:
    return b.getF16Type();
  default:
    llvm_unreachable("getTargetElementType only handles BF16/F16");
  }
}

static Type retypeTensorElement(Type t, Type newElem) {
  if (auto ranked = llvm::dyn_cast<RankedTensorType>(t))
    return RankedTensorType::get(ranked.getShape(), newElem);
  if (llvm::isa<UnrankedTensorType>(t))
    return UnrankedTensorType::get(newElem);
  return t;
}

static bool isFloatTensor(Type t) {
  if (auto shaped = llvm::dyn_cast<ShapedType>(t))
    return llvm::isa<FloatType>(shaped.getElementType());
  return false;
}

static Operation *createGenericOp(IRRewriter &rewriter, Location loc,
    StringRef name, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attrs) {
  OperationState st(loc, name);
  st.addOperands(operands);
  st.addTypes(resultTypes);
  st.addAttributes(attrs);
  return rewriter.create(st);
}

// Ops whose ONNX dialect type constraints (ONNXOps.td.inc, generated from
// the upstream ONNX operator spec) already legally accept BF16/F16
// operands/results alongside F32/F64 today. Only these are safe to retype
// in place; anything else is left untouched (and warned about if it was
// explicitly named in LOWP_NODE_FORMATS).
static bool isSupportedOpForCastRetype(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return name == "onnx.Conv" || name == "onnx.Gemm" || name == "onnx.MatMul" ||
         name == "onnx.Relu" || name == "onnx.Add" || name == "onnx.Sub" ||
         name == "onnx.Mul" || name == "onnx.Div";
}

static Value insertCast(
    IRRewriter &rewriter, Location loc, Value input, Type targetElemType) {
  Type resultType = retypeTensorElement(input.getType(), targetElemType);
  SmallVector<NamedAttribute, 2> attrs;
  attrs.emplace_back(
      rewriter.getStringAttr("to"), TypeAttr::get(targetElemType));
  attrs.emplace_back(rewriter.getStringAttr("saturate"),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 1));
  Operation *castOp = createGenericOp(rewriter, loc, "onnx.Cast",
      TypeRange{resultType}, ValueRange{input}, attrs);
  return castOp->getResult(0);
}

static void retypeCastOp(IRRewriter &rewriter, Operation *op,
    LowPrecisionFormat fmt, StringRef nodeName) {
  Type targetElem = getTargetElementType(fmt, rewriter);
  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  SmallVector<Value, 4> newOperands;
  for (Value operand : op->getOperands()) {
    if (isFloatTensor(operand.getType()))
      newOperands.push_back(insertCast(rewriter, loc, operand, targetElem));
    else
      newOperands.push_back(operand);
  }

  SmallVector<Type, 2> newResultTypes;
  for (Type resTy : op->getResultTypes()) {
    if (isFloatTensor(resTy))
      newResultTypes.push_back(retypeTensorElement(resTy, targetElem));
    else
      newResultTypes.push_back(resTy);
  }

  Operation *newOp = createGenericOp(rewriter, loc,
      op->getName().getStringRef(), newResultTypes, newOperands,
      op->getAttrs());

  rewriter.setInsertionPointAfter(newOp);
  for (auto pair : llvm::zip(op->getResults(), newOp->getResults())) {
    Value oldResult = std::get<0>(pair);
    Value newResult = std::get<1>(pair);
    if (isFloatTensor(oldResult.getType())) {
      Value upcast =
          insertCast(rewriter, loc, newResult, rewriter.getF32Type());
      rewriter.replaceAllUsesWith(oldResult, upcast);
    } else {
      rewriter.replaceAllUsesWith(oldResult, newResult);
    }
  }
  rewriter.eraseOp(op);
}

// ===========================================================================
// INT8 (QLinearConv) support.
//
// Scope: onnx.Conv nodes only (matching src/Conversion/ONNXToKrnl/Math/
// QLinearConv.cpp, group==1, 4D NCHW, auto_pad=NOTSET, statically shaped).
// Gemm/MatMul under :int8 are not wired up yet (QLinearMatMul doesn't take
// Gemm's alpha/beta/transA/transB attrs, so using it there needs pre-folding
// those attrs into the operands first; left for a follow-up).
//
// Activation (x/y) scale+zero_point come from LOWP_NODE_FORMATS, since they
// depend on runtime activation ranges that need external calibration this
// pass has no access to. Weight and bias are compile-time constants, so
// their quantization params are derived directly from the constant data
// (simple per-tensor symmetric min-max: scale = max(abs(values))/127,
// zero_point = 0) rather than requiring the caller to supply them.
// ===========================================================================

static DenseElementsAttr getConstantValueAttr(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return nullptr;
  return llvm::dyn_cast_or_null<DenseElementsAttr>(def->getAttr("value"));
}

static float computeSymmetricScale(DenseElementsAttr attr) {
  float maxAbs = 0.0f;
  for (float f : attr.getValues<float>())
    maxAbs = std::max(maxAbs, std::fabs(f));
  return maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
}

// Max finite value of the given F8 format, used below to scale weight
// quantization to use the format's full dynamic range (mirrors
// computeSymmetricScale's int8 use of 127).
static float f8MaxFinite(Type f8Ty) {
  if (mlir::isa<Float8E5M2Type>(f8Ty))
    return 57344.0f;
  if (mlir::isa<Float8E4M3FNType>(f8Ty))
    return 448.0f;
  llvm_unreachable("unsupported F8 type in f8MaxFinite");
}

static float computeSymmetricScaleF8(DenseElementsAttr attr, Type f8Ty) {
  float maxAbs = 0.0f;
  for (float f : attr.getValues<float>())
    maxAbs = std::max(maxAbs, std::fabs(f));
  return maxAbs > 0.0f ? maxAbs / f8MaxFinite(f8Ty) : 1.0f;
}

static DenseElementsAttr quantizeToI8(
    DenseElementsAttr floatAttr, float scale, int8_t zeroPoint) {
  auto floatTy = mlir::cast<RankedTensorType>(floatAttr.getType());
  auto i8Ty = RankedTensorType::get(
      floatTy.getShape(), IntegerType::get(floatTy.getContext(), 8));
  SmallVector<APInt, 64> vals;
  for (float f : floatAttr.getValues<float>()) {
    int32_t q = static_cast<int32_t>(std::lround(f / scale)) + zeroPoint;
    q = std::clamp(q, -128, 127);
    vals.push_back(APInt(8, static_cast<int64_t>(q), /*isSigned=*/true));
  }
  return DenseElementsAttr::get(i8Ty, vals);
}

// Bias must be quantized with scale = x_scale * w_scale, zero_point = 0
// (ONNX QLinearConv spec requirement).
static DenseElementsAttr quantizeBiasToI32(
    DenseElementsAttr floatAttr, float scale) {
  auto floatTy = mlir::cast<RankedTensorType>(floatAttr.getType());
  auto i32Ty = RankedTensorType::get(
      floatTy.getShape(), IntegerType::get(floatTy.getContext(), 32));
  SmallVector<APInt, 64> vals;
  for (float f : floatAttr.getValues<float>())
    vals.push_back(APInt(32, std::lround(f / scale), /*isSigned=*/true));
  return DenseElementsAttr::get(i32Ty, vals);
}

static Value buildConstant(
    IRRewriter &rewriter, Location loc, Attribute value) {
  auto elements = mlir::cast<ElementsAttr>(value);
  Operation *cst = createGenericOp(rewriter, loc, "onnx.Constant",
      TypeRange{elements.getType()}, ValueRange{},
      {NamedAttribute(rewriter.getStringAttr("value"), value)});
  return cst->getResult(0);
}

static Value buildScalarF32Constant(
    IRRewriter &rewriter, Location loc, float v) {
  auto ty = RankedTensorType::get({}, rewriter.getF32Type());
  return buildConstant(rewriter, loc, DenseElementsAttr::get(ty, v));
}

static Value buildScalarI8Constant(
    IRRewriter &rewriter, Location loc, int8_t v) {
  auto ty = RankedTensorType::get({}, rewriter.getI8Type());
  return buildConstant(rewriter, loc,
      DenseElementsAttr::get(ty, APInt(8, v, /*isSigned=*/true)));
}

// F8E4M3FN/F8E5M2 scalar constant from a real value. Unlike the int8 case,
// no manual bit-encoding is needed here: MLIR's own FloatAttr/APFloat
// machinery already implements correct F8 rounding for these builtin types
// (this is the same rounding logic backing e.g. `onnx.Constant dense<1.5> :
// tensor<f8E4M3FN>` in a .mlir file), so building the attribute directly is
// both simpler and exactly as correct as going through the F8 encode logic
// in DialectBuilder.cpp (which exists for the *runtime* Krnl-level path,
// where there's no attribute/APFloat machinery available).
static Value buildScalarF8Constant(
    IRRewriter &rewriter, Location loc, double v, Type f8Ty) {
  auto ty = RankedTensorType::get({}, f8Ty);
  return buildConstant(
      rewriter, loc, DenseElementsAttr::get(ty, rewriter.getFloatAttr(f8Ty, v)));
}

// Quantize a float32 constant tensor to an F8E4M3FN/F8E5M2 constant tensor
// (dividing by `scale` first), again relying on FloatAttr/APFloat for
// correct F8 rounding rather than reimplementing it.
static DenseElementsAttr quantizeToF8(
    DenseElementsAttr floatAttr, float scale, Type f8Ty) {
  auto floatTy = mlir::cast<RankedTensorType>(floatAttr.getType());
  auto f8TensorTy = RankedTensorType::get(floatTy.getShape(), f8Ty);
  SmallVector<Attribute, 64> vals;
  Builder b(f8Ty.getContext());
  for (float f : floatAttr.getValues<float>())
    vals.push_back(b.getFloatAttr(f8Ty, f / scale));
  return DenseElementsAttr::get(f8TensorTy, vals);
}

static Value buildQuantizeLinear(IRRewriter &rewriter, Location loc,
    Value input, Value scale, Value zeroPoint, Type targetElemType) {
  Type resultType = retypeTensorElement(input.getType(), targetElemType);
  SmallVector<NamedAttribute, 2> attrs;
  attrs.emplace_back(rewriter.getStringAttr("axis"),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 1));
  attrs.emplace_back(rewriter.getStringAttr("saturate"),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 1));
  Operation *op = createGenericOp(rewriter, loc, "onnx.QuantizeLinear",
      TypeRange{resultType}, ValueRange{input, scale, zeroPoint}, attrs);
  return op->getResult(0);
}

static Value buildDequantizeLinear(IRRewriter &rewriter, Location loc,
    Value input, Value scale, Value zeroPoint, Type resultType) {
  SmallVector<NamedAttribute, 1> attrs;
  attrs.emplace_back(rewriter.getStringAttr("axis"),
      rewriter.getIntegerAttr(rewriter.getIntegerType(64, true), 1));
  Operation *op = createGenericOp(rewriter, loc, "onnx.DequantizeLinear",
      TypeRange{resultType}, ValueRange{input, scale, zeroPoint}, attrs);
  return op->getResult(0);
}

// FP8-specific quantize/dequantize, built from onnx.Cast (which correctly
// supports F32<->F8 via MathBuilder::cast's decode/encode logic) plus plain
// f32 arithmetic, instead of literal onnx.QuantizeLinear/DequantizeLinear:
// mainline QuantizeLinear.cpp's own Krnl lowering is hardcoded to int8/uint8
// targets (it asserts on anything else) and hasn't been extended for float8
// targets. This is mathematically equivalent to the QuantizeLinear/
// DequantizeLinear formula for the zero_point=0 case that's realistic for
// float8 (the ONNX spec's own doc notes zero-point is "usually not used"
// for float8 quantization); for a nonzero zero_point, it's added in f32
// before the cast-triggered rounding rather than after in the f8 domain,
// matching the same ordering choice already made in QLinearMatMul.cpp's
// float8 branch (see that file's comment for why the distinction is
// immaterial in the realistic zero_point==0 case).
static Value buildQuantizeCastF8(IRRewriter &rewriter, Location loc,
    Value input, Value scaleF32, Value zeroPointF32, Type f8ElemType) {
  Value divided = createGenericOp(rewriter, loc, "onnx.Div",
      TypeRange{input.getType()}, ValueRange{input, scaleF32}, {})
                      ->getResult(0);
  Value shifted = createGenericOp(rewriter, loc, "onnx.Add",
      TypeRange{input.getType()}, ValueRange{divided, zeroPointF32}, {})
                      ->getResult(0);
  return insertCast(rewriter, loc, shifted, f8ElemType);
}

static Value buildDequantizeCastF8(IRRewriter &rewriter, Location loc,
    Value input, Value scaleF32, Value zeroPointF32, Type resultType) {
  Value decoded = insertCast(rewriter, loc, input, rewriter.getF32Type());
  Value shifted = createGenericOp(rewriter, loc, "onnx.Sub", TypeRange{resultType},
      ValueRange{decoded, zeroPointF32}, {})
                      ->getResult(0);
  return createGenericOp(rewriter, loc, "onnx.Mul", TypeRange{resultType},
      ValueRange{shifted, scaleF32}, {})
      ->getResult(0);
}

// Rewrite a single onnx.Conv node into QuantizeLinear(x) ->
// onnx.QLinearConv -> DequantizeLinear(y). Returns false (leaving the node
// untouched) if the node's shape isn't one this can handle yet.
static bool retypeConvToInt8(IRRewriter &rewriter, Operation *op,
    ArrayRef<double> params, StringRef nodeName) {
  if (params.size() != 4) {
    op->emitWarning() << "LOWP_NODE_FORMATS int8 entry for '" << nodeName
                       << "' needs exactly 4 params "
                          "(x_scale:x_zero_point:y_scale:y_zero_point); "
                          "skipping.";
    return false;
  }
  Value X = op->getOperand(0);
  Value W = op->getOperand(1);
  Value Bias = op->getOperand(2);
  bool hasBias = !llvm::isa<NoneType>(Bias.getType());

  DenseElementsAttr wAttr = getConstantValueAttr(W);
  if (!wAttr) {
    op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                       << "' for int8 but its weight is not a compile-time "
                          "onnx.Constant; skipping.";
    return false;
  }
  DenseElementsAttr biasAttr;
  if (hasBias) {
    biasAttr = getConstantValueAttr(Bias);
    if (!biasAttr) {
      op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                         << "' for int8 but its bias is not a compile-time "
                            "onnx.Constant; skipping.";
      return false;
    }
  }

  float xScale = static_cast<float>(params[0]);
  int8_t xZeroPoint = static_cast<int8_t>(params[1]);
  float yScale = static_cast<float>(params[2]);
  int8_t yZeroPoint = static_cast<int8_t>(params[3]);
  float wScale = computeSymmetricScale(wAttr);
  int8_t wZeroPoint = 0;

  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  Value xScaleC = buildScalarF32Constant(rewriter, loc, xScale);
  Value xZeroPointC = buildScalarI8Constant(rewriter, loc, xZeroPoint);
  Value yScaleC = buildScalarF32Constant(rewriter, loc, yScale);
  Value yZeroPointC = buildScalarI8Constant(rewriter, loc, yZeroPoint);
  Value wScaleC = buildScalarF32Constant(rewriter, loc, wScale);
  Value wZeroPointC = buildScalarI8Constant(rewriter, loc, wZeroPoint);

  Value xQ = buildQuantizeLinear(
      rewriter, loc, X, xScaleC, xZeroPointC, rewriter.getI8Type());
  Value wQ = buildConstant(rewriter, loc, quantizeToI8(wAttr, wScale, wZeroPoint));
  Value biasQ = Bias; // pass the NoneType placeholder through unchanged
  if (hasBias)
    biasQ = buildConstant(
        rewriter, loc, quantizeBiasToI32(biasAttr, xScale * wScale));

  Type origResultType = op->getResult(0).getType();
  Type i8ResultType = retypeTensorElement(origResultType, rewriter.getI8Type());
  Operation *qlcOp = createGenericOp(rewriter, loc, "onnx.QLinearConv",
      TypeRange{i8ResultType},
      ValueRange{
          xQ, xScaleC, xZeroPointC, wQ, wScaleC, wZeroPointC, yScaleC,
          yZeroPointC, biasQ},
      op->getAttrs());

  rewriter.setInsertionPointAfter(qlcOp);
  Value deq = buildDequantizeLinear(
      rewriter, loc, qlcOp->getResult(0), yScaleC, yZeroPointC, origResultType);
  rewriter.replaceAllUsesWith(op->getResult(0), deq);
  rewriter.eraseOp(op);
  return true;
}

// Transpose a rank-2 float constant's data in place (compile time), used for
// Gemm's transB: unlike A (a runtime activation, needs a real onnx.Transpose
// op), B must already be a compile-time constant for int8 quantization
// (there's no calibration path for a non-constant weight), so its transpose
// can just be folded into the constant data directly.
static DenseElementsAttr transposeConstant2D(DenseElementsAttr attr) {
  auto ty = mlir::cast<RankedTensorType>(attr.getType());
  ArrayRef<int64_t> shape = ty.getShape();
  int64_t R = shape[0], Cd = shape[1];
  auto outTy = RankedTensorType::get({Cd, R}, ty.getElementType());
  SmallVector<float, 64> in(attr.getValues<float>().begin(),
      attr.getValues<float>().end());
  SmallVector<float, 64> out(in.size());
  for (int64_t r = 0; r < R; ++r)
    for (int64_t c = 0; c < Cd; ++c)
      out[c * R + r] = in[r * Cd + c];
  return DenseElementsAttr::get(outTy, ArrayRef<float>(out));
}

static bool isF8ElemType(Type t) {
  return mlir::isa<Float8E4M3FNType, Float8E5M2Type>(t);
}

static Value buildScalarZeroPointConstant(
    IRRewriter &rewriter, Location loc, double v, Type targetElemType) {
  if (isF8ElemType(targetElemType))
    return buildScalarF8Constant(rewriter, loc, v, targetElemType);
  return buildScalarI8Constant(rewriter, loc, static_cast<int8_t>(v));
}

static Value buildQuantizedWeightConstant(IRRewriter &rewriter, Location loc,
    DenseElementsAttr bAttr, float wScale, Type targetElemType) {
  if (isF8ElemType(targetElemType))
    return buildConstant(rewriter, loc, quantizeToF8(bAttr, wScale, targetElemType));
  return buildConstant(
      rewriter, loc, quantizeToI8(bAttr, wScale, /*zeroPoint=*/0));
}

// Quantize `val` to targetElemType at (scale, zeroPoint), then immediately
// dequantize it straight back to `val`'s own (f32) type. This simulates the
// precision loss of storing `val` in targetElemType at this scale/zero-point
// without actually keeping it in that format -- used to build the
// elementwise ops' QDQ handling below, since unlike Conv/Gemm/MatMul there's
// no ONNX quantized-arithmetic op to target for them (no QLinearRelu/
// QLinearAdd/etc. exist), so the "low precision" here is entirely the QDQ
// noise on each operand/result; the actual arithmetic stays in f32.
static Value quantDequantRoundTrip(IRRewriter &rewriter, Location loc,
    Value val, double scale, double zeroPoint, Type targetElemType) {
  Value scaleC = buildScalarF32Constant(rewriter, loc, static_cast<float>(scale));
  if (isF8ElemType(targetElemType)) {
    Value zpF32 =
        buildScalarF32Constant(rewriter, loc, static_cast<float>(zeroPoint));
    Value q = buildQuantizeCastF8(rewriter, loc, val, scaleC, zpF32, targetElemType);
    return buildDequantizeCastF8(rewriter, loc, q, scaleC, zpF32, val.getType());
  }
  Value zpC = buildScalarZeroPointConstant(rewriter, loc, zeroPoint, targetElemType);
  Value q = buildQuantizeLinear(rewriter, loc, val, scaleC, zpC, targetElemType);
  return buildDequantizeLinear(rewriter, loc, q, scaleC, zpC, val.getType());
}

// Ops handled by the QDQ-round-trip-only path below: onnx.Relu/AveragePool/
// GlobalAveragePool (1 operand) or one of the binary elementwise ops (2
// operands). No int8/fp8-typed version of the op itself is ever built --
// see quantDequantRoundTrip above. AveragePool/GlobalAveragePool fit the
// same single-input/single-output shape as Relu (their kernel_shape/
// strides/pads attributes, if any, ride along unchanged via op->getAttrs()
// in retypeElementwiseToQuant); like Add/Mul/Div, averaging already-
// quantized-then-dequantized values isn't bit-identical to averaging in a
// truly native low-precision accumulator, but is the same QDQ-simulation
// tradeoff already accepted for the other ops here.
//
// onnx.MaxPool is deliberately NOT in this list despite fitting the same
// shape (1 real operand, though a fixed ODS arity of 2 results -- Y and
// Indices, the latter typed NoneType when unused -- which
// retypeElementwiseToQuant does handle generically, round-tripping only
// result(0) and forwarding the rest). It's held back due to an unresolved,
// reproducible bug: `onnx-mlir-opt ... --convert-onnx-to-lowprecision
// --canonicalize --convert-onnx-to-krnl ...` fails to legalize the rebuilt
// onnx.MaxPool ("failed to legalize operation 'onnx.MaxPool'") *only* when
// `--convert-krnl-to-llvm` is ALSO present later in the same CLI invocation
// -- despite --convert-krnl-to-llvm running strictly after --convert-onnx-
// to-krnl in the pipeline, and despite the exact same IR (round-tripped
// through a text dump and re-parsed as a separate onnx-mlir-opt invocation)
// legalizing fine either way. This isn't a --canonicalize-placement issue
// (unlike the fp8 elementwise dominance bug documented above, which this
// bug is NOT the same as -- that one manifests during --convert-onnx-to-
// krnl itself regardless of what follows it), and root-causing it would
// need digging into how onnx-mlir-opt's CLI pass registration/PassManager
// setup differs based on which flags are co-present -- not yet done. Do not
// re-add "onnx.MaxPool" here without either fixing that or re-verifying the
// exact repro (a hand-written onnx.MaxPool node run through the full
// --convert-onnx-to-lowprecision -> ... -> --convert-krnl-to-llvm pipeline
// in one invocation) no longer fails.
static bool isElementwiseQuantOp(StringRef opName) {
  return opName == "onnx.Relu" || opName == "onnx.Add" ||
         opName == "onnx.Sub" || opName == "onnx.Mul" || opName == "onnx.Div" ||
         opName == "onnx.AveragePool" || opName == "onnx.GlobalAveragePool";
}

// Rewrite a single onnx.Relu/Add/Sub/Mul/Div node by QDQ-round-tripping each
// operand at its own scale/zero-point (quantDequantRoundTrip above), applying
// the *same* op unchanged to the round-tripped values, then QDQ-round-
// tripping the result. Relu (1 operand) takes 4 params
// (x_scale:x_zero_point:y_scale:y_zero_point); Add/Sub/Mul/Div (2 operands,
// each independently scaled/zero-pointed, since they're typically fed by two
// different upstream tensors) take 6
// (a_scale:a_zero_point:b_scale:b_zero_point:y_scale:y_zero_point).
//
// PIPELINE NOTE (fp8e4m3/fp8e5m2 only): when two or more of these nodes
// appear in the same function, `onnx-mlir-opt` needs a `--canonicalize`
// between `--convert-onnx-to-lowprecision` and `--convert-onnx-to-krnl` (not
// just the usual one *after* `--convert-onnx-to-krnl`), or the second QDQ
// chain's Krnl lowering can produce a dominance violation
// (`--convert-onnx-to-krnl` itself fails to verify). Root cause: each fp8
// round trip expands to plain onnx.Div/Add/Cast/Sub/Mul (see
// buildQuantizeCastF8/buildDequantizeCastF8 above -- there's no literal
// QuantizeLinear/DequantizeLinear for float8), which goes through mainline's
// generic Elementwise.cpp lowering; that lowering's own comment states it
// assumes canonicalization has already hoisted constants, a precondition
// this project's manual pipelines hadn't previously had reason to violate
// (a single QDQ chain, or int8's literal QuantizeLinear/DequantizeLinear
// path, never triggered it). Confirmed via a minimal repro: two chained
// fp8 QDQ round trips reliably fail --convert-onnx-to-krnl without a prior
// --canonicalize and reliably succeed (bit-exact numerically) with one;
// the equivalent two-node int8 case never needed it.
static bool retypeElementwiseToQuant(IRRewriter &rewriter, Operation *op,
    ArrayRef<double> params, StringRef nodeName, Type targetElemType) {
  unsigned numOperands = op->getNumOperands();
  size_t expectedParams = 2 * static_cast<size_t>(numOperands) + 2;
  if (params.size() != expectedParams) {
    op->emitWarning() << "LOWP_NODE_FORMATS entry for '" << nodeName << "' ("
                       << op->getName() << ") needs exactly " << expectedParams
                       << " params; skipping.";
    return false;
  }

  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  SmallVector<Value, 2> qdqOperands;
  for (unsigned i = 0; i < numOperands; ++i) {
    Value operand = op->getOperand(i);
    if (!isFloatTensor(operand.getType())) {
      op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                         << "' but operand " << i
                         << " is not a float tensor; skipping.";
      return false;
    }
    qdqOperands.push_back(quantDequantRoundTrip(
        rewriter, loc, operand, params[2 * i], params[2 * i + 1], targetElemType));
  }

  // Rebuild with the op's full original result-type list (e.g. onnx.MaxPool
  // has a fixed arity of 2 results, Y and Indices -- the latter typed
  // NoneType when unused, not omitted). Only result(0) (the actual numeric
  // output) gets QDQ-round-tripped below; any further results (MaxPool's
  // Indices) aren't numeric values to quantize and are forwarded unchanged.
  Operation *newOp = createGenericOp(rewriter, loc, op->getName().getStringRef(),
      op->getResultTypes(), qdqOperands, op->getAttrs());

  rewriter.setInsertionPointAfter(newOp);
  double yScale = params[2 * numOperands];
  double yZeroPoint = params[2 * numOperands + 1];
  Value result = quantDequantRoundTrip(
      rewriter, loc, newOp->getResult(0), yScale, yZeroPoint, targetElemType);

  SmallVector<Value, 2> replacements;
  replacements.push_back(result);
  for (unsigned i = 1; i < op->getNumResults(); ++i)
    replacements.push_back(newOp->getResult(i));
  rewriter.replaceOp(op, replacements);
  return true;
}

// Rewrite a single onnx.Gemm or onnx.MatMul node into QuantizeLinear(A) ->
// onnx.QLinearMatMul -> DequantizeLinear(Y) [-> Add(beta*C) for Gemm's
// optional bias]. Scope: 2D A/B only, B a compile-time constant. Gemm's
// alpha is folded into A's dequant scale (a free multiplicative factor in
// QLinearMatMul's own rescale step); transA/transB are honored by
// transposing A at runtime (onnx.Transpose) and B's constant data at
// compile time. Gemm's C (if present) is kept in float and added back after
// dequantization rather than being pushed through the quantized kernel
// itself, since QLinearMatMul (unlike QLinearConv) has no bias operand to
// fold it into.
//
// Shared between int8 (targetElemType = i8) and the fp8 formats
// (targetElemType = f8E4M3FN/f8E5M2, using the QLinearMatMul opset-21
// float8 branch added in src/Conversion/ONNXToKrnl/Math/QLinearMatMul.cpp):
// the Gemm/MatMul-shape handling (transpose, alpha/beta folding, bias) is
// identical either way, only the zero-point/weight-quantization helpers
// differ (see buildScalarZeroPointConstant/buildQuantizedWeightConstant
// above).
static bool retypeMatMulLikeToQuant(IRRewriter &rewriter, Operation *op,
    ArrayRef<double> params, StringRef nodeName, Type targetElemType) {
  if (params.size() != 4) {
    op->emitWarning() << "LOWP_NODE_FORMATS entry for '" << nodeName
                       << "' needs exactly 4 params "
                          "(x_scale:x_zero_point:y_scale:y_zero_point); "
                          "skipping.";
    return false;
  }
  bool isGemm = op->getName().getStringRef() == "onnx.Gemm";
  Value A = op->getOperand(0);
  Value B = op->getOperand(1);
  Value C = isGemm ? op->getOperand(2) : Value();
  bool hasC = isGemm && !llvm::isa<NoneType>(C.getType());

  float alpha = 1.0f, beta = 1.0f;
  bool transA = false, transB = false;
  if (isGemm) {
    if (auto a = op->getAttrOfType<FloatAttr>("alpha"))
      alpha = static_cast<float>(a.getValueAsDouble());
    if (auto b = op->getAttrOfType<FloatAttr>("beta"))
      beta = static_cast<float>(b.getValueAsDouble());
    if (auto ta = op->getAttrOfType<IntegerAttr>("transA"))
      transA = ta.getSInt() != 0;
    if (auto tb = op->getAttrOfType<IntegerAttr>("transB"))
      transB = tb.getSInt() != 0;
  }

  auto aTy = mlir::dyn_cast<RankedTensorType>(A.getType());
  auto bTy = mlir::dyn_cast<RankedTensorType>(B.getType());
  if (!aTy || !bTy || aTy.getRank() != 2 || bTy.getRank() != 2) {
    op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                       << "' but its A/B are not both 2D tensors; skipping.";
    return false;
  }

  DenseElementsAttr bAttr = getConstantValueAttr(B);
  if (!bAttr) {
    op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                       << "' but its B operand is not a compile-time "
                          "onnx.Constant; skipping.";
    return false;
  }
  if (transB)
    bAttr = transposeConstant2D(bAttr);

  float xScale = static_cast<float>(params[0]) * alpha; // fold alpha in
  double xZeroPoint = params[1];
  float yScale = static_cast<float>(params[2]);
  double yZeroPoint = params[3];
  float wScale = isF8ElemType(targetElemType)
                     ? computeSymmetricScaleF8(bAttr, targetElemType)
                     : computeSymmetricScale(bAttr);

  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  Value AT = A;
  if (transA) {
    ArrayRef<int64_t> shape = aTy.getShape();
    auto outTy = RankedTensorType::get({shape[1], shape[0]}, aTy.getElementType());
    SmallVector<NamedAttribute, 1> tAttrs;
    tAttrs.emplace_back(
        rewriter.getStringAttr("perm"), rewriter.getI64ArrayAttr({1, 0}));
    Operation *t = createGenericOp(
        rewriter, loc, "onnx.Transpose", TypeRange{outTy}, ValueRange{A}, tAttrs);
    AT = t->getResult(0);
  }

  Value xScaleC = buildScalarF32Constant(rewriter, loc, xScale);
  Value xZeroPointC =
      buildScalarZeroPointConstant(rewriter, loc, xZeroPoint, targetElemType);
  Value yScaleC = buildScalarF32Constant(rewriter, loc, yScale);
  Value yZeroPointC =
      buildScalarZeroPointConstant(rewriter, loc, yZeroPoint, targetElemType);
  Value wScaleC = buildScalarF32Constant(rewriter, loc, wScale);
  Value wZeroPointC =
      buildScalarZeroPointConstant(rewriter, loc, 0.0, targetElemType);

  bool isF8 = isF8ElemType(targetElemType);
  Value aQ = isF8 ? buildQuantizeCastF8(rewriter, loc, AT, xScaleC,
                        buildScalarF32Constant(rewriter, loc, xZeroPoint),
                        targetElemType)
                  : buildQuantizeLinear(
                        rewriter, loc, AT, xScaleC, xZeroPointC, targetElemType);
  Value bQ = buildQuantizedWeightConstant(rewriter, loc, bAttr, wScale, targetElemType);

  auto atTy = mlir::cast<RankedTensorType>(AT.getType());
  auto bqTy = mlir::cast<RankedTensorType>(bQ.getType());
  int64_t M = atTy.getShape()[0], N = bqTy.getShape()[1];
  auto qmmResultTy = RankedTensorType::get({M, N}, targetElemType);

  Operation *qmmOp = createGenericOp(rewriter, loc, "onnx.QLinearMatMul",
      TypeRange{qmmResultTy},
      ValueRange{aQ, xScaleC, xZeroPointC, bQ, wScaleC, wZeroPointC, yScaleC,
          yZeroPointC},
      {});

  rewriter.setInsertionPointAfter(qmmOp);
  auto matmulF32Ty = RankedTensorType::get({M, N}, rewriter.getF32Type());
  Value deq = isF8 ? buildDequantizeCastF8(rewriter, loc, qmmOp->getResult(0),
                         yScaleC, buildScalarF32Constant(rewriter, loc, yZeroPoint),
                         matmulF32Ty)
                   : buildDequantizeLinear(rewriter, loc, qmmOp->getResult(0),
                         yScaleC, yZeroPointC, matmulF32Ty);

  Value result = deq;
  if (hasC) {
    Value betaC = buildScalarF32Constant(rewriter, loc, beta);
    Operation *mulOp = createGenericOp(
        rewriter, loc, "onnx.Mul", TypeRange{C.getType()}, ValueRange{C, betaC}, {});
    Operation *addOp = createGenericOp(rewriter, loc, "onnx.Add",
        TypeRange{op->getResult(0).getType()},
        ValueRange{deq, mulOp->getResult(0)}, {});
    result = addOp->getResult(0);
  }

  rewriter.replaceAllUsesWith(op->getResult(0), result);
  rewriter.eraseOp(op);
  return true;
}

// Permute+flatten a [CO,C,KH,KW] f32 weight constant into [K,CO]
// (K=KH*KW*C), matching the (kh outer, kw next, c innermost) flattening
// order used by the runtime im2col unfold in retypeConvToFp8 below, and
// quantizing to f8 in the same pass. W is always a compile-time constant
// here, so this whole step -- which mirrors, in plain C++ over the
// attribute data, the transposeInt64({2,3,1,0})+reshape done at *runtime*
// for the activation side in src/Conversion/ONNXToKrnl/Math/QLinearConv.cpp
// -- avoids needing any runtime Transpose/Reshape ops for the weight at
// all.
// `coStart`/`coCount` select a contiguous range of W's leading (CO) dim --
// used to pull out one group's output-channel slice of a `group`-attribute
// weight tensor shaped [CO, C/group, KH, KW] without needing a runtime Slice
// (W is always a compile-time constant here, so this is plain C++ indexing
// into the constant's flat data). group==1 callers just pass coStart=0,
// coCount=CO, i.e. the whole tensor -- same code path either way.
static DenseElementsAttr quantizeAndReshapeConvWeightToF8(
    DenseElementsAttr wAttr, float wScale, Type f8Ty, int64_t coStart,
    int64_t coCount, int64_t C, int64_t KH, int64_t KW) {
  auto f8TensorTy = RankedTensorType::get({KH * KW * C, coCount}, f8Ty);
  SmallVector<float, 64> in(
      wAttr.getValues<float>().begin(), wAttr.getValues<float>().end());
  SmallVector<Attribute, 64> out(KH * KW * C * coCount);
  Builder b(f8Ty.getContext());
  // in index (row-major [CO,C,KH,KW]): (((coStart+co)*C+c)*KH+kh)*KW+kw
  // out index (row-major [K,coCount], K=(kh,kw,c) flattened): (kh*KW+kw)*C+c, *coCount+co
  for (int64_t co = 0; co < coCount; ++co)
    for (int64_t c = 0; c < C; ++c)
      for (int64_t kh = 0; kh < KH; ++kh)
        for (int64_t kw = 0; kw < KW; ++kw) {
          int64_t inIdx = (((coStart + co) * C + c) * KH + kh) * KW + kw;
          int64_t kIdx = (kh * KW + kw) * C + c;
          int64_t outIdx = kIdx * coCount + co;
          out[outIdx] = b.getFloatAttr(f8Ty, in[inIdx] / wScale);
        }
  return DenseElementsAttr::get(f8TensorTy, out);
}

// Rewrite a single onnx.Conv node into an im2col unfold (Slice/Concat/
// Reshape/Transpose, quantizing X to f8 first) -> onnx.QLinearMatMul ->
// dequantize-cast -> [+bias]. Unlike the int8 path (retypeConvToInt8,
// which builds an onnx.QLinearConv and lets a dedicated Krnl lowering
// pattern in src/Conversion/ONNXToKrnl/Math/QLinearConv.cpp do the im2col),
// this does the im2col decomposition directly at the ONNX-graph level here:
// onnx.QLinearConv has no opset version with float8-typed operands at all
// (unlike QLinearMatMul, which got a real opset-21 upgrade -- see the
// project plan), so it structurally cannot carry f8 X/W/Y, and there was no
// way to reuse it for this case. The im2col formulas themselves (padding,
// per-tap slicing, tap concatenation order) are the exact same ones already
// validated in QLinearConv.cpp, just built here via OnnxBuilder directly on
// ONNX-dialect tensors instead of inside a Krnl conversion pattern.
//
// Bias here is the *original* float32 onnx.Conv bias (not a pre-quantized
// i32 value like QLinearConv's spec-mandated bias), so unlike
// retypeConvToInt8 it's simply added in f32 after dequantization -- the
// same "add C after dequant" pattern retypeMatMulLikeToQuant already uses
// for Gemm's bias.
//
// Scope: 4D NCHW, auto_pad==NOTSET, static shapes, W a compile-time constant
// -- the same restrictions as retypeConvToInt8 / QLinearConv.cpp, which come
// from the im2col approach itself rather than from int8/fp8 specifically.
// group>=1 (including depthwise) is supported, mirroring the restructuring
// already validated for the int8 kernel in QLinearConv.cpp: X is
// channel-sliced per group (runtime onnx.Slice, since X is only known at
// im2col-unfold time), W is channel-sliced per group directly in C++ over
// the constant attribute data (see quantizeAndReshapeConvWeightToF8's
// coStart/coCount), and each group's QLinearMatMul+dequant result is
// concatenated back along the CO axis before bias is added. Unlike the int8
// kernel, w_scale/w_zero_point/y_scale/y_zero_point stay single per-tensor
// scalars shared by every group (matching this function's pre-existing
// group==1 design, which never supported per-channel scale/zero-point to
// begin with), and bias -- being plain f32, not a pre-quantized i32 value --
// is added exactly once, after the per-group results are concatenated,
// rather than needing to be sliced and added per group before rescaling.
static bool retypeConvToFp8(IRRewriter &rewriter, Operation *op,
    ArrayRef<double> params, StringRef nodeName, Type f8ElemType) {
  if (params.size() != 4) {
    op->emitWarning() << "LOWP_NODE_FORMATS entry for '" << nodeName
                       << "' needs exactly 4 params "
                          "(x_scale:x_zero_point:y_scale:y_zero_point); "
                          "skipping.";
    return false;
  }
  Value X = op->getOperand(0);
  Value W = op->getOperand(1);
  Value Bias = op->getOperand(2);
  bool hasBias = !llvm::isa<NoneType>(Bias.getType());

  if (auto ap = op->getAttrOfType<StringAttr>("auto_pad"))
    if (ap.getValue() != "NOTSET") {
      op->emitWarning() << "LOWP_NODE_FORMATS fp8 Conv only supports "
                            "auto_pad=NOTSET; skipping '"
                         << nodeName << "'.";
      return false;
    }
  int64_t group = 1;
  if (auto g = op->getAttrOfType<IntegerAttr>("group"))
    group = g.getSInt();

  auto xTy = mlir::dyn_cast<RankedTensorType>(X.getType());
  auto wTy = mlir::dyn_cast<RankedTensorType>(W.getType());
  if (!xTy || !wTy || xTy.getRank() != 4 || wTy.getRank() != 4 ||
      !xTy.hasStaticShape() || !wTy.hasStaticShape()) {
    op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                       << "' for fp8 but it isn't a statically-shaped 4D "
                          "NCHW Conv; skipping.";
    return false;
  }
  DenseElementsAttr wAttr = getConstantValueAttr(W);
  if (!wAttr) {
    op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                       << "' for fp8 but its weight is not a compile-time "
                          "onnx.Constant; skipping.";
    return false;
  }

  ArrayRef<int64_t> xShape = xTy.getShape(); // [N, C, H, Win]
  ArrayRef<int64_t> wShape = wTy.getShape(); // [CO, C/group, KH, KW]
  int64_t N = xShape[0], C = xShape[1], H = xShape[2], Win = xShape[3];
  int64_t CO = wShape[0], KH = wShape[2], KW = wShape[3];
  if (group < 1 || C % group != 0 || CO % group != 0) {
    op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                       << "' for fp8: group must evenly divide both the "
                          "input and output channel counts; skipping.";
    return false;
  }
  int64_t CPerGroup = C / group;
  int64_t COPerGroup = CO / group;
  if (wShape[1] != CPerGroup) {
    op->emitWarning() << "LOWP_NODE_FORMATS names '" << nodeName
                       << "' for fp8: W's channel-in dim must equal X's "
                          "channel dim divided by group; skipping.";
    return false;
  }

  auto getIntArrayAttr = [&](StringRef name,
                              SmallVectorImpl<int64_t> &out) -> bool {
    if (auto a = op->getAttrOfType<ArrayAttr>(name)) {
      // ArrayAttrIntVals appends (emplace_back) rather than replacing, so
      // callers pre-populating `out` with a default must clear it first, or
      // the real values just get appended after the stale defaults.
      out.clear();
      ArrayAttrIntVals(a, out);
      return true;
    }
    return false;
  };
  SmallVector<int64_t, 4> kernelShape{KH, KW};
  SmallVector<int64_t, 4> ks;
  if (getIntArrayAttr("kernel_shape", ks)) {
    if (ks.size() != 2) {
      op->emitWarning() << "LOWP_NODE_FORMATS fp8 Conv only supports 2D "
                            "kernel_shape; skipping '"
                         << nodeName << "'.";
      return false;
    }
    kernelShape = ks;
  }
  SmallVector<int64_t, 4> strides{1, 1};
  getIntArrayAttr("strides", strides);
  SmallVector<int64_t, 4> dilations{1, 1};
  getIntArrayAttr("dilations", dilations);
  SmallVector<int64_t, 4> pads{0, 0, 0, 0};
  getIntArrayAttr("pads", pads);

  int64_t padHBegin = pads[0], padWBegin = pads[1];
  int64_t padHEnd = pads[2], padWEnd = pads[3];
  int64_t HPad = H + padHBegin + padHEnd;
  int64_t WPad = Win + padWBegin + padWEnd;
  int64_t HO = (HPad - dilations[0] * (kernelShape[0] - 1) - 1) / strides[0] + 1;
  int64_t WO = (WPad - dilations[1] * (kernelShape[1] - 1) - 1) / strides[1] + 1;

  float xScale = static_cast<float>(params[0]);
  double xZeroPoint = params[1];
  float yScale = static_cast<float>(params[2]);
  double yZeroPoint = params[3];
  float wScale = computeSymmetricScaleF8(wAttr, f8ElemType);

  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);
  OnnxBuilder create(rewriter, loc);

  Value xScaleC = buildScalarF32Constant(rewriter, loc, xScale);
  Value xZeroPointF32 =
      buildScalarF32Constant(rewriter, loc, static_cast<float>(xZeroPoint));
  Value xZeroPointC =
      buildScalarF8Constant(rewriter, loc, xZeroPoint, f8ElemType);
  Value yScaleC = buildScalarF32Constant(rewriter, loc, yScale);
  Value yZeroPointF32 =
      buildScalarF32Constant(rewriter, loc, static_cast<float>(yZeroPoint));
  Value wScaleC = buildScalarF32Constant(rewriter, loc, wScale);
  Value wZeroPointC = buildScalarF8Constant(rewriter, loc, 0.0, f8ElemType);
  Value yZeroPointC =
      buildScalarF8Constant(rewriter, loc, yZeroPoint, f8ElemType);

  // --- Pad X (if needed), using x_zero_point's real value as the fill. ---
  // Unlike QLinearConv.cpp's int8 im2col (which operates on already-
  // quantized int8 data, since onnx.Slice's ONNX-spec type list includes
  // int8), im2col here runs on the *original f32* X: onnx.Slice/Concat's
  // type constraint (from the ONNX op spec onnx-mlir imports) doesn't
  // include float8 at all, only the standard int/float types, so an
  // f8-typed Slice/Concat can't legally be constructed. Quantizing X to f8
  // only *after* the unfold (a few lines below) instead is mathematically
  // equivalent -- im2col is pure data movement/reordering, no arithmetic --
  // and sidesteps the type constraint entirely.
  Value paddedX = X;
  if (padHBegin || padWBegin || padHEnd || padWEnd) {
    Value padsVal = create.constantInt64(
        {0, 0, padHBegin, padWBegin, 0, 0, padHEnd, padWEnd});
    paddedX = create.pad(X, padsVal, xZeroPointF32, "constant");
  }

  // --- Per-group im2col (in f32, same formulas as QLinearConv.cpp's int8
  // path) + quantized matmul. Each group only ever sees its own slice of
  // input channels (channel-sliced from paddedX) and output channels
  // (channel-sliced from W's constant data); group==1 (the common case)
  // takes exactly the same path as before, just with a trivial
  // single-iteration loop around it. ---
  int64_t M = N * HO * WO;
  int64_t K = kernelShape[0] * kernelShape[1] * CPerGroup;
  auto matmulF32Ty = RankedTensorType::get({M, CO}, rewriter.getF32Type());
  SmallVector<Value, 4> groupResults;
  for (int64_t g = 0; g < group; ++g) {
    Value paddedXg = paddedX;
    if (group > 1) {
      auto xgTy = RankedTensorType::get(
          {N, CPerGroup, HPad, WPad}, rewriter.getF32Type());
      paddedXg = create.slice(xgTy, paddedX,
          create.constantInt64({g * CPerGroup}),
          create.constantInt64({(g + 1) * CPerGroup}),
          create.constantInt64({1}), create.constantInt64({1}));
    }

    auto tapType =
        RankedTensorType::get({N, CPerGroup, HO, WO}, rewriter.getF32Type());
    SmallVector<Value, 16> taps;
    for (int64_t kh = 0; kh < kernelShape[0]; ++kh) {
      int64_t hStart = kh * dilations[0];
      int64_t hEnd = hStart + (HO - 1) * strides[0] + 1;
      for (int64_t kw = 0; kw < kernelShape[1]; ++kw) {
        int64_t wStart = kw * dilations[1];
        int64_t wEnd = wStart + (WO - 1) * strides[1] + 1;
        Value tap = create.slice(tapType, paddedXg,
            create.constantInt64({hStart, wStart}),
            create.constantInt64({hEnd, wEnd}), create.constantInt64({2, 3}),
            create.constantInt64({strides[0], strides[1]}));
        taps.push_back(create.transposeInt64(tap, {0, 2, 3, 1}));
      }
    }
    Value unfolded =
        taps.size() == 1 ? taps[0]
                          : create.concat(RankedTensorType::get({N, HO, WO, K},
                                              rewriter.getF32Type()),
                                taps, /*axis=*/3);
    Value AMatF32 = create.reshape(
        RankedTensorType::get({M, K}, rewriter.getF32Type()), unfolded,
        create.constantInt64({M, K}));
    Value AMat = buildQuantizeCastF8(
        rewriter, loc, AMatF32, xScaleC, xZeroPointF32, f8ElemType);

    Value BMat = buildConstant(rewriter, loc,
        quantizeAndReshapeConvWeightToF8(wAttr, wScale, f8ElemType,
            g * COPerGroup, COPerGroup, CPerGroup, KH, KW));

    auto qmmResultTy = RankedTensorType::get({M, COPerGroup}, f8ElemType);
    Operation *qmmOp = createGenericOp(rewriter, loc, "onnx.QLinearMatMul",
        TypeRange{qmmResultTy},
        ValueRange{AMat, xScaleC, xZeroPointC, BMat, wScaleC, wZeroPointC,
            yScaleC, yZeroPointC},
        {});

    rewriter.setInsertionPointAfter(qmmOp);
    auto groupF32Ty = RankedTensorType::get({M, COPerGroup}, rewriter.getF32Type());
    Value deq = buildDequantizeCastF8(rewriter, loc, qmmOp->getResult(0),
        yScaleC, yZeroPointF32, groupF32Ty);
    groupResults.push_back(deq);
  }

  Value result = groupResults.size() == 1
                     ? groupResults[0]
                     : create.concat(matmulF32Ty, groupResults, /*axis=*/1);
  if (hasBias) {
    Operation *addOp = createGenericOp(rewriter, loc, "onnx.Add",
        TypeRange{matmulF32Ty}, ValueRange{result, Bias}, {});
    result = addOp->getResult(0);
  }

  // [M,CO] = [N*HO*WO,CO] -> [N,HO,WO,CO] -> [N,CO,HO,WO] (NCHW).
  Value res4d = create.reshape(
      RankedTensorType::get({N, HO, WO, CO}, rewriter.getF32Type()), result,
      create.constantInt64({N, HO, WO, CO}));
  Value resNCHW = create.transposeInt64(res4d, {0, 3, 1, 2});

  rewriter.replaceAllUsesWith(op->getResult(0), resNCHW);
  rewriter.eraseOp(op);
  return true;
}

struct ConvertONNXToLowPrecisionPass
    : public PassWrapper<ConvertONNXToLowPrecisionPass,
          OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToLowPrecisionPass)

  StringRef getArgument() const final { return "convert-onnx-to-lowprecision"; }
  StringRef getDescription() const final {
    return "Retype LOWP_NODE_FORMATS-selected ONNX nodes to BF16/F16 (via "
           "onnx.Cast), INT8 (Conv via QuantizeLinear/QLinearConv, Gemm/"
           "MatMul via QLinearMatMul/DequantizeLinear), or FP8E4M3/FP8E5M2 "
           "(Gemm/MatMul via QLinearMatMul; Conv via an im2col decomposition "
           "targeting QLinearMatMul directly, since QLinearConv has no "
           "float8-capable opset to target). Relu/Add/Sub/Mul/Div/"
           "AveragePool/GlobalAveragePool under INT8/FP8E4M3/FP8E5M2 are "
           "QDQ-round-tripped only (no quantized op exists for them in "
           "ONNX), simulating per-operand precision loss while the "
           "arithmetic itself stays in f32.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &ctx = getContext();
    IRRewriter rewriter(&ctx);

    std::map<std::string, NodeFormatEntry> formatMap = parseNodeFormats();
    if (formatMap.empty())
      return;

    // Collect targets first: rewriting in-place while walking would
    // invalidate the walk (old ops get erased and replaced).
    SmallVector<Operation *, 16> targets;
    module.walk([&](Operation *op) {
      auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name");
      if (!nameAttr)
        return;
      auto it = formatMap.find(nameAttr.getValue().str());
      if (it == formatMap.end())
        return;
      LowPrecisionFormat fmt = it->second.format;
      StringRef opName = op->getName().getStringRef();
      bool isMatMulLike = opName == "onnx.Gemm" || opName == "onnx.MatMul";
      bool supported;
      if (fmt == LowPrecisionFormat::INT8)
        supported =
            opName == "onnx.Conv" || isMatMulLike || isElementwiseQuantOp(opName);
      else if (fmt == LowPrecisionFormat::FP8E4M3 ||
               fmt == LowPrecisionFormat::FP8E5M2)
        // Conv goes through retypeConvToFp8's own im2col decomposition
        // (onnx.QLinearConv has no float8-capable opset to target, unlike
        // QLinearMatMul), not a QLinearConv-based Krnl pattern.
        supported =
            opName == "onnx.Conv" || isMatMulLike || isElementwiseQuantOp(opName);
      else
        supported = isSupportedOpForCastRetype(op);
      if (!supported) {
        op->emitWarning()
            << "LOWP_NODE_FORMATS names node '" << nameAttr.getValue()
            << "' but its op type ('" << op->getName()
            << "') is not supported for the requested format; skipping.";
        return;
      }
      targets.push_back(op);
    });

    for (Operation *op : targets) {
      auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name");
      StringRef nodeName = nameAttr.getValue();
      const NodeFormatEntry &entry = formatMap[nodeName.str()];
      StringRef opName = op->getName().getStringRef();
      switch (entry.format) {
      case LowPrecisionFormat::INT8:
        if (opName == "onnx.Conv")
          retypeConvToInt8(rewriter, op, entry.quantParams, nodeName);
        else if (isElementwiseQuantOp(opName))
          retypeElementwiseToQuant(
              rewriter, op, entry.quantParams, nodeName, rewriter.getI8Type());
        else
          retypeMatMulLikeToQuant(
              rewriter, op, entry.quantParams, nodeName, rewriter.getI8Type());
        break;
      case LowPrecisionFormat::FP8E4M3:
        if (opName == "onnx.Conv")
          retypeConvToFp8(rewriter, op, entry.quantParams, nodeName,
              Float8E4M3FNType::get(&ctx));
        else if (isElementwiseQuantOp(opName))
          retypeElementwiseToQuant(rewriter, op, entry.quantParams, nodeName,
              Float8E4M3FNType::get(&ctx));
        else
          retypeMatMulLikeToQuant(rewriter, op, entry.quantParams, nodeName,
              Float8E4M3FNType::get(&ctx));
        break;
      case LowPrecisionFormat::FP8E5M2:
        if (opName == "onnx.Conv")
          retypeConvToFp8(rewriter, op, entry.quantParams, nodeName,
              Float8E5M2Type::get(&ctx));
        else if (isElementwiseQuantOp(opName))
          retypeElementwiseToQuant(rewriter, op, entry.quantParams, nodeName,
              Float8E5M2Type::get(&ctx));
        else
          retypeMatMulLikeToQuant(rewriter, op, entry.quantParams, nodeName,
              Float8E5M2Type::get(&ctx));
        break;
      default:
        retypeCastOp(rewriter, op, entry.format, nodeName);
        break;
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertONNXToLowPrecisionPass() {
  return std::make_unique<ConvertONNXToLowPrecisionPass>();
}

} // namespace onnx_mlir
