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
  // INT8/FP8E4M3/FP8E5M2 only: [x_scale, x_zero_point, y_scale,
  // y_zero_point], supplied via LOWP_NODE_FORMATS as
  // "Conv_1:int8:<x_scale>:<x_zero_point>:<y_scale>:<y_zero_point>". For
  // int8, x_zero_point/y_zero_point are integer values; for the fp8 formats
  // they're real values (almost always 0.0 in practice -- see the fp8
  // quantize/dequantize helpers below). Weight/bias quantization params are
  // derived from the weight/bias constants themselves (see the
  // quantizeConstant* helpers below), not supplied here, since they're
  // static and don't need calibration.
  SmallVector<double, 4> quantParams;
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

struct ConvertONNXToLowPrecisionPass
    : public PassWrapper<ConvertONNXToLowPrecisionPass,
          OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToLowPrecisionPass)

  StringRef getArgument() const final { return "convert-onnx-to-lowprecision"; }
  StringRef getDescription() const final {
    return "Retype LOWP_NODE_FORMATS-selected ONNX nodes to BF16/F16 (via "
           "onnx.Cast), INT8 (via QuantizeLinear/QLinearConv or "
           "QLinearMatMul/DequantizeLinear; Conv, Gemm, MatMul), or "
           "FP8E4M3/FP8E5M2 (QuantizeLinear/QLinearMatMul/DequantizeLinear; "
           "Gemm, MatMul only -- no QLinearConv-equivalent fp8 kernel yet).";
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
        // QLinearConv.cpp only supports int8; the fp8 formats don't have a
        // QLinearConv-equivalent kernel yet (only QLinearMatMul's opset-21
        // float8 branch), so Conv isn't offered for fp8e4m3/fp8e5m2 below.
        supported = opName == "onnx.Conv" || isMatMulLike;
      else if (fmt == LowPrecisionFormat::FP8E4M3 ||
               fmt == LowPrecisionFormat::FP8E5M2)
        supported = isMatMulLike;
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
      switch (entry.format) {
      case LowPrecisionFormat::INT8:
        if (op->getName().getStringRef() == "onnx.Conv")
          retypeConvToInt8(rewriter, op, entry.quantParams, nodeName);
        else
          retypeMatMulLikeToQuant(
              rewriter, op, entry.quantParams, nodeName, rewriter.getI8Type());
        break;
      case LowPrecisionFormat::FP8E4M3:
        retypeMatMulLikeToQuant(rewriter, op, entry.quantParams, nodeName,
            Float8E4M3FNType::get(&ctx));
        break;
      case LowPrecisionFormat::FP8E5M2:
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
