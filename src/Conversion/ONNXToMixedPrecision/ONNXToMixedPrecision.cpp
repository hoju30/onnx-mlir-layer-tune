#include "onnx-mlir/Conversion/ONNXToMixedPrecision/ONNXToMixedPrecision.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Conversion/ONNXToMixedPrecision/MixedPrecisionBackward.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositLut.hpp"
#include "src/Dialect/Posit/PositOps.h"
#include "src/Dialect/Posit/PositTypes.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

using namespace mlir;

namespace onnx_mlir {
namespace {

// ===========================================================================
// Format spec: family + (bits) for INT, (nbits, es) for POSIT. F16/BF16/FP32
// need no extra params. Unlike the single-format-per-node
// --convert-onnx-to-lowprecision pass, here weight and activation each carry
// their own independent FormatSpec -- see MixedNodeFormatEntry below.
//
// Every format here is realized purely via QDQ (quantize-then-immediately-
// dequantize) simulation: the real op always runs on real f32 operands, only
// the *storage* of a value is simulated as if it had been rounded through
// the target format. This is a deliberate generalization of the
// quantDequantRoundTrip idea already used for Relu/Add/Sub/Mul/Div in
// ONNXToLowPrecision.cpp, extended to Conv/Gemm/MatMul too and to a much
// larger format set (int4-int8, posit p<N>e<ES>, f16, bf16) that has no
// native ONNX quantized-op or Krnl-kernel support to route through -- QDQ
// simulation sidesteps needing any of that.
// ===========================================================================

enum class FormatFamily { FP32, INT, POSIT, F16, BF16 };

struct FormatSpec {
  FormatFamily family = FormatFamily::FP32;
  unsigned bits = 0; // INT: bit width (4-8). POSIT: nbits.
  unsigned es = 0;   // POSIT only: exponent bits.
};

static bool parseFormatTag(StringRef tag, FormatSpec &spec) {
  if (tag == "fp32") {
    spec = {FormatFamily::FP32, 0, 0};
    return true;
  }
  if (tag == "f16") {
    spec = {FormatFamily::F16, 16, 0};
    return true;
  }
  if (tag == "bf16") {
    spec = {FormatFamily::BF16, 16, 0};
    return true;
  }
  if (tag.starts_with("int")) {
    unsigned bits;
    if (!tag.drop_front(3).getAsInteger(10, bits) && bits >= 4 && bits <= 8) {
      spec = {FormatFamily::INT, bits, 0};
      return true;
    }
    return false;
  }
  if (tag.starts_with("p")) {
    // p<N>e<ES>, e.g. "p8e1" -> nbits=8, es=1.
    StringRef rest = tag.drop_front(1);
    size_t ePos = rest.find('e');
    if (ePos == StringRef::npos)
      return false;
    unsigned n, es;
    if (rest.take_front(ePos).getAsInteger(10, n))
      return false;
    if (rest.drop_front(ePos + 1).getAsInteger(10, es))
      return false;
    // PositType's own verifier requires es < nbits; violating it aborts in
    // PositType::get rather than failing gracefully, so reject it here.
    if (n < 2 || n > 32 || es >= n)
      return false;
    spec = {FormatFamily::POSIT, n, es};
    return true;
  }
  return false;
}

// ===========================================================================
// MIXEDP_NODE_FORMATS parsing:
//   <node_name>:w=<weight_fmt>:a=<activation_fmt>[:<params>],...
// `w=` is only meaningful for ops with a real weight operand (Conv/Gemm/
// MatMul) and defaults to fp32 (weight left untouched) if omitted. `a=`
// applies to every other float operand and to the result; extra numeric
// params are needed only when activation_fmt is int4..int8 (needs
// calibration, same convention as LOWP_NODE_FORMATS): 4 params
// (x_scale:x_zero_point:y_scale:y_zero_point) for single-input ops, 6
// (a_scale:a_zero_point:b_scale:b_zero_point:y_scale:y_zero_point) for the
// two-input elementwise ops -- mirrors ONNXToLowPrecision.cpp's
// retypeElementwiseToQuant convention exactly. Posit/f16/bf16/fp32
// activation formats take no params (posit self-scales; f16/bf16/fp32 need
// no external scale at all).
// ===========================================================================

struct MixedNodeFormatEntry {
  FormatSpec weightFormat;     // default fp32 == weight left untouched
  FormatSpec activationFormat; // default fp32 == activation left untouched
  SmallVector<double, 6> activationParams;
  // Set when a "w="/"a=" field's format tag failed to parse (e.g. a typo
  // like "fp16" instead of "f16"), so the node it belongs to can still warn
  // even though weightFormat/activationFormat silently kept their fp32
  // default -- without this, a mistyped format would apply no quantization
  // at all with no indication anything was wrong.
  bool weightFormatInvalid = false;
  bool activationFormatInvalid = false;
};

static std::map<std::string, MixedNodeFormatEntry> parseMixedNodeFormats() {
  std::map<std::string, MixedNodeFormatEntry> result;
  const char *e = std::getenv("MIXEDP_NODE_FORMATS");
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
    if (c1 == std::string::npos) {
      pos = comma + 1;
      continue;
    }
    std::string name = entry.substr(0, c1);
    std::string rest = entry.substr(c1 + 1);

    MixedNodeFormatEntry nfe;
    size_t rp = 0;
    while (rp < rest.size()) {
      size_t fc = rest.find(':', rp);
      if (fc == std::string::npos)
        fc = rest.size();
      std::string field = rest.substr(rp, fc - rp);
      FormatSpec spec;
      if (field.starts_with("w=")) {
        if (parseFormatTag(field.substr(2), spec))
          nfe.weightFormat = spec;
        else
          nfe.weightFormatInvalid = true;
      } else if (field.starts_with("a=")) {
        if (parseFormatTag(field.substr(2), spec))
          nfe.activationFormat = spec;
        else
          nfe.activationFormatInvalid = true;
      } else if (!field.empty()) {
        // A malformed numeric field (e.g. "oops", or "1e9999" overflowing
        // double range) would otherwise throw std::invalid_argument/
        // std::out_of_range straight out of this env-var parser and abort
        // the whole compiler before any diagnostic could be printed.
        // Substituting NaN preserves the parameter *count* (so a bad middle
        // parameter doesn't shift the meaning of the ones after it) and
        // lets the existing activation-params validation in
        // retypeNodeMixedQDQ reject it with a proper warning instead.
        double v = std::numeric_limits<double>::quiet_NaN();
        try {
          v = std::stod(field);
        } catch (const std::exception &) {
        }
        nfe.activationParams.push_back(v);
      }
      rp = fc + 1;
    }
    result[name] = nfe;
    pos = comma + 1;
  }
  return result;
}

// ===========================================================================
// Small IR-building helpers, self-contained in this file (mirrors the same
// helpers in ONNXToLowPrecision.cpp -- duplicated rather than shared since
// there is no common utility header today and this pass has a different
// dialect dependency footprint, see CMakeLists.txt).
// ===========================================================================

static Operation *createGenericOp(IRRewriter &rewriter, Location loc,
    StringRef name, TypeRange resultTypes, ValueRange operands,
    ArrayRef<NamedAttribute> attrs) {
  OperationState st(loc, name);
  st.addOperands(operands);
  st.addTypes(resultTypes);
  st.addAttributes(attrs);
  return rewriter.create(st);
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

static DenseElementsAttr getConstantValueAttr(Value v) {
  Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return nullptr;
  return llvm::dyn_cast_or_null<DenseElementsAttr>(def->getAttr("value"));
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

// ===========================================================================
// INT{4..8} fake-quant: pure f32 arithmetic (div -> round -> clip -> mul),
// never touching a real low-bit MLIR type at all. Deliberately avoids
// onnx.QuantizeLinear/DequantizeLinear, whose Krnl lowering is hardcoded to
// int8/uint8 only (see docs/LowPrecisionFormats.md) and so cannot represent
// int4..int7. Symmetric quant, scale=max_abs/qmax, matching the convention
// ONNXToLowPrecision.cpp's computeSymmetricScale already uses for int8,
// generalized to a parametrized bit width.
// ===========================================================================

// Returns nullptr if the weight can't be safely quantized this way at all
// (every element non-finite, or the computed scale itself isn't a usable
// positive finite number -- e.g. maxAbs a tiny subnormal that underflows to
// exactly 0.0f on division by qmax); the caller treats that the same as an
// unsupported format and leaves the weight untouched. Individual non-finite
// elements (+-Inf/NaN) are passed through unchanged rather than fed into the
// round/clamp/rescale arithmetic, and an element-wise rescale that itself
// overflows to +-Inf (possible in principle when maxAbs is near FLT_MAX,
// since dividing then multiplying back by the same float-rounded scale
// doesn't exactly invert) falls back to the original value -- both confirmed
// by direct testing to otherwise corrupt an originally-finite weight.
// Every fakeQuantWeight* helper below returns the fake-quantized constant, or,
// given `sink`, moves the values there instead and returns `floatAttr` (non-null
// for success) without creating an attribute: MLIR keeps every attribute alive
// in the context, so a caller that only needs the values (the statistics pass's
// weight errors, 10 formats x every weight) must not create one per format.
static DenseElementsAttr fakeQuantResult(DenseElementsAttr floatAttr,
    SmallVectorImpl<float> &out, SmallVectorImpl<float> *sink) {
  if (sink) {
    sink->swap(out);
    return floatAttr;
  }
  return DenseElementsAttr::get(floatAttr.getType(), ArrayRef<float>(out));
}

static DenseElementsAttr fakeQuantWeightInt(
    DenseElementsAttr floatAttr, unsigned bits,
    SmallVectorImpl<float> *sink = nullptr) {
  int32_t qmax = (1 << (bits - 1)) - 1;
  int32_t qmin = -(1 << (bits - 1));
  float maxAbs = 0.0f;
  for (float f : floatAttr.getValues<float>())
    if (std::isfinite(f))
      maxAbs = std::max(maxAbs, std::fabs(f));
  float scale = maxAbs > 0.0f ? maxAbs / static_cast<float>(qmax) : 1.0f;
  if (!std::isfinite(scale) || scale <= 0.0f)
    return nullptr;
  SmallVector<float, 64> out;
  out.reserve(floatAttr.getNumElements());
  for (float f : floatAttr.getValues<float>()) {
    if (!std::isfinite(f)) {
      out.push_back(f);
      continue;
    }
    int32_t q = static_cast<int32_t>(std::lround(f / scale));
    q = std::clamp(q, qmin, qmax);
    float dq = static_cast<float>(q) * scale;
    out.push_back(std::isfinite(dq) ? dq : f);
  }
  return fakeQuantResult(floatAttr, out, sink);
}

// Per-output-channel variant of fakeQuantWeightInt: the same symmetric
// round-half-away-from-zero arithmetic, but one scale per index along
// `channelAxis` (max|w| over that channel / qmax). Per-tensor scales round whole
// output channels to zero when channel magnitudes differ widely (BN-folded
// depthwise convolutions); per-channel scales do not. A channel with no finite
// nonzero value keeps scale 1, as the per-tensor helper does for the tensor.
static DenseElementsAttr fakeQuantWeightIntPerChannel(
    DenseElementsAttr floatAttr, unsigned bits, int64_t channelAxis,
    SmallVectorImpl<float> *sink = nullptr) {
  auto floatTy = mlir::cast<RankedTensorType>(floatAttr.getType());
  ArrayRef<int64_t> shape = floatTy.getShape();
  if (channelAxis < 0 || channelAxis >= static_cast<int64_t>(shape.size()))
    return nullptr;
  int64_t channels = shape[channelAxis];
  int64_t inner = 1;
  for (size_t d = channelAxis + 1; d < shape.size(); ++d)
    inner *= shape[d];
  int32_t qmax = (1 << (bits - 1)) - 1;
  int32_t qmin = -(1 << (bits - 1));
  SmallVector<float, 64> values(floatAttr.getValues<float>());
  SmallVector<float, 64> maxAbs(channels, 0.0f);
  for (size_t i = 0; i < values.size(); ++i)
    if (std::isfinite(values[i])) {
      int64_t c = (static_cast<int64_t>(i) / inner) % channels;
      maxAbs[c] = std::max(maxAbs[c], std::fabs(values[i]));
    }
  SmallVector<float, 64> scale(channels, 1.0f);
  for (int64_t c = 0; c < channels; ++c) {
    if (maxAbs[c] > 0.0f)
      scale[c] = maxAbs[c] / static_cast<float>(qmax);
    if (!std::isfinite(scale[c]) || scale[c] <= 0.0f)
      return nullptr;
  }
  for (size_t i = 0; i < values.size(); ++i) {
    float f = values[i];
    if (!std::isfinite(f))
      continue;
    float s = scale[(static_cast<int64_t>(i) / inner) % channels];
    int32_t q = static_cast<int32_t>(std::lround(f / s));
    q = std::clamp(q, qmin, qmax);
    float dq = static_cast<float>(q) * s;
    values[i] = std::isfinite(dq) ? dq : f;
  }
  return fakeQuantResult(floatAttr, values, sink);
}

// MIXEDP_WEIGHT_PER_CHANNEL=1 switches intN weight quantization from one scale
// per tensor (the default) to one scale per output channel.
static bool weightPerChannelEnabled() {
  const char *e = std::getenv("MIXEDP_WEIGHT_PER_CHANNEL");
  return e && (std::string(e) == "1" || std::string(e) == "true");
}

// Output-channel axis of an op's weight operand, or -1 (use per-tensor):
// Conv W is [M, C/group, k...] -> 0; Gemm B is [K, N] -> 1, or [N, K] with
// transB=1 -> 0; MatMul's constant right operand [..., K, N] -> last axis.
static int64_t weightOutputChannelAxis(Operation *op, ShapedType wTy) {
  StringRef name = op->getName().getStringRef();
  if (!wTy.hasRank())
    return -1;
  int64_t rank = wTy.getRank();
  if (name == "onnx.Conv")
    return rank >= 1 ? 0 : -1;
  if (name == "onnx.Gemm") {
    auto transB = op->getAttrOfType<IntegerAttr>("transB");
    // ONNX attributes are si64, so read them with getSExtValue (getInt asserts
    // a signless integer).
    return rank == 2 ? (transB && transB.getValue().getSExtValue() != 0 ? 0 : 1) : -1;
  }
  if (name == "onnx.MatMul")
    return rank >= 2 ? rank - 1 : -1;
  return -1;
}

// Runtime (activation) counterpart of fakeQuantWeightInt, built as IR ops:
// clip((x/scale)+zero_point, qmin, qmax) rounded, then unshift+rescale back
// to f32. Uses OnnxBuilder::round/clip (src/Dialect/ONNX/DialectBuilder.hpp)
// rather than hand-building those ops, and plain onnx.Div/Add/Sub/Mul via
// createGenericOp for the rest, matching the exact style
// ONNXToLowPrecision.cpp's buildQuantizeCastF8/buildDequantizeCastF8 already
// use for their own from-scratch f32 arithmetic.
// scaleSplat > 1: the scale as a splat constant of that length (broadcast
// along the last axis) instead of a scalar -- the same arithmetic, used by
// --mixedp-collect-stats to keep canonicalization from folding the Div into
// a preceding MatMul's weight.
static Value buildIntActivationQDQ(IRRewriter &rewriter, Location loc,
    Value x, double scale, double zeroPoint, unsigned bits,
    int64_t scaleSplat = 0) {
  OnnxBuilder create(rewriter, loc);
  float qmax = static_cast<float>((1 << (bits - 1)) - 1);
  float qmin = static_cast<float>(-(1 << (bits - 1)));
  Value scaleC =
      scaleSplat > 1
          ? buildConstant(rewriter, loc,
                DenseElementsAttr::get(
                    RankedTensorType::get({scaleSplat}, rewriter.getF32Type()),
                    static_cast<float>(scale)))
          : buildScalarF32Constant(rewriter, loc, static_cast<float>(scale));
  Value zpC = buildScalarF32Constant(rewriter, loc, static_cast<float>(zeroPoint));
  Value minC = buildScalarF32Constant(rewriter, loc, qmin);
  Value maxC = buildScalarF32Constant(rewriter, loc, qmax);

  Value divided = createGenericOp(rewriter, loc, "onnx.Div",
      TypeRange{x.getType()}, ValueRange{x, scaleC}, {})
                      ->getResult(0);
  Value shifted = createGenericOp(rewriter, loc, "onnx.Add",
      TypeRange{x.getType()}, ValueRange{divided, zpC}, {})
                      ->getResult(0);
  Value rounded = create.round(shifted);
  Value clipped = create.clip(rounded, minC, maxC);
  Value unshifted = createGenericOp(rewriter, loc, "onnx.Sub",
      TypeRange{x.getType()}, ValueRange{clipped, zpC}, {})
                         ->getResult(0);
  return createGenericOp(rewriter, loc, "onnx.Mul", TypeRange{x.getType()},
      ValueRange{unshifted, scaleC}, {})
      ->getResult(0);
}

// ===========================================================================
// POSIT p<N>e<ES> fake-quant.
//
// Weight (compile-time constant) path: LUT-accelerated for N<=8
// (p4e1/p6e1/p8e1) and bit-exact with sw::universal::posit<N,ES> -- the
// table of decision boundaries and how it is built live in
// src/Dialect/Posit/PositLut.cpp (shared with other passes); each weight
// element costs one binary search over at most 2^N-2 boundaries instead of a
// general posit encode.
//
// Activation (runtime tensor) path: applyActivationQDQ below inserts one
// posit.round_f32 op, which --convert-posit-to-krnl lowers to an inlined
// lookup in the same table (no runtime call, posit_runtime.cpp untouched).
// MIXEDP_POSIT_ACTIVATION=runtime -- and any (nbits, es) without a table --
// keeps the original posit.from_f32/to_f32 pair instead, executed by
// src/posit_runtime.cpp.
// ===========================================================================

static DenseElementsAttr fakeQuantWeightPosit(DenseElementsAttr floatAttr,
    unsigned nbits, unsigned es, SmallVectorImpl<float> *sink = nullptr) {
  const PositLut *lut = getPositLutCached(nbits, es);
  if (!lut)
    return nullptr;
  SmallVector<float, 64> out;
  out.reserve(floatAttr.getNumElements());
  for (float f : floatAttr.getValues<float>())
    out.push_back(fakeQuantPositViaLut(f, *lut));
  return fakeQuantResult(floatAttr, out, sink);
}

// ===========================================================================
// F16/BF16 weight fake-quant: APFloat round trip through the target type's
// own float semantics, same technique ONNXToLowPrecision.cpp's
// buildScalarF8Constant comment already documents relying on for F8 (MLIR's
// own FloatAttr/APFloat machinery implements correct low-precision
// rounding) -- simpler here since F16/BF16 are "plain" IEEE-shaped types.
// ===========================================================================

static DenseElementsAttr fakeQuantWeightFloat(DenseElementsAttr floatAttr,
    FloatType narrowTy, SmallVectorImpl<float> *sink = nullptr) {
  const llvm::fltSemantics &narrowSem = narrowTy.getFloatSemantics();
  SmallVector<float, 64> out;
  out.reserve(floatAttr.getNumElements());
  bool losesInfo;
  for (float f : floatAttr.getValues<float>()) {
    APFloat af(f);
    af.convert(narrowSem, APFloat::rmNearestTiesToEven, &losesInfo);
    af.convert(APFloat::IEEEsingle(), APFloat::rmNearestTiesToEven, &losesInfo);
    out.push_back(af.convertToFloat());
  }
  return fakeQuantResult(floatAttr, out, sink);
}

// channelAxis >= 0 selects per-output-channel scales for intN weights (ignored
// by the other formats, which have no scale).
static DenseElementsAttr fakeQuantWeight(DenseElementsAttr attr,
    const FormatSpec &spec, int64_t channelAxis = -1,
    SmallVectorImpl<float> *sink = nullptr) {
  if (spec.family == FormatFamily::FP32) {
    if (sink)
      sink->assign(attr.getValues<float>().begin(), attr.getValues<float>().end());
    return attr;
  }
  // Every fake-quant helper below assumes an f32-stored constant
  // (getValues<float>() asserts otherwise); a weight that's natively
  // f16/bf16/f64 needs the caller to have cast it to f32 first, which this
  // pass does not do. Treat that the same as any other unsupported format
  // rather than crashing.
  if (!mlir::cast<ShapedType>(attr.getType()).getElementType().isF32())
    return nullptr;
  switch (spec.family) {
  case FormatFamily::FP32:
    return attr;
  case FormatFamily::INT:
    return channelAxis >= 0 ? fakeQuantWeightIntPerChannel(
                                  attr, spec.bits, channelAxis, sink)
                            : fakeQuantWeightInt(attr, spec.bits, sink);
  case FormatFamily::POSIT:
    return fakeQuantWeightPosit(attr, spec.bits, spec.es, sink);
  case FormatFamily::F16:
    return fakeQuantWeightFloat(
        attr, Builder(attr.getContext()).getF16Type(), sink);
  case FormatFamily::BF16:
    return fakeQuantWeightFloat(
        attr, Builder(attr.getContext()).getBF16Type(), sink);
  }
  return nullptr;
}

// PositToKrnl's own to_f32 lowering (src/Conversion/PositToKrnl/Pattern/
// Math.cpp) folds a directly-adjacent `to_f32(from_f32(x))` straight to `x`
// whenever the from_f32 op's "qalign_key" attribute is absent or 0 -- a
// valid optimization for its own callers, but fatal for this pass's use:
// confirmed by direct testing that an activation-only posit QDQ (weight
// untouched) compiled and ran BIT-IDENTICAL to the plain FP32 baseline,
// i.e. the round trip was silently eliminated rather than simulating any
// precision loss. A nonzero qalign_key is PositToKrnl's own documented
// escape hatch for "this is a real round trip, don't fold it": traced
// through src/posit_runtime.cpp's applyQAlignScaleSnap, an unregistered
// key (nothing ever written to it via POSIT_QALIGN_FILE/
// POSIT_QALIGN_COLLECT_FILE) is numerically transparent -- theta defaults
// to 1.0 and useCompand defaults to false, so it passes the raw value
// through unchanged before the real posit encode still happens -- so this
// only disables the unwanted fold, it does not alter the simulated value
// or turn on any of that file's probing/collection side effects. Each call
// site gets its own key (rather than one shared constant) purely to keep
// every QDQ instance independent of any other key someone might register
// via those env vars; the starting value is just a visually-distinctive,
// unlikely-to-collide offset, not otherwise meaningful.
static int64_t nextMixedQAlignKey() {
  static int64_t counter = 0x4D505F00; // "MP_" - arbitrary distinctive base
  return ++counter;
}

// MIXEDP_POSIT_ACTIVATION=runtime selects the posit.from_f32/to_f32 activation
// path even where posit.round_f32 has a table (default "lut"). Both are
// bit-identical; the switch exists to compare them.
static bool positActivationViaLut() {
  const char *e = std::getenv("MIXEDP_POSIT_ACTIVATION");
  return !(e && std::string(e) == "runtime");
}

// Activation (runtime) QDQ round trip, dispatched by format family. Always
// returns a value of the SAME type `v` started with (f32) -- every format
// here brackets a round trip back to f32, so the real op downstream always
// sees plain f32 operands.
static Value applyActivationQDQ(IRRewriter &rewriter, Location loc,
    Value v, const FormatSpec &spec, double scale, double zeroPoint) {
  switch (spec.family) {
  case FormatFamily::FP32:
    return v;
  case FormatFamily::F16:
  case FormatFamily::BF16: {
    Type narrow = spec.family == FormatFamily::F16
                      ? Type(Builder(v.getContext()).getF16Type())
                      : Type(Builder(v.getContext()).getBF16Type());
    Value down = insertCast(rewriter, loc, v, narrow);
    return insertCast(rewriter, loc, down, Builder(v.getContext()).getF32Type());
  }
  case FormatFamily::POSIT: {
    auto origTy = mlir::dyn_cast<RankedTensorType>(v.getType());
    if (!origTy) {
      // v may be a block argument (e.g. the function's own input tensor),
      // which has no defining op -- emit against its location instead of
      // dereferencing a null Operation*.
      if (Operation *def = v.getDefiningOp())
        def->emitWarning()
            << "posit activation QDQ needs a ranked tensor; leaving untouched.";
      else
        mlir::emitWarning(v.getLoc())
            << "posit activation QDQ needs a ranked tensor; leaving untouched.";
      return v;
    }
    if (positActivationViaLut() && getPositLutCached(spec.bits, spec.es))
      return rewriter
          .create<posit::RoundF32Op>(loc, origTy, v,
              rewriter.getI64IntegerAttr(spec.bits),
              rewriter.getI64IntegerAttr(spec.es))
          .getResult();
    auto posTy = RankedTensorType::get(
        origTy.getShape(), posit::PositType::get(v.getContext(), spec.bits, spec.es));
    Operation *fromOp =
        rewriter.create<posit::FromF32Op>(loc, posTy, v).getOperation();
    fromOp->setAttr("qalign_key",
        rewriter.getI64IntegerAttr(nextMixedQAlignKey()));
    return rewriter.create<posit::ToF32Op>(loc, origTy, fromOp->getResult(0))
        .getResult();
  }
  case FormatFamily::INT:
    return buildIntActivationQDQ(rewriter, loc, v, scale, zeroPoint, spec.bits);
  }
  return v;
}

// ===========================================================================
// Per-op shape info: which operand (if any) is the weight, and whether the
// op has two independently-scaled activation-like operands (the binary
// elementwise ops) or just one.
// ===========================================================================

struct OpShapeInfo {
  int weightIdx = -1; // -1 == no weight operand on this op.
  bool isBinaryElementwise = false;
};

static OpShapeInfo getOpShapeInfo(StringRef opName) {
  OpShapeInfo info;
  if (opName == "onnx.Conv" || opName == "onnx.Gemm" || opName == "onnx.MatMul")
    info.weightIdx = 1;
  else if (opName == "onnx.Add" || opName == "onnx.Sub" ||
           opName == "onnx.Mul" || opName == "onnx.Div")
    info.isBinaryElementwise = true;
  // onnx.Relu/AveragePool/GlobalAveragePool/MaxPoolSingleOut: defaults (no
  // weight, unary) are already correct.
  return info;
}

// MaxPool is supported in its single-result form, onnx.MaxPoolSingleOut --
// the form the ONNX importer produces for any model that does not consume
// MaxPool's optional Indices output, and the only one onnx-mlir can lower.
// The two-result onnx.MaxPool (Y plus an Indices result, typically NoneType) is
// deliberately NOT listed. That is not a limitation of this pass: onnx-mlir's
// own --convert-onnx-to-krnl never lowers that form (it exits successfully but
// leaves the op in place, and a later stage then fails with "failed to legalize
// operation 'onnx.MaxPool'"), even for a MaxPool no quantization pass has
// touched, so rewriting it here would gain nothing. (This is the real cause of
// the failure ONNXToLowPrecision.cpp's isElementwiseQuantOp comment attributes
// to its rebuilt MaxPool; it is unrelated to how the op is rebuilt.)
static bool isSupportedMixedOp(StringRef name) {
  return name == "onnx.Conv" || name == "onnx.Gemm" || name == "onnx.MatMul" ||
         name == "onnx.Relu" || name == "onnx.Add" || name == "onnx.Sub" ||
         name == "onnx.Mul" || name == "onnx.Div" ||
         name == "onnx.AveragePool" || name == "onnx.GlobalAveragePool" ||
         name == "onnx.MaxPoolSingleOut";
}

// Rewrite one node: fake-quant its weight (if any, compile time) and every
// other float operand + the result (activation format, runtime QDQ), then
// rebuild the SAME original op unchanged on the round-tripped f32 operands.
// Unlike ONNXToLowPrecision.cpp's per-op-family split (retypeConvToInt8/
// retypeMatMulLikeToQuant/retypeConvToFp8/retypeElementwiseToQuant), QDQ-
// everywhere makes every supported op structurally identical -- one
// function handles all of them. Bias (Conv's 3rd operand / Gemm's optional
// C) is never touched: it stays real f32, matching the "arithmetic stays
// real f32" QDQ philosophy already established for the elementwise ops in
// ONNXToLowPrecision.cpp, and sidesteps needing to re-derive
// QLinearConv-style pre-quantized-bias handling (there's no real QLinearConv
// being built here at all).
static bool retypeNodeMixedQDQ(IRRewriter &rewriter, Operation *op,
    StringRef nodeName, const MixedNodeFormatEntry &entry) {
  StringRef opName = op->getName().getStringRef();
  OpShapeInfo info = getOpShapeInfo(opName);
  Location loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  if (entry.weightFormatInvalid)
    op->emitWarning() << "MIXEDP_NODE_FORMATS 'w=' entry for '" << nodeName
                       << "' has an unrecognized format tag; weight left "
                          "untouched.";
  if (entry.activationFormatInvalid)
    op->emitWarning() << "MIXEDP_NODE_FORMATS 'a=' entry for '" << nodeName
                       << "' has an unrecognized format tag; activation "
                          "left untouched.";

  SmallVector<Value, 4> newOperands(op->getOperands().begin(), op->getOperands().end());

  // 1. Weight (compile-time constant fake-quant).
  if (entry.weightFormat.family != FormatFamily::FP32) {
    if (info.weightIdx < 0) {
      op->emitWarning() << "MIXEDP_NODE_FORMATS gives '" << nodeName
                         << "' a weight format but its op type ('" << opName
                         << "') has no weight operand; ignoring.";
    } else {
      Value W = op->getOperand(info.weightIdx);
      DenseElementsAttr wAttr = getConstantValueAttr(W);
      if (!wAttr) {
        op->emitWarning() << "MIXEDP_NODE_FORMATS names '" << nodeName
                           << "' with a weight format but its weight operand "
                              "is not a compile-time onnx.Constant; leaving "
                              "weight untouched.";
      } else {
        int64_t channelAxis =
            weightPerChannelEnabled()
                ? weightOutputChannelAxis(op, mlir::cast<ShapedType>(wAttr.getType()))
                : -1;
        DenseElementsAttr qAttr =
            fakeQuantWeight(wAttr, entry.weightFormat, channelAxis);
        if (!qAttr) {
          op->emitWarning() << "MIXEDP_NODE_FORMATS names '" << nodeName
                             << "' with an unsupported weight format; "
                                "leaving weight untouched.";
        } else {
          newOperands[info.weightIdx] = buildConstant(rewriter, loc, qAttr);
        }
      }
    }
  }

  // 2. Activation operands (everything float that isn't the weight).
  unsigned activationOperandCount = info.isBinaryElementwise ? 2 : 1;
  size_t expectedActivationParams =
      entry.activationFormat.family == FormatFamily::INT
          ? 2 * (activationOperandCount + 1)
          : 0;
  // Params come in (scale, zero_point) pairs: even indices are scales, odd
  // indices are zero-points. Both are baked into the generated IR as FP32
  // constants (buildScalarF32Constant), so each must be validated AFTER the
  // double->float narrowing, not before: 1e-100 is a fine positive double but
  // becomes 0.0f (the onnx.Div then divides by zero), and 1e100 becomes +Inf.
  // A scale must additionally be a normal (non-subnormal) positive float -- a
  // subnormal divisor is flushed to zero on FTZ/DAZ targets. A zero-point has
  // no sign/magnitude requirement here, only that it survives narrowing as a
  // finite number (a NaN from a caught std::stod failure above, or Inf, would
  // otherwise silently propagate NaN/Inf through every quantized activation).
  // Shared by both the operand QDQ below and the result QDQ (step 4), so one
  // invalid param disables both consistently rather than only the operands.
  bool activationParamsUsable =
      entry.activationParams.size() == expectedActivationParams;
  if (activationParamsUsable &&
      entry.activationFormat.family == FormatFamily::INT)
    for (size_t p = 0; p < entry.activationParams.size(); ++p) {
      float asF32 = static_cast<float>(entry.activationParams[p]);
      bool isScale = (p % 2 == 0);
      if (isScale ? !(std::isnormal(asF32) && asF32 > 0.0f)
                  : !std::isfinite(asF32))
        activationParamsUsable = false;
    }
  if (entry.activationFormat.family != FormatFamily::FP32) {
    if (entry.activationParams.size() != expectedActivationParams) {
      op->emitWarning() << "MIXEDP_NODE_FORMATS entry for '" << nodeName
                         << "' (" << opName << ") needs exactly "
                         << expectedActivationParams
                         << " activation params; leaving activation "
                            "untouched.";
    } else if (!activationParamsUsable) {
      op->emitWarning() << "MIXEDP_NODE_FORMATS entry for '" << nodeName
                         << "' (" << opName << ") has an invalid activation "
                            "scale/zero-point (a scale must be a positive "
                            "normal FP32 value and a zero-point must be "
                            "finite, after conversion to FP32); leaving "
                            "activation untouched.";
    } else {
      for (unsigned i = 0; i < activationOperandCount; ++i) {
        Value operand = newOperands[i];
        if (!isFloatTensor(operand.getType()))
          continue;
        double scale = entry.activationFormat.family == FormatFamily::INT
                            ? entry.activationParams[2 * i]
                            : 0.0;
        double zp = entry.activationFormat.family == FormatFamily::INT
                        ? entry.activationParams[2 * i + 1]
                        : 0.0;
        newOperands[i] = applyActivationQDQ(
            rewriter, loc, operand, entry.activationFormat, scale, zp);
      }
    }
  }

  // 3. Rebuild the original op unchanged on the round-tripped operands.
  Operation *newOp = createGenericOp(
      rewriter, loc, opName, op->getResultTypes(), newOperands, op->getAttrs());

  // 4. Result activation QDQ (result(0) only; any further result of an op
  // would be non-numeric and must not be QDQ'd, though every op in
  // isSupportedMixedOp has exactly one result).
  rewriter.setInsertionPointAfter(newOp);
  Value result0 = newOp->getResult(0);
  Value finalResult0 = result0;
  if (entry.activationFormat.family != FormatFamily::FP32 &&
      isFloatTensor(result0.getType()) && activationParamsUsable) {
    double yScale = 0.0, yZp = 0.0;
    if (entry.activationFormat.family == FormatFamily::INT) {
      yScale = entry.activationParams[2 * activationOperandCount];
      yZp = entry.activationParams[2 * activationOperandCount + 1];
    }
    finalResult0 = applyActivationQDQ(
        rewriter, loc, result0, entry.activationFormat, yScale, yZp);
  }

  SmallVector<Value, 2> replacements;
  replacements.push_back(finalResult0);
  for (unsigned i = 1; i < op->getNumResults(); ++i)
    replacements.push_back(newOp->getResult(i));
  rewriter.replaceOp(op, replacements);
  return true;
}

struct ConvertONNXToMixedPrecisionPass
    : public PassWrapper<ConvertONNXToMixedPrecisionPass,
          OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertONNXToMixedPrecisionPass)

  StringRef getArgument() const final { return "convert-onnx-to-mixed-precision"; }
  StringRef getDescription() const final {
    return "Retype MIXEDP_NODE_FORMATS-selected ONNX nodes so weight and "
           "activation independently simulate int4-int8, posit p<N>e<ES>, "
           "f16, or bf16 storage via QDQ round trips -- the real op always "
           "runs on real f32 operands. Independent of, and does not modify, "
           "--convert-onnx-to-lowprecision or --convert-onnx-to-posit.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<posit::PositDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext &ctx = getContext();
    IRRewriter rewriter(&ctx);

    std::map<std::string, MixedNodeFormatEntry> formatMap = parseMixedNodeFormats();
    if (formatMap.empty())
      return;

    SmallVector<Operation *, 16> targets;
    module.walk([&](Operation *op) {
      auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name");
      if (!nameAttr)
        return;
      auto it = formatMap.find(nameAttr.getValue().str());
      if (it == formatMap.end())
        return;
      if (!isSupportedMixedOp(op->getName().getStringRef())) {
        op->emitWarning()
            << "MIXEDP_NODE_FORMATS names node '" << nameAttr.getValue()
            << "' but its op type ('" << op->getName()
            << "') is not supported; skipping.";
        return;
      }
      targets.push_back(op);
    });

    for (Operation *op : targets) {
      auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name");
      StringRef nodeName = nameAttr.getValue();
      const MixedNodeFormatEntry &entry = formatMap[nodeName.str()];
      retypeNodeMixedQDQ(rewriter, op, nodeName, entry);
    }
  }
};

// ===========================================================================
// --mixedp-collect-stats: the statistics a mixed-precision assignment needs,
// computed by the compiler itself rather than by running the FP32 model in
// another runtime. Every named supported node is visited (onnx_node_name +
// isSupportedMixedOp, as above, but not only those listed in
// MIXEDP_NODE_FORMATS). MIXEDP_STATS_LAYOUT names the JSON file the pass
// writes.
//
// MIXEDP_STATS_MODE (default "ranges"):
//   ranges   instruments the FP32 graph so a compiled run reports, at every
//            point the pass above can quantize -- the operands
//            retypeNodeMixedQDQ hands to applyActivationQDQ (the activation
//            operands, weight excluded) plus result(0) -- the tensor's
//            (min, max) over all its elements: the per-tensor range int
//            activation params are derived from. All taps are concatenated
//            into one extra f32 function result named "mixedp_stats" (one
//            result instead of one per point keeps the runtime interface
//            small); the JSON maps each (node, slot) to its tap, slots being
//            "x"/"y" for unary ops and "a"/"b"/"y" for the binary elementwise
//            ops (the order of the int activation params). A value seen by
//            several points (a result feeding another supported node) is
//            tapped once, right after it is defined, so the tapped tensor's
//            buffer is not kept alive until the function returns.
//   weights  leaves the IR untouched and writes, for every weight-bearing
//            node with a compile-time f32 weight, the squared quantization
//            error of that weight summed per output channel,
//            sum_{i in o} (Q(W)_i - W_i)^2, for each candidate format -- Q
//            being fakeQuantWeight itself (per-channel intN scales when
//            MIXEDP_WEIGHT_PER_CHANNEL=1, exactly as the pass above), the
//            channel axis weightOutputChannelAxis's.
//   grads    appends the graph's own backward pass (MixedPrecisionBackward)
//            for softmax cross-entropy of result 0 against a new
//            "mixedp_onehot" argument, and per-channel gradient statistics at
//            every point and per weight (see instrumentCriterion).
//   errors   per point and candidate activation format, per-channel sums of
//            the activation QDQ error Q(t) - t and its square; int formats
//            take their calibration range from MIXEDP_STATS_RANGES
//            ({node: {slot: [min, max]}}).
//   criterion  grads + errors in one graph: every statistic the B + C
//            criterion needs from one compiled model.
//   values   (testing) every point tensor, flattened.
// MIXEDP_STATS_FORMATS (weights mode): comma-separated candidate formats,
// default every non-fp32 format of the mixed-precision candidate set.
// ===========================================================================

static std::string onnxOpType(Operation *op) {
  StringRef name = op->getName().getStringRef();
  name.consume_front("onnx.");
  return name == "MaxPoolSingleOut" ? "MaxPool" : name.str();
}

// (min, max) over every element of `v`, as a tensor<2xf32>.
static Value buildRangeTap(IRRewriter &rewriter, Location loc, Value v) {
  OnnxBuilder create(rewriter, loc);
  Type f32 = rewriter.getF32Type();
  auto shaped = mlir::cast<ShapedType>(v.getType());
  if (shaped.getElementType() != f32) {
    v = insertCast(rewriter, loc, v, f32);
    shaped = mlir::cast<ShapedType>(v.getType());
  }
  int64_t n = shaped.hasStaticShape() ? shaped.getNumElements()
                                      : ShapedType::kDynamic;
  Value flat = create.reshape(
      RankedTensorType::get({n}, f32), v, create.constantInt64({-1}));
  Value axes = create.constantInt64({0});
  auto oneTy = RankedTensorType::get({1}, f32);
  Value lo = create.reduceMin(oneTy, flat, axes, /*keepDims=*/true);
  Value hi = create.reduceMax(oneTy, flat, axes, /*keepDims=*/true);
  return create.concat(RankedTensorType::get({2}, f32), {lo, hi}, 0);
}

// (scale, zero point) of an intN activation with calibration range [lo, hi]:
// exactly experiments/mixed_precision/eval_random_mixed_precision.int_qparams,
// whose values MIXEDP_NODE_FORMATS carries -- the range widened to contain 0,
// scale = (hi - lo) / (2^bits - 1) (1 if that is not positive), zero point
// qmin - lo / scale rounded half to even (Python's round) and clamped, all in
// double.
static std::pair<double, double> intActivationParams(
    double lo, double hi, unsigned bits) {
  double qmin = -static_cast<double>(1 << (bits - 1));
  double qmax = static_cast<double>((1 << (bits - 1)) - 1);
  lo = std::min(lo, 0.0);
  hi = std::max(hi, 0.0);
  double scale = (hi - lo) / (qmax - qmin);
  if (!(scale > 0.0))
    scale = 1.0;
  double zeroPoint = std::clamp(std::nearbyint(qmin - lo / scale), qmin, qmax);
  return {scale, zeroPoint};
}

// MIXEDP_STATS_NODES: comma-separated node names; when set, only those nodes'
// points (and weights) are collected -- e.g. just the nodes a selection
// assigns formats to.
// MIXEDP_STATS_INPUT_GRAD=branch (grads modes): each weight node's entry also
// holds the statistics of its own input-edge gradient (see instrumentCriterion).
static bool inputBranchGradients() {
  const char *e = std::getenv("MIXEDP_STATS_INPUT_GRAD");
  return e && std::string(e) == "branch";
}

static std::optional<std::set<std::string>> statsNodeFilter() {
  const char *e = std::getenv("MIXEDP_STATS_NODES");
  if (!e || !*e)
    return std::nullopt;
  SmallVector<StringRef, 64> names;
  StringRef(e).split(names, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
  std::set<std::string> out;
  for (StringRef n : names)
    out.insert(n.trim().str());
  return out;
}

// A json number, or null for a non-finite value (JSON has no NaN/Inf).
static llvm::json::Value jsonNumber(double v) {
  return std::isfinite(v) ? llvm::json::Value(v) : llvm::json::Value(nullptr);
}

struct MixedPrecisionCollectStatsPass
    : public PassWrapper<MixedPrecisionCollectStatsPass,
          OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MixedPrecisionCollectStatsPass)

  StringRef getArgument() const final { return "mixedp-collect-stats"; }
  StringRef getDescription() const final {
    return "Statistics for mixed-precision selection at every point "
           "--convert-onnx-to-mixed-precision can quantize "
           "(MIXEDP_STATS_MODE: 'ranges' appends a 'mixedp_stats' result "
           "with per-tensor min/max, 'weights' writes per-output-channel "
           "weight quantization errors, 'grads' appends the backward pass and "
           "per-channel gradient statistics); output JSON in "
           "MIXEDP_STATS_LAYOUT.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    const char *modeEnv = std::getenv("MIXEDP_STATS_MODE");
    std::string mode = modeEnv && *modeEnv ? modeEnv : "ranges";
    if (mode != "ranges" && mode != "weights" && mode != "grads" &&
        mode != "errors" && mode != "criterion" && mode != "values") {
      module.emitError() << "MIXEDP_STATS_MODE='" << mode
                         << "' is not supported (expected 'ranges', "
                            "'weights', 'grads', 'errors', 'criterion' or "
                            "'values')";
      return signalPassFailure();
    }
    const char *layoutPath = std::getenv("MIXEDP_STATS_LAYOUT");
    if (!layoutPath || !*layoutPath) {
      module.emitError() << "--mixedp-collect-stats needs MIXEDP_STATS_LAYOUT "
                            "(the JSON file to write)";
      return signalPassFailure();
    }

    std::optional<llvm::json::Object> out =
        mode == "ranges"    ? instrumentRanges(module)
        : mode == "weights" ? collectWeightErrors(module)
        : mode == "values"  ? instrumentValues(module)
                            : instrumentCriterion(module,
                                  /*withGrads=*/mode != "errors",
                                  /*withErrors=*/mode != "grads");
    if (!out)
      return signalPassFailure();
    (*out)["mode"] = mode;

    std::error_code ec;
    llvm::raw_fd_ostream os(layoutPath, ec);
    if (ec) {
      module.emitError() << "cannot write MIXEDP_STATS_LAYOUT='" << layoutPath
                         << "': " << ec.message();
      return signalPassFailure();
    }
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(*out))) << "\n";
  }

  // The quantization points (see the header comment) of the one function
  // with named supported nodes. Nodes nested in an If branch or a Loop/Scan
  // body are only listed: a value defined there cannot reach the function's
  // return (it would have to be yielded out of every enclosing op, and a loop
  // body's value exists once per iteration).
  struct StatsPoints {
    struct Point {
      std::string node, op, slot;
      int64_t tap;
    };
    func::FuncOp func; // null: no named supported node anywhere
    SmallVector<Point, 64> points;
    SmallVector<Value, 64> tapped; // distinct point values, in tap order
    llvm::json::Array skippedNested;

    llvm::json::Array pointsJson() const {
      llvm::json::Array out;
      for (const Point &p : points)
        out.push_back(llvm::json::Object{{"node", p.node}, {"op", p.op},
            {"slot", p.slot}, {"tap", p.tap}});
      return out;
    }
  };

  std::optional<StatsPoints> collectPoints(ModuleOp module) {
    SmallVector<func::FuncOp, 1> funcs;
    module.walk([&](func::FuncOp f) { funcs.push_back(f); });
    std::optional<std::set<std::string>> filter = statsNodeFilter();
    StatsPoints result;
    for (func::FuncOp func : funcs) {
      if (func.isExternal())
        continue;
      StatsPoints sp;
      llvm::DenseMap<Value, int64_t> tapOf;
      auto tapIndex = [&](Value v) {
        auto [it, inserted] = tapOf.try_emplace(v, sp.tapped.size());
        if (inserted)
          sp.tapped.push_back(v);
        return it->second;
      };
      Block &body = func.getBody().front();
      func.walk([&](Operation *op) {
        auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name");
        if (!nameAttr || !isSupportedMixedOp(op->getName().getStringRef()))
          return;
        if (filter && !filter->count(nameAttr.getValue().str()))
          return;
        if (op->getBlock() != &body) {
          sp.skippedNested.push_back(nameAttr.getValue().str());
          return;
        }
        OpShapeInfo info = getOpShapeInfo(op->getName().getStringRef());
        unsigned activationOperandCount = info.isBinaryElementwise ? 2 : 1;
        static const char *const operandSlots[2][2] = {{"x", ""}, {"a", "b"}};
        std::string node = nameAttr.getValue().str(), type = onnxOpType(op);
        for (unsigned i = 0;
             i < activationOperandCount && i < op->getNumOperands(); ++i) {
          Value v = op->getOperand(i);
          if (isFloatTensor(v.getType()))
            sp.points.push_back({node, type,
                operandSlots[info.isBinaryElementwise][i], tapIndex(v)});
        }
        Value y = op->getResult(0);
        if (isFloatTensor(y.getType()))
          sp.points.push_back({node, type, "y", tapIndex(y)});
      });
      if (!sp.skippedNested.empty())
        func.emitWarning() << "--mixedp-collect-stats: "
                           << sp.skippedNested.size()
                           << " supported node(s) inside nested regions "
                              "(If/Loop/Scan bodies) are not instrumented; "
                              "see 'skipped_nested' in the layout";
      if (sp.tapped.empty() && sp.skippedNested.empty())
        continue;
      if (result.func) {
        func.emitError() << "--mixedp-collect-stats supports one function "
                            "with supported nodes; found another";
        return std::nullopt;
      }
      if (!func.getBody().hasOneBlock()) {
        func.emitError() << "--mixedp-collect-stats expects a single-block "
                            "function body";
        return std::nullopt;
      }
      sp.func = func;
      result = std::move(sp);
    }
    if (!result.func)
      module.emitWarning() << "--mixedp-collect-stats found no named "
                              "supported node; nothing to collect";
    return result;
  }

  // Appends `stats` to `func`'s results as "mixedp_stats"; returns its index.
  std::optional<unsigned> appendStatsResult(func::FuncOp func, Value stats) {
    unsigned resultIndex = func.getNumResults();
    Builder b(func.getContext());
    NamedAttribute onnxName(
        b.getStringAttr("onnx.name"), b.getStringAttr("mixedp_stats"));
    if (failed(func.insertResult(
            resultIndex, stats.getType(), b.getDictionaryAttr({onnxName})))) {
      func.emitError() << "--mixedp-collect-stats could not add the "
                          "'mixedp_stats' result";
      return std::nullopt;
    }
    Operation *ret = func.getBody().back().getTerminator();
    ret->insertOperands(ret->getNumOperands(), stats);
    return resultIndex;
  }

  std::optional<llvm::json::Object> instrumentRanges(ModuleOp module) {
    std::optional<StatsPoints> sp = collectPoints(module);
    if (!sp)
      return std::nullopt;
    if (!sp->func || sp->tapped.empty()) {
      llvm::json::Object layout{{"num_taps", 0}, {"points", llvm::json::Array{}},
          {"skipped_nested", std::move(sp->skippedNested)}};
      if (sp->func)
        layout["function"] = sp->func.getSymName().str();
      return layout;
    }
    func::FuncOp func = sp->func;
    IRRewriter rewriter(&getContext());
    SmallVector<Value, 64> parts;
    for (Value v : sp->tapped) {
      if (Operation *def = v.getDefiningOp())
        rewriter.setInsertionPointAfter(def);
      else
        rewriter.setInsertionPointToStart(
            mlir::cast<BlockArgument>(v).getOwner());
      parts.push_back(buildRangeTap(rewriter, v.getLoc(), v));
    }
    Operation *ret = func.getBody().back().getTerminator();
    rewriter.setInsertionPoint(ret);
    auto statsTy = RankedTensorType::get(
        {2 * static_cast<int64_t>(sp->tapped.size())}, rewriter.getF32Type());
    Value stats =
        parts.size() == 1
            ? parts.front()
            : OnnxBuilder(rewriter, ret->getLoc()).concat(statsTy, parts, 0);
    std::optional<unsigned> resultIndex = appendStatsResult(func, stats);
    if (!resultIndex)
      return std::nullopt;
    return llvm::json::Object{{"function", func.getSymName().str()},
        {"result_name", "mixedp_stats"}, {"result_index", *resultIndex},
        {"num_taps", static_cast<int64_t>(sp->tapped.size())},
        {"values_per_tap", 2}, {"tap_values", llvm::json::Array{"min", "max"}},
        {"points", sp->pointsJson()},
        {"skipped_nested", std::move(sp->skippedNested)}};
  }

  // MIXEDP_STATS_FORMATS (comma-separated tags), default every non-fp32 format
  // of the candidate set; nullopt after an error.
  std::optional<SmallVector<std::pair<std::string, FormatSpec>, 10>>
  parseStatsFormats(ModuleOp module) {
    const char *formatsEnv = std::getenv("MIXEDP_STATS_FORMATS");
    StringRef formatList = formatsEnv && *formatsEnv
                               ? StringRef(formatsEnv)
                               : "int4,int5,int6,int7,int8,p4e1,p6e1,p8e1,"
                                 "f16,bf16";
    SmallVector<StringRef, 10> tags;
    formatList.split(tags, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
    SmallVector<std::pair<std::string, FormatSpec>, 10> formats;
    for (StringRef tag : tags) {
      tag = tag.trim();
      FormatSpec spec;
      if (!parseFormatTag(tag, spec)) {
        module.emitError() << "MIXEDP_STATS_FORMATS has an unrecognized "
                              "format tag '"
                           << tag << "'";
        return std::nullopt;
      }
      formats.push_back({tag.str(), spec});
    }
    return formats;
  }

  std::optional<llvm::json::Object> loadJsonObject(
      ModuleOp module, StringRef path) {
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer) {
      module.emitError() << "cannot read '" << path
                         << "': " << buffer.getError().message();
      return std::nullopt;
    }
    llvm::Expected<llvm::json::Value> parsed =
        llvm::json::parse((*buffer)->getBuffer());
    if (!parsed) {
      module.emitError() << "'" << path << "' is not valid JSON: "
                         << llvm::toString(parsed.takeError());
      return std::nullopt;
    }
    if (llvm::json::Object *obj = parsed->getAsObject())
      return std::move(*obj);
    module.emitError() << "'" << path << "' is not a JSON object";
    return std::nullopt;
  }

  std::optional<llvm::json::Object> collectWeightErrors(ModuleOp module) {
    std::optional<SmallVector<std::pair<std::string, FormatSpec>, 10>> parsed =
        parseStatsFormats(module);
    if (!parsed)
      return std::nullopt;
    SmallVector<std::pair<std::string, FormatSpec>, 10> formats =
        std::move(*parsed);
    llvm::json::Array formatsJson;
    for (const auto &f : formats)
      formatsJson.push_back(f.first);
    bool perChannel = weightPerChannelEnabled();

    llvm::json::Array nodes, skipped;
    bool ok = true;
    module.walk([&](Operation *op) {
      auto nameAttr = op->getAttrOfType<StringAttr>("onnx_node_name");
      if (!ok || !nameAttr || !isSupportedMixedOp(op->getName().getStringRef()))
        return;
      OpShapeInfo info = getOpShapeInfo(op->getName().getStringRef());
      if (info.weightIdx < 0 ||
          info.weightIdx >= static_cast<int>(op->getNumOperands()))
        return;
      std::string node = nameAttr.getValue().str();
      DenseElementsAttr wAttr =
          getConstantValueAttr(op->getOperand(info.weightIdx));
      if (!wAttr || !wAttr.getElementType().isF32()) {
        skipped.push_back(node); // not a compile-time f32 weight
        return;
      }
      auto wTy = mlir::cast<ShapedType>(wAttr.getType());
      int64_t axis = weightOutputChannelAxis(op, wTy);
      int64_t channels = axis >= 0 ? wTy.getDimSize(axis) : 1;
      int64_t inner = axis >= 0 ? 1 : wTy.getNumElements();
      for (int64_t d = axis + 1; axis >= 0 && d < wTy.getRank(); ++d)
        inner *= wTy.getDimSize(d);
      SmallVector<float, 64> w(wAttr.getValues<float>());

      llvm::json::Object err2;
      for (const auto &[tag, spec] : formats) {
        SmallVector<float, 64> q;          // values only: no attribute per format
        if (!fakeQuantWeight(wAttr, spec, perChannel ? axis : -1, &q)) {
          op->emitError() << "--mixedp-collect-stats cannot quantize the "
                             "weight of '"
                          << node << "' to " << tag;
          ok = false;
          return;
        }
        SmallVector<double, 64> sum(channels, 0.0);
        int64_t i = 0;
        for (float qv : q) {
          double d = static_cast<double>(qv) - static_cast<double>(w[i]);
          sum[(i / inner) % channels] += d * d;
          ++i;
        }
        llvm::json::Array perChannelErr;
        for (double s : sum)
          perChannelErr.push_back(jsonNumber(s));
        err2[tag] = std::move(perChannelErr);
      }
      nodes.push_back(llvm::json::Object{{"node", node},
          {"op", onnxOpType(op)}, {"channel_axis", axis},
          {"channels", channels}, {"elements", wTy.getNumElements()},
          {"err2", std::move(err2)}});
    });
    if (!ok)
      return std::nullopt;
    return llvm::json::Object{{"weight_per_channel", perChannel},
        {"formats", std::move(formatsJson)}, {"nodes", std::move(nodes)},
        {"skipped", std::move(skipped)}};
  }

  // Grads / errors / criterion modes, on the quantization points of
  // collectPoints and the weight-bearing nodes with a compile-time f32 weight.
  //
  // withGrads: the FP32 graph plus its own backward pass (buildBackward). The
  // loss is softmax cross-entropy of result 0 against a new argument
  // "mixedp_onehot" of the same shape (seed: softmax(z) - onehot over the last
  // axis), i.e. the summed cross-entropy of every row. Per tap g =
  // d(loss)/d(value), the total gradient over all its consumers: sum(g^2) then
  // sum(g) over every axis but the channel axis (1 for rank-4 NCHW tensors,
  // else the last), C values each. Per weight node, sum(dW^2) over each output
  // channel (weightOutputChannelAxis).
  //
  // withErrors: per tap and candidate activation format (MIXEDP_STATS_FORMATS),
  // d = Q(t) - t with Q exactly the activation QDQ this pass's
  // --convert-onnx-to-mixed-precision inserts (applyActivationQDQ; int
  // formats take their (scale, zero point) from the tensor's calibration range
  // in MIXEDP_STATS_RANGES, see intActivationParams), and sum(d) then sum(d^2)
  // per channel. Inserted right after the tensor is defined.
  //
  // Everything is concatenated into "mixedp_stats"; offsets in the layout.
  std::optional<llvm::json::Object> instrumentCriterion(
      ModuleOp module, bool withGrads, bool withErrors) {
    std::optional<StatsPoints> sp = collectPoints(module);
    if (!sp)
      return std::nullopt;
    if (!sp->func || sp->tapped.empty()) {
      module.emitError() << "--mixedp-collect-stats found no quantization point";
      return std::nullopt;
    }
    func::FuncOp func = sp->func;
    Operation *ret = func.getBody().back().getTerminator();
    for (Value v : sp->tapped) {
      auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
      if (!ty || !ty.hasStaticShape() || ty.getRank() < 1 ||
          !ty.getElementType().isF32()) {
        func.emitError() << "--mixedp-collect-stats needs static f32 "
                            "quantization points of rank >= 1";
        return std::nullopt;
      }
    }

    SmallVector<std::pair<std::string, FormatSpec>, 10> formats;
    SmallVector<std::optional<std::pair<double, double>>, 64> tapRange(
        sp->tapped.size());
    if (withErrors) {
      std::optional<SmallVector<std::pair<std::string, FormatSpec>, 10>> parsed =
          parseStatsFormats(module);
      if (!parsed)
        return std::nullopt;
      formats = std::move(*parsed);
      bool needRanges = llvm::any_of(formats,
          [](const auto &f) { return f.second.family == FormatFamily::INT; });
      if (needRanges) {
        const char *rangesPath = std::getenv("MIXEDP_STATS_RANGES");
        std::optional<llvm::json::Object> ranges =
            rangesPath && *rangesPath ? loadJsonObject(module, rangesPath)
                                      : std::nullopt;
        if (!ranges) {
          if (!rangesPath || !*rangesPath)
            module.emitError() << "--mixedp-collect-stats: int activation "
                                  "formats need MIXEDP_STATS_RANGES "
                                  "({node: {slot: [min, max]}})";
          return std::nullopt;
        }
        for (const auto &p : sp->points) {
          if (tapRange[p.tap])
            continue;
          const llvm::json::Object *node = ranges->getObject(p.node);
          const llvm::json::Array *r = node ? node->getArray(p.slot) : nullptr;
          if (!r || r->size() != 2 || !(*r)[0].getAsNumber() ||
              !(*r)[1].getAsNumber())
            continue;
          tapRange[p.tap] = std::make_pair(
              *(*r)[0].getAsNumber(), *(*r)[1].getAsNumber());
        }
        for (const auto &p : sp->points)
          if (!tapRange[p.tap]) {
            module.emitError() << "MIXEDP_STATS_RANGES has no [min, max] for "
                               << p.node << ":" << p.slot;
            return std::nullopt;
          }
      }
    }

    IRRewriter rewriter(&getContext());
    Location loc = func.getLoc();
    OnnxBuilder create(rewriter, loc);
    Type f32 = rewriter.getF32Type();
    // sum over every axis of `v` but `axis`, as a [C] tensor. A rank-1 tensor
    // already is one: ReduceSum with an empty axes list would instead reduce
    // every axis (noop_with_empty_axes = 0) and break the [C] result type.
    auto sumAllBut = [&](Value v, int64_t axis) {
      auto ty = mlir::cast<RankedTensorType>(v.getType());
      if (ty.getRank() == 1)
        return v;
      SmallVector<int64_t, 4> axes;
      for (int64_t d = 0; d < ty.getRank(); ++d)
        if (d != axis)
          axes.push_back(d);
      return create.reduceSum(RankedTensorType::get({ty.getDimSize(axis)}, f32),
          v, create.constantInt64(axes), /*keepDims=*/false);
    };

    unsigned argIndex = func.getNumArguments();
    SmallVector<Operation *, 64> weightNodes;
    std::optional<BackwardGradients> grads;
    if (withGrads) {
      auto logitsTy = ret->getNumOperands() > 0
                          ? mlir::dyn_cast<RankedTensorType>(
                                ret->getOperand(0).getType())
                          : RankedTensorType();
      if (!logitsTy || !logitsTy.hasStaticShape() ||
          !logitsTy.getElementType().isF32()) {
        func.emitError() << "--mixedp-collect-stats needs a static f32 "
                            "result 0 (the logits) for gradients";
        return std::nullopt;
      }
      Value logits = ret->getOperand(0);
      if (failed(func.insertArgument(argIndex, logitsTy,
              rewriter.getDictionaryAttr(
                  {NamedAttribute(rewriter.getStringAttr("onnx.name"),
                      rewriter.getStringAttr("mixedp_onehot"))}),
              func.getLoc()))) {
        func.emitError() << "--mixedp-collect-stats could not add the "
                            "'mixedp_onehot' argument";
        return std::nullopt;
      }
      Value onehot = func.getArgument(argIndex);
      llvm::DenseSet<Operation *> weightSet;
      std::optional<std::set<std::string>> filter = statsNodeFilter();
      for (Operation &op : func.getBody().front().without_terminator()) {
        auto nameAttr = op.getAttrOfType<StringAttr>("onnx_node_name");
        if (!nameAttr || !isSupportedMixedOp(op.getName().getStringRef()))
          continue;
        if (filter && !filter->count(nameAttr.getValue().str()))
          continue;
        OpShapeInfo info = getOpShapeInfo(op.getName().getStringRef());
        if (info.weightIdx < 0 ||
            info.weightIdx >= static_cast<int>(op.getNumOperands()))
          continue;
        DenseElementsAttr w =
            getConstantValueAttr(op.getOperand(info.weightIdx));
        if (!w || !w.getElementType().isF32())
          continue;
        weightNodes.push_back(&op);
        weightSet.insert(&op);
      }
      rewriter.setInsertionPoint(ret);
      Value softmax =
          create
              .createTypedOpAndInferShapes<ONNXSoftmaxOp>(logitsTy, logits,
                  rewriter.getIntegerAttr(
                      rewriter.getIntegerType(64, /*isSigned=*/true), -1))
              .getResult();
      // Rows whose one-hot is all zero (no target: e.g. the last position of a
      // causal language model) get no gradient: the seed is masked by each
      // row's one-hot sum (1 for every row of a classifier).
      SmallVector<int64_t, 4> rowShape(logitsTy.getShape());
      rowShape.back() = 1;
      Value rowHasTarget = create.reduceSum(RankedTensorType::get(rowShape, f32),
          onehot, create.constantInt64({logitsTy.getRank() - 1}),
          /*keepDims=*/true);
      FailureOr<BackwardGradients> g = buildBackward(rewriter, func, logits,
          create.mul(create.sub(softmax, onehot), rowHasTarget), weightSet);
      if (failed(g))
        return std::nullopt;
      grads = std::move(*g);
    }

    SmallVector<Value, 256> parts;
    int64_t offset = 0;
    llvm::json::Array tapsJson, weightsJson;
    for (auto [k, v] : llvm::enumerate(sp->tapped)) {
      auto ty = mlir::cast<RankedTensorType>(v.getType());
      int64_t axis = ty.getRank() == 4 ? 1 : ty.getRank() - 1;
      int64_t channels = ty.getDimSize(axis);
      llvm::json::Object tap{{"tap", static_cast<int64_t>(k)},
          {"channels", channels}, {"channel_axis", axis},
          {"positions", ty.getNumElements() / channels}};
      if (withGrads) {
        rewriter.setInsertionPoint(ret);
        Value g = grads->values.lookup(v);
        if (g) {
          parts.push_back(sumAllBut(create.mul(g, g), axis));
          parts.push_back(sumAllBut(g, axis));
        } else { // no path to the loss: the gradient is zero
          parts.push_back(create.constant(DenseElementsAttr::get(
              RankedTensorType::get({2 * channels}, f32), 0.0f)));
        }
        tap["offset"] = offset;
        tap["has_grad"] = static_cast<bool>(g);
        offset += 2 * channels;
      }
      if (withErrors) {
        if (Operation *def = v.getDefiningOp())
          rewriter.setInsertionPointAfter(def);
        else
          rewriter.setInsertionPointToStart(
              mlir::cast<BlockArgument>(v).getOwner());
        llvm::json::Object errOffsets, intParams;
        for (const auto &[tag, spec] : formats) {
          double scale = 0.0, zeroPoint = 0.0;
          if (spec.family == FormatFamily::INT) {
            std::tie(scale, zeroPoint) = intActivationParams(
                tapRange[k]->first, tapRange[k]->second, spec.bits);
            intParams[tag] = llvm::json::Array{scale, zeroPoint};
          }
          // An int QDQ dividing by a SCALAR constant directly after a MatMul
          // with a constant weight is folded by onnx-mlir's
          // FuseScalarDivMatMulPattern into a MatMul with a scaled copy of
          // the weight -- one more weight copy per int format here (BERT's
          // statistics graph grew from 0.9 to 4.3 GB, past what clang reads).
          // A splat scale along the last axis divides by the same value but
          // is not a scalar, so the pattern does not apply.
          Value q = spec.family == FormatFamily::INT
                        ? buildIntActivationQDQ(rewriter, v.getLoc(), v, scale,
                              zeroPoint, spec.bits, ty.getShape().back())
                        : applyActivationQDQ(
                              rewriter, v.getLoc(), v, spec, scale, zeroPoint);
          Value d = create.sub(q, v);
          parts.push_back(sumAllBut(d, axis));
          parts.push_back(sumAllBut(create.mul(d, d), axis));
          errOffsets[tag] = offset;
          offset += 2 * channels;
        }
        tap["errors"] = std::move(errOffsets);
        tap["int_params"] = std::move(intParams);
      }
      tapsJson.push_back(std::move(tap));
    }
    rewriter.setInsertionPoint(ret);
    for (Operation *op : weightNodes) {
      OpShapeInfo info = getOpShapeInfo(op->getName().getStringRef());
      Value dW =
          grads->operands.lookup({op, static_cast<unsigned>(info.weightIdx)});
      auto wTy = mlir::cast<RankedTensorType>(dW.getType());
      int64_t axis = weightOutputChannelAxis(op, wTy);
      if (axis < 0) {
        op->emitError() << "--mixedp-collect-stats: no output channel axis "
                           "for this weight";
        return std::nullopt;
      }
      int64_t channels = wTy.getDimSize(axis);
      parts.push_back(sumAllBut(create.mul(dW, dW), axis));
      llvm::json::Object entry{
          {"node", op->getAttrOfType<StringAttr>("onnx_node_name").getValue().str()},
          {"op", onnxOpType(op)}, {"offset", offset}, {"channels", channels},
          {"channel_axis", axis},
          {"elements_per_channel", wTy.getNumElements() / channels}};
      offset += channels;
      if (inputBranchGradients()) {
        // The input as seen by this node alone (its own share of the input's
        // gradient): the mixed-precision pass quantizes each consumer's copy
        // of a shared input, so this, not the tap's total gradient, is the
        // sensitivity of this node's input QDQ. Same fields as a tap.
        Value x = op->getOperand(0);
        auto xTy = mlir::cast<RankedTensorType>(x.getType());
        int64_t xAxis = xTy.getRank() == 4 ? 1 : xTy.getRank() - 1;
        int64_t xChannels = xTy.getDimSize(xAxis);
        Value gx = grads->operands.lookup({op, 0u});
        if (gx) {
          parts.push_back(sumAllBut(create.mul(gx, gx), xAxis));
          parts.push_back(sumAllBut(gx, xAxis));
        } else {
          parts.push_back(create.constant(DenseElementsAttr::get(
              RankedTensorType::get({2 * xChannels}, f32), 0.0f)));
        }
        entry["x_branch_offset"] = offset;
        entry["x_channels"] = xChannels;
        entry["x_channel_axis"] = xAxis;
        entry["x_has_grad"] = static_cast<bool>(gx);
        offset += 2 * xChannels;
      }
      weightsJson.push_back(std::move(entry));
    }
    Value stats = create.concat(RankedTensorType::get({offset}, f32), parts, 0);
    std::optional<unsigned> resultIndex = appendStatsResult(func, stats);
    if (!resultIndex)
      return std::nullopt;

    llvm::json::Object layout{{"function", func.getSymName().str()},
        {"result_name", "mixedp_stats"}, {"result_index", *resultIndex},
        {"num_values", offset}, {"points", sp->pointsJson()},
        {"taps", std::move(tapsJson)},
        {"skipped_nested", std::move(sp->skippedNested)}};
    if (withGrads) {
      layout["input_name"] = "mixedp_onehot";
      layout["input_index"] = argIndex;
      layout["loss"] = "softmax cross-entropy of result 0 against "
                       "mixedp_onehot, summed over the rows that have a target "
                       "(seed (softmax(z) - onehot) * sum(onehot), last axis)";
      layout["tap_fields"] = llvm::json::Array{"g2", "gsum"};
      layout["weights"] = std::move(weightsJson);
    }
    if (withErrors) {
      llvm::json::Array tags;
      for (const auto &f : formats)
        tags.push_back(f.first);
      layout["formats"] = std::move(tags);
      layout["error_fields"] = llvm::json::Array{"dsum", "d2"};
    }
    return layout;
  }

  // Values mode (for testing): every distinct quantization point tensor,
  // flattened and concatenated into "mixedp_stats".
  std::optional<llvm::json::Object> instrumentValues(ModuleOp module) {
    std::optional<StatsPoints> sp = collectPoints(module);
    if (!sp)
      return std::nullopt;
    if (!sp->func || sp->tapped.empty()) {
      module.emitError() << "--mixedp-collect-stats found no quantization point";
      return std::nullopt;
    }
    func::FuncOp func = sp->func;
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(func.getBody().back().getTerminator());
    OnnxBuilder create(rewriter, func.getLoc());
    SmallVector<Value, 128> parts;
    llvm::json::Array tapsJson;
    int64_t offset = 0;
    for (auto [k, v] : llvm::enumerate(sp->tapped)) {
      auto ty = mlir::dyn_cast<RankedTensorType>(v.getType());
      if (!ty || !ty.hasStaticShape() || !ty.getElementType().isF32()) {
        func.emitError() << "--mixedp-collect-stats needs static f32 "
                            "quantization points";
        return std::nullopt;
      }
      int64_t n = ty.getNumElements();
      parts.push_back(create.reshape(RankedTensorType::get({n}, ty.getElementType()),
          v, create.constantInt64({n})));
      tapsJson.push_back(llvm::json::Object{{"tap", static_cast<int64_t>(k)},
          {"offset", offset}, {"shape", llvm::json::Array(ty.getShape())}});
      offset += n;
    }
    Value values = create.concat(
        RankedTensorType::get({offset}, rewriter.getF32Type()), parts, 0);
    std::optional<unsigned> resultIndex = appendStatsResult(func, values);
    if (!resultIndex)
      return std::nullopt;
    return llvm::json::Object{{"function", func.getSymName().str()},
        {"result_name", "mixedp_stats"}, {"result_index", *resultIndex},
        {"num_values", offset}, {"points", sp->pointsJson()},
        {"taps", std::move(tapsJson)},
        {"skipped_nested", std::move(sp->skippedNested)}};
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConvertONNXToMixedPrecisionPass() {
  return std::make_unique<ConvertONNXToMixedPrecisionPass>();
}

std::unique_ptr<mlir::Pass> createMixedPrecisionCollectStatsPass() {
  return std::make_unique<MixedPrecisionCollectStatsPass>();
}

} // namespace onnx_mlir
