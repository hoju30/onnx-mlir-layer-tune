#include "onnx-mlir/Conversion/ONNXToMixedPrecision/ONNXToMixedPrecision.hpp"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositOps.h"
#include "src/Dialect/Posit/PositTypes.h"

#include <universal/number/posit/posit.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
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
static DenseElementsAttr fakeQuantWeightInt(
    DenseElementsAttr floatAttr, unsigned bits) {
  auto floatTy = mlir::cast<RankedTensorType>(floatAttr.getType());
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
  return DenseElementsAttr::get(floatTy, ArrayRef<float>(out));
}

// Runtime (activation) counterpart of fakeQuantWeightInt, built as IR ops:
// clip((x/scale)+zero_point, qmin, qmax) rounded, then unshift+rescale back
// to f32. Uses OnnxBuilder::round/clip (src/Dialect/ONNX/DialectBuilder.hpp)
// rather than hand-building those ops, and plain onnx.Div/Add/Sub/Mul via
// createGenericOp for the rest, matching the exact style
// ONNXToLowPrecision.cpp's buildQuantizeCastF8/buildDequantizeCastF8 already
// use for their own from-scratch f32 arithmetic.
static Value buildIntActivationQDQ(IRRewriter &rewriter, Location loc,
    Value x, double scale, double zeroPoint, unsigned bits) {
  OnnxBuilder create(rewriter, loc);
  float qmax = static_cast<float>((1 << (bits - 1)) - 1);
  float qmin = static_cast<float>(-(1 << (bits - 1)));
  Value scaleC = buildScalarF32Constant(rewriter, loc, static_cast<float>(scale));
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
// (p4e1/p6e1/p8e1). Posit rounding is NOT "nearest representable value" in
// arithmetic terms -- it rounds on the encoded bit string (regime/exponent/
// fraction), a nonzero input never underflows to 0 (it lands on minpos),
// anything past maxpos saturates, and NaN/+-Inf become NaR. Confirmed by
// direct comparison against sw::universal::posit<N,ES>: a nearest-value
// search over the decoded table disagrees with it on ~30% of probed inputs.
//
// What IS true, and what this LUT exploits, is that Universal's conversion
// from float is a monotone step function (verified: 0 violations over 2M
// sorted random finite float bit patterns per format). So for the sorted
// list of finite representable values v[0..M-1], there is, for each adjacent
// pair (v[i], v[i+1]), one exact decision boundary in float space: every
// float below it rounds to v[i] or lower, every float at or above it rounds
// to v[i+1] or higher. Those M-1 boundaries are found once per (nbits,es) by
// bisecting over the ordered float bit patterns with Universal itself as the
// oracle -- so the table reproduces Universal exactly (ties, underflow,
// saturation and all) by construction rather than by re-deriving posit
// rounding rules -- and then each weight element costs one binary search over
// at most 2^N-2 boundaries instead of a general posit encode. Larger N would
// need a prohibitively large table and is not wired up (not needed by this
// pass's candidate format set).
//
// Activation (runtime tensor) path deliberately does NOT reuse this LUT --
// see applyActivationQDQ below, which inserts real posit.from_f32/to_f32
// ops instead, lowered by the existing, shared PositToKrnl pass /
// src/posit_runtime.cpp at execution time. LUT-accelerating that runtime
// path would mean modifying posit_runtime.cpp itself (shared with the
// existing --convert-onnx-to-posit pass); left as a follow-up, not part of
// this change.
// ===========================================================================

// Maps a finite float to a uint32_t whose unsigned order matches the float's
// numeric order (-0.0 sorts immediately below +0.0), so the float line can be
// bisected as a plain integer range.
static uint32_t floatOrderKey(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static float floatFromOrderKey(uint32_t k) {
  uint32_t u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

struct PositLut {
  // Finite representable values, ascending. NaR is excluded (see below).
  std::vector<float> values;
  // thresholds[i] is the smallest floatOrderKey that Universal rounds to
  // values[i+1] or above; size == values.size() - 1.
  std::vector<uint32_t> thresholds;
};

// Builds the LUT for posit<N,ES>. Returns false if Universal ever behaves
// contrary to what the construction relies on (a finite input converting to
// something not in the value table, e.g. NaR) -- the caller then treats the
// format as unsupported and leaves the weight untouched rather than
// producing a wrong table.
template <int N, int ES>
static bool buildPositLutT(PositLut &lut) {
  using sw::universal::posit;
  int64_t count = int64_t(1) << N;
  int64_t half = count / 2;

  // sVal == -half (raw bit pattern 1000...0: sign bit set, everything else
  // zero) is posit's single reserved NaR (Not-a-Real) encoding, not an
  // ordinary value -- it decodes to NaN and must not be in the table.
  // Every other bit pattern is an ordinary finite value, and interpreted as a
  // signed integer they are already in ascending numeric order (posit's
  // monotone-encoding property).
  lut.values.resize(static_cast<size_t>(count - 1));
  for (int64_t sVal = -half + 1; sVal < half; ++sVal) {
    uint8_t raw = static_cast<uint8_t>(sVal) & static_cast<uint8_t>(count - 1);
    posit<N, ES> p;
    p.setbits(static_cast<uint64_t>(raw));
    lut.values[static_cast<size_t>(sVal + half - 1)] = static_cast<float>(p);
  }

  auto indexOf = [&](float f, size_t &idx) -> bool {
    posit<N, ES> p;
    p = f;
    float d = static_cast<float>(p);
    auto it = std::lower_bound(lut.values.begin(), lut.values.end(), d);
    if (it == lut.values.end() || *it != d)
      return false; // NaN (NaR) or a value outside the table
    idx = static_cast<size_t>(it - lut.values.begin());
    return true;
  };

  const uint32_t kMin = floatOrderKey(-std::numeric_limits<float>::max());
  const uint32_t kMax = floatOrderKey(std::numeric_limits<float>::max());
  size_t M = lut.values.size();
  lut.thresholds.resize(M - 1);
  for (size_t i = 0; i + 1 < M; ++i) {
    // Smallest key whose Universal-rounded index is >= i+1 (index is
    // monotone non-decreasing in the key).
    uint32_t lo = kMin, hi = kMax;
    size_t idxHi;
    if (!indexOf(floatFromOrderKey(hi), idxHi) || idxHi < i + 1)
      return false;
    while (lo < hi) {
      uint32_t mid = lo + (hi - lo) / 2;
      size_t idx;
      if (!indexOf(floatFromOrderKey(mid), idx))
        return false;
      if (idx >= i + 1)
        hi = mid;
      else
        lo = mid + 1;
    }
    lut.thresholds[i] = lo;
  }
  return true;
}

// Dispatch table for the (nbits, es) pairs this pass's candidate format set
// actually needs (p4e1/p6e1/p8e1). sw::universal::posit<N,ES> requires N/ES
// as compile-time template parameters, so -- same as
// experiments/common/posit_fakequant.cpp's own roundtrip<N,ES> dispatch --
// this is a finite switch rather than a fully generic function; extend here
// if more (nbits,es) pairs are needed later.
static bool buildPositLut(unsigned nbits, unsigned es, PositLut &lut) {
  if (nbits == 4 && es == 1)
    return buildPositLutT<4, 1>(lut);
  if (nbits == 6 && es == 1)
    return buildPositLutT<6, 1>(lut);
  if (nbits == 8 && es == 1)
    return buildPositLutT<8, 1>(lut);
  return false;
}

static const PositLut *getPositLutCached(unsigned nbits, unsigned es) {
  static std::map<std::pair<unsigned, unsigned>, PositLut> cache;
  auto key = std::make_pair(nbits, es);
  auto it = cache.find(key);
  if (it != cache.end())
    return &it->second;
  PositLut lut;
  if (!buildPositLut(nbits, es, lut))
    return nullptr;
  return &cache.emplace(key, std::move(lut)).first->second;
}

// Reproduces sw::universal::posit<N,ES>'s float -> posit -> float round trip
// exactly. NaN and +-Inf become NaR in Universal, which decodes to NaN, so
// they map to NaN here too -- posit has no infinity to preserve them as.
// (Different from the INT weight path, which passes non-finite elements
// through unchanged: an INT format is simulated by arithmetic on the value,
// whereas a posit conversion has a defined result for them.)
static float fakeQuantPositViaLut(float f, const PositLut &lut) {
  if (!std::isfinite(f))
    return std::numeric_limits<float>::quiet_NaN();
  uint32_t k = floatOrderKey(f);
  size_t idx = static_cast<size_t>(
      std::upper_bound(lut.thresholds.begin(), lut.thresholds.end(), k) -
      lut.thresholds.begin());
  return lut.values[idx];
}

static DenseElementsAttr fakeQuantWeightPosit(
    DenseElementsAttr floatAttr, unsigned nbits, unsigned es) {
  const PositLut *lut = getPositLutCached(nbits, es);
  if (!lut)
    return nullptr;
  auto floatTy = mlir::cast<RankedTensorType>(floatAttr.getType());
  SmallVector<float, 64> out;
  out.reserve(floatAttr.getNumElements());
  for (float f : floatAttr.getValues<float>())
    out.push_back(fakeQuantPositViaLut(f, *lut));
  return DenseElementsAttr::get(floatTy, ArrayRef<float>(out));
}

// ===========================================================================
// F16/BF16 weight fake-quant: APFloat round trip through the target type's
// own float semantics, same technique ONNXToLowPrecision.cpp's
// buildScalarF8Constant comment already documents relying on for F8 (MLIR's
// own FloatAttr/APFloat machinery implements correct low-precision
// rounding) -- simpler here since F16/BF16 are "plain" IEEE-shaped types.
// ===========================================================================

static DenseElementsAttr fakeQuantWeightFloat(
    DenseElementsAttr floatAttr, FloatType narrowTy) {
  auto floatTy = mlir::cast<RankedTensorType>(floatAttr.getType());
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
  return DenseElementsAttr::get(floatTy, ArrayRef<float>(out));
}

static DenseElementsAttr fakeQuantWeight(
    DenseElementsAttr attr, const FormatSpec &spec) {
  if (spec.family == FormatFamily::FP32)
    return attr;
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
    return fakeQuantWeightInt(attr, spec.bits);
  case FormatFamily::POSIT:
    return fakeQuantWeightPosit(attr, spec.bits, spec.es);
  case FormatFamily::F16:
    return fakeQuantWeightFloat(attr, Builder(attr.getContext()).getF16Type());
  case FormatFamily::BF16:
    return fakeQuantWeightFloat(attr, Builder(attr.getContext()).getBF16Type());
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
        DenseElementsAttr qAttr = fakeQuantWeight(wAttr, entry.weightFormat);
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

} // namespace

std::unique_ptr<mlir::Pass> createConvertONNXToMixedPrecisionPass() {
  return std::make_unique<ConvertONNXToMixedPrecisionPass>();
}

} // namespace onnx_mlir
