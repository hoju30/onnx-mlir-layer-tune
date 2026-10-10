/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ PositLut.cpp - posit<N,ES> rounding LUT -----------===//
//
// Posit rounding is NOT "nearest representable value" in arithmetic terms --
// it rounds on the encoded bit string (regime/exponent/fraction), a nonzero
// input never underflows to 0 (it lands on minpos), anything past maxpos
// saturates, and NaN/+-Inf become NaR. Confirmed by direct comparison against
// sw::universal::posit<N,ES>: a nearest-value search over the decoded table
// disagrees with it on ~30% of probed inputs.
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
// rounding rules -- and then each element costs one binary search over at
// most 2^N-2 boundaries instead of a general posit encode. Larger N would
// need a prohibitively large table and is not wired up.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/Posit/PositLut.hpp"

#include <universal/number/posit/posit.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace onnx_mlir {

uint32_t floatOrderKey(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

float floatFromOrderKey(uint32_t k) {
  uint32_t u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

namespace {

// Builds the LUT for posit<N,ES>. Returns false if Universal ever behaves
// contrary to what the construction relies on (a finite input converting to
// something not in the value table, e.g. NaR) -- the caller then treats the
// format as unsupported rather than producing a wrong table.
template <int N, int ES>
bool buildPositLutT(PositLut &lut) {
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

// Dispatch table for the (nbits, es) pairs the mixed-precision candidate
// format set needs (p4e1/p6e1/p8e1). sw::universal::posit<N,ES> requires N/ES
// as compile-time template parameters, so -- same as
// experiments/common/posit_fakequant.cpp's own roundtrip<N,ES> dispatch --
// this is a finite switch rather than a fully generic function; extend here
// if more (nbits,es) pairs are needed later.
bool buildPositLut(unsigned nbits, unsigned es, PositLut &lut) {
  if (nbits == 4 && es == 1)
    return buildPositLutT<4, 1>(lut);
  if (nbits == 6 && es == 1)
    return buildPositLutT<6, 1>(lut);
  if (nbits == 8 && es == 1)
    return buildPositLutT<8, 1>(lut);
  return false;
}

} // namespace

const PositLut *getPositLutCached(unsigned nbits, unsigned es) {
  static std::mutex mutex;
  static std::map<std::pair<unsigned, unsigned>, PositLut> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto key = std::make_pair(nbits, es);
  auto it = cache.find(key);
  if (it != cache.end())
    return &it->second;
  PositLut lut;
  if (!buildPositLut(nbits, es, lut))
    return nullptr;
  return &cache.emplace(key, std::move(lut)).first->second;
}

// (Different from the INT weight path of --convert-onnx-to-mixed-precision,
// which passes non-finite elements through unchanged: an INT format is
// simulated by arithmetic on the value, whereas a posit conversion has a
// defined result for them.)
float fakeQuantPositViaLut(float f, const PositLut &lut) {
  if (!std::isfinite(f))
    return std::numeric_limits<float>::quiet_NaN();
  uint32_t k = floatOrderKey(f);
  size_t idx = static_cast<size_t>(
      std::upper_bound(lut.thresholds.begin(), lut.thresholds.end(), k) -
      lut.thresholds.begin());
  return lut.values[idx];
}

} // namespace onnx_mlir
