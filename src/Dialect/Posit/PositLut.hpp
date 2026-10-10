/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------ PositLut.hpp - posit<N,ES> rounding LUT -----------===//
//
// Exact sw::universal::posit<N,ES> float -> posit -> float rounding for small
// N, as a table of decision boundaries built once per (nbits, es) with
// Universal itself as the oracle (see PositLut.cpp). Shared by every pass
// that needs posit rounding at compile time: --convert-onnx-to-mixed-precision
// folds posit weights with it. Universal is only included by PositLut.cpp.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>
#include <vector>

namespace onnx_mlir {

struct PositLut {
  // Finite representable values, ascending. NaR is excluded (it decodes to
  // NaN and would break the ordering).
  std::vector<float> values;
  // thresholds[i] is the smallest floatOrderKey that Universal rounds to
  // values[i+1] or above; size == values.size() - 1.
  std::vector<uint32_t> thresholds;
};

// Maps a finite float to a uint32_t whose unsigned order matches the float's
// numeric order (-0.0 sorts immediately below +0.0), so the float line can be
// bisected as a plain integer range.
uint32_t floatOrderKey(float f);
float floatFromOrderKey(uint32_t k);

// The LUT for posit<nbits,es>, built on first use and cached (thread-safe).
// nullptr for an (nbits, es) pair that is not wired up -- (4,1), (6,1), (8,1)
// are -- or if Universal ever behaved contrary to what the construction relies
// on; callers then treat the format as unsupported.
const PositLut *getPositLutCached(unsigned nbits, unsigned es);

// Reproduces sw::universal::posit<N,ES>'s float -> posit -> float round trip
// exactly. NaN and +-Inf become NaR in Universal, which decodes to NaN, so
// they map to NaN here too -- posit has no infinity to preserve them as.
float fakeQuantPositViaLut(float f, const PositLut &lut);

} // namespace onnx_mlir
