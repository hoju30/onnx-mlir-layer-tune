// [EXTEND][2026-02-23] Posit runtime (SoftPosit or Universal backend)
// - Keeps MLIR CRunner ABI (`UnrankedMemRefType` / `DynamicMemRefType`)
// - Provides generic kernels instantiated for posit formats used by the generated .ll
// - Enable Universal backend with: -DPOSIT_USE_UNIVERSAL and include path to universal headers

#if defined(POSIT_USE_UNIVERSAL)
#include <universal/utility/compiler.hpp>
#include <universal/utility/architecture.hpp>
#include <universal/utility/bit_cast.hpp>
#include <universal/utility/long_double.hpp>
#include <universal/traits/number_traits.hpp>
#include <universal/traits/arithmetic_traits.hpp>
#include <universal/common/number_traits_reports.hpp>
#include <universal/number/posit/exceptions.hpp>
#include <universal/number/posit/posit_fwd.hpp>
#include <universal/number/posit/posit_impl.hpp>
#include <universal/number/posit/quire.hpp>
#include <universal/traits/posit_traits.hpp>
#include <universal/number/posit/numeric_limits.hpp>
#else
extern "C" {
#include "softposit.h"
}
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
#include <universal/utility/compiler.hpp>
#include <universal/utility/architecture.hpp>
#include <universal/utility/bit_cast.hpp>
#include <universal/utility/long_double.hpp>
#include <universal/traits/number_traits.hpp>
#include <universal/traits/arithmetic_traits.hpp>
#include <universal/common/number_traits_reports.hpp>
#include <universal/number/posit/exceptions.hpp>
#include <universal/number/posit/posit_fwd.hpp>
#include <universal/number/posit/posit_impl.hpp>
#include <universal/traits/posit_traits.hpp>
#include <universal/number/posit/numeric_limits.hpp>
#endif
#endif

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

template <typename T>
static inline int64_t num_elems(const DynamicMemRefType<T> &m) {
  int64_t n = 1;
  for (int i = 0; i < m.rank; ++i)
    n *= m.sizes[i];
  return n;
}

template <typename T>
static inline int64_t offset_of(const DynamicMemRefType<T> &m, int64_t linear) {
  int64_t off = m.offset;
  int64_t rem = linear;
  for (int i = m.rank - 1; i >= 0; --i) {
    int64_t idx = rem % m.sizes[i];
    rem /= m.sizes[i];
    off += idx * m.strides[i];
  }
  return off;
}

template <typename T>
static inline std::make_unsigned_t<T> to_bits(T v) {
  return static_cast<std::make_unsigned_t<T>>(v);
}

template <typename T>
static inline T from_bits(std::make_unsigned_t<T> v) {
  return static_cast<T>(v);
}

template <typename T>
static inline std::make_unsigned_t<T> load1(const DynamicMemRefType<T> &m, int64_t i) {
  int64_t off = m.offset + i * m.strides[0];
  return to_bits(m.data[off]);
}

template <typename T>
static inline std::make_unsigned_t<T> load2(const DynamicMemRefType<T> &m, int64_t i, int64_t j) {
  int64_t off = m.offset + i * m.strides[0] + j * m.strides[1];
  return to_bits(m.data[off]);
}

template <typename T>
static inline void store2(DynamicMemRefType<T> &m, int64_t i, int64_t j,
                          std::make_unsigned_t<T> bits) {
  int64_t off = m.offset + i * m.strides[0] + j * m.strides[1];
  m.data[off] = from_bits<T>(bits);
}

template <typename T>
static inline std::make_unsigned_t<T> load3(const DynamicMemRefType<T> &m, int64_t i,
                                            int64_t j, int64_t k) {
  int64_t off = m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2];
  return to_bits(m.data[off]);
}

template <typename T>
static inline void store3(DynamicMemRefType<T> &m, int64_t i, int64_t j, int64_t k,
                          std::make_unsigned_t<T> bits) {
  int64_t off = m.offset + i * m.strides[0] + j * m.strides[1] + k * m.strides[2];
  m.data[off] = from_bits<T>(bits);
}

template <typename T>
static inline std::make_unsigned_t<T> load_broadcast_nd(const DynamicMemRefType<T> &m,
                                                        const int64_t *outIdx,
                                                        int64_t outRank) {
  if (m.rank == 0)
    return to_bits(m.data[m.offset]);
  int64_t delta = outRank - m.rank;
  int64_t off = m.offset;
  for (int64_t od = 0; od < outRank; ++od) {
    int64_t md = od - delta;
    if (md < 0)
      continue;
    int64_t idx = outIdx[od];
    if (m.sizes[md] == 1)
      idx = 0;
    off += idx * m.strides[md];
  }
  return to_bits(m.data[off]);
}

template <typename T>
static inline std::make_unsigned_t<T> load4(const DynamicMemRefType<T> &m, int64_t n,
                                            int64_t c, int64_t h, int64_t w) {
  int64_t off = m.offset + n * m.strides[0] + c * m.strides[1] + h * m.strides[2] +
                w * m.strides[3];
  return to_bits(m.data[off]);
}

template <typename T>
static inline void store4(DynamicMemRefType<T> &m, int64_t n, int64_t c, int64_t h,
                          int64_t w, std::make_unsigned_t<T> bits) {
  int64_t off = m.offset + n * m.strides[0] + c * m.strides[1] + h * m.strides[2] +
                w * m.strides[3];
  m.data[off] = from_bits<T>(bits);
}

template <typename T>
static inline std::make_unsigned_t<T> loadC_broadcast(const DynamicMemRefType<T> &c, int64_t i,
                                                      int64_t j) {
  if (c.rank == 0)
    return to_bits(c.data[c.offset]);
  if (c.rank == 1) {
    int64_t jj = (c.sizes[0] == 1) ? 0 : j;
    return to_bits(c.data[c.offset + jj * c.strides[0]]);
  }
  int64_t ii = (c.sizes[0] == 1) ? 0 : i;
  int64_t jj = (c.sizes[1] == 1) ? 0 : j;
  return load2(c, ii, jj);
}

//===----------------------------------------------------------------------===//
// Format traits (backend-specific)
//===----------------------------------------------------------------------===//

#if defined(POSIT_USE_UNIVERSAL)

template <int NBits, int ES, typename MemT_, typename UIntT_>
struct UniversalFmt {
  using MemT = MemT_;
  using UIntT = UIntT_;
  using PositT = sw::universal::posit<NBits, ES>;

  static PositT fromRaw(UIntT b) {
    PositT p;
    p.setbits(static_cast<uint64_t>(b));
    return p;
  }
  static UIntT toRaw(const PositT &p) {
    return static_cast<UIntT>(p.bits().to_ull());
  }
  static PositT fromDouble(double x) {
    PositT p;
    p = x;
    return p;
  }
  static double toDouble(const PositT &p) { return static_cast<double>(p); }
  static PositT add(const PositT &a, const PositT &b) { return a + b; }
  static PositT sub(const PositT &a, const PositT &b) { return a - b; }
  static PositT mul(const PositT &a, const PositT &b) { return a * b; }
  static PositT div(const PositT &a, const PositT &b) { return a / b; }
};

using FmtP8E0 = UniversalFmt<8, 0, int8_t, uint8_t>;
using FmtP8E1 = UniversalFmt<8, 1, int8_t, uint8_t>;
using FmtP8E2 = UniversalFmt<8, 2, int8_t, uint8_t>;
using FmtP4E0 = UniversalFmt<4, 0, int8_t, uint8_t>;
using FmtP4E1 = UniversalFmt<4, 1, int8_t, uint8_t>;
using FmtP4E2 = UniversalFmt<4, 2, int8_t, uint8_t>;
using FmtP4E3 = UniversalFmt<4, 3, int8_t, uint8_t>;
using FmtP5E0 = UniversalFmt<5, 0, int8_t, uint8_t>;
using FmtP5E1 = UniversalFmt<5, 1, int8_t, uint8_t>;
using FmtP5E2 = UniversalFmt<5, 2, int8_t, uint8_t>;
using FmtP5E3 = UniversalFmt<5, 3, int8_t, uint8_t>;
using FmtP6E0 = UniversalFmt<6, 0, int8_t, uint8_t>;
using FmtP6E1 = UniversalFmt<6, 1, int8_t, uint8_t>;
using FmtP6E2 = UniversalFmt<6, 2, int8_t, uint8_t>;
using FmtP6E3 = UniversalFmt<6, 3, int8_t, uint8_t>;
using FmtP7E0 = UniversalFmt<7, 0, int8_t, uint8_t>;
using FmtP7E1 = UniversalFmt<7, 1, int8_t, uint8_t>;
using FmtP7E2 = UniversalFmt<7, 2, int8_t, uint8_t>;
using FmtP7E3 = UniversalFmt<7, 3, int8_t, uint8_t>;
using FmtP9E0 = UniversalFmt<9, 0, int16_t, uint16_t>;
using FmtP9E1 = UniversalFmt<9, 1, int16_t, uint16_t>;
using FmtP9E2 = UniversalFmt<9, 2, int16_t, uint16_t>;
using FmtP9E3 = UniversalFmt<9, 3, int16_t, uint16_t>;
using FmtP16E0 = UniversalFmt<16, 0, int16_t, uint16_t>;
using FmtP16E1 = UniversalFmt<16, 1, int16_t, uint16_t>;
using FmtP16E2 = UniversalFmt<16, 2, int16_t, uint16_t>;
using FmtP32E0 = UniversalFmt<32, 0, int32_t, uint32_t>;
using FmtP32E1 = UniversalFmt<32, 1, int32_t, uint32_t>;
using FmtP32E2 = UniversalFmt<32, 2, int32_t, uint32_t>;

#else

// SoftPosit dynamic posit path for p8e1: map 8-bit raw to/from posit_1_t with x=8.
// This requires SoftPosit to export pX1_* symbols.
// Build with -DPOSIT_USE_SOFTPOSIT_PX1 to enable.
struct FmtP8E1ViaPX1 {
  using MemT = int8_t;
  using UIntT = uint8_t;
  using PositT = posit_1_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b) << 24;
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX1_to_pX1(p, 8);
    return static_cast<UIntT>((p.v >> 24) & 0xFFu);
  }
  static PositT fromDouble(double x) { return convertDoubleToPX1(x, 8); }
  static double toDouble(PositT p) { return convertPX1ToDouble(p); }
  static PositT add(PositT a, PositT b) { return pX1_add(a, b, 8); }
  static PositT sub(PositT a, PositT b) { return pX1_sub(a, b, 8); }
  static PositT mul(PositT a, PositT b) { return pX1_mul(a, b, 8); }
  static PositT div(PositT a, PositT b) { return pX1_div(a, b, 8); }
};

// SoftPosit dynamic posit path for p8e2: map 8-bit raw to/from posit_2_t with x=8.
// This requires SoftPosit to export pX2/qX2 symbols.
// Build with -DPOSIT_USE_SOFTPOSIT_PX2 to enable.
struct FmtP8E2ViaPX2 {
  using MemT = int8_t;
  using UIntT = uint8_t;
  using PositT = posit_2_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b) << 24;
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX2_to_pX2(p, 8);
    return static_cast<UIntT>((p.v >> 24) & 0xFFu);
  }
  static PositT fromDouble(double x) { return convertDoubleToPX2(x, 8); }
  static double toDouble(PositT p) { return convertPX2ToDouble(p); }
  static PositT add(PositT a, PositT b) { return pX2_add(a, b, 8); }
  static PositT sub(PositT a, PositT b) { return pX2_sub(a, b, 8); }
  static PositT mul(PositT a, PositT b) { return pX2_mul(a, b, 8); }
  static PositT div(PositT a, PositT b) { return pX2_div(a, b, 8); }
};

struct FmtP8E0 {
  using MemT = int8_t;
  using UIntT = uint8_t;
  using PositT = posit8_t;
  static PositT fromRaw(UIntT b) { return castP8(b); }
  static UIntT toRaw(PositT p) { return static_cast<UIntT>(castUI(p) & 0xFFu); }
  static PositT fromDouble(double x) { return convertDoubleToP8(x); }
  static double toDouble(PositT p) { return convertP8ToDouble(p); }
  static PositT add(PositT a, PositT b) { return p8_add(a, b); }
  static PositT sub(PositT a, PositT b) { return p8_sub(a, b); }
  static PositT mul(PositT a, PositT b) { return p8_mul(a, b); }
  static PositT div(PositT a, PositT b) { return p8_div(a, b); }
};

struct FmtP16E1 {
  using MemT = int16_t;
  using UIntT = uint16_t;
  using PositT = posit16_t;
  static PositT fromRaw(UIntT b) { return castP16(b); }
  static UIntT toRaw(PositT p) { return static_cast<UIntT>(castUI(p) & 0xFFFFu); }
  static PositT fromDouble(double x) { return convertDoubleToP16(x); }
  static double toDouble(PositT p) { return convertP16ToDouble(p); }
  static PositT add(PositT a, PositT b) { return p16_add(a, b); }
  static PositT sub(PositT a, PositT b) { return p16_sub(a, b); }
  static PositT mul(PositT a, PositT b) { return p16_mul(a, b); }
  static PositT div(PositT a, PositT b) { return p16_div(a, b); }
};

// SoftPosit dynamic posit path for p16e2: map 16-bit raw to/from posit_2_t with x=16.
#if defined(POSIT_USE_SOFTPOSIT_PX2)
struct FmtP16E2ViaPX2 {
  using MemT = int16_t;
  using UIntT = uint16_t;
  using PositT = posit_2_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b) << 16;
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX2_to_pX2(p, 16);
    return static_cast<UIntT>((p.v >> 16) & 0xFFFFu);
  }
  static PositT fromDouble(double x) { return convertDoubleToPX2(x, 16); }
  static double toDouble(PositT p) { return convertPX2ToDouble(p); }
  static PositT add(PositT a, PositT b) { return pX2_add(a, b, 16); }
  static PositT sub(PositT a, PositT b) { return pX2_sub(a, b, 16); }
  static PositT mul(PositT a, PositT b) { return pX2_mul(a, b, 16); }
  static PositT div(PositT a, PositT b) { return pX2_div(a, b, 16); }
};
#endif

#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
template <int NBits, int ES, typename MemT_, typename UIntT_>
struct SoftPositUniversalFallbackFmt {
  using MemT = MemT_;
  using UIntT = UIntT_;
  using PositT = sw::universal::posit<NBits, ES>;

  static PositT fromRaw(UIntT b) {
    PositT p;
    p.setbits(static_cast<uint64_t>(b));
    return p;
  }
  static UIntT toRaw(const PositT &p) {
    return static_cast<UIntT>(p.bits().to_ull());
  }
  static PositT fromDouble(double x) {
    PositT p;
    p = x;
    return p;
  }
  static double toDouble(const PositT &p) { return static_cast<double>(p); }
  static PositT add(const PositT &a, const PositT &b) { return a + b; }
  static PositT sub(const PositT &a, const PositT &b) { return a - b; }
  static PositT mul(const PositT &a, const PositT &b) { return a * b; }
  static PositT div(const PositT &a, const PositT &b) { return a / b; }
};

using FmtP16E0 = SoftPositUniversalFallbackFmt<16, 0, int16_t, uint16_t>;
using FmtP32E0 = SoftPositUniversalFallbackFmt<32, 0, int32_t, uint32_t>;
#endif

// SoftPosit default backend is kept for legacy formats only.
// Full p8/p16/p32 x e0/e1/e2 coverage requires POSIT_USE_UNIVERSAL.

#if defined(POSIT_USE_SOFTPOSIT_PX1)
struct FmtP32E1ViaPX1 {
  using MemT = int32_t;
  using UIntT = uint32_t;
  using PositT = posit_1_t;
  static PositT fromRaw(UIntT b) {
    PositT p;
    p.v = static_cast<uint32_t>(b);
    return p;
  }
  static UIntT toRaw(PositT p) {
    p = pX1_to_pX1(p, 32);
    return static_cast<UIntT>(p.v);
  }
  static PositT fromDouble(double x) { return convertDoubleToPX1(x, 32); }
  static double toDouble(PositT p) { return convertPX1ToDouble(p); }
  static PositT add(PositT a, PositT b) { return pX1_add(a, b, 32); }
  static PositT sub(PositT a, PositT b) { return pX1_sub(a, b, 32); }
  static PositT mul(PositT a, PositT b) { return pX1_mul(a, b, 32); }
  static PositT div(PositT a, PositT b) { return pX1_div(a, b, 32); }
};
#endif

struct FmtP32E2 {
  using MemT = int32_t;
  using UIntT = uint32_t;
  using PositT = posit32_t;
  static PositT fromRaw(UIntT b) { return castP32(b); }
  static UIntT toRaw(PositT p) { return static_cast<UIntT>(castUI(p)); }
  static PositT fromDouble(double x) { return convertDoubleToP32(x); }
  static double toDouble(PositT p) { return convertP32ToDouble(p); }
  static PositT add(PositT a, PositT b) { return p32_add(a, b); }
  static PositT sub(PositT a, PositT b) { return p32_sub(a, b); }
  static PositT mul(PositT a, PositT b) { return p32_mul(a, b); }
  static PositT div(PositT a, PositT b) { return p32_div(a, b); }
};

#endif

template <typename Fmt>
static inline typename Fmt::UIntT bits_from_double_fmt(double x) {
  return Fmt::toRaw(Fmt::fromDouble(x));
}

template <typename Fmt>
static inline double double_from_bits_fmt(typename Fmt::UIntT b) {
  return Fmt::toDouble(Fmt::fromRaw(b));
}

static bool positQuireEnabledForP8() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  const char *e = std::getenv("POSIT_QUIRE_P8");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (s == "0" || s == "off" || s == "false" || s == "no")
    enabled = 0;
  else
    enabled = 1;
  return enabled == 1;
}

static bool positQuireEnabledForSmall() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled == 1;
  // Dedicated switch for extra low-bit formats.
  const char *e = std::getenv("POSIT_QUIRE_SMALL");
  // Backward-compatible: if not set, fall back to p8 switch.
  if (!e || !*e)
    e = std::getenv("POSIT_QUIRE_P8");
  if (!e || !*e) {
    enabled = 1;
    return true;
  }
  std::string s(e);
  for (char &ch : s)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (s == "0" || s == "off" || s == "false" || s == "no")
    enabled = 0;
  else
    enabled = 1;
  return enabled == 1;
}

// Dot-product accumulator:
// - default: plain posit accumulate (acc += a*b in posit domain)
// - SoftPosit formats with quire: accumulate products in quire and convert once
//   at the end. This reduces rounding loss in GEMM/Conv inner loops.
template <typename Fmt>
struct DotAccumulator {
  using PositT = typename Fmt::PositT;
  struct State {
    PositT acc;
  };
  static inline State init() { return State{Fmt::fromDouble(0.0)}; }
  static inline void fdp(State &s, PositT a, PositT b) {
    s.acc = Fmt::add(s.acc, Fmt::mul(a, b));
  }
  static inline PositT finish(const State &s) { return s.acc; }
};

#if defined(POSIT_USE_UNIVERSAL)
template <typename PositT, typename QuireT>
static inline PositT finishUniversalQuire(const QuireT &q) {
  PositT out;
  sw::universal::convert(q.to_value(), out);
  return out;
}

template <>
struct DotAccumulator<FmtP8E0> {
  using PositT = typename FmtP8E0::PositT;
  using QuireT = sw::universal::quire<8, 0, 20>;
  struct State {
    bool useQuire;
    QuireT q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8(), QuireT(0), FmtP8E0::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q += sw::universal::quire_mul(a, b);
      return;
    }
    s.acc = FmtP8E0::add(s.acc, FmtP8E0::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? finishUniversalQuire<PositT>(s.q) : s.acc;
  }
};

template <>
struct DotAccumulator<FmtP8E1> {
  using PositT = typename FmtP8E1::PositT;
  using QuireT = sw::universal::quire<8, 1, 20>;
  struct State {
    bool useQuire;
    QuireT q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8(), QuireT(0), FmtP8E1::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q += sw::universal::quire_mul(a, b);
      return;
    }
    s.acc = FmtP8E1::add(s.acc, FmtP8E1::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? finishUniversalQuire<PositT>(s.q) : s.acc;
  }
};

template <>
struct DotAccumulator<FmtP8E2> {
  using PositT = typename FmtP8E2::PositT;
  using QuireT = sw::universal::quire<8, 2, 20>;
  struct State {
    bool useQuire;
    QuireT q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8(), QuireT(0), FmtP8E2::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q += sw::universal::quire_mul(a, b);
      return;
    }
    s.acc = FmtP8E2::add(s.acc, FmtP8E2::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? finishUniversalQuire<PositT>(s.q) : s.acc;
  }
};

// Use quire accumulation for extra formats with promoted posit storage.
// - p4..p7 -> promote to posit<8,es> + quire<8,es,20>
// - p9     -> promote to posit<16,es> + quire<16,es,20>
// This keeps quire-based accumulation while avoiding unstable small-nbits
// quire template instantiations.
#define DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FMT_ALIAS, QNBITS, ESBITS)                                 \
  template <>                                                                                        \
  struct DotAccumulator<FMT_ALIAS> {                                                                 \
    using PositT = typename FMT_ALIAS::PositT;                                                       \
    using QPositT = sw::universal::posit<QNBITS, ESBITS>;                                            \
    using QuireT = sw::universal::quire<QNBITS, ESBITS, 20>;                                         \
    struct State {                                                                                   \
      bool useQuire;                                                                                 \
      QuireT q;                                                                                      \
      PositT acc;                                                                                    \
    };                                                                                               \
    static inline State init() {                                                                     \
      return State{positQuireEnabledForSmall(), QuireT(0), FMT_ALIAS::fromDouble(0.0)};            \
    }                                                                                                \
    static inline void fdp(State &s, PositT a, PositT b) {                                          \
      if (s.useQuire) {                                                                              \
        QPositT qa = static_cast<double>(a);                                                         \
        QPositT qb = static_cast<double>(b);                                                         \
        s.q += sw::universal::quire_mul(qa, qb);                                                     \
        return;                                                                                      \
      }                                                                                              \
      s.acc = FMT_ALIAS::add(s.acc, FMT_ALIAS::mul(a, b));                                          \
    }                                                                                                \
    static inline PositT finish(const State &s) {                                                    \
      if (!s.useQuire)                                                                               \
        return s.acc;                                                                                \
      QPositT qOut;                                                                                  \
      sw::universal::convert(s.q.to_value(), qOut);                                                  \
      return FMT_ALIAS::fromDouble(static_cast<double>(qOut));                                       \
    }                                                                                                \
  };

DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP4E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP5E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP6E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E0, 8, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E1, 8, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E2, 8, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP7E3, 8, 3)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E0, 16, 0)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E1, 16, 1)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E2, 16, 2)
DEFINE_UNIVERSAL_SMALL_DOT_ACCUM(FmtP9E3, 16, 3)

#undef DEFINE_UNIVERSAL_SMALL_DOT_ACCUM
#endif

#if !defined(POSIT_USE_UNIVERSAL)
template <>
struct DotAccumulator<FmtP8E0> {
  using PositT = typename FmtP8E0::PositT;
  struct State {
    bool useQuire;
    quire8_t q;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8(), q8Clr(), FmtP8E0::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q = q8_fdp_add(s.q, a, b);
      return;
    }
    s.acc = FmtP8E0::add(s.acc, FmtP8E0::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? q8_to_p8(s.q) : s.acc;
  }
};

#if defined(POSIT_USE_SOFTPOSIT_PX1)
template <>
struct DotAccumulator<FmtP8E1ViaPX1> {
  using PositT = typename FmtP8E1ViaPX1::PositT;
  struct State {
    bool useQuire;
    long double qacc;
    PositT acc;
  };
  static inline State init() {
    return State{positQuireEnabledForP8(), 0.0L, FmtP8E1ViaPX1::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.qacc += static_cast<long double>(FmtP8E1ViaPX1::toDouble(a)) *
                static_cast<long double>(FmtP8E1ViaPX1::toDouble(b));
      return;
    }
    s.acc = FmtP8E1ViaPX1::add(s.acc, FmtP8E1ViaPX1::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? FmtP8E1ViaPX1::fromDouble(static_cast<double>(s.qacc)) : s.acc;
  }
};
#endif

#if defined(POSIT_USE_SOFTPOSIT_PX2)
template <>
struct DotAccumulator<FmtP8E2ViaPX2> {
  using PositT = typename FmtP8E2ViaPX2::PositT;
  struct State {
    bool useQuire;
    quire_2_t q;
    PositT acc;
  };
  static inline State init() {
    return State{
        positQuireEnabledForP8(), qX2Clr(), FmtP8E2ViaPX2::fromDouble(0.0)};
  }
  static inline void fdp(State &s, PositT a, PositT b) {
    if (s.useQuire) {
      s.q = qX2_fdp_add(s.q, a, b);
      return;
    }
    s.acc = FmtP8E2ViaPX2::add(s.acc, FmtP8E2ViaPX2::mul(a, b));
  }
  static inline PositT finish(const State &s) {
    return s.useQuire ? qX2_to_pX2(s.q, 8) : s.acc;
  }
};
#endif
#endif

template <typename Fmt, typename LoadA, typename LoadB>
static inline typename Fmt::PositT dot_product_accumulate(
    int64_t K, LoadA &&loadA, LoadB &&loadB) {
  using Acc = DotAccumulator<Fmt>;
  auto st = Acc::init();
  for (int64_t k = 0; k < K; ++k)
    Acc::fdp(st, loadA(k), loadB(k));
  return Acc::finish(st);
}

static bool positTraceEnabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *e = std::getenv("POSIT_TRACE");
    enabled = (e && *e && std::string(e) != "0") ? 1 : 0;
  }
  return enabled == 1;
}

static int64_t positTraceMaxElems() {
  static int64_t v = -1;
  if (v >= 0)
    return v;
  const char *e = std::getenv("POSIT_TRACE_MAX_ELEMS");
  if (!e || !*e) {
    v = 8;
    return v;
  }
  char *end = nullptr;
  long long parsed = std::strtoll(e, &end, 10);
  if (!end || *end != '\0' || parsed <= 0)
    v = 8;
  else
    v = parsed;
  return v;
}

template <typename Fmt>
static void traceMemrefSummary(const char *opName,
                               const DynamicMemRefType<typename Fmt::MemT> &m) {
  if (!positTraceEnabled())
    return;

  int64_t n = num_elems(m);
  double mn = std::numeric_limits<double>::infinity();
  double mx = -std::numeric_limits<double>::infinity();
  double sum = 0.0;
  double absmax = 0.0;
  int64_t k = std::min<int64_t>(n, positTraceMaxElems());
  std::vector<double> sample;
  sample.reserve(static_cast<size_t>(k));

  for (int64_t i = 0; i < n; ++i) {
    auto bits = to_bits(m.data[offset_of(m, i)]);
    double v = double_from_bits_fmt<Fmt>(bits);
    mn = std::min(mn, v);
    mx = std::max(mx, v);
    sum += v;
    absmax = std::max(absmax, std::fabs(v));
    if (i < k)
      sample.push_back(v);
  }

  double mean = (n > 0) ? (sum / static_cast<double>(n)) : 0.0;
  std::fprintf(stderr, "[PTRACE] op=%s rank=%lld shape=[", opName,
               static_cast<long long>(m.rank));
  for (int i = 0; i < m.rank; ++i) {
    std::fprintf(stderr, "%lld%s", static_cast<long long>(m.sizes[i]),
                 (i + 1 == m.rank) ? "" : ",");
  }
  std::fprintf(stderr, "] n=%lld min=%g max=%g mean=%g absmax=%g sample=[",
               static_cast<long long>(n), mn, mx, mean, absmax);
  for (size_t i = 0; i < sample.size(); ++i) {
    std::fprintf(stderr, "%g%s", sample[i], (i + 1 == sample.size()) ? "" : ",");
  }
  std::fprintf(stderr, "]\n");
}

//===----------------------------------------------------------------------===//
// Generic kernels
//===----------------------------------------------------------------------===//

template <typename Fmt>
static void posit_from_f32_kernel(UnrankedMemRefType<float> *In,
                                  UnrankedMemRefType<typename Fmt::MemT> *Out) {
  DynamicMemRefType<float> in(*In);
  DynamicMemRefType<typename Fmt::MemT> out(*Out);
  int64_t n = num_elems(in);
  for (int64_t i = 0; i < n; ++i) {
    float x = in.data[offset_of(in, i)];
    auto bits = bits_from_double_fmt<Fmt>(static_cast<double>(x));
    out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
  }
}

template <typename Fmt>
static void posit_to_f32_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                                UnrankedMemRefType<float> *Out) {
  DynamicMemRefType<typename Fmt::MemT> in(*In);
  DynamicMemRefType<float> out(*Out);
  int64_t n = num_elems(in);
  for (int64_t i = 0; i < n; ++i) {
    auto bits = to_bits(in.data[offset_of(in, i)]);
    out.data[offset_of(out, i)] = static_cast<float>(double_from_bits_fmt<Fmt>(bits));
  }
}

template <typename Fmt, typename OpFn>
static void elemwise_kernel(OpFn op, UnrankedMemRefType<typename Fmt::MemT> *A,
                            UnrankedMemRefType<typename Fmt::MemT> *B,
                            UnrankedMemRefType<typename Fmt::MemT> *Out) {
  DynamicMemRefType<typename Fmt::MemT> a(*A), b(*B), o(*Out);

  int64_t outElems = num_elems(o);
  int64_t outRank = o.rank;
  std::vector<int64_t> outIdx(static_cast<size_t>(std::max<int64_t>(outRank, 1)), 0);

  auto offset_with_broadcast = [&](const DynamicMemRefType<typename Fmt::MemT> &m) {
    if (m.rank == 0)
      return m.offset;
    int64_t rankDelta = outRank - m.rank;
    int64_t off = m.offset;
    for (int64_t od = 0; od < outRank; ++od) {
      int64_t md = od - rankDelta;
      if (md < 0)
        continue;
      int64_t idx = outIdx[static_cast<size_t>(od)];
      if (m.sizes[md] == 1)
        idx = 0;
      off += idx * m.strides[md];
    }
    return off;
  };

  for (int64_t lin = 0; lin < outElems; ++lin) {
    if (outRank > 0) {
      int64_t rem = lin;
      for (int64_t d = outRank - 1; d >= 0; --d) {
        outIdx[static_cast<size_t>(d)] = rem % o.sizes[d];
        rem /= o.sizes[d];
      }
    }

    auto abit = to_bits(a.data[offset_with_broadcast(a)]);
    auto bbit = to_bits(b.data[offset_with_broadcast(b)]);
    auto pa = Fmt::fromRaw(abit);
    auto pb = Fmt::fromRaw(bbit);
    auto pc = op(pa, pb);
    o.data[offset_of(o, lin)] = from_bits<typename Fmt::MemT>(Fmt::toRaw(pc));
  }
}

template <typename Fmt>
static void relu_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                        UnrankedMemRefType<typename Fmt::MemT> *Out) {
  DynamicMemRefType<typename Fmt::MemT> in(*In), out(*Out);
  int64_t n = num_elems(in);
  auto zeroBits = bits_from_double_fmt<Fmt>(0.0);
  for (int64_t i = 0; i < n; ++i) {
    auto bits = to_bits(in.data[offset_of(in, i)]);
    double v = double_from_bits_fmt<Fmt>(bits);
    out.data[offset_of(out, i)] =
        from_bits<typename Fmt::MemT>((v < 0.0) ? zeroBits : bits);
  }
}

template <typename Fmt>
static void gemm_kernel(UnrankedMemRefType<typename Fmt::MemT> *A,
                        UnrankedMemRefType<typename Fmt::MemT> *B,
                        UnrankedMemRefType<typename Fmt::MemT> *C,
                        UnrankedMemRefType<typename Fmt::MemT> *Y, float alpha, float beta,
                        int64_t transA, int64_t transB) {
  DynamicMemRefType<typename Fmt::MemT> a(*A), b(*B), c(*C), y(*Y);

  // Conv-im2col style GEMM for ONNX MatMul lowering:
  // A[M,K], B[N,K,S], Y[N,M,S] where N is batch-like leading dim.
  if (a.rank == 2 && b.rank == 3 && y.rank == 3 && transA == 0 && transB == 0) {
    int64_t M = a.sizes[0];
    int64_t K = a.sizes[1];
    int64_t N = b.sizes[0];
    int64_t S = b.sizes[2];

    auto pAlpha = Fmt::fromDouble(static_cast<double>(alpha));
    auto pBeta = Fmt::fromDouble(static_cast<double>(beta));

    for (int64_t n = 0; n < N; ++n) {
      for (int64_t m = 0; m < M; ++m) {
        for (int64_t s = 0; s < S; ++s) {
          auto dot = dot_product_accumulate<Fmt>(
              K,
              [&](int64_t k) { return Fmt::fromRaw(load2(a, m, k)); },
              [&](int64_t k) { return Fmt::fromRaw(load3(b, n, k, s)); });
          auto acc = dot;
          acc = Fmt::mul(acc, pAlpha);
          int64_t idx3[3] = {n, m, s};
          auto cbit = load_broadcast_nd(c, idx3, 3);
          auto addc = Fmt::mul(Fmt::fromRaw(cbit), pBeta);
          acc = Fmt::add(acc, addc);
          store3(y, n, m, s, Fmt::toRaw(acc));
        }
      }
    }
    return;
  }

  // Standard 2D GEMM.
  int64_t a0 = a.sizes[0], a1 = a.sizes[1];
  int64_t b0 = b.sizes[0], b1 = b.sizes[1];
  int64_t M = transA ? a1 : a0;
  int64_t K = transA ? a0 : a1;
  int64_t N = transB ? b0 : b1;

  auto pAlpha = Fmt::fromDouble(static_cast<double>(alpha));
  auto pBeta = Fmt::fromDouble(static_cast<double>(beta));

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      auto dot = dot_product_accumulate<Fmt>(
          K,
          [&](int64_t k) {
            return Fmt::fromRaw(transA ? load2(a, k, i) : load2(a, i, k));
          },
          [&](int64_t k) {
            return Fmt::fromRaw(transB ? load2(b, j, k) : load2(b, k, j));
          });
      auto acc = dot;
      acc = Fmt::mul(acc, pAlpha);
      int64_t idx2[2] = {i, j};
      auto cbit = load_broadcast_nd(c, idx2, 2);
      auto addc = Fmt::mul(Fmt::fromRaw(cbit), pBeta);
      acc = Fmt::add(acc, addc);
      store2(y, i, j, Fmt::toRaw(acc));
    }
  }
}

template <typename Fmt>
static void conv2d_nchw_kernel(UnrankedMemRefType<typename Fmt::MemT> *X,
                               UnrankedMemRefType<typename Fmt::MemT> *W,
                               UnrankedMemRefType<typename Fmt::MemT> *B,
                               UnrankedMemRefType<typename Fmt::MemT> *Out,
                               int64_t stride_h, int64_t stride_w, int64_t dilation_h,
                               int64_t dilation_w, int64_t pad_top, int64_t pad_left,
                               int64_t pad_bottom, int64_t pad_right, int64_t groups) {
  (void)pad_bottom;
  (void)pad_right;

  DynamicMemRefType<typename Fmt::MemT> x(*X), w(*W), b(*B), out(*Out);

  int64_t N = x.sizes[0];
  int64_t H = x.sizes[2];
  int64_t Wd = x.sizes[3];
  int64_t M = w.sizes[0];
  int64_t Cpg = w.sizes[1];
  int64_t kH = w.sizes[2];
  int64_t kW = w.sizes[3];
  int64_t outH = out.sizes[2];
  int64_t outW = out.sizes[3];

  int64_t g = (groups <= 0) ? 1 : groups;
  int64_t Mpg = M / g;
  using DotAcc = DotAccumulator<Fmt>;

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t gg = 0; gg < g; ++gg) {
      for (int64_t mm = 0; mm < Mpg; ++mm) {
        int64_t oc = gg * Mpg + mm;
        auto bias = Fmt::fromRaw(load1(b, oc));
        for (int64_t oh = 0; oh < outH; ++oh) {
          for (int64_t ow = 0; ow < outW; ++ow) {
            auto st = DotAcc::init();
            for (int64_t cc = 0; cc < Cpg; ++cc) {
              int64_t ic = gg * Cpg + cc;
              for (int64_t kh = 0; kh < kH; ++kh) {
                int64_t ih = oh * stride_h - pad_top + kh * dilation_h;
                if (ih < 0 || ih >= H)
                  continue;
                for (int64_t kw = 0; kw < kW; ++kw) {
                  int64_t iw = ow * stride_w - pad_left + kw * dilation_w;
                  if (iw < 0 || iw >= Wd)
                    continue;
                  auto px = Fmt::fromRaw(load4(x, n, ic, ih, iw));
                  auto pw = Fmt::fromRaw(load4(w, oc, cc, kh, kw));
                  DotAcc::fdp(st, px, pw);
                }
              }
            }
            auto acc = Fmt::add(bias, DotAcc::finish(st));
            store4(out, n, oc, oh, ow, Fmt::toRaw(acc));
          }
        }
      }
    }
  }
}

template <typename Fmt>
static void maxpool2d_nchw_kernel(UnrankedMemRefType<typename Fmt::MemT> *X,
                                  UnrankedMemRefType<typename Fmt::MemT> *Out,
                                  int64_t kernel_h, int64_t kernel_w, int64_t stride_h,
                                  int64_t stride_w, int64_t pad_top, int64_t pad_left,
                                  int64_t pad_bottom, int64_t pad_right,
                                  int64_t ceil_mode) {
  (void)pad_bottom;
  (void)pad_right;
  (void)ceil_mode;

  DynamicMemRefType<typename Fmt::MemT> x(*X), out(*Out);
  int64_t N = x.sizes[0];
  int64_t C = x.sizes[1];
  int64_t H = x.sizes[2];
  int64_t Wd = x.sizes[3];
  int64_t outH = out.sizes[2];
  int64_t outW = out.sizes[3];
  auto zeroBits = bits_from_double_fmt<Fmt>(0.0);

  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t oh = 0; oh < outH; ++oh) {
        for (int64_t ow = 0; ow < outW; ++ow) {
          bool any = false;
          typename Fmt::UIntT best = zeroBits;
          double bestv = 0.0;

          int64_t ih0 = oh * stride_h - pad_top;
          int64_t iw0 = ow * stride_w - pad_left;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            int64_t ih = ih0 + kh;
            if (ih < 0 || ih >= H)
              continue;
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              int64_t iw = iw0 + kw;
              if (iw < 0 || iw >= Wd)
                continue;
              auto bits = load4(x, n, c, ih, iw);
              double v = double_from_bits_fmt<Fmt>(bits);
              if (!any || v > bestv) {
                any = true;
                best = bits;
                bestv = v;
              }
            }
          }
          store4(out, n, c, oh, ow, any ? best : zeroBits);
        }
      }
    }
  }
}

template <typename Fmt>
static void clip_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                        UnrankedMemRefType<typename Fmt::MemT> *Out, float minVal,
                        float maxVal, int64_t hasMin, int64_t hasMax) {
  DynamicMemRefType<typename Fmt::MemT> in(*In), out(*Out);
  int64_t n = num_elems(in);
  for (int64_t i = 0; i < n; ++i) {
    double v = double_from_bits_fmt<Fmt>(to_bits(in.data[offset_of(in, i)]));
    if (hasMin)
      v = std::max(v, static_cast<double>(minVal));
    if (hasMax)
      v = std::min(v, static_cast<double>(maxVal));
    out.data[offset_of(out, i)] =
        from_bits<typename Fmt::MemT>(bits_from_double_fmt<Fmt>(v));
  }
}

template <typename Fmt>
static void reduce_mean_kernel(UnrankedMemRefType<typename Fmt::MemT> *In,
                               UnrankedMemRefType<typename Fmt::MemT> *Out,
                               int64_t axis0, int64_t axis1, int64_t keepdims) {
  DynamicMemRefType<typename Fmt::MemT> in(*In), out(*Out);
  if (!keepdims || in.rank != out.rank || in.rank <= 0) {
    // Conservative fallback.
    int64_t n = num_elems(in);
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i)
      acc += double_from_bits_fmt<Fmt>(to_bits(in.data[offset_of(in, i)]));
    double mean = (n > 0) ? (acc / static_cast<double>(n)) : 0.0;
    int64_t m = num_elems(out);
    auto bits = bits_from_double_fmt<Fmt>(mean);
    for (int64_t i = 0; i < m; ++i)
      out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
    return;
  }

  int64_t r = in.rank;
  if (axis0 < 0)
    axis0 += r;
  if (axis1 < 0)
    axis1 += r;
  if (axis0 < 0 || axis0 >= r || axis1 < 0 || axis1 >= r) {
    int64_t n = num_elems(in);
    double acc = 0.0;
    for (int64_t i = 0; i < n; ++i)
      acc += double_from_bits_fmt<Fmt>(to_bits(in.data[offset_of(in, i)]));
    double mean = (n > 0) ? (acc / static_cast<double>(n)) : 0.0;
    int64_t m = num_elems(out);
    auto bits = bits_from_double_fmt<Fmt>(mean);
    for (int64_t i = 0; i < m; ++i)
      out.data[offset_of(out, i)] = from_bits<typename Fmt::MemT>(bits);
    return;
  }

  int64_t outElems = num_elems(out);
  std::vector<int64_t> outIdx(static_cast<size_t>(r), 0);
  for (int64_t lin = 0; lin < outElems; ++lin) {
    int64_t rem = lin;
    for (int64_t d = r - 1; d >= 0; --d) {
      outIdx[static_cast<size_t>(d)] = rem % out.sizes[d];
      rem /= out.sizes[d];
    }

    double acc = 0.0;
    int64_t cnt = 0;
    for (int64_t i0 = 0; i0 < in.sizes[axis0]; ++i0) {
      for (int64_t i1 = 0; i1 < in.sizes[axis1]; ++i1) {
        int64_t off = in.offset;
        for (int64_t d = 0; d < r; ++d) {
          int64_t idx = outIdx[static_cast<size_t>(d)];
          if (d == axis0)
            idx = i0;
          else if (d == axis1)
            idx = i1;
          off += idx * in.strides[d];
        }
        acc += double_from_bits_fmt<Fmt>(to_bits(in.data[off]));
        ++cnt;
      }
    }
    double mean = (cnt > 0) ? (acc / static_cast<double>(cnt)) : 0.0;
    out.data[offset_of(out, lin)] =
        from_bits<typename Fmt::MemT>(bits_from_double_fmt<Fmt>(mean));
  }
}

template <typename Fmt>
static void dequantize_linear_kernel(UnrankedMemRefType<int8_t> *In,
                                     UnrankedMemRefType<typename Fmt::MemT> *Out,
                                     float scale, int64_t zeroPoint,
                                     int64_t hasZeroPoint, int64_t axis,
                                     int64_t inputSigned) {
  DynamicMemRefType<int8_t> in(*In);
  DynamicMemRefType<typename Fmt::MemT> out(*Out);

  // ONNX DQ uses per-axis broadcasting semantics for scale/zero_point.
  // This runtime ABI currently passes scalar scale/zero_point, so this kernel
  // applies scalar values but keeps axis-normalization/indexing logic explicit.
  int64_t inRank = in.rank;
  if (inRank <= 0) {
    int64_t inOff = in.offset;
    int64_t outOff = out.offset;
    double q = inputSigned ? static_cast<double>(in.data[inOff])
                           : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    if (hasZeroPoint)
      q -= static_cast<double>(zeroPoint);
    double v = q * static_cast<double>(scale);
    out.data[outOff] = from_bits<typename Fmt::MemT>(bits_from_double_fmt<Fmt>(v));
    return;
  }

  int64_t normAxis = axis;
  if (normAxis < 0)
    normAxis += inRank;
  if (normAxis < 0)
    normAxis = 0;
  if (normAxis >= inRank)
    normAxis = inRank - 1;

  int64_t inElems = num_elems(in);
  int64_t outElems = num_elems(out);
  int64_t n = std::min(inElems, outElems);

  auto axis_index_from_linear = [&](int64_t lin) -> int64_t {
    int64_t rem = lin;
    int64_t axisIdx = 0;
    for (int64_t d = inRank - 1; d >= 0; --d) {
      int64_t dim = in.sizes[d];
      int64_t idx = (dim > 0) ? (rem % dim) : 0;
      rem = (dim > 0) ? (rem / dim) : 0;
      if (d == normAxis)
        axisIdx = idx;
    }
    return axisIdx;
  };

  for (int64_t i = 0; i < n; ++i) {
    int64_t inOff = offset_of(in, i);
    int64_t outOff = offset_of(out, i);
    int64_t axisIdx = axis_index_from_linear(i);
    (void)axisIdx; // Placeholder for future per-axis scale/zp buffers in ABI.

    double q = inputSigned ? static_cast<double>(in.data[inOff])
                           : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    if (hasZeroPoint)
      q -= static_cast<double>(zeroPoint);
    double v = q * static_cast<double>(scale);
    out.data[outOff] = from_bits<typename Fmt::MemT>(bits_from_double_fmt<Fmt>(v));
  }
}

template <typename Fmt>
static void dequantize_linear_axis_kernel(
    UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<typename Fmt::MemT> *Out,
    UnrankedMemRefType<float> *Scale, UnrankedMemRefType<int64_t> *ZeroPoint,
    int64_t hasZeroPoint, int64_t axis, int64_t inputSigned) {
  DynamicMemRefType<int8_t> in(*In);
  DynamicMemRefType<typename Fmt::MemT> out(*Out);
  DynamicMemRefType<float> scale(*Scale);
  DynamicMemRefType<int64_t> zp(*ZeroPoint);

  int64_t inRank = in.rank;
  if (inRank <= 0) {
    int64_t inOff = in.offset;
    int64_t outOff = out.offset;
    int64_t sOff = scale.offset;
    int64_t zOff = zp.offset;
    double s = static_cast<double>(scale.data[sOff]);
    double z = hasZeroPoint ? static_cast<double>(zp.data[zOff]) : 0.0;
    double q = inputSigned ? static_cast<double>(in.data[inOff])
                           : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    q -= z;
    out.data[outOff] = from_bits<typename Fmt::MemT>(bits_from_double_fmt<Fmt>(q * s));
    return;
  }

  int64_t normAxis = axis;
  if (normAxis < 0)
    normAxis += inRank;
  if (normAxis < 0)
    normAxis = 0;
  if (normAxis >= inRank)
    normAxis = inRank - 1;

  int64_t scaleElems = std::max<int64_t>(1, num_elems(scale));
  int64_t zpElems = std::max<int64_t>(1, num_elems(zp));
  int64_t inElems = num_elems(in);
  int64_t outElems = num_elems(out);
  int64_t n = std::min(inElems, outElems);

  auto axis_index_from_linear = [&](int64_t lin) -> int64_t {
    int64_t rem = lin;
    int64_t axisIdx = 0;
    for (int64_t d = inRank - 1; d >= 0; --d) {
      int64_t dim = in.sizes[d];
      int64_t idx = (dim > 0) ? (rem % dim) : 0;
      rem = (dim > 0) ? (rem / dim) : 0;
      if (d == normAxis)
        axisIdx = idx;
    }
    return axisIdx;
  };

  auto load_scale = [&](int64_t axisIdx) -> double {
    int64_t i = (scaleElems <= 1) ? 0 : std::min<int64_t>(axisIdx, scaleElems - 1);
    return static_cast<double>(scale.data[offset_of(scale, i)]);
  };

  auto load_zp = [&](int64_t axisIdx) -> double {
    if (!hasZeroPoint)
      return 0.0;
    int64_t i = (zpElems <= 1) ? 0 : std::min<int64_t>(axisIdx, zpElems - 1);
    return static_cast<double>(zp.data[offset_of(zp, i)]);
  };

  for (int64_t i = 0; i < n; ++i) {
    int64_t inOff = offset_of(in, i);
    int64_t outOff = offset_of(out, i);
    int64_t axisIdx = axis_index_from_linear(i);
    double s = load_scale(axisIdx);
    double z = load_zp(axisIdx);

    double q = inputSigned ? static_cast<double>(in.data[inOff])
                           : static_cast<double>(static_cast<uint8_t>(in.data[inOff]));
    double v = (q - z) * s;
    out.data[outOff] = from_bits<typename Fmt::MemT>(bits_from_double_fmt<Fmt>(v));
  }
}

//===----------------------------------------------------------------------===//
// Export wrappers
//===----------------------------------------------------------------------===//

#define DEFINE_POSIT_RUNTIME_EXPORTS(TAG, FMT, MEMTY)                                             \
  extern "C" void _mlir_ciface_posit_from_f32_##TAG(                                              \
      UnrankedMemRefType<float> *In, UnrankedMemRefType<MEMTY> *Out) {                            \
    posit_from_f32_kernel<FMT>(In, Out);                                                           \
    traceMemrefSummary<FMT>("from_f32_" #TAG, DynamicMemRefType<MEMTY>(*Out));                    \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_to_f32_##TAG(                                                \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<float> *Out) {                            \
    posit_to_f32_kernel<FMT>(In, Out);                                                             \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_add_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out) {                                                            \
    elemwise_kernel<FMT>(FMT::add, A, B, Out);                                                    \
    traceMemrefSummary<FMT>("add_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_sub_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out) {                                                            \
    elemwise_kernel<FMT>(FMT::sub, A, B, Out);                                                    \
    traceMemrefSummary<FMT>("sub_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_mul_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out) {                                                            \
    elemwise_kernel<FMT>(FMT::mul, A, B, Out);                                                    \
    traceMemrefSummary<FMT>("mul_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_div_##TAG(                                                   \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *Out) {                                                            \
    elemwise_kernel<FMT>(FMT::div, A, B, Out);                                                    \
    traceMemrefSummary<FMT>("div_" #TAG, DynamicMemRefType<MEMTY>(*Out));                         \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_relu_##TAG(                                                  \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<MEMTY> *Out) {                            \
    relu_kernel<FMT>(In, Out);                                                                     \
    traceMemrefSummary<FMT>("relu_" #TAG, DynamicMemRefType<MEMTY>(*Out));                        \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_gemm_##TAG(                                                  \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *C, UnrankedMemRefType<MEMTY> *Y,                                 \
      float alpha, float beta, int64_t transA, int64_t transB) {                                  \
    gemm_kernel<FMT>(A, B, C, Y, alpha, beta, transA, transB);                                    \
    traceMemrefSummary<FMT>("gemm_" #TAG, DynamicMemRefType<MEMTY>(*Y));                          \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_conv2d_nchw_##TAG(                                           \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *W,                                 \
      UnrankedMemRefType<MEMTY> *B, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t stride_h, int64_t stride_w, int64_t dilation_h, int64_t dilation_w,                 \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,                   \
      int64_t groups) {                                                                            \
    conv2d_nchw_kernel<FMT>(X, W, B, Out, stride_h, stride_w, dilation_h, dilation_w,             \
                            pad_top, pad_left, pad_bottom, pad_right, groups);                     \
    traceMemrefSummary<FMT>("conv2d_nchw_" #TAG, DynamicMemRefType<MEMTY>(*Out));                 \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_maxpool2d_nchw_##TAG(                                        \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,                     \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,                   \
      int64_t ceil_mode) {                                                                         \
    maxpool2d_nchw_kernel<FMT>(X, Out, kernel_h, kernel_w, stride_h, stride_w,                    \
                               pad_top, pad_left, pad_bottom, pad_right, ceil_mode);               \
    traceMemrefSummary<FMT>("maxpool2d_nchw_" #TAG, DynamicMemRefType<MEMTY>(*Out));              \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_clip_##TAG(                                                  \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<MEMTY> *Out, float minVal,                \
      float maxVal, int64_t hasMin, int64_t hasMax) {                                              \
    clip_kernel<FMT>(In, Out, minVal, maxVal, hasMin, hasMax);                                     \
    traceMemrefSummary<FMT>("clip_" #TAG, DynamicMemRefType<MEMTY>(*Out));                        \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_reduce_mean_##TAG(                                            \
      UnrankedMemRefType<MEMTY> *In, UnrankedMemRefType<MEMTY> *Out, int64_t axis0,               \
      int64_t axis1, int64_t keepdims) {                                                           \
    reduce_mean_kernel<FMT>(In, Out, axis0, axis1, keepdims);                                      \
    traceMemrefSummary<FMT>("reduce_mean_" #TAG, DynamicMemRefType<MEMTY>(*Out));                 \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_##TAG(                                      \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<MEMTY> *Out, float scale,                \
      int64_t zeroPoint, int64_t hasZeroPoint, int64_t axis, int64_t inputSigned) {               \
    dequantize_linear_kernel<FMT>(In, Out, scale, zeroPoint, hasZeroPoint, axis, inputSigned);    \
    traceMemrefSummary<FMT>("dequantize_linear_" #TAG, DynamicMemRefType<MEMTY>(*Out));           \
  }                                                                                                \
  extern "C" void _mlir_ciface_posit_dequantize_linear_axis_##TAG(                                 \
      UnrankedMemRefType<int8_t> *In, UnrankedMemRefType<MEMTY> *Out,                              \
      UnrankedMemRefType<float> *Scale, UnrankedMemRefType<int64_t> *ZeroPoint,                    \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned) {                                   \
    dequantize_linear_axis_kernel<FMT>(In, Out, Scale, ZeroPoint, hasZeroPoint, axis, inputSigned);\
    traceMemrefSummary<FMT>("dequantize_linear_axis_" #TAG, DynamicMemRefType<MEMTY>(*Out));      \
  }

DEFINE_POSIT_RUNTIME_EXPORTS(p8e0, FmtP8E0, int8_t)
#if defined(POSIT_USE_UNIVERSAL)
DEFINE_POSIT_RUNTIME_EXPORTS(p4e0, FmtP4E0, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p4e1, FmtP4E1, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p4e2, FmtP4E2, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p4e3, FmtP4E3, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p5e0, FmtP5E0, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p5e1, FmtP5E1, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p5e2, FmtP5E2, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p5e3, FmtP5E3, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p6e0, FmtP6E0, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p6e1, FmtP6E1, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p6e2, FmtP6E2, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p6e3, FmtP6E3, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p7e0, FmtP7E0, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p7e1, FmtP7E1, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p7e2, FmtP7E2, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p7e3, FmtP7E3, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p8e1, FmtP8E1, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p8e2, FmtP8E2, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p9e0, FmtP9E0, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p9e1, FmtP9E1, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p9e2, FmtP9E2, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p9e3, FmtP9E3, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p16e0, FmtP16E0, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p16e1, FmtP16E1, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p16e2, FmtP16E2, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p32e0, FmtP32E0, int32_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p32e1, FmtP32E1, int32_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p32e2, FmtP32E2, int32_t)
#elif defined(POSIT_USE_SOFTPOSIT_PX1) || defined(POSIT_USE_SOFTPOSIT_PX2)
#if defined(POSIT_USE_SOFTPOSIT_PX1)
DEFINE_POSIT_RUNTIME_EXPORTS(p8e1, FmtP8E1ViaPX1, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p32e1, FmtP32E1ViaPX1, int32_t)
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
DEFINE_POSIT_RUNTIME_EXPORTS(p8e2, FmtP8E2ViaPX2, int8_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p16e2, FmtP16E2ViaPX2, int16_t)
#endif
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
DEFINE_POSIT_RUNTIME_EXPORTS(p16e0, FmtP16E0, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p32e0, FmtP32E0, int32_t)
#endif
DEFINE_POSIT_RUNTIME_EXPORTS(p16e1, FmtP16E1, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p32e2, FmtP32E2, int32_t)
#else
DEFINE_POSIT_RUNTIME_EXPORTS(p16e1, FmtP16E1, int16_t)
DEFINE_POSIT_RUNTIME_EXPORTS(p32e2, FmtP32E2, int32_t)
#endif

#define DEFINE_POSIT_CIFACE_FORWARDERS(TAG, MEMTY)                                                 \
  extern "C" {                                                                                     \
  void _mlir_ciface__mlir_ciface_posit_from_f32_##TAG(UnrankedMemRefType<float> *in,              \
                                                      UnrankedMemRefType<MEMTY> *out) {            \
    _mlir_ciface_posit_from_f32_##TAG(in, out);                                                    \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_to_f32_##TAG(UnrankedMemRefType<MEMTY> *in,                \
                                                    UnrankedMemRefType<float> *out) {              \
    _mlir_ciface_posit_to_f32_##TAG(in, out);                                                      \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_add_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_add_##TAG(a, b, out);                                                       \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_sub_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_sub_##TAG(a, b, out);                                                       \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_mul_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_mul_##TAG(a, b, out);                                                       \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_div_##TAG(UnrankedMemRefType<MEMTY> *a,                    \
                                                 UnrankedMemRefType<MEMTY> *b,                     \
                                                 UnrankedMemRefType<MEMTY> *out) {                 \
    _mlir_ciface_posit_div_##TAG(a, b, out);                                                       \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_relu_##TAG(UnrankedMemRefType<MEMTY> *in,                  \
                                                  UnrankedMemRefType<MEMTY> *out) {                \
    _mlir_ciface_posit_relu_##TAG(in, out);                                                        \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_conv2d_nchw_##TAG(                                          \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *W,                                 \
      UnrankedMemRefType<MEMTY> *B, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t stride_h, int64_t stride_w, int64_t dilat_h, int64_t dilat_w,                       \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right, int64_t group) { \
    _mlir_ciface_posit_conv2d_nchw_##TAG(X, W, B, Out, stride_h, stride_w, dilat_h, dilat_w,      \
                                         pad_top, pad_left, pad_bottom, pad_right, group);         \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_maxpool2d_nchw_##TAG(                                       \
      UnrankedMemRefType<MEMTY> *X, UnrankedMemRefType<MEMTY> *Out,                               \
      int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,                     \
      int64_t pad_top, int64_t pad_left, int64_t pad_bottom, int64_t pad_right,                   \
      int64_t ceil_mode) {                                                                         \
    _mlir_ciface_posit_maxpool2d_nchw_##TAG(X, Out, kernel_h, kernel_w, stride_h, stride_w,       \
                                            pad_top, pad_left, pad_bottom, pad_right, ceil_mode);  \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_gemm_##TAG(                                                 \
      UnrankedMemRefType<MEMTY> *A, UnrankedMemRefType<MEMTY> *B,                                 \
      UnrankedMemRefType<MEMTY> *C, UnrankedMemRefType<MEMTY> *Y,                                 \
      float alpha, float beta, int64_t transA, int64_t transB) {                                  \
    _mlir_ciface_posit_gemm_##TAG(A, B, C, Y, alpha, beta, transA, transB);                       \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_clip_##TAG(                                                 \
      UnrankedMemRefType<MEMTY> *in, UnrankedMemRefType<MEMTY> *out,                              \
      float minVal, float maxVal, int64_t hasMin, int64_t hasMax) {                               \
    _mlir_ciface_posit_clip_##TAG(in, out, minVal, maxVal, hasMin, hasMax);                       \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_reduce_mean_##TAG(                                          \
      UnrankedMemRefType<MEMTY> *in, UnrankedMemRefType<MEMTY> *out,                              \
      int64_t axis0, int64_t axis1, int64_t keepdims) {                                            \
    _mlir_ciface_posit_reduce_mean_##TAG(in, out, axis0, axis1, keepdims);                        \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_##TAG(                                    \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<MEMTY> *out, float scale,                \
      int64_t zeroPoint, int64_t hasZeroPoint, int64_t axis, int64_t inputSigned) {               \
    _mlir_ciface_posit_dequantize_linear_##TAG(                                                    \
        in, out, scale, zeroPoint, hasZeroPoint, axis, inputSigned);                              \
  }                                                                                                \
  void _mlir_ciface__mlir_ciface_posit_dequantize_linear_axis_##TAG(                               \
      UnrankedMemRefType<int8_t> *in, UnrankedMemRefType<MEMTY> *out,                              \
      UnrankedMemRefType<float> *scale, UnrankedMemRefType<int64_t> *zeroPoint,                    \
      int64_t hasZeroPoint, int64_t axis, int64_t inputSigned) {                                   \
    _mlir_ciface_posit_dequantize_linear_axis_##TAG(                                               \
        in, out, scale, zeroPoint, hasZeroPoint, axis, inputSigned);                              \
  }                                                                                                \
  }

DEFINE_POSIT_CIFACE_FORWARDERS(p8e0, int8_t)
#if defined(POSIT_USE_UNIVERSAL) || defined(POSIT_USE_SOFTPOSIT_PX1)
DEFINE_POSIT_CIFACE_FORWARDERS(p8e1, int8_t)
#endif
#if defined(POSIT_USE_UNIVERSAL) || defined(POSIT_USE_SOFTPOSIT_PX2)
DEFINE_POSIT_CIFACE_FORWARDERS(p8e2, int8_t)
#endif
#if defined(POSIT_USE_UNIVERSAL)
DEFINE_POSIT_CIFACE_FORWARDERS(p4e0, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p4e1, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p4e2, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p4e3, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p5e0, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p5e1, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p5e2, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p5e3, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p6e0, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p6e1, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p6e2, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p6e3, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p7e0, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p7e1, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p7e2, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p7e3, int8_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p9e0, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p9e1, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p9e2, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p9e3, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p16e0, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p16e1, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p16e2, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p32e0, int32_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p32e1, int32_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p32e2, int32_t)
#else
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
DEFINE_POSIT_CIFACE_FORWARDERS(p16e0, int16_t)
DEFINE_POSIT_CIFACE_FORWARDERS(p32e0, int32_t)
#endif
DEFINE_POSIT_CIFACE_FORWARDERS(p16e1, int16_t)
#if defined(POSIT_USE_SOFTPOSIT_PX2)
DEFINE_POSIT_CIFACE_FORWARDERS(p16e2, int16_t)
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX1)
DEFINE_POSIT_CIFACE_FORWARDERS(p32e1, int32_t)
#endif
DEFINE_POSIT_CIFACE_FORWARDERS(p32e2, int32_t)
#endif

#if !defined(POSIT_USE_UNIVERSAL) && !defined(POSIT_USE_SOFTPOSIT_PX1) && !defined(POSIT_USE_SOFTPOSIT_PX2)
// [EXTEND] SoftPosit default API does not provide p8e1/p8e2 directly. To satisfy .ll
// files that declare _mlir_ciface__mlir_ciface_posit_*_p8e1/p8e2, rebuild with one of:
//   -DPOSIT_USE_UNIVERSAL
//   -DPOSIT_USE_SOFTPOSIT_PX1  (requires libsoftposit with pX1_* symbols)
//   -DPOSIT_USE_SOFTPOSIT_PX2  (requires libsoftposit with pX2_* / qX2_* symbols)
#endif
