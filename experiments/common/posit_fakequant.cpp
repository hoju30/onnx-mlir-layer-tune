// Standalone FP32 <-> posit<N,ES> round-trip (fake-quant) tool for the ML
// feature pipeline. Deliberately NOT wired into the MLIR/memref calling
// convention that src/posit_runtime.cpp's _mlir_ciface_posit_from_f32_*
// exports use (UnrankedMemRefType is painful to construct from Python) --
// this reuses the exact same underlying type, sw::universal::posit<N,ES>
// from src/.deps/universal, via a plain flat-array C ABI so it's trivially
// callable via Python ctypes and stays bit-exact with the real runtime.
//
// Covers the candidate formats in claude.md section 1: posit_{8,16,32}_{0,1,2}.
#include <cstdint>
#include <cstring>
#include <type_traits>

#include <universal/number/posit/posit.hpp>

using sw::universal::posit;

template <int N, int ES>
static void roundtrip(const float *in, float *out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    posit<N, ES> p;
    p = in[i];
    out[i] = static_cast<float>(p);
  }
}

// Decode raw posit<N,ES> bit patterns (as already stored in a posit.constant's
// dense<...> attribute -- iN two's-complement values, little-endian byte
// order matching how MLIR dumps dense hex constants) back to FP32. This is
// the reverse of roundtrip(): roundtrip() starts from a known FP32 value and
// simulates quantizing it; this starts from already-encoded bits with no FP32
// origin available, using the same posit<N,ES>.setbits()+cast the runtime's
// own _mlir_ciface_posit_to_f32_* wrappers use (see UniversalFmt::fromRaw /
// ::toDouble in src/posit_runtime.cpp), so it stays bit-exact.
template <int N, int ES, typename StorageT>
static void decode_bits(const uint8_t *raw, float *out, int64_t n) {
  using UStorageT = typename std::make_unsigned<StorageT>::type;
  for (int64_t i = 0; i < n; ++i) {
    StorageT bits;
    std::memcpy(&bits, raw + i * sizeof(StorageT), sizeof(StorageT));
    posit<N, ES> p;
    p.setbits(static_cast<uint64_t>(static_cast<UStorageT>(bits)));
    out[i] = static_cast<float>(p);
  }
}

extern "C" int posit_roundtrip_f32(
    const float *in, float *out, int64_t n, int nbits, int es) {
  switch (nbits) {
    case 8:
      switch (es) {
        case 0: roundtrip<8, 0>(in, out, n); return 0;
        case 1: roundtrip<8, 1>(in, out, n); return 0;
        case 2: roundtrip<8, 2>(in, out, n); return 0;
      }
      break;
    case 16:
      switch (es) {
        case 0: roundtrip<16, 0>(in, out, n); return 0;
        case 1: roundtrip<16, 1>(in, out, n); return 0;
        case 2: roundtrip<16, 2>(in, out, n); return 0;
      }
      break;
    case 32:
      switch (es) {
        case 0: roundtrip<32, 0>(in, out, n); return 0;
        case 1: roundtrip<32, 1>(in, out, n); return 0;
        case 2: roundtrip<32, 2>(in, out, n); return 0;
      }
      break;
  }
  return -1;  // unsupported (nbits, es)
}

extern "C" int posit_decode_bits_f32(
    const uint8_t *raw, float *out, int64_t n, int nbits, int es) {
  switch (nbits) {
    case 8:
      switch (es) {
        case 0: decode_bits<8, 0, int8_t>(raw, out, n); return 0;
        case 1: decode_bits<8, 1, int8_t>(raw, out, n); return 0;
        case 2: decode_bits<8, 2, int8_t>(raw, out, n); return 0;
      }
      break;
    case 16:
      switch (es) {
        case 0: decode_bits<16, 0, int16_t>(raw, out, n); return 0;
        case 1: decode_bits<16, 1, int16_t>(raw, out, n); return 0;
        case 2: decode_bits<16, 2, int16_t>(raw, out, n); return 0;
      }
      break;
    case 32:
      switch (es) {
        case 0: decode_bits<32, 0, int32_t>(raw, out, n); return 0;
        case 1: decode_bits<32, 1, int32_t>(raw, out, n); return 0;
        case 2: decode_bits<32, 2, int32_t>(raw, out, n); return 0;
      }
      break;
  }
  return -1;  // unsupported (nbits, es)
}
