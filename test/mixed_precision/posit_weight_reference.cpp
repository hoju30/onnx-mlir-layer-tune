// SPDX-License-Identifier: Apache-2.0
// Generate an independent Universal oracle for compiler-folded posit weights.
// This helper never includes the mixed-precision pass or its LUT implementation.
#include <universal/number/posit/posit.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static float fromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

static void writeFloats(const char *path, const std::vector<float> &values) {
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char *>(values.data()),
      values.size() * sizeof(float));
  if (!output)
    throw std::runtime_error(std::string("cannot write ") + path);
}

template <unsigned N>
static void generate(const char *inputPath, const char *expectedPath) {
  using Posit = sw::universal::posit<N, 1>;
  std::vector<float> inputs, decoded;
  auto appendNeighbors = [&](float value) {
    inputs.push_back(std::nextafter(value, -INFINITY));
    inputs.push_back(value);
    inputs.push_back(std::nextafter(value, INFINITY));
  };

  // Every encoded value, including NaR, plus both f32 neighbors of each
  // finite decoded value. This covers both signs and all regime boundaries.
  for (unsigned bits = 0; bits < (1u << N); ++bits) {
    Posit p;
    p.setbits(bits);
    float value = static_cast<float>(p);
    inputs.push_back(value);
    if (std::isfinite(value)) {
      decoded.push_back(value);
      inputs.push_back(std::nextafter(value, -INFINITY));
      inputs.push_back(std::nextafter(value, INFINITY));
    }
  }
  std::sort(decoded.begin(), decoded.end());
  const size_t encodedCount = inputs.size();

  // Arithmetic midpoints are useful adversarial inputs even though posit
  // conversion is not simply nearest-value rounding. Universal computes
  // the answer independently for every midpoint and adjacent f32 value.
  for (size_t i = 1; i < decoded.size(); ++i)
    appendNeighbors(static_cast<float>(
        (static_cast<double>(decoded[i - 1]) + decoded[i]) / 2.0));
  const size_t midpointCount = inputs.size() - encodedCount;

  for (uint32_t bits : {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu,
           0x807fffffu, 0x00800000u, 0x80800000u, 0x7f7fffffu, 0xff7fffffu,
           0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00000u, 0x7fc12345u,
           0xffc12345u, 0x7f800001u, 0xff800001u})
    inputs.push_back(fromBits(bits));
  const float maxpos = decoded.back();
  const float minpos = *std::upper_bound(decoded.begin(), decoded.end(), 0.0f);
  for (float endpoint : {minpos, maxpos})
    for (float factor : {0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f})
      for (float sign : {-1.0f, 1.0f})
        appendNeighbors(sign * endpoint * factor);
  const size_t boundaryCount = inputs.size() - encodedCount - midpointCount;

  // mt19937's integer sequence is standardized. Avoid uniform distributions,
  // whose mapping can vary between standard-library implementations.
  constexpr size_t randomCount = 65536;
  std::mt19937 random(0x504f5349u + N);
  for (size_t i = 0; i < randomCount; ++i)
    inputs.push_back(fromBits(random()));
  for (size_t i = 0; i < randomCount; ++i) {
    double unit = static_cast<double>(random()) / 4294967295.0;
    inputs.push_back(static_cast<float>((2.0 * unit - 1.0) * maxpos));
  }

  std::vector<float> expected;
  expected.reserve(inputs.size());
  for (float input : inputs) {
    Posit p;
    p = input;
    expected.push_back(static_cast<float>(p));
  }
  writeFloats(inputPath, inputs);
  writeFloats(expectedPath, expected);
  std::cout << "{\"encoded_and_neighbors\":" << encodedCount
            << ",\"midpoint_and_neighbors\":" << midpointCount
            << ",\"boundary\":" << boundaryCount
            << ",\"random_raw_bits\":" << randomCount
            << ",\"random_in_range\":" << randomCount
            << ",\"total\":" << inputs.size() << "}\n";
}

int main(int argc, char **argv) {
  static_assert(sizeof(float) == sizeof(uint32_t));
  static_assert(std::numeric_limits<float>::is_iec559);
  if (argc != 4) {
    std::cerr << "usage: posit_weight_reference {4|6|8} input.bin expected.bin\n";
    return 2;
  }
  try {
    const std::string nbits(argv[1]);
    if (nbits == "4")
      generate<4>(argv[2], argv[3]);
    else if (nbits == "6")
      generate<6>(argv[2], argv[3]);
    else if (nbits == "8")
      generate<8>(argv[2], argv[3]);
    else
      throw std::runtime_error("unsupported posit format");
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
