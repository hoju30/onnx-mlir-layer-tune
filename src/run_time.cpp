#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <unistd.h>

#include "mlir/ExecutionEngine/CRunnerUtils.h"

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

using InDesc = StridedMemRefType<float, 4>;
template <typename T>
using OutDescT = StridedMemRefType<T, 2>;
template <typename T>
using MainFnT = void (*)(OutDescT<T> *, InDesc *);

enum class OutType {
  P4E0,
  P4E1,
  P4E2,
  P4E3,
  P5E0,
  P5E1,
  P5E2,
  P5E3,
  P6E0,
  P6E1,
  P6E2,
  P6E3,
  P7E0,
  P7E1,
  P7E2,
  P7E3,
  P8E0,
  P8E1,
  P8E2,
  P9E0,
  P9E1,
  P9E2,
  P9E3,
  P16E0,
  P16E1,
  P16E2,
  P32E0,
  P32E1,
  P32E2,
  F32,
};

struct RefSpec {
  std::string soPath;
  OutType outType = OutType::F32;
  std::string entrySymbol;
};

struct RunStats {
  double avg_us = 0.0;
  double p50_us = 0.0;
  double p95_us = 0.0;
  double min_us = 0.0;
  double max_us = 0.0;
};

struct CompareStats {
  int n = 0;
  double mae = 0.0;
  double rmse = 0.0;
  double max_abs = 0.0;
  double cosine = 0.0;
  double mean_rel_abs = 0.0;
  double js_div = 0.0;
  int top1_target = -1;
  int top1_base = -1;
  bool top1_match = false;
  int top5_overlap = 0;
  bool top5_exact = false;
};

static bool isFlag(const std::string &s) { return s.rfind("--", 0) == 0; }

static int64_t parseI64(const std::string &s, const char *what) {
  try {
    size_t pos = 0;
    long long v = std::stoll(s, &pos, 10);
    if (pos != s.size())
      throw std::runtime_error("trailing");
    return static_cast<int64_t>(v);
  } catch (...) {
    std::cerr << "Invalid " << what << ": " << s << "\n";
    std::exit(1);
  }
}

static bool parseOnOff(const std::string &s, bool &out) {
  std::string v = s;
  std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  if (v == "1" || v == "on" || v == "true" || v == "yes") {
    out = true;
    return true;
  }
  if (v == "0" || v == "off" || v == "false" || v == "no") {
    out = false;
    return true;
  }
  return false;
}

static std::string resolveSoPathForDlopen(const std::string &path) {
  if (path.find('/') != std::string::npos)
    return path;
  std::ifstream f(path);
  if (f.good())
    return std::string("./") + path;
  return path;
}

static std::string basenameOnly(const std::string &path) {
  size_t pos = path.find_last_of("/\\");
  return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

static std::string stripSharedObjectSuffix(std::string s) {
  if (s.size() >= 3 && s.substr(s.size() - 3) == ".so")
    s.resize(s.size() - 3);
  return s;
}

static void appendUnique(std::vector<std::string> &dst, const std::string &s) {
  if (s.empty())
    return;
  if (std::find(dst.begin(), dst.end(), s) == dst.end())
    dst.push_back(s);
}

static std::vector<std::string> deriveEntrypointStemCandidates(
    const std::string &soPath) {
  std::vector<std::string> stems;
  std::string base = stripSharedObjectSuffix(basenameOnly(soPath));
  appendUnique(stems, base);
  for (const std::string &marker : {"-nqdq-", "-qdq-"}) {
    size_t pos = base.find(marker);
    if (pos != std::string::npos)
      appendUnique(stems, base.substr(0, pos));
  }
  return stems;
}

static std::vector<std::string> entrypointCandidates(const std::string &soPath,
    const std::string &preferred, bool explicitPreferred) {
  std::vector<std::string> out;
  appendUnique(out, preferred);
  if (explicitPreferred)
    return out;

  std::vector<std::string> stems = deriveEntrypointStemCandidates(soPath);

  appendUnique(out, "_mlir_ciface_main_graph");
  for (const std::string &stem : stems)
    appendUnique(out, "_mlir_ciface_main_graph_" + stem);
  appendUnique(out, "run_main_graph");
  for (const std::string &stem : stems)
    appendUnique(out, "run_main_graph_" + stem);
  appendUnique(out, "main_graph");
  for (const std::string &stem : stems)
    appendUnique(out, "main_graph_" + stem);
  return out;
}

static std::vector<std::string> queryEntrypointCandidates(void *handle) {
  std::vector<std::string> out;
  using QueryEntryPointsFn = const char **(*)(int64_t *);

  dlerror();
  auto *queryFn = reinterpret_cast<QueryEntryPointsFn>(
      dlsym(handle, "omQueryEntryPoints"));
  const char *err = dlerror();
  if (err || !queryFn)
    return out;

  int64_t count = 0;
  const char **entries = queryFn(&count);
  if (!entries)
    return out;

  std::vector<std::string> rawNames;
  auto appendDerived = [&](const std::string &name) {
    static const std::string runPrefix = "run_main_graph";
    if (name.rfind(runPrefix, 0) == 0)
      appendUnique(out, "_mlir_ciface_main_graph" + name.substr(runPrefix.size()));
    static const std::string mainPrefix = "main_graph";
    if (name.rfind(mainPrefix, 0) == 0)
      appendUnique(out, "_mlir_ciface_main_graph" + name.substr(mainPrefix.size()));
    appendUnique(rawNames, name);
  };

  if (count > 0) {
    for (int64_t i = 0; i < count; ++i) {
      if (!entries[i])
        break;
      appendDerived(entries[i]);
    }
  } else {
    for (const char **p = entries; p && *p; ++p)
      appendDerived(*p);
  }
  for (const std::string &name : rawNames)
    appendUnique(out, name);
  return out;
}

static void *resolveEntrypointSymbol(
    void *handle, const std::string &soPath, const std::string &preferred,
    bool explicitPreferred, std::string &resolvedEntry,
    std::string &attemptedEntries) {
  std::vector<std::string> candidates;
  if (explicitPreferred) {
    candidates = entrypointCandidates(soPath, preferred, explicitPreferred);
  } else {
    appendUnique(candidates, preferred);
    appendUnique(candidates, "_mlir_ciface_main_graph");
    for (const std::string &stem : deriveEntrypointStemCandidates(soPath))
      appendUnique(candidates, "_mlir_ciface_main_graph_" + stem);
    for (const std::string &q : queryEntrypointCandidates(handle))
      appendUnique(candidates, q);
    for (const std::string &fallback :
         entrypointCandidates(soPath, preferred, explicitPreferred))
      appendUnique(candidates, fallback);
  }
  attemptedEntries.clear();
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (i)
      attemptedEntries += ", ";
    attemptedEntries += candidates[i];
    dlerror();
    void *sym = dlsym(handle, candidates[i].c_str());
    const char *err = dlerror();
    if (!err && sym) {
      resolvedEntry = candidates[i];
      return sym;
    }
  }
  resolvedEntry.clear();
  return nullptr;
}

static bool tryParseOutType(const std::string &s, OutType &out) {
  if (s == "p4e0")
    out = OutType::P4E0;
  else if (s == "p4e1")
    out = OutType::P4E1;
  else if (s == "p4e2")
    out = OutType::P4E2;
  else if (s == "p4e3")
    out = OutType::P4E3;
  else if (s == "p5e0")
    out = OutType::P5E0;
  else if (s == "p5e1")
    out = OutType::P5E1;
  else if (s == "p5e2")
    out = OutType::P5E2;
  else if (s == "p5e3")
    out = OutType::P5E3;
  else if (s == "p6e0")
    out = OutType::P6E0;
  else if (s == "p6e1")
    out = OutType::P6E1;
  else if (s == "p6e2")
    out = OutType::P6E2;
  else if (s == "p6e3")
    out = OutType::P6E3;
  else if (s == "p7e0")
    out = OutType::P7E0;
  else if (s == "p7e1")
    out = OutType::P7E1;
  else if (s == "p7e2")
    out = OutType::P7E2;
  else if (s == "p7e3")
    out = OutType::P7E3;
  else if (s == "p8e0")
    out = OutType::P8E0;
  else if (s == "p8e1")
    out = OutType::P8E1;
  else if (s == "p8e2")
    out = OutType::P8E2;
  else if (s == "p9e0")
    out = OutType::P9E0;
  else if (s == "p9e1")
    out = OutType::P9E1;
  else if (s == "p9e2")
    out = OutType::P9E2;
  else if (s == "p9e3")
    out = OutType::P9E3;
  else if (s == "p16e0")
    out = OutType::P16E0;
  else if (s == "p16e1")
    out = OutType::P16E1;
  else if (s == "p16e2")
    out = OutType::P16E2;
  else if (s == "p32e0")
    out = OutType::P32E0;
  else if (s == "p32e1")
    out = OutType::P32E1;
  else if (s == "p32e2")
    out = OutType::P32E2;
  else if (s == "f32")
    out = OutType::F32;
  else
    return false;
  return true;
}

static OutType parseOutType(const std::string &s) {
  OutType out = OutType::F32;
  if (tryParseOutType(s, out))
    return out;
  std::cerr << "Unsupported output type: " << s
            << " (supported: p4e0..p7e3, p8e0..p8e2, p9e0..p9e3, "
            << "p16e0..p16e2, p32e0..p32e2, f32)\n";
  std::exit(1);
}

static const char *outTypeToString(OutType t) {
  switch (t) {
  case OutType::P4E0:
    return "p4e0";
  case OutType::P4E1:
    return "p4e1";
  case OutType::P4E2:
    return "p4e2";
  case OutType::P4E3:
    return "p4e3";
  case OutType::P5E0:
    return "p5e0";
  case OutType::P5E1:
    return "p5e1";
  case OutType::P5E2:
    return "p5e2";
  case OutType::P5E3:
    return "p5e3";
  case OutType::P6E0:
    return "p6e0";
  case OutType::P6E1:
    return "p6e1";
  case OutType::P6E2:
    return "p6e2";
  case OutType::P6E3:
    return "p6e3";
  case OutType::P7E0:
    return "p7e0";
  case OutType::P7E1:
    return "p7e1";
  case OutType::P7E2:
    return "p7e2";
  case OutType::P7E3:
    return "p7e3";
  case OutType::P8E0:
    return "p8e0";
  case OutType::P8E1:
    return "p8e1";
  case OutType::P8E2:
    return "p8e2";
  case OutType::P9E0:
    return "p9e0";
  case OutType::P9E1:
    return "p9e1";
  case OutType::P9E2:
    return "p9e2";
  case OutType::P9E3:
    return "p9e3";
  case OutType::P16E0:
    return "p16e0";
  case OutType::P16E1:
    return "p16e1";
  case OutType::P16E2:
    return "p16e2";
  case OutType::P32E0:
    return "p32e0";
  case OutType::P32E1:
    return "p32e1";
  case OutType::P32E2:
    return "p32e2";
  case OutType::F32:
    return "f32";
  }
  return "unknown";
}

static RefSpec parseCmpSpec(const std::string &s) {
  size_t last = s.rfind(':');
  if (last == std::string::npos || last == 0 || last + 1 >= s.size()) {
    std::cerr << "Invalid --cmp format: " << s << " (expect <path>:<type>[:<entry>])\n";
    std::exit(1);
  }

  RefSpec r;
  std::string maybeType = s.substr(last + 1);
  OutType ot = OutType::F32;
  if (tryParseOutType(maybeType, ot)) {
    r.soPath = s.substr(0, last);
    r.outType = ot;
    return r;
  }

  size_t mid = s.rfind(':', last - 1);
  if (mid == std::string::npos || mid == 0 || mid + 1 >= last) {
    std::cerr << "Invalid --cmp format: " << s << " (expect <path>:<type>[:<entry>])\n";
    std::exit(1);
  }

  std::string typeStr = s.substr(mid + 1, last - mid - 1);
  if (!tryParseOutType(typeStr, ot)) {
    std::cerr << "Invalid --cmp type in: " << s << "\n";
    std::exit(1);
  }
  r.soPath = s.substr(0, mid);
  r.outType = ot;
  r.entrySymbol = s.substr(last + 1);
  return r;
}

static bool parseShape4(const std::string &s, int64_t shape[4]) {
  std::vector<int64_t> vals;
  size_t start = 0;
  while (start < s.size()) {
    size_t end = s.find('x', start);
    if (end == std::string::npos)
      end = s.size();
    if (end == start)
      return false;
    int64_t v = parseI64(s.substr(start, end - start), "shape dim");
    if (v <= 0)
      return false;
    vals.push_back(v);
    start = end + 1;
  }
  if (vals.size() != 4)
    return false;
  for (int i = 0; i < 4; ++i)
    shape[i] = vals[static_cast<size_t>(i)];
  return true;
}

static int64_t numInputElems(const int64_t shape[4]) {
  return shape[0] * shape[1] * shape[2] * shape[3];
}

static bool readTxtFloats(const std::string &path, std::vector<float> &out) {
  std::ifstream fin(path);
  if (!fin.is_open())
    return false;
  out.clear();
  float v;
  while (fin >> v)
    out.push_back(v);
  return !out.empty();
}

static std::string shellQuote(const std::string &s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  out += "'";
  return out;
}

static std::string shapeToString(const int64_t shape[4]) {
  std::ostringstream os;
  os << shape[0] << "x" << shape[1] << "x" << shape[2] << "x" << shape[3];
  return os.str();
}

static bool preprocessImageToFloats(const std::string &imagePath,
    const std::string &scriptPath, const int64_t shape[4], int64_t resizeShort,
    int64_t cropSize, std::vector<float> &out) {
  if (scriptPath.empty()) {
    std::cerr << "--image requires --image-preprocess-script or "
                 "POSIT_IMAGE_PREPROCESS_SCRIPT\n";
    return false;
  }

  char tmpTemplate[] = "/tmp/posit_image_tensor_XXXXXX";
  int fd = mkstemp(tmpTemplate);
  if (fd < 0) {
    std::cerr << "failed to create temporary image tensor file\n";
    return false;
  }
  close(fd);
  std::string tmpPath = tmpTemplate;

  std::ostringstream cmd;
  cmd << "python3 " << shellQuote(scriptPath) << " --image "
      << shellQuote(imagePath) << " --output " << shellQuote(tmpPath)
      << " --shape " << shellQuote(shapeToString(shape)) << " --resize-short "
      << resizeShort << " --crop-size " << cropSize;
  int rc = std::system(cmd.str().c_str());
  if (rc != 0) {
    std::cerr << "image preprocess failed rc=" << rc << ": " << imagePath
              << "\n";
    unlink(tmpPath.c_str());
    return false;
  }

  bool ok = readTxtFloats(tmpPath, out);
  unlink(tmpPath.c_str());
  return ok;
}

static bool writeLogitsCsv(const std::string &path, const std::vector<double> &logits) {
  std::ofstream fout(path);
  if (!fout.is_open())
    return false;
  fout << "label,logit\n";
  fout << std::setprecision(17);
  for (size_t i = 0; i < logits.size(); ++i)
    fout << i << "," << logits[i] << "\n";
  return true;
}

static void makeRandom(std::vector<float> &x, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (float &v : x)
    v = dist(rng);
}

static void fillInputDesc(InDesc &in, std::vector<float> &buf, const int64_t shape[4]) {
  in.basePtr = buf.data();
  in.data = buf.data();
  in.offset = 0;
  for (int i = 0; i < 4; ++i)
    in.sizes[i] = shape[i];
  in.strides[3] = 1;
  in.strides[2] = shape[3];
  in.strides[1] = shape[2] * shape[3];
  in.strides[0] = shape[1] * shape[2] * shape[3];
}

template <typename T>
using DecodeFn = double (*)(T);

#if defined(POSIT_USE_UNIVERSAL) || defined(POSIT_USE_UNIVERSAL_FALLBACK)
template <int NBits, int ES, typename MemT, typename UIntT>
static double universal_bits_to_double(MemT bits) {
  using P = sw::universal::posit<NBits, ES>;
  P p;
  p.setbits(static_cast<uint64_t>(static_cast<UIntT>(bits)));
  return static_cast<double>(p);
}
#endif

#if defined(POSIT_USE_UNIVERSAL)
static double p4e0_bits_to_double(int8_t bits) {
  return universal_bits_to_double<4, 0, int8_t, uint8_t>(bits);
}
static double p4e1_bits_to_double(int8_t bits) {
  return universal_bits_to_double<4, 1, int8_t, uint8_t>(bits);
}
static double p4e2_bits_to_double(int8_t bits) {
  return universal_bits_to_double<4, 2, int8_t, uint8_t>(bits);
}
static double p4e3_bits_to_double(int8_t bits) {
  return universal_bits_to_double<4, 3, int8_t, uint8_t>(bits);
}
static double p5e0_bits_to_double(int8_t bits) {
  return universal_bits_to_double<5, 0, int8_t, uint8_t>(bits);
}
static double p5e1_bits_to_double(int8_t bits) {
  return universal_bits_to_double<5, 1, int8_t, uint8_t>(bits);
}
static double p5e2_bits_to_double(int8_t bits) {
  return universal_bits_to_double<5, 2, int8_t, uint8_t>(bits);
}
static double p5e3_bits_to_double(int8_t bits) {
  return universal_bits_to_double<5, 3, int8_t, uint8_t>(bits);
}
static double p6e0_bits_to_double(int8_t bits) {
  return universal_bits_to_double<6, 0, int8_t, uint8_t>(bits);
}
static double p6e1_bits_to_double(int8_t bits) {
  return universal_bits_to_double<6, 1, int8_t, uint8_t>(bits);
}
static double p6e2_bits_to_double(int8_t bits) {
  return universal_bits_to_double<6, 2, int8_t, uint8_t>(bits);
}
static double p6e3_bits_to_double(int8_t bits) {
  return universal_bits_to_double<6, 3, int8_t, uint8_t>(bits);
}
static double p7e0_bits_to_double(int8_t bits) {
  return universal_bits_to_double<7, 0, int8_t, uint8_t>(bits);
}
static double p7e1_bits_to_double(int8_t bits) {
  return universal_bits_to_double<7, 1, int8_t, uint8_t>(bits);
}
static double p7e2_bits_to_double(int8_t bits) {
  return universal_bits_to_double<7, 2, int8_t, uint8_t>(bits);
}
static double p7e3_bits_to_double(int8_t bits) {
  return universal_bits_to_double<7, 3, int8_t, uint8_t>(bits);
}
static double p8e0_bits_to_double(int8_t bits) {
  return universal_bits_to_double<8, 0, int8_t, uint8_t>(bits);
}
static double p8e1_bits_to_double(int8_t bits) {
  return universal_bits_to_double<8, 1, int8_t, uint8_t>(bits);
}
static double p8e2_bits_to_double(int8_t bits) {
  return universal_bits_to_double<8, 2, int8_t, uint8_t>(bits);
}
static double p9e0_bits_to_double(int16_t bits) {
  return universal_bits_to_double<9, 0, int16_t, uint16_t>(bits);
}
static double p9e1_bits_to_double(int16_t bits) {
  return universal_bits_to_double<9, 1, int16_t, uint16_t>(bits);
}
static double p9e2_bits_to_double(int16_t bits) {
  return universal_bits_to_double<9, 2, int16_t, uint16_t>(bits);
}
static double p9e3_bits_to_double(int16_t bits) {
  return universal_bits_to_double<9, 3, int16_t, uint16_t>(bits);
}
static double p16e0_bits_to_double(int16_t bits) {
  return universal_bits_to_double<16, 0, int16_t, uint16_t>(bits);
}
static double p16e1_bits_to_double(int16_t bits) {
  return universal_bits_to_double<16, 1, int16_t, uint16_t>(bits);
}
static double p16e2_bits_to_double(int16_t bits) {
  return universal_bits_to_double<16, 2, int16_t, uint16_t>(bits);
}
static double p32e0_bits_to_double(int32_t bits) {
  return universal_bits_to_double<32, 0, int32_t, uint32_t>(bits);
}
static double p32e1_bits_to_double(int32_t bits) {
  return universal_bits_to_double<32, 1, int32_t, uint32_t>(bits);
}
static double p32e2_bits_to_double(int32_t bits) {
  return universal_bits_to_double<32, 2, int32_t, uint32_t>(bits);
}
#else
static double unsupported_extra_posit_bits_to_double() {
  std::cerr << "p4..p7/p9 require POSIT_USE_UNIVERSAL\n";
  std::exit(1);
}
static double p4e0_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p4e1_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p4e2_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p4e3_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p5e0_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p5e1_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p5e2_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p5e3_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p6e0_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p6e1_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p6e2_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p6e3_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p7e0_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p7e1_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p7e2_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p7e3_bits_to_double(int8_t) { return unsupported_extra_posit_bits_to_double(); }
static double p8e0_bits_to_double(int8_t bits) {
  return convertP8ToDouble(castP8(static_cast<uint8_t>(bits)));
}
#if defined(POSIT_USE_SOFTPOSIT_PX1)
static double p8e1_bits_to_double(int8_t bits) {
  posit_1_t p;
  p.v = static_cast<uint32_t>(static_cast<uint8_t>(bits)) << 24;
  p = pX1_to_pX1(p, 8);
  return convertPX1ToDouble(p);
}
#else
static double p8e1_bits_to_double(int8_t) {
  std::cerr
      << "p8e1 requires POSIT_USE_UNIVERSAL or POSIT_USE_SOFTPOSIT_PX1\n";
  std::exit(1);
}
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX2)
static double p8e2_bits_to_double(int8_t bits) {
  posit_2_t p;
  p.v = static_cast<uint32_t>(static_cast<uint8_t>(bits)) << 24;
  p = pX2_to_pX2(p, 8);
  return convertPX2ToDouble(p);
}
#else
static double p8e2_bits_to_double(int8_t) {
  std::cerr
      << "p8e2 requires POSIT_USE_UNIVERSAL or POSIT_USE_SOFTPOSIT_PX2\n";
  std::exit(1);
}
#endif
static double p9e0_bits_to_double(int16_t) { return unsupported_extra_posit_bits_to_double(); }
static double p9e1_bits_to_double(int16_t) { return unsupported_extra_posit_bits_to_double(); }
static double p9e2_bits_to_double(int16_t) { return unsupported_extra_posit_bits_to_double(); }
static double p9e3_bits_to_double(int16_t) { return unsupported_extra_posit_bits_to_double(); }
static double unsupported_posit_bits_to_double() {
  std::cerr << "p16e0/p32e0 require POSIT_USE_UNIVERSAL or POSIT_USE_UNIVERSAL_FALLBACK\n";
  std::exit(1);
}
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
static double p16e0_bits_to_double(int16_t bits) {
  return universal_bits_to_double<16, 0, int16_t, uint16_t>(bits);
}
#else
static double p16e0_bits_to_double(int16_t) { return unsupported_posit_bits_to_double(); }
#endif
static double p16e1_bits_to_double(int16_t bits) {
  return convertP16ToDouble(castP16(static_cast<uint16_t>(bits)));
}
#if defined(POSIT_USE_SOFTPOSIT_PX2)
static double p16e2_bits_to_double(int16_t bits) {
  posit_2_t p;
  p.v = static_cast<uint32_t>(static_cast<uint16_t>(bits)) << 16;
  p = pX2_to_pX2(p, 16);
  return convertPX2ToDouble(p);
}
#else
static double p16e2_bits_to_double(int16_t) { return unsupported_posit_bits_to_double(); }
#endif
#if defined(POSIT_USE_UNIVERSAL_FALLBACK)
static double p32e0_bits_to_double(int32_t bits) {
  return universal_bits_to_double<32, 0, int32_t, uint32_t>(bits);
}
#else
static double p32e0_bits_to_double(int32_t) { return unsupported_posit_bits_to_double(); }
#endif
#if defined(POSIT_USE_SOFTPOSIT_PX1)
static double p32e1_bits_to_double(int32_t bits) {
  posit_1_t p;
  p.v = static_cast<uint32_t>(bits);
  p = pX1_to_pX1(p, 32);
  return convertPX1ToDouble(p);
}
#else
static double p32e1_bits_to_double(int32_t) { return unsupported_posit_bits_to_double(); }
#endif
static double p32e2_bits_to_double(int32_t bits) {
  return convertP32ToDouble(castP32(static_cast<uint32_t>(bits)));
}
#endif

static double f32_to_double(float x) { return static_cast<double>(x); }

template <typename T>
static void freeOut(OutDescT<T> &out, bool doFree) {
  if (!doFree)
    return;
  if (out.basePtr)
    std::free(out.basePtr);
  else if (out.data)
    std::free(out.data);
  out.basePtr = nullptr;
  out.data = nullptr;
}

template <typename T>
static RunStats benchmarkMain(void *sym, InDesc &in, int64_t warmup, int64_t iters,
                              bool stats, bool doFree) {
  auto fn = reinterpret_cast<MainFnT<T>>(sym);
  for (int64_t i = 0; i < warmup; ++i) {
    OutDescT<T> out;
    fn(&out, &in);
    freeOut(out, doFree);
  }

  RunStats rs;
  if (!stats) {
    auto t0 = std::chrono::steady_clock::now();
    for (int64_t i = 0; i < iters; ++i) {
      OutDescT<T> out;
      fn(&out, &in);
      freeOut(out, doFree);
    }
    auto t1 = std::chrono::steady_clock::now();
    rs.avg_us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                static_cast<double>(iters);
    return rs;
  }

  std::vector<double> us;
  us.reserve(static_cast<size_t>(iters));
  for (int64_t i = 0; i < iters; ++i) {
    OutDescT<T> out;
    auto t0 = std::chrono::steady_clock::now();
    fn(&out, &in);
    auto t1 = std::chrono::steady_clock::now();
    freeOut(out, doFree);
    us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
  }
  std::sort(us.begin(), us.end());
  rs.min_us = us.front();
  rs.max_us = us.back();
  rs.p50_us = us[static_cast<size_t>(0.50 * (iters - 1))];
  rs.p95_us = us[static_cast<size_t>(0.95 * (iters - 1))];
  rs.avg_us = std::accumulate(us.begin(), us.end(), 0.0) / static_cast<double>(iters);
  return rs;
}

template <typename T>
static bool inferLogits(void *sym, InDesc &in, bool doFree, DecodeFn<T> decodeBits,
                        std::vector<double> &logitsOut) {
  auto fn = reinterpret_cast<MainFnT<T>>(sym);
  OutDescT<T> out;
  fn(&out, &in);

  if (out.sizes[0] <= 0 || out.sizes[1] <= 0) {
    std::cerr << "[ERROR] unexpected output shape: [" << out.sizes[0] << "," << out.sizes[1]
              << "]\n";
    freeOut(out, doFree);
    return false;
  }

  int64_t cols = out.sizes[1];
  logitsOut.assign(static_cast<size_t>(cols), 0.0);
  for (int64_t j = 0; j < cols; ++j) {
    int64_t idx = out.offset + j * out.strides[1];
    logitsOut[static_cast<size_t>(j)] = decodeBits(out.data[idx]);
  }
  freeOut(out, doFree);
  return true;
}

static std::vector<int> topkIndices(const std::vector<double> &v, int k) {
  std::vector<int> idx(v.size());
  for (size_t i = 0; i < v.size(); ++i)
    idx[i] = static_cast<int>(i);
  if (k < 0)
    k = 0;
  if (k > static_cast<int>(idx.size()))
    k = static_cast<int>(idx.size());
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                    [&](int a, int b) { return v[static_cast<size_t>(a)] > v[static_cast<size_t>(b)]; });
  idx.resize(static_cast<size_t>(k));
  return idx;
}

static std::vector<double> softmaxStable(const std::vector<double> &x) {
  std::vector<double> p(x.size(), 0.0);
  if (x.empty())
    return p;
  double mx = *std::max_element(x.begin(), x.end());
  double sum = 0.0;
  for (size_t i = 0; i < x.size(); ++i) {
    p[i] = std::exp(x[i] - mx);
    sum += p[i];
  }
  if (sum > 0.0) {
    for (double &v : p)
      v /= sum;
  }
  return p;
}

static CompareStats compareLogits(const std::vector<double> &target,
                                  const std::vector<double> &base) {
  CompareStats s;
  s.n = static_cast<int>(std::min(target.size(), base.size()));
  if (s.n <= 0)
    return s;

  double absSum = 0.0;
  double sqSum = 0.0;
  double dot = 0.0;
  double na2 = 0.0;
  double nb2 = 0.0;
  double relSum = 0.0;
  constexpr double eps = 1e-12;

  for (int i = 0; i < s.n; ++i) {
    double a = target[static_cast<size_t>(i)];
    double b = base[static_cast<size_t>(i)];
    double d = a - b;
    double ad = std::fabs(d);
    absSum += ad;
    sqSum += d * d;
    s.max_abs = std::max(s.max_abs, ad);
    dot += a * b;
    na2 += a * a;
    nb2 += b * b;
    relSum += ad / (std::fabs(b) + eps);
  }

  s.mae = absSum / static_cast<double>(s.n);
  s.rmse = std::sqrt(sqSum / static_cast<double>(s.n));
  s.mean_rel_abs = relSum / static_cast<double>(s.n);
  s.cosine = (na2 > 0.0 && nb2 > 0.0) ? (dot / (std::sqrt(na2) * std::sqrt(nb2))) : 0.0;

  std::vector<double> ta(target.begin(), target.begin() + s.n);
  std::vector<double> ba(base.begin(), base.begin() + s.n);
  auto pa = softmaxStable(ta);
  auto pb = softmaxStable(ba);
  std::vector<double> m(static_cast<size_t>(s.n), 0.0);
  for (int i = 0; i < s.n; ++i)
    m[static_cast<size_t>(i)] = 0.5 * (pa[static_cast<size_t>(i)] + pb[static_cast<size_t>(i)]);
  auto kl = [&](const std::vector<double> &p, const std::vector<double> &q) {
    double v = 0.0;
    for (int i = 0; i < s.n; ++i) {
      double pi = std::max(p[static_cast<size_t>(i)], eps);
      double qi = std::max(q[static_cast<size_t>(i)], eps);
      v += pi * std::log(pi / qi);
    }
    return v;
  };
  s.js_div = 0.5 * kl(pa, m) + 0.5 * kl(pb, m);

  auto t1 = topkIndices(ta, 1);
  auto b1 = topkIndices(ba, 1);
  s.top1_target = t1.empty() ? -1 : t1[0];
  s.top1_base = b1.empty() ? -1 : b1[0];
  s.top1_match = (s.top1_target == s.top1_base);

  auto t5 = topkIndices(ta, 5);
  auto b5 = topkIndices(ba, 5);
  int overlap = 0;
  for (int x : t5)
    for (int y : b5)
      if (x == y)
        ++overlap;
  s.top5_overlap = overlap;
  s.top5_exact = (overlap == static_cast<int>(std::min(t5.size(), b5.size())) &&
                  t5.size() == b5.size());
  return s;
}

template <typename T>
static bool inferByType(void *sym, InDesc &in, bool doFree, DecodeFn<T> decode,
                        std::vector<double> &logitsOut) {
  return inferLogits<T>(sym, in, doFree, decode, logitsOut);
}

static bool inferByOutType(void *sym, InDesc &in, bool doFree, OutType outType,
                           std::vector<double> &logitsOut) {
  switch (outType) {
  case OutType::P4E0:
    return inferByType<int8_t>(sym, in, doFree, p4e0_bits_to_double, logitsOut);
  case OutType::P4E1:
    return inferByType<int8_t>(sym, in, doFree, p4e1_bits_to_double, logitsOut);
  case OutType::P4E2:
    return inferByType<int8_t>(sym, in, doFree, p4e2_bits_to_double, logitsOut);
  case OutType::P4E3:
    return inferByType<int8_t>(sym, in, doFree, p4e3_bits_to_double, logitsOut);
  case OutType::P5E0:
    return inferByType<int8_t>(sym, in, doFree, p5e0_bits_to_double, logitsOut);
  case OutType::P5E1:
    return inferByType<int8_t>(sym, in, doFree, p5e1_bits_to_double, logitsOut);
  case OutType::P5E2:
    return inferByType<int8_t>(sym, in, doFree, p5e2_bits_to_double, logitsOut);
  case OutType::P5E3:
    return inferByType<int8_t>(sym, in, doFree, p5e3_bits_to_double, logitsOut);
  case OutType::P6E0:
    return inferByType<int8_t>(sym, in, doFree, p6e0_bits_to_double, logitsOut);
  case OutType::P6E1:
    return inferByType<int8_t>(sym, in, doFree, p6e1_bits_to_double, logitsOut);
  case OutType::P6E2:
    return inferByType<int8_t>(sym, in, doFree, p6e2_bits_to_double, logitsOut);
  case OutType::P6E3:
    return inferByType<int8_t>(sym, in, doFree, p6e3_bits_to_double, logitsOut);
  case OutType::P7E0:
    return inferByType<int8_t>(sym, in, doFree, p7e0_bits_to_double, logitsOut);
  case OutType::P7E1:
    return inferByType<int8_t>(sym, in, doFree, p7e1_bits_to_double, logitsOut);
  case OutType::P7E2:
    return inferByType<int8_t>(sym, in, doFree, p7e2_bits_to_double, logitsOut);
  case OutType::P7E3:
    return inferByType<int8_t>(sym, in, doFree, p7e3_bits_to_double, logitsOut);
  case OutType::P8E0:
    return inferByType<int8_t>(sym, in, doFree, p8e0_bits_to_double, logitsOut);
  case OutType::P8E1:
    return inferByType<int8_t>(sym, in, doFree, p8e1_bits_to_double, logitsOut);
  case OutType::P8E2:
    return inferByType<int8_t>(sym, in, doFree, p8e2_bits_to_double, logitsOut);
  case OutType::P9E0:
    return inferByType<int16_t>(sym, in, doFree, p9e0_bits_to_double, logitsOut);
  case OutType::P9E1:
    return inferByType<int16_t>(sym, in, doFree, p9e1_bits_to_double, logitsOut);
  case OutType::P9E2:
    return inferByType<int16_t>(sym, in, doFree, p9e2_bits_to_double, logitsOut);
  case OutType::P9E3:
    return inferByType<int16_t>(sym, in, doFree, p9e3_bits_to_double, logitsOut);
  case OutType::P16E0:
    return inferByType<int16_t>(sym, in, doFree, p16e0_bits_to_double, logitsOut);
  case OutType::P16E1:
    return inferByType<int16_t>(sym, in, doFree, p16e1_bits_to_double, logitsOut);
  case OutType::P16E2:
    return inferByType<int16_t>(sym, in, doFree, p16e2_bits_to_double, logitsOut);
  case OutType::P32E0:
    return inferByType<int32_t>(sym, in, doFree, p32e0_bits_to_double, logitsOut);
  case OutType::P32E1:
    return inferByType<int32_t>(sym, in, doFree, p32e1_bits_to_double, logitsOut);
  case OutType::P32E2:
    return inferByType<int32_t>(sym, in, doFree, p32e2_bits_to_double, logitsOut);
  case OutType::F32:
    return inferByType<float>(sym, in, doFree, f32_to_double, logitsOut);
  }
  return false;
}

static bool timedInferByOutType(void *sym, InDesc &in, bool doFree, OutType outType,
                                std::vector<double> &logitsOut, double &inferUsOut) {
  auto t0 = std::chrono::high_resolution_clock::now();
  bool ok = inferByOutType(sym, in, doFree, outType, logitsOut);
  auto t1 = std::chrono::high_resolution_clock::now();
  inferUsOut = std::chrono::duration<double, std::micro>(t1 - t0).count();
  return ok;
}

static RunStats benchmarkByOutType(void *sym, InDesc &in, int64_t warmup, int64_t iters,
                                   bool stats, bool doFree, OutType outType) {
  switch (outType) {
  case OutType::P4E0:
  case OutType::P4E1:
  case OutType::P4E2:
  case OutType::P4E3:
  case OutType::P5E0:
  case OutType::P5E1:
  case OutType::P5E2:
  case OutType::P5E3:
  case OutType::P6E0:
  case OutType::P6E1:
  case OutType::P6E2:
  case OutType::P6E3:
  case OutType::P7E0:
  case OutType::P7E1:
  case OutType::P7E2:
  case OutType::P7E3:
  case OutType::P8E0:
  case OutType::P8E1:
  case OutType::P8E2:
    return benchmarkMain<int8_t>(sym, in, warmup, iters, stats, doFree);
  case OutType::P9E0:
  case OutType::P9E1:
  case OutType::P9E2:
  case OutType::P9E3:
  case OutType::P16E0:
  case OutType::P16E1:
  case OutType::P16E2:
    return benchmarkMain<int16_t>(sym, in, warmup, iters, stats, doFree);
  case OutType::P32E0:
  case OutType::P32E1:
  case OutType::P32E2:
    return benchmarkMain<int32_t>(sym, in, warmup, iters, stats, doFree);
  case OutType::F32:
    return benchmarkMain<float>(sym, in, warmup, iters, stats, doFree);
  }
  return RunStats{};
}

static void usage(const char *prog) {
  std::cerr
      << "Usage:\n"
      << "  " << prog << " <model.so> [input.txt]\n"
      << "    [--image FILE --image-preprocess-script FILE]\n"
      << "    [--shape NxCxHxW] [--zeros|--random [seed]]\n"
      << "    [--out-type p*e*|f32]\n"
      << "    [--posit p*e*] [--entry symbol] [--cmp so:type[:entry] ...]\n"
      << "    [--baseline main|cmp:N] [--label class_id]\n"
      << "    [--resize-short N] [--crop-size N]\n"
      << "    [--dump-logits FILE] [--dump-cmp-logits-dir DIR]\n"
      << "    [--warmup N] [--iters N] [--quire on|off]\n"
      << "    [--stats] [--quiet] [--no-free] [--no-benchmark]\n";
}

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 1;
  }
  if (argc == 2) {
    std::string a = argv[1];
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    }
  }

  std::string soPath = argv[1];
  std::string inputPath;
  std::string imagePath;
  std::string imagePreprocessScript;
  int64_t inputShape[4] = {1, 3, 224, 224};
  int64_t resizeShort = 256;
  int64_t cropSize = 224;
  bool forceZeros = false;
  bool forceRandom = true;
  uint32_t randomSeed = 12345u;

  int64_t warmup = 20;
  int64_t iters = 200;
  bool stats = false;
  bool quiet = false;
  bool doFree = true;
  bool noBenchmark = false;
  bool quireP8 = true;
  // Default to f32 decode at graph boundary for stable metric comparisons.
  OutType mainOutType = OutType::F32;
  std::string mainEntry = "_mlir_ciface_main_graph";
  std::string refEntry = "_mlir_ciface_main_graph";
  bool mainEntryExplicit = false;
  bool refEntryExplicit = false;
  std::string baselineSpec = "main";
  int64_t label = -1;
  std::string dumpLogitsPath;
  std::string dumpCmpLogitsDir;

  std::vector<RefSpec> refs;

  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    } else if (a == "--shape") {
      if (i + 1 >= argc) {
        std::cerr << "--shape requires NxCxHxW\n";
        return 1;
      }
      if (!parseShape4(argv[++i], inputShape)) {
        std::cerr << "Invalid --shape, expect NxCxHxW with 4 positive dims\n";
        return 1;
      }
    } else if (a == "--zeros") {
      forceZeros = true;
      forceRandom = false;
    } else if (a == "--image") {
      if (i + 1 >= argc)
        return 1;
      imagePath = argv[++i];
      forceRandom = false;
    } else if (a == "--image-preprocess-script") {
      if (i + 1 >= argc)
        return 1;
      imagePreprocessScript = argv[++i];
    } else if (a == "--resize-short") {
      if (i + 1 >= argc)
        return 1;
      resizeShort = parseI64(argv[++i], "resize-short");
    } else if (a == "--crop-size") {
      if (i + 1 >= argc)
        return 1;
      cropSize = parseI64(argv[++i], "crop-size");
    } else if (a == "--random") {
      forceRandom = true;
      forceZeros = false;
      if (i + 1 < argc && !isFlag(argv[i + 1]))
        randomSeed = static_cast<uint32_t>(parseI64(argv[++i], "seed"));
    } else if (a == "--out-type") {
      if (i + 1 >= argc)
        return 1;
      mainOutType = parseOutType(argv[++i]);
    } else if (a == "--posit") {
      if (i + 1 >= argc)
        return 1;
      mainOutType = parseOutType(argv[++i]);
    } else if (a == "--entry") {
      if (i + 1 >= argc)
        return 1;
      mainEntry = argv[++i];
      mainEntryExplicit = true;
    } else if (a == "--ref-entry") {
      if (i + 1 >= argc)
        return 1;
      refEntry = argv[++i];
      refEntryExplicit = true;
    } else if (a == "--cmp") {
      if (i + 1 >= argc)
        return 1;
      refs.push_back(parseCmpSpec(argv[++i]));
    } else if (a == "--baseline") {
      if (i + 1 >= argc)
        return 1;
      baselineSpec = argv[++i];
    } else if (a == "--label") {
      if (i + 1 >= argc)
        return 1;
      label = parseI64(argv[++i], "label");
    } else if (a == "--dump-logits") {
      if (i + 1 >= argc)
        return 1;
      dumpLogitsPath = argv[++i];
    } else if (a == "--dump-cmp-logits-dir") {
      if (i + 1 >= argc)
        return 1;
      dumpCmpLogitsDir = argv[++i];
    } else if (a == "--warmup") {
      if (i + 1 >= argc)
        return 1;
      warmup = parseI64(argv[++i], "warmup");
      if (warmup < 0)
        warmup = 0;
    } else if (a == "--iters") {
      if (i + 1 >= argc)
        return 1;
      iters = parseI64(argv[++i], "iters");
      if (iters < 1)
        iters = 1;
    } else if (a == "--quire") {
      if (i + 1 >= argc)
        return 1;
      bool parsed = false;
      if (!parseOnOff(argv[++i], parsed)) {
        std::cerr << "Invalid --quire value, expected on|off|1|0\n";
        return 1;
      }
      quireP8 = parsed;
    } else if (a == "--stats") {
      stats = true;
    } else if (a == "--quiet") {
      quiet = true;
    } else if (a == "--no-free") {
      doFree = false;
    } else if (a == "--no-benchmark") {
      noBenchmark = true;
    } else if (!isFlag(a) && inputPath.empty()) {
      inputPath = a;
      forceRandom = false;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      usage(argv[0]);
      return 1;
    }
  }

  if (!imagePath.empty() && !inputPath.empty()) {
    std::cerr << "Use either positional input.txt or --image, not both\n";
    return 1;
  }
  if (imagePreprocessScript.empty()) {
    const char *envScript = std::getenv("POSIT_IMAGE_PREPROCESS_SCRIPT");
    if (envScript)
      imagePreprocessScript = envScript;
  }

  bool baselineIsMain = true;
  size_t baselineCmpIdx = 0;
  if (baselineSpec == "main") {
    baselineIsMain = true;
  } else if (baselineSpec.rfind("cmp:", 0) == 0) {
    baselineIsMain = false;
    int64_t idx = parseI64(baselineSpec.substr(4), "baseline cmp index");
    if (idx < 1) {
      std::cerr << "--baseline cmp:N requires N >= 1\n";
      return 1;
    }
    baselineCmpIdx = static_cast<size_t>(idx - 1);
    if (baselineCmpIdx >= refs.size()) {
      std::cerr << "--baseline " << baselineSpec << " out of range\n";
      return 1;
    }
  } else {
    std::cerr << "Unsupported --baseline: " << baselineSpec << "\n";
    return 1;
  }

  setenv("POSIT_QUIRE_P8", quireP8 ? "1" : "0", 1);
  setenv("POSIT_QUIRE_SMALL", quireP8 ? "1" : "0", 1);
  if (!quiet) {
    std::cout << "CONFIG quire_p8=" << (quireP8 ? "on" : "off") << "\n";
    std::cout << "CONFIG quire_small=" << (quireP8 ? "on" : "off") << "\n";
  }

  std::vector<float> input;
  input.resize(static_cast<size_t>(numInputElems(inputShape)));
  if (!imagePath.empty()) {
    std::vector<float> img;
    if (!preprocessImageToFloats(
            imagePath, imagePreprocessScript, inputShape, resizeShort, cropSize, img)) {
      return 1;
    } else if (img.size() != input.size()) {
      std::cerr << "[ERROR] image tensor size mismatch (file=" << img.size()
                << ", need=" << input.size() << ")\n";
      return 1;
    } else {
      input = std::move(img);
    }
  } else if (!inputPath.empty()) {
    std::vector<float> txt;
    if (!readTxtFloats(inputPath, txt)) {
      std::cerr << "[WARN] failed to read input file, fallback random: " << inputPath << "\n";
      makeRandom(input, randomSeed);
    } else if (txt.size() != input.size()) {
      std::cerr << "[WARN] input size mismatch (file=" << txt.size() << ", need="
                << input.size() << "), fallback random\n";
      makeRandom(input, randomSeed);
    } else {
      input = std::move(txt);
    }
  } else if (forceZeros) {
    std::fill(input.begin(), input.end(), 0.0f);
  } else {
    makeRandom(input, randomSeed);
  }

  InDesc in;
  fillInputDesc(in, input, inputShape);

  std::string soResolved = resolveSoPathForDlopen(soPath);
  void *h = dlopen(soResolved.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    std::cerr << "dlopen failed: " << dlerror() << "\n";
    return 2;
  }

  std::string resolvedMainEntry;
  std::string attemptedMainEntries;
  auto *mainSym = resolveEntrypointSymbol(
      h, soPath, mainEntry, mainEntryExplicit, resolvedMainEntry,
      attemptedMainEntries);
  if (!mainSym) {
    std::cerr << "dlsym failed. attempted entries: [" << attemptedMainEntries
              << "]\n";
    dlclose(h);
    return 3;
  }
  if (!quiet && resolvedMainEntry != mainEntry) {
    std::cout << "CONFIG main_entry_resolved=" << resolvedMainEntry
              << " (requested=" << mainEntry << ")\n";
  }

  RunStats mainBench{};
  if (!noBenchmark)
    mainBench = benchmarkByOutType(mainSym, in, warmup, iters, stats, doFree, mainOutType);
  std::vector<double> mainLogits;
  double mainInferUs = 0.0;
  if (!timedInferByOutType(mainSym, in, doFree, mainOutType, mainLogits, mainInferUs)) {
    dlclose(h);
    return 4;
  }
  if (noBenchmark)
    mainBench.avg_us = mainInferUs;

  std::cout << "MAIN type=" << outTypeToString(mainOutType)
            << " avg=" << mainBench.avg_us << " us"
            << " (iters=" << iters << ", warmup=" << warmup << ")\n";
  if (stats && !noBenchmark) {
    std::cout << "  p50=" << mainBench.p50_us << " us"
              << " p95=" << mainBench.p95_us << " us"
              << " min=" << mainBench.min_us << " us"
              << " max=" << mainBench.max_us << " us\n";
  }

  auto mainTop1 = topkIndices(mainLogits, 1);
  auto mainTop5 = topkIndices(mainLogits, 5);
  std::cout << "  top1=" << (mainTop1.empty() ? -1 : mainTop1[0]) << " top5=[";
  for (size_t i = 0; i < mainTop5.size(); ++i)
    std::cout << mainTop5[i] << (i + 1 == mainTop5.size() ? "" : ",");
  std::cout << "]\n";
  if (label >= 0) {
    bool hit1 = (!mainTop1.empty() && mainTop1[0] == label);
    bool hit5 = false;
    for (int c : mainTop5)
      if (c == label)
        hit5 = true;
    std::cout << "  label=" << label << " top1_hit=" << (hit1 ? "yes" : "no")
              << " top5_hit=" << (hit5 ? "yes" : "no") << "\n";
  }
  if (!dumpLogitsPath.empty()) {
    if (!writeLogitsCsv(dumpLogitsPath, mainLogits)) {
      std::cerr << "Failed to write --dump-logits: " << dumpLogitsPath << "\n";
      dlclose(h);
      return 8;
    }
  }

  struct RefRun {
    RefSpec spec;
    RunStats bench;
    std::vector<double> logits;
  };
  std::vector<RefRun> refRuns;
  refRuns.reserve(refs.size());

  for (size_t i = 0; i < refs.size(); ++i) {
    const RefSpec &ref = refs[i];
    std::string refResolved = resolveSoPathForDlopen(ref.soPath);
    void *hr = dlopen(refResolved.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!hr) {
      std::cerr << "CMP#" << (i + 1) << " dlopen failed: " << dlerror() << "\n";
      dlclose(h);
      return 5;
    }

    const bool explicitCmpEntry = !ref.entrySymbol.empty();
    const std::string &entry = explicitCmpEntry ? ref.entrySymbol : refEntry;
    std::string resolvedCmpEntry;
    std::string attemptedCmpEntries;
    auto *sym = resolveEntrypointSymbol(hr, ref.soPath, entry,
        explicitCmpEntry || refEntryExplicit, resolvedCmpEntry,
        attemptedCmpEntries);
    if (!sym) {
      std::cerr << "CMP#" << (i + 1) << " dlsym failed. attempted entries: ["
                << attemptedCmpEntries << "]\n";
      dlclose(hr);
      dlclose(h);
      return 6;
    }
    if (!quiet && resolvedCmpEntry != entry) {
      std::cout << "CONFIG cmp#" << (i + 1)
                << "_entry_resolved=" << resolvedCmpEntry
                << " (requested=" << entry << ")\n";
    }

    RefRun rr;
    rr.spec = ref;
    if (!noBenchmark)
      rr.bench = benchmarkByOutType(sym, in, warmup, iters, stats, doFree, ref.outType);
    double refInferUs = 0.0;
    if (!timedInferByOutType(sym, in, doFree, ref.outType, rr.logits, refInferUs)) {
      dlclose(hr);
      dlclose(h);
      return 7;
    }
    if (noBenchmark)
      rr.bench.avg_us = refInferUs;
    refRuns.push_back(std::move(rr));

    std::cout << "CMP#" << (i + 1) << " type=" << outTypeToString(ref.outType)
              << " avg=" << refRuns.back().bench.avg_us << " us"
              << " so=" << ref.soPath << "\n";
    if (stats && !noBenchmark) {
      std::cout << "  p50=" << refRuns.back().bench.p50_us << " us"
                << " p95=" << refRuns.back().bench.p95_us << " us"
                << " min=" << refRuns.back().bench.min_us << " us"
                << " max=" << refRuns.back().bench.max_us << " us\n";
    }

    auto rTop1 = topkIndices(refRuns.back().logits, 1);
    auto rTop5 = topkIndices(refRuns.back().logits, 5);
    std::cout << "  top1=" << (rTop1.empty() ? -1 : rTop1[0]) << " top5=[";
    for (size_t k = 0; k < rTop5.size(); ++k)
      std::cout << rTop5[k] << (k + 1 == rTop5.size() ? "" : ",");
    std::cout << "]\n";
    if (label >= 0) {
      bool hit1 = (!rTop1.empty() && rTop1[0] == label);
      bool hit5 = false;
      for (int c : rTop5)
        if (c == label)
          hit5 = true;
      std::cout << "  label=" << label << " top1_hit=" << (hit1 ? "yes" : "no")
                << " top5_hit=" << (hit5 ? "yes" : "no") << "\n";
    }

    if (!dumpCmpLogitsDir.empty()) {
      std::string cmpPath = dumpCmpLogitsDir + "/cmp" + std::to_string(i + 1) + "_" +
                            outTypeToString(ref.outType) + ".csv";
      if (!writeLogitsCsv(cmpPath, refRuns.back().logits)) {
        std::cerr << "Failed to write cmp logits: " << cmpPath << "\n";
        dlclose(hr);
        dlclose(h);
        return 9;
      }
    }

    dlclose(hr);
  }

  const char *baseTag = outTypeToString(mainOutType);
  std::string baseSo = soPath;
  std::vector<double> *baseLogits = &mainLogits;
  double baseLatency = mainBench.avg_us;
  if (!baselineIsMain) {
    baseTag = outTypeToString(refRuns[baselineCmpIdx].spec.outType);
    baseSo = refRuns[baselineCmpIdx].spec.soPath;
    baseLogits = &refRuns[baselineCmpIdx].logits;
    baseLatency = refRuns[baselineCmpIdx].bench.avg_us;
  }

  std::cout << "COMPARE baseline=" << baseTag << " so=" << baseSo << "\n";

  size_t cmpNo = 1;
  auto printOneCompare = [&](const char *tag, const std::string &targetSo,
                             const std::vector<double> &targetLogits, double targetLatency) {
    CompareStats cs = compareLogits(targetLogits, *baseLogits);
    std::cout << "  C" << cmpNo++ << " target=" << tag
              << " so=" << targetSo
              << " N=" << cs.n
              << " MAE=" << cs.mae
              << " RMSE=" << cs.rmse
              << " MaxAbs=" << cs.max_abs
              << " RMAE=" << cs.mean_rel_abs
              << " Cosine=" << cs.cosine
              << " JS=" << cs.js_div
              << " Top1(" << cs.top1_target << "," << cs.top1_base << ")="
              << (cs.top1_match ? "match" : "diff")
              << " Top5Overlap=" << cs.top5_overlap
              << " LatencyRatio=" << ((baseLatency > 0.0) ? (targetLatency / baseLatency) : 0.0)
              << "x\n";
  };

  if (baselineIsMain) {
    for (const RefRun &rr : refRuns)
      printOneCompare(outTypeToString(rr.spec.outType), rr.spec.soPath, rr.logits,
                      rr.bench.avg_us);
  } else {
    printOneCompare(outTypeToString(mainOutType), soPath, mainLogits, mainBench.avg_us);
    for (size_t i = 0; i < refRuns.size(); ++i) {
      if (i == baselineCmpIdx)
        continue;
      const RefRun &rr = refRuns[i];
      printOneCompare(outTypeToString(rr.spec.outType), rr.spec.soPath, rr.logits,
                      rr.bench.avg_us);
    }
  }

  if (!quiet) {
    std::cout << "logits(main) size=" << mainLogits.size() << "\n";
  }

  dlclose(h);
  return 0;
}
