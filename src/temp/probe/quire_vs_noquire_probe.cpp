#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

extern "C" {
#include "softposit.h"
}

namespace {

template <typename T>
static inline int64_t idx3(int64_t a, int64_t b, int64_t c, int64_t B, int64_t C) {
  return (a * B + b) * C + c;
}

template <typename T>
static inline int64_t idx4(
    int64_t a, int64_t b, int64_t c, int64_t d, int64_t B, int64_t C, int64_t D) {
  return ((a * B + b) * C + c) * D + d;
}

static inline int8_t quant_i8(double x, double s, int zp = 0) {
  long q = std::lround(x / s) + zp;
  q = std::max<long>(-128, std::min<long>(127, q));
  return static_cast<int8_t>(q);
}

struct FmtP8E0 {
  using Posit = posit8_t;
  static const char *name() { return "p8e0"; }
  static Posit fromDouble(double x) { return convertDoubleToP8(x); }
  static double toDouble(Posit p) { return convertP8ToDouble(p); }
  static Posit add(Posit a, Posit b) { return p8_add(a, b); }
  static Posit mul(Posit a, Posit b) { return p8_mul(a, b); }
};

struct FmtP8E2 {
  using Posit = posit_2_t;
  static const char *name() { return "p8e2"; }
  static Posit fromDouble(double x) { return convertDoubleToPX2(x, 8); }
  static double toDouble(Posit p) { return convertPX2ToDouble(pX2_to_pX2(p, 8)); }
  static Posit add(Posit a, Posit b) { return pX2_add(a, b, 8); }
  static Posit mul(Posit a, Posit b) { return pX2_mul(a, b, 8); }
};

struct FmtP16E1 {
  using Posit = posit16_t;
  static const char *name() { return "p16e1"; }
  static Posit fromDouble(double x) { return convertDoubleToP16(x); }
  static double toDouble(Posit p) { return convertP16ToDouble(p); }
  static Posit add(Posit a, Posit b) { return p16_add(a, b); }
  static Posit mul(Posit a, Posit b) { return p16_mul(a, b); }
};

struct FmtP32E2 {
  using Posit = posit32_t;
  static const char *name() { return "p32e2"; }
  static Posit fromDouble(double x) { return convertDoubleToP32(x); }
  static double toDouble(Posit p) { return convertP32ToDouble(p); }
  static Posit add(Posit a, Posit b) { return p32_add(a, b); }
  static Posit mul(Posit a, Posit b) { return p32_mul(a, b); }
};

template <typename Fmt>
struct QuireTrait;

template <>
struct QuireTrait<FmtP8E0> {
  using Posit = FmtP8E0::Posit;
  using Quire = quire8_t;
  static Quire clear() { return q8Clr(); }
  static void fdp(Quire &q, Posit a, Posit b) { q = q8_fdp_add(q, a, b); }
  static Posit toPosit(Quire q) { return q8_to_p8(q); }
};

template <>
struct QuireTrait<FmtP8E2> {
  using Posit = FmtP8E2::Posit;
  using Quire = quire_2_t;
  static Quire clear() { return qX2Clr(); }
  static void fdp(Quire &q, Posit a, Posit b) { q = qX2_fdp_add(q, a, b); }
  static Posit toPosit(Quire q) { return qX2_to_pX2(q, 8); }
};

template <>
struct QuireTrait<FmtP16E1> {
  using Posit = FmtP16E1::Posit;
  using Quire = quire16_t;
  static Quire clear() { return q16Clr(); }
  static void fdp(Quire &q, Posit a, Posit b) { q = q16_fdp_add(q, a, b); }
  static Posit toPosit(Quire q) { return q16_to_p16(q); }
};

template <>
struct QuireTrait<FmtP32E2> {
  using Posit = FmtP32E2::Posit;
  using Quire = quire32_t;
  static Quire clear() { return q32Clr(); }
  static void fdp(Quire &q, Posit a, Posit b) { q = q32_fdp_add(q, a, b); }
  static Posit toPosit(Quire q) { return q32_to_p32(q); }
};

template <typename Fmt>
static std::vector<typename Fmt::Posit> encodeVec(const std::vector<double> &src) {
  std::vector<typename Fmt::Posit> out(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    out[i] = Fmt::fromDouble(src[i]);
  return out;
}

template <typename Fmt>
static std::vector<double> conv2d(
    const std::vector<typename Fmt::Posit> &x, const std::vector<typename Fmt::Posit> &w,
    const std::vector<typename Fmt::Posit> &b, int64_t C, int64_t H, int64_t W,
    int64_t M, int64_t kH, int64_t kW, bool useQuire) {
  int64_t outH = H - kH + 1;
  int64_t outW = W - kW + 1;
  std::vector<double> out(static_cast<size_t>(M * outH * outW), 0.0);

  for (int64_t oc = 0; oc < M; ++oc) {
    for (int64_t oh = 0; oh < outH; ++oh) {
      for (int64_t ow = 0; ow < outW; ++ow) {
        typename Fmt::Posit dot;
        if (useQuire) {
          auto q = QuireTrait<Fmt>::clear();
          for (int64_t ic = 0; ic < C; ++ic) {
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                auto px = x[idx3<int64_t>(ic, oh + kh, ow + kw, H, W)];
                auto pw = w[idx4<int64_t>(oc, ic, kh, kw, C, kH, kW)];
                QuireTrait<Fmt>::fdp(q, px, pw);
              }
            }
          }
          dot = QuireTrait<Fmt>::toPosit(q);
        } else {
          dot = Fmt::fromDouble(0.0);
          for (int64_t ic = 0; ic < C; ++ic) {
            for (int64_t kh = 0; kh < kH; ++kh) {
              for (int64_t kw = 0; kw < kW; ++kw) {
                auto px = x[idx3<int64_t>(ic, oh + kh, ow + kw, H, W)];
                auto pw = w[idx4<int64_t>(oc, ic, kh, kw, C, kH, kW)];
                dot = Fmt::add(dot, Fmt::mul(px, pw));
              }
            }
          }
        }
        auto y = Fmt::add(dot, b[static_cast<size_t>(oc)]);
        out[idx3<int64_t>(oc, oh, ow, outH, outW)] = Fmt::toDouble(y);
      }
    }
  }
  return out;
}

template <typename Fmt>
static std::vector<double> gemm(
    const std::vector<typename Fmt::Posit> &A, const std::vector<typename Fmt::Posit> &B,
    const std::vector<typename Fmt::Posit> &C, int64_t M, int64_t K, int64_t N,
    double alpha, double beta, bool useQuire) {
  std::vector<double> out(static_cast<size_t>(M * N), 0.0);
  auto pAlpha = Fmt::fromDouble(alpha);
  auto pBeta = Fmt::fromDouble(beta);

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      typename Fmt::Posit dot;
      if (useQuire) {
        auto q = QuireTrait<Fmt>::clear();
        for (int64_t k = 0; k < K; ++k)
          QuireTrait<Fmt>::fdp(
              q, A[idx3<int64_t>(i, k, 0, K, 1)], B[idx3<int64_t>(k, j, 0, N, 1)]);
        dot = QuireTrait<Fmt>::toPosit(q);
      } else {
        dot = Fmt::fromDouble(0.0);
        for (int64_t k = 0; k < K; ++k) {
          auto pa = A[idx3<int64_t>(i, k, 0, K, 1)];
          auto pb = B[idx3<int64_t>(k, j, 0, N, 1)];
          dot = Fmt::add(dot, Fmt::mul(pa, pb));
        }
      }
      auto y = Fmt::add(Fmt::mul(dot, pAlpha), Fmt::mul(C[idx3<int64_t>(i, j, 0, N, 1)], pBeta));
      out[idx3<int64_t>(i, j, 0, N, 1)] = Fmt::toDouble(y);
    }
  }
  return out;
}

static std::vector<double> conv2d_f32(
    const std::vector<double> &x, const std::vector<double> &w, const std::vector<double> &b,
    int64_t C, int64_t H, int64_t W, int64_t M, int64_t kH, int64_t kW) {
  int64_t outH = H - kH + 1;
  int64_t outW = W - kW + 1;
  std::vector<double> out(static_cast<size_t>(M * outH * outW), 0.0);
  for (int64_t oc = 0; oc < M; ++oc) {
    for (int64_t oh = 0; oh < outH; ++oh) {
      for (int64_t ow = 0; ow < outW; ++ow) {
        double acc = b[static_cast<size_t>(oc)];
        for (int64_t ic = 0; ic < C; ++ic)
          for (int64_t kh = 0; kh < kH; ++kh)
            for (int64_t kw = 0; kw < kW; ++kw)
              acc += x[idx3<int64_t>(ic, oh + kh, ow + kw, H, W)] *
                     w[idx4<int64_t>(oc, ic, kh, kw, C, kH, kW)];
        out[idx3<int64_t>(oc, oh, ow, outH, outW)] = acc;
      }
    }
  }
  return out;
}

static std::vector<double> gemm_f32(
    const std::vector<double> &A, const std::vector<double> &B, const std::vector<double> &C,
    int64_t M, int64_t K, int64_t N, double alpha, double beta) {
  std::vector<double> out(static_cast<size_t>(M * N), 0.0);
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      double dot = 0.0;
      for (int64_t k = 0; k < K; ++k)
        dot += A[idx3<int64_t>(i, k, 0, K, 1)] * B[idx3<int64_t>(k, j, 0, N, 1)];
      out[idx3<int64_t>(i, j, 0, N, 1)] = alpha * dot + beta * C[idx3<int64_t>(i, j, 0, N, 1)];
    }
  }
  return out;
}

static std::vector<double> conv2d_int8(
    const std::vector<double> &x, const std::vector<double> &w, const std::vector<double> &b,
    int64_t C, int64_t H, int64_t W, int64_t M, int64_t kH, int64_t kW,
    double sx, double sw) {
  int64_t outH = H - kH + 1;
  int64_t outW = W - kW + 1;
  std::vector<double> out(static_cast<size_t>(M * outH * outW), 0.0);
  std::vector<int8_t> qx(x.size()), qw(w.size());
  for (size_t i = 0; i < x.size(); ++i)
    qx[i] = quant_i8(x[i], sx);
  for (size_t i = 0; i < w.size(); ++i)
    qw[i] = quant_i8(w[i], sw);

  for (int64_t oc = 0; oc < M; ++oc) {
    for (int64_t oh = 0; oh < outH; ++oh) {
      for (int64_t ow = 0; ow < outW; ++ow) {
        int32_t acc = 0;
        for (int64_t ic = 0; ic < C; ++ic)
          for (int64_t kh = 0; kh < kH; ++kh)
            for (int64_t kw = 0; kw < kW; ++kw)
              acc += static_cast<int32_t>(qx[idx3<int64_t>(ic, oh + kh, ow + kw, H, W)]) *
                     static_cast<int32_t>(qw[idx4<int64_t>(oc, ic, kh, kw, C, kH, kW)]);
        out[idx3<int64_t>(oc, oh, ow, outH, outW)] =
            static_cast<double>(acc) * sx * sw + b[static_cast<size_t>(oc)];
      }
    }
  }
  return out;
}

static std::vector<double> gemm_int8(
    const std::vector<double> &A, const std::vector<double> &B, const std::vector<double> &C,
    int64_t M, int64_t K, int64_t N, double alpha, double beta, double sA, double sB) {
  std::vector<double> out(static_cast<size_t>(M * N), 0.0);
  std::vector<int8_t> qA(A.size()), qB(B.size());
  for (size_t i = 0; i < A.size(); ++i)
    qA[i] = quant_i8(A[i], sA);
  for (size_t i = 0; i < B.size(); ++i)
    qB[i] = quant_i8(B[i], sB);

  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (int64_t k = 0; k < K; ++k)
        acc += static_cast<int32_t>(qA[idx3<int64_t>(i, k, 0, K, 1)]) *
               static_cast<int32_t>(qB[idx3<int64_t>(k, j, 0, N, 1)]);
      double dot = static_cast<double>(acc) * sA * sB;
      out[idx3<int64_t>(i, j, 0, N, 1)] = alpha * dot + beta * C[idx3<int64_t>(i, j, 0, N, 1)];
    }
  }
  return out;
}

static double mae(const std::vector<double> &a, const std::vector<double> &b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    s += std::abs(a[i] - b[i]);
  return a.empty() ? 0.0 : s / static_cast<double>(a.size());
}

static void dumpVec(const std::string &path, const std::string &name, const std::vector<double> &v) {
  std::ofstream os(path, std::ios::app);
  os << name << " (size=" << v.size() << ")\n";
  os << std::setprecision(10);
  for (size_t i = 0; i < v.size(); ++i)
    os << i << " " << v[i] << "\n";
  os << "\n";
}

template <typename Fmt>
static void dumpResult(
    std::ofstream &os, const std::string &tag, const std::vector<double> &convNoQ,
    const std::vector<double> &convQ, const std::vector<double> &convRef,
    const std::vector<double> &gemmNoQ, const std::vector<double> &gemmQ,
    const std::vector<double> &gemmRef) {
  os << "[" << tag << "]\n";
  os << "conv_mae_noquire_vs_f32=" << mae(convNoQ, convRef) << "\n";
  os << "conv_mae_quire_vs_f32=" << mae(convQ, convRef) << "\n";
  os << "gemm_mae_noquire_vs_f32=" << mae(gemmNoQ, gemmRef) << "\n";
  os << "gemm_mae_quire_vs_f32=" << mae(gemmQ, gemmRef) << "\n";
  os << "conv_out_noquire=";
  for (double v : convNoQ)
    os << " " << v;
  os << "\nconv_out_quire=";
  for (double v : convQ)
    os << " " << v;
  os << "\n";
  os << "gemm_out_noquire=";
  for (double v : gemmNoQ)
    os << " " << v;
  os << "\ngemm_out_quire=";
  for (double v : gemmQ)
    os << " " << v;
  os << "\n\n";
}

template <typename Fmt>
static void traceP8Style(
    std::ofstream &os, const std::vector<double> &x, const std::vector<double> &w,
    const std::vector<double> &A, const std::vector<double> &B,
    int64_t C, int64_t H, int64_t W, int64_t kH, int64_t kW,
    int64_t K) {
  auto ex = encodeVec<Fmt>(x);
  auto ew = encodeVec<Fmt>(w);
  auto eA = encodeVec<Fmt>(A);
  auto eB = encodeVec<Fmt>(B);

  os << "=== trace " << Fmt::name() << " conv(oc=0,oh=0,ow=0) ===\n";
  auto accNoQ = Fmt::fromDouble(0.0);
  auto q = QuireTrait<Fmt>::clear();
  int step = 0;
  for (int64_t ic = 0; ic < C; ++ic) {
    for (int64_t kh = 0; kh < kH; ++kh) {
      for (int64_t kw = 0; kw < kW; ++kw) {
        auto px = ex[idx3<int64_t>(ic, kh, kw, H, W)];
        auto pw = ew[idx4<int64_t>(0, ic, kh, kw, C, kH, kW)];
        auto prod = Fmt::mul(px, pw);
        accNoQ = Fmt::add(accNoQ, prod);
        QuireTrait<Fmt>::fdp(q, px, pw);
        auto accQasP = QuireTrait<Fmt>::toPosit(q);
        os << "step=" << step
           << " x=" << Fmt::toDouble(px)
           << " w=" << Fmt::toDouble(pw)
           << " prod=" << Fmt::toDouble(prod)
           << " acc_noq=" << Fmt::toDouble(accNoQ)
           << " acc_q_as_p=" << Fmt::toDouble(accQasP)
           << "\n";
        ++step;
      }
    }
  }

  os << "=== trace " << Fmt::name() << " gemm(i=0,j=0) ===\n";
  auto gNoQ = Fmt::fromDouble(0.0);
  auto gq = QuireTrait<Fmt>::clear();
  for (int64_t k = 0; k < K; ++k) {
    auto pa = eA[idx3<int64_t>(0, k, 0, K, 1)];
    auto pb = eB[idx3<int64_t>(k, 0, 0, 3, 1)];
    auto prod = Fmt::mul(pa, pb);
    gNoQ = Fmt::add(gNoQ, prod);
    QuireTrait<Fmt>::fdp(gq, pa, pb);
    auto gqasP = QuireTrait<Fmt>::toPosit(gq);
    os << "k=" << k
       << " a=" << Fmt::toDouble(pa)
       << " b=" << Fmt::toDouble(pb)
       << " prod=" << Fmt::toDouble(prod)
       << " acc_noq=" << Fmt::toDouble(gNoQ)
       << " acc_q_as_p=" << Fmt::toDouble(gqasP)
       << "\n";
  }
  os << "\n";
}

static void traceInt8(
    std::ofstream &os, const std::vector<double> &x, const std::vector<double> &w,
    const std::vector<double> &A, const std::vector<double> &B,
    int64_t C, int64_t H, int64_t W, int64_t kH, int64_t kW,
    int64_t K, double sx, double sw, double sA, double sB) {
  std::vector<int8_t> qx(x.size()), qw(w.size()), qA(A.size()), qB(B.size());
  for (size_t i = 0; i < x.size(); ++i)
    qx[i] = quant_i8(x[i], sx);
  for (size_t i = 0; i < w.size(); ++i)
    qw[i] = quant_i8(w[i], sw);
  for (size_t i = 0; i < A.size(); ++i)
    qA[i] = quant_i8(A[i], sA);
  for (size_t i = 0; i < B.size(); ++i)
    qB[i] = quant_i8(B[i], sB);

  os << "=== trace int8 conv(oc=0,oh=0,ow=0) ===\n";
  int32_t acc = 0;
  int step = 0;
  for (int64_t ic = 0; ic < C; ++ic) {
    for (int64_t kh = 0; kh < kH; ++kh) {
      for (int64_t kw = 0; kw < kW; ++kw) {
        auto qxi = qx[idx3<int64_t>(ic, kh, kw, H, W)];
        auto qwi = qw[idx4<int64_t>(0, ic, kh, kw, C, kH, kW)];
        int32_t prod = static_cast<int32_t>(qxi) * static_cast<int32_t>(qwi);
        acc += prod;
        os << "step=" << step << " qx=" << static_cast<int>(qxi)
           << " qw=" << static_cast<int>(qwi)
           << " prod=" << prod << " acc_i32=" << acc << "\n";
        ++step;
      }
    }
  }
  os << "conv_dequant_without_bias=" << (static_cast<double>(acc) * sx * sw) << "\n\n";

  os << "=== trace int8 gemm(i=0,j=0) ===\n";
  int32_t gacc = 0;
  for (int64_t k = 0; k < K; ++k) {
    auto qa = qA[idx3<int64_t>(0, k, 0, K, 1)];
    auto qb = qB[idx3<int64_t>(k, 0, 0, 3, 1)];
    int32_t prod = static_cast<int32_t>(qa) * static_cast<int32_t>(qb);
    gacc += prod;
    os << "k=" << k << " qa=" << static_cast<int>(qa)
       << " qb=" << static_cast<int>(qb)
       << " prod=" << prod << " acc_i32=" << gacc << "\n";
  }
  os << "gemm_dequant_without_alpha_beta=" << (static_cast<double>(gacc) * sA * sB) << "\n\n";
}

} // namespace

int main() {
  const int64_t C = 3, H = 4, W = 4, M = 2, kH = 3, kW = 3;
  const int64_t GM = 2, GK = 12, GN = 3;
  const double alpha = 1.0, beta = 0.5;

  std::vector<double> x(static_cast<size_t>(C * H * W));
  std::vector<double> w(static_cast<size_t>(M * C * kH * kW));
  std::vector<double> b(static_cast<size_t>(M));
  std::vector<double> A(static_cast<size_t>(GM * GK));
  std::vector<double> B(static_cast<size_t>(GK * GN));
  std::vector<double> Cmat(static_cast<size_t>(GM * GN));

  for (size_t i = 0; i < x.size(); ++i)
    x[i] = std::sin(0.37 * static_cast<double>(i)) * 2.5 + (static_cast<int>(i % 5) - 2) * 0.2;
  for (size_t i = 0; i < w.size(); ++i)
    w[i] = std::cos(0.29 * static_cast<double>(i)) * 1.8 + (static_cast<int>(i % 3) - 1) * 0.15;
  b[0] = 0.35;
  b[1] = -0.22;

  for (size_t i = 0; i < A.size(); ++i)
    A[i] = std::sin(0.41 * static_cast<double>(i)) * 1.7 + (static_cast<int>(i % 4) - 1) * 0.11;
  for (size_t i = 0; i < B.size(); ++i)
    B[i] = std::cos(0.33 * static_cast<double>(i)) * 1.4 + (static_cast<int>(i % 6) - 2) * 0.09;
  for (size_t i = 0; i < Cmat.size(); ++i)
    Cmat[i] = std::sin(0.19 * static_cast<double>(i)) * 0.6;

  const double sx = 0.125;
  const double sw = 0.09375;
  const double sA = 0.11;
  const double sB = 0.07;

  const std::string inConv = "temp/probe/inputs_conv.txt";
  const std::string inGemm = "temp/probe/inputs_gemm.txt";
  const std::string summary = "temp/probe/results_summary.txt";
  const std::string trace = "temp/probe/trace_steps_p8e0_int8.txt";

  {
    std::ofstream clr(inConv);
  }
  {
    std::ofstream clr(inGemm);
  }
  {
    std::ofstream clr(summary);
  }
  {
    std::ofstream clr(trace);
  }

  dumpVec(inConv, "X(CxHxW=3x4x4)", x);
  dumpVec(inConv, "W(MxCxkHxkW=2x3x3x3)", w);
  dumpVec(inConv, "Bias(M=2)", b);
  dumpVec(inGemm, "A(MxK=2x12)", A);
  dumpVec(inGemm, "B(KxN=12x3)", B);
  dumpVec(inGemm, "C(MxN=2x3)", Cmat);

  auto convRef = conv2d_f32(x, w, b, C, H, W, M, kH, kW);
  auto gemmRef = gemm_f32(A, B, Cmat, GM, GK, GN, alpha, beta);
  auto convI8 = conv2d_int8(x, w, b, C, H, W, M, kH, kW, sx, sw);
  auto gemmI8 = gemm_int8(A, B, Cmat, GM, GK, GN, alpha, beta, sA, sB);

  {
    std::ofstream os(summary, std::ios::app);
    os << std::setprecision(10);
    os << "[step_count]\n";
    os << "conv_each_output_mac=" << (C * kH * kW) << "\n";
    os << "conv_noquire_ops=mac*(1 mul + 1 add) + 1 bias_add\n";
    os << "conv_quire_ops=mac*(1 fdp) + 1 q_to_p + 1 bias_add\n";
    os << "gemm_each_output_mac=" << GK << "\n";
    os << "gemm_noquire_ops=mac*(1 mul + 1 add) + alpha/beta paths\n";
    os << "gemm_quire_ops=mac*(1 fdp) + 1 q_to_p + alpha/beta paths\n\n";

    os << "[int8]\n";
    os << "conv_mae_int8_vs_f32=" << mae(convI8, convRef) << "\n";
    os << "gemm_mae_int8_vs_f32=" << mae(gemmI8, gemmRef) << "\n";
    os << "conv_out_int8=";
    for (double v : convI8)
      os << " " << v;
    os << "\ngemm_out_int8=";
    for (double v : gemmI8)
      os << " " << v;
    os << "\n\n";
  }

  auto ex8 = encodeVec<FmtP8E0>(x);
  auto ew8 = encodeVec<FmtP8E0>(w);
  auto eb8 = encodeVec<FmtP8E0>(b);
  auto eA8 = encodeVec<FmtP8E0>(A);
  auto eB8 = encodeVec<FmtP8E0>(B);
  auto eC8 = encodeVec<FmtP8E0>(Cmat);

  auto conv8NoQ = conv2d<FmtP8E0>(ex8, ew8, eb8, C, H, W, M, kH, kW, false);
  auto conv8Q = conv2d<FmtP8E0>(ex8, ew8, eb8, C, H, W, M, kH, kW, true);
  auto gemm8NoQ = gemm<FmtP8E0>(eA8, eB8, eC8, GM, GK, GN, alpha, beta, false);
  auto gemm8Q = gemm<FmtP8E0>(eA8, eB8, eC8, GM, GK, GN, alpha, beta, true);

  auto ex82 = encodeVec<FmtP8E2>(x);
  auto ew82 = encodeVec<FmtP8E2>(w);
  auto eb82 = encodeVec<FmtP8E2>(b);
  auto eA82 = encodeVec<FmtP8E2>(A);
  auto eB82 = encodeVec<FmtP8E2>(B);
  auto eC82 = encodeVec<FmtP8E2>(Cmat);

  auto conv82NoQ = conv2d<FmtP8E2>(ex82, ew82, eb82, C, H, W, M, kH, kW, false);
  auto conv82Q = conv2d<FmtP8E2>(ex82, ew82, eb82, C, H, W, M, kH, kW, true);
  auto gemm82NoQ = gemm<FmtP8E2>(eA82, eB82, eC82, GM, GK, GN, alpha, beta, false);
  auto gemm82Q = gemm<FmtP8E2>(eA82, eB82, eC82, GM, GK, GN, alpha, beta, true);

  auto ex16 = encodeVec<FmtP16E1>(x);
  auto ew16 = encodeVec<FmtP16E1>(w);
  auto eb16 = encodeVec<FmtP16E1>(b);
  auto eA16 = encodeVec<FmtP16E1>(A);
  auto eB16 = encodeVec<FmtP16E1>(B);
  auto eC16 = encodeVec<FmtP16E1>(Cmat);

  auto conv16NoQ = conv2d<FmtP16E1>(ex16, ew16, eb16, C, H, W, M, kH, kW, false);
  auto conv16Q = conv2d<FmtP16E1>(ex16, ew16, eb16, C, H, W, M, kH, kW, true);
  auto gemm16NoQ = gemm<FmtP16E1>(eA16, eB16, eC16, GM, GK, GN, alpha, beta, false);
  auto gemm16Q = gemm<FmtP16E1>(eA16, eB16, eC16, GM, GK, GN, alpha, beta, true);

  auto ex32 = encodeVec<FmtP32E2>(x);
  auto ew32 = encodeVec<FmtP32E2>(w);
  auto eb32 = encodeVec<FmtP32E2>(b);
  auto eA32 = encodeVec<FmtP32E2>(A);
  auto eB32 = encodeVec<FmtP32E2>(B);
  auto eC32 = encodeVec<FmtP32E2>(Cmat);

  auto conv32NoQ = conv2d<FmtP32E2>(ex32, ew32, eb32, C, H, W, M, kH, kW, false);
  auto conv32Q = conv2d<FmtP32E2>(ex32, ew32, eb32, C, H, W, M, kH, kW, true);
  auto gemm32NoQ = gemm<FmtP32E2>(eA32, eB32, eC32, GM, GK, GN, alpha, beta, false);
  auto gemm32Q = gemm<FmtP32E2>(eA32, eB32, eC32, GM, GK, GN, alpha, beta, true);

  {
    std::ofstream os(summary, std::ios::app);
    os << std::setprecision(10);
    dumpResult<FmtP8E0>(os, FmtP8E0::name(), conv8NoQ, conv8Q, convRef, gemm8NoQ, gemm8Q, gemmRef);
    dumpResult<FmtP8E2>(os, FmtP8E2::name(), conv82NoQ, conv82Q, convRef, gemm82NoQ, gemm82Q, gemmRef);
    dumpResult<FmtP16E1>(os, FmtP16E1::name(), conv16NoQ, conv16Q, convRef, gemm16NoQ, gemm16Q, gemmRef);
    dumpResult<FmtP32E2>(os, FmtP32E2::name(), conv32NoQ, conv32Q, convRef, gemm32NoQ, gemm32Q, gemmRef);
  }

  {
    std::ofstream os(trace, std::ios::app);
    os << std::setprecision(10);
    traceP8Style<FmtP8E0>(os, x, w, A, B, C, H, W, kH, kW, GK);
    traceInt8(os, x, w, A, B, C, H, W, kH, kW, GK, sx, sw, sA, sB);
  }

  std::cout << "Wrote:\n"
            << "  temp/probe/inputs_conv.txt\n"
            << "  temp/probe/inputs_gemm.txt\n"
            << "  temp/probe/results_summary.txt\n"
            << "  temp/probe/trace_steps_p8e0_int8.txt\n";
  return 0;
}
