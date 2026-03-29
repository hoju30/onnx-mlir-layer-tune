extern "C" {
#include "softposit.h"
}

#include <cstdint>
#include <cstdio>
#include "mlir/ExecutionEngine/CRunnerUtils.h"

// MLIR 會生成並呼叫這個 C 介面 wrapper：_mlir_ciface_posit_add_p8e0_4xi8
extern "C" void _mlir_ciface_posit_add_p8e0_4xi8(
    StridedMemRefType<int8_t, 1> *A,
    StridedMemRefType<int8_t, 1> *B,
    StridedMemRefType<int8_t, 1> *Out) {

  int8_t *a = A->data + A->offset;
  int8_t *b = B->data + B->offset;
  int8_t *o = Out->data + Out->offset;

  int64_t n  = A->sizes[0];
  int64_t sa = A->strides[0];
  int64_t sb = B->strides[0];
  int64_t so = Out->strides[0];

  std::printf("[posit_add_p8e0_4xi8] n=%ld\n", (long)n);

  for (int64_t i = 0; i < n; ++i) {
    uint8_t abit = (uint8_t)a[i * sa];
    uint8_t bbit = (uint8_t)b[i * sb];

    posit8_t pa = castP8(abit);
    posit8_t pb = castP8(bbit);

    posit8_t pc = p8_add(pa, pb);

    // 取 raw bits
    uint8_t outBits = (uint8_t)castUI(pc); // SoftPosit README 也用 castUI 
    o[i * so] = (int8_t)outBits;

    // 轉成 double 印出可讀數值（README 有 convertP8ToDouble）
    double outVal = convertP8ToDouble(pc);

    double aVal = convertP8ToDouble(pa);
    double bVal = convertP8ToDouble(pb);
    std::printf("i=%ld A=0x%02X(A=%.10f) B=0x%02X(B=%.10f) Out=0x%02X Out=%.10f\n",
            (long)i, abit, aVal, bbit, bVal, outBits, outVal);
  }

  std::fflush(stdout);
}
