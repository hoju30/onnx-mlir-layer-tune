#pragma once

#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/MLIRContext.h"

namespace onnx_mlir {

// Storage bit-width for posit raw payload in memrefs/runtime ABI.
// Keep posit format as (nbits, es), but use byte-aligned integers for storage.
// Example: p6e0 uses i8 payload and only low 6 bits are significant.
inline unsigned getPositStorageBitWidth(unsigned nbits) {
  if (nbits <= 8)
    return 8;
  if (nbits <= 16)
    return 16;
  if (nbits <= 32)
    return 32;
  if (nbits <= 64)
    return 64;
  return nbits;
}

struct PositToKrnlTypeConverter : public mlir::TypeConverter {
  PositToKrnlTypeConverter(unsigned nbits, unsigned es, mlir::MLIRContext *ctx);

  unsigned getNbits() const { return nbits_; }
  unsigned getEs() const { return es_; }

private:
  unsigned nbits_;
  unsigned es_;
};

} // namespace onnx_mlir
