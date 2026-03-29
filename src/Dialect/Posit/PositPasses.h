#ifndef MLIR_POSIT_PASSES_H
#define MLIR_POSIT_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace posit {
std::unique_ptr<mlir::Pass> createLowerPositToArithPass();
} // namespace posit
} // namespace mlir

#endif // MLIR_POSIT_PASSES_H

//namespace mlir { void registerPositPasses(); }

// #define GEN_PASS_DECL
// #include "posit/PositPasses.h.inc"
