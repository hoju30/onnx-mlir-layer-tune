#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

namespace onnx_mlir {

std::unique_ptr<mlir::Pass> createConvertPositToKrnlPass();
std::unique_ptr<mlir::Pass> createConvertPositToKrnlPass(unsigned nbits,
                                                         unsigned es);

} // namespace onnx_mlir