#pragma once
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"




namespace onnx_mlir {


// 型別轉換器
struct PositTypeConverter : public mlir::TypeConverter {
  PositTypeConverter(unsigned nbits, unsigned es, mlir::MLIRContext *ctx);
};

} // namespace onnx_mlir