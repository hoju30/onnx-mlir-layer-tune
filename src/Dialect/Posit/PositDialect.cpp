#include "llvm/ADT/TypeSwitch.h"      
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h" //op need fail
#include "mlir/IR/BuiltinTypes.h"  //op need fail
#include "llvm/Support/Casting.h"  //op need

#include "src/Dialect/Posit/PositDialect.h"
#include "mlir/IR/DialectImplementation.h"

#include "src/Dialect/Posit/PositOps.h"     

#define GET_TYPEDEF_CLASSES
#include "src/Dialect/Posit/PositTypes.cpp.inc"

#include "src/Dialect/Posit/PositDialect.cpp.inc"


using namespace mlir;
using namespace mlir::posit;

LogicalResult PositType::verify(function_ref<InFlightDiagnostic()> emitError,
                                unsigned nbits, unsigned es) {
  if (nbits == 0)
    return emitError() << "nbits must be > 0";
  if (es >= nbits)
    return emitError() << "es (" << es << ") must be < nbits (" << nbits << ")";
  return success();
}

void PositDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "src/Dialect/Posit/PositTypes.cpp.inc"
  >();

  addOperations<
#define GET_OP_LIST
#include "src/Dialect/Posit/PositOps.cpp.inc"
  >();
}

Operation *PositDialect::materializeConstant(OpBuilder &builder,
                                             Attribute value, 
                                             Type type,
                                             Location loc) {
  // 只接受 posit scalar 型別 + i64屬性
  if (!llvm::isa<mlir::posit::PositType>(type)) return nullptr;
  if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(value))
    return builder.create<mlir::posit::ConstantOp>(loc, type, intAttr);

  return nullptr;
}