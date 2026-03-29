#include "src/Dialect/Posit/PositOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::posit;

//  這行把 ODS 產生的東西編進來
#define GET_OP_CLASSES
#include "src/Dialect/Posit/PositOps.cpp.inc"