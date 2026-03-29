

#ifndef POSIT_POSITOPS_H
#define POSIT_POSITOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "mlir/IR/OpImplementation.h"
#include "src/Dialect/Posit/PositDialect.h"
#include "src/Dialect/Posit/PositTypes.h"

#define GET_OP_CLASSES
#include "src/Dialect/Posit/PositOps.h.inc"

#endif // POSIT_POSITOPS_H