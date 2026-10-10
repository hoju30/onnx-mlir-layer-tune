# Dialect defs
# 指定 Dialect 的 .td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/PositDialect.td)
mlir_tablegen(PositDialect.h.inc  -gen-dialect-decls -dialect=posit)
mlir_tablegen(PositDialect.cpp.inc -gen-dialect-defs  -dialect=posit)
add_public_tablegen_target(MLIRPositDialectIncGen)

# Types defs
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/PositTypes.td)
mlir_tablegen(PositTypes.h.inc  -gen-typedef-decls)
mlir_tablegen(PositTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRPositTypesIncGen)

# Ops defs
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/PositOps.td)
mlir_tablegen(PositOps.h.inc   -gen-op-decls)
mlir_tablegen(PositOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRPositOpsIncGen)

# 讓上面產物成為可依賴的目標
add_public_tablegen_target(OMPositOpsIncGen)

# 方言庫
add_onnx_mlir_library(OMPositOps
  PositDialect.cpp
  PositOps.cpp

  DEPENDS
        OMPositOpsIncGen

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
    ${ONNX_MLIR_BIN_ROOT}/include

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
)

# Exact posit<N,ES> rounding LUT (PositLut.hpp), built with the header-only
# Universal posit template as its oracle -- the same UNIVERSAL_ROOT
# resolution src/Conversion/ONNXToPosit and ONNXToMixedPrecision use. Kept
# out of OMPositOps so the dialect itself does not depend on Universal.
set(UNIVERSAL_ROOT "${UNIVERSAL_ROOT}" CACHE PATH
  "Path to the Universal posit headers used by ONNXToPosit/ONNXToMixedPrecision compile-time constant encoding.")
if (NOT UNIVERSAL_ROOT)
  if (EXISTS "${ONNX_MLIR_SRC_ROOT}/src/.deps/universal")
    set(UNIVERSAL_ROOT "${ONNX_MLIR_SRC_ROOT}/src/.deps/universal" CACHE PATH
      "Path to the Universal posit headers used by ONNXToPosit/ONNXToMixedPrecision compile-time constant encoding." FORCE)
  elseif (EXISTS "${ONNX_MLIR_SRC_ROOT}/../universal")
    set(UNIVERSAL_ROOT "${ONNX_MLIR_SRC_ROOT}/../universal" CACHE PATH
      "Path to the Universal posit headers used by ONNXToPosit/ONNXToMixedPrecision compile-time constant encoding." FORCE)
  endif()
endif()

add_onnx_mlir_library(OMPositLut
  PositLut.cpp

  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
)

target_compile_features(OMPositLut PRIVATE cxx_std_20)
target_include_directories(OMPositLut PRIVATE ${UNIVERSAL_ROOT}/include/sw)
