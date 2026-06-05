add_subdirectory(Pattern)
set(SOFTPOSIT_ROOT "${SOFTPOSIT_ROOT}" CACHE PATH
  "Path to the SoftPosit checkout used by posit-related builds.")
if (NOT SOFTPOSIT_ROOT)
  if (EXISTS "${ONNX_MLIR_SRC_ROOT}/src/.deps/SoftPosit")
    set(SOFTPOSIT_ROOT "${ONNX_MLIR_SRC_ROOT}/src/.deps/SoftPosit" CACHE PATH
      "Path to the SoftPosit checkout used by posit-related builds." FORCE)
  elseif (EXISTS "${ONNX_MLIR_SRC_ROOT}/../SoftPosit")
    set(SOFTPOSIT_ROOT "${ONNX_MLIR_SRC_ROOT}/../SoftPosit" CACHE PATH
      "Path to the SoftPosit checkout used by posit-related builds." FORCE)
  endif()
endif()

set(SOFTPOSIT_INCLUDE_DIR "${SOFTPOSIT_ROOT}/source/include")
set(SOFTPOSIT_PLATFORM_INCLUDE_DIR "${SOFTPOSIT_ROOT}/build/Linux-x86_64-GCC")
if (NOT EXISTS "${SOFTPOSIT_PLATFORM_INCLUDE_DIR}")
  set(SOFTPOSIT_PLATFORM_INCLUDE_DIR "${SOFTPOSIT_ROOT}/build/Linux_x86_64_GCC")
endif()

set(UNIVERSAL_ROOT "${UNIVERSAL_ROOT}" CACHE PATH
  "Path to the Universal posit headers used by ONNXToPosit compile-time constant encoding.")
if (NOT UNIVERSAL_ROOT)
  if (EXISTS "${ONNX_MLIR_SRC_ROOT}/src/.deps/universal")
    set(UNIVERSAL_ROOT "${ONNX_MLIR_SRC_ROOT}/src/.deps/universal" CACHE PATH
      "Path to the Universal posit headers used by ONNXToPosit compile-time constant encoding." FORCE)
  elseif (EXISTS "${ONNX_MLIR_SRC_ROOT}/../universal")
    set(UNIVERSAL_ROOT "${ONNX_MLIR_SRC_ROOT}/../universal" CACHE PATH
      "Path to the Universal posit headers used by ONNXToPosit compile-time constant encoding." FORCE)
  endif()
endif()

set(UNIVERSAL_INCLUDE_DIR "${UNIVERSAL_ROOT}/include/sw")

add_onnx_mlir_library(OMONNXToPosit
  ONNXToPosit.cpp
  TypeConverters.cpp
  Pattern/Math.cpp

  LINK_LIBS PUBLIC
    OMONNXOps          
    MLIRIR
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRPass
    MLIRSupport
    MLIRParser
    MLIRDialectUtils
    MLIRRewrite
    MLIRTransforms
    MLIRFuncTransforms
    MLIRMemRefDialect
    MLIRShapeDialect
    MLIRDialect

    OMPositOps

  DEPENDS
    OMONNXOps
)

target_compile_features(OMONNXToPosit PRIVATE cxx_std_20)

target_include_directories(OMONNXToPosit PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_BIN_ROOT}/include
  ${SOFTPOSIT_INCLUDE_DIR}
  ${SOFTPOSIT_PLATFORM_INCLUDE_DIR}
  ${UNIVERSAL_INCLUDE_DIR}
)
