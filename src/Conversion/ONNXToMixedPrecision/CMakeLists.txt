# Weight/activation-separated QDQ mixed-precision pass. Needs the Posit
# dialect (for the posit.from_f32/to_f32 activation QDQ path) and the
# header-only Universal posit<N,ES> template (for the LUT-accelerated
# compile-time weight QDQ path, N<=8) -- same dependencies
# src/Conversion/ONNXToPosit/CMakeLists.txt already resolves, reused here
# rather than re-deriving the search logic.
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

set(UNIVERSAL_INCLUDE_DIR "${UNIVERSAL_ROOT}/include/sw")

add_onnx_mlir_library(OMONNXToMixedPrecision
  ONNXToMixedPrecision.cpp

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
    OMPositOps
)

target_compile_features(OMONNXToMixedPrecision PRIVATE cxx_std_20)

target_include_directories(OMONNXToMixedPrecision PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_BIN_ROOT}/include
  ${UNIVERSAL_INCLUDE_DIR}
)
