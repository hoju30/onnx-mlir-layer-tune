add_onnx_mlir_library(OMONNXToLowPrecision
  ONNXToLowPrecision.cpp

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

  DEPENDS
    OMONNXOps
)

target_compile_features(OMONNXToLowPrecision PRIVATE cxx_std_20)

target_include_directories(OMONNXToLowPrecision PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_BIN_ROOT}/include
)
