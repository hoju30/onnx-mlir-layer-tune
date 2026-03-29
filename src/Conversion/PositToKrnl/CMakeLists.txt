add_subdirectory(Pattern)

add_onnx_mlir_library(OMPositToKrnl
  PositToKrnl.cpp
  TypeConverters.cpp
  # Pattern/Math.cpp

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

target_include_directories(OMONNXToPosit PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_BIN_ROOT}/include
)