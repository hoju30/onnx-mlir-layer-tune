add_subdirectory(Pattern)
set(SOFTPOSIT_ROOT "/home/lai/mlir_toy/SoftPosit/SoftPosit")

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

target_include_directories(OMONNXToPosit PRIVATE
  ${ONNX_MLIR_SRC_ROOT}/include
  ${ONNX_MLIR_BIN_ROOT}/include
  ${SOFTPOSIT_ROOT}/source/include
  ${SOFTPOSIT_ROOT}/build/Linux-x86_64-GCC
)
