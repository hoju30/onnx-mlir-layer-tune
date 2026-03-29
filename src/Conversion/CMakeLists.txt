# SPDX-License-Identifier: Apache-2.0

add_subdirectory(ONNXConversionCommon)
add_subdirectory(ONNXToKrnl)
add_subdirectory(KrnlToLLVM)
add_subdirectory(KrnlToAffine)
add_subdirectory(KrnlSeqToMemref)
add_subdirectory(ONNXToTOSA)
# 11/13
add_subdirectory(ONNXToPosit)
# 12/17 posit to krnl
add_subdirectory(PositToKrnl)

if (ONNX_MLIR_ENABLE_STABLEHLO)
  add_subdirectory(ONNXToStablehlo)
endif()
