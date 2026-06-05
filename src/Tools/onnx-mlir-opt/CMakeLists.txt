# SPDX-License-Identifier: Apache-2.0

get_property(OMLibs GLOBAL PROPERTY ONNX_MLIR_LIBS)

set(SOFTPOSIT_LIBRARY "${SOFTPOSIT_LIBRARY}" CACHE FILEPATH
  "Path to libsoftposit used by posit-enabled tooling.")
if (NOT SOFTPOSIT_LIBRARY)
  find_library(SOFTPOSIT_LIBRARY
    NAMES softposit
    HINTS
      "${ONNX_MLIR_SRC_ROOT}/src/.deps/softposit-px1"
      "${ONNX_MLIR_SRC_ROOT}/../SoftPosit/build/Linux-x86_64-GCC"
      "${ONNX_MLIR_SRC_ROOT}/../SoftPosit/build/Linux_x86_64_GCC"
      "${SOFTPOSIT_ROOT}/build/Linux-x86_64-GCC"
      "${SOFTPOSIT_ROOT}/build/Linux_x86_64_GCC"
  )
endif()
if (NOT SOFTPOSIT_LIBRARY)
  message(FATAL_ERROR
    "softposit library not found. Set SOFTPOSIT_LIBRARY or install dependencies "
    "under src/.deps with src/bash/install_posit_deps.sh.")
endif()
if (NOT TARGET softposit)
  add_library(softposit STATIC IMPORTED GLOBAL)
  set_target_properties(softposit PROPERTIES
    IMPORTED_LOCATION "${SOFTPOSIT_LIBRARY}")
endif()

add_onnx_mlir_executable(onnx-mlir-opt
  onnx-mlir-opt.cpp
  RegisterPasses.cpp

  LINK_LIBS PRIVATE
  ${OMLibs}
  OMCompilerDialects
  OMCompilerOptions
  OMCompilerPasses
  OMAccelerator
  OMVersion
  MLIRAffineTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIROpenMPToLLVM
  MLIROptLib
  MLIRSCFToOpenMP

  )

# 11/13
target_link_libraries(onnx-mlir-opt PUBLIC
  OMONNXToPosit      # ONNX to Posit pass
  OMPositOps         # Posit dialect
  OMPositToKrnl      # Posit to Krnl pass
  softposit          # softposit
)

message(STATUS "test run this cmake")
