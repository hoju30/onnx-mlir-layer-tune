# SPDX-License-Identifier: Apache-2.0

# These IR tests also run in the normal check-onnx-lit suite. A focused target
# keeps it inexpensive to validate changes to the mixed-precision pass.
add_lit_testsuite(check-onnx-mixed-precision-lit
  "Running mixed-precision IR regression tests"
  ${ONNX_MLIR_BIN_ROOT}/test/mlir/conversion/onnx_to_mixed_precision
  DEPENDS onnx-mlir-opt FileCheck
  )
set_target_properties(check-onnx-mixed-precision-lit PROPERTIES FOLDER "Tests")

# Runtime fixtures emit the Linux x86-64 data layout and execute locally.
# Do not register them with CTest on other hosts or in cross builds.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux" AND
    CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND
    NOT CMAKE_CROSSCOMPILING)
  add_test(NAME MixedPrecisionNumerical
    COMMAND ${CMAKE_COMMAND} -E env
      "ONNX_MLIR_ROOT=${ONNX_MLIR_SRC_ROOT}"
      "ONNX_MLIR_BUILD=${ONNX_MLIR_BIN_ROOT}"
      "LLVM_BUILD_BIN=${LLVM_TOOLS_BINARY_DIR}"
      "UNIVERSAL_ROOT=${UNIVERSAL_ROOT}"
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
  set_tests_properties(MixedPrecisionNumerical PROPERTIES
    LABELS "mixed-precision"
    TIMEOUT 600
    )

  add_custom_target(check-onnx-mixed-precision-numerical
    COMMENT "Running mixed-precision numerical tests (skips are failures)"
    COMMAND ${CMAKE_CTEST_COMMAND} -R "^MixedPrecisionNumerical$"
      --output-on-failure --no-tests=error -C $<CONFIG>
    WORKING_DIRECTORY ${ONNX_MLIR_BIN_ROOT}
    DEPENDS onnx-mlir onnx-mlir-opt PyRuntimeC cruntime
    USES_TERMINAL
    )
  set(MIXED_PRECISION_RUNTIME_SUPPORTED TRUE)
else()
  # An explicit request must fail instead of succeeding with zero tests.
  add_custom_target(check-onnx-mixed-precision-numerical
    COMMAND ${CMAKE_COMMAND} -E echo
      "Mixed-precision runtime tests require native Linux x86-64."
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM
    )
endif()
set_target_properties(check-onnx-mixed-precision-numerical PROPERTIES FOLDER "Tests")

add_custom_target(check-onnx-mixed-precision
  DEPENDS check-onnx-mixed-precision-lit check-onnx-mixed-precision-numerical
  )
set_target_properties(check-onnx-mixed-precision PROPERTIES FOLDER "Tests")

if (MIXED_PRECISION_RUNTIME_SUPPORTED)
  add_dependencies(check-onnx-backend-numerical check-onnx-mixed-precision)
endif()
