#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check actual compiler-folded weights against independent numeric references.

Run with ``python3 -m unittest -v test_mixed_precision_weights``. Requires numpy
and onnx-mlir-opt; posit cases also need a C++20 compiler and Universal headers.
Uses the runtime suite's tool discovery but does not require PyRuntimeC,
libcruntime, mlir-translate, or a linked model. No pre-existing audit files or
temporary-directory contents are used. Environment overrides are documented in
test_mixed_precision_runtime.py.
"""

import json
import os
import re
import subprocess
import tempfile
import unittest

from test_mixed_precision_runtime import ENV, np


def _run(command, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, **kwargs)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr[-4000:]}"
        )
    return result


class FoldedWeightTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if np is None:
            raise unittest.SkipTest("numpy not installed")
        cls.opt = os.path.join(ENV.get("bin") or "", "onnx-mlir-opt")
        if not os.path.isfile(cls.opt):
            raise unittest.SkipTest("onnx-mlir-opt not built")
        cls.temp = tempfile.TemporaryDirectory(prefix="mixedprecision-weights-")
        cls.addClassCleanup(cls.temp.cleanup)

    def folded(self, values, fmt, label):
        """Read the exact constant wired to MatMul's weight operand in the IR."""
        values = np.ascontiguousarray(values, dtype=np.float32).reshape(-1)
        count = values.size
        blob = values.tobytes().hex().upper()
        source = f'''module {{
  func.func @weights(%arg0: tensor<1x1xf32>) -> tensor<1x{count}xf32> {{
    %w = onnx.Constant dense<"0x{blob}"> : tensor<1x{count}xf32>
    %r = "onnx.MatMul"(%arg0, %w) {{onnx_node_name = "W"}} : (tensor<1x1xf32>, tensor<1x{count}xf32>) -> tensor<1x{count}xf32>
    return %r : tensor<1x{count}xf32>
  }}
}}
'''
        path = os.path.join(self.temp.name, f"{label}-{fmt}.mlir")
        with open(path, "w") as output:
            output.write(source)
        env = dict(os.environ)
        env["MIXEDP_NODE_FORMATS"] = f"W:w={fmt}"
        result = _run(
            [self.opt, path, "--convert-onnx-to-mixed-precision",
             "--mlir-print-elementsattrs-with-hex-if-larger=0"], env=env)
        operand = re.search(r'"onnx\.MatMul"\(%arg0,\s*(%[\w.]+)\)', result.stdout)
        self.assertIsNotNone(operand, "rewritten MatMul missing from output")
        constant = re.search(
            rf'^\s*{re.escape(operand.group(1))} = onnx\.Constant dense<(.+?)>\s*:',
            result.stdout, re.MULTILINE)
        self.assertIsNotNone(constant, "MatMul does not consume a folded constant")
        payload = constant.group(1)
        if payload.startswith('"0x'):
            actual = np.frombuffer(bytes.fromhex(payload[3:-1]), dtype=np.float32)
        else:
            # MLIR prints splats as scalars even with hex-array printing enabled.
            if payload.startswith("0x"):
                scalar = np.array([int(payload, 16)], dtype=np.uint32).view(np.float32)[0]
            else:
                scalar = np.float32(payload)
            actual = np.full(count, scalar, dtype=np.float32)
        self.assertEqual(actual.size, count, "folded tensor element count changed")
        return actual

    def assertBitsEqual(self, actual, expected, label, nan_equivalent=False):
        actual = np.ascontiguousarray(actual, dtype=np.float32)
        expected = np.ascontiguousarray(expected, dtype=np.float32)
        equal = actual.view(np.uint32) == expected.view(np.uint32)
        if nan_equivalent:
            equal |= np.isnan(actual) & np.isnan(expected)
        bad = np.flatnonzero(~equal)
        examples = [
            f"index {i}: got 0x{actual.view(np.uint32)[i]:08x}, "
            f"expected 0x{expected.view(np.uint32)[i]:08x}" for i in bad[:8]
        ]
        self.assertEqual(bad.size, 0, f"{label}: {bad.size} mismatches; " + "; ".join(examples))


class TestPositWeights(FoldedWeightTest):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        include = ENV.get("universal_inc")
        cxx = ENV.get("clangxx")
        if not include or not os.path.isfile(os.path.join(include, "universal/number/posit/posit.hpp")):
            raise unittest.SkipTest("Universal posit headers not found")
        if not cxx or not os.path.isfile(cxx):
            raise unittest.SkipTest("C++ compiler not found")
        cls.reference = os.path.join(cls.temp.name, "posit_weight_reference")
        helper = os.path.join(os.path.dirname(__file__), "posit_weight_reference.cpp")
        _run([cxx, "-std=c++20", "-O2", f"-I{include}", helper, "-o", cls.reference])

    def check_format(self, bits):
        fmt = f"p{bits}e1"
        input_path = os.path.join(self.temp.name, f"{fmt}-input.bin")
        expected_path = os.path.join(self.temp.name, f"{fmt}-expected.bin")
        counts = json.loads(_run([self.reference, str(bits), input_path, expected_path]).stdout)
        inputs = np.fromfile(input_path, dtype=np.float32)
        expected = np.fromfile(expected_path, dtype=np.float32)
        self.assertEqual(counts["encoded_and_neighbors"], 3 * (1 << bits) - 2)
        self.assertEqual(counts["midpoint_and_neighbors"], 3 * ((1 << bits) - 2))
        self.assertEqual(counts["random_raw_bits"], 65536)
        self.assertEqual(counts["random_in_range"], 65536)
        self.assertEqual(inputs.size, counts["total"])
        self.assertEqual(expected.size, inputs.size)
        actual = self.folded(inputs, fmt, "universal-corpus")
        self.assertBitsEqual(actual, expected, f"{fmt}, {inputs.size} inputs", nan_equivalent=True)

    def test_p4e1_universal_corpus(self):
        self.check_format(4)

    def test_p6e1_universal_corpus(self):
        self.check_format(6)

    def test_p8e1_universal_corpus(self):
        self.check_format(8)


class TestIntegerWeights(FoldedWeightTest):
    def check_all_formats(self, values, expected, label):
        for bits in range(4, 9):
            fmt = f"int{bits}"
            with self.subTest(format=fmt):
                self.assertBitsEqual(self.folded(values, fmt, label), expected, fmt)

    def test_half_away_rounding_all_widths(self):
        # The extrema set scale to exactly 1/4 for every width. Answers below
        # are explicit lattice points, independent of the pass's arithmetic.
        # Both neighbors of each tie distinguish half-away from ties-to-even.
        for bits in range(4, 9):
            with self.subTest(format=f"int{bits}"):
                endpoint = ((1 << (bits - 1)) - 1) / 4
                values = [0, -0.0, 0.125, -0.125, 0.375, -0.375, 0.625, -0.625]
                expected = [0, 0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75]
                for tie, below, above in [(0.125, 0, 0.25), (0.375, 0.25, 0.5)]:
                    for sign in [1, -1]:
                        values += [sign * np.nextafter(np.float32(tie), np.float32(-np.inf)),
                                   sign * np.nextafter(np.float32(tie), np.float32(np.inf))]
                        expected += [sign * below if below else 0, sign * above]
                values += [0.31, -0.31, 0.49, -0.49, endpoint, -endpoint]
                expected += [0.25, -0.25, 0.5, -0.5, endpoint, -endpoint]
                actual = self.folded(values, f"int{bits}", "rounding")
                self.assertBitsEqual(actual, expected, f"int{bits} rounding")

    def test_all_zero(self):
        self.check_all_formats([0, -0.0, 0, -0.0], [0, 0, 0, 0], "zero")

    def test_subnormal_scale_underflow_preserves_weights(self):
        values = np.array([1, 2, 0x80000001, 0x80000002, 0], dtype=np.uint32).view(np.float32)
        self.check_all_formats(values, values, "subnormal")

    def test_max_finite_never_overflows(self):
        maximum = np.finfo(np.float32).max
        values = [maximum, -maximum, 0]
        self.check_all_formats(values, values, "max-finite")

    def test_nonfinite_does_not_change_finite_quantization(self):
        special = np.array([0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345],
                           dtype=np.uint32).view(np.float32)
        for bits in range(4, 9):
            with self.subTest(format=f"int{bits}"):
                endpoint = ((1 << (bits - 1)) - 1) / 4
                values = np.concatenate(([endpoint, -endpoint, 0.125, -0.375], special))
                expected = np.concatenate(([endpoint, -endpoint, 0.25, -0.5], special))
                self.assertBitsEqual(self.folded(values, f"int{bits}", "mixed-nonfinite"),
                                     expected, f"int{bits} mixed nonfinite")

    def test_all_nonfinite_preserved(self):
        values = np.array([0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345],
                          dtype=np.uint32).view(np.float32)
        self.check_all_formats(values, values, "all-nonfinite")


if __name__ == "__main__":
    unittest.main(verbosity=2)
