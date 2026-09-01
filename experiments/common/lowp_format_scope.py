"""Single source of truth for the low-precision candidate format set
(claude_lowprecision.md section 1/8.4), shared by the config generator,
calibration, graph/feature extraction, and the cost calculator so none of
them can drift out of sync with docs/LowPrecisionFormats.md.

Per-op scope mirrors docs/LowPrecisionFormats.md's "Per-op scope" table:
Conv/Gemm/MatMul accept all 5 non-FP32 formats; Relu/Add/Sub/Mul/Div only
accept bf16/f16 (int8/fp8 aren't implemented for them). Any op type not
listed here (e.g. MaxPool, Reshape, Softmax in the MNIST demo net) has no
low-precision candidates at all and stays FP32 in every configuration.
"""

FORMATS = ["FP32", "bf16", "f16", "int8", "fp8e4m3", "fp8e5m2"]

# bits, is_float_format (0 = linear fixed-point int8, 1 = float-shaped),
# requires_calibration (needs external activation scale/zero-point)
FORMAT_INFO = {
    "FP32":    {"bitwidth": 32, "is_float_format": 1, "requires_calibration": 0},
    "bf16":    {"bitwidth": 16, "is_float_format": 1, "requires_calibration": 0},
    "f16":     {"bitwidth": 16, "is_float_format": 1, "requires_calibration": 0},
    "int8":    {"bitwidth": 8,  "is_float_format": 0, "requires_calibration": 1},
    "fp8e4m3": {"bitwidth": 8,  "is_float_format": 1, "requires_calibration": 1},
    "fp8e5m2": {"bitwidth": 8,  "is_float_format": 1, "requires_calibration": 1},
}

# Symmetric-quant max representable magnitude, used to derive scale = max_abs
# / QMAX and zero_point = 0 for calibration (same convention
# docs/LowPrecisionFormats.md describes for automatic weight/bias
# quantization -- reused here for activations since the pass itself does no
# calibration).
QMAX = {
    "int8": 127.0,
    "fp8e4m3": 448.0,     # max finite magnitude of float8_e4m3fn
    "fp8e5m2": 57344.0,   # max finite magnitude of float8_e5m2
}

FULL_SCOPE_OPS = {"Conv", "Gemm", "MatMul"}              # all 5 non-FP32 formats
NARROW_SCOPE_OPS = {"Relu", "Add", "Sub", "Mul", "Div"}   # bf16/f16 only


def candidate_formats(op_type):
    """Formats legal for a node of this op_type, including FP32."""
    if op_type in FULL_SCOPE_OPS:
        return list(FORMATS)
    if op_type in NARROW_SCOPE_OPS:
        return ["FP32", "bf16", "f16"]
    return ["FP32"]


def is_tunable(op_type):
    return len(candidate_formats(op_type)) > 1
