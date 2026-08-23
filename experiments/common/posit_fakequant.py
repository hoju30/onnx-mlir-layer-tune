"""Python ctypes wrapper around libpositfakequant.so -- bit-exact FP32 <->
posit<N,ES> round-trip using the project's actual Universal posit type
(see posit_fakequant.cpp), for computing claude.md 8.5's fake-quant node
features (normalized MSE, SQNR, underflow/zero-after-quant ratios).
"""
import ctypes
import os

import numpy as np

_lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "libpositfakequant.so"))
_lib.posit_roundtrip_f32.argtypes = [
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int, ctypes.c_int,
]
_lib.posit_roundtrip_f32.restype = ctypes.c_int

_lib.posit_decode_bits_f32.argtypes = [
    ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_float),
    ctypes.c_int64, ctypes.c_int, ctypes.c_int,
]
_lib.posit_decode_bits_f32.restype = ctypes.c_int


def roundtrip(arr, nbits, es):
    """FP32 -> posit<nbits,es> -> FP32, elementwise. Returns a new float32 array."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    out = np.empty_like(arr)
    rc = _lib.posit_roundtrip_f32(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        arr.size, nbits, es)
    if rc != 0:
        raise ValueError(f"unsupported posit format: nbits={nbits}, es={es}")
    return out.reshape(arr.shape)


def decode_posit_bits(raw_bytes, n, nbits, es):
    """Decode already-encoded posit<nbits,es> bit patterns (raw storage bytes,
    e.g. from a posit.constant's dense<...> attribute) back to FP32 -- the
    reverse of roundtrip(): there is no original FP32 value here, only bits."""
    raw = np.frombuffer(raw_bytes, dtype=np.uint8)
    out = np.empty(n, dtype=np.float32)
    rc = _lib.posit_decode_bits_f32(
        raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        n, nbits, es)
    if rc != 0:
        raise ValueError(f"unsupported posit format: nbits={nbits}, es={es}")
    return out


def fakequant_metrics(x, nbits, es, is_fp32):
    """claude.md 8.5 fake-quant features for array x at the given format."""
    x = np.asarray(x, dtype=np.float64)
    if is_fp32 or x.size == 0:
        return {
            "fakequant_mse": 0.0, "fakequant_normalized_mse": 0.0, "fakequant_sqnr_db": None,
            "fakequant_underflow_ratio": 0.0,
            "fakequant_zero_after_quant_ratio": float((x == 0).mean()) if x.size else 0.0,
        }
    x_hat = roundtrip(x, nbits, es).astype(np.float64)
    err = x - x_hat
    mse = float(np.mean(err ** 2))
    signal_power = float(np.mean(x ** 2))
    nonzero = x != 0
    underflow = float((nonzero & (x_hat == 0)).sum() / max(nonzero.sum(), 1))
    return {
        "fakequant_mse": mse,
        "fakequant_normalized_mse": mse / signal_power if signal_power > 0 else 0.0,
        "fakequant_sqnr_db": (10.0 * np.log10(signal_power / mse)) if mse > 0 and signal_power > 0 else None,
        "fakequant_underflow_ratio": underflow,
        "fakequant_zero_after_quant_ratio": float((x_hat == 0).mean()),
    }
