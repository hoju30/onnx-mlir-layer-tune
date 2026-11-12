"""Shared helpers of the Qwen3-0.6B SwiGLU gate/up criterion study (qwen3_swiglu_*.py).

PyTorch side (offline references only): the HF model in the same eval / eager / constant-causal-mask form that
qwen3_prepare.py exported, hooks on the 196 projections, and quantizers that reproduce the mixed-precision pass's
arithmetic exactly:
  int activations  Div(x, scale) -> Add(zp) -> Round (half to even) -> Clip -> Sub(zp) -> Mul(scale), float32,
                   (scale, zp) = eval_random_mixed_precision.int_qparams of the native calibration range
  int weights      per output channel, symmetric, std::lround (run_w8_per_channel_check.per_channel_int)
  posit            Universal's float -> posit<N,1> -> float, as a table of decision boundaries found by bisection
                   on Universal itself (PositLut), checked against Universal on a corpus
  f16 / bf16       casts
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import eval_random_mixed_precision as erm  # noqa: E402
from run_w8_per_channel_check import per_channel_int  # noqa: E402

PREP = HERE / "runs/qwen3_prepare_20261011"
MODEL_DIR = HERE / "qwen3_assets/model"
VOCAB = 151936


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def environment(extra_files=()):
    """Code versions and hashes of everything an experiment depends on."""
    root = HERE.parent.parent
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True).stdout.strip()
    files = ["experiments/mixed_precision/qwen3_prepare.py", "experiments/mixed_precision/qwen3_grad_check.py",
             "experiments/mixed_precision/qwen3_swiglu_common.py", "experiments/mixed_precision/run_native_bc_pipeline.py",
             "experiments/mixed_precision/run_native_auto.py", "experiments/mixed_precision/eval_random_mixed_precision.py",
             "src/Conversion/ONNXToMixedPrecision/MixedPrecisionBackward.cpp",
             "src/Conversion/ONNXToMixedPrecision/ONNXToMixedPrecision.cpp", *extra_files]
    tools = erm.Tools()
    import transformers
    return {"git_head": git("rev-parse", "HEAD"), "git_dirty": git("status", "--short", "--untracked-files=no"),
            "files": {f: sha256(root / f) for f in files if (root / f).exists()},
            "onnx_mlir_opt": {"sha256": sha256(tools.opt), "mtime": Path(tools.opt).stat().st_mtime},
            "model_onnx": sha256(PREP / "model.onnx"), "model_onnx_data": sha256(PREP / "model.onnx.data"),
            "validation_tokens": sha256(PREP / "validation_tokens.npy"),
            "checkpoint": json.loads((HERE / "qwen3_assets/provenance.json").read_text()),
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__}


def node_mapping():
    """ONNX node -> (layer, kind, parameter), kind in q/k/v/o/gate/up/down."""
    out = {}
    for node, param in json.loads((PREP / "node_mapping.json").read_text()).items():
        parts = param.split(".")
        out[node] = (int(parts[2]), parts[4].removesuffix("_proj"), param)
    return out


def load_model(dtype=torch.float32):
    """The HF model and the exported wrapper (constant causal mask), eval mode, eager attention."""
    from transformers import Qwen3ForCausalLM
    from qwen3_prepare import CausalLM
    model = Qwen3ForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32, attn_implementation="eager").eval()
    model.config.use_cache = False
    wrapper = CausalLM(model, 128).eval().to(dtype)
    return model, wrapper


def summed_ce(z, ids):
    """Summed next-token cross-entropy over positions 0..L-2 (the native seed: (softmax - onehot) x row mask)."""
    return torch.nn.functional.cross_entropy(z[0, :-1], ids[1:], reduction="sum")


# --------------------------------------------------------------------------- #
# Quantizers matching the mixed-precision pass
# --------------------------------------------------------------------------- #
def int_act_qdq(x, scale, zp, bits):
    # scale / zero-point on x's device: CUDA divides by a 0-dim CPU tensor as a multiply by its reciprocal, which
    # differs from the CPU's (and QuantizeLinear's) true division in the last bit (runs/olmoe_search_final_20261016)
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    s = torch.tensor(np.float32(scale), dtype=x.dtype, device=x.device)
    z = torch.tensor(np.float32(zp), dtype=x.dtype, device=x.device)
    q = torch.clamp(torch.round(x / s + z), qmin, qmax)
    return (q - z) * s


def _order_key_np(x):
    b = np.ascontiguousarray(x, np.float32).view(np.uint32).astype(np.int64)
    return np.where(b & 0x80000000, 0xFFFFFFFF - b, b | 0x80000000)


def _order_key_torch(x):
    b = x.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return torch.where((b & 0x80000000) != 0, 0xFFFFFFFF - b, b | 0x80000000)


def _from_key_np(k):
    k = np.asarray(k, np.int64)
    b = np.where(k & 0x80000000, k & 0x7FFFFFFF, 0xFFFFFFFF - k).astype(np.uint32)
    return b.view(np.float32)


class PositLut:
    """Universal's float32 -> posit<N,ES> -> float32 as sorted values + decision boundaries (order keys): the value
    of x is values[searchsorted(boundaries, key(x), 'right')]. Built by bisection over float order keys with
    Universal as the oracle, so it reproduces Universal for every finite float (verify() checks a corpus)."""

    def __init__(self, oracle, fmt):
        self.oracle, self.fmt = oracle, fmt
        probe = np.concatenate([np.logspace(-40, 38, 400001, dtype=np.float64), [0.0]]).astype(np.float32)
        probe = np.concatenate([-probe, probe])
        vals = np.unique(oracle(probe, fmt))
        vals = vals[np.isfinite(vals)]
        self.values = np.sort(vals).astype(np.float32)
        lo, hi = _order_key_np(self.values[:-1]), _order_key_np(self.values[1:])
        # smallest key mapping to values[i+1], between key(values[i]) (maps to values[i]) and key(values[i+1])
        while np.any(hi - lo > 1):
            mid = (lo + hi) // 2
            up = oracle(_from_key_np(mid), fmt) >= self.values[1:]
            hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
        self.bounds = hi
        self.t_values = torch.from_numpy(self.values)
        self.t_bounds = torch.from_numpy(self.bounds)

    def __call__(self, x):
        idx = torch.searchsorted(self.t_bounds, _order_key_torch(x.float()), right=True)
        return self.t_values[idx].to(x.dtype)

    def verify(self, n=2_000_000, seed=0):
        rng = np.random.default_rng(seed)
        x = np.concatenate([rng.standard_normal(n // 2) * 10.0 ** rng.uniform(-6, 4, n // 2),
                            rng.uniform(-300, 300, n // 4), [0.0, -0.0, 1e-38, -1e-38, 3.4e38, -3.4e38],
                            self.values, np.nextafter(self.values, np.inf), np.nextafter(self.values, -np.inf),
                            _from_key_np(self.bounds), _from_key_np(self.bounds - 1)]).astype(np.float32)
        ref = self.oracle(x, self.fmt)
        got = self(torch.from_numpy(x)).numpy()
        return int(np.sum(ref.view(np.uint32) != got.view(np.uint32))), len(x)


class Quantizers:
    """Activation / weight fake-quantization with the pass's semantics."""

    def __init__(self, work, formats=("p4e1", "p6e1", "p8e1")):
        self.tools = erm.Tools()
        self.oracle = erm.PositOracle(self.tools, Path(work))
        self.posit = {f: PositLut(self.oracle, f) for f in formats}

    def act(self, x, fmt, rng=None):
        """x quantized as an activation; rng = (lo, hi) of its native calibration range (int formats)."""
        if fmt == "fp32":
            return x
        if fmt.startswith("int"):
            bits = int(fmt[3:])
            scale, zp = erm.int_qparams(float(rng[0]), float(rng[1]), bits)
            return int_act_qdq(x, scale, zp, bits)
        if fmt == "f16":
            return x.to(torch.float16).to(x.dtype)
        if fmt == "bf16":
            return x.to(torch.bfloat16).to(x.dtype)
        return self.posit[fmt](x)

    def weight(self, w, fmt):
        """torch Linear weight [out, in] quantized per output channel (axis 0), as the pass does on W^T[in, out]."""
        if fmt == "fp32":
            return w
        a = w.detach().cpu().numpy().astype(np.float32)
        if fmt.startswith("int"):
            q = per_channel_int(a, int(fmt[3:]))
        elif fmt == "f16":
            q = a.astype(np.float16).astype(np.float32)
        elif fmt == "bf16":
            q = torch.from_numpy(a).to(torch.bfloat16).to(torch.float32).numpy()
        else:
            q = self.posit[fmt](torch.from_numpy(a)).numpy()
        return torch.from_numpy(q).to(w.dtype)


def compare(a, b):
    """native (or candidate) a against reference b: max-relative (the gate's metric), relative L2, cosine."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    d = np.abs(a - b)
    nb = np.linalg.norm(b)
    return {"rel_max": float(d.max() / max(np.abs(b).max(), 1e-300)), "abs_max": float(d.max()),
            "ref_max": float(np.abs(b).max()), "rel_l2": float(np.linalg.norm(a - b) / max(nb, 1e-300)),
            "cosine": float(a @ b / max(np.linalg.norm(a) * nb, 1e-300))}
