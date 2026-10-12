#!/usr/bin/env python3
"""Run a real ImageNet-1k classifier through --convert-onnx-to-mixed-precision
with a RANDOM per-layer weight/activation format assignment, and report the
accuracy.

What it does (all steps cached under --out-dir / --work-dir):
  1. Sample N ImageNet-1k *validation* images (default 100) from a parquet shard
     of an ImageNet-1k dataset and pick K of them (default 10) as the
     calibration set.
  2. Preprocess with the model's own timm config (metadata.json pretrained_cfg).
  3. Calibrate: run the FP32 ONNX in onnxruntime with every activation tensor of
     the pass-supported nodes exposed, take min/max over the K calibration
     images, and turn them into asymmetric (affine) scale/zero-point for the
     int4..int8 activation formats.
  4. Draw a random format for every supported node (weight and activation
     independently; seeded) -> MIXEDP_NODE_FORMATS.
  5. Compile: onnx-mlir --EmitONNXBasic -> --convert-onnx-to-mixed-precision ->
     the default ONNX-level pipeline -> Krnl -> LLVM -> .so, then run all N images
     through the .so (each .so in its own process: a hung or crashing model cannot take the driver down).
  6. Report top-1/top-5 for: PyTorch(timm) FP32, onnxruntime FP32, onnx-mlir FP32,
     onnx-mlir + pass with an all-fp32 config (identity control), and the random
     mixed-precision config(s).

The pass must run on the ORIGINAL graph: a plain `onnx-mlir --EmitONNXIR` first
rewrites every 1x1 Conv into Reshape/MatMul/Add and renames the nodes, so the
node names in MIXEDP_NODE_FORMATS (and the calibration, which is per ONNX node)
would no longer match. Hence --EmitONNXBasic, our pass, and only then the
ONNX-level passes that --EmitONNXIR would have run (same list, same order as
addONNXToMLIRPasses in src/Compiler/CompilerPasses.cpp, hybrid pass enabled).

Needs: numpy, onnx, onnxruntime, Pillow, pyarrow, and (for the PyTorch reference
and the preprocessing) torch + timm. A built onnx-mlir (onnx-mlir, onnx-mlir-opt,
PyRuntimeC, libcruntime.a) plus the LLVM build it was configured against
(mlir-translate, clang++) and the Universal headers (posit runtime).

Usage:
  python experiments/mixed_precision/eval_random_mixed_precision.py \
      --parquet validation-00000-of-00014.parquet --seeds 0
  # or fetch one validation shard (~470MB) from an ungated HuggingFace mirror:
  python experiments/mixed_precision/eval_random_mixed_precision.py --download-shard
"""
import argparse
import glob
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("ONNX_MLIR_ROOT") or HERE.parents[1])
DEFAULT_MODEL_DIR = ROOT / "experiments/imagenet100/model_dataset/mobilenetv2/timm_ra"

# Candidate formats: the set docs/MixedPrecisionFormats.md is about.
FORMATS = ["int4", "int5", "int6", "int7", "int8", "p4e1", "p6e1", "p8e1", "f16", "bf16", "fp32"]
POSIT_FORMATS = ("p4e1", "p6e1", "p8e1")     # the only posit formats the pass can quantize weights for
WEIGHT_OPS = ("Conv", "Gemm", "MatMul")      # operand 1 is the weight
UNARY_OPS = WEIGHT_OPS + ("Relu", "AveragePool", "GlobalAveragePool", "MaxPool")
BINARY_OPS = ("Add", "Sub", "Mul", "Div")
SUPPORTED_OPS = UNARY_OPS + BINARY_OPS       # == isSupportedMixedOp() in ONNXToMixedPrecision.cpp

# Same order as addONNXToMLIRPasses() (hybrid pass on, conv-opt on), i.e. what
# `onnx-mlir --EmitONNXIR` runs after the import.
ONNX_LEVEL_PASSES = [
    "--shape-inference", "--decompose-onnx", "--recompose-onnx", "--onnx-hybrid-transform",
    "--conv-opt-onnx", "--onnx-hybrid-transform", "--simplify-shape-related-ops-onnx",
    "--onnx-hybrid-transform", "--standard-func-return", "--symbol-dce", "--scrub-disposable",
    "--set-onnx-node-name",
]
# addKrnlToAffinePasses + addKrnlToLLVMPasses. The buffer-deallocation part is not optional: the shorter
# `--convert-krnl-to-affine --convert-krnl-to-llvm` recipe of lowp_lower_general.py emits no dealloc at all, so every
# inference leaks all of its intermediate buffers (64 MB per MobileNetV2 image, ~130 MB with QDQ chains).
KRNL_TO_LLVM_PASSES = [
    "--convert-krnl-to-affine", "--convert-vector-to-scf", "--lower-affine", "--lower-krnl-region",
    "--buffer-loop-hoisting", "--buffer-deallocation-pipeline", "--optimize-allocation-liveness",
    "--fold-memref-alias-ops", "--convert-vector-to-llvm", "--convert-krnl-to-llvm",
    "--reconcile-unrealized-casts", "--canonicalize",
]

def _cpu_has_f16c():
    try:
        with open("/proc/cpuinfo") as f:
            return any(" f16c" in line for line in f if line.startswith("flags"))
    except OSError:
        return False


# f16 activations are runtime f32->f16->f32 conversions. On a baseline x86-64 target LLVM turns them into calls to
# __truncsfhf2/__extendhfsf2, and libcruntime.a (src/Runtime/OMTensor.inc) defines those with the OLD ABI
# (`uint16_t __truncsfhf2(float)`, `float __extendhfsf2(uint16_t)`) while current clang passes/returns `half` in xmm0.
# The mismatch reads stale registers: results are garbage and intermittent (a fresh session can be right or wrong).
# Compiling with F16C makes LLVM emit vcvtps2ph/vcvtph2ps inline, so no libcall is involved.
HAS_F16C = _cpu_has_f16c()
SYMBOL_POSTFIX = "model"                     # <work>/model.onnx.mlir -> "onnx-mlir.symbol-postfix" = "model"
HF_MIRROR = ("benjamin-paine/imagenet-1k", "data/validation-00000-of-00014.parquet")


def log(msg=""):
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Toolchain
# --------------------------------------------------------------------------- #
def _cache_var(build, name):
    try:
        with open(build / "CMakeCache.txt") as f:
            for line in f:
                m = re.match(rf"^{name}:[A-Z]+=(.*)$", line.strip())
                if m:
                    return m.group(1)
    except OSError:
        pass
    return None


class Tools:
    def __init__(self):
        build = Path(os.environ.get("ONNX_MLIR_BUILD") or ROOT / "build")
        bindirs = [Path(b) for b in glob.glob(str(build / "*" / "bin")) if (Path(b) / "onnx-mlir-opt").exists()]
        if not bindirs:
            sys.exit(f"no built onnx-mlir-opt under {build} (set ONNX_MLIR_BUILD)")
        self.bin, self.lib = bindirs[0], bindirs[0].parent / "lib"
        self.onnx_mlir, self.opt = self.bin / "onnx-mlir", self.bin / "onnx-mlir-opt"
        llvm_dir, mlir_dir = _cache_var(build, "LLVM_DIR"), _cache_var(build, "MLIR_DIR")
        llvm_bin = Path(os.environ["LLVM_BUILD_BIN"]) if os.environ.get("LLVM_BUILD_BIN") else (
            Path(llvm_dir).parents[2] / "bin" if llvm_dir else None)
        self.translate = llvm_bin / "mlir-translate" if llvm_bin else None
        self.clangxx = llvm_bin / "clang++" if llvm_bin else None
        self.mlir_inc = Path(mlir_dir).parents[3] / "mlir" / "include" if mlir_dir else None
        uni = os.environ.get("UNIVERSAL_ROOT") or _cache_var(build, "UNIVERSAL_ROOT")
        self.universal_inc = Path(uni) / "include" / "sw" if uni else None
        self.cruntime = self.lib / "libcruntime.a"
        self.posit_rt = ROOT / "src" / "posit_runtime.cpp"
        missing = [k for k in ("onnx_mlir", "translate", "clangxx", "cruntime")
                   if not getattr(self, k) or not Path(getattr(self, k)).exists()]
        if missing:
            sys.exit("missing prerequisites: " + ", ".join(missing))
        if not glob.glob(str(self.lib / "PyRuntimeC*.so")):
            sys.exit("PyRuntimeC not built (ninja PyRuntimeC)")


def run(cmd, env=None, timeout=900, cwd=None):
    """Every compiler step gets a hard timeout: lowp_lower_general.py documents a
    MobileNetV2 configuration where convert-krnl-to-llvm never returned."""
    e = dict(os.environ)
    e.update(env or {})
    t0 = time.time()
    try:
        r = subprocess.run([str(c) for c in cmd], env=e, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"TIMEOUT after {timeout}s: {' '.join(str(c) for c in cmd)[:300]}")
    if r.returncode != 0:
        raise RuntimeError(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)[:300]}\n{r.stderr[-2500:]}")
    return r, time.time() - t0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def download_shard(dest_dir):
    from huggingface_hub import hf_hub_download
    repo, fname = HF_MIRROR
    log(f"downloading {repo}:{fname} (~470MB, ungated mirror of the ImageNet-1k validation split) ...")
    return hf_hub_download(repo_id=repo, repo_type="dataset", filename=fname, local_dir=str(dest_dir))


def prepare_images(img_dir, parquet, n, n_calib, seed):
    """Sample n validation images (original JPEG bytes, untouched) and choose the
    calibration subset. Cached in img_dir/manifest.json."""
    manifest_path = img_dir / "manifest.json"
    want = {"num_images": n, "num_calibration": n_calib, "seed": seed}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        if all(m.get(k) == v for k, v in want.items()):
            return m
    if not parquet:
        sys.exit("no cached images for these parameters: pass --parquet FILE or --download-shard")
    import pyarrow.parquet as pq
    table = pq.ParquetFile(parquet).read()
    labels, images = table.column("label").to_pylist(), table.column("image")
    rows = sorted(np.random.default_rng(seed).choice(len(labels), n, replace=False).tolist())
    calib = set(np.random.default_rng(seed + 1).choice(n, n_calib, replace=False).tolist())
    img_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for k, r in enumerate(rows):
        cell = images[r].as_py()
        fname = f"img_{k:03d}.JPEG"
        (img_dir / fname).write_bytes(cell["bytes"])
        entries.append({"file": fname, "original_name": cell["path"], "label": int(labels[r]),
                        "row": int(r), "calibration": k in calib})
    m = {**want, "parquet": Path(parquet).name, "images": entries}
    manifest_path.write_text(json.dumps(m, indent=1))
    return m


def preprocess_all(img_dir, manifest, model_meta):
    """timm's own eval transform for this checkpoint (resize+center crop+normalize)."""
    from PIL import Image
    from timm.data import create_transform, resolve_data_config
    cfg = model_meta.get("pretrained_cfg")
    tf = create_transform(**resolve_data_config({}, pretrained_cfg=cfg), is_training=False)
    xs = []
    for e in manifest["images"]:
        with Image.open(img_dir / e["file"]) as im:
            xs.append(tf(im.convert("RGB")).numpy())
    return np.ascontiguousarray(np.stack(xs), dtype=np.float32)


def preprocess_torchvision(img_dir, manifest, meta):
    """Standard ImageNet eval transform (Resize(256) -> CenterCrop(224) ->
    Normalize), for models this repo sourced from torchvision rather than timm
    -- preprocess_all()'s timm transform needs a timm pretrained_cfg, which a
    torchvision-sourced model's metadata.json doesn't have (model_meta.get()
    would silently fall back to timm's generic default instead of this
    checkpoint's actual eval recipe)."""
    from PIL import Image
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    xs = []
    for e in manifest["images"]:
        with Image.open(img_dir / e["file"]) as im:
            xs.append(tf(im.convert("RGB")).numpy())
    return np.ascontiguousarray(np.stack(xs), dtype=np.float32)


def preprocess_by_source(img_dir, manifest, meta):
    return (preprocess_all(img_dir, manifest, meta) if meta.get("source") == "timm"
            else preprocess_torchvision(img_dir, manifest, meta))


# --------------------------------------------------------------------------- #
# Graph, calibration, random assignment
# --------------------------------------------------------------------------- #
def supported_nodes(model):
    inits = {i.name for i in model.graph.initializer}
    nodes = []
    for n in model.graph.node:
        outs = [o for o in n.output if o]
        if n.op_type not in SUPPORTED_OPS or not n.name:
            continue
        if n.op_type == "MaxPool" and len(outs) != 1:      # 2-result form: not lowerable by onnx-mlir at all
            continue
        node = {"name": n.name, "op": n.op_type, "output": outs[0]}
        node["act_tensors"] = list(n.input[:2]) + outs if n.op_type in BINARY_OPS else [n.input[0]] + outs
        if n.op_type in WEIGHT_OPS:
            node["weight_const"] = len(n.input) > 1 and n.input[1] in inits
        nodes.append(node)
    return nodes


def calibrate_ranges(model, nodes, calib_inputs):
    """Per-tensor (min, max) over the calibration images, from the FP32 ONNX in onnxruntime."""
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper
    inits = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    graph_in = {i.name for i in model.graph.input} - set(inits)
    need = sorted({t for n in nodes for t in n["act_tensors"]})
    taps = [t for t in need if t not in inits and t not in graph_in]
    m = onnx.ModelProto()
    m.CopyFrom(model)
    have = {o.name for o in m.graph.output}
    for t in taps:
        if t not in have:
            m.graph.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    in_name, out_names = sess.get_inputs()[0].name, [o.name for o in sess.get_outputs()]
    lo, hi = {t: np.inf for t in need}, {t: -np.inf for t in need}
    for x in calib_inputs:
        outs = dict(zip(out_names, sess.run(None, {in_name: x[None]})))
        for t in taps:
            lo[t], hi[t] = min(lo[t], float(outs[t].min())), max(hi[t], float(outs[t].max()))
        for t in need:
            if t in graph_in:
                lo[t], hi[t] = min(lo[t], float(x.min())), max(hi[t], float(x.max()))
    for t in need:
        if t in inits:
            lo[t], hi[t] = float(inits[t].min()), float(inits[t].max())
    return {t: (lo[t], hi[t]) for t in need}


def int_qparams(lo, hi, bits):
    """Affine min/max calibration for a signed `bits`-bit integer. The range is widened
    to contain 0 so that 0.0 is exactly representable (standard practice)."""
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    lo, hi = min(lo, 0.0), max(hi, 0.0)
    scale = (hi - lo) / (qmax - qmin)
    if not scale > 0.0:                      # tensor is identically zero
        scale = 1.0
    zp = max(qmin, min(qmax, int(round(qmin - lo / scale))))
    return scale, zp


def random_config(nodes, weight_formats, act_formats, seed):
    rng = random.Random(seed)
    cfg = {}
    for n in nodes:
        w = rng.choice(weight_formats) if n["op"] in WEIGHT_OPS and n.get("weight_const") else None
        cfg[n["name"]] = {"op": n["op"], "w": w, "a": rng.choice(act_formats)}
    return cfg


def uniform_config(nodes, w, a):
    """Every node gets the same weight format `w` and activation format `a`."""
    return {n["name"]: {"op": n["op"], "w": w if n["op"] in WEIGHT_OPS and n.get("weight_const") else None, "a": a}
            for n in nodes}


def identity_config(nodes):
    return {n["name"]: {"op": n["op"], "w": "fp32" if n["op"] in WEIGHT_OPS and n.get("weight_const") else None,
                        "a": "fp32"} for n in nodes}


def node_formats_env(cfg, nodes, ranges):
    """MIXEDP_NODE_FORMATS for a config: int activation formats get the calibrated
    scale/zero-point of the node's own tensors (x,y for unary ops; a,b,y for binary)."""
    entries = []
    for n in nodes:
        c = cfg[n["name"]]
        parts = [n["name"]]
        if c["w"] is not None:
            parts.append(f"w={c['w']}")
        parts.append(f"a={c['a']}")
        if c["a"].startswith("int"):
            for t in n["act_tensors"]:
                scale, zp = int_qparams(*ranges[t], int(c["a"][3:]))
                parts += [repr(float(scale)), str(zp)]
        entries.append(":".join(parts))
    return ",".join(entries)


def expected_sites(cfg, nodes):
    """How many activation QDQ sites the pass should have inserted, by family."""
    sites = {"int": 0, "half": 0, "posit": 0}
    for n in nodes:
        a = cfg[n["name"]]["a"]
        k = len(n["act_tensors"])
        if a.startswith("int"):
            sites["int"] += k
        elif a in ("f16", "bf16"):
            sites["half"] += k
        elif a.startswith("p"):
            sites["posit"] += k
    return sites


# --------------------------------------------------------------------------- #
# Compile + run
# --------------------------------------------------------------------------- #
def build_posit_runtime(tools, work):
    obj = work / "posit_runtime_p4e1_p6e1_p8e1.o"
    if not obj.exists():
        flags = ["-DPOSIT_USE_UNIVERSAL", "-DPOSIT_RUNTIME_SINGLE_FORMAT=1", "-DPOSIT_BUILD_MIXED_OFF=1",
                 *[f"-DPOSIT_RUNTIME_FMT_{f.upper()}=1" for f in POSIT_FORMATS],
                 f"-I{tools.universal_inc}", f"-I{tools.mlir_inc}", "-I/usr/local/include"]
        _, dt = run([tools.clangxx, "-std=c++20", "-O3", "-fPIC", "-c", tools.posit_rt, *flags, "-o", obj],
                    timeout=1800)
        log(f"  built posit runtime (p4e1+p6e1+p8e1) in {dt:.0f}s")
    return obj


def ir_stats(text):
    return {"onnx.Round": len(re.findall(r'"onnx\.Round"', text)),
            "onnx.Div": len(re.findall(r'"onnx\.Div"', text)),
            "onnx.Cast": len(re.findall(r'"onnx\.Cast"', text)),
            "posit.round_f32": len(re.findall(r"posit\.round_f32", text)),
            "posit.from_f32": len(re.findall(r"posit\.from_f32", text)),
            "posit.to_f32": len(re.findall(r"posit\.to_f32", text))}


def set_symbol_postfix(mlir_path, postfix):
    """Rename the module's onnx-mlir symbol postfix, in place. Every .so built from the same imported model exports
    main_graph_<postfix>, run_main_graph_<postfix>, ...; loaded into one process, the second binds the first one's
    main_graph and silently computes the wrong graph. A distinct postfix per .so avoids that; load it with
    OMExecutionSession(so, postfix)."""
    p = Path(mlir_path)
    text = p.read_text()
    old = f'"onnx-mlir.symbol-postfix" = "{SYMBOL_POSTFIX}"'
    if old not in text:
        raise ValueError(f"{p}: no {old}")
    p.write_text(text.replace(old, f'"onnx-mlir.symbol-postfix" = "{postfix}"', 1))


def build_so(tools, work, tag, base_mlir, node_formats, posit_obj, uses_posit=None, uses_f16=None, opt="O3",
             constants_to_file=False, krnl_opt=None, bytecode=False):
    """node_formats=None -> plain FP32 build without the pass. uses_posit / uses_f16 override what is otherwise read
    off node_formats (needed for graphs whose posit / f16 QDQ does not come from it, e.g. --mixedp-collect-stats).
    opt is the optimization level of both the Krnl lowering and clang (O3 for evaluated models; the very large
    statistics graphs compile much faster at a lower level); krnl_opt, if given, overrides it for the Krnl lowering
    alone (-O3 there tiles / vectorizes MatMul, which clang's level does not recover).
    constants_to_file: --convert-krnl-to-llvm writes every constant over 1 KB to <work>/model.constants.bin instead
    of the .ll, so translation and clang only see code (GPT-2's .ll otherwise holds ~1.3 GB of weights as text). The
    .so then mmaps that file from OM_CONSTANT_PATH, which OMExecutionSession sets to the .so's folder -- once per
    process: give every such build its own `work` folder and load at most one such .so per process.
    bytecode: every intermediate file is MLIR bytecode (.mlirbc, onnx-mlir-opt --emit-bytecode), base_mlir may be
    either form. Weights are raw bytes there instead of hex text: Qwen3-0.6B's import is 3.0 GB instead of 6.0 GB,
    one read + write 4.3 s instead of 19 s at a 6.3 GB instead of 10.9 GB peak, and the output is the same (mlir-
    translate reads the bytecode; the .ll is byte-identical). The IR op counts (ir_after_pass, ...) need text: None."""
    steps, warnings, stats = {}, [], None
    ext, emit = (".mlirbc", ["--emit-bytecode"]) if bytecode else (".mlir", [])
    src = base_mlir
    if node_formats is not None:
        src = work / f"{tag}.mixed.onnx{ext}"
        r, steps["mixed_pass"] = run([tools.opt, base_mlir, "--shape-inference", "--convert-onnx-to-mixed-precision",
                                      *emit, "-o", src], env={"MIXEDP_NODE_FORMATS": node_formats})
        warnings = [l for l in r.stderr.splitlines() if "warning" in l.lower()]
        stats = None if bytecode else ir_stats(src.read_text())
    if uses_posit is None:
        uses_posit = node_formats is not None and re.search(r"=p\d+e\d+", node_formats) is not None
    onnxir, krnl, llvm = (work / f"{tag}.{x}{ext}" for x in ("onnxir", "krnl", "llvm"))
    ll, so = work / f"{tag}.ll", work / f"{tag}.so"
    _, steps["onnx_level"] = run([tools.opt, src, *ONNX_LEVEL_PASSES, *emit, "-o", onnxir])
    stats_after = ir_stats(onnxir.read_text()) if node_formats is not None and not bytecode else None
    _, steps["to_krnl"] = run([tools.opt, onnxir, "--canonicalize", "--shape-inference", f"--{krnl_opt or opt}",
                               "--convert-onnx-to-krnl", *(["--convert-posit-to-krnl"] if uses_posit else []),
                               "--canonicalize", *emit, "-o", krnl])
    if constants_to_file:       # the pass writes ./model.constants.bin: run it inside `work`, with absolute paths
        to_llvm = [p if p != "--convert-krnl-to-llvm" else
                   "--convert-krnl-to-llvm=store-constants-to-file=true constants-to-file-single-threshold=1 "
                   "constants-to-file-total-threshold=0" for p in KRNL_TO_LLVM_PASSES]
        _, steps["to_llvm"] = run([tools.opt, Path(krnl).resolve(), *to_llvm, *emit, "-o", Path(llvm).resolve()],
                                  cwd=work)
    else:
        _, steps["to_llvm"] = run([tools.opt, krnl, *KRNL_TO_LLVM_PASSES, *emit, "-o", llvm])
    _, steps["translate"] = run([tools.translate, "--mlir-to-llvmir", llvm, "-o", ll])
    if ll.stat().st_size > 1_500_000_000:    # clang rejects textual inputs over 2 GiB: hand it bitcode instead
        bc = work / f"{tag}.bc"
        _, steps["llvm_as"] = run([tools.translate.parent / "llvm-as", ll, "-o", bc])
        ll = bc
    if uses_f16 is None:
        uses_f16 = node_formats is not None and re.search(r"a=f16(?=[:,]|$)", node_formats) is not None
    if uses_f16 and not HAS_F16C:
        raise RuntimeError("f16 activations need a CPU with F16C: without it the generated code calls the "
                           "ABI-mismatched __truncsfhf2/__extendhfsf2 of libcruntime.a and returns garbage")
    link = [tools.clangxx, "-std=c++20", f"-{opt}", "-fPIC", "-shared", *(["-mf16c"] if uses_f16 else []), ll] + \
           ([posit_obj] if uses_posit and posit_obj else []) + [tools.cruntime, "-o", so]
    _, steps["link"] = run(link)
    return so, {"steps": steps, "warnings": warnings, "ir_after_pass": stats, "ir_after_onnx_passes": stats_after}


def run_so(tools, so, inputs_path, out_path, timeout):
    """Fresh process per .so: isolation (timeout/crash) and everything is returned to the OS on exit."""
    r, dt = run([sys.executable, __file__, "--_eval-so", so, "--_inputs", inputs_path, "--_out", out_path,
                 "--_lib", tools.lib], timeout=timeout)
    return np.load(out_path), json.loads(r.stdout.strip().splitlines()[-1])


def child_eval(so, inputs_path, out_path, lib):
    sys.path.insert(0, lib)
    from PyRuntime import OMExecutionSession
    x = np.load(inputs_path)
    sess = OMExecutionSession(so, SYMBOL_POSTFIX)
    t0 = time.time()
    outs = [np.array(sess.run([np.ascontiguousarray(x[i:i + 1])])[0][0]) for i in range(len(x))]
    np.save(out_path, np.stack(outs).astype(np.float32))
    # VmHWM, not getrusage().ru_maxrss: after fork+exec the latter still carries the parent's peak RSS.
    peak_kb = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM"))
    print(json.dumps({"seconds": time.time() - t0, "n": len(x), "peak_rss_gb": peak_kb / 1048576}))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def accuracy(logits, labels, mask):
    finite = np.isfinite(logits).all(axis=1)
    top1 = (logits.argmax(axis=1) == labels) & finite
    top5 = (np.argsort(-logits, axis=1)[:, :5] == labels[:, None]).any(axis=1) & finite
    m = np.asarray(mask)
    return {"n": int(m.sum()), "top1": float(top1[m].mean()), "top5": float(top5[m].mean()),
            "top1_correct": int(top1[m].sum()), "top5_correct": int(top5[m].sum())}


def summarize(logits, ref_logits, labels, calib_mask):
    allm = np.ones(len(labels), bool)
    s = {"all": accuracy(logits, labels, allm), "calibration": accuracy(logits, labels, calib_mask),
         "held_out": accuracy(logits, labels, ~calib_mask),
         "non_finite_rows": int((~np.isfinite(logits).all(axis=1)).sum())}
    if ref_logits is not None:
        fin = np.isfinite(logits).all(axis=1)
        s["same_top1_as_fp32"] = int(((logits.argmax(axis=1) == ref_logits.argmax(axis=1)) & fin).sum())
        s["max_abs_logit_diff_vs_fp32"] = float(np.abs(logits - ref_logits).max()) if fin.all() else None
        s["mean_abs_logit_diff_vs_fp32"] = float(np.abs(logits - ref_logits).mean()) if fin.all() else None
    return s


def fmt_acc(a):
    return f"{a['top1'] * 100:5.1f}% / {a['top5'] * 100:5.1f}%"


def torch_reference(meta, inputs):
    import torch
    if meta.get("source") == "timm":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import timm
        model = timm.create_model(meta["timm_model"], pretrained=True).eval()
    else:
        import torchvision.models as tv_models
        model = tv_models.get_model(meta["torchvision_model"], weights=meta["torchvision_weights"]).eval()
    with torch.no_grad():
        return torch.cat([model(torch.from_numpy(inputs[i:i + 25])) for i in range(0, len(inputs), 25)]).numpy()


def ort_logits(model_path, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.stack([sess.run(None, {name: x[None]})[0][0] for x in inputs]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Independent reference: the same fake-quantization spliced into the ONNX graph and run by onnxruntime
# --------------------------------------------------------------------------- #
POSIT_ORACLE_SRC = r"""
#include <universal/number/posit/posit.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
using sw::universal::posit;
template <int N, int ES> void go(const char *i, const char *o) {
  FILE *f = fopen(i, "rb"); std::vector<float> v; float x;
  while (fread(&x, 4, 1, f) == 1) v.push_back(x); fclose(f);
  for (auto &e : v) { posit<N, ES> p; p = e; e = static_cast<float>(p); }
  f = fopen(o, "wb"); fwrite(v.data(), 4, v.size(), f); fclose(f);
}
int main(int, char **a) {
  int n = atoi(a[1]), es = atoi(a[2]);
  if (n == 4 && es == 1) go<4, 1>(a[3], a[4]);
  else if (n == 6 && es == 1) go<6, 1>(a[3], a[4]);
  else if (n == 8 && es == 1) go<8, 1>(a[3], a[4]);
  else return 1;
}
"""


class PositOracle:
    """Universal's float -> posit<N,ES> -> float, elementwise (same helper as test/mixed_precision/)."""

    def __init__(self, tools, work):
        self.tools, self.work, self.exe, self.lock = tools, work, None, threading.Lock()

    def __call__(self, x, fmt):
        with self.lock:
            if self.exe is None:
                src, self.exe = self.work / "posit_oracle.cpp", self.work / "posit_oracle"
                src.write_text(POSIT_ORACLE_SRC)
                run([self.tools.clangxx, "-std=c++20", "-O2", f"-I{self.tools.universal_inc}", src, "-o", self.exe])
        x = np.ascontiguousarray(x, dtype="<f4")
        with tempfile.TemporaryDirectory(dir=self.work) as d:
            fi, fo = Path(d) / "in.bin", Path(d) / "out.bin"
            x.tofile(fi)
            run([self.exe, fmt[1], fmt[3], fi, fo])
            return np.fromfile(fo, dtype="<f4").reshape(x.shape)


def np_fake_quant_weight(w, fmt, posit_oracle):
    """Same arithmetic as fakeQuantWeightInt/Float/Posit in ONNXToMixedPrecision.cpp, in numpy."""
    w = np.asarray(w, np.float32)
    if fmt.startswith("int"):
        bits = int(fmt[3:])
        qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
        fin = np.isfinite(w)
        maxabs = np.float32(np.abs(w[fin]).max()) if fin.any() else np.float32(0)
        scale = np.float32(maxabs / np.float32(qmax)) if maxabs > 0 else np.float32(1)
        t = (w / scale).astype(np.float64)                                # float32 division, like `f / scale`
        q = np.clip(np.sign(t) * np.floor(np.abs(t) + 0.5), qmin, qmax)   # std::lround: half away from zero
        return (q.astype(np.float32) * scale).astype(np.float32)
    if fmt == "f16":
        return w.astype(np.float16).astype(np.float32)
    if fmt == "bf16":
        import torch
        return torch.from_numpy(w).to(torch.bfloat16).to(torch.float32).numpy()
    if fmt in POSIT_FORMATS:
        return posit_oracle(w, fmt)
    raise ValueError(fmt)


def has_posit_activation(cfg):
    return any(c["a"].startswith("p") for c in cfg.values())


def simulate_in_onnxruntime(model, nodes, cfg, ranges, inputs, posit_oracle):
    """Fake-quantize the weight initializers in numpy and splice the same Div/Add/Round/Clip/Sub/Mul (int) or
    Cast-down/Cast-up (f16/bf16) chains into the ONNX graph, then run it in onnxruntime. Nothing here goes through
    onnx-mlir or the pass. Posit ACTIVATIONS have no onnxruntime equivalent: such configs are not supported."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper
    if has_posit_activation(cfg):
        raise ValueError("posit activations cannot be simulated in onnxruntime")
    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    users = {}
    for n in g.node:
        for t in n.input:
            users[t] = users.get(t, 0) + 1
    by_name = {n["name"]: n for n in nodes}
    counter = [0]

    def fresh(prefix):
        counter[0] += 1
        return f"__{prefix}{counter[0]}"

    def const(value):
        name = fresh("c")
        g.initializer.append(numpy_helper.from_array(np.asarray(value, np.float32), name))
        return name

    def fake_quant(src, dst, fmt, lohi):
        if fmt.startswith("int"):
            bits = int(fmt[3:])
            scale, zp = int_qparams(*lohi, bits)
            s, z = const(np.float32(scale)), const(np.float32(zp))
            lo, hi = const(np.float32(-(1 << (bits - 1)))), const(np.float32((1 << (bits - 1)) - 1))
            t = [fresh("t") for _ in range(5)]
            return [helper.make_node("Div", [src, s], [t[0]]), helper.make_node("Add", [t[0], z], [t[1]]),
                    helper.make_node("Round", [t[1]], [t[2]]), helper.make_node("Clip", [t[2], lo, hi], [t[3]]),
                    helper.make_node("Sub", [t[3], z], [t[4]]), helper.make_node("Mul", [t[4], s], [dst])]
        to = {"f16": TensorProto.FLOAT16, "bf16": TensorProto.BFLOAT16}[fmt]
        mid = fresh("t")
        return [helper.make_node("Cast", [src], [mid], to=to),
                helper.make_node("Cast", [mid], [dst], to=TensorProto.FLOAT)]

    new_nodes = []
    for orig in g.node:
        info, c = by_name.get(orig.name), cfg.get(orig.name)
        if info is None or c is None:
            new_nodes.append(orig)
            continue
        n = onnx.NodeProto()
        n.CopyFrom(orig)
        pre, post = [], []
        if c["w"] not in (None, "fp32"):
            wname = n.input[1]
            if users.get(wname, 0) != 1:
                raise ValueError(f"weight {wname} is shared; not handled by this reference")
            inits[wname].CopyFrom(numpy_helper.from_array(
                np_fake_quant_weight(numpy_helper.to_array(inits[wname]), c["w"], posit_oracle), wname))
        if c["a"] != "fp32":
            tensors = info["act_tensors"]                       # [x, y] or [a, b, y]
            for i, t in enumerate(tensors[:-1]):
                dst = f"{t}__q{i}_{orig.name}"
                pre += fake_quant(n.input[i], dst, c["a"], ranges[t])
                n.input[i] = dst
            y = n.output[0]
            raw = f"{y}__raw_{orig.name}"
            n.output[0] = raw                                   # consumers keep reading `y`, now the quantized value
            post = fake_quant(raw, y, c["a"], ranges[tensors[-1]])
        new_nodes += pre + [n] + post
    del g.node[:]
    g.node.extend(new_nodes)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(m.SerializeToString(), opts, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    return np.stack([sess.run(None, {name: x[None]})[0][0] for x in inputs]).astype(np.float32)


# --------------------------------------------------------------------------- #
def check_formats(label, fmts):
    bad = [f for f in fmts if f not in FORMATS]
    if bad:
        sys.exit(f"unknown {label} {bad}; choose from {FORMATS}")
    return fmts


def format_counts(cfg):
    wf, af = {}, {}
    for c in cfg.values():
        if c["w"] is not None:
            wf[c["w"]] = wf.get(c["w"], 0) + 1
        af[c["a"]] = af.get(c["a"], 0) + 1
    return dict(sorted(wf.items())), dict(sorted(af.items()))


def fmt_sim(s):
    v = s.get("vs_onnxruntime_simulation")
    if v is None:
        return f"failed ({s['simulation_error']})" if "simulation_error" in s else "n/a (posit activations)"
    return f"max|dlogit| {v['max_abs_diff']:.1e}, same top-1 {v['same_top1']}/{s['all']['n']}"


def variant_label(name):
    if name.startswith("random_seed"):
        return f"pass, RANDOM formats, seed {name[len('random_seed'):]}"
    if name.startswith("uniform_w_"):
        return f"pass, every weight = {name[len('uniform_w_'):]} (act fp32)"
    if name.startswith("uniform_a_"):
        return f"pass, every activation = {name[len('uniform_a_'):]} (w fp32)"
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                    help="dir with model.onnx and metadata.json (timm pretrained_cfg)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--parquet", type=Path, help="ImageNet-1k validation parquet shard (image{bytes,path}, label)")
    src.add_argument("--download-shard", action="store_true", help=f"fetch {HF_MIRROR[1]} from {HF_MIRROR[0]}")
    ap.add_argument("--num-images", type=int, default=100)
    ap.add_argument("--num-calibration", type=int, default=10)
    ap.add_argument("--data-seed", type=int, default=0, help="image sampling + calibration split")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds for the random format assignment")
    ap.add_argument("--formats", default=",".join(FORMATS), help="candidate formats to draw from")
    ap.add_argument("--weight-formats", default=None, help="candidates for weights only (default: --formats)")
    ap.add_argument("--act-formats", default=None, help="candidates for activations only (default: --formats)")
    ap.add_argument("--uniform-sweep", action="store_true",
                    help="also run every non-fp32 format applied to ALL weights (activations fp32) and to ALL "
                         "activations (weights fp32): shows which formats this model tolerates at all")
    ap.add_argument("--sweep-formats", default=None, help="formats for --uniform-sweep (default: all except fp32)")
    ap.add_argument("--htq-configs", type=Path, default=None,
                    help="sample_configs_htq.py output JSON: evaluate its SNIP-sensitivity-guided "
                         "Pareto configs as additional variants, same as random_seed*/uniform_*")
    ap.add_argument("--jobs", type=int, default=1,
                    help="variants compiled+evaluated in parallel (a run can need >10GB RAM: keep it small)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="small persistent results (default experiments/mixed_precision/runs/<arch>_<variant>)")
    ap.add_argument("--images-dir", type=Path, default=None,
                    help="sampled ImageNet images + manifest.json, shareable between runs (default <out-dir>/images)")
    ap.add_argument("--work-dir", type=Path, default=None, help="big intermediates (default: a temp dir)")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--eval-timeout", type=int, default=3600)
    for hidden in ("--_eval-so", "--_inputs", "--_out", "--_lib"):
        ap.add_argument(hidden, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args._eval_so:
        return child_eval(args._eval_so, args._inputs, args._out, args._lib)

    formats = check_formats("--formats", args.formats.split(","))
    wformats = check_formats("--weight-formats", args.weight_formats.split(",")) if args.weight_formats else formats
    aformats = check_formats("--act-formats", args.act_formats.split(",")) if args.act_formats else formats
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else []
    model_path = args.model_dir / "model.onnx"
    meta = json.loads((args.model_dir / "metadata.json").read_text())
    tag = f"{args.model_dir.parent.name}_{args.model_dir.name}"
    out = args.out_dir or HERE / "runs" / tag
    out.mkdir(parents=True, exist_ok=True)
    images_dir = args.images_dir or out / "images"
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="mixedp-"))
    work.mkdir(parents=True, exist_ok=True)
    tools = Tools()
    t_start = time.time()
    results = {"model": str(model_path.relative_to(ROOT)) if model_path.is_relative_to(ROOT) else str(model_path),
               "weight_formats": wformats, "act_formats": aformats, "variants": {}}
    try:
        # ---- data -------------------------------------------------------------------------------
        parquet = args.parquet
        if args.download_shard and not (images_dir / "manifest.json").exists():
            parquet = download_shard(work)
        manifest = prepare_images(images_dir, parquet, args.num_images, args.num_calibration, args.data_seed)
        labels = np.array([e["label"] for e in manifest["images"]])
        calib_mask = np.array([e["calibration"] for e in manifest["images"]])
        inputs = preprocess_by_source(images_dir, manifest, meta)
        inputs_path = work / "inputs.npy"
        np.save(inputs_path, inputs)
        log(f"images: {len(labels)} ImageNet-1k validation images ({len(set(labels.tolist()))} distinct classes), "
            f"{int(calib_mask.sum())} used for calibration; input {inputs.shape}")

        # ---- graph + calibration ----------------------------------------------------------------
        import onnx
        model = onnx.load(str(model_path))
        nodes = supported_nodes(model)
        by_op, covered = {}, {}
        for n in model.graph.node:
            by_op[n.op_type] = by_op.get(n.op_type, 0) + 1
        for n in nodes:
            covered[n["op"]] = covered.get(n["op"], 0) + 1
        log(f"graph: {len(model.graph.node)} nodes {by_op}")
        log(f"  supported by the pass: {len(nodes)} nodes {covered}; the other "
            f"{len(model.graph.node) - len(nodes)} nodes always stay FP32")
        t = time.time()
        ranges = calibrate_ranges(model, nodes, inputs[calib_mask])
        (out / "calibration.json").write_text(json.dumps(
            {"num_calibration_images": int(calib_mask.sum()), "min_max": ranges}, indent=1))
        log(f"calibration: min/max of {len(ranges)} activation tensors over {int(calib_mask.sum())} images "
            f"({time.time() - t:.1f}s)")

        # ---- references -------------------------------------------------------------------------
        refs = {}
        try:
            refs["torch_fp32"] = torch_reference(meta, inputs)
        except Exception as e:                                     # noqa: BLE001 - optional reference
            log(f"  (PyTorch reference skipped: {type(e).__name__}: {str(e)[:100]})")
        refs["onnxruntime_fp32"] = ort_logits(model_path, inputs)

        # ---- builds -----------------------------------------------------------------------------
        base = work / f"{SYMBOL_POSTFIX}.onnx.mlir"
        _, dt = run([tools.onnx_mlir, "--EmitONNXBasic", "-o", work / SYMBOL_POSTFIX,
                     "--mlir-elide-resource-strings-if-larger=1000000000",
                     "--mlir-elide-elementsattrs-if-larger=1000000000", model_path])
        log(f"import (EmitONNXBasic, node names preserved): {dt:.1f}s")
        lock, posit_box = threading.Lock(), []
        oracle = PositOracle(tools, work)

        def get_posit_obj():
            with lock:
                if not posit_box:
                    posit_box.append(build_posit_runtime(tools, work))
                return posit_box[0]

        def evaluate(name, cfg):
            """cfg=None -> plain FP32 build without the pass."""
            env = None if cfg is None else node_formats_env(cfg, nodes, ranges)
            posit_obj = get_posit_obj() if env and re.search(r"=p\d+e\d+", env) else None
            t0 = time.time()
            so, info = build_so(tools, work, name, base, env, posit_obj)
            compile_s = time.time() - t0
            logits, ev = run_so(tools, so, inputs_path, work / f"{name}.logits.npy", args.eval_timeout)
            np.save(out / f"logits_{name}.npy", logits)
            if not args.keep_work:
                for p in work.glob(f"{name}.*"):
                    if p.suffix in (".mlir", ".ll", ".so", ".npy"):
                        p.unlink()
            s = summarize(logits, refs["onnxruntime_fp32"], labels, calib_mask)
            s.update({"compile_seconds": compile_s, "run_seconds_per_image": ev["seconds"] / ev["n"],
                      "peak_rss_gb": ev["peak_rss_gb"], **info})
            if cfg is not None and name.startswith("random_seed"):
                s["config"], s["node_formats_env"] = cfg, env
            if cfg is not None and not has_posit_activation(cfg):
                try:
                    sim = simulate_in_onnxruntime(model, nodes, cfg, ranges, inputs, oracle)
                    d = np.abs(logits - sim)
                    s["vs_onnxruntime_simulation"] = {
                        "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                        "same_top1": int((logits.argmax(axis=1) == sim.argmax(axis=1)).sum()),
                        "simulated_top1": float((sim.argmax(axis=1) == labels).mean())}
                except Exception as e:                             # noqa: BLE001 - the compiled result is still valid
                    s["simulation_error"] = f"{type(e).__name__}: {str(e)[:300]}"
            if cfg is not None:
                exp, ir = expected_sites(cfg, nodes), s["ir_after_pass"]
                checks = {"int": (exp["int"], ir["onnx.Round"]), "half(cast pairs)": (exp["half"] * 2, ir["onnx.Cast"]),
                          # one posit.round_f32 per site, or a from_f32/to_f32 pair under MIXEDP_POSIT_ACTIVATION=runtime
                          "posit": (exp["posit"], ir["posit.round_f32"] + ir["posit.from_f32"]),
                          "posit to_f32 pairs": (ir["posit.from_f32"], ir["posit.to_f32"])}
                s["site_check"] = {k: {"expected": e, "found": f} for k, (e, f) in checks.items()}
                s["sites_match"] = all(e == f for e, f in checks.values())
                after = s["ir_after_onnx_passes"]
                s["qdq_survived_onnx_passes"] = all(after[k] == ir[k] for k in ir)
            results["variants"][name] = s
            return logits, s

        log("\n== baseline: onnx-mlir FP32 (no pass) ==")
        fp32_logits, s = evaluate("onnx_mlir_fp32", None)
        log(f"  compile {s['compile_seconds']:.0f}s, run {s['run_seconds_per_image'] * 1000:.0f} ms/image, "
            f"top-1/top-5 {fmt_acc(s['all'])}")
        log("== control: pass with every node fp32 (must be identical to the baseline) ==")
        ident_logits, s = evaluate("pass_identity_fp32", identity_config(nodes))
        s["bit_identical_to_fp32_baseline"] = bool(np.array_equal(ident_logits, fp32_logits))
        log(f"  warnings {len(s['warnings'])}, top-1/top-5 {fmt_acc(s['all'])}, "
            f"bit-identical to baseline: {s['bit_identical_to_fp32_baseline']}")

        # ---- random (and optionally uniform) configurations -------------------------------------
        variants = [(f"random_seed{sd}", random_config(nodes, wformats, aformats, sd)) for sd in seeds]
        if args.uniform_sweep:
            for f in check_formats("--sweep-formats", args.sweep_formats.split(",")) if args.sweep_formats \
                    else [x for x in FORMATS if x != "fp32"]:
                variants += [(f"uniform_w_{f}", uniform_config(nodes, f, "fp32")),
                             (f"uniform_a_{f}", uniform_config(nodes, "fp32", f))]
        if args.htq_configs:
            htq_cfgs = json.loads(args.htq_configs.read_text())["configs"]
            variants += [(c["id"], c["formats"]) for c in htq_cfgs]
            log(f"loaded {len(htq_cfgs)} HTQ Pareto configs from {args.htq_configs}")
        for name, cfg in variants:
            if name.startswith("random_seed"):
                (out / f"{name}.node_formats.txt").write_text(node_formats_env(cfg, nodes, ranges) + "\n")
                (out / f"{name}.config.json").write_text(json.dumps(cfg, indent=1))

        def safe_evaluate(name, cfg):
            try:
                return evaluate(name, cfg)
            except Exception as e:                                 # noqa: BLE001 - keep the other variants going
                results["variants"][name] = {"error": str(e)[:2000]}
                return None, results["variants"][name]

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = [(name, cfg, ex.submit(safe_evaluate, name, cfg)) for name, cfg in variants]
            for name, cfg, fut in futs:
                _, s = fut.result()
                if "error" in s:
                    log(f"\n== {variant_label(name)}: FAILED ==\n  {s['error'][:600]}")
                elif name.startswith("random_seed"):
                    wf, af = format_counts(cfg)
                    ir, exp = s["ir_after_pass"], expected_sites(cfg, nodes)
                    log(f"\n== {variant_label(name)}: {len(cfg)} nodes ==")
                    log(f"  weight formats     : {wf}")
                    log(f"  activation formats : {af}")
                    log(f"  pass warnings: {len(s['warnings'])};  QDQ sites requested vs inserted (int/half/posit): "
                        f"{exp['int']}/{exp['half']}/{exp['posit']} vs {ir['onnx.Round']}/{ir['onnx.Cast'] // 2}/"
                        f"{ir['posit.round_f32'] + ir['posit.from_f32']} -> {'match' if s['sites_match'] else 'MISMATCH'}; "
                        f"survive ONNX-level passes: {s['qdq_survived_onnx_passes']}")
                    log(f"  compile {s['compile_seconds']:.0f}s (steps {{{', '.join(f'{k} {v:.1f}' for k, v in s['steps'].items())}}}), "
                        f"run {s['run_seconds_per_image'] * 1000:.0f} ms/image, peak RSS {s['peak_rss_gb']:.1f} GB")
                    log(f"  top-1/top-5 {fmt_acc(s['all'])};  same top-1 as FP32 on {s['same_top1_as_fp32']}/{len(labels)}; "
                        f"distinct predicted classes {len(set(np.load(out / f'logits_{name}.npy').argmax(1).tolist()))}")
                    log(f"  vs onnxruntime fake-quant simulation: {fmt_sim(s)}")
                else:
                    log(f"  {variant_label(name):<40} top-1/top-5 {fmt_acc(s['all'])}  "
                        f"same top-1 as FP32 {s['same_top1_as_fp32']:>3}/{len(labels)}  "
                        f"warnings {len(s['warnings'])}  sites {'ok' if s['sites_match'] else 'MISMATCH'}  "
                        f"{s['run_seconds_per_image'] * 1000:.0f} ms/img  {s['peak_rss_gb']:.1f} GB  | ort-sim: {fmt_sim(s)}")

        # ---- summary ----------------------------------------------------------------------------
        log("\n" + "=" * 96)
        log(f"ImageNet-1k validation, {len(labels)} random images ({int(calib_mask.sum())} of them = calibration set); "
            "top-1 / top-5")
        log(f"{'':<50}{'all ' + str(len(labels)):>16}{'calib ' + str(int(calib_mask.sum())):>14}"
            f"{'held-out ' + str(int((~calib_mask).sum())):>16}")
        rows = [(f"{k} (reference)", summarize(v, None, labels, calib_mask)) for k, v in refs.items()]
        results["references"] = {label: v for label, v in rows}
        rows += [("onnx-mlir FP32 (no pass)", results["variants"]["onnx_mlir_fp32"]),
                 ("pass, all nodes fp32 (identity control)", results["variants"]["pass_identity_fp32"])]
        rows += [(variant_label(name), results["variants"][name]) for name, _ in variants]
        for label, v in rows:
            if "error" in v:
                log(f"{label:<50}  FAILED")
            else:
                log(f"{label:<50}{fmt_acc(v['all']):>16}{fmt_acc(v['calibration']):>14}{fmt_acc(v['held_out']):>16}")
        log("=" * 96)
        results["wall_seconds"] = time.time() - t_start
        (out / "results.json").write_text(json.dumps(results, indent=1))
        log(f"results: {out / 'results.json'}   (total {results['wall_seconds']:.0f}s)")
    finally:
        if not args.keep_work and args.work_dir is None:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
