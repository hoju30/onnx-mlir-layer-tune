#!/usr/bin/env python3
"""Experiment C of the LP pipeline regression (runs/lp_pipeline_regression_20261016): new-window quality of the distinct
final configurations of experiment B, for the language-model problems that need it (Qwen3, Qwen3.5).

Primary backend: GPU PyTorch replay (labelled as such, not a native whole-model result): FP32 model, every searched
projection's weight quantized per output channel (qwen3_swiglu_common.Quantizers.weight) and its input (x) and output
(y) QDQ'd with the calibration ranges of the current BC table, the int activation quantizer with scale / zero-point on
the input's device (the correction validated in olmoe_search_final), TF32 off. Cross-check: the native onnx-mlir build
(as the models' evaluation scripts compile it) on a fixed subset, reported separately.

Stages (--problem qwen3 | qwen35_ffn | qwen35_ffn_attn; --out-dir = the regression's out dir):
  c_freeze    distinct final configurations of B (+ FP32), k, n from the plan's rule, windows; nothing read yet
  c_windows   tokenization of the new windows, overlap check against every earlier window range of that model
  c_check     the replay against earlier native per-window results of the same model (earlier windows)
  c_replay    GPU replay of every frozen configuration on the new windows
  c_native    native build + run of every frozen configuration on the cross-check subset
  c_analyze   dNLL, paired comparisons against the reference, decisions
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEQ = 128
BOOT = 10000
Q35 = HERE / "runs/qwen35_bc_20261015"
Q3T = HERE / "runs/native_eps_fix_20261014/qwen3"
Q3P = HERE / "runs/qwen3_prepare_20261011"
MODELS = {
    "qwen3": {"kind": "qwen3", "ranges": Q3T / "ranges_by_point.json", "mapping": Q3P / "node_mapping.json",
              "prior": {"dnll": 0.0750, "source": "runs/qwen3_branch_ablation_20261014/heldout.json knee_total, 32 test "
                                                   "windows, native"},
              "sd": {"value": 0.036, "source": "same heldout: SD of knee_all_branch - knee_total per window"},
              "first": {"V": 65536, "T": 65536},
              "used": {"validation": [[0, 15360, "windows 0-119: calibration, tuning, curvature / FD studies"]],
                       "test": [[0, 4096, "prepare / heldout test windows 0-31"]]}},
    "qwen35_ffn_attn": {"kind": "qwen35", "scope": "ffn_attn", "ranges": Q35 / "ranges_by_point.json",
                        "mapping": Q35 / "node_mapping.json",
                        "prior": {"dnll": 0.0323, "source": "runs/qwen35_multipoint_20261015 native ffn_attn_knee, 64 windows"},
                        "sd": {"value": 0.030, "source": "same run: SD of ffn_attn_knee - ffn_attn_conservative per window"},
                        "first": {"V": 32768, "T": 16384}},
    "qwen35_ffn": {"kind": "qwen35", "scope": "ffn", "ranges": Q35 / "ranges_by_point.json", "mapping": Q35 / "node_mapping.json",
                   "prior": {"dnll": 0.0098, "source": "runs/qwen35_multipoint_20261015 native ffn_knee, 64 windows"},
                   "sd": {"value": 0.018, "source": "same run: SD of ffn_knee - ffn_conservative per window"},
                   "first": {"V": 32768, "T": 16384}},
}
QWEN35_USED = {"validation": [[0, 1280], [1280, 5120], [8192, 10240], [10240, 14336], [16384, 17664], [20480, 24320],
                              [24576, 28416], [131072, 135168]],
               "test": [[0, 8192], [131072, 135168]]}
N_CAP = 1024
NATIVE_SUBSET_PER_SEG = 16


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zq(p):
    from scipy.stats import norm
    return float(norm.ppf(p))


# --------------------------------------------------------------------------- #
def stage_c_freeze(args, out, log):
    from mp_auto import config_hash
    key = args.problem
    M = MODELS[key]
    d = out / "C" / key
    d.mkdir(parents=True, exist_ok=True)
    if (d / "freeze.json").exists():
        raise SystemExit("frozen")
    B = out / "B" / key
    runs = {f.stem.replace("search_", ""): jload(f) for f in sorted(B.glob("search_*.json"))}
    main = ["milp_orig", "lpfirst_orig", "milp_grouped", "lpfirst_grouped"]
    by_hash, labels = {}, {}
    for r in main:
        fm = runs[r]["selected_formats"]
        h = config_hash(fm)
        labels.setdefault(h, []).append(r)
        by_hash.setdefault(h, fm)
    ref_h = config_hash(runs["milp_orig"]["selected_formats"])
    (d / "configs").mkdir(exist_ok=True)
    names = {}
    for i, (h, fm) in enumerate(by_hash.items()):
        nm = "ref" if h == ref_h else f"alt{i}"
        names[nm] = {"hash": h, "runs": labels[h]}
        jdump(d / "configs" / f"{nm}.json", fm)
    k = len(names) - 1
    margin = 0.10 * M["prior"]["dnll"]
    if k:
        n_req = ((zq(1 - 0.025 / k) + zq(0.8)) * M["sd"]["value"] / margin) ** 2
    else:
        n_req = 0
    n = int(min(N_CAP, 2 * math.ceil(max(n_req, 64) / 2 / 64) * 64)) if k else 0
    power = None
    if k:
        from scipy.stats import norm
        se = M["sd"]["value"] / math.sqrt(n)
        power = float(norm.cdf(margin / se - zq(1 - 0.025 / k)))
    fz = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "problem": key, "configs": names, "reference": "ref (orig + milp)",
          "alternatives_k": k, "margin_nat_per_token": margin, "margin_basis": M["prior"], "sd_assumed": M["sd"],
          "n_required": n_req, "n_windows": n, "power_at_true_difference_0": power, "cap": N_CAP,
          "segments": {s: {"split": "validation" if s == "V" else "test", "first_token": M["first"][s], "windows": n // 2}
                       for s in ("V", "T")},
          "native_subset": [*range(NATIVE_SUBSET_PER_SEG), *range(n // 2, n // 2 + NATIVE_SUBSET_PER_SEG)] if n else [],
          "decision": {"primary": f"non-inferiority: upper bound of the two-sided {100 * (1 - 0.05 / max(k, 1)):.2f}% paired "
                                  f"CI of dNLL(alt) - dNLL(ref) < margin", "secondary": "equivalence: that CI inside +-margin",
                       "backend": "GPU PyTorch replay (primary, labelled); native subset reported separately"},
          "configs_sha256": {nm: sha256(d / "configs" / f"{nm}.json") for nm in names}}
    jdump(d / "freeze.json", fz)
    log(json.dumps({kk: fz[kk] for kk in ("configs", "alternatives_k", "margin_nat_per_token", "n_required", "n_windows",
                                          "power_at_true_difference_0")}))


def tokenize(kind, split):
    if kind == "qwen35":
        import qwen35_common as q35
        return q35.wikitext_ids(split)
    import pyarrow.parquet as pq
    sys.path.insert(0, str(HERE / "gpt2_assets" / "python_deps"))      # the transformers qwen3_swiglu_common uses
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HERE / "qwen3_assets" / "model")
    text = "\n\n".join(pq.read_table(HERE / "gpt2_assets/data" / f"{split}-00000-of-00001.parquet")["text"].to_pylist())
    return tok.encode(text, add_special_tokens=False)


def stage_c_windows(args, out, log):
    key = args.problem
    M = MODELS[key]
    d = out / "C" / key
    fz = jload(d / "freeze.json")
    if (d / "tokens_eval.npy").exists():
        raise SystemExit("windows exist")
    used = QWEN35_USED if M["kind"] == "qwen35" else {s: [r[:2] for r in v] for s, v in M["used"].items()}
    toks, segs = [], {}
    for s, seg in fz["segments"].items():
        ids = tokenize(M["kind"], seg["split"])
        a, b = seg["first_token"], seg["first_token"] + seg["windows"] * SEQ
        if b > len(ids):
            raise SystemExit(f"{s}: {len(ids)} tokens < {b}")
        for lo, hi in used[seg["split"]]:
            if a < hi and b > lo:
                raise SystemExit(f"{s}: overlaps earlier tokens [{lo}, {hi})")
        toks.append(np.asarray(ids[a:b], np.int64).reshape(seg["windows"], SEQ))
        segs[s] = {**seg, "tokens": [a, b], "split_tokens": len(ids)}
    np.save(d / "tokens_eval.npy", np.concatenate(toks))
    jdump(d / "windows.json", {"segments": segs, "sha256": sha256(d / "tokens_eval.npy"), "used_checked": used,
                               "frozen_before": fz["time"]})
    log(f"windows {segs}")


class LMReplay:
    """GPU (or CPU) replay of a whole configuration of the searched projections."""

    def __init__(self, key, device, work):
        import copy
        import torch
        from qwen3_swiglu_common import Quantizers
        M = MODELS[key]
        self.torch = torch
        if device != "cpu":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        self.device = device
        if M["kind"] == "qwen35":
            import qwen35_common as q35
            hf = q35.load_hf(torch.float32)
            self.ad = q35.TextExport(hf, SEQ).eval().to(device)
            mp = jload(M["mapping"])
            self.param = {n: v["param"] for n, v in mp.items()}
        else:                                   # qwen3_swiglu_common.load_model without qwen3_prepare's onnxruntime import
            import qwen3_swiglu_common as q3
            from transformers import Qwen3ForCausalLM
            hf = Qwen3ForCausalLM.from_pretrained(q3.MODEL_DIR, dtype=torch.float32, attn_implementation="eager").eval()
            hf.config.use_cache = False

            class CausalLM(torch.nn.Module):    # qwen3_prepare.CausalLM: constant 4-D causal mask
                def __init__(self, model, seq):
                    super().__init__()
                    self.model = model
                    mask = torch.full((seq, seq), torch.finfo(torch.float32).min).triu(1)
                    self.register_buffer("mask", mask[None, None], persistent=False)

                def forward(self, input_ids):
                    return self.model(input_ids=input_ids, attention_mask=self.mask, use_cache=False).logits
            self.ad = CausalLM(hf, SEQ).eval().to(device)
            self.param = jload(M["mapping"])
        self.mods = dict(hf.named_modules())
        self.ranges = jload(M["ranges"])
        self.q = Quantizers(work, formats=("p4e1", "p6e1", "p8e1"))
        for f, lut in self.q.posit.items():
            bad, n = lut.verify()
            if bad:
                raise SystemExit(f"posit LUT {f}: {bad} of {n} differ")
        self.lut = {}
        for f, lut in self.q.posit.items():
            self.lut[f] = copy.copy(lut)
            self.lut[f].t_values, self.lut[f].t_bounds = lut.t_values.to(device), lut.t_bounds.to(device)
        self.orig = {n: self.mod(n).weight.data.cpu().clone() for n in self.param}
        self.cache, self.act = {}, {}
        for n in self.param:
            m = self.mod(n)
            m.register_forward_pre_hook(lambda mod, i, n=n: (self.qact(i[0], self.act[n], self.ranges[n]["x"]),)
                                        if self.act.get(n, "fp32") != "fp32" else None)
            m.register_forward_hook(lambda mod, i, o, n=n: self.qact(o, self.act[n], self.ranges[n]["y"])
                                    if self.act.get(n, "fp32") != "fp32" else None)

    def mod(self, n):
        return self.mods[self.param[n].removesuffix(".weight")]

    def qact(self, t, fmt, rng):
        torch = self.torch
        import eval_random_mixed_precision as erm
        if fmt.startswith("int"):
            bits = int(fmt[3:])
            qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
            scale, zp = erm.int_qparams(float(rng[0]), float(rng[1]), bits)
            s = torch.tensor(np.float32(scale), dtype=t.dtype, device=t.device)
            z = torch.tensor(np.float32(zp), dtype=t.dtype, device=t.device)
            return (torch.clamp(torch.round(t / s + z), qmin, qmax) - z) * s
        if fmt == "f16":
            return t.to(torch.float16).to(t.dtype)
        if fmt == "bf16":
            return t.to(torch.bfloat16).to(t.dtype)
        return self.lut[fmt](t)

    def set(self, cfg):
        for n in self.param:
            w, a = (cfg[n]["w"], cfg[n]["a"]) if n in cfg else ("fp32", "fp32")
            w = w or "fp32"
            if (n, w) not in self.cache:
                self.cache[(n, w)] = self.orig[n] if w == "fp32" else self.q.weight(self.orig[n], w)
            self.mod(n).weight.data = self.cache[(n, w)].to(self.device)
            self.act[n] = a

    def ce(self, toks, batch=8):
        torch = self.torch
        x = torch.from_numpy(toks).to(self.device)
        r = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                z = self.ad(x[i:i + batch])
                r.append(torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(),
                                                           x[i:i + batch, 1:].reshape(-1), reduction="none")
                         .view(z.shape[0], -1).double().sum(1))
        return torch.cat(r).cpu().numpy()


def stage_c_check(args, out, log):
    """Replay vs earlier native per-window CE on earlier windows (FP32 and the reference knee)."""
    key = args.problem
    M = MODELS[key]
    d = out / "C" / key
    rp = LMReplay(key, args.device, d)
    res = {}
    if M["kind"] == "qwen3":
        h = jload(HERE / "runs/qwen3_branch_ablation_20261014/heldout.json")["results"]
        toks = np.load(Q3P / "test_tokens.npy")
        from mp_auto import config_hash
        ref = jload(d / "configs" / "ref.json")
        prior = {"fp32": ({}, np.array(h["fp32"]["per_window_mean_nll"])),
                 "knee": (ref, np.array(h["knee_total"]["per_window_mean_nll"]))}
        res["ref_is_prior_knee"] = config_hash(ref) == h["knee_total"]["config_hash"]
    else:
        nd = HERE / "runs/qwen35_multipoint_20261015"
        toks = np.load(nd / "tokens_eval.npy")
        sc = MODELS[key]["scope"]
        fmk = jload(nd / "selected_points.json")[f"{sc}_knee"] if (nd / "selected_points.json").exists() else None
        knee_f = jload(nd / fmk["formats_file"]) if fmk and "formats_file" in fmk else jload(d / "configs" / "ref.json")
        prior = {"fp32": ({}, np.array(jload(nd / "native" / "fp32.json")["summed_ce"]) / (SEQ - 1)),
                 "knee": (knee_f, np.array(jload(nd / "native" / f"{sc}_knee.json")["summed_ce"]) / (SEQ - 1))}
        from mp_auto import config_hash
        res["ref_is_prior_knee"] = config_hash(jload(d / "configs" / "ref.json")) == config_hash(knee_f)
    sims = {}
    for nm, (cfg, nat) in prior.items():
        rp.set(cfg)
        t0 = time.time()
        s = rp.ce(toks) / (SEQ - 1)
        sims[nm] = s
        res[nm] = {"replay_nll": float(s.mean()), "native_nll": float(nat.mean()), "max_abs_window": float(np.abs(s - nat).max()),
                   "seconds": time.time() - t0, "windows": int(len(toks))}
    dn = prior["knee"][1] - prior["fp32"][1]
    ds = sims["knee"] - sims["fp32"]
    res["dnll"] = {"replay": float(ds.mean()), "native": float(dn.mean()), "per_window_corr": float(np.corrcoef(ds, dn)[0, 1]),
                   "difference_mean": float((ds - dn).mean()), "difference_se": float((ds - dn).std(ddof=1) / np.sqrt(len(ds)))}
    jdump(d / "replay_check.json", res)
    log(json.dumps(res))


def stage_c_replay(args, out, log):
    key = args.problem
    d = out / "C" / key
    fz = jload(d / "freeze.json")
    for nm, h in fz["configs_sha256"].items():
        if sha256(d / "configs" / f"{nm}.json") != h:
            raise SystemExit(f"{nm} changed after the freeze")
    f = d / "replay.json"
    res = jload(f) if f.exists() else {"device": args.device, "summed_ce": {}, "seconds": {}}
    toks = np.load(d / "tokens_eval.npy")
    rp = LMReplay(key, args.device, d)
    for nm in ["fp32", *fz["configs"]]:
        if nm in res["summed_ce"]:
            continue
        t0 = time.time()
        rp.set({} if nm == "fp32" else jload(d / "configs" / f"{nm}.json"))
        res["summed_ce"][nm] = rp.ce(toks).tolist()
        res["seconds"][nm] = time.time() - t0
        jdump(f, res)
        log(f"{nm}: NLL {np.sum(res['summed_ce'][nm]) / (len(toks) * (SEQ - 1)):.5f} ({res['seconds'][nm]:.0f}s)")


def stage_c_native(args, out, log):
    """Native build + run (onnx-mlir, as the model's evaluation scripts) of every frozen configuration on the subset."""
    from evaluate_native_search_chunked import chunked_run_so
    from run_native_bc_pipeline import SLOT_INPUT, cross_entropy, erm
    import onnx
    key = args.problem
    M = MODELS[key]
    d = out / "C" / key
    fz = jload(d / "freeze.json")
    toks = np.load(d / "tokens_eval.npy")[fz["native_subset"]]
    nd = d / "native"
    nd.mkdir(exist_ok=True)
    np.save(nd / "tokens.npy", toks)
    if M["kind"] == "qwen35":
        from run_qwen35_bc import native_import
        from run_native_bc_pipeline import qwen35_spec
        tools, _, base = native_import(Q35, log)
        graph = onnx.load(str(Q35 / "model.onnx"), load_external_data=False)
        spec = qwen35_spec(nd / "unused", Q35)
    else:
        from run_native_bc_pipeline import imported_model, qwen3_spec
        tools = erm.Tools()
        base = imported_model(Q3T)
        graph = onnx.load(str(Q3P / "model.onnx"), load_external_data=False)
        spec = qwen3_spec(nd / "unused")
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in jload(M["ranges"]).items():
        n = by_name[node]
        for slot, r in slots.items():
            ranges[n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]] = tuple(r)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    posit_obj = erm.build_posit_runtime(tools, nd)
    for nm in ["fp32", *fz["configs"]]:
        f = nd / f"{nm}.json"
        if f.exists():
            continue
        t0 = time.time()
        target = nd / nm
        target.mkdir(exist_ok=True)
        env = None
        if nm != "fp32":
            fm = jload(d / "configs" / f"{nm}.json")
            env = erm.node_formats_env({n["name"]: fm.get(n["name"], {"w": None, "a": "fp32"}) for n in spec.nodes},
                                       spec.nodes, ranges)
        so, info = erm.build_so(tools, target, nm, base, env, posit_obj if env else None, opt="O0", krnl_opt="O3",
                                constants_to_file=True, bytecode=True, uses_posit=None if env else False,
                                uses_f16=None if env else False)
        if info["warnings"]:
            raise RuntimeError(f"{nm}: compilation warnings {info['warnings']}")
        for p in target.iterdir():
            if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
                p.unlink()
        z, run = chunked_run_so(args.chunks)(tools, so, nd / "tokens.npy", target / "logits.npy", 36000)
        (target / "logits.npy").unlink()
        ce = [float(cross_entropy(z[i, :-1], toks[i, 1:]).sum()) for i in range(len(toks))]
        for p in (so, target / "model.constants.bin"):
            p.unlink(missing_ok=True)
        jdump(f, {"config": nm, "summed_ce": ce, "finite": bool(np.isfinite(z).all()), "build_steps": info.get("steps"),
                  "run_wall_s": run["wall_seconds"], "stage_s": time.time() - t0})
        log(f"native {nm}: NLL {sum(ce) / (len(toks) * (SEQ - 1)):.5f} ({time.time() - t0:.0f}s)")


def stage_c_analyze(args, out, log):
    key = args.problem
    d = out / "C" / key
    fz = jload(d / "freeze.json")
    tr = jload(d / "replay.json")["summed_ce"]
    n = fz["n_windows"]
    h = n // 2
    rng = np.random.default_rng(0)
    draws = np.concatenate([rng.integers(0, h, (BOOT, h)), h + rng.integers(0, h, (BOOT, h))], 1)
    T = SEQ - 1
    ce = {k: np.array(v) for k, v in tr.items()}
    k = fz["alternatives_k"]
    lvl = 0.05 / max(k, 1)

    def st(x, lv=0.05):
        b = x[draws].mean(1)
        return {"mean": float(x.mean()), "ci": [float(np.percentile(b, 100 * lv / 2)), float(np.percentile(b, 100 * (1 - lv / 2)))],
                "level": 1 - lv, "V": float(x[:h].mean()), "T": float(x[h:].mean()), "sd": float(x.std(ddof=1))}
    res = {"backend": "GPU PyTorch replay (corrected device quantizer, TF32 off)", "fp32": {
        "nll": float(ce["fp32"].sum() / (n * T)), "ppl": float(np.exp(ce["fp32"].sum() / (n * T)))}, "configs": {}, "comparisons": {}}
    m = fz["margin_nat_per_token"]
    for nm in fz["configs"]:
        x = (ce[nm] - ce["fp32"]) / T
        res["configs"][nm] = {"runs": fz["configs"][nm]["runs"], "nll": float(ce[nm].sum() / (n * T)),
                              "ppl": float(np.exp(ce[nm].sum() / (n * T))), "dnll": st(x)}
    for nm in fz["configs"]:
        if nm == "ref":
            continue
        x = (ce[nm] - ce["ref"]) / T
        s = st(x, lvl)
        lo, hi = s["ci"]
        res["comparisons"][f"{nm}-ref"] = {**s, "non_inferior": bool(hi < m), "equivalent": bool(-m < lo and hi < m),
                                           "decision": ("non-inferior and equivalent" if -m < lo and hi < m else
                                                        "non-inferior" if hi < m else
                                                        "inferior (CI above margin)" if lo > m else "not shown"),
                                           "ci95_unadjusted": st(x)["ci"]}
    nat = d / "native"
    if nat.exists() and (nat / "fp32.json").exists():
        sub = fz["native_subset"]
        nc = {nm: np.array(jload(nat / f"{nm}.json")["summed_ce"]) for nm in ["fp32", *fz["configs"]] if (nat / f"{nm}.json").exists()}
        r = {"subset": sub}
        for nm in nc:
            r[nm] = {"native_nll": float(nc[nm].sum() / (len(sub) * T)),
                     "replay_nll_same_windows": float(ce[nm][sub].sum() / (len(sub) * T)),
                     "max_abs_window_per_token": float(np.abs(nc[nm] - ce[nm][sub]).max() / T)}
        for nm in nc:
            if nm not in ("fp32", "ref") and "ref" in nc:
                xn = (nc[nm] - nc["ref"]) / T
                xr = (ce[nm][sub] - ce["ref"][sub]) / T
                r[f"{nm}-ref"] = {"native_mean": float(xn.mean()), "native_se": float(xn.std(ddof=1) / np.sqrt(len(xn))),
                                  "replay_mean_same_windows": float(xr.mean())}
        res["native_subset"] = r
    jdump(d / "analysis.json", res)
    log(json.dumps({kk: res[kk] for kk in ("fp32", "configs", "comparisons")})[:3000])


CNN = {"resnet18": {"loss_margin": 0.10 * (1.2786015835559341 - 1.2582988739013672),
                    "top1_drop": 0.706 - 0.701},
       "efficientnet_b0": {"loss_margin": 0.10 * (0.9516699012004123 - 0.9326018691062927),
                           "top1_drop": 0.774 - 0.774}}
HTQ_POOL = Path("/home/hoju/test/HTQ/imagenet1k_2050/images")
REPO_IMAGES = HERE / "runs/mobilenetv2_timm_ra_htq_10cal_1000eval/images"


def stage_c_cnn(args, out, log):
    """CNN: native (as run_native_auto) FP32 / reference / alternative on the 1000 held-out images of the HTQ pool that
    this repository never used; configurations frozen first."""
    from mp_auto import config_hash
    from evaluate_native_search_chunked import chunked_run_so
    from run_native_bc_pipeline import CNNS, SLOT_INPUT, cnn_spec, cross_entropy, erm
    import onnx
    key = args.problem
    d = out / "C" / key
    d.mkdir(parents=True, exist_ok=True)
    B = out / "B" / key
    runs = {f.stem.replace("search_", ""): jload(f) for f in sorted(B.glob("search_*.json"))}
    if not (d / "freeze.json").exists():
        cf = {}
        for r in ("milp_orig", "lpfirst_orig", "milp_grouped", "lpfirst_grouped"):
            h = config_hash(runs[r]["selected_formats"])
            nm = "ref" if h == config_hash(runs["milp_orig"]["selected_formats"]) else f"alt_{h[:8]}"
            cf.setdefault(nm, {"hash": h, "runs": []})["runs"].append(r)
            (d / "configs").mkdir(exist_ok=True)
            jdump(d / "configs" / f"{nm}.json", runs[r]["selected_formats"])
        pool = jload(HTQ_POOL / "manifest.json")["images"]
        used = {e["original_name"] for e in jload(REPO_IMAGES / "manifest.json")["images"]}
        held = [e for e in pool if not e["calibration"]]
        new = [e for e in held if e["original_name"] not in used]
        if len(new) < 1000:
            raise SystemExit(f"only {len(new)} unused images")
        new = new[:1000]
        jdump(d / "images.json", {"images": new, "pool": str(HTQ_POOL), "excluded_used": len(used),
                                  "excluded_pool_calibration": sum(e["calibration"] for e in pool)})
        k = len(cf) - 1
        m = CNN[key]["loss_margin"]
        from scipy.stats import norm
        se = 0.21 / math.sqrt(1000)
        jdump(d / "freeze.json", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "configs": cf, "alternatives_k": k,
                                  "loss_margin": m, "top1_margin": 0.10 * CNN[key]["top1_drop"] or None,
                                  "n_images": 1000, "power_loss_at_0": float(norm.cdf(m / se - zq(1 - 0.025 / max(k, 1)))),
                                  "images_sha256": sha256(d / "images.json"),
                                  "configs_sha256": {nm: sha256(d / "configs" / f"{nm}.json") for nm in cf}})
        log(f"frozen {cf}")
    fz = jload(d / "freeze.json")
    model_dir = HERE / "../imagenet100/model_dataset" / CNNS[key]
    meta = jload(model_dir / "metadata.json")
    imgs = jload(d / "images.json")["images"]
    spec = cnn_spec(key, REPO_IMAGES, 1, d / "unused", with_references=False)
    labels = np.array([e["label"] for e in imgs])
    nd = d / "native"
    nd.mkdir(exist_ok=True)
    if not (nd / "eval_inputs.npy").exists():
        np.save(nd / "eval_inputs.npy", erm.preprocess_by_source(HTQ_POOL, {"images": imgs}, meta).astype(np.float32))
    tools = erm.Tools()
    base = nd / f"{erm.SYMBOL_POSTFIX}.mlirbc"
    if not base.exists():
        text = nd / f"{erm.SYMBOL_POSTFIX}.onnx.mlir"
        erm.run([tools.onnx_mlir, "--EmitONNXBasic", "-o", nd / erm.SYMBOL_POSTFIX,
                 "--mlir-elide-resource-strings-if-larger=1000000000",
                 "--mlir-elide-elementsattrs-if-larger=1000000000", model_dir / "model.onnx"])
        erm.run([tools.opt, text, "--emit-bytecode", "-o", base])
        text.unlink()
    graph = onnx.load(str(model_dir / "model.onnx"))
    by_name = {n.name: n for n in graph.graph.node}
    rfile = HERE / f"runs/native_auto_2d_20261011/{key}/ranges_by_point.json"
    ranges = {}
    for node, slots in jload(rfile).items():
        n = by_name[node]
        for slot, r in slots.items():
            ranges[n.output[0] if slot == "y" else n.input[SLOT_INPUT[slot]]] = tuple(r)
    os.environ["MIXEDP_WEIGHT_PER_CHANNEL"] = "1"
    posit_obj = erm.build_posit_runtime(tools, nd)
    for nm in ["fp32", *fz["configs"]]:
        f = nd / f"{nm}.json"
        if f.exists():
            continue
        if nm != "fp32" and sha256(d / "configs" / f"{nm}.json") != fz["configs_sha256"][nm]:
            raise SystemExit(f"{nm} changed after the freeze")
        t0 = time.time()
        target = nd / nm
        target.mkdir(exist_ok=True)
        env = None
        if nm != "fp32":
            fm = jload(d / "configs" / f"{nm}.json")
            env = erm.node_formats_env({n["name"]: fm.get(n["name"], {"w": None, "a": "fp32"}) for n in spec.nodes},
                                       spec.nodes, ranges)
        so, info = erm.build_so(tools, target, nm, base, env, posit_obj if env else None, opt=spec.eval_levels[1],
                                krnl_opt=spec.eval_levels[0], constants_to_file=True, bytecode=True)
        if info["warnings"]:
            raise RuntimeError(f"{nm}: compilation warnings {info['warnings']}")
        for p in target.iterdir():
            if p.suffix in (".mlir", ".mlirbc", ".ll", ".bc"):
                p.unlink()
        z, run = chunked_run_so(args.chunks)(tools, so, nd / "eval_inputs.npy", target / "logits.npy", 36000)
        (target / "logits.npy").unlink()
        finite = np.isfinite(z).all(1)
        ce = cross_entropy(np.where(finite[:, None], z, 0.0), labels)
        top1 = ((z.argmax(1) == labels) & finite)
        jdump(f, {"config": nm, "per_image_ce": ce.tolist(), "per_image_top1": top1.astype(int).tolist(),
                  "loss": float(ce.mean()), "top1": float(top1.mean()), "nonfinite_images": int((~finite).sum()),
                  "run_wall_s": run["wall_seconds"], "stage_s": time.time() - t0})
        log(f"{key} {nm}: top1 {top1.mean():.4f} loss {ce.mean():.5f} ({time.time() - t0:.0f}s)")
    res = {"configs": {}, "comparisons": {}}
    R = {nm: jload(nd / f"{nm}.json") for nm in ["fp32", *fz["configs"]]}
    rng = np.random.default_rng(0)
    draws = rng.integers(0, 1000, (BOOT, 1000))
    k = fz["alternatives_k"]
    lv = 0.05 / max(k, 1)

    def st(x, level):
        b = x[draws].mean(1)
        return {"mean": float(x.mean()), "ci": [float(np.percentile(b, 100 * level / 2)), float(np.percentile(b, 100 * (1 - level / 2)))],
                "level": 1 - level}
    for nm in R:
        res["configs"][nm] = {"top1": R[nm]["top1"], "loss": R[nm]["loss"]}
        if nm != "fp32":
            res["configs"][nm]["dloss_vs_fp32"] = st(np.array(R[nm]["per_image_ce"]) - np.array(R["fp32"]["per_image_ce"]), 0.05)
            res["configs"][nm]["dtop1_vs_fp32"] = st(np.array(R[nm]["per_image_top1"]) - np.array(R["fp32"]["per_image_top1"]), 0.05)
    for nm in fz["configs"]:
        if nm == "ref":
            continue
        dl = st(np.array(R[nm]["per_image_ce"]) - np.array(R["ref"]["per_image_ce"]), lv)
        dt = st(np.array(R[nm]["per_image_top1"]) - np.array(R["ref"]["per_image_top1"]), lv)
        m = fz["loss_margin"]
        lo, hi = dl["ci"]
        dl["decision"] = ("non-inferior and equivalent" if -m < lo and hi < m else "non-inferior" if hi < m else
                          "inferior (CI above margin)" if lo > m else "not shown")
        res["comparisons"][f"{nm}-ref"] = {"loss": dl, "top1": dt, "loss_margin": m, "top1": dt}
    jdump(d / "analysis.json", res)
    log(json.dumps(res)[:2000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--problem", required=True, choices=list(MODELS) + list(CNN))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunks", type=int, default=4)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    t0 = time.time()
    (out / "logs").mkdir(exist_ok=True)
    logf = open(out / "logs" / f"{args.stage}_{args.problem}.log", "a")

    def log(msg):
        line = f"[{time.time() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()
    status = "ok"
    try:
        globals()[f"stage_{args.stage}"](args, out, log)
    except BaseException as e:
        status = f"failed: {type(e).__name__}: {str(e)[:300]}"
        raise
    finally:
        tf = out / "timing.json"
        timing = jload(tf) if tf.exists() else {"runs": []}
        timing["runs"].append({"tag": f"{args.stage}_{args.problem}", "status": status, "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
                               "script_sha256": sha256(HERE / "run_lp_pipeline_eval.py")})
        jdump(tf, timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
