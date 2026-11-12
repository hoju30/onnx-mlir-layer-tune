#!/usr/bin/env python3
"""OLMoE fixed route: does the single-point candidate ranking (stage 1's 1080 candidates, 2583 same-resource pairs)
reproduce across independent texts, and is BC stably wrong on the reproducible part? Pre-registered in plan.json.

Data: 32 WikiText-2 articles (16 validation, 16 test) never touched by an earlier OLMoE window, one segment of 4
consecutive 128-token windows per article; 16 segments form the diagnostic set D, 16 the confirmation set V (8 + 8 per
split). Every window uses its own FP32 route. BC scores, ranges, candidates, resource model and single-point semantics
are stage 1's; the shared (corrected) quantizer; no perturbation.

Stages (one --out-dir):
  plan        articles, overlap checks, sampling, D/V assignment, window tokens, plan.json (+ hashes)
  pilot       time / memory of one candidate group on stage 1's diagnostic windows (no loss stored)
  capture     --set D|V: FP32 routes of every window
  truth       --set D|V: FP32 + all 1080 candidates on every window (windows in chunks)
  check       determinism (fixed candidates x D's first 2 segments, twice) and cached-layer replay vs full forward
  analyze_d   D results and the frozen confirmation list (freeze_V.json)
  analyze     V results against the frozen list, all pairs, the 31 tracked pairs, traffic diagnostics
"""
import argparse
import bisect
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S1 = HERE / "runs/olmoe_fixed_route_20261015"
NUM = HERE / "runs/olmoe_bc_numerics_20261016"
SEQ, T_PRED = 128, 127
WPS = 4                                       # windows per segment
NSEG = 32
CHUNK = 32                                    # windows per replay chunk (memory)
BOOT = 10000
Q_D, Q_V = 0.10, 0.05


def jdump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n")


def jload(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def used_spans():
    used = {"validation": [], "test": []}
    s1 = jload(S1 / "plan.json")["data"]
    for n in ("calibration", "diagnostic", "check"):
        used[s1[n]["split"]] += [tuple(x) for x in s1[n]["char_spans"]]
    for v in jload(HERE / "runs/olmoe_multipoint_20261015/plan.json")["eval_windows"]["segments"].values():
        used[v["split"]].append(tuple(v["char_spans"]))
    for v in jload(HERE / "runs/solver_regression_20261016/plan.json")["olmoe_eval"]["windows"]["segments"].values():
        used[v["split"]].append(tuple(v["char_spans"]))
    for v in jload(HERE / "runs/olmoe_search_final_20261016/C/windows.json")["segments"].values():
        used[v["split"]].append(tuple(v["char_spans"]))
    used["validation"].append(tuple(jload(NUM / "B/windows.json")["char_spans"]))
    return used


def articles(split, tok):
    """Top-level WikiText articles (' = Title = ' lines) as (title, char start, char end, token start, token end) in
    olmoe_common.wikitext(split) and its tokenization (OLMoE tokenizer, no special tokens)."""
    import olmoe_common as oc
    import pyarrow.parquet as pq
    lines = pq.read_table(HERE / "gpt2_assets/data" / f"{split}-00000-of-00001.parquet")["text"].to_pylist()
    text = oc.wikitext(split)
    starts, pos = [], 0
    for line in lines:
        if re.match(r"^ = [^=].* = \n$", line):
            starts.append((line.strip(" =\n"), pos))
        pos += len(line) + 2
    if pos - 2 != len(text):
        raise SystemExit("article offsets do not match the joined text")
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, off = enc["input_ids"], enc["offset_mapping"]
    ts = [o[0] for o in off]
    out = []
    for k, (title, a) in enumerate(starts):
        b = starts[k + 1][1] if k + 1 < len(starts) else len(text)
        out.append({"title": title, "chars": [a, b], "tokens": [bisect.bisect_left(ts, a), bisect.bisect_left(ts, b)]})
    return out, ids, off


# --------------------------------------------------------------------------- #
def stage_plan(args, out, log):
    import olmoe_common as oc
    if (out / "plan.json").exists():
        raise SystemExit("plan.json exists")
    tok = oc.tokenizer()
    used = used_spans()
    rng = np.random.default_rng(0)
    segs, toks = [], {"D": [], "V": []}
    avail = {}
    for split in ("validation", "test"):
        arts, ids, off = articles(split, tok)
        free = [a for a in arts if not any(a["chars"][0] < u1 and a["chars"][1] > u0 for u0, u1 in used[split])
                and a["tokens"][1] - a["tokens"][0] >= WPS * SEQ]
        avail[split] = {"articles": len(arts), "free_long_enough": len(free),
                        "excluded_overlap": sum(any(a["chars"][0] < u1 and a["chars"][1] > u0 for u0, u1 in used[split]) for a in arts)}
        pick = sorted(rng.choice(len(free), NSEG // 2, replace=False).tolist())
        order = rng.permutation(len(pick))                       # first 8 -> D, last 8 -> V
        for r, k in enumerate(pick):
            a = free[k]
            grp = "D" if order[r] < NSEG // 4 else "V"
            t0 = int(a["tokens"][0] + rng.integers(0, a["tokens"][1] - a["tokens"][0] - WPS * SEQ + 1))
            w = np.asarray(ids[t0:t0 + WPS * SEQ], np.int64).reshape(WPS, SEQ)
            cs = [int(off[t0][0]), int(off[t0 + WPS * SEQ - 1][1])]
            if any(cs[0] < u1 and cs[1] > u0 for u0, u1 in used[split]) or not (a["chars"][0] <= cs[0] and cs[1] <= a["chars"][1]):
                raise SystemExit("segment check failed")
            toks[grp].append(w)
            segs.append({"group": grp, "split": split, "title": a["title"], "article_chars": a["chars"],
                         "window_chars": cs, "tokens": [t0, t0 + WPS * SEQ], "index_in_group": len(toks[grp]) - 1})
    for g in ("D", "V"):
        (out / g).mkdir(exist_ok=True)
        np.save(out / g / "tokens.npy", np.concatenate(toks[g]))
    titles = [s["title"] for s in segs]
    if len(set(titles)) != len(titles):
        raise SystemExit("an article appears twice")
    jdump(out / "segments.json", segs)
    from run_olmoe_bc_numerics import cpu_gpu_subset
    cands = jload(S1 / "candidates.json")
    files = {k: S1 / k for k in ("candidates.json", "table.json", "tables.json", "ranges.json", "freeze.json")}
    jdump(out / "plan.json", {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "questions": ["does the cross-text average ranking of single-point candidates reproduce on independent texts",
                      "is BC stably wrong on the reproducible part", "is it concentrated in experts, modes, traffic"],
        "background_verified": {"candidates": "9 experts, 1080 candidates (w, a, wa x 360), 2583 same-resource pairs - "
                                              "confirmed (stage 1 candidates.json; pairs rebuilt as stage 1)",
                                "tracked_31": "the 31 pairs stable-wrong on both A and B of runs/olmoe_bc_numerics_20261016 "
                                              "are a tracking table only; they do not enter sampling or thresholds"},
        "fixed": "BC scores and formula, calibration, activation ranges / scale / zero-point, posit LUTs, candidate formats, "
                 "resource model, single-point semantics (stage 1 truth), fixed route; shared corrected quantizer; no "
                 "perturbation; GPU FP32, TF32 off; CPU (if used) OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE",
        "data": {"source": "WikiText-2 raw (gpt2_assets/data parquet, the only local splits: validation 60 and test 62 "
                           "articles); article = text from a top-level ' = Title = ' line to the next; documents are "
                           "Wikipedia articles, so different articles are different documents",
                 "availability": avail,
                 "exclusion": "an article is excluded if any character of it overlaps any earlier OLMoE window set "
                              "(stage 1 calibration / diagnostic / check, multipoint, solver regression, search final C, "
                              "bc numerics B) or if it has fewer than 512 tokens",
                 "sampling": "numpy seed 0: per split 16 of the free articles uniformly without replacement, a random "
                             "permutation assigns 8 to D and 8 to V; one segment per article = 4 consecutive 128-token "
                             "windows starting at a uniformly random token of the article (the segment stays inside the "
                             "article); no use of traffic, BC or loss",
                 "segments": "segments.json", "D_tokens_sha256": sha256(out / "D/tokens.npy"),
                 "V_tokens_sha256": sha256(out / "V/tokens.npy"),
                 "limit": "WikiText articles are on different subjects but come from one source; no other source is mixed in"},
        "evaluation": "per window: FP32 and all 1080 single-point candidates (each quantizes only its listed projection(s), "
                      "rest FP32; w without activation QDQ), the window's own FP32 route, replay from the expert's layer "
                      "with the window's FP32 hidden state (same process); summed CE over 127 predictions and the "
                      "difference to FP32",
        "checks": "determinism: the 54 fixed candidates of runs/olmoe_bc_numerics_20261016 (gate / up int4 per expert and "
                  "mode) on D's first 2 segments replayed twice; cached-layer replay vs full forward from the embedding "
                  "for 3 candidates (one per mode, layer 8)",
        "target": "average loss difference per model token over the target distribution: segment value = mean over its 4 "
                  "windows of (CE_A - CE_B) / 127 (equal tokens), pair value = mean over the 16 segments of a group; "
                  "signed, no absolute values, negative damages kept",
        "statistics": {"resampling": f"segments (= articles) with replacement, {BOOT} draws, the same indices for every "
                                     "candidate and pair; D seed 2, V seed 3; windows of a segment are never resampled alone",
                       "p_value": "two-sided bootstrap p = min(1, 2 min(P*(mean <= 0), P*(mean >= 0))); one-sided in the "
                                  "predicted direction for V",
                       "intervals": "95% percentile bootstrap; t interval (15 df) reported alongside",
                       "loso": "leave-one-segment-out: the 16 means without one segment; robust if all have the full mean's sign",
                       "dominance": "largest |segment value| / sum of |segment values|",
                       "ties": "BC tie (equal scores) excluded and counted; a pair with every segment value 0 has no direction"},
        "discovery_D": f"identified = Benjamini-Hochberg at q = {Q_D} over all 2583 two-sided p (Benjamini-Yekutieli "
                       "count reported as the dependence-robust sensitivity); predicted direction = sign of D's mean; "
                       "identified pairs split into BC-agree / BC-disagree; the rest 'not reliably distinguished'",
        "confirmation_V": {"list": "frozen in freeze_V.json before any V candidate loss is read",
                           "confirmed": f"one-sided p in the predicted direction, BH at q = {Q_V} within the identified set",
                           "opposite": f"one-sided p in the opposite direction, BH at q = {Q_V} within the identified set",
                           "same_insufficient": "V mean in the predicted direction, not confirmed",
                           "undetermined": "V mean opposite, not significant (or zero)",
                           "note": "'same direction' is not 'reproduced'; 'not significant' is not 'no difference'"},
        "reproducible_bc_error": "identified in D with BC-disagree and confirmed in V; reported with the number of "
                                 "distinct experts and candidates involved",
        "secondary": ["all 2583 pairs: D vs V sign agreement, Spearman of means, coverage of V means by D's 95% interval",
                      "BC agreement with D, V and pooled D+V signs (pooled is secondary, never a confirmation)",
                      "per expert, mode, pair class", "the 31 tracked pairs", "traffic diagnostics (descriptive)"],
        "traffic": "per segment and expert: routed tokens, windows with >= 1 token, routing weight mean / sum; related to "
                   "pair variability and identification only descriptively; no segment is dropped, no candidate reselected",
        "multiplicity": "BH within D over all pairs (q 0.10) and within the identified set in V (q 0.05); nothing else is "
                        "tested",
        "limits": {"gpu_hours": 3.0, "stop": "if the pilot projects more than 75 min per group (D or V), the plan is "
                                             "amended before any candidate loss of D or V is computed"},
        "determinism_subset": cpu_gpu_subset(cands),
        "inputs_sha256": {k: sha256(v) for k, v in files.items()},
        "scripts_sha256": {s: sha256(HERE / s) for s in ("run_olmoe_crosstext.py", "run_olmoe_bc_numerics.py",
                                                          "run_olmoe_fixed_route.py", "olmoe_common.py",
                                                          "qwen3_swiglu_common.py")}})
    log(f"plan: {avail}; D {len(toks['D'])} V {len(toks['V'])} segments")


def stage_capture(args, out, log):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, head_ce
    gpu_exact(args.device)
    d = out / args.set
    if (d / "routes_manifest.json").exists():
        return
    (d / "routes").mkdir(exist_ok=True)
    cfg, ck = oc.config(), oc.Checkpoint()
    toks_np = np.load(d / "tokens.npy")
    g = ck.global_(args.device, torch.float32)
    cos, sin = oc.rope(cfg, SEQ, args.device, torch.float32)
    hashes, ces = [], []
    with torch.no_grad():
        for a in range(0, len(toks_np), CHUNK):
            tk = torch.from_numpy(toks_np[a:a + CHUNK]).to(args.device)
            h = F.embedding(tk, g["embed"])
            idxs, rws = [], []
            for l in range(cfg.num_hidden_layers):
                h, idx, rw = oc.layer_dynamic(ck.layer(l, args.device, torch.float32), h, cfg, cos, sin)
                idxs.append(idx.view(len(tk), SEQ, -1).to(torch.int16).cpu().numpy())
                rws.append(rw.view(len(tk), SEQ, -1).float().cpu().numpy())
            ces += head_ce(g, h, tk, cfg).cpu().numpy().tolist()
            ia, ra = np.stack(idxs, 1), np.stack(rws, 1)
            for k in range(len(tk)):
                np.savez(d / "routes" / f"w_{a + k:03d}.npz", ids=ia[k], weights=ra[k], tokens=toks_np[a + k])
                hashes.append(oc.route_hash(ia[k], ra[k]))
    jdump(d / "routes_manifest.json", {"w": {"windows": len(toks_np), "hash": hashes, "tokens_sha256": sha256(d / "tokens.npy"),
                                             "summed_ce_dynamic": ces}})
    log(f"{args.set}: routes of {len(toks_np)} windows")


def replay(out, toks_np, routes, cands, dev, log, label, work="work", full_forward=None):
    """stage 1 truth semantics, windows in chunks of CHUNK: per window the FP32 summed CE and, per candidate, summed CE
    minus FP32. full_forward: names whose replay restarts from the embedding (cache check)."""
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_bc_numerics import quantizers
    from run_olmoe_fixed_route import head_ce
    cfg, ck = oc.config(), oc.Checkpoint()
    ranges = jload(S1 / "ranges.json")
    (out / work).mkdir(exist_ok=True)
    q, qa = quantizers(out / work, dev, "corrected")
    cos, sin = oc.rope(cfg, SEQ, dev, torch.float32)
    g = ck.global_(dev, torch.float32)
    rows = {c["name"]: [] for c in cands}
    base_all, t0 = [], time.time()
    with torch.no_grad():
        for a in range(0, len(toks_np), CHUNK):
            sl = slice(a * SEQ, min(len(toks_np), a + CHUNK) * SEQ)
            rts = [(i[sl], w[sl]) for i, w in routes]
            toks = torch.from_numpy(toks_np[a:a + CHUNK]).to(dev)
            hs = [F.embedding(toks, g["embed"])]
            for l in range(cfg.num_hidden_layers):
                hs.append(oc.layer_fixed(ck.layer(l, dev, torch.float32), hs[-1], cfg, cos, sin, *rts[l], l)[0])
            base = head_ce(g, hs[-1], toks, cfg).cpu().numpy()
            base_all += base.tolist()
            for L, mode in sorted({(c["layer"], c["mode"]) for c in cands}):
                group = [c for c in cands if c["layer"] == L and c["mode"] == mode]
                start = {c["name"]: (0 if full_forward and c["name"] in full_forward else L) for c in group}
                H = {c["name"]: hs[start[c["name"]]].clone() for c in group}
                for l in range(min(start.values()), cfg.num_hidden_layers):
                    w = ck.layer(l, dev, torch.float32)
                    for c in group:
                        if l < start[c["name"]]:
                            continue
                        if l == L:
                            e = c["expert"]
                            saved, hk = {}, oc.ExpertHooks()
                            for p, (wf, af) in c["formats"].items():
                                if wf != "fp32":
                                    saved[p] = w[p][e].clone()
                                    w[p][e] = q.weight(w[p][e].cpu(), wf).to(dev)
                                if af != "fp32":
                                    for s_ in ("x", "y"):
                                        hk.act[(l, e, p, s_)] = (lambda t, af=af, r=ranges[f"{c['node']}.{p}"][s_]: qa.act(t, af, r))
                            H[c["name"]] = oc.layer_fixed(w, H[c["name"]], cfg, cos, sin, *rts[l], l, hk)[0]
                            for p, v in saved.items():
                                w[p][e] = v
                        else:
                            H[c["name"]] = oc.layer_fixed(w, H[c["name"]], cfg, cos, sin, *rts[l], l)[0]
                    del w
                for c in group:
                    rows[c["name"]] += (head_ce(g, H.pop(c["name"]), toks, cfg).cpu().numpy() - base).tolist()
            del hs
            log(f"{label}: windows {a}-{min(len(toks_np), a + CHUNK) - 1} ({time.time() - t0:.0f}s, peak GPU "
                f"{torch.cuda.max_memory_allocated() / 2 ** 30 if dev != 'cpu' else 0:.1f} GB)")
    return {"rows": rows, "base": base_all, "seconds": time.time() - t0}


def load_set_routes(out, name, dev):
    import olmoe_common as oc
    from run_olmoe_fixed_route import load_routes
    rts, _, _ = load_routes(out / name, "w", oc.config(), dev)
    return rts


def stage_pilot(args, out, log):
    """Time and peak memory only (no loss kept): stage 1's diagnostic windows (30, already used), one group."""
    import torch
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, load_routes
    gpu_exact("cuda")
    cands = [c for c in jload(S1 / "candidates.json") if c["layer"] == 0 and c["mode"] == "w"]
    rts, _, _ = load_routes(S1, "diagnostic", oc.config(), "cuda")
    torch.cuda.reset_peak_memory_stats()
    r = replay(out, np.load(S1 / "tokens_diagnostic.npy"), rts, cands, "cuda", log, "pilot")
    res = {"windows": 30, "candidates": len(cands), "group": "layer 0, w", "seconds": r["seconds"],
           "peak_gpu_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
           "projection_note": "stage 1 weights: L0 groups replay 16 layers, L8 8, L15 1; 9 groups"}
    per_window_layer = r["seconds"] / (30 * 120 * 16)
    res["projected_minutes_per_64_windows_all_1080"] = per_window_layer * 64 * 120 * 3 * (16 + 8 + 1) / 60
    jdump(out / "pilot.json", res)
    log(json.dumps(res))


def stage_truth(args, out, log):
    import torch
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact(args.device)
    if args.set == "V" and not (out / "freeze_V.json").exists():
        raise SystemExit("V is evaluated only after freeze_V.json")
    d = out / args.set
    f = d / "truth.json"
    if f.exists():
        raise SystemExit("exists")
    cands = jload(S1 / "candidates.json")
    r = replay(out, np.load(d / "tokens.npy"), load_set_routes(out, args.set, args.device), cands, args.device, log, args.set)
    r.update(set=args.set, tokens_per_window=T_PRED, torch=torch.__version__, tf32=bool(torch.backends.cuda.matmul.allow_tf32),
             script_sha256=sha256(HERE / "run_olmoe_crosstext.py"))
    jdump(f, r)
    log(f"{args.set}: {len(cands)} candidates x {len(r['base'])} windows in {r['seconds']:.0f}s")


def stage_check(args, out, log):
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact("cuda")
    sub = set(jload(out / "plan.json")["determinism_subset"])
    allc = jload(S1 / "candidates.json")
    cands = [c for c in allc if c["name"] in sub]
    toks = np.load(out / "D/tokens.npy")[:2 * WPS]
    rts = load_set_routes(out, "D", "cuda")
    rts = [(i[:2 * WPS * SEQ], w[:2 * WPS * SEQ]) for i, w in rts]
    r1 = replay(out, toks, rts, cands, "cuda", log, "check 1")
    r2 = replay(out, toks, rts, cands, "cuda", log, "check 2")
    ff = [c for c in allc if c["layer"] == 8 and c["set"] == "gate_proj" and c["formats"]["gate_proj"] in (["int4", "fp32"], ["fp32", "int4"], ["int4", "int4"])]
    r3 = replay(out, toks, rts, ff, "cuda", log, "check full", full_forward={c["name"] for c in ff})
    r4 = replay(out, toks, rts, ff, "cuda", log, "check cached")
    D = jload(out / "D/truth.json")["rows"] if (out / "D/truth.json").exists() else None
    res = {"determinism_max_abs": float(max(np.abs(np.array(r1["rows"][n]) - np.array(r2["rows"][n])).max() for n in r1["rows"])),
           "full_forward_vs_cached_max_abs": float(max(np.abs(np.array(r3["rows"][n]) - np.array(r4["rows"][n])).max() for n in r3["rows"])),
           "full_forward_candidates": [c["name"] for c in ff],
           "vs_main_D_run_max_abs": (float(max(np.abs(np.array(r1["rows"][n]) - np.array(D[n][:2 * WPS])).max() for n in r1["rows"]))
                                     if D else None)}
    jdump(out / "check.json", res)
    log(json.dumps(res))


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def seg_values(out, name, cands):
    """[candidate, segment] per-token mean damage of each segment, and per-window values."""
    r = jload(out / name / "truth.json")
    W = np.array([r["rows"][c["name"]] for c in cands]) / T_PRED
    return W.reshape(len(cands), -1, WPS).mean(2), W


def boot_idx(nseg, seed):
    return np.random.default_rng(seed).integers(0, nseg, (BOOT, nseg))


def pair_summary(S, P, idx):
    I, J = np.array([p["i"] for p in P]), np.array([p["j"] for p in P])
    d = S[I] - S[J]                                   # [pair, segment]
    m = d.mean(1)
    b = np.stack([d[:, x].mean(1) for x in idx], 1)    # [pair, BOOT]
    le, ge = (b <= 0).mean(1), (b >= 0).mean(1)
    p2 = np.minimum(1.0, 2 * np.minimum(le, ge))
    lo, hi = np.percentile(b, 2.5, axis=1), np.percentile(b, 97.5, axis=1)
    n = d.shape[1]
    sd = d.std(1, ddof=1)
    from scipy.stats import t
    tq = t.ppf(0.975, n - 1)
    loso = np.stack([(m * n - d[:, k]) / (n - 1) for k in range(n)], 1)
    loso_robust = np.all(np.sign(loso) == np.sign(m)[:, None], 1) & (m != 0)
    dom = np.abs(d).max(1) / np.maximum(np.abs(d).sum(1), 1e-300)
    pos = (d > 0).sum(1)
    neg = (d < 0).sum(1)
    return {"d": d, "mean": m, "p2": p2, "le": le, "ge": ge, "lo": lo, "hi": hi, "t_lo": m - tq * sd / math.sqrt(n),
            "t_hi": m + tq * sd / math.sqrt(n), "sd": sd, "loso_robust": loso_robust, "dominance": dom, "pos": pos, "neg": neg}


def bh(p, q):
    n = len(p)
    o = np.argsort(p)
    thr = q * (np.arange(1, n + 1)) / n
    ok = p[o] <= thr
    k = np.where(ok)[0].max() + 1 if ok.any() else 0
    rej = np.zeros(n, bool)
    rej[o[:k]] = True
    return rej


def by(p, q):
    n = len(p)
    return bh(p, q / np.sum(1.0 / np.arange(1, n + 1)))


def stage_analyze_d(args, out, log):
    from run_olmoe_bc_numerics import pair_list
    if (out / "freeze_V.json").exists():
        raise SystemExit("frozen")
    cands = jload(S1 / "candidates.json")
    P = pair_list(cands)
    bc = np.array([c["bc"] for c in cands])
    I, J = np.array([p["i"] for p in P]), np.array([p["j"] for p in P])
    bcd = np.sign(bc[I] - bc[J])
    S, _ = seg_values(out, "D", cands)
    s = pair_summary(S, P, boot_idx(S.shape[1], 2))
    ident = bh(s["p2"], Q_D) & (bcd != 0) & (s["mean"] != 0)
    pred = np.sign(s["mean"])
    lst = [{"k": int(k), "a": cands[P[k]["i"]]["name"], "b": cands[P[k]["j"]]["name"], "predicted_sign": int(pred[k]),
            "bc_sign": int(bcd[k]), "bc_agrees": bool(pred[k] == bcd[k]), "D_mean": float(s["mean"][k]),
            "D_ci95": [float(s["lo"][k]), float(s["hi"][k])], "D_p2": float(s["p2"][k])} for k in np.where(ident)[0]]
    jdump(out / "freeze_V.json", {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "identified": lst,
                                  "D_truth_sha256": sha256(out / "D/truth.json"), "plan_sha256": sha256(out / "plan.json"),
                                  "script_sha256": sha256(HERE / "run_olmoe_crosstext.py"),
                                  "rule": f"BH q={Q_D} over all pairs on D; V: one-sided BH q={Q_V} within this list"})
    res = {"pairs": len(P), "identified": int(ident.sum()), "identified_BY": int((by(s["p2"], Q_D) & (bcd != 0)).sum()),
           "identified_bc_agree": int(sum(x["bc_agrees"] for x in lst)), "identified_bc_disagree": int(sum(not x["bc_agrees"] for x in lst)),
           "zero_pairs": int((s["mean"] == 0).sum())}
    jdump(out / "D/analysis_D.json", res)
    log(json.dumps(res))


def stage_analyze(args, out, log):
    from scipy.stats import spearmanr
    from run_olmoe_bc_numerics import pair_list
    import csv
    cands = jload(S1 / "candidates.json")
    names = [c["name"] for c in cands]
    P = pair_list(cands)
    bc = np.array([c["bc"] for c in cands])
    I, J = np.array([p["i"] for p in P]), np.array([p["j"] for p in P])
    bcd = np.sign(bc[I] - bc[J])
    fz = jload(out / "freeze_V.json")
    SD_, WD = seg_values(out, "D", cands)
    SV_, WV = seg_values(out, "V", cands)
    sD, sV = pair_summary(SD_, P, boot_idx(SD_.shape[1], 2)), pair_summary(SV_, P, boot_idx(SV_.shape[1], 3))
    res = {"pairs": len(P), "segments": {"D": int(SD_.shape[1]), "V": int(SV_.shape[1])}}
    # confirmation of the frozen list
    lst = fz["identified"]
    ks = np.array([x["k"] for x in lst], int)
    pred = np.array([x["predicted_sign"] for x in lst])
    p_same = np.where(pred > 0, sV["le"][ks], sV["ge"][ks]) if len(ks) else np.array([])
    p_opp = np.where(pred > 0, sV["ge"][ks], sV["le"][ks]) if len(ks) else np.array([])
    conf = bh(p_same, Q_V) if len(ks) else np.array([], bool)
    oppo = bh(p_opp, Q_V) if len(ks) else np.array([], bool)
    same_sign = np.sign(sV["mean"][ks]) == pred
    cat = np.where(conf, "confirmed_same", np.where(oppo, "opposite_significant",
                                                    np.where(same_sign, "same_direction_insufficient", "undetermined")))
    agree = np.array([x["bc_agrees"] for x in lst], bool)

    def cnt(sel):
        return {c_: int(((cat == c_) & sel).sum()) for c_ in ("confirmed_same", "same_direction_insufficient",
                                                              "opposite_significant", "undetermined")}
    res["confirmation"] = {"identified_in_D": int(len(ks)), "bc_agree": {"pairs": int(agree.sum()), **cnt(agree)},
                           "bc_disagree": {"pairs": int((~agree).sum()), **cnt(~agree)},
                           "coverage_V_mean_in_D_ci95": float(np.mean((sV["mean"][ks] >= sD["lo"][ks]) & (sV["mean"][ks] <= sD["hi"][ks]))) if len(ks) else None}
    rep_err = [k for k, c_, a in zip(ks, cat, agree) if c_ == "confirmed_same" and not a]
    rep_ok = [k for k, c_, a in zip(ks, cat, agree) if c_ == "confirmed_same" and a]

    def unique(kl):
        cs = set()
        for k in kl:
            cs |= {names[P[k]["i"]], names[P[k]["j"]]}
        return {"pairs": len(kl), "experts": sorted({P[k]["node"] for k in kl}), "candidates": len(cs),
                "by_mode": {m: sum(P[k]["mode"] == m for k in kl) for m in ("w", "a", "wa")},
                "by_expert": {e: sum(P[k]["node"] == e for k in kl) for e in sorted({P[k]["node"] for k in kl})},
                "by_class": {cl: sum(P[k]["class"] == cl for k in kl) for cl in sorted({P[k]["class"] for k in kl})}}
    res["reproducible_bc_errors"] = unique(rep_err)
    res["reproducible_bc_correct"] = unique(rep_ok)
    res["reproducible_bc_errors_list"] = [[names[P[k]["i"]], names[P[k]["j"]], float(sD["mean"][k]), float(sV["mean"][k])] for k in rep_err]
    # all pairs
    nz = (sD["mean"] != 0) & (sV["mean"] != 0)
    res["all_pairs"] = {
        "D_V_same_sign": float((np.sign(sD["mean"][nz]) == np.sign(sV["mean"][nz])).mean()),
        "spearman_D_V_means": float(spearmanr(sD["mean"], sV["mean"])[0]),
        "coverage_V_mean_in_D_ci95": float(np.mean((sV["mean"] >= sD["lo"]) & (sV["mean"] <= sD["hi"]))),
        "bc_agreement_D_sign": float((np.sign(sD["mean"][sD["mean"] != 0]) == bcd[sD["mean"] != 0]).mean()),
        "bc_agreement_V_sign": float((np.sign(sV["mean"][sV["mean"] != 0]) == bcd[sV["mean"] != 0]).mean()),
        "identified_V_alone_BH": int((bh(sV["p2"], Q_D) & (bcd != 0)).sum()),
        "loso_robust_share": {"D": float(sD["loso_robust"].mean()), "V": float(sV["loso_robust"].mean())},
        "median_dominance": {"D": float(np.median(sD["dominance"])), "V": float(np.median(sV["dominance"]))},
        "median_abs_mean": {"D": float(np.median(np.abs(sD["mean"]))), "V": float(np.median(np.abs(sV["mean"])))},
        "median_abs_mean_over_seg_sd": {"D": float(np.median(np.abs(sD["mean"]) / np.maximum(sD["sd"], 1e-30))),
                                        "V": float(np.median(np.abs(sV["mean"]) / np.maximum(sV["sd"], 1e-30)))}}
    # pooled (secondary)
    SP = np.concatenate([SD_, SV_], 1)
    sP = pair_summary(SP, P, boot_idx(SP.shape[1], 4))
    idP = bh(sP["p2"], Q_D) & (bcd != 0)
    res["pooled_secondary"] = {"identified": int(idP.sum()), "bc_agreement_identified": float((np.sign(sP["mean"][idP]) == bcd[idP]).mean()) if idP.any() else None}
    # candidate-level reproducibility
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault((c["node"], c["mode"]), []).append(i)
    res["candidate_spearman_D_V"] = {f"{k[0]}|{k[1]}": float(spearmanr(SD_[v].mean(1), SV_[v].mean(1))[0]) for k, v in groups.items()}
    res["candidate_spearman_median_by_mode"] = {m: float(np.median([v for k, v in res["candidate_spearman_D_V"].items() if k.endswith("|" + m)])) for m in ("w", "a", "wa")}
    # by expert / mode / class
    def block(sel):
        sel = np.asarray(sel, bool)
        idd = np.zeros(len(P), bool)
        idd[ks] = True
        return {"pairs": int(sel.sum()), "identified_D": int((idd & sel).sum()),
                "D_V_same_sign": float((np.sign(sD["mean"][sel & nz]) == np.sign(sV["mean"][sel & nz])).mean()) if (sel & nz).any() else None,
                "bc_agree_D": float((np.sign(sD["mean"][sel]) == bcd[sel]).mean()), "bc_agree_V": float((np.sign(sV["mean"][sel]) == bcd[sel]).mean()),
                "median_abs_mean_D": float(np.median(np.abs(sD["mean"][sel]))),
                "loso_robust_D": float(sD["loso_robust"][sel].mean()), "median_dominance_D": float(np.median(sD["dominance"][sel]))}
    res["by_mode"] = {m: block([p["mode"] == m for p in P]) for m in ("w", "a", "wa")}
    res["by_expert"] = {e: block([p["node"] == e for p in P]) for e in sorted({p["node"] for p in P})}
    res["by_class"] = {cl: block([p["class"] == cl for p in P]) for cl in sorted({p["class"] for p in P})}
    # tracked 31
    tr = jload(NUM / "consistent_pairs.json")["stable_wrong_both"]["pairs_list"]
    key = {(names[p["i"]], names[p["j"]]): k for k, p in enumerate(P)}
    rows31 = []
    pos = {int(k): n for n, k in enumerate(ks)}
    for a, b in tr:
        k = key.get((a, b), key.get((b, a)))
        r = {"a": a, "b": b, "bc_sign": int(bcd[k]), "D_mean": float(sD["mean"][k]), "D_ci95": [float(sD["lo"][k]), float(sD["hi"][k])],
             "D_p2": float(sD["p2"][k]), "V_mean": float(sV["mean"][k]), "V_ci95": [float(sV["lo"][k]), float(sV["hi"][k])],
             "identified_D": k in pos, "V_category": str(cat[pos[k]]) if k in pos else None,
             "D_against_bc": bool(np.sign(sD["mean"][k]) == -bcd[k]), "V_against_bc": bool(np.sign(sV["mean"][k]) == -bcd[k])}
        rows31.append(r)
    res["tracked_31"] = {"pairs": len(rows31), "identified_D": sum(r["identified_D"] for r in rows31),
                         "confirmed_bc_error": sum(r["V_category"] == "confirmed_same" and r["D_against_bc"] for r in rows31),
                         "D_and_V_mean_against_bc": sum(r["D_against_bc"] and r["V_against_bc"] for r in rows31), "rows": rows31}
    # traffic
    tf = {}
    for g in ("D", "V"):
        man = jload(out / g / "routes_manifest.json")["w"]
        for c in cands:
            if c["node"] in tf.get(g, {}):
                continue
            pass
        per = {}
        for e_node in sorted({c["node"] for c in cands}):
            L, E = int(e_node.split(".")[0][1:]), int(e_node.split(".")[1][1:])
            seg = []
            for s_ in range(man["windows"] // WPS):
                tok, act, ws = 0, 0, []
                for wdx in range(s_ * WPS, (s_ + 1) * WPS):
                    z = np.load(out / g / "routes" / f"w_{wdx:03d}.npz")
                    hit = z["ids"][L] == E
                    n = int(hit.sum())
                    tok += n
                    act += n > 0
                    ws += z["weights"][L][hit].tolist()
                seg.append({"tokens": tok, "windows_active": act, "rw_mean": float(np.mean(ws)) if ws else 0.0, "rw_sum": float(np.sum(ws))})
            per[e_node] = seg
        tf[g] = per
    res["traffic"] = {g: {e: {"zero_segments": sum(s_["tokens"] == 0 for s_ in v), "median_tokens": float(np.median([s_["tokens"] for s_ in v])),
                              "min_tokens": min(s_["tokens"] for s_ in v), "max_tokens": max(s_["tokens"] for s_ in v)}
                          for e, v in tf[g].items()} for g in tf}
    # traffic vs pair behaviour (descriptive): per expert, identification share and D/V agreement against median traffic
    res["traffic_vs_pairs"] = {e: {"median_tokens_D": res["traffic"]["D"][e]["median_tokens"], "zero_segments_D": res["traffic"]["D"][e]["zero_segments"],
                                   "identified_share_D": res["by_expert"][e]["identified_D"] / max(res["by_expert"][e]["pairs"], 1),
                                   "D_V_same_sign": res["by_expert"][e]["D_V_same_sign"], "calibration_coverage": next(c["coverage"] for c in cands if c["node"] == e)}
                               for e in res["by_expert"]}
    # does the segment-level pair difference follow traffic? (per expert: corr of |segment value| with segment tokens)
    corr = {}
    for e in res["by_expert"]:
        sel = [k for k, p in enumerate(P) if p["node"] == e]
        toksD = np.array([s_["tokens"] for s_ in tf["D"][e]])
        if toksD.std() > 0:
            corr[e] = float(np.median([np.corrcoef(np.abs(sD["d"][k]), toksD)[0, 1] for k in sel if np.abs(sD["d"][k]).std() > 0]))
    res["traffic_abs_segment_diff_corr_D"] = corr
    jdump(out / "traffic.json", tf)
    jdump(out / "analysis.json", res)
    with open(out / "pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["a", "b", "node", "mode", "class", "bc_sign", "D_mean", "D_lo", "D_hi", "D_p2", "D_loso_robust", "D_dominance",
                    "V_mean", "V_lo", "V_hi", "V_p2", "identified_D", "V_category"])
        catk = {int(k): str(c_) for k, c_ in zip(ks, cat)}
        for k, p in enumerate(P):
            w.writerow([names[p["i"]], names[p["j"]], p["node"], p["mode"], p["class"], int(bcd[k]), sD["mean"][k], sD["lo"][k], sD["hi"][k],
                        sD["p2"][k], int(sD["loso_robust"][k]), sD["dominance"][k], sV["mean"][k], sV["lo"][k], sV["hi"][k], sV["p2"][k],
                        int(k in catk), catk.get(k, "")])
    np.save(out / "segment_values_D.npy", SD_)
    np.save(out / "segment_values_V.npy", SV_)
    log(json.dumps({k: res[k] for k in ("confirmation", "reproducible_bc_errors", "reproducible_bc_correct", "all_pairs", "pooled_secondary")})[:3000])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--set", default="D", choices=("D", "V"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("truth", "capture", "check", "pilot"):
        import torch
        torch.set_num_threads(args.threads)
    t0 = time.time()
    tag = args.stage + (f"_{args.set}" if args.stage in ("truth", "capture") else "")
    (out / "logs").mkdir(exist_ok=True)
    logf = open(out / "logs" / f"{tag}.log", "a")

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
        timing["runs"].append({"tag": tag, "status": status, "wall_s": time.time() - t0,
                               "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0))})
        jdump(tf, timing)
    log(f"stage {args.stage} done")


if __name__ == "__main__":
    main()
