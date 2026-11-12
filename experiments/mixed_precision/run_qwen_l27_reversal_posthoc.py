#!/usr/bin/env python3
"""POST HOC (not pre-registered) companion of run_qwen_l27_reversal_diag.py: the L27 Down f16 vs p8e1 difference with the
target weight as in the original knee A (int8 per output channel) instead of FP32, on the same new windows and the
calibration windows. Everything else FP32. Written after the planned analysis showed the FP32-weight x+y difference
near zero; exploratory only.
Run: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python run_qwen_l27_reversal_posthoc.py --out-dir <run dir>
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_l27_reversal_diag as d  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    out = ap.parse_args().out_dir.resolve()
    t0 = time.perf_counter()
    core = d.Core(out / 'work')
    torch = core.torch
    lin = torch.nn.functional.linear
    W32 = core.W.detach()
    Wq = core.q.weight(W32.cpu(), 'int8').cuda()
    res = {'note': 'post hoc, exploratory; target Down weight fp32 vs int8 (A), activation f16 / p8e1 / int4 / p4e1 on x, y, xy',
           'weights_sha256': {'int8': __import__('hashlib').sha256(Wq.cpu().numpy().tobytes()).hexdigest()}}
    rng = np.random.default_rng(d.SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, d.N_ART, d.N_ART), minlength=d.N_ART) for _ in range(d.BOOT)]).astype(np.float64) / d.N_ART
    for label, toks in (('calibration', np.load(d.NATIVE / 'calib_inputs.npy')), ('new', np.load(out / 'tokens.npy'))):
        acc = {}
        for c0 in range(0, len(toks), d.CHUNK):
            ids = torch.from_numpy(toks[c0:c0 + d.CHUNK]).cuda()
            ce_full, mid, h, y = core.capture(ids)
            with torch.no_grad():
                for wn, Wt in (('fp32', W32), ('int8', Wq)):
                    base = core.head_ce(mid + lin(h, Wt), ids)
                    acc.setdefault(f'{wn}|weight_only', []).append((base - ce_full).cpu().numpy())
                    for f in ('f16', 'p8e1', 'int4', 'p4e1'):
                        qx = core.qact(h, f, 'x')
                        yx = lin(qx, Wt)
                        y0 = lin(h, Wt)
                        for s, yp in (('x', yx), ('y', core.qact(y0, f, 'y')), ('xy', core.qact(yx, f, 'y'))):
                            acc.setdefault(f'{wn}|{s}|{f}', []).append((core.head_ce(mid + yp, ids) - base).cpu().numpy())
        acc = {k: np.concatenate(v) / d.TP for k, v in acc.items()}
        R = {}
        for k, v in acc.items():
            R[k] = {'mean': float(v.mean())}
            if label == 'new':
                a = v.reshape(d.N_ART, d.WPA).mean(1)
                R[k]['ci95'] = np.percentile(Wb @ a, [2.5, 97.5]).tolist()
        for wn in ('fp32', 'int8'):
            for s in ('x', 'y', 'xy'):
                for p, q in (('f16', 'p8e1'), ('int4', 'p4e1')):
                    D = acc[f'{wn}|{s}|{p}'] - acc[f'{wn}|{s}|{q}']
                    R[f'D|{wn}|{s}|{p}-{q}'] = {'mean': float(D.mean())}
                    if label == 'new':
                        R[f'D|{wn}|{s}|{p}-{q}']['ci95'] = np.percentile(Wb @ D.reshape(d.N_ART, d.WPA).mean(1), [2.5, 97.5]).tolist()
        res[label] = R
    res['seconds'] = time.perf_counter() - t0
    (out / 'posthoc').mkdir(exist_ok=True)
    d.write(out / 'posthoc' / 'weight_context.json', res)
    print(json.dumps({k: v for k, v in res['new'].items() if k.startswith('D|')}, indent=1))


if __name__ == '__main__':
    main()
