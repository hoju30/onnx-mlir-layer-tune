#!/usr/bin/env python3
"""Second follow-up of the L27 Down f16 -> p8e1 context effect (runs/qwen3_l27_context_followup_20261026).
(a) GPU: split the quantization of the earlier Down projections (L0-26 down_proj, A formats) into weight only / activation only,
    and the activation into input x only / output y only; the L27 Down a is f16 or p8e1 on the whole node and the per-position CE
    is kept, so the position-0 effect is read exactly (after L27 only per-position operations follow).
(b) native (onnx-mlir): the same position-0 background effect, U (all other projections FP32) vs F (all per A = the original
    knee), from per-position native CE of whole-node f16 / p8e1 models (no position-specific format is needed).
D = NLL(p8e1) - NLL(f16) (negative = p8e1 better); D_pos0 = the change of the position-0 CE / targets of the window(s).

Stages (one new --out-dir): plan, gpu_controls, build, native_controls, audit, data, gpu_eval, native_eval, analyze (worker: internal).
GPU stages: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python; the others ../../venv/bin/python.
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen3_l27_context_attribution as ca  # noqa: E402
import run_qwen_prefix_rank_shift as prs  # noqa: E402

read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
TARGET, ENDP, L27C, NATIVE, PREP = ca.TARGET, ca.ENDP, ca.L27C, ca.NATIVE, prs.PREP
FUP = HERE / 'runs/qwen3_l27_context_followup_20261026'
GPU_BG = ('U', 'dn', 'dn_w', 'dn_a', 'dn_ax', 'dn_ay', 'F')
SPLITS = ('dn_w', 'dn_a', 'dn_ax', 'dn_ay')
NATIVE_CFGS = ('U_f16', 'U_p8e1', 'F_f16', 'F_p8e1')
REUSED = {'F_f16': L27C / 'native/A/A.so', 'F_p8e1': L27C / 'native/C/C.so'}
N_ART, WPA, SEQ, TP, BOOT, VOCAB = 128, 2, 128, 127, 20000, 151936
NATIVE_WORKERS = 6
THREAD_ENV = {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE', 'MKL_DYNAMIC': 'FALSE'}
SEEDS = {'order': 26102651, 'start': 26102652, 'bootstrap': 26102653}
CODE = ['run_qwen3_l27_down_source.py', 'run_qwen3_l27_context_followup.py', 'run_qwen_l27_native_confirm.py', 'run_native_bc_pipeline.py'] + ca.CODE
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def check_plan(out, inputs_too=True):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment)')
    if inputs_too:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


def is_down(n, upto=26):
    l, p = ca.parse(n)
    return p == 'down' and l <= upto


def gpu_background(A, b):
    """(config for LMReplay, slot overrides {node: (x fmt, y fmt)}) of a background; the target gets its slot separately."""
    cfg, ovr = {}, {}
    for n in A:
        if n == TARGET:
            cfg[n] = dict(A[n])
            continue
        if b == 'F' or (b == 'dn' and is_down(n)):
            cfg[n] = dict(A[n])
        elif b == 'dn_w' and is_down(n):
            cfg[n] = {'w': A[n]['w'], 'a': 'fp32'}
        elif b in ('dn_a', 'dn_ax', 'dn_ay') and is_down(n):
            cfg[n] = {'w': 'fp32', 'a': 'fp32'}
            a = A[n]['a']
            ovr[n] = {'dn_a': (a, a), 'dn_ax': (a, 'fp32'), 'dn_ay': ('fp32', a)}[b]
        else:
            cfg[n] = {'w': 'fp32', 'a': 'fp32'}
    return cfg, ovr


def native_config(A, name):
    bg, a = name.split('_')
    cfg = {n: (dict(A[n]) if bg == 'F' else {'w': 'fp32', 'a': 'fp32'}) for n in A}
    cfg[TARGET] = {'w': 'int8', 'a': a}
    return cfg


def inputs():
    return ca.inputs() + [FUP / 'eval_ce.npz', FUP / 'tokens.npy', FUP / 'report.md', REUSED['F_f16'], REUSED['F_f16'].parent / 'model.constants.bin',
                          REUSED['F_p8e1'], REUSED['F_p8e1'].parent / 'model.constants.bin', L27C / 'builds.json', L27C / 'controls.json',
                          L27C / 'configs/A.json', L27C / 'configs/C.json', NATIVE / 'model.mlirbc', PREP / 'model.onnx']


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out, smoke):
    import wt_manifest_scan as wms
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    A = read(ENDP / 'configs/A_auto.json')
    if ca.config_hash(A) != ca.A_HASH:
        raise SystemExit('A is not the original knee')
    nc = {k: native_config(A, k) for k in NATIVE_CFGS}
    if nc['F_f16'] != read(L27C / 'configs/A.json') or nc['F_p8e1'] != read(L27C / 'configs/C.json'):
        raise SystemExit('reused native configurations differ')
    downs = sorted((n for n in A if is_down(n)), key=lambda n: ca.parse(n)[0])
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    seen = [str(p.relative_to(HERE)) for p in HERE.glob('runs/*/plan.json') if any(str(v) in p.read_text() for v in SEEDS.values())]
    if seen:
        raise SystemExit(f'seeds already used: {seen}')
    from run_olmoe_dynamic_route import env_info
    pl = {'time': now(), 'request': 'user 2026-10-26: split the earlier Down quantization into weight / activation, then activation input / '
                                    'output; native confirmation of the position-0 background effect',
          'gpu': {'backgrounds': {'U': 'all other projections FP32', 'dn': 'L0-26 down_proj w and a per A', 'dn_w': 'their weights only',
                                  'dn_a': 'their activations only (x and y)', 'dn_ax': 'activation on input x only', 'dn_ay': 'activation on '
                                  'output y only', 'F': 'all per A'}, 'earlier_downs': {n: A[n] for n in downs},
                  'target': f'{TARGET}: w int8; a f16 or p8e1 on the whole node; per-position CE kept',
                  'exactness': 'L27 is followed only by per-position operations, so the change of the position-p CE under whole-node p8e1 equals '
                               'the effect of p8e1 at position p only (checked: interaction 0 in the previous round)',
                  'conditions': len(GPU_BG) * 2, 'windows': f'{N_ART} new articles x {WPA}'},
          'native': {'configs': {k: 'all other projections ' + ('per A' if k.startswith('F') else 'FP32') + f', target w int8 / a {k.split("_")[1]}'
                                 for k in NATIVE_CFGS}, 'reused': {k: prs.rel(v) for k, v in REUSED.items()},
                     'reused_check': 'F_f16 / F_p8e1 equal runs/qwen3_l27_native_confirm_20261024 configs A / C; .so / constants hashes recorded',
                     'built': 'U_f16, U_p8e1: erm.build_so as the previous rounds (bytecode, constants to file, Krnl O3, clang O0, per-channel '
                              'weight); compile env checked for all 196 entries',
                     'windows': f'the first window of each of the {N_ART} new articles ({N_ART} windows)', 'output': 'per-position float64 CE '
                     '(127 targets) per window', 'workers': NATIVE_WORKERS},
          'statistics': {'D': 'NLL(p8e1) - NLL(f16); pos0 = the position-0 CE change / targets (contribution to the per-token NLL); rest = '
                              'positions 1..126 targets; all = both',
                         'gpu_primary': 'position-0 context change of each split vs U: (D_b,pos0 - D_U,pos0) for dn_w, dn_a, dn_ax, dn_ay, Holm over '
                                        '4; dn and F re-measured (replication); additivity dn - (w + a), a - (ax + ay) (descriptive)',
                         'native_primary': 'position-0 background effect (D_F,pos0 - D_U,pos0) native, 95% CI; also all positions; the GPU '
                                           'value on the same 128 windows and the per-article backend difference reported',
                         'bootstrap': f'{BOOT} paired article resamples (seed {SEEDS["bootstrap"]})'},
          'data': {'n_articles': N_ART, 'windows_per_article': WPA, 'rule': 'run_qwen_prefix_rank_shift.data', 'excluded_previous_wt103': len(used),
                   'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS},
          'controls': ['GPU: U / F / dn f16 and p8e1 reproduce the follow-up round on its tokens (bitwise per window); the slot overrides equal '
                       'LMReplay\'s own path for dn on calibration (bitwise); per-position CE sums equal the window CE',
                       'native: F_f16 / F_p8e1 calibration windows 0-1 equal the l27-confirm controls within 1e-8; U_f16 window 0 repeated in '
                       'another process within 1e-8; per-position CE sums equal the window CE; compile env of the new builds'],
          'boundaries': 'no criterion / config / pipeline change', 'smoke': smoke or 'none',
          'inputs': {prs.rel(p): sha(p) for p in inputs()}, 'code_hashes': {f: sha(HERE / f) for f in CODE}, 'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    (out / 'plan.md').write_text('\n'.join([
        '# 計畫：前層 Down 的權重／activation 拆分與 position 0 背景效應的 native 確認', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        '- GPU：背景 U（全 FP32）、dn（L0–26 Down w 與 a 依 A）、dn_w（只權重）、dn_a（只 activation，x 與 y）、dn_ax（只 input）、dn_ay（只 output）、F（全 A）；L27 Down w int8、a = f16／p8e1 整節點；保存逐位置 CE，position 0 的變化即「只在 position 0 用 p8e1」的效果（L27 之後只有逐位置運算）。',
        '- GPU 主要：各拆分的 position 0 情境改變（D_b,pos0 − D_U,pos0），4 個 Holm；加成性描述；dn、F 重現。',
        '- native：U_f16、U_p8e1（新編譯）、F_f16、F_p8e1（重用 L27 confirm 的 A、C .so）；每篇第 1 個窗口（128 個）的逐位置 CE；主要：position 0 背景效應 D_F,pos0 − D_U,pos0 與 95% CI；同窗口 GPU 對照。',
        f'- 資料：{N_ART} 篇新 WT103 × {WPA}；seed {SEEDS}。',
        f'- 冒煙測試：{smoke or "無"}', '']) + '\n')
    log(f'plan written; used {len(used)}')


# --------------------------------------------------------------------------- #
# GPU
# --------------------------------------------------------------------------- #
class SlotRunner:
    """LMReplay with per-node (x, y) activation overrides (the overridden nodes skip LMReplay's own activation hooks) and per-position CE."""

    def __init__(self, work, nodes):
        from run_qwen_down_fp16_protection import replay
        self.rp, self.init_s = replay(work)
        self.torch = self.rp.torch
        self.ovr = {}
        rp = self.rp
        for n in nodes:
            m, rng = rp.mod(n), rp.ranges[n]

            def pre(mod, i, n=n, rng=rng):
                f = self.ovr.get(n)
                return None if f is None or f[0] == 'fp32' else (rp.qact(i[0], f[0], rng['x']),)

            def post(mod, i, o, n=n, rng=rng):
                f = self.ovr.get(n)
                return None if f is None or f[1] == 'fp32' else rp.qact(o, f[1], rng['y'])
            m.register_forward_pre_hook(pre)
            m.register_forward_hook(post)

    def ce_pos(self, toks, batch=8):
        torch = self.torch
        x = torch.from_numpy(toks).to(self.rp.device)
        r = []
        with torch.no_grad():
            for i in range(0, len(x), batch):
                z = self.rp.ad(x[i:i + batch])
                r.append(torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), x[i:i + batch, 1:].reshape(-1),
                                                           reduction='none').view(z.shape[0], -1).double())
        return torch.cat(r).cpu().numpy()                    # [windows, 127]

    def run(self, cfg, ovr, toks, own=True):
        c = {n: dict(v) for n, v in cfg.items()}
        if own:
            for n in ovr:
                c[n] = {'w': c[n]['w'], 'a': 'fp32'}
            self.ovr = dict(ovr)
        else:
            self.ovr = {}
        self.rp.set(c)
        r = self.ce_pos(toks)
        self.ovr = {}
        return r


def gpu_setting(A, b, s):
    cfg, ovr = gpu_background(A, b)
    ovr[TARGET] = (s, s)
    return cfg, ovr


def gpu_controls(out):
    t0 = time.perf_counter()
    check_plan(out)
    A = read(ENDP / 'configs/A_auto.json')
    r = SlotRunner(out / 'work', [n for n in A if is_down(n, 27)])
    ftok = np.load(FUP / 'tokens.npy')
    fz = np.load(FUP / 'eval_ce.npz')
    ctl = {}
    rep = {}
    for b_new, b_old in (('U', 'U'), ('F', 'F'), ('dn', 'ffn_down')):
        for s in ('f16', 'p8e1'):
            cfg, ovr = gpu_setting(A, b_new, s)
            pp = r.run(cfg, ovr, ftok)
            rep[f'{b_new}|{s}'] = float(np.abs(pp.sum(1) - fz[f'{b_old}__{s}']).max())
    ctl['reproduce_followup_window_ce_abs'] = rep
    cal = np.load(NATIVE / 'calib_inputs.npy')
    own_ = {}
    for s in ('f16', 'p8e1'):
        cfg, ovr = gpu_setting(A, 'dn_a', s)                 # dn_a is also expressible in LMReplay's own (one format per node) path
        a = r.run(cfg, ovr, cal)
        lm_cfg = {n: dict(v) for n, v in cfg.items()}
        for n, (fx, fy) in ovr.items():
            assert fx == fy
            lm_cfg[n]['a'] = fx
        b = r.run(lm_cfg, {}, cal, own=False)
        own_[s] = float(np.abs(a - b).max())
    ctl['overrides_equal_lmreplay_path_abs'] = own_
    ctl['tf32'] = [r.torch.backends.cuda.matmul.allow_tf32, r.torch.backends.cudnn.allow_tf32]
    ctl['deterministic'] = r.torch.are_deterministic_algorithms_enabled()
    ctl['passed'] = bool(max(rep.values()) == 0 and max(own_.values()) == 0 and ctl['tf32'] == [False, False] and ctl['deterministic'])
    write(out / 'gpu_controls.json', ctl)
    ca.log_stage(out, 'gpu_controls', time.perf_counter() - t0)
    log(f'gpu controls passed={ctl["passed"]}: {ctl}')
    if not ctl['passed']:
        raise SystemExit('gpu controls failed')


def gpu_eval(out):
    t0 = time.perf_counter()
    verify_data(out)
    A = read(ENDP / 'configs/A_auto.json')
    r = SlotRunner(out / 'work', [n for n in A if is_down(n, 27)])
    tok = np.load(out / 'tokens.npy')
    res = {}
    for b in GPU_BG:
        for s in ('f16', 'p8e1'):
            cfg, ovr = gpu_setting(A, b, s)
            res[f'{b}__{s}'] = r.run(cfg, ovr, tok)
    np.savez(out / 'gpu_ce_pos.npz', **res)
    ca.log_stage(out, 'gpu_eval', time.perf_counter() - t0, {'conditions': len(res), 'windows': len(tok)})


# --------------------------------------------------------------------------- #
# native
# --------------------------------------------------------------------------- #
def build(out):
    import onnx
    import eval_random_mixed_precision as erm
    from run_native_bc_pipeline import SLOT_INPUT, qwen3_spec
    import run_qwen_l27_native_confirm as l27
    t0 = time.perf_counter()
    check_plan(out)
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    A = read(ENDP / 'configs/A_auto.json')
    (out / 'configs').mkdir(exist_ok=True)
    for k in NATIVE_CFGS:
        write(out / 'configs' / f'{k}.json', native_config(A, k))
    tools, tf = l27.tool_files()
    root = out / 'native'
    root.mkdir(exist_ok=True)
    spec = qwen3_spec(out / 'work' / 'spec')
    graph = onnx.load(str(PREP / 'model.onnx'), load_external_data=False)
    by_name = {n.name: n for n in graph.graph.node}
    ranges = {}
    for node, slots in read(NATIVE / 'ranges_by_point.json').items():
        n = by_name[node]
        for slot, rr in slots.items():
            ranges[n.output[0] if slot == 'y' else n.input[SLOT_INPUT[slot]]] = tuple(rr)
    obj = erm.build_posit_runtime(tools, root)
    builds = {'tools_sha256': {k: sha(v) for k, v in tf.items()}, 'compile_env_checks': {}}
    for k, so in REUSED.items():
        builds[k] = {'so': prs.rel(so), 'so_sha256': sha(so), 'constants_sha256': sha(so.parent / 'model.constants.bin'), 'reused_from': prs.rel(so.parent)}
    for k in ('U_f16', 'U_p8e1'):
        cfg = read(out / 'configs' / f'{k}.json')
        env = erm.node_formats_env({n['name']: cfg.get(n['name'], {'w': None, 'a': 'fp32'}) for n in spec.nodes}, spec.nodes, ranges)
        entries = {e.split(':')[0]: e.split(':')[1:] for e in env.split(',')}
        ok = all(f'w={v["w"]}' in entries[n] and f'a={v["a"]}' in entries[n] for n, v in cfg.items())
        builds['compile_env_checks'][k] = {'entries': len(entries), 'all_196_match_config': bool(ok and len(entries) == 196)}
        if not builds['compile_env_checks'][k]['all_196_match_config']:
            raise SystemExit(f'{k}: compile env check failed')
        target = root / k
        target.mkdir(exist_ok=True)
        write(target / 'compile_env.json', {'MIXEDP_NODE_FORMATS': env, 'MIXEDP_WEIGHT_PER_CHANNEL': '1'})
        t1 = time.perf_counter()
        so, info = erm.build_so(tools, target, k, NATIVE / 'model.mlirbc', env, obj, opt='O0', krnl_opt='O3', constants_to_file=True, bytecode=True)
        if info['warnings']:
            raise SystemExit(f'{k}: warnings {info["warnings"]}')
        for f in target.iterdir():
            if f.suffix in ('.mlir', '.mlirbc', '.ll', '.bc'):
                f.unlink()
        builds[k] = {'so': prs.rel(so), 'so_sha256': sha(so), 'constants_sha256': sha(target / 'model.constants.bin'), 'seconds': time.perf_counter() - t1}
        write(out / 'builds.json', builds)
        log(f'built {k} in {builds[k]["seconds"]:.0f}s')
    write(out / 'builds.json', builds)
    ca.log_stage(out, 'build', time.perf_counter() - t0)


def worker(args):
    sys.path.insert(0, args.lib)
    from PyRuntime import OMExecutionSession
    from run_native_bc_pipeline import cross_entropy
    import fcntl
    hdr = json.loads(Path(args.header).read_text())
    if sha(args.tokens) != hdr['tokens_sha256'] or sha(Path(__file__).resolve()) != hdr['script_sha256']:
        raise SystemExit('worker: tokens or script differ from the header')
    path = Path(args.result)
    lock = open(str(path) + '.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    done = set()
    if path.exists():
        good = []
        for line in path.read_text().split('\n'):
            if not line:
                continue
            try:
                good.append(json.loads(line))
            except json.JSONDecodeError:
                break
        if not good or good[0].get('header') != hdr:
            raise SystemExit(f'worker: checkpoint {path} belongs to another version / data')
        path.write_text(''.join(json.dumps(r) + '\n' for r in good))
        done = {r['index'] for r in good[1:]}
    else:
        path.write_text(json.dumps({'header': hdr}) + '\n')
    x = np.load(args.tokens, mmap_mode='r')
    sess = OMExecutionSession(args.so, 'model')
    for i in range(hdr['start'], hdr['stop']):
        if i in done:
            continue
        t0 = time.perf_counter()
        w = np.ascontiguousarray(x[i:i + 1])
        z = np.array(sess.run([w])[0][0])
        if z.dtype != np.float32 or z.shape != (SEQ, VOCAB) or not np.isfinite(z).all():
            raise SystemExit(f'worker: bad logits at {i}')
        ce = cross_entropy(z[:-1], np.asarray(x[i, 1:]))
        row = {'index': i, 'ce_pos': [float(v) for v in ce], 'summed_ce': float(ce.sum()), 'window_sha256': hashlib.sha256(w.tobytes()).hexdigest(),
               'seconds': time.perf_counter() - t0}
        with path.open('a') as f:
            f.write(json.dumps(row) + '\n')
            f.flush()
            os.fsync(f.fileno())


def run_native(out, name, tokens, label, workers):
    import eval_random_mixed_precision as erm
    tools = erm.Tools()
    builds = read(out / 'builds.json')
    so = HERE / builds[name]['so']
    if sha(so) != builds[name]['so_sha256'] or sha(so.parent / 'model.constants.bin') != builds[name]['constants_sha256']:
        raise SystemExit(f'{name}: .so / constants differ from builds.json')
    target = out / 'native_results' / label / name
    target.mkdir(parents=True, exist_ok=True)
    x = np.load(tokens, mmap_mode='r')
    base = {'tokens_sha256': sha(tokens), 'so_sha256': builds[name]['so_sha256'], 'constants_sha256': builds[name]['constants_sha256'],
            'script_sha256': sha(Path(__file__).resolve()), 'config': name, 'config_sha256': sha(out / 'configs' / f'{name}.json')}
    comp = target / 'complete.json'
    if comp.exists():
        c = read(comp)
        if {k: c['header'][k] for k in base} != base:
            raise SystemExit(f'{comp}: from another version / data')
        return np.array(c['ce_pos'])
    idxs = [v for v in np.array_split(np.arange(len(x)), min(workers, len(x))) if len(v)]
    env = dict(os.environ, **THREAD_ENV)

    def one(k, inds):
        hdr = base | {'start': int(inds[0]), 'stop': int(inds[-1]) + 1, 'part': k, 'workers': len(idxs)}
        hp = target / f'part{k}.header.json'
        if hp.exists() and read(hp) != hdr:
            raise SystemExit(f'{hp}: header differs')
        write(hp, hdr)
        cmd = [sys.executable, str(Path(__file__).resolve()), '--stage', 'worker', '--out-dir', str(out), '--so', str(so), '--lib', str(tools.lib),
               '--tokens', str(tokens), '--result', str(target / f'part{k}.jsonl'), '--header', str(hp)]
        with (target / f'part{k}.log').open('a') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(idxs)) as pool:
        for f in as_completed([pool.submit(one, k, v) for k, v in enumerate(idxs)]):
            f.result()
    rows = []
    for k in range(len(idxs)):
        lines = [json.loads(l) for l in (target / f'part{k}.jsonl').read_text().splitlines() if l]
        rows += lines[1:]
    rows.sort(key=lambda r: r['index'])
    if [r['index'] for r in rows] != list(range(len(x))):
        raise SystemExit(f'{label} {name}: windows missing or duplicated')
    ce = np.array([r['ce_pos'] for r in rows])
    sum_dev = max(abs(sum(r['ce_pos']) - r['summed_ce']) for r in rows)
    write(comp, {'time': now(), 'header': base, 'windows': len(x), 'ce_pos': ce.tolist(), 'wall_s': time.perf_counter() - t0, 'pos_sum_dev': sum_dev})
    log(f'{label} {name}: {len(x)} windows in {time.perf_counter() - t0:.0f}s')
    return ce


def native_controls(out):
    t0 = time.perf_counter()
    check_plan(out)
    if (out / 'freeze.json').exists():
        raise SystemExit('frozen')
    if not read(out / 'gpu_controls.json')['passed']:
        raise SystemExit('gpu controls did not pass')
    tokens = out / 'work' / 'control_tokens.npy'
    np.save(tokens, np.load(NATIVE / 'calib_inputs.npy')[:2])
    prev = read(L27C / 'controls.json')
    res = {}
    for k in NATIVE_CFGS:
        res[k] = {'summed_ce': run_native(out, k, tokens, 'controls', 2).sum(1).tolist()}
    for k, ref in (('F_f16', 'A'), ('F_p8e1', 'C')):
        res[k]['previous'] = prev[ref]['summed_ce']
        res[k]['max_abs_error'] = float(np.abs(np.array(res[k]['summed_ce']) - np.array(prev[ref]['summed_ce'])).max())
    rep = out / 'work' / 'repeat_tokens.npy'
    np.save(rep, np.load(tokens)[:1])
    res['repeat_U_f16_window0_abs'] = float(abs(run_native(out, 'U_f16', rep, 'repeat', 1).sum() - res['U_f16']['summed_ce'][0]))
    b = read(out / 'builds.json')
    res['passed'] = bool(res['F_f16']['max_abs_error'] <= 1e-8 and res['F_p8e1']['max_abs_error'] <= 1e-8 and res['repeat_U_f16_window0_abs'] <= 1e-8
                         and all(v['all_196_match_config'] for v in b['compile_env_checks'].values()))
    write(out / 'native_controls.json', res)
    ca.log_stage(out, 'native_controls', time.perf_counter() - t0)
    if not res['passed']:
        raise SystemExit(f'native controls failed: {res}')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'builds_sha256': sha(out / 'builds.json'),
                                'gpu_controls_sha256': sha(out / 'gpu_controls.json'), 'native_controls_sha256': sha(out / 'native_controls.json'),
                                'configs_sha256': {k: sha(out / 'configs' / f'{k}.json') for k in NATIVE_CFGS}})
    log('native controls passed; frozen')


def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    getattr(prs, stage)(out)
    ca.log_stage(out, stage, time.perf_counter() - t0)


def verify_data(out):
    check_plan(out)
    dz = read(out / 'data_freeze.json')
    if sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'freeze.json') != dz['freeze_sha256']:
        raise SystemExit('data / freeze changed')


def native_eval(out):
    t0 = time.perf_counter()
    verify_data(out)
    tok = np.load(out / 'tokens.npy')
    first = out / 'work' / 'native_tokens.npy'
    if not first.exists():
        np.save(first, tok[0::WPA])
    vals, tm = {}, {}
    for k in NATIVE_CFGS:
        t1 = time.perf_counter()
        vals[k] = run_native(out, k, first, 'fresh', NATIVE_WORKERS)
        tm[k] = time.perf_counter() - t1
    np.savez(out / 'native_ce_pos.npz', **vals)
    ca.log_stage(out, 'native_eval', time.perf_counter() - t0, {'per_config_wall_s': tm, 'workers': NATIVE_WORKERS})


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(out):
    t0 = time.perf_counter()
    verify_data(out)
    rng = np.random.default_rng(SEEDS['bootstrap'])
    Wb = np.stack([np.bincount(rng.integers(0, N_ART, N_ART), minlength=N_ART) for _ in range(BOOT)]) / N_ART

    def st(v):
        b = Wb @ v
        return {'mean': float(v.mean()), 'ci95': np.percentile(b, [2.5, 97.5]).tolist(),
                'p_two_sided': float(min(1.0, 2 * min((b >= 0).mean(), (b <= 0).mean()))), 'articles_negative': int((v < 0).sum())}

    def parts(cp, wpa):
        """per article: pos0, rest, all contributions to the per-token NLL ([windows, 127] -> [articles])."""
        n = cp.shape[0] // wpa
        c = cp.reshape(n, wpa, -1)
        return {'pos0': c[:, :, 0].sum(1) / (wpa * TP), 'rest': c[:, :, 1:].sum((1, 2)) / (wpa * TP), 'all': c.sum((1, 2)) / (wpa * TP)}
    g = np.load(out / 'gpu_ce_pos.npz')
    G = {k: parts(g[k], WPA) for k in g.files}
    D = {b: {p: G[f'{b}__p8e1'][p] - G[f'{b}__f16'][p] for p in ('pos0', 'rest', 'all')} for b in GPU_BG}
    res = {'units': 'nat/token contributions; D = p8e1 - f16 (negative = p8e1 better)',
           'gpu_D': {b: {p: st(v) for p, v in d.items()} for b, d in D.items()}}
    ctx = {b: st(D[b]['pos0'] - D['U']['pos0']) for b in SPLITS}
    order = sorted(ctx, key=lambda k: ctx[k]['p_two_sided'])
    run_ = 0.0
    for j, k in enumerate(order):
        run_ = max(run_, min(1.0, (len(order) - j) * ctx[k]['p_two_sided']))
        ctx[k]['p_holm'] = run_
    res['gpu_pos0_context_change_splits'] = ctx
    res['gpu_pos0_context_change_reference'] = {b: st(D[b]['pos0'] - D['U']['pos0']) for b in ('dn', 'F')}
    res['gpu_rest_context_change'] = {b: st(D[b]['rest'] - D['U']['rest']) for b in ('dn', 'dn_w', 'dn_a', 'dn_ax', 'dn_ay', 'F')}
    cu = lambda b: D[b]['pos0'] - D['U']['pos0']
    res['gpu_additivity_pos0'] = {'dn - (w + a)': st(cu('dn') - cu('dn_w') - cu('dn_a')), 'a - (ax + ay)': st(cu('dn_a') - cu('dn_ax') - cu('dn_ay'))}
    nz = np.load(out / 'native_ce_pos.npz')
    Nn = {k: parts(nz[k], 1) for k in nz.files}
    gfirst = {k: parts(g[k][0::WPA], 1) for k in g.files}
    nat = {}
    for p in ('pos0', 'rest', 'all'):
        dU, dF = Nn['U_p8e1'][p] - Nn['U_f16'][p], Nn['F_p8e1'][p] - Nn['F_f16'][p]
        gU, gF = gfirst['U__p8e1'][p] - gfirst['U__f16'][p], gfirst['F__p8e1'][p] - gfirst['F__f16'][p]
        nat[p] = {'native_D_U': st(dU), 'native_D_F': st(dF), 'native_context_change': st(dF - dU), 'gpu_same_windows_context_change': st(gF - gU),
                  'backend_difference': st((dF - dU) - (gF - gU)),
                  'correlation': float(np.corrcoef(dF - dU, gF - gU)[0, 1]), 'sign_agreement': int((np.sign(dF - dU) == np.sign(gF - gU)).sum())}
    res['native'] = nat
    res['nll'] = {'native': {k: float(Nn[k]['all'].mean()) for k in Nn}, 'gpu_first_windows': {k: float(gfirst[k]['all'].mean()) for k in ('U__f16', 'U__p8e1', 'F__f16', 'F__p8e1')}}
    write(out / 'stats.json', res)
    ca.log_stage(out, 'analyze', time.perf_counter() - t0)
    log('done')


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'gpu_controls', 'build', 'native_controls', 'audit', 'data', 'gpu_eval', 'native_eval',
                                                       'analyze', 'worker'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--smoke-note')
    for k in ('so', 'lib', 'tokens', 'result', 'header'):
        ap.add_argument('--' + k)
    a = ap.parse_args()
    if a.stage == 'worker':
        worker(a)
        return
    out = a.out_dir.resolve()
    if a.stage == 'plan':
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f'{out} is not empty')
    elif not (out / 'plan.json').exists():
        raise SystemExit('plan first')
    out.mkdir(parents=True, exist_ok=True)
    for d in ('logs', 'work'):
        (out / d).mkdir(exist_ok=True)
    LOGF = open(out / 'logs' / f'{a.stage}.log', 'a')
    t0 = time.time()
    if a.stage == 'plan':
        plan(out, a.smoke_note)
    elif a.stage in ('audit', 'data'):
        sampler(out, a.stage)
    else:
        globals()[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen3_l27_down_source.py')}) + '\n')


if __name__ == '__main__':
    main()
