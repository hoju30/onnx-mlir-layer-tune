#!/usr/bin/env python3
"""Qwen3-0.6B L27 Down: why does the original BC rank int4 above p4e1 and f16 far above p8e1 while the measured loss
prefers p4e1 and p8e1? A single-point diagnostic (no BC / search / knee / range / quantizer change, no fine-tuning).

Setting (as the BC activation term C assumes): every weight and activation FP32 except the L27 Down activation QDQ.
For format f and slot s in {x (Down input), y (Down output), xy (both, the configuration semantics)}, position mask m
in {all, pos0 (position 0 of each window), rest (positions 1..127)}:
  measured   dL(alpha) = summed CE with the perturbation alpha * e_s * m  minus the FP32 summed CE  (e_s = Q(s) - s);
             alpha = +1 (the real quantization), -1, +0.5, -0.5 for x and y; alpha = 1 for xy
             linear part a = (dL(1) - dL(-1)) / 2,  quadratic part b = (dL(1) + dL(-1)) / 2
  in-sample  BC  T_old(e_s*m) = sum_c G2_c / n (S2_c - n mu_c^2) + GS_c mu_c^2  (the native formula, n = windows x 128),
             V_pair, F = sum g.e (first order), dir = sum_windows (g_w.e)^2 (empirical-Fisher quadratic, x2)
L27 is the last block: each condition recomputes only the Down projection, the final norm and the LM head.
Data: the original 10 calibration windows (BC's own data) and 128 new WikiText-103 articles x 2 windows.

Stages (one new --out-dir): plan, calib, freeze, audit, data, eval, analyze.
CPU: ../../venv/bin/python; calib / eval: PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_qwen_prefix_rank_shift as prs  # noqa: E402  (read / write / sha / now / sampler; standard imports only)

NATIVE = HERE / 'runs/native_eps_fix_20261014/qwen3'
PREP = HERE / 'runs/qwen3_prepare_20261011'
PRS = HERE / 'runs/qwen3_prefix_rank_shift_20261024'
NODE = '/model/model/layers.27/mlp/down_proj/MatMul'
LAYER, SEQ, TP, CAL = 27, 128, 127, 10
FORMATS = ('int4', 'p4e1', 'int6', 'p6e1', 'int8', 'p8e1', 'f16')
PAIRS = (('int4', 'p4e1'), ('int6', 'p6e1'), ('int8', 'p8e1'), ('f16', 'p8e1'))
FOCUS = (('int4', 'p4e1'), ('f16', 'p8e1'))
SLOTS, MASKS, ALPHAS = ('x', 'y'), ('all', 'pos0', 'rest'), (1.0, -1.0, 0.5, -0.5)
N_ART, WPA, BOOT, CHUNK, LM_SUB = 128, 2, 20000, 32, 8
SEEDS = {'order': 31101, 'start': 31102, 'bootstrap': 31103}
TOL = {'baseline_vs_full': 0.0, 'native_C_rel': 1e-2}
CODE = ['run_qwen_l27_reversal_diag.py', 'run_qwen_prefix_rank_shift.py', 'run_qwen_down_spike_diag.py',
        'run_qwen_down_fp16_protection.py', 'qwen3_swiglu_common.py', 'eval_random_mixed_precision.py', 'wt_manifest_scan.py',
        'run_olmoe_route_mass_restore.py', 'run_olmoe_dynamic_route.py', 'run_lp_pipeline_eval.py']
read, write, sha, now = prs.read, prs.write, prs.sha, prs.now
LOGF = None


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def log_stage(out, stage, seconds, extra=None):
    p = out / 'timing.json'
    d = read(p) if p.exists() else {}
    d[stage] = {'seconds': seconds, **(extra or {})}
    write(p, d)


def input_files():
    import run_olmoe_route_mass_restore as mr
    return ([HERE / 'qwen3_assets/model' / f for f in ('config.json', 'model.safetensors', 'tokenizer.json')] +
            [NATIVE / 'table.json', NATIVE / 'ranges_by_point.json', NATIVE / 'calib_inputs.npy', PREP / 'validation_tokens.npy',
             PREP / 'test_tokens.npy', PREP / 'node_mapping.json', PRS / 'segments.json',
             HERE / 'qwen35_assets/model/tokenizer.json'] + mr.WT103 + mr.WT2)


def check_plan(out, inputs=True):
    pl = read(out / 'plan.json')
    for f, h in pl['code_hashes'].items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan: {f} (record an amendment first)')
    if inputs:
        for f, h in pl['inputs'].items():
            if sha(HERE / f) != h:
                raise SystemExit(f'input changed: {f}')
    return pl


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(out):
    import wt_manifest_scan as wms
    t0 = time.perf_counter()
    if (out / 'plan.json').exists():
        raise SystemExit('plan.json exists')
    table = read(NATIVE / 'table.json')
    w103, wt2 = wms.corpora()
    used, _ = wms.used_wt103_verified(out, w103, wt2)
    prior = {s['content_sha256'] for s in read(PRS / 'segments.json')}
    if not prior <= used:
        raise SystemExit('previous round articles not in the verified scan')
    T2, C2 = {a['nt'] for a in wt2}, {a['ch'] for a in wt2}
    ct, cc = Counter(a['nt'] for a in w103), Counter(a['ch'] for a in w103)
    elig = [a for a in w103 if a['nt'] not in T2 and a['ch'] not in C2 and ct[a['nt']] == 1 and cc[a['ch']] == 1 and a['ch'] not in used]
    bc = {f: table['C'][NODE][f] for f in FORMATS}
    from run_olmoe_dynamic_route import env_info
    pl = {
        'time': now(),
        'question': 'L27 Down: why do the BC table (C: int4 6482 < p4e1 7177; f16 0.0003 << p8e1 160) and the measured loss (p4e1 '
                    'and p8e1 better) disagree? Which slot (input x / output y), which positions (0 / rest), which term (linear / '
                    'curvature) and whether the BC formula is wrong on its own windows (approximation) or only on new text (transfer)',
        'scope': 'single-point diagnostic; every weight and activation FP32 except the L27 Down activation QDQ (the BC C context); '
                 'no BC / search / knee / range / quantizer / pipeline change; GPU FP32 + QDQ (no native claim)',
        'bc_table_L27': bc,
        'conditions': {'formats': list(FORMATS), 'slots': ['x', 'y', 'xy'], 'masks': list(MASKS), 'alphas_x_y': list(ALPHAS),
                       'alpha_xy': [1.0], 'count': len(FORMATS) * (len(SLOTS) * len(MASKS) * len(ALPHAS) + len(MASKS)),
                       'quantizers': 'native calibration x / y ranges of the L27 Down node (int: erm.int_qparams + int_act_qdq on the '
                                     'device), posit LUT (Universal), f16 cast; never recalibrated',
                       'perturbation': 'x: Down(h + alpha e_x m); y: y + alpha e_y m; xy: Q_y on Down(Q_x(h) on m) on m; alpha = 1 and '
                                       'mask all is exactly the QDQ'},
        'measures': {'dL': 'summed CE minus the same-path FP32 summed CE, per window; per token = / 127',
                     'a, b': 'a = (dL(1) - dL(-1)) / 2 (odd part), b = (dL(1) + dL(-1)) / 2 (even part); cubic check r = dL(0.5) - '
                             '(0.5 a + 0.25 b)',
                     'in-sample scores (FP32 forward / backward on the same windows, g = d summed CE / d slot, position 127 g = 0, '
                     'n = windows x 128)': {'BC': 'T_old(e m) (native formula)', 'V_old, U': 'its variance and bias parts',
                                            'V_pair': 'sum g^2 (e - mu)^2 + U', 'F': 'sum g.e (first-order prediction of a)',
                                            'Fisher2': '1/2 sum_w (g_w.e)^2 (empirical-Fisher prediction of b)',
                                            'BC2': '1/2 T_old (BC read as a prediction of b)'}},
        'rules': {'pair_D': 'D = dL(INT or f16) - dL(Posit) per article; positive = Posit better; same for a, b and BC terms',
                  'slot': 'a slot carries the reversal if its measured D (all positions) has the sign opposite to its BC prediction',
                  'position': 'position 0 explains the reversal if BC computed without position 0 (mask rest) has the measured sign',
                  'term': 'curvature right / linear wins: sign(D_b) = BC sign and sign(D) = sign(D_a) != BC sign; curvature wrong: '
                          'sign(D_b) != BC sign',
                  'approximation_vs_transfer': 'the in-sample BC on the calibration windows (BC\'s own data) and on the new articles '
                                               'both reversed -> approximation of the formula; only new articles -> transfer',
                  'uncertainty': 'new articles: paired article bootstrap (20000, seed 31103) 95% CI for every measured mean; a sign is '
                                 'claimed only if its CI excludes 0, otherwise undetermined; calibration (10 windows) descriptive',
                  'additivity': 'xy vs x + y (interaction), descriptive'},
        'controls': ['same-path FP32 replay equals the complete forward bitwise (calibration and every new chunk)',
                     'alpha = 1, mask all equals the QDQ; alpha = 0 equals FP32 (zero perturbation)',
                     f'calibration in-sample BC x + y vs native table C[L27] within {TOL["native_C_rel"]} relative',
                     'y is the Down output (= F.linear(h, W)), x the Down input after SiLU(gate) * up; position 127 gradient 0',
                     'posit LUT verified; TF32 off; deterministic; float32 forward, float64 sums'],
        'data': {'calibration': 'original 10 x 128 windows', 'new': f'{N_ART} new WikiText-103 articles x {WPA} windows (rule of '
                 'run_qwen_prefix_rank_shift.data: verified scan, WT2 / near duplicates, historical Qwen windows, window-WT2 >= 50%)',
                 'excluded_previous_wt103': len(used), 'eligible_before_length_and_neardup': len(elig), 'seeds': SEEDS},
        'boundaries': 'diagnostic only; no criterion proposal or pipeline change in this round',
        'pre_plan_smoke_test': 'before this plan a scratch run (outside runs/) executed calib on the 10 calibration windows and '
                               'eval / analyze on those windows tiled to 256 (no new article): the calibration-side measurements '
                               'and in-sample BC terms were seen; nothing of the design was changed',
        'inputs': {prs.rel(p): sha(p) for p in input_files()}, 'code_hashes': {f: sha(HERE / f) for f in CODE},
        'environment_cpu': env_info()}
    write(out / 'plan.json', pl)
    md = ['# 計畫：Qwen3 L27 Down 的 BC 預測與實測誤差為何反轉', '',
          f'時間：{pl["time"]}。以 plan.json 為準。', '',
          f'- 問題：BC 表 C[L27 Down]：int4 {bc["int4"]:.0f} < p4e1 {bc["p4e1"]:.0f}、f16 {bc["f16"]:.4f} ≪ p8e1 {bc["p8e1"]:.0f}；實測卻是 p4e1、p8e1 較好。拆解：槽位（input x／output y）、位置（0／其餘）、誤差項（一階／曲率），以及 BC 公式在自己的 calibration 上是否就反轉（近似誤差）或只在新文本反轉（轉移）。',
          '- 設定：與 BC 的 C 項相同，除 L27 Down activation QDQ 外全部 FP32（含權重）。L27 是最後一層，只重算 Down、final norm、LM head。',
          f'- 條件：{len(FORMATS)} 格式 × x、y 各 3 遮罩 × α（1、−1、0.5、−0.5）+ xy × 3 遮罩（α=1）= {pl["conditions"]["count"]}。ranges、LUT 固定。',
          '- 量測：dL(α)；a = (dL(1) − dL(−1))/2（奇次／線性部分）；b = (dL(1) + dL(−1))/2（偶次／曲率部分）；三次檢查。',
          '- in-sample 分數（同一批窗口的 FP32 前向／反向）：BC = T_old(e·m)（native 公式）及其 V_old、U；V_pair；F = Σg·e（一階預測）；½Σ_w(g_w·e)²（經驗 Fisher 的二階預測）；½T_old（把 BC 當成 b 的預測）。',
          '- 判讀規則：成對 D = dL(INT 或 f16) − dL(Posit)（正值 Posit 較好）。某槽位的實測 D 與其 BC 預測反號 → 該槽位承載反轉；BC 去掉 position 0 後與實測同號 → position 0 解釋反轉；D_b 與 BC 同號而 D、D_a 反號 → 曲率對、一階項翻盤；D_b 與 BC 反號 → 曲率近似錯。calibration 與新文章的 in-sample BC 都反轉 → 公式近似誤差；只有新文章 → 轉移。',
          f'- 不確定性：新文章以文章 paired bootstrap（{BOOT} 次，seed {SEEDS["bootstrap"]}）95% CI；CI 跨 0 即未定；calibration 只作描述。',
          f'- 控制：同路徑 FP32 = 完整前向（逐位元）；α=1 全遮罩 = QDQ；calibration in-sample BC x+y vs native C[L27] 相對差 ≤ {TOL["native_C_rel"]}；梯度 position 127 = 0；LUT 驗證；TF32 關閉、deterministic。',
          f'- 資料：原 10 個 calibration 窗口；{N_ART} 篇新 WT103 文章 × {WPA} 窗口（沿用前輪抽樣規則；verified 掃描排除 {len(used)} 篇，含前輪 128 篇）；seed 順序 {SEEDS["order"]}、起點 {SEEDS["start"]}。',
          '- 邊界：只做診斷，不提出 criterion 修改、不改 pipeline。',
          '- 計畫前冒煙測試（scratchpad，不在 runs/）：calib 跑過原 10 個 calibration 窗口，eval／analyze 跑過重複填充成 256 窗口的 calibration（沒有新文章）；已看過 calibration 端的量測與 in-sample BC 項；設計未改。', '']
    (out / 'plan.md').write_text('\n'.join(md) + '\n')
    log_stage(out, 'plan', time.perf_counter() - t0)
    log(f'plan written; used {len(used)}; eligible {len(elig)}')


# --------------------------------------------------------------------------- #
# GPU core
# --------------------------------------------------------------------------- #
class Core:
    def __init__(self, work):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        import qwen3_swiglu_common as qc
        from run_qwen_down_spike_diag import Model
        self.torch, self.qc, self.erm = torch, qc, qc.erm
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        torch.use_deterministic_algorithms(True)
        self.M = Model('cuda')
        self.q = qc.Quantizers(work, formats=('p4e1', 'p6e1', 'p8e1'))
        self.lut = {f: (l.t_bounds.cuda(), l.t_values.cuda()) for f, l in self.q.posit.items()}
        self.ranges = read(NATIVE / 'ranges_by_point.json')[NODE]
        self.layer = self.M.layers[LAYER]
        self.W = self.layer.mlp.down_proj.weight

    def qact(self, t, f, slot):
        torch = self.torch
        if f.startswith('int'):
            bits = int(f[3:])
            lo, hi = self.ranges[slot]
            scale, zp = self.erm.int_qparams(float(lo), float(hi), bits)
            return self.qc.int_act_qdq(t, scale, zp, bits)
        if f == 'f16':
            return t.to(torch.float16).to(t.dtype)
        b, v = self.lut[f]
        return v[torch.searchsorted(b, self.qc._order_key_torch(t.float()), right=True)].to(t.dtype)

    def capture(self, ids):
        """complete forward; returns per-window summed CE and the L27 residual before the MLP, Down input and output."""
        torch = self.torch
        cap = {}
        hk = [self.layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a: cap.__setitem__('mid', a[0].detach())),
              self.layer.mlp.down_proj.register_forward_hook(lambda m, a, o: cap.update(h=a[0].detach(), y=o.detach()))]
        with torch.no_grad():
            hs = self.M.model.model(input_ids=ids, attention_mask=self.M.wrapper.mask, use_cache=False).last_hidden_state
            ce = self.head_ce(hs, ids, normed=True)
        for h in hk:
            h.remove()
        return ce, cap['mid'], cap['h'], cap['y']

    def head_ce(self, t, ids, normed=False):
        torch = self.torch
        out = []
        for j in range(0, t.shape[0], LM_SUB):
            n_ = t[j:j + LM_SUB] if normed else self.M.model.model.norm(t[j:j + LM_SUB])
            z = self.M.model.lm_head(n_)
            ce = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), ids[j:j + LM_SUB, 1:].reshape(-1), reduction='none')
            out.append(ce.view(z.shape[0], -1).double().sum(1))
            del z
        return torch.cat(out)

    def grads(self, mid, y, ids):
        """g_y = d(summed CE of each window)/dy through the final norm and LM head; g_x = g_y W."""
        torch = self.torch
        gy = torch.empty_like(y)
        for j in range(0, y.shape[0], LM_SUB):
            yj = y[j:j + LM_SUB].clone().requires_grad_(True)
            z = self.M.model.lm_head(self.M.model.model.norm(mid[j:j + LM_SUB] + yj))
            loss = torch.nn.functional.cross_entropy(z[:, :-1].reshape(-1, z.shape[-1]).float(), ids[j:j + LM_SUB, 1:].reshape(-1), reduction='sum')
            loss.backward()
            gy[j:j + LM_SUB] = yj.grad
            del z, loss
        with torch.no_grad():
            gx = gy @ self.W
        return gx, gy


def masks(torch, nb, device):
    pos = torch.arange(SEQ, device=device)[None, :, None]
    return {'all': torch.ones(1, SEQ, 1, dtype=torch.bool, device=device), 'pos0': pos == 0, 'rest': pos > 0}


def run_block(core, ids, acc, F_rows):
    """all conditions on one block of windows: per-window dL into a dict, in-sample sums accumulated into acc."""
    torch = core.torch
    lin = torch.nn.functional.linear
    ce_full, mid, h, y = core.capture(ids)
    with torch.no_grad():
        base = core.head_ce(mid + lin(h, core.W), ids)
    ok_base = bool(torch.equal(base, ce_full))
    gx, gy = core.grads(mid, y, ids)
    last_zero = bool((gx[:, -1] == 0).all() and (gy[:, -1] == 0).all())
    M = masks(torch, ids.shape[0], ids.device)
    res = {}
    with torch.no_grad():
        y_check = bool(torch.equal(lin(h, core.W), y))
        g = {'x': gx.double(), 'y': gy.double()}
        for s, t in (('x', h), ('y', y)):
            g2 = g[s] ** 2
            acc.setdefault(f'G2|{s}', 0)
            acc[f'G2|{s}'] = acc[f'G2|{s}'] + g2.sum((0, 1))
            acc[f'GS|{s}'] = acc.get(f'GS|{s}', 0) + (g[s].sum(1) ** 2).sum(0)
            acc[f'n|{s}'] = acc.get(f'n|{s}', 0) + t.shape[0] * t.shape[1]
            for f in FORMATS:
                qt = core.qact(t, f, s)
                e = (qt - t)
                for mk, m in M.items():
                    em = (e * m).double()
                    k = f'{s}|{f}|{mk}'
                    acc[f'S1|{k}'] = acc.get(f'S1|{k}', 0) + em.sum((0, 1))
                    acc[f'S2|{k}'] = acc.get(f'S2|{k}', 0) + (em ** 2).sum((0, 1))
                    acc[f'P2|{k}'] = acc.get(f'P2|{k}', 0) + (g2 * em ** 2).sum((0, 1))
                    acc[f'P1|{k}'] = acc.get(f'P1|{k}', 0) + (g2 * em).sum((0, 1))
                    Fw = (g[s] * em).sum((1, 2))
                    F_rows.setdefault(k, []).append(Fw.cpu().numpy())
                    for al in ALPHAS:
                        if al == 1.0:
                            tp = torch.where(m, qt, t)
                        else:
                            tp = t + al * e * m
                        yp = lin(tp, core.W) if s == 'x' else tp
                        res[f'{k}|{al:g}'] = (core.head_ce(mid + yp, ids) - base).cpu().numpy()
            if s == 'x':
                for f in FORMATS:
                    qx = core.qact(h, f, 'x')
                    for mk, m in M.items():
                        y0 = lin(torch.where(m, qx, h), core.W)
                        yp = torch.where(m, core.qact(y0, f, 'y'), y0)
                        res[f'xy|{f}|{mk}|1'] = (core.head_ce(mid + yp, ids) - base).cpu().numpy()
        zero = (core.head_ce(mid + lin(h + 0.0 * (core.qact(h, 'int4', 'x') - h), core.W), ids) - base).abs().max().item()
        full_q = bool(torch.equal(core.head_ce(mid + lin(core.qact(h, 'int4', 'x'), core.W), ids) - base,
                                  torch.from_numpy(res['x|int4|all|1']).to(base.device)))
    ctl = {'baseline_equals_full_forward': ok_base, 'grad_last_position_zero': last_zero, 'y_equals_linear_h': y_check,
           'alpha0_max_abs': zero, 'alpha1_all_equals_qdq': full_q}
    return res, base.cpu().numpy(), ctl


def scores_from(acc, F_rows):
    """in-sample BC terms per slot / format / mask from the accumulated sums."""
    out = {}
    for s in SLOTS:
        G2, GS, n = acc[f'G2|{s}'], acc[f'GS|{s}'], acc[f'n|{s}']
        for f in FORMATS:
            for mk in MASKS:
                k = f'{s}|{f}|{mk}'
                S1, S2, P2, P1 = acc[f'S1|{k}'], acc[f'S2|{k}'], acc[f'P2|{k}'], acc[f'P1|{k}']
                mu = S1 / n
                V = float((G2 / n * (S2 - n * mu ** 2)).sum())
                U = float((GS * mu ** 2).sum())
                Vp = float((P2 - 2 * mu * P1 + mu ** 2 * G2).sum())
                Fw = np.concatenate(F_rows[k])
                out[k] = {'BC': V + U, 'V_old': V, 'U': U, 'T_pair': Vp + U, 'F': float(Fw.sum()), 'Fisher2': float(0.5 * (Fw ** 2).sum()),
                          'BC2': 0.5 * (V + U), 'F_per_window': Fw.tolist()}
    return out


def run_dataset(core, toks, label, out):
    torch = core.torch
    acc, F_rows, res_all, base_all, ctls = {}, {}, {}, [], []
    for c0 in range(0, len(toks), CHUNK):
        ids = torch.from_numpy(toks[c0:c0 + CHUNK]).cuda()
        res, base, ctl = run_block(core, ids, acc, F_rows)
        for k, v in res.items():
            res_all.setdefault(k, []).append(v)
        base_all.append(base)
        ctls.append(ctl)
        log(f'{label}: windows {c0}-{c0 + len(ids)} done; {ctl}')
    res_all = {k: np.concatenate(v) for k, v in res_all.items()}
    sc = scores_from(acc, F_rows)
    np.savez(out / f'{label}_dL.npz', base=np.concatenate(base_all), **{k.replace('|', '__'): v for k, v in res_all.items()})
    write(out / f'{label}_scores.json', sc)
    return res_all, sc, ctls


def calib(out):
    t0 = time.perf_counter()
    check_plan(out)
    core = Core(out / 'work')
    torch = core.torch
    toks = np.load(NATIVE / 'calib_inputs.npy')
    res, sc, ctls = run_dataset(core, toks, 'calibration', out)
    table = read(NATIVE / 'table.json')['C'][NODE]
    nat = {f: {'pytorch_x_plus_y': sc[f'x|{f}|all']['BC'] + sc[f'y|{f}|all']['BC'], 'native_C': table[f]} for f in FORMATS}
    for f in nat:
        nat[f]['rel'] = abs(nat[f]['pytorch_x_plus_y'] - nat[f]['native_C']) / abs(nat[f]['native_C'])
    ctl = {'chunks': ctls, 'posit_lut': {f: list(core.q.posit[f].verify()) for f in core.q.posit}, 'native_C_check': nat,
           'tf32': [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32], 'deterministic': torch.are_deterministic_algorithms_enabled(),
           'calibration_tokens_equal_prepare': bool(np.array_equal(toks, np.load(PREP / 'validation_tokens.npy')[:CAL]))}
    ctl['passed'] = bool(all(c['baseline_equals_full_forward'] and c['grad_last_position_zero'] and c['y_equals_linear_h'] and c['alpha0_max_abs'] == 0
                             and c['alpha1_all_equals_qdq'] for c in ctls) and all(v[0] == 0 for v in ctl['posit_lut'].values())
                         and all(v['rel'] <= TOL['native_C_rel'] for v in nat.values()) and ctl['tf32'] == [False, False]
                         and ctl['deterministic'] and ctl['calibration_tokens_equal_prepare'])
    write(out / 'controls_calib.json', ctl)
    log_stage(out, 'calib', time.perf_counter() - t0)
    log(f'calibration controls passed={ctl["passed"]}; native C rel max {max(v["rel"] for v in nat.values()):.2e}')
    if not ctl['passed']:
        raise SystemExit('controls failed')


def freeze(out):
    check_plan(out)
    if not read(out / 'controls_calib.json')['passed']:
        raise SystemExit('controls failed')
    write(out / 'freeze.json', {'time': now(), 'plan_sha256': sha(out / 'plan.json'), 'controls_sha256': sha(out / 'controls_calib.json'),
                                'calibration_scores_sha256': sha(out / 'calibration_scores.json'), 'calibration_dL_sha256': sha(out / 'calibration_dL.npz'),
                                'script_sha256': sha(HERE / 'run_qwen_l27_reversal_diag.py')})


def sampler(out, stage):
    check_plan(out)
    if not (out / 'freeze.json').exists():
        raise SystemExit('freeze first')
    t0 = time.perf_counter()
    prs.check_plan, prs.log_stage, prs.N_ART, prs.WPA, prs.SEEDS = check_plan, (lambda *a, **k: None), N_ART, WPA, SEEDS
    prs.LOGF = LOGF
    getattr(prs, stage)(out)
    log_stage(out, stage, time.perf_counter() - t0)


def eval_(out):
    t0 = time.perf_counter()
    check_plan(out)
    dz = read(out / 'data_freeze.json')
    if sha(out / 'tokens.npy') != dz['tokens_sha256'] or sha(out / 'freeze.json') != dz['freeze_sha256']:
        raise SystemExit('data / freeze changed')
    core = Core(out / 'work')
    _, _, ctls = run_dataset(core, np.load(out / 'tokens.npy'), 'new', out)
    ok = all(c['baseline_equals_full_forward'] and c['grad_last_position_zero'] and c['alpha0_max_abs'] == 0 and c['alpha1_all_equals_qdq'] for c in ctls)
    write(out / 'controls_eval.json', {'chunks': ctls, 'passed': bool(ok)})
    log_stage(out, 'eval', time.perf_counter() - t0, {'peak_gpu_memory_bytes': int(core.torch.cuda.max_memory_allocated())})
    if not ok:
        raise SystemExit('eval controls failed')


# --------------------------------------------------------------------------- #
# analyze
# --------------------------------------------------------------------------- #
def analyze(out):
    t0 = time.perf_counter()
    check_plan(out)
    rng = np.random.default_rng(SEEDS['bootstrap'])
    W = np.stack([np.bincount(rng.integers(0, N_ART, N_ART), minlength=N_ART) for _ in range(BOOT)]).astype(np.float64) / N_ART
    data = {}
    for label, nwin in (('calibration', CAL), ('new', N_ART * WPA)):
        z = np.load(out / f'{label}_dL.npz')
        data[label] = ({k.replace('__', '|'): z[k] / TP for k in z.files if k != 'base'}, read(out / f'{label}_scores.json'), nwin)

    def art(v):
        return v.reshape(N_ART, WPA).mean(1)

    def ci(v):
        b = W @ v
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    rows, pairs = [], []
    for label, (dl, sc, nwin) in data.items():
        for s in ('x', 'y', 'xy'):
            for f in FORMATS:
                for mk in MASKS:
                    k = f'{s}|{f}|{mk}'
                    r = {'data': label, 'slot': s, 'format': f, 'mask': mk, 'dL': float(dl[f'{k}|1'].mean())}
                    if s != 'xy':
                        a = (dl[f'{k}|1'] - dl[f'{k}|-1']) / 2
                        b = (dl[f'{k}|1'] + dl[f'{k}|-1']) / 2
                        cub = dl[f'{k}|0.5'] - (0.5 * a + 0.25 * b)
                        r.update(a=float(a.mean()), b=float(b.mean()), cubic_resid=float(cub.mean()),
                                 **{kk: sc[k][kk] / (nwin * TP) for kk in ('BC', 'V_old', 'U', 'T_pair', 'F', 'Fisher2', 'BC2')})
                    if label == 'new':
                        r['dL_ci'] = ci(art(dl[f'{k}|1']))
                        if s != 'xy':
                            r['a_ci'], r['b_ci'] = ci(art(a)), ci(art(b))
                            r['F_ci'] = ci(art(np.array(sc[k]['F_per_window']) / TP))
                    rows.append(r)
        for s in ('x', 'y', 'xy'):
            for p, q in PAIRS:
                for mk in MASKS:
                    kp, kq = f'{s}|{p}|{mk}', f'{s}|{q}|{mk}'
                    D = dl[f'{kp}|1'] - dl[f'{kq}|1']
                    r = {'data': label, 'slot': s, 'pair': f'{p}-{q}', 'mask': mk, 'D': float(D.mean())}
                    if s != 'xy':
                        Da = (dl[f'{kp}|1'] - dl[f'{kp}|-1']) / 2 - (dl[f'{kq}|1'] - dl[f'{kq}|-1']) / 2
                        Db = (dl[f'{kp}|1'] + dl[f'{kp}|-1']) / 2 - (dl[f'{kq}|1'] + dl[f'{kq}|-1']) / 2
                        r.update(D_a=float(Da.mean()), D_b=float(Db.mean()), **{f'BC_D_{kk}': (sc[kp][kk] - sc[kq][kk]) / (nwin * TP)
                                                                              for kk in ('BC', 'V_old', 'U', 'T_pair', 'Fisher2', 'F')})
                    if label == 'new':
                        r['D_ci'] = ci(art(D))
                        if s != 'xy':
                            r['D_a_ci'], r['D_b_ci'] = ci(art(Da)), ci(art(Db))
                    pairs.append(r)
    # additivity xy vs x + y (all positions)
    add = []
    for label, (dl, _, _) in data.items():
        for f in FORMATS:
            I = dl[f'xy|{f}|all|1'] - dl[f'x|{f}|all|1'] - dl[f'y|{f}|all|1']
            add.append({'data': label, 'format': f, 'interaction': float(I.mean()), **({'ci': ci(art(I))} if label == 'new' else {})})
    # attribution per focus pair, by the planned rules
    sgn = lambda v: 0 if v == 0 else (1 if v > 0 else -1)
    det = lambda c: c is not None and (c[0] > 0 or c[1] < 0)
    attr = {}
    P = {(r['data'], r['slot'], r['pair'], r['mask']): r for r in pairs}
    for p, q in PAIRS:
        for label in data:
            for s in SLOTS:
                al, re, p0 = P[(label, s, f'{p}-{q}', 'all')], P[(label, s, f'{p}-{q}', 'rest')], P[(label, s, f'{p}-{q}', 'pos0')]
                meas = sgn(al['D'])
                certain = label == 'calibration' or det(al.get('D_ci'))
                bc_all, bc_rest = sgn(al['BC_D_BC']), sgn(re['BC_D_BC'])
                attr[f'{label}|{s}|{p}-{q}'] = {
                    'measured_D': al['D'], 'measured_D_ci': al.get('D_ci'), 'measured_sign_certain': bool(certain),
                    'bc_sign': bc_all, 'measured_sign': meas, 'slot_reversed': bool(certain and meas != bc_all),
                    'bc_without_pos0_sign': bc_rest, 'position0_explains': bool(certain and meas != bc_all and bc_rest == meas),
                    'D_a': al['D_a'], 'D_b': al['D_b'], 'D_a_ci': al.get('D_a_ci'), 'D_b_ci': al.get('D_b_ci'),
                    'term': ('curvature right, linear term wins' if sgn(al['D_b']) == bc_all and meas == sgn(al['D_a']) != bc_all else
                             'curvature wrong' if sgn(al['D_b']) != bc_all else 'BC right'),
                    'term_certain': bool(label == 'calibration' or (det(al.get('D_b_ci')) and det(al.get('D_a_ci')))),
                    'measured_pos0_D': p0['D'], 'measured_rest_D': re['D'], 'bc_pos0_D': p0['BC_D_BC'], 'bc_rest_D': re['BC_D_BC'],
                    'measured_b_pos0': p0['D_b'], 'measured_b_rest': re['D_b'], 'fisher2_D': al['BC_D_Fisher2'], 'T_pair_D': al['BC_D_T_pair'],
                    'U_D': al['BC_D_U'], 'V_D': al['BC_D_V_old'], 'F_D': al['BC_D_F']}
    res = {'conditions': rows, 'pairs': pairs, 'additivity': add, 'attribution': attr, 'units': 'nat/token (summed CE / 127 per window, averaged)'}
    write(out / 'stats.json', res)
    with open(out / 'conditions.csv', 'w', newline='') as fh:
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)
    with open(out / 'pairs.csv', 'w', newline='') as fh:
        keys = sorted({k for r in pairs for k in r}, key=lambda k: list(pairs[0]).index(k) if k in pairs[0] else 99)
        wr = csv.DictWriter(fh, fieldnames=keys)
        wr.writeheader()
        wr.writerows(pairs)
    figures(out, rows)
    log_stage(out, 'analyze', time.perf_counter() - t0)
    log(json.dumps({k: {kk: v[kk] for kk in ('measured_D', 'bc_sign', 'slot_reversed', 'position0_explains', 'term', 'D_a', 'D_b')}
                    for k, v in attr.items() if 'new' in k and any(f'{p}-{q}' in k for p, q in FOCUS)}, ensure_ascii=False, indent=1))


def figures(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = {(r['data'], r['slot'], r['format'], r['mask']): r for r in rows}
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    xs = np.arange(len(FORMATS))
    for row, s in enumerate(SLOTS):
        for col, mk in enumerate(MASKS):
            ax = axes[row, col]
            for off, (key, lab, c) in enumerate((('dL', 'measured dL', '#1f77b4'), ('a', 'linear a', '#2ca02c'), ('b', 'curvature b', '#9467bd'),
                                                 ('BC2', 'BC / 2 (implied b)', '#d62728'), ('F', 'first order g.e', '#8c564b'))):
                v = [R[('new', s, f, mk)][key] for f in FORMATS]
                ax.bar(xs + (off - 2) * 0.16, v, 0.16, label=lab, color=c)
            ax.set_yscale('symlog', linthresh=1e-3)
            ax.axhline(0, color='k', lw=.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(FORMATS, fontsize=8)
            ax.set_title(f'L27 Down {s}, positions {mk}', fontsize=9)
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    fig.suptitle('new articles: measured loss change vs its linear / curvature parts and the BC-implied curvature (nat/token)')
    fig.tight_layout()
    fig.savefig(out / 'fig_terms.png', dpi=120)
    plt.close(fig)


def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'calib', 'freeze', 'audit', 'data', 'eval', 'analyze'))
    ap.add_argument('--out-dir', type=Path, required=True)
    a = ap.parse_args()
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
    if a.stage in ('audit', 'data'):
        sampler(out, a.stage)
    else:
        {'plan': plan, 'calib': calib, 'freeze': freeze, 'eval': eval_, 'analyze': analyze}[a.stage](out)
    with open(out / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'seconds': time.time() - t0, 'time': now(), 'script_sha256': sha(HERE / 'run_qwen_l27_reversal_diag.py')}) + '\n')


if __name__ == '__main__':
    main()
