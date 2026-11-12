#!/usr/bin/env python3
"""Calibration-only signed weight term: retrospective audit + frozen fresh-article test.

No production policy changes. Scores are fixed before evaluation; no D/V gradient
or loss is used to fit coefficients. See the generated plan for scale limitations.
"""
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
S1 = HERE / 'runs/olmoe_fixed_route_20261015'
CT = HERE / 'runs/olmoe_crosstext_20261017'
CASE = HERE / 'runs/olmoe_up_case_20261017'
FORMATS = ('int4', 'int6', 'int8', 'p6e1', 'p8e1')
METHODS = ('bc', 'first', 'first_plus_half_bc', 'first_plus_bc')
PRIMARY = 'first_plus_half_bc'
DENOM = 10 * 127


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def candidates():
    return [c for c in read(S1 / 'candidates.json') if c['mode'] == 'w']


def plan(out, log):
    if (out / 'plan.json').exists():
        raise RuntimeError('plan already exists')
    cs = candidates()
    write(out / 'plan.json', {
        'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'question': 'Does a calibration-only signed weight term improve ranking beyond the selected Up case without damaging correct comparisons?',
        'scope': 'All 360 weight-only candidates / 909 same-resource pairs, nine experts, fixed route. Activation criterion and production search unchanged.',
        'scores': {
            'linear': 'F=sum_s <grad_W summed_CE_s, Q(W)-W> / (10*127); sum projections for joint candidates',
            'bc': 'B=original candidate BC/(10*127); positive rescaling leaves BC ranking unchanged',
            'first': 'F', 'first_plus_half_bc': 'F+0.5*B (primary)', 'first_plus_bc': 'F+B (prespecified scale sensitivity)',
            'warning': 'B is a squared-gradient proxy, not a Hessian or calibrated loss estimate. Dividing by calibration tokens only fixes sample aggregation. 0.5 is a Taylor-inspired convention, not a validated physical conversion. No fitted lambda, clipping, or posthoc best-method selection.'},
        'calibration': 'Original 10 calibration windows and cached routes only. Recompute signed gradients with layerwise backward; float64 contraction/accumulation. No D/V or fresh gradients.',
        'checks': 'Reconstruct all 27x5 original weight BC values (max relative error <=1e-3); compare L15.E17.up signed term to earlier independent autograd diagnostic (relative <=1e-3). Abort interpretation if checks fail.',
        'retrospective': 'Existing cross-text D and V are already seen and selected the motivating case; their evaluation is retrospective, not independent confirmation. Report all pairs and the frozen 28 BC-correct / 7 BC-wrong weight pairs.',
        'fresh': '16 unused articles (8 validation,8 test), 4x128-token windows/article, seed 17031. Exclude whole articles overlapping every earlier OLMoE set, including cross-text D/V; sample without using route counts or loss. Freeze scores/config choices before capture/truth.',
        'primary_endpoint': 'For each (expert,M_bytes,BOPs) group with >=2 candidates, choose minimum score (name tie-break). Mean of paired chosen loss differences primary minus BC across all groups, then article-level mean. Also report equal-expert weighting (each expert has same groups). This is single-point decision regret, not a jointly deployed full configuration.',
        'statistics': '10000 article bootstrap replicates stratified by validation/test (8 per stratum), same draws across methods/pairs. Primary CI95 is descriptive fixed-method fresh confirmation; secondary scores not selected by outcomes. Pair sign accuracy on all 909 is descriptive; no independent-binomial CI for shared pairs. No global claims from nine experts.',
        'protected': 'Track 28 previously confirmed BC-correct weight comparisons; report prediction flips and fresh consequences; >=2 flips (more than 5%) fails the conservative non-regression gate.',
        'full_configuration_gate': 'Only proceed to a separately frozen full-configuration pilot if fresh primary decision-difference CI95 upper < 0 and protected flips <=1 and gradient checks pass. Otherwise finish with a negative/inconclusive result and do not alter full-model search. Single-point success would not itself prove full-model benefit.',
        'execution': 'GPU FP32, TF32 off; corrected shared quantizers; reuse verified fixed-route replay; keep activation and routes fixed. No teacher or evaluation loss used for scoring.',
        'inputs': {str(p.relative_to(HERE)): sha(p) for p in [S1/'candidates.json', S1/'table.json', S1/'tokens_calibration.npy', CT/'segments.json', CT/'analysis.json', CASE/'analysis.json']},
        'script_sha256': sha(__file__)
    })
    write(out / 'candidates.json', cs)
    log('plan frozen: 360 weight candidates; calibration-only scores; fresh confirmation required')


def score(out, log):
    import torch
    import torch.nn.functional as F
    import olmoe_common as oc
    from run_olmoe_fixed_route import gpu_exact, load_routes
    from run_olmoe_bc_numerics import quantizers
    if (out / 'scores.json').exists():
        raise RuntimeError('scores already exist')
    gpu_exact('cuda')
    torch.set_num_threads(8)
    dev = 'cuda'
    cfg, ck = oc.config(), oc.Checkpoint()
    cs = read(out/'candidates.json')
    want = {(c['layer'], c['expert']) for c in cs}
    rt, _, _ = load_routes(S1, 'calibration', cfg, dev)
    toks = torch.from_numpy(np.load(S1/'tokens_calibration.npy')).to(dev)
    assert toks.shape == (10,128)
    co, si = oc.rope(cfg,128,dev,torch.float32)
    g = ck.global_(dev,torch.float32)
    t0=time.time()
    with torch.no_grad():
        hs=[F.embedding(toks,g['embed'])]
        for l in range(cfg.num_hidden_layers):
            hs.append(oc.layer_fixed(ck.layer(l,dev,torch.float32),hs[-1],cfg,co,si,*rt[l],l)[0])
    ht=hs[-1].detach().requires_grad_(True)
    ce=oc.summed_ce(oc.head(g,ht,cfg),toks)
    gn,=torch.autograd.grad(ce.sum(),ht)
    (out/'work').mkdir(exist_ok=True)
    (out/'gradients').mkdir(exist_ok=True)
    q,_=quantizers(out/'work',dev,'corrected')
    original=read(S1/'table.json')['B']
    result={}
    errs=[]
    for l in reversed(range(cfg.num_hidden_layers)):
        w=ck.layer(l,dev,torch.float32)
        cap={}
        hk=oc.ExpertHooks()
        def capture(ll,e,p,s,t):
            if (ll,e) in want:
                t.retain_grad()
                cap[(e,p,s)]=t
        hk.capture=capture
        hi=hs[l].detach().requires_grad_(True)
        o=oc.layer_fixed(w,hi,cfg,co,si,*rt[l],l,hk)[0]
        o.backward(gn)
        gn=hi.grad.detach()
        for e in sorted(e for ll,e in want if ll==l):
            _,tok=torch.where(F.one_hot(rt[l][0],num_classes=cfg.num_experts+1).permute(2,1,0)[e])
            win=tok//128
            for p in oc.PROJ:
                node=f'L{l}.E{e}.{p}'
                x=cap[(e,'gate_proj','x')] if p!='down_proj' else cap[(e,p,'x')]
                gy=cap[(e,p,'y')].grad
                W=w[p][e].detach().cpu()
                gs=torch.zeros(W.shape,device=dev,dtype=torch.float64)
                g2=torch.zeros(W.shape[0],device=dev,dtype=torch.float64)
                per=[]
                deltas={f:(q.weight(W,f)-W).to(device=dev,dtype=torch.float64) for f in FORMATS}
                for s in range(10):
                    m=win==s
                    grad=gy[m].double().T @ x[m].detach().double()
                    gs+=grad
                    g2+=grad.square().sum(1)
                    per.append({f:float((grad*d).sum())/127 for f,d in deltas.items()})
                vals={}
                for f,d in deltas.items():
                    rec=float((g2/W.shape[1]*d.square().sum(1)).sum())
                    orig=original[node][f]
                    err=abs(rec-orig)/max(abs(orig),1e-20)
                    errs.append({'node':node,'format':f,'relative':err,'reconstructed':rec,'original':orig})
                    vals[f]={'first':float((gs*d).sum())/DENOM,'bc':orig/DENOM,'bc_reconstructed':rec/DENOM}
                torch.save(gs.cpu(),out/'gradients'/f'{node}.pt')
                result[node]={'formats':vals,'first_per_window':per}
                del deltas,gs,g2,grad
        del cap,hk,o,hi,w
        log(f'calibration backward layer {l} done ({time.time()-t0:.1f}s)')
    checks={'bc_max_relative':max(x['relative'] for x in errs),'bc_comparisons':errs}
    case=read(CASE/'analysis.json')['C']['first_order']
    checks['up_case_relative']={f:abs(result['L15.E17.up_proj']['formats'][f]['first']-case[f]['mean'])/max(abs(case[f]['mean']),1e-20) for f in ('int8','p8e1')}
    checks['pass']=checks['bc_max_relative']<1e-3 and max(checks['up_case_relative'].values())<1e-3
    write(out/'gradient_checks.json',checks)
    write(out/'projection_scores.json',result)
    if not checks['pass']:
        raise RuntimeError('gradient check failed; see gradient_checks.json')
    rows=[]
    for c in cs:
        f=sum(result[c['node']+'.'+p]['formats'][wf]['first'] for p,(wf,af) in c['formats'].items())
        b=c['bc']/DENOM
        rows.append({'name':c['name'],'bc':b,'first':f,'first_plus_half_bc':f+0.5*b,'first_plus_bc':f+b})
    write(out/'scores.json',rows)
    write(out/'score_freeze.json',{'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'scores_sha256':sha(out/'scores.json'),'plan_sha256':sha(out/'plan.json'),'seconds':time.time()-t0,'torch':torch.__version__,'peak_gpu_gb':torch.cuda.max_memory_allocated()/2**30,'source':'calibration only'})
    log(f'scores frozen; checks pass, max BC relative={checks["bc_max_relative"]:.3g}')


def fresh(out,log):
    import olmoe_common as oc
    from run_olmoe_crosstext import used_spans, articles
    if not (out/'score_freeze.json').exists():
        raise RuntimeError('freeze scores first')
    if (out/'segments.json').exists():
        raise RuntimeError('fresh data already frozen')
    used=used_spans()
    old=read(CT/'segments.json')
    titles={s['title'] for s in old}
    for s in old:
        used[s['split']].append(tuple(s['article_chars']))
    tok=oc.tokenizer()
    rng=np.random.default_rng(17031)
    segs=[]; chunks=[]; avail={}
    for split in ('validation','test'):
        arts,ids,offsets=articles(split,tok)
        free=[a for a in arts if a['title'] not in titles and a['tokens'][1]-a['tokens'][0]>=512 and not any(a['chars'][0]<b and a['chars'][1]>a0 for a0,b in used[split])]
        avail[split]=len(free)
        if len(free)<8:
            raise RuntimeError('insufficient unused articles')
        for k in sorted(rng.choice(len(free),8,replace=False).tolist()):
            a=free[k]; start=int(rng.integers(a['tokens'][0],a['tokens'][1]-512+1))
            chunks.append(np.array(ids[start:start+512],dtype=np.int64).reshape(4,128))
            segs.append({'split':split,'title':a['title'],'article_chars':a['chars'],'window_chars':[int(offsets[start][0]),int(offsets[start+511][1])],'tokens':[start,start+512],'index':len(segs)})
            titles.add(a['title'])
    d=out/'H';d.mkdir(exist_ok=True)
    np.save(d/'tokens.npy',np.concatenate(chunks))
    write(out/'segments.json',segs)
    write(out/'fresh_freeze.json',{'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'scores_sha256':sha(out/'scores.json'),'tokens_sha256':sha(d/'tokens.npy'),'available':avail,'segments':16,'overlap':'whole prior articles excluded; unique titles'})
    log(f'16 fresh articles frozen; available={avail}')


def truth(out,log):
    from run_olmoe_crosstext import stage_capture,load_set_routes,replay
    from run_olmoe_fixed_route import gpu_exact
    gpu_exact('cuda')
    if sha(out/'scores.json')!=read(out/'fresh_freeze.json')['scores_sha256']:
        raise RuntimeError('score hash mismatch')
    if (out/'H/truth.json').exists():
        raise RuntimeError('truth exists')
    stage_capture(SimpleNamespace(set='H',device='cuda'),out,log)
    r=replay(out,np.load(out/'H/tokens.npy'),load_set_routes(out,'H','cuda'),read(out/'candidates.json'),'cuda',log,'H weight-only',work='truth_work')
    r['script_sha256']=sha(__file__)
    write(out/'H/truth.json',r)


def bootstrap_stats(x,idx):
    x=np.asarray(x)
    b=x[idx].mean(1)
    return {'mean':float(x.mean()),'ci95':np.percentile(b,[2.5,97.5]).tolist()}


def analyze(out,log):
    from scipy.stats import spearmanr
    from run_olmoe_bc_numerics import pair_list
    cs=read(out/'candidates.json'); nms=[c['name'] for c in cs]
    scr=read(out/'scores.json')
    assert nms==[s['name'] for s in scr]
    vals={m:np.array([s[m] for s in scr]) for m in METHODS}
    pairs=pair_list(cs); ii=np.array([p['i'] for p in pairs]);jj=np.array([p['j'] for p in pairs])
    dirs={m:np.sign(v[ii]-v[jj]) for m,v in vals.items()}
    byresource=defaultdict(list)
    for i,c in enumerate(cs):
        byresource[(c['node'],c['M_bytes'],c['BOPs_calibration'])].append(i)
    groups=[v for v in byresource.values() if len(v)>1]
    choices={m:np.array([min(g,key=lambda i:(vals[m][i],nms[i])) for g in groups]) for m in METHODS}
    result={'candidates':len(cs),'pairs':len(pairs),'resource_groups':len(groups),'primary':PRIMARY,'scores_frozen':read(out/'score_freeze.json'),'sets':{},'choice_changes':{m:int((v!=choices['bc']).sum()) for m,v in choices.items()}}
    # Previously independently confirmed pairs are retrospective diagnostics now.
    oldrows=list(csv.DictReader(open(CT/'pairs.csv')))
    confirmed={}
    # Obtain exact former selection list and confirmed names from analysis and CSV.
    oldanalysis=read(CT/'analysis.json')
    wrong={tuple(x[:2]) for x in oldanalysis['reproducible_bc_errors_list']}
    # CSV schema checked directly rather than relying on ordering of 1080 candidates.
    for r in oldrows:
        if r.get('V_category')=='confirmed_same':
            a,b=r.get('a'),r.get('b')
            if a in nms and b in nms:
                confirmed[(a,b)]='wrong' if (a,b) in wrong else 'right'
    if sum(v=='right' for v in confirmed.values())!=28 or sum(v=='wrong' for v in confirmed.values())!=7:
        raise RuntimeError('historical confirmation schema/count mismatch: '+str(confirmed))
    lookup={(nms[p['i']],nms[p['j']]):k for k,p in enumerate(pairs)}
    protected=np.array([lookup[p] for p,t in confirmed.items() if t=='right'])
    erroneous=np.array([lookup[p] for p,t in confirmed.items() if t=='wrong'])
    result['historical_prediction_changes']={m:{'correct_28_flipped':int((dirs[m][protected]!=dirs['bc'][protected]).sum()),'wrong_7_corrected_direction':int((dirs[m][erroneous]==-dirs['bc'][erroneous]).sum())} for m in METHODS}
    pair_rows=[]
    for label,folder in [('D',CT/'D'),('V',CT/'V'),('H',out/'H')]:
        if not (folder/'truth.json').exists():
            continue
        t=read(folder/'truth.json')
        s=np.array([t['rows'][nm] for nm in nms])/127
        seg=s.reshape(len(cs),16,4).mean(2)
        # D/V use source stratum membership, H first 8 validation then 8 test.
        meta=read(CT/'segments.json') if label!='H' else read(out/'segments.json')
        meta=sorted([x for x in meta if label=='H' or x['group']==label],key=lambda x:x['index'] if label=='H' else x['index_in_group'])
        rng=np.random.default_rng({'D':17041,'V':17042,'H':17043}[label])
        idx=np.concatenate([rng.choice([i for i,x in enumerate(meta) if x['split']==sp],(10000,sum(x['split']==sp for x in meta)),replace=True) for sp in ('validation','test')],1)
        diff=seg[ii]-seg[jj]; mean=diff.mean(1)
        # shared article bootstrap; statistic averages shared candidate pairs, not independent trials
        draws=np.stack([diff[:,row].mean(1) for row in idx],axis=1)
        ci=np.percentile(draws,[2.5,97.5],axis=1)
        resolved=(ci[0]>0)|(ci[1]<0)  # descriptive unadjusted diagnostic only
        methods={}
        for m in METHODS:
            acc=(dirs[m]==np.sign(mean)).mean()
            accdraw=(dirs[m][:,None]==np.sign(draws)).mean(0)
            delta=(seg[choices[m]]-seg[choices['bc']]).mean(0)
            corr=[]
            for node in sorted({c['node'] for c in cs}):
                mask=np.array([c['node']==node for c in cs])
                corr.append(float(spearmanr(vals[m][mask],seg[mask].mean(1)).statistic))
            methods[m]={'all_pair_sign_agreement':float(acc),'agreement_bootstrap_ci95':np.percentile(accdraw,[2.5,97.5]).tolist(),'unadjusted_resolved_agreement':float((dirs[m][resolved]==np.sign(mean[resolved])).mean()) if resolved.any() else None,'spearman_per_expert':dict(zip(sorted({c['node'] for c in cs}),corr)),'choice_loss_minus_bc':bootstrap_stats(delta,idx),'choice_mean_damage':float(seg[choices[m]].mean()),'historical_correct_fresh_sign_agreement':float((dirs[m][protected]==np.sign(mean[protected])).mean()),'historical_wrong_fresh_sign_agreement':float((dirs[m][erroneous]==np.sign(mean[erroneous])).mean())}
            if m!='bc':
                ad=((dirs[m][:,None]==np.sign(draws)).mean(0)-(dirs['bc'][:,None]==np.sign(draws)).mean(0))
                methods[m]['agreement_improvement_ci95']=np.percentile(ad,[2.5,97.5]).tolist()
            methods[m]['choice_loss_minus_bc_by_expert']={
                node:bootstrap_stats((seg[choices[m][mask]]-seg[choices['bc'][mask]]).mean(0),idx)
                for node in sorted({c['node'] for c in cs})
                for mask in [np.array([cs[g[0]]['node']==node for g in groups])]
            }
        result['sets'][label]={'fresh':label=='H','articles':16,'unadjusted_resolved_pairs':int(resolved.sum()),'methods':methods}
        for k,p in enumerate(pairs):
            pair_rows.append({'set':label,'a':nms[p['i']],'b':nms[p['j']],'node':p['node'],'mean_loss_a_minus_b':mean[k],'ci_low':ci[0,k],'ci_high':ci[1,k],'historical':confirmed.get((nms[p['i']],nms[p['j']]),''),**{m+'_sign':dirs[m][k] for m in METHODS}})
    if 'H' in result['sets']:
        upper=result['sets']['H']['methods'][PRIMARY]['choice_loss_minus_bc']['ci95'][1]
        flips=result['historical_prediction_changes'][PRIMARY]['correct_28_flipped']
        result['full_configuration_gate']={'fresh_choice_ci_upper':upper,'protected_flips':flips,'pass':bool(upper<0 and flips<=1),'decision':'eligible for separately frozen full-configuration pilot' if upper<0 and flips<=1 else 'do not promote or change full-model search; single-point gate not passed'}
    write(out/'analysis.json',result)
    with open(out/'pairs.csv','w') as f:
        wr=csv.DictWriter(f,fieldnames=list(pair_rows[0]));wr.writeheader();wr.writerows(pair_rows)
    write(out/'choices.json',{m:[{'node':cs[i]['node'],'name':nms[i],'M_bytes':cs[i]['M_bytes'],'BOPs_calibration':cs[i]['BOPs_calibration']} for i in vv] for m,vv in choices.items()})
    log(json.dumps({'changes':result['historical_prediction_changes'],'sets':{s:{m:v['choice_loss_minus_bc'] for m,v in x['methods'].items()} for s,x in result['sets'].items()},'gate':result.get('full_configuration_gate')},ensure_ascii=False))


def report(out,log):
    a=read(out/'analysis.json')
    if 'H' not in a['sets']:
        raise RuntimeError('fresh evaluation required for final report')
    if (out/'report.md').exists():
        raise RuntimeError('report.md exists (the 20261017 report is hand-written from analysis.json + supplement.json)')
    labels={'bc':'原 BC','first':'純一階 F','first_plus_half_bc':'F + 0.5 B（主要方案）','first_plus_bc':'F + B（尺度敏感性）'}
    checks=read(out/'gradient_checks.json')
    gate=a['full_configuration_gate']
    fmt=lambda v:f"{v['mean']:+.7f} [{v['ci95'][0]:+.7f}, {v['ci95'][1]:+.7f}]"
    lines=['# OLMoE fixed-route：calibration 一階項的跨候選驗證','',
           '日期：2026-10-17。腳本：`run_olmoe_first_order_validation.py`。未修改正式 BC、搜尋或共用量化器。','',
           '## 結果與決策','',
           f"預先固定的完整配置測試門檻：**{'通過' if gate['pass'] else '未通過'}**。新文章的主要選擇損傷差 CI 上界為 {gate['fresh_choice_ci_upper']:+.7f}；原 28 對確認正確配對中，主要方案翻轉 {gate['protected_flips']} 對。門檻要求上界 < 0 且翻轉最多 1 對。這是保守的研究推進規則，不是 criterion 普遍無效的統計證明。",'',
           '本輪完成單點候選的校準建表、回溯分析與新文章確認。若門檻未通過，不推進完整配置搜尋，不改預設 criterion。完整配置效果仍未驗證。','',
           '## 設計與尺度','',
           '- 原 9 個 expert、27 個投影、全部 360 個 weight-only 候選、909 對同資源比較；沒有 activation 一階項。',
           '- 僅使用原 10 個 calibration 窗口與固定 FP32 路由重算 signed gradient。沒有使用 D、V、H 的梯度或 loss 來建立分數、挑係數或配置。',
           '- F = Σ_s〈g_s, Q(W)−W〉/(10×127)。B = 原 BC/(10×127)。聯合候選加總各投影分數，不截斷負值。',
           '- 主要方案固定 F+0.5B；另列 F、F+B。不依結果選擇係數。B 仍是平方梯度 proxy，正規化只統一樣本加總，不能證明 B 就是真正二階損傷。0.5 只是 Taylor-inspired 待測約定。',
           '- 舊 D/V 已參與案例選取，因此本輪對它們僅為回溯分析；H 為 16 篇新文章（validation/test 各 8）、每篇 4×128 token，整篇排除先前使用文章。',
           '- 分數先凍結，再抽 H、擷取 H 路由與計算 truth。主評估 GPU FP32，TF32 關閉，fixed route，非 native 全模型。',
           '- 63 個同資源群組各自取分數最低候選。主要指標是群組平均的「新方法選擇 loss − BC 選擇 loss」，不是把 63 個選擇聯合量化的 loss。',
           '- 以文章為單位、按 validation/test 分層做 10000 次 paired bootstrap；配對不當成獨立 Bernoulli 樣本。次要指標不作多方法最佳者選擇。','',
           '## 原有案例的預測變化（回溯）','',
           '| 方法 | 原 7 對錯誤改成原確認方向 | 原 28 對正確被翻轉 | 63 群組中改變選擇 |',
           '|---|---:|---:|---:|']
    for m in METHODS:
        x=a['historical_prediction_changes'][m]
        lines.append(f"| {labels[m]} | {x['wrong_7_corrected_direction']} | {x['correct_28_flipped']} | {a['choice_changes'][m]} |")
    lines+=['','原 7 對錯誤有共用候選，不能視為 7 個獨立機制。翻轉原確認方向也不代表新文本必然受害，須同時看 H 的結果。','',
            '## 同資源候選選擇的實際損傷差','',
            '負值代表相對 BC 更好；單位 nat/token。區間為 95% 文章 bootstrap。','',
            '| 方法 | D（回溯） | V（回溯） | H（新文章） |','|---|---|---|---|']
    for m in METHODS:
        lines.append('| '+labels[m]+' | '+' | '.join(fmt(a['sets'][s]['methods'][m]['choice_loss_minus_bc']) for s in ('D','V','H'))+' |')
    lines+=['','## 全部配對的方向一致率','',
            '包含大量實測差異接近零的配對，僅作描述；不是確定性配對的準確率。','',
            '| 方法 | D | V | H |','|---|---:|---:|---:|']
    for m in METHODS:
        lines.append('| '+labels[m]+' | '+' | '.join(f"{100*a['sets'][s]['methods'][m]['all_pair_sign_agreement']:.1f}%" for s in ('D','V','H'))+' |')
    lines+=['','## H 上各 expert 的主要選擇差','', '| Expert | F+0.5B − BC，平均 [CI95] |','|---|---|']
    for n,v in a['sets']['H']['methods'][PRIMARY]['choice_loss_minus_bc_by_expert'].items():
        lines.append(f'| {n} | {fmt(v)} |')
    lines+=['','各 expert 區間屬未校正的診斷資訊，不能逐項挑顯著結果作總結。','',
            '## 檢查、成本與限制','',
            f"- 全部 135 個原 BC 權重分數重建：最大相對差 {checks['bc_max_relative']:.3g}。",
            f"- L15.E17 Up 的 calibration 一階項對獨立 autograd 診斷：相對差 {checks['up_case_relative']}。",
            '- 360 候選分數加總、weight-only 語意、63 個群組的精確 M/B 相等、新文章與舊 D/V 不重疊：見 verification.json。',
            f"- calibration 建分數 {a['scores_frozen']['seconds']:.1f} 秒，含一次 forward/backward、27 投影誤差及 contraction；不是全模型 3051 投影的成本估計。",
            f"- 新文章 truth {read(out/'H/truth.json')['seconds']:.1f} 秒，不含 route capture；各階段時間見 timing.jsonl。",
            '- 只有一個模型、9 個 expert、16 篇新文章。區間跨零是無法確認改善，不是已證明完全相同。',
            '- 一階項在本例有效，不代表可直接加到所有 activation 或完整配置。跨層交互作用、總和效果與原生推論尚未驗證。',
            '- 本輪採固定尺度的少量方案，結果不能排除所有可能的一階校準方式；也不允許事後調係數後仍稱獨立確認。','',
            '## 重跑','', '```bash',
            'cd experiments/mixed_precision',
            'O=runs/olmoe_first_order_validation_20261017  # 新實驗請改用新目錄',
            '../../venv/bin/python run_olmoe_first_order_validation.py --stage plan --out-dir "$O"',
            'env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python run_olmoe_first_order_validation.py --stage score --out-dir "$O"',
            '../../venv/bin/python run_olmoe_first_order_validation.py --stage fresh --out-dir "$O"',
            'env OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE PYTHONPATH=qwen35_assets/gpu_deps /home/hoju/test/HTQ/.venv/bin/python run_olmoe_first_order_validation.py --stage truth --out-dir "$O"',
            '../../venv/bin/python run_olmoe_first_order_validation.py --stage analyze --out-dir "$O"',
            '../../venv/bin/python run_olmoe_first_order_validation.py --stage report --out-dir "$O"',
            '```','',
            '## 產物','',
            '`plan.json`、`projection_scores.json`、`gradients/*.pt`、`scores.json`、`score_freeze.json`、`fresh_freeze.json`、`segments.json`、`H/truth.json`、`analysis.json`、`pairs.csv`、`choices.json`、`gradient_checks.json`、`verification.json`、`timing.jsonl`。','']
    (out/'report.md').write_text('\n'.join(lines))
    log('report.md written')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['plan','score','fresh','truth','analyze','report'],required=True)
    ap.add_argument('--out-dir',type=Path,required=True)
    a=ap.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True)
    def log(msg):
        print(time.strftime('%H:%M:%S'),msg,flush=True)
    t=time.time()
    if a.stage!='plan' and not (a.out_dir/'plan.json').exists():
        raise RuntimeError('plan required')
    globals()[a.stage](a.out_dir,log)
    with open(a.out_dir/'timing.jsonl','a') as f:
        f.write(json.dumps({'stage':a.stage,'seconds':time.time()-t,'time':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'script_sha256':sha(__file__)})+'\n')


if __name__=='__main__':
    main()
