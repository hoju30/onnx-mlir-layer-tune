#!/usr/bin/env python3
"""GPT-2 small pilot: shared, relative W+O, relative W+I+O; ORT QDQ evaluation.

Only the 48 constant-weight transformer projection nodes are searched. Embeddings,
the tied LM head, LayerNorm, and activation-only attention MatMuls stay FP32.
Budget costs refer only to searched projections. PPL uses independent fixed windows.
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'gpt2_assets'
sys.path.insert(0, str(ASSETS / 'python_deps'))

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import time
import numpy as np
import onnx
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from check_shared_fix import tap_samples
from eval_random_mixed_precision import (calibrate_ranges, supported_nodes, simulate_in_onnxruntime)
from htq_perturbation import weight_perturbation, activation_perturbation
from mp_milp import milp_select
from sample_configs_sqnr import config_cost

FORMATS = ['int4', 'int5', 'int6', 'int7', 'int8', 'f16', 'bf16', 'fp32']


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda v: v.item())+'\n')


def write_report(out, results, settings):
    text = ['# GPT-2 small：混合精度 criterion 初步測試', '',
        '**評測後端：ONNX Runtime QDQ 模擬；尚未宣稱 ONNX-MLIR 編譯後驗證。**', '',
        '| 設定 | 內容 |','|---|---|',
        '| 模型 | openai-community/gpt2，預訓練 small 模型 |',
        '| 資料 | WikiText-2 raw，validation 校準／test 評測 |',
        f"| 校準／評測 | {settings['calibration_windows']}／{settings['eval_windows']} 個獨立視窗 |",
        f"| 視窗長度／預測 token 數 | {settings['sequence_length']}／{settings['eval_windows']*(settings['sequence_length']-1)} |",
        '| 格式 | int4–8、f16、bf16、fp32 |',
        '| 搜尋 | MILP；只包含 48 個 transformer 投影節點 |',
        '| 固定 FP32 | Embedding、共享權重的 LM head、LayerNorm、attention 動態 MatMul 與其他節點 |',
        '| f 的基準 | 48 個投影層全 W8A8 的理論 weight bytes 與 BOPs；不是全模型成本 |',
        '| Sensitivity | 相同預訓練權重，以 causal next-token CE 計算 mean(abs(W*dL/dW))，再平均校準視窗 |', '',
        'shared 使用 raw weight/output L2；W＋O 使用相對平方 weight/output 誤差；W＋I＋O 再加入相對平方 input 誤差。三者均使用相同 sensitivity。實際量化在各選定節點同時作用於 weight、input 和 output。', '',
        '| 方法 | f | NLL ↓ | Perplexity ↓ | Next-token top-1 ↑ |', '|---|---:|---:|---:|---:|']
    for r in results['rows']:
        ppl=f"{r['perplexity']:.2f}" if r['perplexity'] is not None else 'non-finite'
        nll=f"{r['nll']:.4f}" if r['nll'] is not None else 'non-finite'
        text.append(f"| {r['criterion']} | {r.get('budget', '—')} | {nll} | {ppl} | {100*r['top1']:.2f}% |")
    text += ['', '本結果使用固定短視窗的 test 子集，視窗之間不傳遞 KV cache；不能與完整 WikiText-2 或長滑動視窗的公開 PPL 直接比較。這是同設定下的 criterion 對照。', '',
        '[原始結果](results.json) · [設定與資料來源](experiment.json) · [配置](configs.json)']
    smoke = out/'compiler_smoke/status.json'
    if smoke.exists():
        state = json.loads(smoke.read_text())
        text += ['', 'ONNX-MLIR 相容性檢查：' + state['stage'] +
            '。已驗證匯入與 shared_v2、f=0.80 的量化 pass；這項檢查不代表已完成原生執行。',
            '[編譯相容性檢查紀錄](compiler_smoke/status.json)']
    if results['status'] == 'complete':
        winners=[]
        for f in settings['budgets']:
            group=[r for r in results['rows'] if r.get('budget')==f and r['perplexity'] is not None]
            if group:
                best=min(group,key=lambda r:r['perplexity'])
                winners.append(f"f={f:.2f}：{best['criterion']}")
        text += ['', '本輪各預算最低 PPL：' + '；'.join(winners) + '。這是小型子集結果，不能據此宣稱某個 criterion 普遍較好。',
            '即使投影層全部 W8A8，PPL 也高於 FP32。後續若要找出退化原因，應固定配置拆分 weight-only、activation-only 與 W+A，並增加校準文字；目前結果本身無法判定主要成因。',
            '這裡的 shared 是目前 ONNX 流程的 raw weight/output L2 criterion；不是官方 HTQ 以權重誤差代理 activation 的完整重現。',
            '', '重跑方式（輸出目錄須尚未存在）：', '```bash',
            'venv/bin/python3 experiments/mixed_precision/run_gpt2_criteria.py --out-dir experiments/mixed_precision/runs/gpt2_new_run', '```']
    (out/'report.md').write_text('\n'.join(text)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--sequence-length', type=int, default=128)
    ap.add_argument('--calibration-windows', type=int, default=10)
    ap.add_argument('--eval-windows', type=int, default=32)
    ap.add_argument('--budgets', default='0.6,0.65,0.7,0.75,0.8')
    args=ap.parse_args(); out=args.out_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4); torch.manual_seed(0)
    t0=time.time()
    def log(s):print(f'[{time.time()-t0:.1f}s] {s}',flush=True)
    tokenizer=GPT2TokenizerFast.from_pretrained(ASSETS/'model',local_files_only=True)
    model=GPT2LMHeadModel.from_pretrained(ASSETS/'model',local_files_only=True,attn_implementation='eager').eval()
    model.config.use_cache=False
    seq=args.sequence_length
    tokens={}
    for split,count in [('validation',args.calibration_windows),('test',args.eval_windows)]:
        text='\n\n'.join(pq.read_table(ASSETS/'data'/f'{split}-00000-of-00001.parquet')['text'].to_pylist())
        ids=tokenizer.encode(text,add_special_tokens=False)
        assert len(ids)>=count*seq
        tokens[split]=np.asarray(ids[:count*seq],dtype=np.int64).reshape(count,seq)
        np.save(out/f'{split}_tokens.npy',tokens[split])
    cx,ex=tokens['validation'],tokens['test']
    projection_names=[name for name,_ in model.named_parameters() if name.startswith('transformer.h.') and name.endswith(
        ('attn.c_attn.weight','attn.c_proj.weight','mlp.c_fc.weight','mlp.c_proj.weight'))]
    assert len(projection_names)==48, projection_names
    params=dict(model.named_parameters())
    class Export(torch.nn.Module):
        def __init__(self,model):super().__init__(); self.model=model
        def forward(self,input_ids):
            mask=torch.zeros((1,1,seq,seq),device=input_ids.device,dtype=torch.float32)
            return self.model(input_ids,attention_mask=mask,use_cache=False,return_dict=False)[0]
    wrapper=Export(model).eval()
    example=torch.from_numpy(cx[:1])
    with torch.no_grad():
        ref=model(example,use_cache=False).logits.numpy()
        wrapped=wrapper(example).numpy()
    np.testing.assert_allclose(wrapped,ref,rtol=1e-4,atol=1e-3)
    log('Model, tokenizer and data ready; exporting fixed-shape ONNX')
    torch.onnx.export(wrapper,(example,),str(out/'model.onnx'),input_names=['input_ids'],output_names=['logits'],
        opset_version=14,dynamo=False,do_constant_folding=True)
    graph=onnx.load(str(out/'model.onnx')); onnx.checker.check_model(graph)
    opts=ort.SessionOptions();opts.intra_op_num_threads=4
    sess=ort.InferenceSession(str(out/'model.onnx'),opts,providers=['CPUExecutionProvider'])
    ort_ref=sess.run(None,{'input_ids':cx[:1]})[0]
    np.testing.assert_allclose(ort_ref,ref,rtol=1e-3,atol=2e-3)
    equivalence={'max_abs_logit_diff':float(np.abs(ort_ref-ref).max()),
                 'same_top1_tokens':int((ort_ref.argmax(-1)==ref.argmax(-1)).sum()), 'tokens':seq}
    del sess,ref,ort_ref,wrapped
    all_nodes=supported_nodes(graph); by_name={n.name:n for n in graph.graph.node}
    inits={i.name:onnx.numpy_helper.to_array(i) for i in graph.graph.initializer}
    mapping={}
    for n in all_nodes:
        if not n.get('weight_const'):continue
        init=by_name[n['name']].input[1]
        param=init.removeprefix('model.')
        if param in projection_names:
            np.testing.assert_array_equal(inits[init],params[param].detach().numpy())
            mapping[n['name']]=param
    assert len(mapping)==48, f'Projection mapping failed: {len(mapping)}; initializers={list(inits)[:20]}'
    nodes=[n for n in all_nodes if n['name'] in mapping]
    save(out/'node_mapping.json',mapping)
    for name,p in model.named_parameters():p.requires_grad_(name in projection_names)
    sensitivities=dict.fromkeys(projection_names,0.)
    for k,ids in enumerate(cx):
        model.zero_grad(set_to_none=True)
        loss=model(torch.from_numpy(ids[None]),labels=torch.from_numpy(ids[None]),use_cache=False).loss
        loss.backward()
        for name in projection_names:
            p=params[name];sensitivities[name]+=float((p*p.grad).abs().mean())/len(cx)
        log(f'SNIP calibration {k+1}/{len(cx)}: causal CE={float(loss.detach()):.4f}')
    S={n:sensitivities[p] for n,p in mapping.items()};save(out/'sensitivity.json',S)
    del model,wrapper,params,example,loss;gc.collect()
    ranges=calibrate_ranges(graph,nodes,cx);save(out/'calibration.json',ranges)
    samples=tap_samples(graph,sorted({t for n in nodes for t in n['act_tensors']}),cx)
    tables={c:{} for c in ['shared','relative_wo','shared_v2']}
    elems={};flops={}
    for k,n in enumerate(nodes):
        name=n['name'];x,y=n['act_tensors'][0],n['act_tensors'][-1]
        W=inits[by_name[name].input[1]].astype(np.float64)
        wn,xn,yn=np.linalg.norm(W),np.linalg.norm(samples[x]),np.linalg.norm(samples[y])
        assert wn>0 and xn>0 and yn>0
        we={f:weight_perturbation(W,f,None) for f in FORMATS}
        xe={f:activation_perturbation(samples[x],f,ranges[x],None) for f in FORMATS}
        ye={f:activation_perturbation(samples[y],f,ranges[y],None) for f in FORMATS}
        rw={f:(we[f]/wn)**2 for f in FORMATS};ry={f:(ye[f]/yn)**2 for f in FORMATS}
        tables['shared'][name]={'w':we,'a':ye}
        tables['relative_wo'][name]={'w':rw,'a':ry}
        tables['shared_v2'][name]={'w':rw,'a':{f:(xe[f]/xn)**2+ry[f] for f in FORMATS}}
        elems[name]=W.size;flops[name]=W.size*seq
        if k%12==11:log(f'Perturbation tables: {k+1}/48 projection nodes')
    save(out/'perturbation_tables.json',tables)
    del samples,inits,W;gc.collect()
    settings={'model':'openai-community/gpt2','sequence_length':seq,'calibration_windows':len(cx),
        'eval_windows':len(ex),'eval_predicted_tokens':len(ex)*(seq-1),'formats':FORMATS,
        'budgets':[float(f) for f in args.budgets.split(',')],'fp32_equivalence':equivalence,
        'model_sha256':hashlib.sha256((out/'model.onnx').read_bytes()).hexdigest(),
        'assets':json.loads((ASSETS/'provenance.json').read_text()),'quantized_projection_nodes':48,
        'calibration_split':'validation','evaluation_split':'test','window_scheme':'independent, nonoverlapping, no padding',
        'frozen':'embeddings/tied lm_head/LayerNorm/activation-only attention MatMuls and other nodes',
        'sensitivity_backend':'PyTorch causal next-token CE; ONNX initializer equality and FP32 logits verified'}
    save(out/'experiment.json',settings)
    shutil.copy2(__file__,out/'run_gpt2_criteria.py')
    configs=[];ref_size=sum(elems.values());ref_bops=sum(64*v for v in flops.values())
    for i,budget in enumerate(settings['budgets']):
        for criterion,table in tables.items():
            def value(name,w,a):return S[name]*(table[name]['w'][w]+table[name]['a'][a])
            cfg,obj,status=milp_select(nodes,FORMATS,FORMATS,value,elems,flops,budget*ref_size,budget*ref_bops)
            assert status==0,(criterion,budget,status)
            for n in nodes:cfg[n['name']]['op']=n['op']
            size,bops=config_cost(cfg,elems,flops)
            configs.append({'id':f'{criterion}_{i:03d}','criterion':criterion,'budget':budget,'formats':cfg,
                'objective':obj,'solver_status':status,'size_vs_int8':size/ref_size,'bops_vs_int8':bops/ref_bops})
            log(f'MILP optimal: {criterion} f={budget:.2f}')
    save(out/'configs.json',{'configs':configs})
    results={'backend':'ORT QDQ simulation','status':'running','rows':[]}
    refs=[{'id':'fp32','criterion':'FP32','formats':{}},
          {'id':'w8a8','criterion':'W8A8 (projections)','formats':{n['name']:{'w':'int8','a':'int8'} for n in nodes}}]
    for c in refs+configs:
        z=simulate_in_onnxruntime(graph,nodes,c['formats'],ranges,ex,None)
        assert z.shape==(len(ex),seq,model_vocab:=50257),z.shape
        pred=z[:,:-1,:];target=ex[:,1:]
        finite=bool(np.isfinite(pred).all())
        correct=int((pred.argmax(-1)==target).sum()) if finite else 0
        if finite:
            # Float64 reduction per window; never average per-window perplexities.
            nlls=[]
            for scores,labels in zip(pred,target):
                a=scores.astype(np.float64)
                nlls.append(logsumexp(a,axis=-1)-a[np.arange(seq-1),labels])
            nll=float(np.concatenate(nlls).mean());ppl=math.exp(nll) if nll<700 else None
        else:nll,ppl=None,None
        r={'id':c['id'],'criterion':c['criterion'],'nll':nll,'perplexity':ppl,'top1':correct/target.size,
           'predicted_tokens':int(target.size),'finite_logits':finite}
        if 'budget' in c:r.update({k:c[k] for k in ['budget','size_vs_int8','bops_vs_int8']})
        results['rows'].append(r);save(out/'results.json',results);write_report(out,results,settings)
        log(f"{c['id']}: PPL={ppl}, next-token top1={r['top1']:.2%}")
        del z,pred;gc.collect()
    results['status']='complete';results['wall_seconds']=time.time()-t0
    save(out/'results.json',results);write_report(out,results,settings)
    log('All GPT-2 pilot comparisons complete')


if __name__=='__main__':main()
