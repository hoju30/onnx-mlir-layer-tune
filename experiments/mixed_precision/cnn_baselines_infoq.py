#!/usr/bin/env python3
"""InfoQ SMI score table (run_cnn_baselines.score_infoq launches this in InfoQ's venv): --model resnet18 (torchvision
checkpoint, InfoQ's resnet18_gluon configuration) or mobilenetv2 (timm_ra weights in InfoQ's own MobileNetV2 definition,
InfoQ's mobilenetv2 configuration; see setup_mobilenetv2).

InfoQ's own code does the work, unchanged: mqbench_quantization.quant (uniform_quantized_models / mpq_quantize: MQBench
Academic backend, per-tensor symmetric MSE-observed weights, 8-bit asymmetric EMA-MSE activations, BN statistics
re-estimated in calibration), SMI.utils.smi_calculate.calculate_smi (k-SMI with KSG on GPU via cupy / cuml, m = 100), the
DINOv2 input embedding of SMI.input_compressor and SMI_score.score_utils.get_smi_scores. The port only replaces the
data source (the run's calibration pool through the common model preprocessing; DINOv2 gets InfoQ's own CenterCrop(224) +
ToTensor items), the model (torchvision checkpoint instead of gluon_resnet18_v1b) and the observer names: InfoQ's
resnet18_gluon observers (block outputs after the second activation, the flattened global pool, fc) are found in the
prepared torchvision graph and the score table is built with InfoQ's gluon column names, so get_smi_scores and
conv_to_block_output_dict run exactly as for resnet18_gluon. Bits 4 and 6 for the searched convolutions; 8 is the
background (cost 0). Author SMI caches (RUNS/) are not read; the per-layer accuracy column of the original script is not
computed (it never enters the score).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

INFOQ = Path('/home/hoju/test/InfoQ')
sys.path.insert(0, str(INFOQ))
TV_CKPT = Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
GLUON_OBS = ['layer1_0_act2', 'layer1_1_act2', 'layer2_0_act2', 'layer2_1_act2', 'layer3_0_act2', 'layer3_1_act2',
             'layer4_0_act2', 'global_pool_flatten', 'fc']
BITS = (4, 6)
SMI_BATCH, CAL_BATCH, N_SMI = 32, 64, 1000


class DictLoader:
    """the {'image', 'label'} batches InfoQ's loaders yield, from preprocessed arrays."""
    def __init__(self, x, y, batch):
        import torch
        self.x, self.y, self.batch, self.torch = x, y, batch, torch

    def __len__(self):
        return -(-len(self.x) // self.batch)

    def __iter__(self):
        for i in range(0, len(self.x), self.batch):
            yield {'image': self.torch.from_numpy(np.ascontiguousarray(self.x[i:i + self.batch])),
                   'label': self.torch.from_numpy(self.y[i:i + self.batch])}


def observer_names(prepared):
    """torchvision names of InfoQ's resnet18_gluon observers in the MQBench-prepared graph."""
    names = [n for n, _ in prepared.named_modules()]
    out = {}
    for g in GLUON_OBS[:7]:
        layer, blk = g.split('_')[0], g.split('_')[1]
        want = f'{layer}_{blk}_relu_1_post_act_fake_quantizer'   # block output: the ReLU after the residual add
        if want not in names:
            raise SystemExit(f'no block output quantizer for {g}: {want}')
        out[g] = want
    pool = [n for n in names if n.startswith(('flatten', 'avgpool')) and n.endswith('_post_act_fake_quantizer')]
    if not pool:
        raise SystemExit(f'no global pool quantizer: {[n for n in names if "fake_quantizer" in n][-8:]}')
    out['global_pool_flatten'] = sorted(pool, key=lambda n: (not n.startswith('flatten'), n))[0]
    out['fc'] = 'fc'
    return out


def setup_resnet18(plan, torch, x):
    """torchvision ResNet18 with InfoQ's resnet18_gluon observers found under their torchvision names."""
    import torchvision
    import yaml
    cfg = yaml.safe_load(open(INFOQ / 'SMI/configs/resnet18_imagenet.yaml'))
    model = torchvision.models.resnet18(weights=None)
    model.load_state_dict(torch.load(TV_CKPT, map_location='cpu'))
    module_of = plan['search_space']['module_of_node']
    return {'cfg': cfg, 'model': model, 'layer_of_node': dict(module_of), 'model_name': 'resnet18_gluon', 'n_cal': 64,
            'cal_batches': 1, 'n_classes': cfg['num_target_classes'], 'per_class': cfg['num_samples_per_class'],
            'columns': [f'{g}_post_act_fake_quantizer' if g != 'fc' else 'fc' for g in GLUON_OBS], 'control': None}


def setup_mobilenetv2(plan, torch, x):
    """InfoQ's own MobileNetV2 definition (models/mobilenetv2_imagenet.py, tonylins layout: the names its mobilenetv2
    config, layer-to-block map and get_smi_scores use) carrying the timm mobilenetv2_100.ra_in1k weights: the two
    definitions are the same network (ReLU6, same channels / strides / padding, BN eps 1e-5), so every Conv2d, BN and the
    classifier are copied in module order with shapes checked; control: its logits equal the timm model's."""
    import timm
    import yaml
    from safetensors.torch import load_file
    from models.mobilenetv2_imagenet import mobilenetv2
    cfg = yaml.safe_load(open(INFOQ / 'SMI/configs/mobilenetv2_imagenet.yaml'))
    src = timm.create_model('mobilenetv2_100.ra_in1k', pretrained=False)
    src.load_state_dict(load_file(plan['model']['torch_checkpoint']))
    src.eval()
    torch.manual_seed(0)
    model = mobilenetv2().eval()

    def mods(m, types):
        return [(n, v) for n, v in m.named_modules() if isinstance(v, types)]
    sc, dc = mods(src, torch.nn.Conv2d), mods(model, torch.nn.Conv2d)
    sb, db = mods(src, torch.nn.BatchNorm2d), mods(model, torch.nn.BatchNorm2d)
    sl, dl = mods(src, torch.nn.Linear), mods(model, torch.nn.Linear)
    if not (len(sc) == len(dc) and len(sb) == len(db) and len(sl) == len(dl) == 1):
        raise SystemExit(f'MobileNetV2 layouts differ: conv {len(sc)}/{len(dc)}, bn {len(sb)}/{len(db)}, fc {len(sl)}/{len(dl)}')
    with torch.no_grad():
        for (a, s_), (b, d_) in zip(sc + sl, dc + dl):
            if s_.weight.shape != d_.weight.shape or (s_.groups if hasattr(s_, 'groups') else 1) != (d_.groups if hasattr(d_, 'groups') else 1):
                raise SystemExit(f'shape mismatch {a} {tuple(s_.weight.shape)} vs {b} {tuple(d_.weight.shape)}')
            d_.weight.copy_(s_.weight)
            if d_.bias is not None:
                d_.bias.copy_(s_.bias)
        for (a, s_), (b, d_) in zip(sb, db):
            if s_.num_features != d_.num_features or s_.eps != d_.eps:
                raise SystemExit(f'BN mismatch {a} vs {b}')
            for k in ('weight', 'bias', 'running_mean', 'running_var'):
                getattr(d_, k).copy_(getattr(s_, k))
        xx = torch.from_numpy(np.ascontiguousarray(x[:10]))
        diff = float((src(xx) - model(xx)).abs().max())
    if diff > 1e-3:
        raise SystemExit(f'InfoQ MobileNetV2 with timm weights differs from timm: {diff}')
    conv_name = dict(zip([n for n, _ in sc], [n for n, _ in dc]))          # timm conv -> InfoQ conv
    layer_of_node = {node: (conv_name[mod] if mod in conv_name else mod)
                     for node, mod in plan['search_space']['module_of_node'].items()}
    return {'cfg': cfg, 'model': model, 'layer_of_node': layer_of_node, 'model_name': 'mobilenetv2',
            'n_cal': 64 * cfg['num_calibration_batches'], 'cal_batches': cfg['num_calibration_batches'],
            'n_classes': 10, 'per_class': 100,                              # the pool (user choice), not 80 x 60
            'columns': list(cfg['target_modules']),
            'control': {'timm_vs_infoq_definition_logits_max_abs': diff, 'conv_map_timm_to_infoq': conv_name}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--max-layers', type=int, help='smoke test only: the first k searched layers')
    ap.add_argument('--model', choices=('resnet18', 'mobilenetv2'), default='resnet18')
    a = ap.parse_args()
    run = a.run_dir
    tm = {}
    t0 = time.perf_counter()
    import torch
    import torchvision
    import cupy as cp
    import pandas as pd
    import yaml
    from PIL import Image
    from torchvision import transforms
    np.random.seed(0)
    torch.manual_seed(0)
    cp.random.seed(0)
    # SMI/vae.py loads the whole imagenet-1k-256x256 train + validation splits at module level; input_compressor imports
    # it only for the retired VAE compressor (X is the DINOv2 embedding), so a stub stands in for it (no data loaded)
    import types
    stub = types.ModuleType('SMI.vae')

    def _retired(*args, **kwargs):
        raise RuntimeError('SMI.vae (retired VAE compressor) is not used: X = DINOv2 embeddings')
    stub.load_model_with_fc = _retired
    sys.modules['SMI.vae'] = stub
    from mqbench_quantization.quant import mpq_quantize, uniform_quantized_models
    from SMI.utils.smi_calculate import calculate_smi
    from SMI.input_compressor import dinov2_model, imagenet_embeddings
    from SMI.utils.mqbench_layer_to_block import conv_to_block_output_dict
    from SMI_score.score_utils import get_smi_scores
    plan = json.loads((run / 'plan.json').read_text())
    fixed = set(plan['search_space']['fixed_int8_weight'])
    x = np.load(run / 'assets/pool_inputs.npy', mmap_mode='r')[:N_SMI]
    y = np.load(run / 'assets/pool_labels.npy')[:N_SMI]
    st = (setup_resnet18 if a.model == 'resnet18' else setup_mobilenetv2)(plan, torch, x)
    cfg, layer_of_node = st['cfg'], st['layer_of_node']
    searched = [layer_of_node[n] for n in plan['search_space']['searched']]
    if a.max_layers:
        searched = searched[:a.max_layers]
    dev = 'cuda'
    model = st['model'].eval().to(dev)
    N_CAL = st['n_cal']
    smi_loader = DictLoader(x, y, SMI_BATCH)
    cal_loader = DictLoader(x[:N_CAL], y[:N_CAL], CAL_BATCH)
    from SMI.dataloaders import get_targets
    targets = get_targets(smi_loader, 'imagenet_256x256', st['n_classes'], st['per_class'])   # remaps to 0..9
    torch.cuda.synchronize()
    tm['prepare_s'] = time.perf_counter() - t0
    # X: InfoQ's DINOv2 embedding of the same images (its items: CenterCrop(224) + ToTensor, no normalization)
    t1 = time.perf_counter()
    man = json.loads((run / 'assets/pool_images/manifest.json').read_text())['images'][:N_SMI]
    tf = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()])
    items = []
    for e in man:
        with Image.open(run / 'assets/pool_images' / e['file']) as im:
            items.append({'image': tf(im.convert('RGB'))})
    processor, dino = dinov2_model()
    X = np.array(imagenet_embeddings(processor, dino, items))
    del dino
    torch.cuda.empty_cache()
    tm['input_embedding_s'] = time.perf_counter() - t1
    # MQBench background models (all 8-bit, and the 4 / 6-bit weight sources)
    t2 = time.perf_counter()
    uniform = uniform_quantized_models(bitwidths=list(BITS), model_to_quantize=model, num_calibration_batches=st['cal_batches'],
                                       calib_dataloader=cal_loader, per_channel=cfg['per_channel'],
                                       activation_bit=cfg['activation_bit'])
    if a.model == 'resnet18':
        obs = observer_names(uniform[8])
        targets_q = [obs[g] for g in GLUON_OBS]
        rename = {obs[g]: c for g, c in zip(GLUON_OBS, st['columns'])}
        xl = [obs[g.removesuffix('_post_act_fake_quantizer')] for g in cfg['XL_OBSERVERS']]
        yl = [obs[g.removesuffix('_post_act_fake_quantizer')] for g in cfg['YL_OBSERVERS']]
        needed_of = lambda b: obs[b.removesuffix('_post_act_fake_quantizer')]   # noqa: E731
    else:
        names = {n for n, _ in uniform[8].named_modules()}
        missing = [t for t in cfg['target_modules'] if t not in names]
        if missing:
            raise SystemExit(f'InfoQ mobilenetv2 target modules not in the prepared graph: {missing}')
        obs = {t: t for t in cfg['target_modules']}
        targets_q, rename = list(cfg['target_modules']), {t: t for t in cfg['target_modules']}
        xl, yl = list(cfg['XL_OBSERVERS']), list(cfg['YL_OBSERVERS'])
        needed_of = lambda b: b                                                  # noqa: E731
    block = conv_to_block_output_dict(model=st['model_name'], dataset='imagenet_256x256')
    torch.cuda.synchronize()
    tm['background_models_s'] = time.perf_counter() - t2

    def smi(model_q, layer_needed):
        r = calculate_smi(model_q, smi_loader, targets, X, number_of_class=st['n_classes'], m=cfg['smi_m'],
                          target_modules=targets_q, layer_needed=layer_needed, XL_OBSERVERS=xl, YL_OBSERVERS=yl)
        return {rename[k]: v for k, v in r.items()}
    t3 = time.perf_counter()
    records, per = [], {}
    for layer in searched:
        for bw in BITS:
            tl = time.perf_counter()
            qd ={name: 8 for name, mod in model.named_modules() if isinstance(mod, torch.nn.Conv2d)}
            qd[layer] = bw
            model_q = mpq_quantize(layer_bitwidth_dict=qd, num_calibration_batches=st['cal_batches'], calib_dataloader=cal_loader,
                                   bit_model_dict=uniform)
            needed = needed_of(block[layer])
            res = smi(model_q, needed)
            rec = {'layer': layer, 'bw': bw, 'acc': np.nan}
            for key in st['columns']:
                if key in res:
                    rec[key + '_mi_ly'], rec[key + '_mi_xl'] = float(res[key][1]), float(res[key][0])
            records.append(rec)
            torch.cuda.synchronize()
            per[f'{layer}|{bw}'] = time.perf_counter() - tl
            del model_q
            torch.cuda.empty_cache()
            print(f'{layer} {bw}: {per[f"{layer}|{bw}"]:.1f}s', flush=True)
    t8 = time.perf_counter()
    res8 = smi(uniform[8], None)
    rec = {'layer': 'all', 'bw': 8, 'acc': np.nan}
    for key in st['columns']:
        rec[key + '_mi_ly'], rec[key + '_mi_xl'] = float(res8[key][1]), float(res8[key][0])
    records.append(rec)
    per['all|8'] = time.perf_counter() - t8
    tm['smi_s'] = time.perf_counter() - t3
    # score: InfoQ's get_smi_scores on the gluon-named table
    t4 = time.perf_counter()
    cols = ['layer', 'bw', 'acc'] + [c for key in st['columns'] for c in (key + '_mi_ly', key + '_mi_xl')]
    df = pd.DataFrame(records, columns=cols)
    df.insert(0, 'Unnamed: 0', range(len(df)))
    layer_bit = {name: 8 for name, mod in model.named_modules() if isinstance(mod, torch.nn.Conv2d)}
    scores = get_smi_scores(layer_bit_dict=layer_bit, column_names=df.columns.tolist(), score_df=df, XL_CORR=None,
                            LY_CORR=None, model_name=st['model_name'], dataset='imagenet_256x256', BIT_WIDTHS=[4, 6, 8])
    lnames = list(layer_bit)
    cost = {}
    for node, mod in layer_of_node.items():
        if node in fixed:
            cost[node] = {'int4': 0.0, 'int6': 0.0, 'int8': 0.0}          # never variables in the search
            continue
        s = scores[lnames.index(mod)] if mod in searched else {4: 0.0, 6: 0.0, 8: 0.0}
        cost[node] = {f'int{b}': float(-s[b]) for b in (4, 6, 8)}
    tm['score_s'] = time.perf_counter() - t4
    tm['total_s'] = time.perf_counter() - t0
    df.to_csv(a.out.with_suffix('.records.csv'), index=False)
    a.out.write_text(json.dumps({
        'method': 'InfoQ', 'cost': cost, 'raw_scores': {str(k): {str(b): float(v) for b, v in s.items()} for k, s in scores.items()},
        'model': a.model, 'infoq_model_name': st['model_name'], 'observer_map': obs, 'xl_observers': xl, 'yl_observers': yl,
        'layer_of_node': layer_of_node, 'targets': {'classes': st['n_classes'], 'per_class': st['per_class']},
        'control': st['control'],
        'calibration_images': {'smi': N_SMI, 'mqbench_calibration': N_CAL}, 'per_layer_bit_s': per, 'timing': tm,
        'device': torch.cuda.get_device_name(0), 'versions': {'torch': torch.__version__, 'cupy': cp.__version__},
        'note': 'cost = -get_smi_scores score (>= 0, minimized); int8 = background, cost 0'}, indent=1) + '\n')


if __name__ == '__main__':
    main()
