#!/usr/bin/env python3
"""CNN / ImageNet weight mixed-precision baseline comparison with activations fixed at INT8 (A8).

--model resnet18 (torchvision, runs/cnn_baselines_20261029, the first run), mobilenetv2 / efficientnet_b0 (timm_ra, the
formal native suite's checkpoints; images reused from the ResNet18 run, preprocessed with each model's own config).
Five scorers (each a nonnegative additive cost per (node, weight format)) on the same model, candidates, quantizer,
resource model and searcher (InfoQ only where InfoQ has a configuration: ResNet18, MobileNetV2):
  BC     the formal native weight criterion B[n][w] = sum_c mean(g^2)_c ||dW_c||^2 (run_native_bc_pipeline's weights +
         grads statistics; only the parts B needs)
  SQNR   model-output (logit) noise of the single-node QDQ model, N = sum ||z_q - z_fp32||^2 (sqnr_tables; per-channel
         weights, fixed A8 ranges) -- not weight SQNR
  HTQ    HTQ's own score: SNIP connection sensitivity of a randomly re-initialized network (HTQ/sensitivity.py) times
         HTQ's own per-tensor weight perturbation ||W - Q_b(W)||_F (HTQ/utils/pareto_utils.compute_perturbation)
  HAWQ   CLADO_SampleRun/prep_hawq.py + optimize.py: mean Hutchinson trace of the layer Hessian (pyhessian, Rademacher,
         maxIter 100, tol 1e-3, seed 0) / #params x ||Q_b(W) - W||^2 with the common per-channel quantizer
  InfoQ  InfoQ's SMI flow (cnn_baselines_infoq.py in its own venv): per layer and bit, change of k-sliced MI at InfoQ's
         observer blocks relative to its W8A8 MQBench background, score of SMI_score/score_utils.get_smi_scores
Searched: every weight node except the first convolution and the classifier (fixed at INT8 weights for every method;
ResNet18: the 19 inner convolutions, conv1 and fc fixed); weight int4/int6/int8
(per output channel, symmetric), every weight node's input / output activation INT8 with the fixed ranges of the
existing native run. Searcher: mp_milp.milp_select (exact HiGHS; the formal rule for dense models) with an LP-first
control, and mp_auto.auto_select as the common automatic shell. Final models: onnx-mlir native, evaluated on a
never-used ImageNet validation shard. No training, no QAT, no test-based tuning.

Stages (one --run-dir): plan, assets, controls, score_bc, score_sqnr, score_htq, score_hawq, score_infoq, caps, search,
evaluate, timing, analyze. CPU stages: ../../venv/bin/python; score_htq / score_hawq:
/home/hoju/test/HTQ/.venv/bin/python; score_infoq launches /home/hoju/test/InfoQ/.venv_infoq/bin/python itself.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R18_RUN = HERE / 'runs/cnn_baselines_20261029'                        # first run; its images and pyhessian are reused
TV_CKPT = Path.home() / '.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
# per model: ONNX dir, existing formal native run (fixed A8 ranges, B reference table), torch source, InfoQ support
MODELS = {
    'resnet18': {'label': 'ResNet18', 'dir': 'resnet18/torchvision_v1', 'run': R18_RUN, 'torch': ('torchvision', 'resnet18'),
                 'infoq': 'resnet18_gluon', 'infoq_mqb_cal': 64, 'matched_R': False, 'hawq_batch': None},
    'mobilenetv2': {'label': 'MobileNetV2', 'dir': 'mobilenetv2/timm_ra', 'run': HERE / 'runs/cnn_baselines_20261029_mobilenetv2',
                    'torch': ('timm', 'mobilenetv2_100.ra_in1k'), 'infoq': 'mobilenetv2', 'infoq_mqb_cal': 320, 'matched_R': True,
                    'hawq_batch': None},
    'efficientnet_b0': {'label': 'EfficientNet-B0', 'dir': 'efficientnet_b0/timm_ra',
                        'run': HERE / 'runs/cnn_baselines_20261029_efficientnet_b0', 'torch': ('timm', 'efficientnet_b0.ra_in1k'),
                        'infoq': None, 'infoq_mqb_cal': None, 'matched_R': True,
                        'hawq_batch': 16},                              # single batch of 64 exceeds the 12 GB GPU
}
MODEL = 'resnet18'
SPEC = MODELS[MODEL]
RUN = SPEC['run']
MODEL_DIR = HERE / '../imagenet100/model_dataset' / SPEC['dir']
NATIVE_REF = HERE / 'runs/native_auto_2d_20261011' / MODEL           # existing formal native run (10 val images)
VAL_SHARD = 'data/validation-00001-of-00014.parquet'                  # benjamin-paine/imagenet-1k, never used before
HF256 = 'benjamin-paine/imagenet-1k-256x256'
INFOQ_DIR = Path('/home/hoju/test/InfoQ')
INFOQ_PY = INFOQ_DIR / '.venv_infoq/bin/python'
HTQ_DIR, CLADO_DIR = Path('/home/hoju/test/HTQ'), Path('/home/hoju/test/CLADO_MPQ/CLADO_SampleRun')
PYHESSIAN_DIR = R18_RUN / 'deps/pyhessian'                            # pip --target install, isolated from the shared venv
FIXED = ('node_Conv_291', 'node_linear')                               # first conv, classifier: INT8 weights for every method
WFMT, AFMT = ('int4', 'int6', 'int8'), ('int8',)
METHODS = ('BC', 'SQNR', 'HTQ', 'HAWQ', 'InfoQ')
CAL_N = {'BC': 10, 'SQNR': 10, 'HTQ': 32, 'HAWQ': 64, 'InfoQ': 1000}  # nested prefixes of one calibration pool
INFOQ_MQB_CAL = 64                                                      # InfoQ's MQBench calibration images (first k)
INFOQ_SEED, INFOQ_NCLS, INFOQ_PER_CLASS = 42, 10, 100                  # SMI/dataloaders.load_imagenet_256x256_ds
HTQ_SEED, HAWQ_SEED, BOOT_SEED, BOOT = 0, 0, 26102901, 20000
HAWQ_MAXITER, HAWQ_TOL = 100, 1e-3
AUTO = {'samples': 17, 'refinements': 2, 'time_limit': 60}
MILP_TIME_LIMIT = 60
LOGF = None


def configure(model):
    """set the module globals of one model (the import-time defaults are ResNet18's, the first run)."""
    global MODEL, SPEC, RUN, MODEL_DIR, NATIVE_REF, FIXED, METHODS, CAL_N, INFOQ_MQB_CAL
    MODEL, SPEC = model, MODELS[model]
    RUN = SPEC['run']
    MODEL_DIR = HERE / '../imagenet100/model_dataset' / SPEC['dir']
    NATIVE_REF = HERE / 'runs/native_auto_2d_20261011' / model
    METHODS = ('BC', 'SQNR', 'HTQ', 'HAWQ') + (('InfoQ',) if SPEC['infoq'] else ())
    CAL_N = {m: n for m, n in {'BC': 10, 'SQNR': 10, 'HTQ': 32, 'HAWQ': 64, 'InfoQ': 1000}.items() if m in METHODS}
    INFOQ_MQB_CAL = SPEC['infoq_mqb_cal']
    try:                                                               # first convolution, classifier
        import onnx
        import eval_random_mixed_precision as erm
        w = [n['name'] for n in erm.supported_nodes(onnx.load(str(MODEL_DIR / 'model.onnx'))) if n.get('weight_const')]
        FIXED = (w[0], w[-1])
    except ImportError:                                                # GPU venv without onnx: check_plan reads them
        FIXED = None


def torch_ckpt():
    """the checkpoint file the PyTorch scorers load (torchvision .pth or timm safetensors in the HF hub cache)."""
    if SPEC['torch'][0] == 'torchvision':
        return TV_CKPT
    return next((Path.home() / f'.cache/huggingface/hub/models--timm--{SPEC["torch"][1]}/snapshots').glob('*/model.safetensors'))


def now():
    return datetime.now(ZoneInfo('Asia/Taipei')).isoformat(timespec='seconds')


def log(msg):
    line = f'{now()} {msg}'
    print(line, flush=True)
    if LOGF:
        LOGF.write(line + '\n')
        LOGF.flush()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, obj):
    p = Path(p)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 24), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return os.path.relpath(p, HERE)


def timing(run, stage, method, seconds, **extra):
    with open(run / 'timing.jsonl', 'a') as fh:
        fh.write(json.dumps({'time': now(), 'stage': stage, 'method': method, 'seconds': seconds, **extra}) + '\n')


def code_files():
    return ['run_cnn_baselines.py', 'cnn_baselines_infoq.py', 'run_native_bc_pipeline.py', 'eval_random_mixed_precision.py',
            'mp_milp.py', 'mp_auto.py', 'sqnr_tables.py', 'run_w8_per_channel_check.py', 'run_olmoe_search_final.py',
            'evaluate_native_search_chunked.py', 'htq_pareto_search.py', 'sample_configs_htq.py']


def baseline_files():
    return [HTQ_DIR / 'sensitivity.py', HTQ_DIR / 'utils/pareto_utils.py', HTQ_DIR / 'main.py', HTQ_DIR / 'pareto_frontier.py',
            CLADO_DIR / 'prep_hawq.py', CLADO_DIR / 'optimize.py', INFOQ_DIR / 'SMI/smi_calculation.py',
            INFOQ_DIR / 'SMI/utils/smi_calculate.py', INFOQ_DIR / 'SMI/utils/basic_functions.py',
            INFOQ_DIR / 'SMI_score/score_utils.py', INFOQ_DIR / 'SMI/configs/resnet18_imagenet.yaml',
            INFOQ_DIR / 'SMI/utils/mqbench_layer_to_block.py', INFOQ_DIR / 'mqbench_quantization/quant.py',
            INFOQ_DIR / 'SMI/input_compressor.py', INFOQ_DIR / 'SMI/dataloaders.py',
            INFOQ_DIR / 'SMI/configs/mobilenetv2_imagenet.yaml', INFOQ_DIR / 'models/mobilenetv2_imagenet.py']


def check_plan(run):
    global FIXED
    pl = read(run / 'plan.json')
    if pl.get('model_key', 'resnet18') != MODEL:
        raise SystemExit(f'{run} is a {pl.get("model_key", "resnet18")} run, not {MODEL}')
    fixed = tuple(pl['search_space']['fixed_int8_weight'])
    if FIXED is not None and FIXED != fixed:
        raise SystemExit(f'fixed nodes {FIXED} differ from the plan {fixed}')
    FIXED = fixed
    am = run / 'amendments.json'
    hashes = read(am)[-1]['code_hashes'] if am.exists() else pl['code_hashes']
    for f, h in hashes.items():
        if sha(HERE / f) != h:
            raise SystemExit(f'code changed since the plan / last amendment: {f} (record an amendment)')
    return pl


# --------------------------------------------------------------------------- #
# model / graph helpers (CPU)
# --------------------------------------------------------------------------- #
def graph_info():
    import onnx
    import eval_random_mixed_precision as erm
    from sample_configs_htq import compute_weight_elems_and_flops_proxy
    graph = onnx.load(str(MODEL_DIR / 'model.onnx'))
    nodes = erm.supported_nodes(graph)
    names = [n['name'] for n in nodes if n.get('weight_const')]
    elems, macs = compute_weight_elems_and_flops_proxy(graph, nodes)
    by = {n.name: n for n in graph.graph.node}
    module = {n: by[n].input[1].removesuffix('.weight') for n in names}            # ONNX node -> torchvision module
    return graph, nodes, names, {n: elems[n] for n in names}, {n: macs[n] for n in names}, module


def tensor_ranges(nodes):
    """the fixed A8 ranges by tensor (existing native ranges_by_point of the 10 validation calibration images)."""
    rp = read(NATIVE_REF / 'ranges_by_point.json')
    out = {}
    for n in nodes:
        if n['name'] not in rp:
            continue
        for slot, t in (('x', n['act_tensors'][0]), ('y', n['act_tensors'][-1])):
            if slot in rp[n['name']]:
                r = tuple(rp[n['name']][slot])
                if out.setdefault(t, r) != r:
                    raise SystemExit(f'tensor {t} has two ranges')
    return out


def resources(cfg, names, elems, macs):
    """independently recomputed resources of a {node: {'w', 'a'}} config over the weight nodes."""
    bits = {'int4': 4, 'int6': 6, 'int8': 8, 'fp32': 32}
    wb = sum(elems[n] * bits[cfg[n]['w']] for n in names) / 8
    bops = sum(macs[n] * bits[cfg[n]['w']] * bits[cfg[n]['a']] for n in names)
    ref_b, ref_o = sum(elems.values()), sum(64 * m for m in macs.values())
    hist = {}
    for n in names:
        hist[cfg[n]['w']] = hist.get(cfg[n]['w'], 0) + 1
    return {'weight_bytes': wb, 'weight_MB': wb / 1e6, 'weight_MiB': wb / 2 ** 20, 'bops': bops, 'GBOPs': bops / 1e9,
            'size_ratio': wb / ref_b, 'bops_ratio': bops / ref_o, 'R': max(wb / ref_b, bops / ref_o), 'w_hist': hist}


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def plan(run, smoke):
    if (run / 'plan.json').exists():
        raise SystemExit('plan exists')
    from run_olmoe_dynamic_route import env_info
    graph, nodes, names, elems, macs, module = graph_info()
    searched = [n for n in names if n not in FIXED]
    label, nw = SPEC['label'], len(names)
    others = [m for m in METHODS if m != 'BC']
    r18 = MODEL == 'resnet18'
    if r18:
        request = ('2026-10-29 CNN mixed-precision baseline comparison (ResNet18 / ImageNet, five scorers, common searcher); '
                   'user choices of 2026-10-29: one calibration pool with each method\'s native sample count (nested '
                   'prefixes), one new ImageNet validation shard as held-out, InfoQ in an isolated venv with its original '
                   'dependencies')
    else:
        request = (f'2026-10-29 follow-up of the ResNet18 comparison on {label} (user: "測試另外兩個model"; choices of '
                   '2026-10-29: MobileNetV2 + EfficientNet-B0 timm_ra; InfoQ on MobileNetV2 with the 1000-image pool of the '
                   'ResNet18 run; InfoQ on EfficientNet-B0 marked not run (no InfoQ configuration); BC at the R of every '
                   'other method\'s auto pick pre-registered)')
    pl = {
        'time': now(),
        'model_key': MODEL,
        'request': request,
        'instructions_found': 'no AGENTS.md applies (only aimet/CLAUDE.md and universal/CLAUDE.md exist, for other projects)',
        'question': 'with activations fixed at INT8 and a common searcher, does the formal BC weight criterion reach a '
                    f'competitive accuracy / resource tradeoff at lower offline cost than {", ".join(others)} scores?',
        'scope': f'{label} / ImageNet only; fixed A8 weight mixed precision (not a full W+A pipeline); the comparison is of '
                 'scores and the configurations they select under one searcher, not of the papers\' native pipelines; '
                 'baselines are controlled adaptations to the common candidates, not reproductions of the papers',
        'model': {'onnx': rel(MODEL_DIR / 'model.onnx'), 'onnx_sha256': sha(MODEL_DIR / 'model.onnx'),
                  'checkpoint': ('torchvision resnet18 IMAGENET1K_V1' if r18 else f'timm {SPEC["torch"][1]}'),
                  'torch_checkpoint': str(torch_ckpt()), 'torch_checkpoint_sha256': sha(torch_ckpt()),
                  'note': 'the ONNX model has BN folded into the convolutions (conv weight / bias initializers named after '
                          'the torch modules); PyTorch scorers load the same checkpoint; a control checks that '
                          'folding it reproduces the ONNX initializers and logits',
                  'preprocessing': ('eval_random_mixed_precision.preprocess_torchvision: Resize(256), CenterCrop(224), '
                                    'ToTensor, Normalize(ImageNet mean/std) for every model input of every method' if r18 else
                                    'eval_random_mixed_precision.preprocess_by_source with the model\'s own timm '
                                    'pretrained_cfg (bicubic, crop_pct 0.875, ImageNet mean / std) for every model input of '
                                    'every method')},
        'search_space': {'weight_nodes': len(names), 'searched': searched, 'fixed_int8_weight': list(FIXED),
                         'weight_formats': list(WFMT), 'activation': 'int8 on the input and output tensor of every weight '
                         'node (pass semantics of a=int8), per-tensor asymmetric, fixed ranges; activation-only nodes FP32',
                         'quantizer': 'pass weights per output channel, symmetric max-abs, round half away from zero '
                                      '(MIXEDP_WEIGHT_PER_CHANNEL=1; run_w8_per_channel_check.per_channel_int); biases '
                                      'FP32; no FP / Posit candidates',
                         'ranges': f'{rel(NATIVE_REF / "ranges_by_point.json")} (sha256 {sha(NATIVE_REF / "ranges_by_point.json")}), '
                                   'the existing native ranges of the 10 validation calibration images, never recomputed',
                         'module_of_node': module},
        'resources': {'weight_bytes': f'sum over the {nw} weight nodes of N_i x b_w,i / 8 (bytes; MB = 1e6, MiB = 2^20)',
                      'bops': f'sum over the {nw} weight nodes of MAC_i x b_w,i x b_a,i; MAC_i = weight elements x output '
                              'positions (Conv) or weight elements (Gemm), not doubled; ReLU / Add / Mul / pooling not counted',
                      'reference': f'common all-INT8 configuration (W8A8 on the {nw} weight nodes): size_ratio = bytes / '
                                   'bytes_int8, bops_ratio = BOPs / BOPs_int8, R = max(size_ratio, bops_ratio)',
                      'non_searched': 'FP32 biases (folded BN), per-channel weight scales, activation scale / zero points '
                                      'reported separately; .so / constants file sizes are FP32 + QDQ artifacts, never a '
                                      'packed low-bit size',
                      'int8_reference_bytes': sum(elems.values()), 'int8_reference_bops': sum(64 * m for m in macs.values()),
                      'weight_elems': elems, 'macs': macs},
        'calibration': {'pool': f'{INFOQ_NCLS} classes x {INFOQ_PER_CLASS} images of {HF256} train selected by InfoQ\'s own '
                                f'rule (SMI/dataloaders.load_imagenet_256x256_ds: np.random.seed({INFOQ_SEED}), np.random.choice '
                                'of the sorted labels without replacement, then np.random.shuffle of each selected class\'s '
                                'index list in selection order, first 100); pool order = round robin over the classes in '
                                'selection order, each class in its shuffled order, so every prefix is class-balanced',
                        'per_method_count': CAL_N,
                        'infoq_mqbench_calibration': (None if not SPEC['infoq'] else f'first {INFOQ_MQB_CAL} pool images '
                                                      f'(InfoQ {SPEC["infoq"]} config: num_calibration_batches x 64)'),
                        'labels': 'true labels of the pool images',
                        'a8_ranges': 'fixed (see search_space.ranges), not from the pool',
                        'note': 'pool images are ImageNet train images (256x256 resized by the dataset); every method sees '
                                'a prefix of the same ordered pool; data volume differences are part of the method costs'
                                + ('' if r18 else f'; the pool images of {rel(R18_RUN)} (symlinked, sha256 of every image '
                                   'checked against its manifest), preprocessed with this model\'s config')},
        'evaluation': {'data': f'every image of benjamin-paine/imagenet-1k {VAL_SHARD} (no earlier run references this shard; '
                               'checked by grep before the plan)' + ('' if r18 else f'; the same 3572 images as {rel(R18_RUN)} '
                               '(symlinked, sha256 checked), preprocessed with this model\'s config; the shard was read by '
                               'the ResNet18 run only, never to select or tune anything for this model'),
                       'backend': 'onnx-mlir native (run_native_auto.compile_and_eval '
                               'arithmetic: erm.build_so, constants to file, Krnl -O3 / clang -O3, MIXEDP_WEIGHT_PER_CHANNEL=1), '
                               'chunked processes (bit-identical to one process)', 'configs': 'FP32, all-INT8 reference, '
                               f'A: 3 caps x {len(METHODS)} methods, B: {len(METHODS)} auto picks'
                               + (f', C: BC at the R of each of the {len(others)} other auto picks' if SPEC['matched_R'] else '')
                               + '; identical configurations compiled and evaluated once',
                       'per_image': 'logits, loss, top-1 / top-5 correctness', 'isolation': 'configurations frozen '
                       '(configs.json hash) before any evaluation; evaluation never selects or tunes anything'},
        'scorers': {
            'BC': 'B[n][w] of run_native_bc_pipeline.native_tables (weights pass + grads statistics on the first 10 pool '
                  'images, true labels; ranges / errors chains are not needed for weight-only B and are not run); control: '
                  'the same driver on the existing 10 validation calibration inputs reproduces '
                  f'{rel(NATIVE_REF / "table.json")} B',
            'SQNR': 'sqnr_tables.sqnr_table with per-channel weights (run_w8pc_full_compare patch) and the fixed ranges, on '
                    'the first 10 pool images: N[n][(w, int8)] = sum ||z - z_fp32||^2 with only node n quantized (others '
                    'FP32); additive cost N (total logit noise if single-node noises add = -SQNR up to a constant); this is '
                    'model-output noise, not weight SQNR',
            'HTQ': f'HTQ/sensitivity.snip unchanged in logic: deep copy of {SPEC["torch"][0]} {SPEC["torch"][1]} '
                   f'(architecture only) re-initialized (torch seed '
                   f'{HTQ_SEED}, xavier_normal_ on every Conv2d / Linear weight), train mode, weight masks, F.nll_loss on the '
                   'raw logits (as in the original), one batch = first 32 pool images, score = mean |d loss / d mask| per '
                   'layer; cost = score x HTQ/utils/pareto_utils.compute_perturbation(W, b) = ||W - Q_b(W)||_F with HTQ\'s '
                   'per-tensor asymmetric quantizer on the unfolded checkpoint weights (as in the original); port '
                   'differences: batch from the common pool instead of 32 random-crop train images, common checkpoint',
            'HAWQ': f'CLADO_SampleRun/prep_hawq.estimate_trace: torch / numpy seed {HAWQ_SEED}, only the layer\'s weight '
                    f'requires grad, pyhessian.hessian(model, CrossEntropyLoss, data=(first 64 pool images, labels)).trace('
                    f'maxIter={HAWQ_MAXITER}, tol={HAWQ_TOL}) (Rademacher probes, stop when the running mean changes by '
                    '< tol relative); optimize.py: cost = mean(trace samples) / numel x ||Q_b(W) - W||^2 with the common '
                    'per-channel quantizer on the unfolded weights (equal to the deployed folded error mapped back: per-'
                    'channel symmetric quantization commutes with the BN scale); model in eval mode (pyhessian); negative '
                    'mean trace: cost 0 for that layer (flagged, the rule CLADO\'s README mentions), non-finite: the run fails',
            'InfoQ': infoq_plan_text(len(searched))}}
    pl['scorers'] = {k: v for k, v in pl['scorers'].items() if k in METHODS}
    if not SPEC['infoq']:
        pl['scorers_not_run'] = {'InfoQ': 'not run: InfoQ has no configuration for this model (no observers, no layer-to-'
                                          'block map); marked not run as decided by the user on 2026-10-29, no substitute '
                                          'score under its name'}
    pl.update({
        'search': {'solver': 'mp_milp.milp_select (scipy HiGHS, mip_rel_gap 0, attempt at the exact caps, 1 ppm retry, '
                             f'labelled repair; time limit {MILP_TIME_LIMIT} s), the project rule for dense models '
                             '(memory lp-pipeline-regression: MILP for dense, LP-first for large MoE); the request named '
                             'LP-first: run_olmoe_search_final.make_solver("lpfirst") is solved at every A cap as a control and '
                             'its agreement reported', 'objective': 'sum of the method\'s own cost; options of fixed nodes '
                             'other than int8 forbidden; activation int8 only', 'auto_shell': f'mp_auto.auto_select {AUTO}: '
                             '17 log-spaced budgets on R, two refinements, non-dominated frontier, normalized chord knee'},
        'resource_points_A': {'rule': 'from BC\'s calibration-only auto frontier (B shell run on BC costs): knee R_k = R of '
                                      'BC\'s auto pick; R_min = R of all searched int4 (+ fixed int8); loose R_L = R_k + '
                                      '(1 - R_k) / 2; aggressive R_A = R_min + (R_k - R_min) / 2; caps size <= R x '
                                      'int8 bytes and BOPs <= R x int8 BOPs (one factor bounds both ratios, the pipeline '
                                      'convention); exact caps written to caps.json before any other method is searched '
                                      'and before any evaluation', 'per_point': 'every method re-searched at the same caps'},
        'statistics': {'metrics': 'top-1 (primary), top-5, CE loss; accuracy drop vs common FP32 in percentage points',
                       'primary': f'A: at each of the 3 points, each of {" / ".join(others)} minus BC on the same images '
                                  f'({3 * len(others)} comparisons): top-1 difference, exact two-sided McNemar p-value, Holm '
                                  f'correction over the {3 * len(others)} at family alpha 0.05; verdict BC better / method '
                                  'better / undetermined',
                       'intervals': f'paired bootstrap 95% CI ({BOOT} image resamples, seed {BOOT_SEED}, one index set) for '
                                    'top-1 and CE differences; descriptive',
                       'no_equivalence': 'no equivalence or non-inferiority margin is pre-set: a non-significant difference '
                                         'is "undetermined", never "equal"',
                       'B': 'common automatic shell picks: accuracy with their own resources; descriptive only, read with A',
                       **({'C_matched_R': f'secondary family: each other method\'s auto pick minus BC re-searched (MILP) at '
                                          'the exact R of that pick (caps size and BOPs <= R x all-INT8), same images, '
                                          f'exact McNemar, Holm over the {len(others)} at alpha 0.05, same bootstrap; '
                                          'answers "is an auto-shell advantage only more resources"'}
                          if SPEC['matched_R'] else {})},
        'timing': {'stages': ['model / data preparation', 'range calibration (fixed, reused: 0 for every method)',
                              'sensitivity / SMI / Hessian statistics', 'candidate errors and table', 'search at one cap',
                              'complete automatic search', 'final model compilation', 'final evaluation (reported apart)'],
                   'repeats': '3 interleaved repetitions (method order rotated) for BC, SQNR, HTQ, HAWQ; InfoQ 1 full run '
                              '(+1 if it takes < 30 min), and any other scorer whose repetition 1 takes >= 10 min the same; median and range; tables of repetition 1 are the official ones, '
                              'later repetitions must reproduce them', 'devices': 'BC native and SQNR (onnxruntime) on CPU '
                              '(12 cores), HTQ / HAWQ / InfoQ on the RTX 3060; measured pipeline costs, not algorithmic '
                              'speedups', 'cold_vs_warm': 'BC native statistics builds are a real cost: cold (fresh builds) '
                              'measured; warm (reused .so) derived as total minus the grads build'},
        'controls': ['checkpoint folded with BN equals the ONNX initializers; torch FP32 logits equal '
                     'onnxruntime logits on the first 10 pool images', 'preprocessing reproduces the existing native '
                     'calibration inputs from their source images', 'BC driver reproduces the existing B table',
                     'each method\'s search reads its own table (hashes recorded)', 'configurations legal: candidates, fixed '
                     'nodes, resources recomputed from the formats within the exact caps (else invalid)',
                     'identical configurations evaluated once; one configuration evaluated twice through the common entry '
                     'gives identical logits', 'no training; the evaluation images are read only by the evaluate stage '
                     'after the configs freeze', 'InfoQ author caches unused']
                    + (['InfoQ: its own MobileNetV2 definition carrying the timm weights reproduces the timm logits']
                       if MODEL == 'mobilenetv2' else []),
        'boundaries': 'no change to the formal BC criterion, pipeline defaults or earlier artifacts; no proxy refresh, '
                      f'spike fix, fine-tuning, CLADO IQP, AIMET, MPQCO, LIMPQ; {label} only in this run',
        'smoke': smoke or 'none',
        'code_hashes': {f: sha(HERE / f) for f in code_files() if (HERE / f).exists()},
        'baseline_source_hashes': {str(p): sha(p) for p in baseline_files() if p.exists()},
        'environment': env_info()})
    write(run / 'plan.json', pl)
    write_plan_md(run, pl, searched)
    log(f'plan written: {len(searched)} searched nodes, fixed {FIXED}')


def infoq_plan_text(n_searched):
    if SPEC['infoq'] == 'mobilenetv2':
        return ('cnn_baselines_infoq.py --model mobilenetv2 in /home/hoju/test/InfoQ/.venv_infoq: InfoQ\'s own MobileNetV2 '
                'definition (models/mobilenetv2_imagenet.py, tonylins layout) carrying the timm_ra weights (Conv / BN copied '
                'in module order, shapes checked; control: logits equal the timm model), so InfoQ\'s mobilenetv2 config '
                '(SMI/configs/mobilenetv2_imagenet.yaml: 19 target modules, 15 XL observers, 3 YL observers, per-tensor, A8, '
                f'm = 100, {INFOQ_MQB_CAL // 64} calibration batches of 64 = first {INFOQ_MQB_CAL} pool images), its layer-to-'
                'block map and get_smi_scores(model_name="mobilenetv2") run with their own names; X = DINOv2 embeddings, Y = '
                'labels of the 1000 pool images (user choice 2026-10-29; the config\'s 80 classes x 60 images are not used, '
                'get_targets / number_of_class follow the pool: 10 x 100); cost = -score, int8 = background (cost 0); bits 4 '
                f'and 6 for the {n_searched} searched convolutions; no QAT, no author caches; named "InfoQ score / '
                'allocation, training-free controlled comparison"')
    return ('cnn_baselines_infoq.py in /home/hoju/test/InfoQ/.venv_infoq: InfoQ\'s own mqbench_quantization.quant '
                     '(uniform_quantized_models / mpq_quantize, Academic backend, per-tensor weights, 8-bit activations, '
                     'calibration = first 64 pool images), SMI.utils.smi_calculate.calculate_smi (k-SMI, KSG, m = 100) at '
                     'its ResNet18 observers mapped to torchvision names (XL: layer3.0 / layer3.1 / layer4.0 block outputs, '
                     'LY: flattened global pool and fc), X = DINOv2 embeddings of the pool images (SMI.input_compressor), '
                     'Y = labels, all 1000 pool images; SMI_score.score_utils.get_smi_scores formula (cost = (sum |dMI_XL| + '
                     'sum |dMI_LY|) / (sum MI_8) x 1/b, the negative of its maximized score; b = 8 is the background, cost 0); '
                     f'bits 4 and 6 for the {n_searched} searched convolutions; no QAT, no author SMI caches (RUNS/ is not '
                     'read), the accuracy column of the original script is not computed (it never enters the score); named '
                     '"InfoQ score / allocation, training-free controlled comparison"')


def write_plan_md(run, pl, searched):
    label, others = SPEC['label'], [m for m in METHODS if m != 'BC']
    desc = {'BC': 'BC（native weight criterion）', 'SQNR': 'SQNR（模型輸出 logit 雜訊，不是 weight SQNR）',
            'HTQ': 'HTQ（隨機初始化 SNIP × HTQ 自己的 per-tensor 擾動）',
            'HAWQ': 'HAWQ（CLADO 附帶實作：Hessian trace／參數數 × ‖ΔW‖²）', 'InfoQ': 'InfoQ（SMI 評分的無訓練受控比較）'}
    lines = [
        f'# 計畫：{label} weight mixed-precision 的 baseline 比較（A8 固定）', '', f'時間：{pl["time"]}。以 plan.json 為準。', '',
        f'- 範圍：{label}／ImageNet；activation 全部 INT8（固定 ranges）；weight 候選 int4／int6／int8（per-channel 對稱）；'
        f'搜尋 {len(searched)} 個 weight node，第一層 conv（{FIXED[0]}）與 classifier（{FIXED[1]}）固定 INT8。',
        f'- {len(METHODS)} 種評分：' + '、'.join(desc[m] for m in METHODS) + '。'
        + ('' if SPEC['infoq'] else ' InfoQ：未執行（InfoQ 沒有此模型的設定；2026-10-29 使用者決定）。'),
        f'- 校準：同一個 1000 張的池（InfoQ 的選類規則，10 類 × 100 張 train），各方法取前 k 張：{CAL_N}；A8 ranges 用既有 10 張 validation 的 native ranges，不重算。',
        f'- 評估：ImageNet validation shard（{VAL_SHARD}）全部 3572 張，native 編譯評估；配置凍結後才評估。',
        '- 搜尋：共同 MILP（dense 模型的正式規則）；LP-first 在每個上限當對照；共同自動外殼 = mp_auto.auto_select。',
        '- 資源點（A）：由 BC 的 calibration-only 自動 frontier 決定：knee R_k；寬鬆 R_k + (1 − R_k)/2；激進 R_min + (R_k − R_min)/2；size 與 BOPs 都 ≤ R × all-INT8；精確上限寫入 caps.json 後才搜尋其他方法。',
        f'- 統計：每點 {len(others)} 個方法各與 BC 比 top-1（共 {3 * len(others)} 個比較），exact McNemar + Holm（α = 0.05）；配對 bootstrap {BOOT} 次 95% CI 只作描述；不設等效界線，未顯著＝未定。']
    if SPEC['matched_R']:
        lines.append(f'- 次要（C，預先登記）：每個方法的自動選點對「BC 在同一 R 下重新搜尋的配置」，{len(others)} 個比較另成一族做 Holm。')
    lines += ['- 計時：BC、SQNR、HTQ、HAWQ 交錯重複 3 次取中位數與範圍；InfoQ 至少 1 次；裝置不同時只作實測 pipeline 成本比較。',
              f'- 冒煙測試：{pl["smoke"]}', '']
    (run / 'plan.md').write_text('\n'.join(lines) + '\n')


# --------------------------------------------------------------------------- #
# assets: evaluation shard and calibration pool
# --------------------------------------------------------------------------- #
def assets(run):
    import pyarrow.parquet as pq
    import eval_random_mixed_precision as erm
    check_plan(run)
    if MODEL != 'resnet18':
        return assets_reuse(run)
    t0 = time.perf_counter()
    meta = read(MODEL_DIR / 'metadata.json')
    # evaluation: every image of the new validation shard, bytes untouched
    vdir = run / 'assets/eval_images'
    vdir.mkdir(parents=True, exist_ok=True)
    vpath = run / 'assets/val_shard' / VAL_SHARD
    tab = pq.ParquetFile(vpath).read()
    labels, imgs = tab.column('label').to_pylist(), tab.column('image').to_pylist()
    entries = []
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        b = im['bytes']
        f = f'val1_{i:05d}.JPEG'
        (vdir / f).write_bytes(b)
        entries.append({'file': f, 'original_name': im.get('path'), 'label': int(lab), 'row': i, 'sha256': hashlib.sha256(b).hexdigest()})
    man = {'source': f'benjamin-paine/imagenet-1k {VAL_SHARD}', 'parquet_sha256': sha(vpath), 'num_images': len(entries),
           'images': entries}
    write(vdir / 'manifest.json', man)
    x = erm.preprocess_by_source(vdir, man, meta)
    np.save(run / 'assets/eval_inputs.npy', x)
    np.save(run / 'assets/eval_labels.npy', np.array(labels, np.int64))
    t_eval = time.perf_counter() - t0
    # calibration pool: InfoQ's class / sample selection on the full train label order
    t1 = time.perf_counter()
    foot = read(run / 'assets/hf256_train_footer.json')
    offsets = np.cumsum([0] + [s['num_rows'] for s in foot['shards']])
    np.random.seed(INFOQ_SEED)
    all_labels = list(range(1000))      # every class is present (footer label ranges cover 0..999 contiguously)
    sel = np.random.choice(all_labels, size=INFOQ_NCLS, replace=False).tolist()
    if sel != foot['selected']:
        raise SystemExit('class selection differs from the footer probe')
    loaded = {}
    picks = {}
    for c in sel:
        rows = []
        for s in foot['shards']:
            if s['label_min'] <= c <= s['label_max']:
                k = s['shard']
                if k not in loaded:
                    p = run / f'assets/hf256_train/data/train-{k:05d}-of-00040.parquet'
                    loaded[k] = pq.ParquetFile(p).read(columns=['label']).column('label').to_numpy()
                rows += (offsets[k] + np.nonzero(loaded[k] == c)[0]).tolist()
        idx = sorted(rows)
        np.random.shuffle(idx)          # the original shuffles a list copy of the class's global indices
        picks[c] = idx[:INFOQ_PER_CLASS]
    order = [(c, picks[c][r]) for r in range(INFOQ_PER_CLASS) for c in sel]
    pdir = run / 'assets/pool_images'
    pdir.mkdir(parents=True, exist_ok=True)
    by_shard = {}
    for pos, (c, g) in enumerate(order):
        k = int(np.searchsorted(offsets, g, side='right') - 1)
        by_shard.setdefault(k, []).append((pos, c, g, int(g - offsets[k])))
    pent = [None] * len(order)
    for k, items in by_shard.items():
        t = pq.ParquetFile(run / f'assets/hf256_train/data/train-{k:05d}-of-00040.parquet').read()
        col, lab = t.column('image'), t.column('label').to_numpy()
        for pos, c, g, r in items:
            if int(lab[r]) != c:
                raise SystemExit('label mismatch in the pool')
            b = col[r].as_py()['bytes']
            ext = '.png' if b[:8] == b'\x89PNG\r\n\x1a\n' else '.JPEG'
            f = f'pool_{pos:04d}{ext}'
            (pdir / f).write_bytes(b)
            pent[pos] = {'file': f, 'label': c, 'global_index': int(g), 'shard': k, 'row': r, 'sha256': hashlib.sha256(b).hexdigest(),
                         'class_rank': pos // INFOQ_NCLS}
    pman = {'source': f'{HF256} train', 'selected_classes': sel, 'seed': INFOQ_SEED, 'images': pent,
            'shard_sha256': {k: sha(run / f'assets/hf256_train/data/train-{k:05d}-of-00040.parquet') for k in sorted(by_shard)}}
    write(pdir / 'manifest.json', pman)
    px = erm.preprocess_by_source(pdir, pman, meta)
    np.save(run / 'assets/pool_inputs.npy', px)
    np.save(run / 'assets/pool_labels.npy', np.array([e['label'] for e in pent], np.int64))
    t_pool = time.perf_counter() - t1
    write(run / 'assets/assets_freeze.json', {'time': now(), 'eval_manifest_sha256': sha(vdir / 'manifest.json'),
                                              'eval_inputs_sha256': sha(run / 'assets/eval_inputs.npy'), 'eval_images': len(entries),
                                              'pool_manifest_sha256': sha(pdir / 'manifest.json'),
                                              'pool_inputs_sha256': sha(run / 'assets/pool_inputs.npy'), 'pool_images': len(pent),
                                              'eval_prepare_s': t_eval, 'pool_prepare_s': t_pool})
    timing(run, 'assets_eval', 'common', t_eval, images=len(entries))
    timing(run, 'assets_pool', 'common', t_pool, images=len(pent))
    log(f'assets: eval {len(entries)} images ({t_eval:.0f}s), pool {len(pent)} images classes {sel} ({t_pool:.0f}s)')


def assets_reuse(run):
    """the ResNet18 run's evaluation images and calibration pool (symlinked, every image's sha256 checked against its
    manifest), preprocessed with this model's own config."""
    import eval_random_mixed_precision as erm
    meta = read(MODEL_DIR / 'metadata.json')
    out = {'time': now(), 'source_run': rel(R18_RUN)}
    for kind, d, inp, lab in (('eval', 'eval_images', 'eval_inputs.npy', 'eval_labels.npy'),
                              ('pool', 'pool_images', 'pool_inputs.npy', 'pool_labels.npy')):
        t0 = time.perf_counter()
        src = R18_RUN / 'assets' / d
        dst = run / 'assets' / d
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
        man = read(dst / 'manifest.json')
        bad = [e['file'] for e in man['images'] if hashlib.sha256((dst / e['file']).read_bytes()).hexdigest() != e['sha256']]
        if bad:
            raise SystemExit(f'{kind}: {len(bad)} images differ from the manifest')
        x = erm.preprocess_by_source(dst, man, meta)
        np.save(run / 'assets' / inp, x)
        y = np.load(R18_RUN / 'assets' / lab)
        if not np.array_equal(y, np.array([e['label'] for e in man['images']], np.int64)):
            raise SystemExit(f'{kind}: labels differ from the manifest')
        np.save(run / 'assets' / lab, y)
        dt = time.perf_counter() - t0
        out.update({f'{kind}_manifest_sha256': sha(dst / 'manifest.json'), f'{kind}_inputs_sha256': sha(run / 'assets' / inp),
                    f'{kind}_images': len(man['images']), f'{kind}_prepare_s': dt})
        timing(run, f'assets_{kind}', 'common', dt, images=len(man['images']))
    write(run / 'assets/assets_freeze.json', out)
    log(f'assets reused from {rel(R18_RUN)}: eval {out["eval_images"]}, pool {out["pool_images"]} images')


# --------------------------------------------------------------------------- #
# controls: model weights, logits, preprocessing (CPU)
# --------------------------------------------------------------------------- #
def tv_model(pretrained=True):
    """the model's torch definition (pretrained: the common checkpoint; else freshly initialized, the caller seeds)."""
    import torch
    src, arch = SPEC['torch']
    if src == 'torchvision':
        import torchvision
        m = getattr(torchvision.models, arch)(weights=None)
        if pretrained:
            m.load_state_dict(torch.load(TV_CKPT, map_location='cpu'))
        return m.eval()
    import timm
    from safetensors.torch import load_file
    m = timm.create_model(arch, pretrained=False)
    if pretrained:
        m.load_state_dict(load_file(str(torch_ckpt())))
    return m.eval()


def bn_of(model):
    """conv module name -> the BatchNorm2d module that directly follows it in definition order (None: no BN, e.g. SE
    convolutions, the classifier); ResNet18: conv -> bn, downsample.0 -> downsample.1."""
    import torch
    mods = list(model.named_modules())
    out = {}
    for i, (n, m) in enumerate(mods):
        if isinstance(m, torch.nn.Conv2d):
            nxt = mods[i + 1][1] if i + 1 < len(mods) else None
            out[n] = mods[i + 1][0] if isinstance(nxt, torch.nn.BatchNorm2d) else None
    return out


def controls(run):
    import onnx
    import onnxruntime as ort
    import torch
    from onnx import numpy_helper
    import eval_random_mixed_precision as erm
    check_plan(run)
    graph, nodes, names, elems, macs, module = graph_info()
    inits = {i.name: numpy_helper.to_array(i) for i in graph.graph.initializer}
    m = tv_model()
    sd = dict(m.named_modules())
    bns = bn_of(m)
    wd = bd = 0.0
    for n in names:
        mod = sd[module[n]]
        W = mod.weight.detach().double().numpy()
        if bns.get(module[n]) is None:
            Wf, bf = W, mod.bias.detach().double().numpy()
        else:
            bn = sd[bns[module[n]]]
            s = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().double().numpy()
            Wf = W * s[:, None, None, None]
            bf = bn.bias.detach().double().numpy() - bn.running_mean.detach().double().numpy() * s
        node = next(x for x in graph.graph.node if x.name == n)
        wd = max(wd, float(np.abs(Wf - inits[node.input[1]]).max()))
        bd = max(bd, float(np.abs(bf - inits[node.input[2]]).max()))
    x = np.load(run / 'assets/pool_inputs.npy', mmap_mode='r')[:10]
    with torch.no_grad():
        zt = m(torch.from_numpy(np.ascontiguousarray(x))).numpy()
    sess = ort.InferenceSession(str(MODEL_DIR / 'model.onnx'), providers=['CPUExecutionProvider'])
    zo = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: np.ascontiguousarray(x[i:i + 1])})[0] for i in range(len(x))])
    # preprocessing: the existing native calibration inputs from their source images
    src = Path('/home/hoju/test/HTQ/imagenet1k_2050/images')
    man = read(src / 'manifest.json')
    cal = [e for e in man['images'] if e['calibration']][:10]
    xr = erm.preprocess_by_source(src, {'images': cal}, read(MODEL_DIR / 'metadata.json'))
    ref = np.load(NATIVE_REF / 'calib_inputs.npy')
    out = {'bn_fold_weight_max_abs': wd, 'bn_fold_bias_max_abs': bd,
           'bn_of_weight_node': {n: bns.get(module[n]) for n in names},
           'torch_vs_ort_logits_max_abs': float(np.abs(zt - zo).max()),
           'torch_vs_ort_top1_agree': int((zt.argmax(1) == zo.argmax(1)).sum()), 'preprocess_reproduces_native_calib':
           bool(np.array_equal(xr, ref)), 'preprocess_max_abs': float(np.abs(xr - ref).max())}
    out['passed'] = bool(wd < 1e-5 and bd < 1e-5 and out['torch_vs_ort_logits_max_abs'] < 1e-3 and out['torch_vs_ort_top1_agree'] == 10
                         and out['preprocess_reproduces_native_calib'])
    write(run / 'controls_model.json', out)
    log(f'controls: {out}')
    if not out['passed']:
        raise SystemExit('model controls failed')


# --------------------------------------------------------------------------- #
# BC: native weights pass + grads statistics (the part of run_native_bc_pipeline.native_tables that B needs)
# --------------------------------------------------------------------------- #
def native_b_table(work, inputs, labels):
    from concurrent.futures import ThreadPoolExecutor
    import run_native_bc_pipeline as nbp
    erm = nbp.erm
    tools = erm.Tools()
    sys.path.insert(0, str(tools.lib))
    work.mkdir(parents=True, exist_ok=True)
    graph, nodes, names, elems, macs, module = graph_info()
    tm, t_all = {}, time.perf_counter()
    budget = nbp.MemoryBudget(0.8 * nbp.mem_available_gb())

    def stage(name, fn, *a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        tm[f'{name}_s'] = time.perf_counter() - t0
        return r
    text, base = work / f'{erm.SYMBOL_POSTFIX}.onnx.mlir', work / f'{erm.SYMBOL_POSTFIX}.mlirbc'
    stage('import', erm.run, [tools.onnx_mlir, '--EmitONNXBasic', '-o', work / erm.SYMBOL_POSTFIX,
                              '--mlir-elide-resource-strings-if-larger=1000000000',
                              '--mlir-elide-elementsattrs-if-larger=1000000000', MODEL_DIR / 'model.onnx'])
    stage('to_bytecode', erm.run, [tools.opt, text, '--emit-bytecode', '-o', base])
    text.unlink()
    inputs_path = work / 'calib_inputs.npy'
    np.save(inputs_path, inputs)
    onehot = work / 'stats_grads' / 'onehot'
    onehot.mkdir(parents=True, exist_ok=True)
    for i, y in enumerate(labels):
        np.save(onehot / f'{i}.npy', np.eye(1000, dtype=np.float32)[[int(y)]])
    levels = nbp.STATS_LEVELS['conv']

    def weights():
        wjson = work / 'weight_err.json'
        with budget.hold(nbp.build_gb(base)):
            erm.run([tools.opt, base, '--mixedp-collect-stats', '--emit-bytecode', '-o', work / 'unchanged.mlirbc'],
                    env={'MIXEDP_STATS_MODE': 'weights', 'MIXEDP_WEIGHT_PER_CHANNEL': '1',
                         'MIXEDP_STATS_FORMATS': ','.join(WFMT), 'MIXEDP_STATS_LAYOUT': str(wjson)})
        (work / 'unchanged.mlirbc').unlink()
        return {e['node']: e for e in json.loads(wjson.read_text())['nodes']}

    def grads_chain(runs):
        so, lay, _ = stage('grads_build', nbp.stats_build, tools, work, base, 'grads',
                           {'MIXEDP_STATS_MODE': 'grads', 'MIXEDP_STATS_NODES': ','.join(names)}, levels['grads'], budget)
        v, info = stage('grads_runs', nbp.stats_runs, runs, tools, so, lay, inputs_path, len(inputs), budget, onehot)
        tm['grads_run_s_per_input'] = info['run_s_per_input']
        return lay, v
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as runs, ThreadPoolExecutor(max_workers=2) as chains:
        fw = chains.submit(stage, 'weights', weights)
        fg = chains.submit(grads_chain, runs)
        werr, (lay, vg) = fw.result(), fg.result()
    t0 = time.perf_counter()
    B = {}
    for w in lay['weights']:
        n = w['node']
        G = vg[:, w['offset']:w['offset'] + w['channels']].sum(0) / w['elements_per_channel']
        B[n] = {f: float((G * np.array([np.inf if v is None else v for v in werr[n]['err2'][f]])).sum()) for f in WFMT}
    tm['table_s'] = time.perf_counter() - t0
    tm['total_s'] = time.perf_counter() - t_all
    return B, tm


def score_bc(run, rep=1, control=False):
    check_plan(run)
    if control:
        x, y = np.load(NATIVE_REF / 'calib_inputs.npy'), None
        src = Path('/home/hoju/test/HTQ/imagenet1k_2050/images')
        y = np.array([e['label'] for e in read(src / 'manifest.json')['images'] if e['calibration']][:10])
        B, tm = native_b_table(run / 'work/bc_control', x, y)
        ref = read(NATIVE_REF / 'table.json')['B']
        diff = max(abs(B[n][f] - ref[n][f]) / max(abs(ref[n][f]), 1e-300) for n in B for f in WFMT)
        out = {'max_rel_vs_existing_table': diff, 'nodes': len(B), 'passed': bool(diff <= 1e-9), 'timing': tm}
        write(run / 'controls_bc_reconstruction.json', out)
        log(f'BC reconstruction control: max rel {diff:.2e}')
        if not out['passed']:
            raise SystemExit('BC reconstruction control failed')
        return
    x = np.load(run / 'assets/pool_inputs.npy', mmap_mode='r')[:CAL_N['BC']]
    y = np.load(run / 'assets/pool_labels.npy')[:CAL_N['BC']]
    B, tm = native_b_table(run / f'work/bc_rep{rep}', np.ascontiguousarray(x), y)
    write(run / f'scores/BC_rep{rep}.json', {'method': 'BC', 'cost': B, 'calibration_images': CAL_N['BC'], 'timing': tm,
                                            'device': 'CPU (onnx-mlir native statistics, 12 cores)'})
    timing(run, 'score', 'BC', tm['total_s'], rep=rep, detail=tm)
    log(f'BC rep {rep}: {tm["total_s"]:.1f}s')


# --------------------------------------------------------------------------- #
# SQNR: model-output logit noise of single-node QDQ models (onnxruntime, CPU)
# --------------------------------------------------------------------------- #
def score_sqnr(run, rep=1):
    import eval_random_mixed_precision as erm
    from run_w8_per_channel_check import per_channel_int
    from sqnr_tables import sqnr_table
    check_plan(run)
    t0 = time.perf_counter()
    original = erm.np_fake_quant_weight
    erm.np_fake_quant_weight = lambda w, fmt, oracle: (per_channel_int(w, int(fmt[3:])) if fmt.startswith('int')
                                                        else original(w, fmt, oracle))
    graph, nodes, names, elems, macs, module = graph_info()
    ranges = tensor_ranges(nodes)
    x = np.ascontiguousarray(np.load(run / 'assets/pool_inputs.npy', mmap_mode='r')[:CAL_N['SQNR']])
    t_prep = time.perf_counter() - t0
    t1 = time.perf_counter()
    table = sqnr_table(graph, nodes, WFMT, AFMT, ranges, x, None, log=lambda m: None)
    t_tab = time.perf_counter() - t1
    erm.np_fake_quant_weight = original
    cost = {n: {w: table[n][f'{w}|int8'] for w in WFMT} for n in names}
    if any(v is None or not np.isfinite(v) for r in cost.values() for v in r.values()):
        raise SystemExit('SQNR: non-finite option')
    tm = {'prepare_s': t_prep, 'table_s': t_tab, 'total_s': time.perf_counter() - t0, 'ort_runs': 1 + len(names) * len(WFMT)}
    write(run / f'scores/SQNR_rep{rep}.json', {'method': 'SQNR', 'cost': cost, 'calibration_images': CAL_N['SQNR'], 'timing': tm,
                                              'definition': 'sum over images of ||logits(node at (w, int8), rest FP32) - logits_fp32||^2',
                                              'device': 'CPU (onnxruntime)'})
    timing(run, 'score', 'SQNR', tm['total_s'], rep=rep, detail=tm)
    log(f'SQNR rep {rep}: {tm["total_s"]:.1f}s')


# --------------------------------------------------------------------------- #
# HTQ and HAWQ (GPU, HTQ venv)
# --------------------------------------------------------------------------- #
def per_channel_int_torch(w, bits):
    """run_w8_per_channel_check.per_channel_int in torch (float32 scale, round half away from zero)."""
    import torch
    qmax, qmin = (1 << (bits - 1)) - 1, -(1 << (bits - 1))
    flat = w.reshape(w.shape[0], -1).float()
    maxabs = flat.abs().max(1).values
    scale = torch.where(maxabs > 0, maxabs / qmax, torch.ones_like(maxabs))
    t = (flat / scale[:, None]).double()
    q = torch.clamp(torch.sign(t) * torch.floor(t.abs() + 0.5), qmin, qmax)
    return (q.float() * scale[:, None]).reshape(w.shape)


def module_map(run):
    return read(run / 'plan.json')['search_space']['module_of_node']


def score_htq(run, rep=1):
    import torch
    check_plan(run)
    sys.path.insert(0, str(HTQ_DIR))
    import sensitivity as htq_sens
    from utils.pareto_utils import compute_perturbation
    t0 = time.perf_counter()
    torch.manual_seed(HTQ_SEED)
    np.random.seed(HTQ_SEED)
    net0 = tv_model(pretrained=False).train().cuda()                   # HTQ main.py: pretrained=False (train mode)
    x = torch.from_numpy(np.array(np.load(run / 'assets/pool_inputs.npy', mmap_mode='r')[:CAL_N['HTQ']]))
    y = torch.from_numpy(np.load(run / 'assets/pool_labels.npy')[:CAL_N['HTQ']])
    torch.cuda.synchronize()
    t_prep = time.perf_counter() - t0
    t1 = time.perf_counter()
    scores, names = htq_sens.snip(net0, [(x, y)])
    torch.cuda.synchronize()
    t_snip = time.perf_counter() - t1
    t2 = time.perf_counter()
    m = tv_model().cuda()
    mods = dict(m.named_modules())
    s = dict(zip(names, scores))
    cost = {n: {w: float(compute_perturbation(mods[mm].weight, int(w[3:]), s[mm])) for w in WFMT} for n, mm in module_map(run).items()}
    torch.cuda.synchronize()
    t_tab = time.perf_counter() - t2
    tm = {'prepare_s': t_prep, 'sensitivity_s': t_snip, 'table_s': t_tab, 'total_s': time.perf_counter() - t0}
    write(run / f'scores/HTQ_rep{rep}.json', {'method': 'HTQ', 'cost': cost, 'snip': s, 'calibration_images': CAL_N['HTQ'],
                                             'timing': tm, 'device': torch.cuda.get_device_name(0)})
    timing(run, 'score', 'HTQ', tm['total_s'], rep=rep, detail=tm)
    log(f'HTQ rep {rep}: {tm["total_s"]:.1f}s')


def score_hawq(run, rep=1):
    import torch
    check_plan(run)
    sys.path.insert(0, str(PYHESSIAN_DIR))
    from pyhessian import hessian
    t0 = time.perf_counter()
    model = tv_model().cuda()
    x = torch.from_numpy(np.array(np.load(run / 'assets/pool_inputs.npy', mmap_mode='r')[:CAL_N['HAWQ']]))
    y = torch.from_numpy(np.load(run / 'assets/pool_labels.npy')[:CAL_N['HAWQ']])
    criterion = torch.nn.CrossEntropyLoss()
    mods = dict(model.named_modules())
    torch.cuda.synchronize()
    t_prep = time.perf_counter() - t0
    traces, info = {}, {}
    t1 = time.perf_counter()
    for n, mm in module_map(run).items():
        torch.manual_seed(HAWQ_SEED)                      # CLADO prep_hawq.estimate_trace
        np.random.seed(HAWQ_SEED)
        for p in model.parameters():
            p.requires_grad = False
        mods[mm].weight.requires_grad = True
        tl = time.perf_counter()
        if SPEC['hawq_batch']:                            # pyhessian dataloader mode: the same 64 images in batches,
            b = SPEC['hawq_batch']                         # Hv averaged by batch size (the Hessian of the mean CE)
            comp = hessian(model, criterion, dataloader=[(x[i:i + b], y[i:i + b]) for i in range(0, len(x), b)], cuda=True)
        else:
            comp = hessian(model, criterion, data=(x, y), cuda=True)
        res = comp.trace(maxIter=HAWQ_MAXITER, tol=HAWQ_TOL)
        torch.cuda.synchronize()
        traces[n] = [float(v) for v in res]
        info[n] = {'probes': len(res), 'converged': len(res) < HAWQ_MAXITER, 'seconds': time.perf_counter() - tl,
                   'mean': float(np.mean(res)), 'std': float(np.std(res))}
        del comp
        torch.cuda.empty_cache()
    t_hess = time.perf_counter() - t1
    t2 = time.perf_counter()
    cost, flags = {}, {}
    for n, mm in module_map(run).items():
        W = mods[mm].weight.detach()
        tr = float(np.mean(traces[n]))
        if not np.isfinite(tr):
            raise SystemExit(f'HAWQ: non-finite trace at {n}')
        if tr < 0:
            flags[n] = 'negative mean trace -> cost 0'
            tr = 0.0
        cost[n] = {w: tr / W.numel() * float(((per_channel_int_torch(W, int(w[3:])) - W) ** 2).sum()) for w in WFMT}
    torch.cuda.synchronize()
    t_tab = time.perf_counter() - t2
    tm = {'prepare_s': t_prep, 'hessian_s': t_hess, 'table_s': t_tab, 'total_s': time.perf_counter() - t0,
          'probes_total': sum(v['probes'] for v in info.values())}
    write(run / f'scores/HAWQ_rep{rep}.json', {'method': 'HAWQ', 'cost': cost, 'trace_samples': traces, 'trace_info': info,
                                              'flags': flags, 'calibration_images': CAL_N['HAWQ'], 'timing': tm,
                                              'pyhessian_mode': (f'dataloader, batches of {SPEC["hawq_batch"]}'
                                                                 if SPEC['hawq_batch'] else 'data (one batch)'),
                                              'device': torch.cuda.get_device_name(0)})
    timing(run, 'score', 'HAWQ', tm['total_s'], rep=rep, detail=tm)
    log(f'HAWQ rep {rep}: {tm["total_s"]:.1f}s, probes {tm["probes_total"]}, flags {flags}')


# --------------------------------------------------------------------------- #
# InfoQ (own venv; cnn_baselines_infoq.py)
# --------------------------------------------------------------------------- #
def score_infoq(run, rep=1):
    import subprocess
    check_plan(run)
    out = run / f'scores/InfoQ_rep{rep}.json'
    t0 = time.perf_counter()
    env = dict(os.environ, PYTHONPATH=str(INFOQ_DIR), CUDA_VISIBLE_DEVICES='0', HF_HUB_OFFLINE='1')
    extra = ['--max-layers', os.environ['CNN_SMOKE_INFOQ_LAYERS']] if os.environ.get('CNN_SMOKE_INFOQ_LAYERS') else []
    if not SPEC['infoq']:
        raise SystemExit(f'InfoQ has no configuration for {MODEL} (marked not run)')
    r = subprocess.run([str(INFOQ_PY), str(HERE / 'cnn_baselines_infoq.py'), '--run-dir', str(run), '--out', str(out),
                        '--model', MODEL] + extra,
                       cwd=str(run / 'work/infoq_cwd'), env=env, capture_output=True, text=True)
    (run / f'logs/infoq_rep{rep}.log').write_text(r.stdout + '\n' + r.stderr)
    if r.returncode != 0:
        raise SystemExit(f'InfoQ adapter failed ({r.returncode}); see logs/infoq_rep{rep}.log')
    wall = time.perf_counter() - t0
    d = read(out)
    d['timing']['wall_including_process_start_s'] = wall
    write(out, d)
    timing(run, 'score', 'InfoQ', wall, rep=rep, detail=d['timing'])
    log(f'InfoQ rep {rep}: {wall:.0f}s')


# --------------------------------------------------------------------------- #
# timing repetitions of the fast scorers (fresh processes, rotated order)
# --------------------------------------------------------------------------- #
GPU_PY = '/home/hoju/test/HTQ/.venv/bin/python'
CPU_PY = str(HERE / '../../venv/bin/python')


LONG_S, VERY_LONG_S = 600, 1800      # a scorer whose repetition 1 takes >= 10 min follows the InfoQ rule (amendment)


def reps_allowed(run, m):
    t = read(run / f'scores/{m}_rep1.json')['timing']['total_s']
    return 3 if t < LONG_S else 2 if t < VERY_LONG_S else 1


def timing_rep(run, rep):
    import subprocess
    check_plan(run)
    order = ['BC', 'SQNR', 'HTQ', 'HAWQ']
    k = (rep - 1) % len(order)
    order = order[k:] + order[:k]
    for m in order:
        if rep > reps_allowed(run, m):
            log(f'rep {rep}: {m} skipped (repetition 1 took >= {LONG_S if rep > 2 else VERY_LONG_S}s)')
            continue
        py = GPU_PY if m in ('HTQ', 'HAWQ') else CPU_PY
        t0 = time.perf_counter()
        r = subprocess.run([py, __file__, '--stage', f'score_{m.lower()}', '--run-dir', str(run), '--rep', str(rep),
                            '--model', MODEL],
                           capture_output=True, text=True)
        wall = time.perf_counter() - t0
        (run / f'logs/score_{m}_rep{rep}.log').write_text(r.stdout + '\n' + r.stderr)
        if r.returncode != 0:
            raise SystemExit(f'{m} rep {rep} failed; see logs/score_{m}_rep{rep}.log')
        timing(run, 'score_process_wall', m, wall, rep=rep, order=order)
        log(f'rep {rep}: {m} {wall:.1f}s (process wall)')


# --------------------------------------------------------------------------- #
# caps (BC frontier) and search
# --------------------------------------------------------------------------- #
def value_fn(cost):
    def v(n, w, a):
        if a != 'int8':
            return None
        if n in FIXED:
            return 0.0 if w == 'int8' else None
        return float(cost[n][w])
    return v


def node_list():
    graph, nodes, names, elems, macs, module = graph_info()
    return nodes, names, elems, macs


def full_cfg(formats, nodes):
    return {n['name']: formats.get(n['name'], {'w': None, 'a': 'fp32'}) for n in nodes}


def caps(run):
    from mp_auto import auto_select
    check_plan(run)
    if (run / 'caps.json').exists():
        raise SystemExit('caps exist')
    nodes, names, elems, macs = node_list()
    cost = read(run / 'scores/BC_rep1.json')['cost']
    t0 = time.perf_counter()
    sel = auto_select(nodes, WFMT, AFMT, value_fn(cost), elems, macs, **AUTO)
    t_auto = time.perf_counter() - t0
    rk = sel['selected']['resource']
    rmin = resources({n: {'w': 'int8' if n in FIXED else 'int4', 'a': 'int8'} for n in names}, names, elems, macs)['R']
    rl, ra = rk + (1 - rk) / 2, rmin + (rk - rmin) / 2
    ref_b, ref_o = sum(elems.values()), sum(64 * m for m in macs.values())
    out = {'time': now(), 'rule': read(run / 'plan.json')['resource_points_A']['rule'], 'R_min': rmin, 'R_knee_BC': rk,
           'bc_auto_selection_hash': hashlib.sha256(json.dumps(sel['selected']['formats'], sort_keys=True).encode()).hexdigest(),
           'points': {p: {'R': r, 'max_weight_bytes': r * ref_b, 'max_bops': r * ref_o} for p, r in
                      (('loose', rl), ('knee', rk), ('aggressive', ra))},
           'int8_reference': {'weight_bytes': ref_b, 'bops': ref_o}, 'bc_scores_sha256': sha(run / 'scores/BC_rep1.json'),
           'bc_auto_frontier': [{k: r[k] for k in ('size', 'bops', 'resource', 'objective', 'origin', 'solver_status')}
                                for r in sel['frontier']], 'auto_s': t_auto,
           'note': 'calibration-only BC frontier; written before any other method is searched and before any evaluation'}
    write(run / 'caps.json', out)
    log(f'caps: R_min {rmin:.4f}, aggressive {ra:.4f}, knee {rk:.4f}, loose {rl:.4f}')


def search(run):
    from mp_auto import auto_select
    from mp_milp import milp_select
    from run_olmoe_search_final import make_solver
    check_plan(run)
    cp = read(run / 'caps.json')
    nodes, names, elems, macs = node_list()
    rows = []
    for m in METHODS:
        sp = run / f'scores/{m}_rep1.json'
        cost = read(sp)['cost']
        if any(not np.isfinite(cost[n][w]) or cost[n][w] < 0 for n in names if n not in FIXED for w in WFMT):
            raise SystemExit(f'{m}: costs must be finite and nonnegative')
        v = value_fn(cost)
        for p, c in cp['points'].items():
            info = {}
            t0 = time.perf_counter()
            fmt, obj, status = milp_select(nodes, WFMT, AFMT, v, elems, macs, c['max_weight_bytes'], c['max_bops'],
                                           time_limit=MILP_TIME_LIMIT, info=info)
            t_one = time.perf_counter() - t0
            recs = []
            (run / 'work/lpfirst').mkdir(parents=True, exist_ok=True)
            lp = make_solver('lpfirst', recs, run / 'work/lpfirst', time_limit=MILP_TIME_LIMIT, tag=f'{m}_{p}')
            fmt_lp, obj_lp, st_lp = lp(nodes, WFMT, AFMT, v, elems, macs, c['max_weight_bytes'], c['max_bops'])
            res = resources(full_cfg(fmt, nodes), names, elems, macs)
            rows.append({'group': 'A', 'method': m, 'point': p, 'formats': {n: fmt[n] for n in names}, 'objective': obj,
                         'solver_status': status, 'solver_info': info, 'search_s': t_one, 'resources': res,
                         'valid': bool(res['weight_bytes'] <= c['max_weight_bytes'] * (1 + 1e-12) and res['bops'] <= c['max_bops'] * (1 + 1e-12)),
                         'caps': c, 'lpfirst_same': {n: fmt_lp[n] for n in names} == {n: fmt[n] for n in names},
                         'lpfirst_objective': obj_lp, 'lpfirst_status': st_lp, 'scores_sha256': sha(sp)})
        t0 = time.perf_counter()
        sel = auto_select(nodes, WFMT, AFMT, v, elems, macs, **AUTO)
        t_auto = time.perf_counter() - t0
        res = resources(full_cfg(sel['selected']['formats'], nodes), names, elems, macs)
        rows.append({'group': 'B', 'method': m, 'point': 'auto_knee', 'formats': {n: sel['selected']['formats'][n] for n in names},
                     'objective': sel['selected']['objective'], 'solver_status': sel['selected']['solver_status'], 'search_s': t_auto,
                     'resources': res, 'valid': True, 'knee': {'resource': sel['selected']['resource'], 'size': sel['selected']['size'],
                     'bops': sel['selected']['bops'], 'frontier_points': len(sel['frontier']), 'solves': len(sel['attempted_budgets'])},
                     'scores_sha256': sha(sp)})
        log(f'{m}: A ' + ', '.join(f"{r['point']} R={r['resources']['R']:.3f}" for r in rows if r['method'] == m and r['group'] == 'A')
            + f'; auto R={res["R"]:.3f} ({t_auto:.1f}s)')
    if SPEC['matched_R']:
        # C (pre-registered secondary): BC re-searched at the exact R of every other method's auto pick
        sp = run / 'scores/BC_rep1.json'
        v = value_fn(read(sp)['cost'])
        ref_b, ref_o = sum(elems.values()), sum(64 * m for m in macs.values())
        for b in [r for r in rows if r['group'] == 'B' and r['method'] != 'BC']:
            R = b['resources']['R']
            c = {'R': R, 'max_weight_bytes': R * ref_b, 'max_bops': R * ref_o}
            t0 = time.perf_counter()
            fmt, obj, status = milp_select(nodes, WFMT, AFMT, v, elems, macs, c['max_weight_bytes'], c['max_bops'],
                                           time_limit=MILP_TIME_LIMIT)
            res = resources(full_cfg(fmt, nodes), names, elems, macs)
            rows.append({'group': 'C', 'method': 'BC', 'point': f'R_of_{b["method"]}', 'formats': {n: fmt[n] for n in names},
                         'objective': obj, 'solver_status': status, 'search_s': time.perf_counter() - t0, 'resources': res,
                         'valid': bool(res['weight_bytes'] <= c['max_weight_bytes'] * (1 + 1e-12) and
                                       res['bops'] <= c['max_bops'] * (1 + 1e-12)), 'caps': c, 'scores_sha256': sha(sp)})
        log('C: ' + ', '.join(f"{r['point']} R={r['resources']['R']:.3f}" for r in rows if r['group'] == 'C'))
    refs = {'FP32': None, 'INT8_ref': {n: {'w': 'int8', 'a': 'int8'} for n in names}}
    uniq = {}
    for r in rows:
        if not r['valid']:
            raise SystemExit(f'invalid configuration {r["method"]} {r["point"]}')
        h = hashlib.sha256(json.dumps(r['formats'], sort_keys=True).encode()).hexdigest()[:16]
        r['config_id'] = h
        uniq.setdefault(h, r['formats'])
    uniq['INT8_ref'] = refs['INT8_ref']
    write(run / 'configs.json', {'time': now(), 'rows': rows, 'unique': uniq, 'fp32': True,
                                 'int8_reference_resources': resources(full_cfg(refs['INT8_ref'], nodes), names, elems, macs)})
    write(run / 'configs_freeze.json', {'time': now(), 'configs_sha256': sha(run / 'configs.json'), 'caps_sha256': sha(run / 'caps.json'),
                                        'scores_sha256': {m: sha(run / f'scores/{m}_rep1.json') for m in METHODS}})
    log(f'search: {len(rows)} rows, {len(uniq)} unique configurations (+FP32)')


# --------------------------------------------------------------------------- #
# native compile + evaluation of the frozen configurations (CPU)
# --------------------------------------------------------------------------- #
def evaluate(run, chunks=12):
    import run_native_bc_pipeline as nbp
    from evaluate_native_search_chunked import chunked_run_so
    erm = nbp.erm
    check_plan(run)
    fz = read(run / 'configs_freeze.json')
    if sha(run / 'configs.json') != fz['configs_sha256']:
        raise SystemExit('configs changed after the freeze')
    tools = erm.Tools()
    sys.path.insert(0, str(tools.lib))
    work = run / 'work/eval'
    work.mkdir(parents=True, exist_ok=True)
    if not (work / f'{erm.SYMBOL_POSTFIX}.mlirbc').exists():
        erm.run([tools.onnx_mlir, '--EmitONNXBasic', '-o', work / erm.SYMBOL_POSTFIX, '--mlir-elide-resource-strings-if-larger=1000000000',
                 '--mlir-elide-elementsattrs-if-larger=1000000000', MODEL_DIR / 'model.onnx'])
        erm.run([tools.opt, work / f'{erm.SYMBOL_POSTFIX}.onnx.mlir', '--emit-bytecode', '-o', work / f'{erm.SYMBOL_POSTFIX}.mlirbc'])
        (work / f'{erm.SYMBOL_POSTFIX}.onnx.mlir').unlink()
    nodes, names, elems, macs = node_list()
    ranges = tensor_ranges(nodes)
    labels = np.load(run / 'assets/eval_labels.npy')
    cfgs = dict(read(run / 'configs.json')['unique'])
    todo = [('FP32', None)] + sorted(cfgs.items())
    os.environ['MIXEDP_WEIGHT_PER_CHANNEL'] = '1'
    for cid, fmt in todo:
        d = run / f'eval/{cid}'
        if (d / 'result.json').exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        env = None if fmt is None else erm.node_formats_env(full_cfg(fmt, nodes), nodes, ranges)
        t0 = time.perf_counter()
        so, info = erm.build_so(tools, d, cid, nbp.imported_model(work), env, None, opt='O3', krnl_opt='O3',
                                constants_to_file=True, bytecode=True)
        t_comp = time.perf_counter() - t0
        if info['warnings']:
            raise SystemExit(f'{cid}: compile warnings {info["warnings"]}')
        for p in d.iterdir():
            if p.suffix in ('.mlir', '.mlirbc', '.ll', '.bc'):
                p.unlink()
        z, ev = chunked_run_so(chunks)(tools, so, run / 'assets/eval_inputs.npy', d / 'logits.npy', 36000)
        if not np.isfinite(z).all():
            raise SystemExit(f'{cid}: non-finite logits')
        loss = nbp.cross_entropy(z, labels)
        top5 = np.argsort(-z, 1)[:, :5]
        np.savez_compressed(d / 'per_image.npz', loss=loss, pred=z.argmax(1), top5=top5,
                            correct1=(z.argmax(1) == labels), correct5=(top5 == labels[:, None]).any(1))
        np.save(d / 'logits.npy', z.astype(np.float32))
        art = {'so_bytes': so.stat().st_size, 'constants_bytes': sum(p.stat().st_size for p in d.glob('*.constants.bin'))}
        res = {'config_id': cid, 'env': env, 'top1': float((z.argmax(1) == labels).mean()), 'top5': float((top5 == labels[:, None]).any(1).mean()),
               'loss': float(loss.mean()), 'n': int(len(labels)), 'compile_s': t_comp, 'eval': ev, 'artifact': art,
               'artifact_note': 'FP32 + QDQ library and FP32 constants; not a packed low-bit model size'}
        write(d / 'result.json', res)
        timing(run, 'compile', cid, t_comp)
        timing(run, 'final_eval', cid, ev['wall_seconds'], images=int(len(labels)), processes=ev['processes'])
        log(f'eval {cid}: top1 {res["top1"]:.4f} loss {res["loss"]:.4f} (compile {t_comp:.0f}s, eval {ev["wall_seconds"]:.0f}s)')
    # control: one configuration evaluated again through the common entry (other chunking) must give identical logits
    d = run / 'eval/INT8_ref'
    so = next(d.glob('*.so'))
    z2, _ = chunked_run_so(max(1, chunks // 2))(tools, so, run / 'assets/eval_inputs.npy', run / 'work/eval/int8_again.npy', 36000)
    same = bool(np.array_equal(z2, np.load(d / 'logits.npy')))
    write(run / 'controls_eval.json', {'int8_ref_reevaluated_identical': same, 'chunks': [chunks, max(1, chunks // 2)]})
    (run / 'work/eval/int8_again.npy').unlink()
    log(f'evaluation control: identical re-evaluation {same}')


# --------------------------------------------------------------------------- #
# analysis (CPU)
# --------------------------------------------------------------------------- #
COLORS = {'BC': '#2a78d6', 'SQNR': '#eb6834', 'HTQ': '#1baf7a', 'HAWQ': '#eda100', 'InfoQ': '#e87ba4'}   # dataviz slots 1-5
MARKERS = {'BC': 'o', 'SQNR': 's', 'HTQ': '^', 'HAWQ': 'D', 'InfoQ': 'P'}
SAME_CAP_MS = {'BC': 15, 'SQNR': 12, 'HTQ': 10, 'HAWQ': 7.5, 'InfoQ': 9}


def holm(p):
    order = np.argsort(p)
    adj, run_max = np.empty(len(p)), 0.0
    for k, i in enumerate(order):
        run_max = max(run_max, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run_max
    return adj


def boot_ci(d, rng_seed=BOOT_SEED, n=BOOT, chunk=1000):
    rng = np.random.default_rng(rng_seed)
    N = len(d)
    means = []
    for s in range(0, n, chunk):
        idx = rng.integers(0, N, (min(chunk, n - s), N))
        means.append(d[idx].mean(1))
    return np.percentile(np.concatenate(means), [2.5, 97.5]).tolist()


def score_time_summary(run):
    out = {}
    for m in METHODS:
        reps = sorted(run.glob(f'scores/{m}_rep*.json'))
        tt = [read(p)['timing']['total_s'] for p in reps]
        walls = [json.loads(l)['seconds'] for l in open(run / 'timing.jsonl') if json.loads(l)['stage'] in
                 ('score_process_wall',) and json.loads(l)['method'] == m]
        if m == 'InfoQ':
            walls = [read(p)['timing'].get('wall_including_process_start_s', np.nan) for p in reps]
        same = None
        if len(reps) > 1:
            c0 = read(reps[0])['cost']
            same = max(abs(read(p)['cost'][n][w] - c0[n][w]) / max(abs(c0[n][w]), 1e-300) for p in reps[1:] for n in c0 for w in WFMT)
        out[m] = {'reps': len(reps), 'table_total_s': tt, 'median_s': float(np.median(tt)) if tt else None,
                  'range_s': [float(min(tt)), float(max(tt))] if tt else None, 'process_wall_s': walls,
                  'median_wall_s': float(np.median(walls)) if walls else None, 'max_rel_table_diff_vs_rep1': same,
                  'detail_rep1': read(reps[0])['timing'] if reps else None}
    return out


def analyze(run):
    from scipy.stats import binomtest
    check_plan(run)
    cf = read(run / 'configs.json')
    rows = cf['rows']
    nodes, names, elems, macs = node_list()
    pi = {cid: np.load(run / f'eval/{cid}/per_image.npz') for cid in list(cf['unique']) + ['FP32']}
    er = {cid: read(run / f'eval/{cid}/result.json') for cid in pi}
    N = len(pi['FP32']['loss'])
    fp = er['FP32']
    tsum = score_time_summary(run)
    caps = read(run / 'caps.json')
    # table 1: same caps
    t1, tests = [], []
    for p in ('loose', 'knee', 'aggressive'):
        bc = next(r for r in rows if r['group'] == 'A' and r['method'] == 'BC' and r['point'] == p)
        for m in METHODS:
            r = next(x for x in rows if x['group'] == 'A' and x['method'] == m and x['point'] == p)
            a, b = pi[r['config_id']], pi[bc['config_id']]
            d1 = a['correct1'].astype(float) - b['correct1'].astype(float)
            dl = a['loss'] - b['loss']
            row = {'point': p, 'method': m, 'config_id': r['config_id'], 'same_config_as_BC': r['config_id'] == bc['config_id'],
                   'top1': er[r['config_id']]['top1'], 'top5': er[r['config_id']]['top5'], 'loss': er[r['config_id']]['loss'],
                   'drop_vs_fp32_pp': 100 * (fp['top1'] - er[r['config_id']]['top1']), **{k: r['resources'][k] for k in
                   ('weight_bytes', 'weight_MB', 'weight_MiB', 'bops', 'GBOPs', 'size_ratio', 'bops_ratio', 'R')},
                   'cap_R': r['caps']['R'], 'size_util': r['resources']['weight_bytes'] / r['caps']['max_weight_bytes'],
                   'bops_util': r['resources']['bops'] / r['caps']['max_bops'], 'w_hist': r['resources']['w_hist'],
                   'table_time_median_s': tsum[m]['median_s'], 'search_s': r['search_s'], 'lpfirst_same': r['lpfirst_same']}
            if m != 'BC':
                bw, cw = int(((a['correct1']) & (~b['correct1'])).sum()), int(((~a['correct1']) & (b['correct1'])).sum())
                pv = binomtest(bw, bw + cw, 0.5).pvalue if bw + cw else 1.0
                row.update({'d_top1_pp_vs_BC': 100 * d1.mean(), 'd_top1_ci_pp': [100 * v for v in boot_ci(d1)],
                            'd_loss_vs_BC': float(dl.mean()), 'd_loss_ci': boot_ci(dl), 'mcnemar_method_right_BC_wrong': bw,
                            'mcnemar_method_wrong_BC_right': cw, 'mcnemar_p': pv})
                tests.append(row)
            t1.append(row)
    adj = holm(np.array([r['mcnemar_p'] for r in tests]))
    for r, a in zip(tests, adj):
        r['holm_p'] = float(a)
        r['verdict'] = ('method better' if r['d_top1_pp_vs_BC'] > 0 else 'BC better') if a < 0.05 else 'undetermined'
    # table 2: common automatic shell
    t2 = []
    for m in METHODS:
        r = next(x for x in rows if x['group'] == 'B' and x['method'] == m)
        cid = r['config_id']
        comp = er[cid]['compile_s']
        t2.append({'method': m, 'config_id': cid, 'top1': er[cid]['top1'], 'top5': er[cid]['top5'], 'loss': er[cid]['loss'],
                   'drop_vs_fp32_pp': 100 * (fp['top1'] - er[cid]['top1']), **{k: r['resources'][k] for k in
                   ('weight_MB', 'weight_MiB', 'GBOPs', 'size_ratio', 'bops_ratio', 'R')}, 'w_hist': r['resources']['w_hist'],
                   'knee': r['knee'], 'table_time_median_s': tsum[m]['median_s'], 'auto_search_s': r['search_s'],
                   'compile_s': comp, 'offline_total_s': (tsum[m]['median_wall_s'] or tsum[m]['median_s']) + r['search_s'] + comp})
    # C (secondary, pre-registered where the plan has it): each other auto pick minus BC at the same R
    tc = []
    for r in [x for x in rows if x['group'] == 'C']:
        m = r['point'].removeprefix('R_of_')
        b = next(x for x in rows if x['group'] == 'B' and x['method'] == m)
        a, c = pi[b['config_id']], pi[r['config_id']]
        d1 = a['correct1'].astype(float) - c['correct1'].astype(float)
        dl = a['loss'] - c['loss']
        mw, cw = int((a['correct1'] & ~c['correct1']).sum()), int((~a['correct1'] & c['correct1']).sum())
        tc.append({'method': m, 'R_cap': r['caps']['R'], 'method_config_id': b['config_id'], 'bc_config_id': r['config_id'],
                   'same_config': b['config_id'] == r['config_id'],
                   'method_top1': er[b['config_id']]['top1'], 'bc_top1': er[r['config_id']]['top1'],
                   'method_top5': er[b['config_id']]['top5'], 'bc_top5': er[r['config_id']]['top5'],
                   'method_loss': er[b['config_id']]['loss'], 'bc_loss': er[r['config_id']]['loss'],
                   'method_resources': {k: b['resources'][k] for k in ('weight_MB', 'GBOPs', 'size_ratio', 'bops_ratio', 'R')},
                   'bc_resources': {k: r['resources'][k] for k in ('weight_MB', 'GBOPs', 'size_ratio', 'bops_ratio', 'R')},
                   'bc_w_hist': r['resources']['w_hist'], 'd_top1_pp': 100 * d1.mean(),
                   'd_top1_ci_pp': [100 * v for v in boot_ci(d1)], 'd_loss': float(dl.mean()), 'd_loss_ci': boot_ci(dl),
                   'mcnemar_method_right_BC_wrong': mw, 'mcnemar_method_wrong_BC_right': cw,
                   'mcnemar_p': binomtest(mw, mw + cw, 0.5).pvalue if mw + cw else 1.0})
    if tc:
        for r, a in zip(tc, holm(np.array([r['mcnemar_p'] for r in tc]))):
            r['holm_p'] = float(a)
            r['verdict'] = ('method better' if r['d_top1_pp'] > 0 else 'BC better') if a < 0.05 else 'undetermined'
    # resources in detail (every configuration, independently recomputed)
    import onnx
    from onnx import numpy_helper
    graph = onnx.load(str(MODEL_DIR / 'model.onnx'))
    inits = {i.name: numpy_helper.to_array(i) for i in graph.graph.initializer}
    byn = {n.name: n for n in graph.graph.node}
    bias_bytes = sum(inits[byn[n].input[2]].size * 4 for n in names)
    scale_bytes = sum(inits[byn[n].input[1]].shape[0] * 4 for n in names)
    act_tensors = {t for n in nodes if n['name'] in names for t in (n['act_tensors'][0], n['act_tensors'][-1])}
    meta = {'fp32_bias_bytes': bias_bytes, 'per_channel_weight_scale_bytes_fp32': scale_bytes,
            'activation_qparam_bytes': len(act_tensors) * 5, 'activation_qparam_note': 'one FP32 scale + one int8 zero point per '
            'quantized activation tensor', 'weight_zero_points': 'symmetric weights: zero points are 0 and not stored'}
    res_rows = []
    for r in rows + [{'group': 'ref', 'method': 'INT8_ref', 'point': '-', 'config_id': 'INT8_ref', 'formats': cf['unique']['INT8_ref']}]:
        rr = resources(full_cfg(r['formats'], nodes), names, elems, macs)
        res_rows.append({'group': r['group'], 'method': r['method'], 'point': r['point'], 'config_id': r['config_id'], **{k: v for k, v in rr.items() if k != 'w_hist'},
                         'w_hist': json.dumps(rr['w_hist']), 'total_quantized_bytes': rr['weight_bytes'] + bias_bytes + scale_bytes + meta['activation_qparam_bytes'],
                         'formats': ' '.join(r['formats'][n]['w'] for n in names),
                         'so_bytes': er[r['config_id']]['artifact']['so_bytes'], 'constants_bytes': er[r['config_id']]['artifact']['constants_bytes']})
    out = {'units': 'top-1 / top-5 as fractions; differences in percentage points; loss = mean CE (nats); MB = 1e6 bytes',
           'n_eval_images': N, 'fp32': {k: fp[k] for k in ('top1', 'top5', 'loss')},
           'int8_ref': {k: er['INT8_ref'][k] for k in ('top1', 'top5', 'loss')}, 'caps': caps['points'], 'table1': t1, 'table2': t2,
           'score_timing': tsum, 'metadata_bytes': meta, 'holm_family': len(tests),
           'lpfirst_agreement': sum(r.get('lpfirst_same', False) for r in rows if r['group'] == 'A'),
           'controls': {k: read(run / f) for k, f in (('model', 'controls_model.json'), ('bc', 'controls_bc_reconstruction.json'),
                                                     ('eval', 'controls_eval.json')) if (run / f).exists()}}
    if tc:
        out['table_matched_R'] = tc
        out['holm_family_matched_R'] = len(tc)
    write(run / 'summary.json', out)
    import csv
    for name, rr in (('summary_table1.csv', t1), ('summary_table2.csv', t2), ('resources.csv', res_rows))\
            + ((('summary_table_matched_R.csv', tc),) if tc else ()):
        keys = list(dict.fromkeys(k for r in rr for k in r))
        with open(run / name, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rr])
    # BC at the R of other auto picks: table C, or (ResNet18) the follow-up run_cnn_baselines_matched.py
    extra = [(r['bc_resources'], r['bc_top1'], r['method']) for r in tc]
    fu = run / 'followup_matched/summary.json'
    if not tc and fu.exists():
        extra = [(r['bc_resources'], r['bc_top1'], r['method']) for r in read(fu)['rows']]
    plots(run, t1, t2, fp, er['INT8_ref'], cf, extra)
    log(f'analysis written; FP32 top1 {fp["top1"]:.4f}, INT8 ref {er["INT8_ref"]["top1"]:.4f}')


def plots(run, t1, t2, fp, i8, cf, extra=()):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    surf, ink, muted, grid = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
    nodes, names, elems, macs = node_list()
    i8r = resources(full_cfg(cf['unique']['INT8_ref'], nodes), names, elems, macs)
    for key, xlabel, fname in (('size_ratio', 'Weight size / all-INT8 weight size', 'top1_vs_size.png'),
                               ('bops_ratio', 'BOPs / all-INT8 BOPs', 'top1_vs_bops.png')):
        fig, ax = plt.subplots(figsize=(7.6, 5.4), facecolor=surf)
        ax.set_facecolor(surf)
        for m in METHODS:
            xs = [r[key] for r in t1 if r['method'] == m]
            ys = [100 * r['top1'] for r in t1 if r['method'] == m]
            # identical configurations coincide: concentric sizes keep every marker visible
            ax.plot(xs, ys, MARKERS[m], color=COLORS[m], ms=SAME_CAP_MS[m], mec=surf, mew=1.2, label=f'{m} (same caps)', zorder=3)
            b = next(r for r in t2 if r['method'] == m)
            ax.plot([b[key]], [100 * b['top1']], MARKERS[m], ms=9, mfc='none', mec=COLORS[m], mew=2, zorder=4,
                    label=f'{m} (auto shell)')
        if extra:
            ax.plot([r[key] for r, _, _ in extra], [100 * t for _, t, _ in extra], 'o', color=COLORS['BC'], ms=8,
                    mec=ink, mew=0.8, label='BC (at the R of an auto pick)', zorder=5)
            # labels stacked in a column right of the cluster, with leader lines (points may nearly coincide)
            (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
            ex = sorted(extra, key=lambda e: -e[1])
            xt = max(r[key] for r, _, _ in ex) + 0.08 * (x1 - x0)
            yc, dy = np.mean([100 * t for _, t, _ in ex]), 0.065 * (y1 - y0)
            for i, (r, t, m) in enumerate(ex):
                ax.annotate(f'BC @ R of {m}', (r[key], 100 * t), xytext=(xt, yc + ((len(ex) - 1) / 2 - i) * dy),
                            ha='left', va='center', fontsize=7, color=muted,
                            arrowprops={'arrowstyle': '-', 'color': muted, 'lw': 0.6, 'shrinkA': 0, 'shrinkB': 4})
        for p, (dx, dy, ha) in (('loose', (0, 12, 'center')), ('knee', (12, -14, 'left')), ('aggressive', (12, -3, 'left'))):
            grp = {}
            for r in t1:
                if r['point'] == p:
                    grp.setdefault(r['config_id'], []).append(r)
            for g in grp.values():
                if len(g) > 1:
                    ax.annotate(' = '.join(r['method'] for r in g) + ' (same config)', (g[0][key], 100 * g[0]['top1']),
                                textcoords='offset points', xytext=(dx, dy), ha=ha, fontsize=8, color=ink)
        ax.axhline(100 * fp['top1'], color=muted, lw=1, ls='--', zorder=1)
        ax.text(ax.get_xlim()[0], 100 * fp['top1'], ' FP32', va='bottom', ha='left', color=muted, fontsize=9)
        ax.plot([i8r[key]], [100 * i8['top1']], 'x', color=ink, ms=8, mew=2, label='all-INT8 reference', zorder=4)
        ax.set_xlabel(xlabel, color=ink)
        ax.set_ylabel('Top-1 accuracy (%)', color=ink)
        ax.grid(True, color=grid, lw=0.8)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted)
        ax.legend(fontsize=8, ncol=3, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.16))
        fig.tight_layout()
        fig.savefig(run / fname, dpi=150, facecolor=surf, bbox_inches='tight')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=surf)
    ax.set_facecolor(surf)
    for i, b in enumerate(sorted(t2, key=lambda b: b['offline_total_s'])):
        ax.plot([b['offline_total_s']], [100 * b['top1']], MARKERS[b['method']], color=COLORS[b['method']], ms=9, mec=surf, mew=1.5,
                label=b['method'], zorder=3)
        left = b['method'] == 'InfoQ'
        # alternate above / below so neighbours in time with similar accuracy do not overlap
        ax.annotate(f"{b['method']}  R={b['R']:.2f}", (b['offline_total_s'], 100 * b['top1']), textcoords='offset points',
                    xytext=(-7 if left else 7, 4 if i % 2 == 0 else -12), ha='right' if left else 'left', fontsize=8, color=ink)
    ax.set_xscale('log')
    ax.set_xlabel('Offline time to a configuration (s, log scale; table + auto search + compile)', color=ink)
    ax.set_ylabel('Top-1 accuracy of the auto-shell pick (%)', color=ink)
    ax.grid(True, color=grid, lw=0.8)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=muted)
    fig.tight_layout()
    fig.savefig(run / 'quality_vs_offline_time.png', dpi=150, facecolor=surf)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main():
    global LOGF
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, choices=('plan', 'assets', 'controls', 'score_bc', 'score_bc_control', 'score_sqnr',
                                                      'score_htq', 'score_hawq', 'score_infoq', 'timing', 'caps', 'search',
                                                      'evaluate', 'analyze'))
    ap.add_argument('--model', choices=tuple(MODELS), default='resnet18')
    ap.add_argument('--run-dir', type=Path, help='default: the model\'s run directory')
    ap.add_argument('--rep', type=int, default=1)
    ap.add_argument('--smoke-note')
    a = ap.parse_args()
    configure(a.model)
    run = (a.run_dir or RUN).resolve()
    for d in ('logs', 'scores', 'work', 'eval', 'work/infoq_cwd'):
        (run / d).mkdir(parents=True, exist_ok=True)
    LOGF = open(run / 'logs' / f'{a.stage}.log', 'a')
    t0 = time.time()
    if a.stage == 'plan':
        plan(run, a.smoke_note)
    elif a.stage == 'score_bc_control':
        score_bc(run, control=True)
    elif a.stage in ('score_bc', 'score_sqnr', 'score_htq', 'score_hawq', 'score_infoq'):
        globals()[a.stage](run, a.rep)
    elif a.stage == 'timing':
        timing_rep(run, a.rep)
    else:
        {'assets': assets, 'controls': controls, 'caps': caps, 'search': search, 'evaluate': evaluate, 'analyze': analyze}[a.stage](run)
    with open(run / 'stages.jsonl', 'a') as fh:
        fh.write(json.dumps({'stage': a.stage, 'rep': a.rep, 'seconds': time.time() - t0, 'time': now(),
                             'script_sha256': sha(Path(__file__)), 'argv': sys.argv}) + '\n')


if __name__ == '__main__':
    main()
