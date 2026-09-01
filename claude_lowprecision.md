ML-Guided Whole-Configuration Mixed Low-Precision Optimization on ONNX-MLIR

0. Relation to the Posit Variant

This document is the low-precision counterpart of `claude.md`. The Posit variant
(`claude.md`) is unchanged and remains implemented; it is not removed or
deprioritized, just parked. This variant targets the numeric formats already
implemented by the `--convert-onnx-to-lowprecision` pass and documented in
`docs/LowPrecisionFormats.md`: bf16, f16, int8, fp8e4m3, fp8e5m2, and FP32. The
research problem, optimization objectives, and overall Configuration
Generator / Accuracy Predictor / Cost Calculator / Pareto Selection loop are
the same as the Posit variant. What differs is everything downstream of "how
a chosen format is realized in the IR", because low-precision formats do not
introduce a dedicated dialect the way Posit does. Section 6 and Section 7
below are the sections that actually change; the rest largely carries over
with terminology swapped.

1. Project Overview

This project develops an ML-guided whole-configuration mixed low-precision
optimization framework on ONNX-MLIR. Given a trained ONNX model, the
framework assigns a numerical format to each supported ONNX operation. A
configuration may contain bf16, f16, int8, fp8e4m3, fp8e5m2, and FP32
simultaneously.

Conv_0    -> bf16
Relu_1    -> f16
MatMul_2  -> int8
Add_3     -> f16
Softmax_4 -> FP32

Candidate formats are bf16, f16, int8, fp8e4m3, fp8e5m2, and FP32 (6 formats).
Unlike the Posit format set, not every candidate format is legal on every op:
per `docs/LowPrecisionFormats.md`'s per-op scope table, `Conv`/`Gemm`/`MatMul`
accept all 5 non-FP32 formats, while `Relu`/`Add`/`Sub`/`Mul`/`Div` only
accept bf16/f16 (int8 and fp8 are not implemented for those ops). The
low-precision backend is already implemented as an `onnx-mlir-opt`-only pass
(`--convert-onnx-to-lowprecision`, driven by `LOWP_NODE_FORMATS`), and
supported nodes can be assigned, lowered, compiled, and executed with
different formats today. The current research focuses on complete
configuration generation, graph-based accuracy-loss prediction, static cost
analysis, Pareto selection, real evaluation, and iterative model updating —
exactly as in the Posit variant, just over this format set.

2. Research Problem

The number of possible mixed-precision configurations grows exponentially
with the number of tunable nodes. Let (N) be the number of tunable nodes and
(F_i) be the supported format set of node (i). (SearchSpace=\prod_{i=1}^{N}|F_i|)
Because (F_i) is now op-type dependent (6 choices for Conv/Gemm/MatMul, 3 for
elementwise ops), the space is smaller than a uniform-format-set assumption
would suggest, but still exponential. Exhaustively compiling and evaluating
every configuration is impractical because real low-precision execution is
expensive. The framework therefore learns a surrogate model that predicts the
whole-model accuracy loss of a complete mixed-precision configuration.
(\min_P\left(AccuracyLoss(P),WeightStorage(P),EstimatedPeakActivationMemory(P)\right))
where (P) is a complete node-wise mixed-precision configuration.
(AccuracyLoss(P)=Accuracy_{FP32}-Accuracy_P) A deployment-stage tolerance (T)
may additionally be imposed: (AccuracyLoss(P)\leq T) The framework searches
for Pareto-optimal trade-offs instead of selecting only the safest or
lowest-bit configuration.

3. Research Objectives

The objectives are to represent ONNX-MLIR as a graph, encode complete
low-precision configurations, predict whole-model accuracy loss, calculate
weight storage and peak activation memory, search Pareto configurations,
validate them through real execution, update the predictor, and reduce
tuning cost.

4. Overall Framework

Configuration Generator
          ↓
Complete Mixed-Precision Configurations
          ↓
 ┌────────┴─────────┐
 ↓                  ↓
Accuracy Predictor  Cost Calculator
GNN Regression      Static Analysis
 ↓                  ↓
Predicted            Weight Storage
Accuracy Loss        Estimated Peak Activation Memory
 └────────┬─────────┘
          ↓
     Pareto Selection
          ↓
  Selected Configurations
          ↓
   Real Low-Precision Evaluator
          ↓
   True Accuracy Loss
          ↓
Dataset Update + Fine-tune
          ↓
 Updated Accuracy Predictor

The Accuracy Predictor estimates the accuracy effect of a complete
configuration. The Cost Calculator computes storage and memory costs without
ML prediction. The Real Low-Precision Evaluator remains the source of
ground-truth accuracy.

5. Configuration Generator

5.1 Initial Stage

Before the predictor is available, the generator uses stratified random
sampling plus manually designed representative configurations. Strata cover
low-precision ratio; bf16/f16/int8/fp8e4m3/fp8e5m2/FP32 proportions; front,
middle, and back regions; operator distribution (Conv/Gemm/MatMul vs.
elementwise, since their legal format sets differ); weight and activation
size; graph depth and branches; and average bit-width. Representative samples
include all FP32; uniform bf16; uniform f16; uniform int8 (on
Conv/Gemm/MatMul only); uniform fp8e4m3/fp8e5m2 (on Conv/Gemm/MatMul only);
one low-precision region; front-only or back-only quantization; alternating
formats; random mixtures; and sensitivity-guided configurations when
available. Each sampled configuration is compiled and evaluated by the Real
Low-Precision Evaluator. The measured whole-model accuracy loss becomes the
regression label.

Because int8/fp8e4m3/fp8e5m2 require externally supplied activation
scale/zero-point (the pass itself does no calibration — see Section 20), any
configuration that selects one of these formats for a node needs that node's
calibration parameters resolved before compilation. Calibration is run once
per (node, format) pair on a calibration subset and cached, rather than
per sampled configuration, since the scale/zero-point only depend on the
node's activation distribution and the target format, not on the rest of the
configuration.

5.2 Predictor-Assisted Stage

After initial training, the generator changes to surrogate-assisted
NSGA-II. NSGA-II generates complete configurations through population
initialization, crossover, mutation, non-dominated sorting, and
crowding-distance selection. The predictor estimates accuracy loss, while the
Cost Calculator computes weight storage and peak activation memory. Only
selected Pareto, uncertain, or underrepresented configurations are sent to
real evaluation. Mutation and crossover must respect the per-op format mask
(Section 1): a mutation on an elementwise node may only draw from
{bf16, f16, FP32}, never int8/fp8.

6. ONNX-MLIR Placement

This is where the low-precision framework diverges structurally from the
Posit variant. The Posit placement lowers selected nodes into the Posit
Dialect before graph extraction, because Posit assigns each node a genuinely
different op type carrying N/ES. Low-precision lowering does not: bf16/f16
retype in place with a `onnx.Cast` wrapper and leave the original op
untouched; int8/fp8 rewrite one node into a small `QuantizeLinear ->
QLinear* -> DequantizeLinear`-style subgraph, but only when the pass actually
runs at compile time. Consequently the ONNX Dialect graph's topology is the
same for every configuration — only the per-node chosen format changes. This
means graph extraction does not need to happen after a selective lowering
pass; it can happen once, at the ONNX Dialect stage, with the configuration
attached purely as a node feature.

ONNX model
  -> ONNX Dialect + shape inference
  -> identify supported ONNX nodes + per-op format eligibility mask
  -> cache high-level static metadata by onnx_node_name
  -> calibration: resolve activation scale/zero-point per (node, int8/fp8 format)
     on a calibration subset, cached
  -> Configuration Generator selects per-node formats
  -> extract ONNX Dialect graph once; attach selected format as a node feature
     (no selective lowering needed for graph/feature extraction)
  -> Accuracy Predictor / Cost Calculator
  -> [real evaluation only] annotate selected ONNX nodes with LOWP_NODE_FORMATS
     (+ calibration scale/zero-point for int8/fp8 nodes)
  -> selective ONNX-to-LowPrecision lowering (--convert-onnx-to-lowprecision)
  -> Krnl / SCF / LLVM
  -> executable mixed-precision model

The ONNX Dialect is both the configuration-decision stage and the predictor
graph-extraction stage — there is no separate "lowered dialect" stage to
extract from, unlike Posit. The Configuration Generator uses stable ONNX node
identities (`onnx_node_name`) to decide which nodes get which format. Because
the graph structure never changes across configurations, the same
extracted graph and edge index can be reused across every sampled
configuration in the dataset; only the per-node format feature vector differs
between samples. Selective lowering into the actual
`QuantizeLinear`/`QLinear*`/`DequantizeLinear`/`Cast` IR is deferred to the
Real Low-Precision Evaluator, i.e. it only happens when a configuration is
actually compiled and run, not during feature/graph extraction.

7. Graph Construction

7.1 Node Definition

Each ONNX operation is represented as one graph node, using its
`onnx_node_name` as identity. Unlike the Posit variant, this identity does
not need to be recovered after lowering into a different dialect, because
graph extraction happens before lowering and the graph topology is
configuration-invariant. Constant weights and biases are incorporated into
the owning operation's node features rather than represented as independent
graph nodes.

7.2 Dependency Definition

The primary dependency is the SSA tensor producer-consumer relation in the
ONNX Dialect. An edge is created when an ONNX operation result is used as an
operand of another operation. The graph is an operation-level ONNX data-flow
graph. Its topology is fixed per model and does not depend on the applied
configuration (contrast with the Posit variant, where the graph topology
reflects which nodes were actually lowered to Posit ops).

7.3 Forward and Reverse Edges

For each producer-consumer dependency, the graph contains one forward
data-flow edge and one reverse message-passing edge.

Gemm_3 -> Relu_4  forward
Relu_4 -> Gemm_3  reverse

The reverse edge is not a real backward dependency or execution order. It
only allows the GNN to propagate downstream context toward earlier nodes.

7.4 Order Information

The graph preserves topological_position, operand_index, and
producer_output_index, computed once at the ONNX stage since the graph
structure is shared across every configuration sample.

8. Node Features

Node features are assembled from ONNX-stage metadata plus the configuration
being scored. Unlike the Posit variant there is no separate "lowered-stage
extraction" to join back via node identity, since everything is read from
the single ONNX Dialect graph.

8.1 Static Structural Features

Unchanged from the Posit variant: original op_type; input/output rank and log
element counts; log weight count and FLOPs; kernel, stride, four-direction
padding, dilation, group, log bias count; normalized topological position;
in/out degree; and has_weight. Large count features use:
(log_value=\log(1+value))

8.2 Weight Statistics

Unchanged: weight_mean, weight_std, weight_p99_abs, weight_zero_ratio, and
weight_dynamic_range. Nodes without weights use zero-filled statistics and
has_weight = 0.

8.3 Activation Statistics

Unchanged: activation_mean, activation_std, activation_p99_abs,
activation_zero_ratio, and activation_dynamic_range, collected on a
calibration subset before applying a candidate configuration. These same
statistics also feed int8/fp8 scale/zero-point calibration (Section 5.1),
so they only need to be computed once per model.

8.4 Low-Precision Configuration Features

Each graph node receives the format selected for it by the current
configuration. Unlike Posit's N/ES (a continuous format family), the
low-precision candidate set is a fixed small enumeration, so format is
primarily encoded as a categorical tag, with a few derived scalar features:

FP32     -> is_fp32=1, bitwidth=32, is_float_format=1, requires_calibration=0
bf16     -> is_fp32=0, bitwidth=16, is_float_format=1, requires_calibration=0
f16      -> is_fp32=0, bitwidth=16, is_float_format=1, requires_calibration=0
int8     -> is_fp32=0, bitwidth=8,  is_float_format=0, requires_calibration=1
fp8e4m3  -> is_fp32=0, bitwidth=8,  is_float_format=1, requires_calibration=1
fp8e5m2  -> is_fp32=0, bitwidth=8,  is_float_format=1, requires_calibration=1

`is_float_format` distinguishes int8 (linear fixed-point, scale/zero-point)
from the floating-point-shaped formats (bf16/f16/fp8e4m3/fp8e5m2), since they
behave very differently even at the same bit-width. `requires_calibration`
flags formats whose deployment additionally depends on externally supplied
scale/zero-point rather than being a pure retype. A learned categorical
format embedding (one-hot or trainable embedding over the 6 formats) may be
added alongside these scalars, analogous to Posit's optional format
embedding.

8.5 Optional Node-Format Features

Optional node-format features are fake-quant normalized MSE and SQNR,
underflow/zero-after-quantization ratios, and single-node accuracy or loss
sensitivity — same idea as the Posit variant. `experiments/imagenet100/eval_fp_formats.py`
already implements per-format fake-quant rounding (`fake_quant()`, using
`torch.float16`/`bfloat16`/`float8_e4m3fn`/`float8_e5m2`, plus an int8
round-trip to add) for bf16/f16/fp8e4m3/fp8e5m2 at the whole-model level;
that rounding function is directly reusable per-node for these auxiliary
features without needing a compiled onnx-mlir backend.

9. Edge Features

Unchanged: direction, operand_index, producer_output_index, tensor rank, and
log tensor elements. A format-transition edge exists when producer output
format differs from consumer input format; for low-precision formats this
transition corresponds to an actual `Cast`/`DequantizeLinear`->`QuantizeLinear`
boundary inserted at real-compile time, not just an abstract feature.

10. Feature Encoding and Embedding

10.1 Operation Embedding

Unchanged: op_type is categorical, converted to an integer ID, then passed
through a trainable embedding table. (e_{op}=Embedding(op_id))

10.2 Format Encoding

Replaces Posit's normalized-N/normalized-ES encoding with a categorical
encoding over the fixed 6-format set (Section 8.4): a one-hot or trainable
embedding for the format tag, concatenated with the bitwidth,
is_float_format, and requires_calibration scalars.

10.3 Numerical Feature Processing

Unchanged: count-based features use logarithmic transformation; continuous
features are standardized using training-set statistics
(x'=\frac{x-\mu}{\sigma}), with normalization parameters fit on the
training split and reused for validation/test models.

10.4 Node and Edge Vectors

Unchanged in structure: (x_i=[e_{op}\Vert x_{static}\Vert x_{weight}\Vert
x_{activation}\Vert x_{format}]), forming (X\in\mathbb{R}^{|V|\times D_n}).
(e_{ij}=[e_{direction}\Vert operand_index\Vert output_index\Vert rank\Vert
log_elements]). The graph input contains X, edge_index, and edge_attr.

11. Accuracy Predictor

Unchanged: an edge-aware graph neural network (GINE or another
message-passing layer accepting edge attributes). Because the graph
structure is now shared across every configuration sample of a given model
(Section 6/7), the dataset can reuse one cached (edge_index, edge_attr) per
model and only vary X's format columns per sample — a data-loading
simplification relative to the Posit variant.

Node and edge vectors
        ↓
Edge-aware GNN layers
        ↓
Updated node embeddings
        ↓
Global mean pooling + global max pooling
        ↓
Graph/configuration embedding
        ↓
MLP regression head
        ↓
Predicted whole-model accuracy loss

(f(G_{onnx},X_{onnx},X_{format}(P))\rightarrow\widehat{AccuracyLoss}(P)) where
(G_{onnx}) is the (configuration-invariant) ONNX Dialect graph, (X_{onnx})
contains cached high-level ONNX metadata, and (X_{format}(P)) contains the
per-node format features implied by configuration (P). The predictor does
not estimate weight storage or peak activation memory.

12. Regression Label and Training

Each dataset sample corresponds to one complete configuration.

Gemm_3 = int8
Relu_4 = f16
Gemm_5 = FP32
label  = measured whole-model accuracy loss

(y_P=Accuracy_{FP32}-Accuracy_P) Initial training objective may use mean
squared error: (L_{MSE}=\frac{1}{M}\sum_{P=1}^{M}(\widehat{y}_P-y_P)^2). Mean
absolute error and Huber loss may be compared. A ranking-aware loss may be
considered later.

13. Cost Calculator

The Cost Calculator performs static analysis for every complete
configuration. It produces persistent weight storage and estimated peak live
activation memory. No learned model is required for these objectives.

14. Persistent Weight Storage

(WeightStorage(P)=\sum_{w\in Weights}Elements(w)\times\frac{Bits_P(w)}{8})
with Bits_P taken from the table in Section 8.4 (32/16/16/8/8/8 for
FP32/bf16/f16/int8/fp8e4m3/fp8e5m2). Shared initializers are counted once.
Note per `docs/LowPrecisionFormats.md`: INT8/FP8 Gemm/MatMul require the
weight to be a compile-time constant to be eligible at all, so any node
without a constant weight is excluded from those formats' candidate set
regardless of what the Cost Calculator computes.

15. Estimated Peak Activation Memory

Unchanged formula and conservative liveness model, with Bits_P(a) again
taken from the Section 8.4 table:
(TensorBytes(a)=Elements(a)\times\frac{Bits_P(a)}{8})
(LiveMemory(t)=\sum_{a\in Live(t)}TensorBytes(a))
(EstimatedPeakActivationMemory(P)=\max_t LiveMemory(t))

16. Initial Dataset Construction

Unchanged in structure. Each sample stores the model ID and ONNX graph;
complete configuration; node and edge tensors; FP32 and low-precision
accuracy; measured accuracy loss; weight storage; estimated peak activation
memory; and build/execution metadata. The cache key should include model,
complete configuration, compiler version, flags, calibration subset/version
(new — since int8/fp8 accuracy depends on which calibration data produced
the scale/zero-point), dataset subset, and evaluation settings.

17. Multi-Fidelity Data

Unchanged: real full-dataset evaluation is the highest-fidelity label
source. Lower-cost signals may include calibration-subset accuracy,
fake-quant output error (directly available today via
`eval_fp_formats.py`'s `fake_quant()`), calibration loss increase, logit
divergence, and single-node sensitivity.

18. Surrogate-Assisted NSGA-II

Unchanged in structure. Mutation changes one or more node formats, masked by
the per-op eligibility table (Section 1); crossover combines format regions
from two configurations; FP32 remains available as a fallback for every op.

19. Pareto Selection

Unchanged: minimize AccuracyLoss(P), WeightStorage(P), and
EstimatedPeakActivationMemory(P) jointly, retaining non-dominated
configurations with diversity preserved.

20. Real Low-Precision Evaluator and Model Update

Selected configuration
  -> update ONNX node-format mapping
  -> for each int8/fp8e4m3/fp8e5m2 node: look up (or compute) cached
     activation scale/zero-point from the calibration subset
  -> annotate selected ONNX nodes: LOWP_NODE_FORMATS=<node>:<format>[:x_scale:x_zp:y_scale:y_zp],...
  -> selective ONNX-to-LowPrecision lowering (onnx-mlir-opt --convert-onnx-to-lowprecision)
  -> compilation
  -> validation execution
  -> measured accuracy
  -> true accuracy loss

Per `docs/LowPrecisionFormats.md`, only `onnx-mlir-opt` currently wires in
`--convert-onnx-to-lowprecision` — the main `onnx-mlir` driver does not (same
precedent as the existing Posit pass). The evaluator must therefore drive
`onnx-mlir-opt` directly (or an equivalent explicit pipeline) rather than the
standard `onnx-mlir` compile entry point, until/unless the pass is wired into
the main driver. The evaluator records compilation status, execution status,
accuracy, true loss, build time, evaluation time, and optional binary size.
A predicted Pareto configuration is accepted only after real evaluation.
Every newly evaluated configuration is added to the dataset.

Predict -> select -> real evaluate -> add labels -> fine-tune -> predict again

21. Precision Plan Output

precision_plan.json records the strategy, per-node formats (including
resolved calibration scale/zero-point for int8/fp8 nodes), predicted and
measured loss, weight storage, peak activation memory, Pareto rank, and
validation status.

22. Experimental Design

Unchanged: candidate model families include small CNN or MLP, ResNet,
MobileNet, ShuffleNet, and EfficientNet. Selection criteria include
ONNX-MLIR compatibility, low-precision op coverage, operator diversity,
residual structure, depthwise convolution, activation diversity, and
evaluation cost. Recommended protocols include leave-one-model-out and
leave-one-family-out, with normalization statistics fit only on the training
split.

23. Research Questions and Metrics

Unchanged (RQ1 accuracy-loss prediction, RQ2 configuration ranking, RQ3
search efficiency, RQ4 Pareto quality, RQ5 weight storage, RQ6 peak
activation memory, RQ7 feature contribution) — same metrics, evaluated over
the low-precision format set instead of Posit.

24. Baselines

Required baselines are FP32; uniform bf16; uniform f16; uniform int8 (on
eligible ops only); uniform fp8e4m3; uniform fp8e5m2; random and stratified
random search; sensitivity-guided greedy search; NSGA-II without an accuracy
predictor; and tabular whole-configuration regression. Optional baselines
include single-node XGBoost prediction, exhaustive search on small models,
GNN without edge attributes, and GNN without reverse edges. All search
methods should be compared under equal real-evaluation budgets.

25. Implementation Status and Plan

Implemented today (per `docs/LowPrecisionFormats.md`): the
`--convert-onnx-to-lowprecision` `onnx-mlir-opt` pass, driven by
`LOWP_NODE_FORMATS`; bf16/f16 in-place `Cast` retyping for
Conv/Gemm/MatMul/Relu/Add/Sub/Mul/Div; int8 QDQ (QuantizeLinear /
QLinearConv / QLinearMatMul / DequantizeLinear) for Conv (including grouped
and depthwise) and Gemm/MatMul; fp8e4m3/fp8e5m2 quantize/dequantize boundary
+ opset-21 `QLinearMatMul` for Gemm/MatMul, and a hand-rolled im2col
decomposition at the ONNX-graph level for Conv; and
`experiments/imagenet100/eval_fp_formats.py`'s whole-model fake-quant
evaluation for bf16/f16/fp8e4m3/fp8e5m2.

Remaining work: a calibration pass/script producing per-node activation
scale/zero-point for int8/fp8 nodes (currently must be supplied externally);
a `LOWP_NODE_FORMATS`-generating counterpart to
`experiments/imagenet100/gen_node_formats.py`; a per-node sensitivity sweep
analogous to `experiments/common/sweep_node_sensitivity.py` but over this
format set and its per-op mask; ONNX Dialect graph + feature extraction
(Section 6/7); the GNN accuracy predictor; NSGA-II with the per-op format
mask; the Cost Calculator's low-precision bit-width table (Section 14/15);
and wiring `--convert-onnx-to-lowprecision` into the main `onnx-mlir` driver
if the evaluator needs a non-`onnx-mlir-opt` compile path.

26. Scope and Limitations

In addition to the general limitations shared with the Posit variant
(latency not a primary objective; prediction quality depends on
configuration diversity; activation statistics depend on the calibration
dataset; peak activation memory is a logical static estimate; reverse edges
do not represent execution dependencies; the full configuration space cannot
be exhaustively evaluated for large models), this variant carries the
limitations documented in `docs/LowPrecisionFormats.md`:

- No calibration automation: activation scale/zero-point must be supplied
  externally for every int8/fp8 node; there is no calibration pass in the
  compiler itself.
- INT8/FP8 Gemm/MatMul require the weight (`B`) to be a compile-time
  constant — a non-constant weight makes those formats ineligible for that
  node regardless of predicted benefit.
- FP8 weight/output quantization is per-tensor only (no per-channel scale),
  for both Conv and Gemm/MatMul.
- Only `onnx-mlir-opt` wires in `--convert-onnx-to-lowprecision`; the main
  `onnx-mlir` driver does not.
- Elementwise ops (Relu/Add/Sub/Mul/Div) only support bf16/f16 — int8 and
  fp8e4m3/fp8e5m2 are not implemented for them, shrinking (F_i) for those
  nodes to 3 formats instead of 6.

27. Expected Contributions

Expected contributions are an ONNX Dialect operation graph with per-node
low-precision configuration features (extracted once, reused across every
sampled configuration), a configuration-aware GNN accuracy predictor, static
storage and liveness calculators, surrogate-assisted NSGA-II under a
per-op format mask, an iterative real-evaluation loop, and an empirical
analysis of accuracy-storage-memory trade-offs for bf16/f16/int8/fp8e4m3/fp8e5m2
mixes — directly comparable to the Posit-variant results produced by
`claude.md`.

28. Success Criteria

Unchanged: executable configurations, useful prediction and ranking
quality, reduced storage and peak memory, acceptable real accuracy loss,
fewer expensive evaluations, improved generalization over tabular baselines,
and clear Pareto trade-offs.

29. Final Summary

(\min_P\left(AccuracyLoss(P),WeightStorage(P),EstimatedPeakActivationMemory(P)\right))
(f(G_{onnx},X_{onnx},X_{format}(P))\rightarrow\widehat{AccuracyLoss}(P)) The
Configuration Generator operates on ONNX node identities and a per-op format
eligibility mask to decide which format each node uses. Because
low-precision lowering preserves ONNX Dialect topology (unlike Posit
lowering, which introduces a distinct dialect), the graph is extracted once
from the ONNX Dialect and the configuration is attached purely as a node
feature — no re-extraction after lowering is needed for training data.
Selective lowering into `Cast`/QDQ IR (`--convert-onnx-to-lowprecision`) is
reserved for the Real Low-Precision Evaluator's actual compile/run step.
Weight storage and peak activation memory are calculated through static
analysis using the format bit-width table. Selected Pareto configurations
are compiled and executed by the real low-precision backend, with int8/fp8
nodes requiring resolved calibration parameters first. Measured results are
added to the dataset and used to update the Accuracy Predictor. The Posit
variant (`claude.md`) remains the parallel track for the Posit format
family and is unaffected by this document.
