ML-Guided Whole-Configuration Mixed-Posit Optimization on ONNX-MLIR

1. Project Overview

This project develops an ML-guided whole-configuration mixed-Posit optimization framework on ONNX-MLIR.Given a trained ONNX model, the framework assigns a numerical format to each supported ONNX operation or output tensor.A configuration may contain Posit8, Posit16, Posit32, and FP32 simultaneously.

Conv_0    -> posit_16_1
Relu_1    -> posit_8_1
MatMul_2  -> posit_16_2
Add_3     -> posit_8_0
Softmax_4 -> FP32

Candidate formats currently include posit_8_{0,1,2}, posit_16_{0,1,2}, posit_32_{0,1,2}, and FP32.The Posit backend is already implemented, and supported nodes can be assigned, lowered, compiled, and executed with different formats.The current research focuses on complete-configuration generation, graph-based accuracy-loss prediction, static cost analysis, Pareto selection, real evaluation, and iterative model updating.

2. Research Problem

The number of possible mixed-precision configurations grows exponentially with the number of tunable nodes.Let (N) be the number of tunable nodes and (F_i) be the supported format set of node (i).(SearchSpace=\prod_{i=1}^{N}|F_i|)Exhaustively compiling and evaluating every configuration is impractical because real Posit execution is expensive.The framework therefore learns a surrogate model that predicts the whole-model accuracy loss of a complete mixed-Posit configuration.(\min_P\left(AccuracyLoss(P),WeightStorage(P),EstimatedPeakActivationMemory(P)\right))where (P) is a complete node-wise mixed-Posit configuration.(AccuracyLoss(P)=Accuracy_{FP32}-Accuracy_P)A deployment-stage tolerance (T) may additionally be imposed:(AccuracyLoss(P)\leq T)The framework searches for Pareto-optimal trade-offs instead of selecting only the safest or lowest-bit configuration.

3. Research Objectives

The objectives are to represent ONNX-MLIR as a graph, encode complete Posit configurations, predict whole-model accuracy loss, calculate weight storage and peak activation memory, search Pareto configurations, validate them through real execution, update the predictor, and reduce tuning cost.

4. Overall Framework

Configuration Generator
          ↓
Complete Mixed-Posit Configurations
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
   Real Posit Evaluator
          ↓
   True Accuracy Loss
          ↓
Dataset Update + Fine-tune
          ↓
 Updated Accuracy Predictor

The Accuracy Predictor estimates the accuracy effect of a complete configuration.The Cost Calculator computes storage and memory costs without ML prediction.The Real Posit Evaluator remains the source of ground-truth accuracy.

5. Configuration Generator

5.1 Initial Stage

Before the predictor is available, the generator uses stratified random sampling plus manually designed representative configurations.Strata cover low-precision ratio; Posit8/16/32/FP32 proportions; front, middle, and back regions; operator distribution; weight and activation size; graph depth and branches; and average bit-width.Representative samples include all FP32; uniform Posit8/16/32; one low-precision region; front-only or back-only quantization; alternating formats; random mixtures; and sensitivity-guided configurations when available.Each sampled configuration is compiled and evaluated by the Real Posit Evaluator.The measured whole-model accuracy loss becomes the regression label.

5.2 Predictor-Assisted Stage

After initial training, the generator changes to surrogate-assisted NSGA-II.NSGA-II generates complete configurations through population initialization, crossover, mutation, non-dominated sorting, and crowding-distance selection.The predictor estimates accuracy loss, while the Cost Calculator computes weight storage and peak activation memory.Only selected Pareto, uncertain, or underrepresented configurations are sent to real evaluation.

6. ONNX-MLIR Placement

ONNX model
  -> ONNX Dialect + shape inference
  -> identify supported ONNX nodes
  -> cache high-level static metadata by onnx_node_name
  -> Configuration Generator selects per-node formats
  -> annotate selected ONNX nodes
  -> selective ONNX-to-Posit lowering
  -> Posit Dialect graph and feature extraction
  -> Accuracy Predictor / Cost Calculator
  -> Krnl / SCF / LLVM
  -> executable mixed-Posit model

The ONNX Dialect is the configuration-decision stage.The Configuration Generator uses stable ONNX node identities, such as onnx_node_name, to decide which nodes should be lowered to Posit and which Posit format each selected node should use.The Posit Dialect is the predictor graph-extraction stage.After applying a complete configuration, selected ONNX operations are lowered into Posit operations, and the resulting Posit-level SSA producer-consumer graph is extracted for the Accuracy Predictor.High-level information that may not be fully preserved after lowering, including original ONNX op type, shapes, attributes, FLOPs, and weight statistics, is cached at the ONNX stage and joined back to Posit nodes through the preserved node identity.LLVM IR is not used as the primary predictor input because one high-level operation may be expanded into many loops, loads, stores, and calls.

7. Graph Construction

7.1 Node Definition

Each operation in the Posit Dialect graph is represented as one graph node.A Posit node preserves or references the original ONNX node identity so that its high-level metadata and selected configuration can be recovered.For example, an ONNX node named Gemm_3 may be selectively lowered to a Posit Gemm operation carrying the same logical node ID and its selected N and ES.Constant weights and biases are initially incorporated into the owning operation's node features rather than represented as independent graph nodes.

7.2 Dependency Definition

The primary dependency is the SSA tensor producer-consumer relation in the Posit Dialect.An edge is created when a Posit operation result is used as an operand of another operation.The graph is therefore an operation-level Posit data-flow graph, not a graph formed by connecting adjacent textual instructions.The graph topology reflects the applied complete configuration because the selected ONNX nodes have already been lowered into Posit operations.

7.3 Forward and Reverse Edges

For each producer-consumer dependency, the graph contains one forward data-flow edge and one reverse message-passing edge.

Gemm_3 -> Relu_4  forward
Relu_4 -> Gemm_3  reverse

The reverse edge is not a real backward dependency or execution order.It only allows the GNN to propagate downstream context toward earlier nodes.

7.4 Order Information

The graph preserves topological_position, operand_index, and producer_output_index.topological_position represents normalized operation order.operand_index identifies which consumer input receives the tensor.producer_output_index identifies which producer output generates the tensor.

8. Node Features

Node features are assembled from two sources.ONNX-stage metadata provides high-level structural and numerical information, while Posit-stage extraction provides the actual lowered operation, selected format, and Posit SSA graph context.The two records are joined using the preserved ONNX node identity.

8.1 Static Structural Features

The first version includes original op_type; input/output rank and log element counts; log weight count and FLOPs; kernel, stride, four-direction padding, dilation, group, log bias count; normalized topological position; in/out degree; and has_weight.Most of these values are collected or cached before selective lowering and then attached to the corresponding Posit graph node.Large count features use:(log_value=\log(1+value))

8.2 Weight Statistics

Weight features are weight_mean, weight_std, weight_p99_abs, weight_zero_ratio, and weight_dynamic_range.Nodes without weights use zero-filled statistics and has_weight = 0.

8.3 Activation Statistics

Activation statistics are collected using a calibration subset.Activation features are activation_mean, activation_std, activation_p99_abs, activation_zero_ratio, and activation_dynamic_range.These values describe the FP32 activation distribution before applying a candidate configuration.

8.4 Posit Configuration Features

Each Posit graph node receives the format selected earlier for its original ONNX node.The selected configuration is applied during ONNX-to-Posit lowering, so N and ES are read from the resulting Posit operation or its type/attributes rather than inferred only from an external plan.The initial representation includes N, ES, and is_fp32.

FP32      -> is_fp32=1, N=32, ES=0
Posit8E1  -> is_fp32=0, N=8,  ES=1
Posit16E2 -> is_fp32=0, N=16, ES=2

A learned format embedding may be added when the candidate set is fixed.Keeping N and ES supports future experiments with previously unseen formats.

8.5 Optional Node-Format Features

Optional node-format features are fake-quant normalized MSE and SQNR, underflow and zero-after-quantization ratios, and single-node accuracy or loss sensitivity.These values depend on both the node and the selected format.They are auxiliary features rather than whole-configuration labels.

9. Edge Features

The first edge vector includes direction, operand_index, producer_output_index, tensor rank, and log tensor elements; future versions may add residual, format-transition, tensor-statistics, alias/view, and lifetime features.A format-transition edge exists when producer output precision differs from consumer input precision.

10. Feature Encoding and Embedding

10.1 Operation Embedding

op_type is categorical and is converted to an integer ID, then passed through a trainable embedding table.(e_{op}=Embedding(op_id))The operation ID is not directly treated as a continuous number.

10.2 Format Encoding

Posit configuration is represented using normalized N, normalized ES, and is_fp32.An optional learned format embedding can be concatenated with these values.

10.3 Numerical Feature Processing

Count-based features first use logarithmic transformation.Continuous features are standardized using training-set statistics:(x'=\frac{x-\mu}{\sigma})The same normalization parameters are reused for validation and test models.

10.4 Node and Edge Vectors

The node vector is:(x_i=[e_{op}\Vert x_{static}\Vert x_{weight}\Vert x_{activation}\Vert x_{format}])The node vectors form:(X\in\mathbb{R}^{|V|\times D_n})The edge vector is:(e_{ij}=[e_{direction}\Vert operand_index\Vert output_index\Vert rank\Vert log_elements])The graph input contains X, edge_index, and edge_attr.

11. Accuracy Predictor

The primary predictor is an edge-aware graph neural network.The first implementation may use GINE or another message-passing layer that accepts edge attributes.

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

The same original ONNX model produces a different Posit Dialect graph sample when a different complete configuration is applied and selectively lowered.(f(G_{posit}(P),X_{onnx},X_{posit})\rightarrow\widehat{AccuracyLoss}(P))where (G_{posit}(P)) is the Posit Dialect graph produced by configuration (P), (X_{onnx}) contains cached high-level ONNX metadata, and (X_{posit}) contains lowered Posit operation and format features.The predictor does not estimate weight storage or peak activation memory.

12. Regression Label and Training

Each dataset sample corresponds to one complete configuration.

Gemm_3 = posit_8_1
Relu_4 = posit_16_1
Gemm_5 = FP32
label  = measured whole-model accuracy loss

The label is:(y_P=Accuracy_{FP32}-Accuracy_P)The initial training objective may use mean squared error:(L_{MSE}=\frac{1}{M}\sum_{P=1}^{M}(\widehat{y}_P-y_P)^2)Mean absolute error and Huber loss may be compared.A ranking-aware loss may be considered later because configuration ordering matters for Pareto selection.

13. Cost Calculator

The Cost Calculator performs static analysis for every complete configuration.It produces persistent weight storage and estimated peak live activation memory.No learned model is required for these objectives.

14. Persistent Weight Storage

(WeightStorage(P)=\sum_{w\in Weights}Elements(w)\times\frac{Bits_P(w)}{8})The calculation includes constant weights and biases when biases follow the assigned format.Shared initializers are counted only once.The precision of each weight tensor is determined by its owning operation or an explicit weight-format rule.The metric excludes executable code, Posit runtime libraries, external shared libraries, compiler metadata, and unmodeled alignment overhead.Compiled .so size may be reported as a secondary metric, but it is not equivalent to parameter storage.

15. Estimated Peak Activation Memory

For every activation tensor, record producer, consumers, shape, element count, assigned bit-width, creation step, last-use step, and alias relation.(TensorBytes(a)=Elements(a)\times\frac{Bits_P(a)}{8})A tensor becomes live after its producer creates it and remains live until its final consumer finishes using it.(LiveMemory(t)=\sum_{a\in Live(t)}TensorBytes(a))(EstimatedPeakActivationMemory(P)=\max_t LiveMemory(t))The first version uses a conservative model in which operation inputs remain live while outputs are allocated.View-like operations such as Reshape, Flatten, and Unsqueeze normally reuse the same buffer.Backend workspace, allocator fragmentation, and external-library temporary buffers are excluded unless statically available.

16. Initial Dataset Construction

Each sample stores the model ID and ONNX graph; complete configuration; node and edge tensors; FP32 and Posit accuracy; measured accuracy loss; weight storage; estimated peak activation memory; and build/execution metadata.The initial sampling strategy combines stratified random configurations and representative configurations.Sampling should cover both high-accuracy and high-loss regions.Sampling only near FP32 may fail on aggressive configurations, while sampling only uniform formats may fail to learn interactions.The cache key should include model, complete configuration, compiler version, flags, dataset subset, and evaluation settings.

17. Multi-Fidelity Data

Real full-dataset Posit evaluation is the highest-fidelity label source.Lower-cost signals may include calibration-subset accuracy, fake-quant output error, calibration loss increase, logit divergence, and single-node sensitivity.These proxies do not replace the whole-model accuracy label.They may be added as features or used to prioritize expensive evaluations.

18. Surrogate-Assisted NSGA-II

After initial predictor training, NSGA-II becomes the main generator.Each individual is a complete configuration vector:(P=[f_1,f_2,\ldots,f_N],\quad f_i\in F_i)For each individual:

Construct node-level configuration features.

Predict whole-model accuracy loss.

Calculate weight storage.

Calculate estimated peak activation memory.

Apply non-dominated sorting.

Apply crossover and mutation.

Generate the next population.Mutation changes one or more node formats.Crossover combines format regions from two configurations.Unsupported formats are masked, and FP32 remains available as a fallback.

19. Pareto Selection

The objectives are:(\min AccuracyLoss(P))(\min WeightStorage(P))(\min EstimatedPeakActivationMemory(P))A configuration is dominated when another configuration is no worse in every objective and strictly better in at least one objective.The system retains non-dominated configurations.When tolerance (T) is used, configurations above the predicted tolerance may be filtered or treated as constraint violations.Pareto diversity is preserved so that the final results expose multiple trade-offs.

20. Real Posit Evaluator and Model Update

Selected configuration
  -> update ONNX node-format mapping
  -> annotate selected ONNX nodes
  -> selective ONNX-to-Posit lowering
  -> extract Posit Dialect graph/features
  -> compilation
  -> validation execution
  -> measured accuracy
  -> true accuracy loss

The evaluator records compilation status, execution status, accuracy, true loss, build time, evaluation time, and optional binary size.A predicted Pareto configuration is accepted only after real evaluation.Every newly evaluated configuration is added to the dataset.New evaluations should prioritize Pareto-front configurations, uncertain predictions, large errors, underrepresented formats, and configurations near tolerance.

Predict -> select -> real evaluate -> add labels -> fine-tune -> predict again

This forms a surrogate-assisted optimization and active-learning loop.

21. Precision Plan Output

precision_plan.json records the strategy, per-node formats, predicted and measured loss, weight storage, peak activation memory, Pareto rank, and validation status.

22. Experimental Design

Candidate model families include small CNN or MLP, ResNet, MobileNet, ShuffleNet, and EfficientNet.Selection criteria include ONNX-MLIR compatibility, Posit support, operator diversity, residual structure, depthwise convolution, activation diversity, and evaluation cost.Training and testing splits should be performed by complete models or model families.Recommended protocols include leave-one-model-out and leave-one-family-out.Normalization statistics must be fitted only on the training split.Within-model and cross-model results should be reported separately.

23. Research Questions and Metrics

RQ1 Accuracy-Loss Prediction: Can the graph predictor estimate whole-configuration loss? Metrics: MAE, RMSE, (R^2), and maximum absolute error.

RQ2 Configuration Ranking: Can it rank configurations correctly? Metrics: Spearman correlation, Kendall's tau, and Top-K recall.

RQ3 Search Efficiency: Does surrogate search reduce real evaluation? Metrics: configuration, compilation, and execution counts; tuning time; predictor inference time.

RQ4 Pareto Quality: Does it find better trade-offs? Metrics: hypervolume, Pareto coverage, non-dominated count, and best feasible solution under each tolerance.

RQ5 Weight Storage: Metrics: FP32 bytes, mixed-Posit bytes, reduction ratio, and weighted average weight bit-width.

RQ6 Peak Activation Memory: Metrics: FP32 and mixed peak memory, reduction ratio, peak step, and live tensors at the peak.

RQ7 Feature Contribution: Ablate configuration, static, weight-statistics, activation-statistics, edge, fake-quant, and single-node-sensitivity features.

24. Baselines

Required baselines are FP32; uniform Posit8/16/32; random and stratified random search; sensitivity-guided greedy search; NSGA-II without an accuracy predictor; and tabular whole-configuration regression.Optional baselines include single-node XGBoost prediction, exhaustive search on small models, GNN without edge attributes, and GNN without reverse edges.All search methods should be compared under equal real-evaluation budgets.

25. Implementation Status and Plan

Implemented components are Posit operation representation, external-library integration, Posit-aware lowering, node-level format assignment, mixed-Posit compilation/execution, and low-bit Posit8/16/32 storage.Remaining work is to preserve ONNX-to-Posit node mapping, construct the Posit graph and joined ONNX/Posit features; implement storage and liveness analysis; generate and evaluate initial configurations; train the GNN; integrate NSGA-II and Pareto search; update the predictor with real evaluations; and compare methods under equal budgets.

26. Scope and Limitations

Latency is not a primary objective in the first version because runtime may be dominated by software emulation, external calls, conversions, and non-vectorized kernels.Latency may be reported as a secondary implementation metric.Prediction quality depends on configuration diversity and label quantity.Activation statistics depend on the calibration dataset.Peak activation memory is a logical static estimate and may exclude backend workspace or allocator overhead.Reverse edges support GNN message passing but do not represent execution dependencies.Fake-quant error and single-node sensitivity are optional initial features.Cross-model generalization may require several model families.The full configuration space cannot be exhaustively evaluated for large models.

27. Expected Contributions

Expected contributions are a Posit Dialect operation graph linked to original ONNX node metadata, a configuration-aware GNN accuracy predictor, static storage and liveness calculators, surrogate-assisted NSGA-II, an iterative real-evaluation loop, and an empirical analysis of accuracy-storage-memory trade-offs.

28. Success Criteria

Success requires executable configurations, useful prediction and ranking quality, reduced storage and peak memory, acceptable real accuracy loss, fewer expensive evaluations, improved generalization over tabular baselines, and clear Pareto trade-offs.

29. Final Summary

(\min_P\left(AccuracyLoss(P),WeightStorage(P),EstimatedPeakActivationMemory(P)\right))The ML task is:(f(G_{posit}(P),X_{onnx},X_{posit})\rightarrow\widehat{AccuracyLoss}(P))The Configuration Generator operates on ONNX node identities and determines which ONNX nodes are selectively lowered to each Posit format.After applying the configuration, the graph is extracted from the Posit Dialect, where nodes represent lowered Posit operations and edges represent Posit SSA tensor producer-consumer dependencies.Cached ONNX metadata is joined to Posit nodes through the preserved node identity.Reverse edges are added only for bidirectional GNN message passing.The predictor uses structural, graph, weight-distribution, activation-distribution, and Posit-configuration features.Initial configurations are generated through stratified random sampling and representative configurations.After initial training, surrogate-assisted NSGA-II generates complete configurations.Weight storage and peak activation memory are calculated through static analysis.Selected Pareto configurations are compiled and executed by the real Posit backend.Measured results are added to the dataset and used to update the Accuracy Predictor.The final framework combines compiler IR graph representation, whole-configuration prediction, static cost analysis, multi-objective search, and real execution feedback.