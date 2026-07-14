# ML-Guided Mixed-Posit Precision Tuning on ONNX-MLIR

## 1. Project Overview
This project investigates an **ML-guided layer-wise / entity-wise posit precision tuning framework on ONNX-MLIR**.

Core goal:
> Given a trained ONNX model and a user-defined error tolerance, select a posit format for each layer, op group, or computation entity while **minimizing total precision cost** and keeping whole-model output error or accuracy loss within tolerance.

This is not whole-model single-format posit conversion. The goal is **mixed-posit assignment**:
```text
Conv_0   -> posit_16_1
Relu_1   -> posit_8_2
MatMul_2 -> posit_16_2
Add_3    -> posit_8_1
Softmax_4 -> FP32
```

Target formats: `posit_8_0`, `posit_8_1`, `posit_8_2`, `posit_16_0`, `posit_16_1`, `posit_16_2`, `posit_32_0`, `posit_32_1`, `posit_32_2`, and `FP32`.

Core idea:
1. Use an **FPLearner-style ML cost model** to predict promising layer-format candidates and reduce search space.
2. Use a **TuneQn-style selective search / validation process** to find a layer-wise mixed-posit plan.
3. Use a **cost-aware planner** so the final plan is not merely safe, but also lower-cost than FP32 / high-precision baselines.
4. Apply the selected plan through **ONNX-MLIR annotation and later posit lowering passes**.

## 2. Research Problem and Objective
The research should not be defined only as:
```text
Find any posit plan such that whole_model_error <= tolerance.
```

That formulation is too weak because a trivial high-precision plan, such as all FP32 or all posit32, can satisfy the tolerance without reducing cost.

Instead, this project defines posit format selection as a **tolerance-constrained cost minimization problem**:
```text
minimize    total_precision_cost(plan)
subject to  whole_model_error(plan) <= tolerance
```

A simple first version of total precision cost can be:
```text
total_precision_cost(plan) = Σ bit_width(format_i) × tensor_size_i
```

where `format_i` is the chosen format for layer/entity `i`, and `tensor_size_i` can be weight size, activation size, or weight + activation size.

This makes the trade-off explicit:
```text
lower precision cost  <->  higher possible error risk
higher precision cost <->  lower possible error risk
```

Therefore, the planner should search for the **lowest-cost feasible mixed-posit plan**, not simply the safest plan.

## 3. Research Direction
Current direction:
> Build an ONNX-MLIR-based framework that uses an ML cost model to predict promising posit formats for each layer/entity, keeps only Top-K candidates, and then performs cost-aware selective search to produce a mixed-posit precision plan.

Main contribution targets:
1. ONNX-MLIR-based layer/entity-wise posit precision tuning.
2. ML-based Top-K layer-posit candidate selection.
3. Cost-aware tolerance-constrained planning.
4. TuneQn-style candidate evaluation with tolerance-based and Pareto-style analysis.
5. Evaluation of whether ML-guided search reduces search space while preserving model accuracy and reducing precision cost.

## 4. Why ONNX-MLIR?
ONNX-MLIR lowering pipeline:
```text
ONNX model -> ONNX Dialect -> Krnl / SCF / Affine lowering -> LLVM IR -> Code generation
```

The **ONNX Dialect** stage preserves layer-level semantics, so posit decisions can be attached to high-level operations such as `onnx.Conv`, `onnx.Gemm`, `onnx.MatMul`, `onnx.Add`, `onnx.Relu`, and `onnx.Softmax`.

After lowering to Krnl, SCF, or LLVM IR, one ONNX layer may become loops, memory operations, and arithmetic instructions. Therefore, the first posit selection / annotation should happen **before full lowering**.

Recommended placement:
```text
ONNX model
  -> ONNX Dialect
  -> [Posit Selection / Annotation Pass]
  -> Lower ONNX to Krnl / SCF
  -> [Optional Posit Propagation / Refinement Pass]
  -> Lower to LLVM
```

## 5. Main Pipeline
```text
Original ONNX model
  -> Import into ONNX-MLIR
  -> ONNX Dialect
  -> Layer / entity identification
  -> Feature extraction
  -> ML-based cost model
  -> Top-K posit candidate selection
  -> Reduced posit search space
  -> Cost-aware planner / selective search
  -> Validation / benchmarking
  -> Tolerance-based selection + Pareto-style analysis
  -> Layer-wise posit precision plan
  -> ONNX-MLIR annotation pass
  -> Optional type rewrite / posit lowering
```

## 6. ML-Based Cost Model
The ML model is a **cost model / error-risk predictor**, not the final decision maker.

Input features:
- **Layer / structural:** layer type, op type, input/output shape, parameter count, FLOPs, graph depth, fan-in/fan-out, residual indicator, FP op count, arithmetic counts, memory access pattern.
- **Runtime / profiling:** mean, std, min, max, abs max, p01, p50, p99, near-zero ratio, positive/negative ratio, outlier ratio, skewness, kurtosis, activation range, weight range.
- **Posit metadata:** `nbits`, `es`, dynamic range, precision behavior, estimated bit cost, estimated runtime or hardware cost.

Possible outputs:
```text
f(layer_features, posit_format) -> predicted_relative_error
f(layer_features, posit_format, tolerance) -> feasible / infeasible
f(layer_features, posit_format) -> quantization_risk_score
```

Initial recommendation: start with risk score or feasibility prediction, because exact error prediction is harder and can be affected by error propagation across layers.

## 7. Top-K Layer-Posit Candidate Selection
For each layer/entity, the ML cost model predicts the risk, error, or feasibility of each posit format. Instead of allowing every format for every layer, only the **Top-K promising candidates** are kept.

| Layer | Original Candidates | After Top-K |
|---|---|---|
| `Conv_0` | all posit formats + FP32 | `posit_16_1`, `posit_16_2`, `posit_32_0` |
| `Relu_1` | all posit formats + FP32 | `posit_8_1`, `posit_8_2`, `posit_16_0` |
| `MatMul_2` | all posit formats + FP32 | `posit_16_2`, `posit_32_0`, `posit_32_1` |
| `Softmax_4` | all posit formats + FP32 | `posit_32_0`, `posit_32_1`, `FP32` |

Search-space comparison:
```text
Exhaustive search with 30 entities: 10^30
Top-2 per entity: 2^30
Top-3 per entity: 3^30
```

The ML model only decides **which posit formats are worth trying**. The final choice is made by the planner and validated by benchmarking.

## 8. Why Top-K Is Preferred
**Method 3: Top-K Layer-Format Candidate Selection** is preferred because it is easier to implement, easier to explain, does not require the ML model to directly choose the final format, keeps validation / benchmarking as the final decision step, and makes search-space reduction clear and measurable.

**Method 4: ML-Guided Layer-Format Ordering** ranks transition actions, such as `Layer_3: posit_8_0 -> posit_16_0`. This is more complex because posit formats are not ordered only by bit-width. The `es` value changes dynamic range and precision distribution, so transition paths are not always obvious.

Conclusion: use Top-K candidate selection as the main method. Treat ML-guided ordering as optional future work.

## 9. Cost-Aware Search and Candidate Plan Generation
After Top-K candidate selection, the reduced search space is passed to a cost-aware selective search process.

Possible search methods:
- Greedy search
- Beam search
- Sensitivity-based candidate generation
- Pareto Front analysis
- Tolerance-based filtering

Planner objective:
```text
minimize    Σ cost(layer_i, format_i)
subject to  estimated_whole_model_error(plan) <= tolerance
```

A practical cost function:
```text
cost(layer_i, format_i) = bit_width(format_i) × tensor_size(layer_i)
```

Greedy plan generation:
```text
1. Start from a safe high-precision plan, such as FP32 or posit32.
2. Use the ML cost model to predict risk/error for each layer-format pair.
3. Keep only Top-K candidates for each layer/entity.
4. For each possible replacement, compute cost_saved and error_increase.
5. Rank replacements by cost_saved / error_increase.
6. Try the best replacement.
7. If accuracy loss is within tolerance, keep the replacement.
8. If accuracy loss exceeds tolerance, rollback the replacement.
9. Repeat until no valid lower-cost replacement can be applied.
```

Cost-aware greedy score:
```text
score = cost_saved / (sensitivity_weighted_error_increase + epsilon)
```

This avoids choosing high precision simply because it is safe. The planner is forced to look for lower-cost feasible replacements.

## 10. High-Precision Outcome Is Possible but Interpretable
It is possible that the final tuned model still uses many high-precision formats, especially when:
- The tolerance is very strict.
- The model is sensitive to low-precision arithmetic.
- The error predictor or error aggregation is conservative.
- Some operations, such as MatMul, Conv, Softmax, or reductions, are numerically sensitive.

This is not necessarily a failure. It may indicate that the model has limited low-precision opportunity under that tolerance.

Evaluation should therefore include:
```text
1. tolerance sweep, such as 1e-2, 1e-3, 1e-4, 1e-5
2. percentage of posit8 / posit16 / posit32 / FP32 selected
3. Pareto-style trade-off between error and precision cost
4. comparison between predicted error and actual error
```

The purpose of mixed-precision tuning is not to force every layer into low precision. The goal is to automatically identify which layers can safely use low precision and which layers must remain high precision.

## 11. Precision Plan Format
The ML cost model and planner should output a JSON file containing strategy, tolerance, candidate formats, cost, and per-entity decisions.

```json
{
  "strategy": "cost_aware_top_k_posit_search",
  "tolerance": 0.001,
  "objective": "minimize_precision_cost_under_error_tolerance",
  "formats": ["posit_8_0", "posit_8_1", "posit_8_2", "posit_16_0", "posit_16_1", "posit_16_2", "posit_32_0", "posit_32_1", "posit_32_2", "FP32"],
  "entities": {
    "Conv_0": {
      "chosen": "posit_16_1",
      "candidates": ["posit_16_1", "posit_16_2", "posit_32_0"],
      "pred_error": 0.00042,
      "risk_score": 0.18,
      "precision_cost": 12582912
    }
  },
  "estimated_whole_model_error": 0.00087,
  "total_precision_cost": 216000000
}
```

## 12. ONNX-MLIR Pass Structure
### 12.1 ONNX-Level Posit Annotation Pass
Runs on ONNX Dialect. It reads `precision_plan.json`, matches plan entries to ONNX operations, attaches posit attributes to ONNX ops, and starts with annotation-only mode.

Example:
```mlir
%0 = "onnx.Conv"(%input, %weight, %bias) {
  posit.chosen = "posit_16_1",
  posit.candidates = ["posit_16_1", "posit_16_2", "posit_32_0"],
  posit.pred_error = 0.00042,
  posit.precision_cost = 12582912
} : (...) -> tensor<...>
```

### 12.2 Krnl / SCF-Level Posit Propagation Pass
Runs after ONNX-to-Krnl lowering. It checks whether posit attributes survive lowering, propagates source layer information to generated loops or computation regions, and attaches `posit.source_layer` and `posit.chosen` to Krnl or SCF operations.

### 12.3 Future Type Rewrite / Posit Lowering Pass
Later-stage work: rewrite selected floating-point operations to posit-aware operations, insert posit runtime or library calls, lower posit operations to LLVM-compatible code, and connect to a posit backend such as Stillwater Universal.

Conceptual lowering:
```text
fadd -> posit_add
fmul -> posit_mul
matmul -> posit_matmul_kernel
```

## 13. Posit Execution Method
The final goal is to evaluate real posit execution inside the ONNX-MLIR compilation flow.

Execution flow:
```text
ONNX model -> ONNX-MLIR import -> ONNX Dialect -> Apply posit precision plan -> Lower to Krnl / SCF / LLVM -> Generate executable code -> Run validation and benchmarking
```

The selected posit operations can be lowered to posit-aware runtime or library calls. A possible backend is the Stillwater Universal posit library.

Measured outputs: real model output, accuracy, accuracy loss, precision cost, model size, runtime behavior, and latency if supported.

## 14. Relation to TuneQn and FPLearner
TuneQn is a conceptual reference for selective quantization, candidate generation, benchmarking, accuracy/model-size measurement, and Pareto Front selection.

| Aspect | TuneQn | This Project |
|---|---|---|
| Platform | ONNX + ONNX Runtime / TVM | ONNX-MLIR |
| Format | INT quantization | Posit 8/16/32 with es=0/1/2 |
| Unit | Layer | Layer / op group / entity |
| Search reduction | Sensitivity ranking | ML-based Top-K candidate selection |
| Final objective | Accuracy-size trade-off | Minimize precision cost under error tolerance |
| Output | Quantized ONNX model | Posit plan + MLIR annotations |
| Lowering | ONNX Quantizer | ONNX-MLIR pass / posit runtime |

FPLearner provides the idea that ML can reduce precision-tuning search space:
```text
Do not exhaustively search all precision configurations.
Use ML to identify promising precision candidates first.
Then search only the reduced space.
```

Differences: this project targets ONNX / ONNX-MLIR, performs layer-wise or entity-wise posit precision tuning, and tunes posit formats rather than ordinary floating-point precision choices.

## 15. Experimental Evaluation Design
The evaluation verifies:
1. Whether the selected mixed-posit precision plan maintains acceptable model accuracy.
2. Whether selected posit formats reduce precision cost, such as model size, average bit-width, or activation memory.
3. Whether the ML-based cost model reduces the precision tuning search space compared with exhaustive or unguided search.
4. Whether the final planner avoids trivial all-high-precision solutions when lower-cost feasible plans exist.

Evaluation models: MobileNet, ShuffleNet, EfficientNet, and ResNet.

## 16. Baseline and Metrics
Baseline:
- Original FP32 ONNX model.
- Optional all-posit32 model if posit32 execution is supported.
- Optional random search and unguided greedy search.

Main metrics:
```text
Accuracy = correct predictions / total predictions
Accuracy Loss = FP32 Accuracy - Mixed-Posit Accuracy
Weight Cost = number_of_weights × bit_width
Activation Cost = number_of_activation_elements × bit_width
Total Precision Cost = Weight Cost + Activation Cost
Model Size Reduction = 1 - Mixed-Posit Model Size / FP32 Model Size
Search Space Reduction Ratio = 1 - K^N / F^N
```

Also report:
- Number of evaluated plans.
- Number of accepted / rejected plans.
- Search time if supported.
- Distribution of selected formats: posit8, posit16, posit32, FP32.

## 17. Search Space Reduction Evaluation
Assume:
```text
N = number of quantizable layers/entities
F = number of original candidate formats per layer
K = number of Top-K candidates kept per layer
```

In this project:
```text
F = 10
Original Search Space = F^N = 10^N
Reduced Search Space = K^N
Search Space Reduction Ratio = 1 - K^N / F^N
```

Example with Top-3:
```text
Original Search Space = 10^N
Reduced Search Space = 3^N
Reduction Ratio = 1 - 3^N / 10^N
```

## 18. Comparison Against Random Search
To show that the cost model is useful, compare ML-guided Top-K search with random search under the same evaluation budget.

Example:
```text
If ML-guided search evaluates 100 candidate plans,
random search should also be limited to 100 candidate plans.
```

Compare best accuracy loss, best precision cost reduction, number of valid plans, and whether each method finds a plan within tolerance.

The proposed method is better if it finds a lower-cost valid plan than random search under the same number of evaluations.

## 19. Success Criteria
The proposed search-space reduction method is considered effective if:
1. The reduced search space is much smaller than the original exhaustive search space.
2. The number of evaluated candidate plans is lower than exhaustive or unguided search.
3. The final mixed-posit plan keeps accuracy loss within tolerance.
4. The final mixed-posit plan reduces precision cost compared with the FP32 or all-high-precision baseline.
5. The final plan is comparable to or better than random search under the same evaluation budget.

Therefore, the ML cost model is not evaluated only by prediction accuracy. It is also evaluated by whether it can guide the planner to search fewer candidate plans while still finding a valid low-cost mixed-posit configuration.

## 20. Current Implementation Plan
### Stage 1: Annotation-Only Prototype
Goal: generate a precision plan, insert ONNX-level posit annotations, and verify layer/entity mapping.
Tasks: import ONNX model, identify ONNX operations/entities, extract features, run ML cost model, keep Top-K candidates, generate `precision_plan.json`, implement `PositAnnotationPass`, and confirm attributes are attached correctly.

### Stage 2: Propagation Through Lowering
Goal: preserve or propagate posit annotations from ONNX ops to Krnl/SCF loops.
Tasks: lower annotated ONNX Dialect to Krnl/SCF, track source layer/entity IDs, attach `posit.source_layer` and `posit.chosen`, and verify mapping correctness.

### Stage 3: Candidate Evaluation
Goal: evaluate whether selected posit plans satisfy tolerance and reduce precision cost.
Tasks: generate candidate plans from Top-K sets, execute or simulate posit behavior, compare output against FP64 or FP32 reference, measure error/cost/size/latency, and select the lowest-cost feasible plan.

### Stage 4: Posit Lowering
Goal: actually execute selected posit operations.
Tasks: define posit operation representation, rewrite selected operations or regions to posit-aware calls, connect to a posit runtime/library implementation, and validate correctness and performance.

## 21. Final Summary
This project builds an ONNX-MLIR-based ML-guided framework that predicts promising posit formats for each layer/entity, reduces the mixed-posit search space using Top-K candidate selection, and generates a layer-wise posit precision plan through cost-aware selective search.

The central formulation is:
```text
minimize total_precision_cost(plan)
subject to whole_model_error(plan) <= tolerance
```

The final goal is to show that ML-guided Top-K search can reduce precision tuning search space while still finding a mixed-posit plan that satisfies accuracy tolerance and reduces precision cost.
