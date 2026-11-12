## Table 1: same caps (method - BC, top-1 pp, 95% bootstrap CI, Holm-adjusted McNemar)

| model | point | R cap | SQNR | HTQ | HAWQ | InfoQ |
|---|---|---|---|---|---|---|
| ResNet18 | loose | 0.8124 | same config | -0.20 [-0.70, +0.31] undetermined | same config | -1.12 [-1.88, -0.39] BC better |
| ResNet18 | knee | 0.6247 | same config | -0.70 [-1.54, +0.14] undetermined | same config | -3.89 [-4.87, -2.94] BC better |
| ResNet18 | aggressive | 0.5787 | same config | same config | same config | -3.39 [-4.37, -2.44] BC better |
| EfficientNet-B0 | loose | 0.8205 | -1.48 [-2.52, -0.48] BC better | -2.35 [-3.30, -1.40] BC better | -0.48 [-1.48, +0.50] undetermined | not run |
| EfficientNet-B0 | knee | 0.6410 | -0.50 [-1.60, +0.59] undetermined | -15.87 [-17.30, -14.45] BC better | -0.53 [-1.51, +0.48] undetermined | not run |
| EfficientNet-B0 | aggressive | 0.6317 | -0.25 [-1.29, +0.78] undetermined | -9.74 [-11.09, -8.40] BC better | +1.09 [+0.11, +2.04] undetermined | not run |

## Table 2: common auto shell (top-1 %, R)

| model | BC | SQNR | HTQ | HAWQ | InfoQ |
|---|---|---|---|---|---|
| ResNet18 | 67.75 (R 0.625) | 67.53 (R 0.623) | 68.11 (R 0.673) | 68.53 (R 0.672) | 63.05 (R 0.594) |
| EfficientNet-B0 | 39.84 (R 0.641) | 39.56 (R 0.641) | 23.49 (R 0.636) | 39.31 (R 0.641) | not run |

## Table C: auto pick - BC re-searched at the same R (top-1 pp)

| model | method | R | method top-1 | BC top-1 | diff [CI] | verdict | source |
|---|---|---|---|---|---|---|---|
| ResNet18 | HAWQ | 0.6716 | 68.53 | 68.42 | +0.11 [-0.56, +0.76] | undetermined | followup_matched (post hoc, pre-registered) |
| ResNet18 | HTQ | 0.6734 | 68.11 | 68.25 | -0.14 [-0.81, +0.56] | undetermined | followup_matched (post hoc, pre-registered) |
| EfficientNet-B0 | SQNR | 0.6409 | 39.56 | 40.26 | -0.70 [-1.74, +0.34] | undetermined | pre-registered C |
| EfficientNet-B0 | HTQ | 0.6362 | 23.49 | 36.87 | -13.38 [-14.81, -11.98] | BC better | pre-registered C |
| EfficientNet-B0 | HAWQ | 0.6410 | 39.31 | 40.57 | -1.26 [-2.21, -0.28] | BC better | pre-registered C |

## Table 3: offline scoring cost (median table time, s; ratio vs BC; repetitions)

| model | BC | SQNR | HTQ | HAWQ | InfoQ |
|---|---|---|---|---|---|
| ResNet18 | 16.1 (1.00x, n=3) | 11.3 (0.70x, n=3) | 0.8 (0.05x, n=3) | 27.3 (1.69x, n=3) | 600.7 (37.28x, n=2) |
| EfficientNet-B0 | 44.8 (1.00x, n=3) | 28.0 (0.63x, n=3) | 1.1 (0.02x, n=3) | 3511.5 (78.45x, n=1) | not run |
