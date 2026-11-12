# 更正：token-spike 輪的文章重複

時間：2026-10-23T04:14:19+0800

## 原因
- run_olmoe_selective_sensitivity.used_wt103_all classified manifests only by their source / split fields; runs/olmoe_act_outlier_diag_20261023/segments.json has neither, so its 64 WikiText-103 articles were not excluded when the token-spike round drew its confirmation articles

## 修正
- wt_manifest_scan.used_wt103_verified (content-hash membership; manifests without hashes must be shown to be WikiText-2; unresolved manifests are errors); the old function is kept unchanged for the recorded code hashes

## 驗證
- 舊掃描 6074 篇，驗證掃描 6138 篇（舊集合是子集：True）；舊掃描漏掉 64 篇，全部來自缺來源欄位的 manifest：{'runs/olmoe_act_outlier_diag_20261023/segments.json': 64}。
- 已知重複：[{'index': 39, 'title': 'Brushstrokes series', 'content_sha256': 'ea9b0017ead03f3347c2c30806fa68b0491f41a67e01aaa1355a6657dddf686e', 'in_old_scanner': False}]；驗證掃描會排除，舊掃描不會。近似重複（shingle ≥ 0.1）：{'39': 1.0}。
- 沒有 hash 的 manifest 全部以證據判定為 WikiText-2（WT2 文章本來就從 WT103 抽樣池整批排除）：
  - runs/olmoe_calibration_stability_20261018/segments.json：55 titles are WikiText-2 articles; 1 by WikiText-2 character offsets with the run plan naming WikiText-2 (plan names WikiText-2)
  - runs/olmoe_crosstext_20261017/segments.json：32 titles are WikiText-2 articles; 0 by WikiText-2 character offsets with the run plan naming WikiText-2 (plan names WikiText-2)
  - runs/olmoe_dynamic_multipoint_20261018/segments.json：300 titles are WikiText-2 articles; 0 by WikiText-2 character offsets with the run plan naming WikiText-2 (plan names WikiText-2)
  - runs/olmoe_dynamic_route_20261018/segments.json：128 titles are WikiText-2 articles; 0 by WikiText-2 character offsets with the run plan naming WikiText-2 (plan names WikiText-2)
  - runs/olmoe_first_order_validation_20261017/segments.json：16 titles are WikiText-2 articles; 0 by WikiText-2 character offsets with the run plan naming WikiText-2 (plan does not name the corpus)
  - runs/olmoe_route_decomposition_20261018/segments.json：156 titles are WikiText-2 articles; 0 by WikiText-2 character offsets with the run plan naming WikiText-2 (plan names WikiText-2)

## 以 95 篇重算（seed 不變；bootstrap 抽樣因篇數不同而不同）
```
{
 "articles": 95,
 "ranking": {
  "all_new": {
   "pairs": 180,
   "decisive": 63,
   "undeterminable": 117,
   "bc_ties": 0,
   "decisive_bc_ties": 0,
   "evaluated": 63,
   "correct": 54,
   "accuracy": 0.8571428571428571,
   "abs_diff_lt_1e-6": 1
  },
  "known_case": {
   "pairs": 15,
   "decisive": 9,
   "undeterminable": 6,
   "bc_ties": 0,
   "decisive_bc_ties": 0,
   "evaluated": 9,
   "correct": 8,
   "accuracy": 0.8888888888888888,
   "abs_diff_lt_1e-6": 0
  },
  "same_bits_new": {
   "pairs": 36,
   "decisive": 8,
   "undeterminable": 28,
   "bc_ties": 0,
   "decisive_bc_ties": 0,
   "evaluated": 8,
   "correct": 8,
   "accuracy": 1.0,
   "abs_diff_lt_1e-6": 1
  },
  "family:int-int": {
   "pairs": 36,
   "decisive": 13,
   "undeterminable": 23,
   "bc_ties": 0,
   "decisive_bc_ties": 0,
   "evaluated": 13,
   "correct": 9,
   "accuracy": 0.6923076923076923,
   "abs_diff_lt_1e-6": 0
  },
  "family:posit-posit": {
   "pairs": 36,
   "decisive": 16,
   "undeterminable": 20,
   "bc_ties": 0,
   "decisive_bc_ties": 0,
   "evaluated": 16,
   "correct": 16,
   "accuracy": 1.0,
   "abs_diff_lt_1e-6": 0
  },
  "family:cross": {
   "pairs": 108,
   "decisive": 34,
   "undeterminable": 74,
   "bc_ties": 0,
   "decisive_bc_ties": 0,
   "evaluated": 34,
   "correct": 29,
   "accuracy": 0.8529411764705882,
   "abs_diff_lt_1e-6": 1
  }
 },
 "P4": {
  "int4": {
   "mean_diff": -0.0004736839686769431,
   "ci97.5": [
    -0.0011617047605631022,
    0.0001831079182831688
   ],
   "ci95": [
    -0.0010704232006948581,
    0.00010653676468805497
   ],
   "primary": true
  },
  "int6": {
   "mean_diff": 0.0004805769001635655,
   "ci97.5": [
    -4.244821246995662e-05,
    0.001056632138001247
   ],
   "ci95": [
    2.0317228997618193e-05,
    0.0009763796857241081
   ],
   "primary": false
  },
  "int8": {
   "mean_diff": -4.064508164048509e-05,
   "ci97.5": [
    -0.0003721752410227664,
    0.0003047094852278871
   ],
   "ci95": [
    -0.00033530863828353663,
    0.0002618386675890677
   ],
   "primary": false
  },
  "p4e1": {
   "mean_diff": 0.0013440668725015076,
   "ci97.5": [
    0.0006148974175607763,
    0.0020723815671336767
   ],
   "ci95": [
    0.0006995370605875673,
    0.0019781000850749887
   ],
   "primary": true
  },
  "p6e1": {
   "mean_diff": 0.00012821375631895613,
   "ci97.5": [
    -0.00011386186135133395,
    0.00037298315357159713
   ],
   "ci95": [
    -8.171614663173307e-05,
    0.000343687020488769
   ],
   "primary": false
  },
  "p8e1": {
   "mean_diff": -0.00010701375090182766,
   "ci97.5": [
    -0.0002281322666211383,
    1.4030956436152016e-05
   ],
   "ci95": [
    -0.0002112393219239408,
    -2.6394962649435336e-06
   ],
   "primary": false
  }
 },
 "known_case": {
  "now": [
   {
    "f1": "int4",
    "f2": "p6e1",
    "measured_diff": 0.0035481530846586977,
    "ci95": [
     0.0019400384252391057,
     0.0051600618196681255
    ],
    "decisive": true,
    "bc_correct": false
   },
   {
    "f1": "int6",
    "f2": "p6e1",
    "measured_diff": 0.0012618022892895622,
    "ci95": [
     -1.0202144746307635e-05,
     0.002512745340892946
    ],
    "decisive": false,
    "bc_correct": false
   },
   {
    "f1": "int8",
    "f2": "p6e1",
    "measured_diff": 0.0003988169597749088,
    "ci95": [
     -0.00044795888980097664,
     0.0012500906130993765
    ],
    "decisive": false,
    "bc_correct": false
   },
   {
    "f1": "int8",
    "f2": "p8e1",
    "measured_diff": 0.00023798715351688662,
    "ci95": [
     -0.0005978236329716235,
     0.0010971308077830783
    ],
    "decisive": false,
    "bc_correct": false
   }
  ],
  "replicated": 1
 },
 "misranked": [
  {
   "tap": 3,
   "node": "L1.E37",
   "f1": "int8",
   "f2": "p6e1",
   "diff": 0.00044558988001892826,
   "ci95": [
    1.6123857860090657e-05,
    0.0008962456747173932
   ]
  },
  {
   "tap": 6,
   "node": "L6.E59",
   "f1": "int4",
   "f2": "int6",
   "diff": -0.0024215745532516868,
   "ci95": [
    -0.0033525899010677703,
    -0.0015189129030832435
   ]
  },
  {
   "tap": 6,
   "node": "L6.E59",
   "f1": "int4",
   "f2": "int8",
   "diff": -0.0024766483658515024,
   "ci95": [
    -0.0036615994624033866,
    -0.0013400138289145424
   ]
  },
  {
   "tap": 6,
   "node": "L6.E59",
   "f1": "int4",
   "f2": "p6e1",
   "diff": -0.0021437305862561586,
   "ci95": [
    -0.0033683612575213903,
    -0.0009509288430090704
   ]
  },
  {
   "tap": 6,
   "node": "L6.E59",
   "f1": "int4",
   "f2": "p8e1",
   "diff": -0.0021171038147434285,
   "ci95": [
    -0.0033508583465245256,
    -0.0009166229075398951
   ]
  },
  {
   "tap": 11,
   "node": "L2.E14",
   "f1": "int4",
   "f2": "int6",
   "diff": -0.0021003241549079946,
   "ci95": [
    -0.0032485688553677105,
    -0.0009392571941977703
   ]
  },
  {
   "tap": 11,
   "node": "L2.E14",
   "f1": "int4",
   "f2": "int8",
   "diff": -0.001771296684345621,
   "ci95": [
    -0.002841905169281933,
    -0.0006409612931547571
   ]
  },
  {
   "tap": 11,
   "node": "L2.E14",
   "f1": "int4",
   "f2": "p6e1",
   "diff": -0.0019021969827887526,
   "ci95": [
    -0.0030603810621943283,
    -0.0007045054863492554
   ]
  },
  {
   "tap": 11,
   "node": "L2.E14",
   "f1": "int4",
   "f2": "p8e1",
   "diff": -0.00167485058133817,
   "ci95": [
    -0.0028608678800852902,
    -0.0004516381359858321
   ]
  },
  {
   "tap": 12,
   "node": "L1.E9",
   "f1": "int4",
   "f2": "p6e1",
   "diff": 0.0035481530846586977,
   "ci95": [
    0.0019400384252391057,
    0.0051600618196681255
   ]
  }
 ],
 "per_tap_dnll": {
  "L0.E46": {
   "int4": 0.000980330953787772,
   "int6": 0.0004515835078565883,
   "int8": -0.0002939319292189209,
   "p4e1": 0.00412042732293775,
   "p6e1": -5.1524313942784485e-05,
   "p8e1": -0.00017521320676154286
  },
  "L0.E12": {
   "int4": -0.0006126857766314905,
   "int6": -0.00012540692205022187,
   "int8": -0.00012780314427662275,
   "p4e1": 0.0014121770681230883,
   "p6e1": -0.0003056028189378985,
   "p8e1": -5.513913192195048e-05
  },
  "L1.E49": {
   "int4": 0.0
```
