TABLE 4
| Estimator | Brier (19→20) | MCB (19→20) | Brier (20→19) | MCB (20→19) | Pooled Brier | Pooled MCB [95% CI] | Pooled DSC |
|---|---|---|---|---|---|---|---|
| constant (training prevalence) | 0.173 | 0.029 | 0.255 | 0.029 | 0.211 | 0.022 [0.016, 0.029] | 0.000 |
| norm | 0.195 | 0.058 | 0.246 | 0.029 | 0.219 | 0.038 [0.024, 0.057] | 0.008 |
| Platt | 0.177 | 0.037 | 0.252 | 0.036 | 0.211 | 0.023 [0.015, 0.034] | 0.001 |
| isotonic | 0.177 | 0.037 | 0.250 | 0.033 | 0.211 | 0.024 [0.017, 0.035] | 0.002 |
| norm+Platt | 0.163 | 0.026 | 0.245 | 0.028 | 0.201 | 0.013 [0.009, 0.020] | 0.001 |
| norm+isotonic | 0.165 | 0.027 | 0.246 | 0.029 | 0.202 | 0.015 [0.011, 0.022] | 0.002 |
UNC: 19->20 0.144, 20->19 0.226, pooled 0.189

TABLE 7 NEW ROWS
| QRC-mean (norm+Platt) − QRC-mean (norm) | calibration on vs. off, same input | 0.000 [−0.006, 0.007] |
| QRC-mean (norm+isotonic) − QRC-mean (norm) | calibration on vs. off, same input | +0.006 [−0.044, 0.059] |

TABLE A1
| Quantity (pooled unless stated) | Point estimate | 95% CI, calibrators fixed | 95% CI, calibrators refitted | Interval excludes 0 (fixed / refitted) |
|---|---|---|---|---|
| Brier, norm+Platt − Platt | −0.0105 | [−0.0178, −0.0035] | [−0.0244, −0.0001] | yes / yes |
| Brier, norm+isotonic − isotonic | −0.0088 | [−0.0167, −0.0018] | [−0.0239, 0.0020] | yes / no |
| Brier, norm+Platt − norm | −0.0179 | [−0.0391, 0.0031] | [−0.0487, 0.0163] | no / no |
| Brier, Platt − constant | +0.0002 | [−0.0076, 0.0079] | [−0.0104, 0.0133] | no / no |
| Brier, norm+Platt − constant | −0.0102 | [−0.0152, −0.0052] | [−0.0174, −0.0030] | yes / yes |
| Brier, Platt − norm (19→20) | −0.0187 | [−0.0351, −0.0033] | [−0.0461, 0.0185] | yes / no |
| Brier, norm+Platt − norm (19→20) | −0.0321 | [−0.0433, −0.0213] | [−0.0546, −0.0042] | yes / yes |
| MCB, norm+Platt − Platt | −0.0102 | [−0.0176, −0.0030] | [−0.0241, 0.0003] | yes / no |
| MCB, norm+isotonic − isotonic | −0.0084 | [−0.0156, −0.0028] | [−0.0227, 0.0011] | yes / no |
| MCB, norm+Platt − norm | −0.0246 | [−0.0424, −0.0081] | [−0.0511, 0.0056] | yes / no |
| MCB, norm+isotonic − norm | −0.0225 | [−0.0410, −0.0056] | [−0.0500, 0.0094] | yes / no |
| MCB, norm+Platt − constant | −0.0088 | [−0.0126, −0.0037] | [−0.0151, −0.0003] | yes / yes |
| MCB, norm+isotonic − constant | −0.0067 | [−0.0104, −0.0021] | [−0.0124, 0.0012] | yes / no |
| AUROC, QRC-mean (Platt) − mean@10 | −0.002 | [−0.008, 0.000] | [−0.009, 0.002] | no / no |
| AUROC, QRC-mean (norm+Platt) − QRC-mean (Platt) | −0.046 | [−0.185, 0.106] | [−0.185, 0.114] | no / no |
| AUROC, QRC-mean (norm+isotonic) − QRC-mean (isotonic) | −0.031 | [−0.189, 0.130] | [−0.191, 0.133] | no / no |
| AUROC, QRC-mean (norm+Platt) − QRC-mean (norm) | 0.000 | [−0.006, 0.007] | [−0.009, 0.009] | no / no |
| AUROC, QRC-mean (Platt) − CV@100 | −0.123 | [−0.272, 0.032] | [−0.276, 0.032] | no / no |
| AUROC, QRC-mean (isotonic) − CV@100 | −0.132 | [−0.291, 0.025] | [−0.275, 0.036] | no / no |
| AUROC, QRC-mean (norm) − CV@100 | −0.170 | [−0.307, −0.018] | [−0.307, −0.018] | yes / yes |
| AUROC, QRC-mean (norm+Platt) − CV@100 | −0.170 | [−0.307, −0.017] | [−0.306, −0.018] | yes / yes |
| AUROC, QRC-mean (norm+isotonic) − CV@100 | −0.163 | [−0.305, −0.015] | [−0.292, −0.007] | yes / yes |
| AUROC, QRC-max (Platt) − CV@100 | −0.107 | [−0.254, 0.046] | [−0.257, 0.043] | no / no |
| AUROC, QRC-max (isotonic) − CV@100 | −0.096 | [−0.245, 0.055] | [−0.242, 0.055] | no / no |

TABLE A2
| Signal | AUROC DL19 | AUROC DL20 | AUROC stratified [95% CI] | Δ vs. CV@100 [95% CI] |
|---|---|---|---|---|
| CV@100 | 0.716 | 0.741 | 0.731 [0.625, 0.824] | — |
| QRC-mean@100, norm | 0.485 | 0.401 | 0.433 [0.322, 0.552] | **−0.298 [−0.481, −0.094]** |
| QRC-mean@100, Platt | 0.613 | 0.484 | 0.534 [0.410, 0.652] | **−0.197 [−0.370, −0.016]** |
| QRC-mean@100, isotonic | 0.604 | 0.497 | 0.538 [0.412, 0.659] | **−0.193 [−0.364, −0.011]** |
| QRC-mean@100, norm+Platt | 0.487 | 0.398 | 0.432 [0.322, 0.551] | **−0.299 [−0.481, −0.096]** |
| QRC-mean@100, norm+isotonic | 0.511 | 0.373 | 0.427 [0.316, 0.545] | **−0.305 [−0.487, −0.102]** |
| std of P(Rel)@100, norm | 0.652 | 0.653 | 0.652 [0.531, 0.761] | −0.079 [−0.159, 0.003] |
| std of P(Rel)@100, Platt | 0.747 | 0.754 | 0.751 [0.645, 0.842] | +0.020 [−0.064, 0.105] |
| std of P(Rel)@100, isotonic | 0.669 | 0.705 | 0.691 [0.584, 0.791] | −0.040 [−0.138, 0.062] |
| std of P(Rel)@100, norm+Platt | 0.669 | 0.652 | 0.658 [0.539, 0.766] | −0.073 [−0.159, 0.015] |
| std of P(Rel)@100, norm+isotonic | 0.623 | 0.628 | 0.626 [0.511, 0.741] | **−0.105 [−0.201, −0.007]** |
A2 contrasts:
  E-a depth (calibrated level): qrcmean100_norm - qrcmean_norm = -0.128 [−0.245, −0.007] excl0=True
  E-a depth (calibrated level): qrcmean100_platt - qrcmean_platt = -0.074 [−0.125, −0.025] excl0=True
  E-a depth (calibrated level): qrcmean100_iso - qrcmean_iso = -0.061 [−0.117, −0.012] excl0=True
  E-a depth (calibrated level): qrcmean100_nplatt - qrcmean_nplatt = -0.129 [−0.246, −0.010] excl0=True
  E-a depth (calibrated level): qrcmean100_niso - qrcmean_niso = -0.141 [−0.267, −0.014] excl0=True
  E-b QRC-mean@100 vs CV@100: qrcmean100_norm - cv100 = -0.298 [−0.481, −0.094] excl0=True
  E-b QRC-mean@100 vs CV@100: qrcmean100_platt - cv100 = -0.197 [−0.370, −0.016] excl0=True
  E-b QRC-mean@100 vs CV@100: qrcmean100_iso - cv100 = -0.193 [−0.364, −0.011] excl0=True
  E-b QRC-mean@100 vs CV@100: qrcmean100_nplatt - cv100 = -0.299 [−0.481, −0.096] excl0=True
  E-b QRC-mean@100 vs CV@100: qrcmean100_niso - cv100 = -0.305 [−0.487, −0.102] excl0=True
  E-c std P@100 vs CV@100: pstd100_norm - cv100 = -0.079 [−0.159, 0.003] excl0=False
  E-c std P@100 vs CV@100: pstd100_platt - cv100 = +0.020 [−0.064, 0.105] excl0=False
  E-c std P@100 vs CV@100: pstd100_iso - cv100 = -0.040 [−0.138, 0.062] excl0=False
  E-c std P@100 vs CV@100: pstd100_nplatt - cv100 = -0.073 [−0.159, 0.015] excl0=False
  E-c std P@100 vs CV@100: pstd100_niso - cv100 = -0.105 [−0.201, −0.007] excl0=True
  E-d dispersion vs level @100 (calibrated): pstd100_platt - qrcmean100_platt = +0.217 [0.087, 0.339] excl0=True
  E-d dispersion vs level @100 (calibrated): pstd100_iso - qrcmean100_iso = +0.153 [−0.009, 0.307] excl0=False
  E-e calibration effect @100: qrcmean100_platt - mean100 = +0.001 [−0.004, 0.007] excl0=False
  E-e calibration effect @100: qrcmean100_nplatt - qrcmean100_platt = -0.102 [−0.226, 0.014] excl0=False
  E-e calibration effect @100: qrcmean100_niso - qrcmean100_iso = -0.112 [−0.242, 0.017] excl0=False
  E-f calibration on vs off (same input) @10: qrcmean_nplatt - qrcmean_norm = 0.000 [−0.006, 0.007] excl0=False
  E-f calibration on vs off (same input) @10: qrcmean_niso - qrcmean_norm = +0.006 [−0.044, 0.059] excl0=False
  E-f calibration on vs off (same input) @100: qrcmean100_nplatt - qrcmean100_norm = -0.001 [−0.011, 0.008] excl0=False
  E-f calibration on vs off (same input) @100: qrcmean100_niso - qrcmean100_norm = -0.007 [−0.038, 0.022] excl0=False

TABLE A3
| Signal | < 0.35 | < 0.40 | < 0.45 | < 0.50 | < 0.55 | < 0.60 | < 0.65 |
|---|---|---|---|---|---|---|---|
| Positive queries (DL19 / DL20) | 10 / 16 | 13 / 24 | 16 / 26 | 21 / 29 | 26 / 34 | 31 / 37 | 32 / 38 |
| CV@100 | 0.715 | 0.709 | 0.698 | 0.739 | 0.763 | 0.797 | 0.782 |
| Raw top-1 | 0.622 | 0.570 | 0.590 | 0.664 | 0.614 | **0.604** | 0.603 |
| Raw mean@10 | 0.574 | 0.559 | 0.582 | 0.652 | 0.602 | **0.586** | 0.579 |
| QRC-mean, norm | **0.418** | 0.548 | 0.547 | **0.565** | **0.574** | **0.592** | **0.584** |
| QRC-mean, Platt | 0.571 | 0.557 | 0.580 | 0.650 | 0.600 | **0.585** | 0.578 |
| QRC-mean, isotonic | 0.570 | 0.554 | 0.570 | 0.641 | **0.582** | **0.566** | **0.562** |
| QRC-mean, norm+Platt | **0.416** | 0.546 | 0.547 | **0.566** | **0.575** | **0.591** | **0.583** |
| QRC-mean, norm+isotonic | **0.410** | 0.557 | 0.566 | **0.576** | **0.593** | **0.584** | **0.573** |
| QRC-max, Platt | 0.622 | 0.570 | 0.590 | 0.664 | 0.614 | **0.604** | 0.603 |
| QRC-max, isotonic | 0.626 | 0.578 | 0.600 | 0.667 | 0.620 | **0.614** | 0.614 |
PRIMARY DETAIL:
  t=0.35 CV@100: 0.715 [0.591, 0.821] d=0.000 [0.000, 0.000] excl=False
  t=0.35 Raw top-1: 0.622 [0.479, 0.755] d=-0.094 [−0.267, 0.071] excl=False
  t=0.35 Raw mean@10: 0.574 [0.433, 0.706] d=-0.142 [−0.314, 0.026] excl=False
  t=0.35 QRC-mean norm: 0.418 [0.294, 0.555] d=-0.297 [−0.446, −0.140] excl=True
  t=0.35 QRC-mean Platt: 0.571 [0.432, 0.703] d=-0.144 [−0.316, 0.023] excl=False
  t=0.35 QRC-mean isotonic: 0.570 [0.437, 0.700] d=-0.145 [−0.320, 0.024] excl=False
  t=0.35 QRC-mean norm+Platt: 0.416 [0.291, 0.551] d=-0.300 [−0.448, −0.143] excl=True
  t=0.35 QRC-mean norm+isotonic: 0.410 [0.289, 0.546] d=-0.305 [−0.450, −0.153] excl=True
  t=0.35 QRC-max Platt: 0.622 [0.479, 0.755] d=-0.094 [−0.267, 0.071] excl=False
  t=0.35 QRC-max isotonic: 0.626 [0.485, 0.754] d=-0.089 [−0.259, 0.076] excl=False
  t=0.5 CV@100: 0.739 [0.635, 0.832] d=0.000 [0.000, 0.000] excl=False
  t=0.5 Raw top-1: 0.664 [0.546, 0.772] d=-0.075 [−0.224, 0.076] excl=False
  t=0.5 Raw mean@10: 0.652 [0.538, 0.759] d=-0.087 [−0.238, 0.071] excl=False
  t=0.5 QRC-mean norm: 0.565 [0.452, 0.684] d=-0.174 [−0.315, −0.018] excl=True
  t=0.5 QRC-mean Platt: 0.650 [0.537, 0.757] d=-0.088 [−0.240, 0.069] excl=False
  t=0.5 QRC-mean isotonic: 0.641 [0.525, 0.755] d=-0.098 [−0.252, 0.062] excl=False
  t=0.5 QRC-mean norm+Platt: 0.566 [0.453, 0.686] d=-0.173 [−0.314, −0.020] excl=True
  t=0.5 QRC-mean norm+isotonic: 0.576 [0.457, 0.694] d=-0.163 [−0.301, −0.013] excl=True
  t=0.5 QRC-max Platt: 0.664 [0.546, 0.772] d=-0.075 [−0.224, 0.076] excl=False
  t=0.5 QRC-max isotonic: 0.667 [0.551, 0.769] d=-0.072 [−0.219, 0.078] excl=False
  t=0.35 qrcmean_platt - mean10: -0.002 [−0.009, 0.000] excl=False
  t=0.35 qrcmean_nplatt - qrcmean_platt: -0.156 [−0.315, 0.019] excl=False
  t=0.35 qrcmean_niso - qrcmean_iso: -0.160 [−0.329, 0.023] excl=False
  t=0.35 qrcmean_nplatt - qrcmean_norm: -0.002 [−0.009, 0.000] excl=False
  t=0.4 qrcmean_platt - mean10: -0.002 [−0.008, 0.000] excl=False
  t=0.4 qrcmean_nplatt - qrcmean_platt: -0.011 [−0.157, 0.150] excl=False
  t=0.4 qrcmean_niso - qrcmean_iso: +0.003 [−0.161, 0.177] excl=False
  t=0.4 qrcmean_nplatt - qrcmean_norm: -0.002 [−0.008, 0.000] excl=False
  t=0.45 qrcmean_platt - mean10: -0.002 [−0.008, 0.000] excl=False
  t=0.45 qrcmean_nplatt - qrcmean_platt: -0.034 [−0.177, 0.120] excl=False
  t=0.45 qrcmean_niso - qrcmean_iso: -0.004 [−0.160, 0.160] excl=False
  t=0.45 qrcmean_nplatt - qrcmean_norm: 0.000 [−0.006, 0.007] excl=False
  t=0.5 qrcmean_platt - mean10: -0.002 [−0.008, 0.000] excl=False
  t=0.5 qrcmean_nplatt - qrcmean_platt: -0.084 [−0.225, 0.066] excl=False
  t=0.5 qrcmean_niso - qrcmean_iso: -0.064 [−0.226, 0.102] excl=False
  t=0.5 qrcmean_nplatt - qrcmean_norm: +0.001 [−0.004, 0.008] excl=False
  t=0.55 qrcmean_platt - mean10: -0.002 [−0.008, 0.000] excl=False
  t=0.55 qrcmean_nplatt - qrcmean_platt: -0.025 [−0.180, 0.133] excl=False
  t=0.55 qrcmean_niso - qrcmean_iso: +0.010 [−0.155, 0.180] excl=False
  t=0.55 qrcmean_nplatt - qrcmean_norm: +0.001 [−0.004, 0.008] excl=False
  t=0.6 qrcmean_platt - mean10: -0.001 [−0.006, 0.000] excl=False
  t=0.6 qrcmean_nplatt - qrcmean_platt: +0.006 [−0.157, 0.179] excl=False
  t=0.6 qrcmean_niso - qrcmean_iso: +0.018 [−0.167, 0.209] excl=False
  t=0.6 qrcmean_nplatt - qrcmean_norm: -0.001 [−0.007, 0.004] excl=False
  t=0.65 qrcmean_platt - mean10: -0.001 [−0.007, 0.000] excl=False
  t=0.65 qrcmean_nplatt - qrcmean_platt: +0.005 [−0.165, 0.190] excl=False
  t=0.65 qrcmean_niso - qrcmean_iso: +0.011 [−0.185, 0.214] excl=False
  t=0.65 qrcmean_nplatt - qrcmean_norm: -0.001 [−0.007, 0.004] excl=False