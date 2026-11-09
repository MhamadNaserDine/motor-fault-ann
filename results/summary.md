# Results summary

Dataset: 1500 simulated motors (300 per class), loads 30–110 % of rated,
measurement noise ~0.5–1.5 % of RMS, ±5 % parameter spread. 19 MCSA features.
Split: training 60 % / test 20 % (early stopping) / validation 20 % (final, unbiased score).

## Fault detection (healthy vs faulty) – validation set
| Model | Acc % | Prec % | Rec % | F1 % | Time (s) | Epochs | Point A |
|---|---|---|---|---|---|---|---|
| Perceptron | 91.7 | 93.1 | 90.0 | 91.5 | 1.53 | 300 | 196 |
| ADALINE (LMS) | 88.3 | 92.6 | 83.3 | 87.7 | 1.05 | 300 | 131 |
| MLP-BP | 91.7 | 94.6 | 88.3 | 91.4 | 5.75 | 300 | 164 |
| MLP-BP + momentum | 92.5 | 96.4 | 88.3 | 92.2 | 5.83 | 300 | 34 |
| MLP-BP + momentum + adaptive η | 92.5 | 96.4 | 88.3 | 92.2 | 5.79 | 300 | 26 |
| MLP-Levenberg-Marquardt | 91.7 | 93.1 | 90.0 | 91.5 | 46.59 | 100 | 7 |
| RBF (fixed centres + pseudo-inverse) | 92.5 | 91.8 | 93.3 | 92.6 | 0.02 | 1 | 1 |
| scikit-learn MLP (check) | 95.8 | 98.2 | 93.3 | 95.7 | 1.17 | 1334 | — |

## Fault diagnosis (5 classes) – validation set
| Model | Acc % | Prec % | Rec % | F1 % | Time (s) | Epochs | Point A |
|---|---|---|---|---|---|---|---|
| Perceptron | 93.7 | 93.8 | 93.7 | 93.7 | 6.19 | 300 | 75 |
| ADALINE (LMS) | 91.3 | 91.6 | 91.3 | 91.4 | 4.90 | 300 | 69 |
| MLP-BP | 95.3 | 95.5 | 95.3 | 95.3 | 21.06 | 300 | 149 |
| MLP-BP + momentum | 94.3 | 94.6 | 94.3 | 94.4 | 21.33 | 300 | 37 |
| MLP-BP + momentum + adaptive η | 94.3 | 94.5 | 94.3 | 94.4 | 22.03 | 300 | 52 |
| MLP-Levenberg-Marquardt | 95.3 | 95.5 | 95.3 | 95.4 | 60.81 | 100 | 7 |
| RBF (fixed centres + pseudo-inverse) | 75.7 | 78.4 | 75.7 | 75.7 | 0.03 | 1 | 1 |
| scikit-learn MLP (check) | 94.3 | 94.5 | 94.3 | 94.3 | 2.82 | 1119 | — |

## Learning-rate experiment (classic BP, 150 epochs)
- eta_0.001: final training cost 0.0803, validation accuracy 94.3 %
- eta_0.05: final training cost 0.0248, validation accuracy 94.7 %
- eta_5.0: final training cost 0.1187, validation accuracy 88.3 %

## Non-linear separability (2 features: I2/I1 and BRB sideband; healthy vs BRB-or-ITSC), validation accuracy
- Perceptron: 86.7 %
- ADALINE (LMS): 85.0 %
- MLP-BP + momentum (10 hidden): 92.5 %

## RBF: test error vs number of centres M
- M = 5: training error 38.4 %, test error 33.7 %
- M = 10: training error 26.0 %, test error 26.7 %
- M = 20: training error 16.2 %, test error 18.7 %
- M = 40: training error 24.8 %, test error 25.3 %
- M = 80: training error 17.3 %, test error 21.7 %
- M = 120: training error 17.9 %, test error 21.7 %
- M = 200: training error 17.2 %, test error 25.0 %
- M = 300: training error 26.0 %, test error 43.3 %
