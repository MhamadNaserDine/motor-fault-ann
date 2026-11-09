# Report / slides notes

## Results summary (paste-ready)

**Data.** 1500 motors simulated with the 2.2 kW dq model: 300 per class (healthy, broken rotor bars,
inter-turn short, bearing defect, supply unbalance), 3 severities, 30–110 % load, 0.5–2 % sensor noise,
±5 % parameter spread. 19 MCSA features. Split 60 / 20 / 20 into training / test / validation.

**Diagnosis (5 classes).** The MLP trained with back-propagation and the MLP trained with
Levenberg-Marquardt are the best networks: **95.3 % validation accuracy (F1 ≈ 95.4 %)**, against 93.7 %
for the Perceptron, 91.3 % for ADALINE and 75.7 % for the RBF network. A scikit-learn MLP reaches 94.3 %,
which confirms that the from-scratch implementation is correct.
Broken bars are always recognised (60/60). Most errors are incipient faults at light load that the
network confuses with *healthy*, and healthy motors on a slightly unbalanced grid classified as *supply unbalance*.

**Detection (healthy / faulty).** All networks give 88–93 %. The MLP with momentum and the RBF network
reach 92.5 %, with the MLP having the best precision (96 % of the alarms are real faults).
Detection is harder than it looks because the smallest faults (1.5 % shorted turns, 2 % torque ripple)
are of the same order as the normal imperfections of a healthy motor.

**Training algorithms.**
* Momentum and adaptive η reach the early-stopping point A **4× sooner** (epoch 37–52 vs 149 for classic BP).
* Levenberg-Marquardt reaches point A at **epoch 7** and is the fastest MLP (2.9 s vs 8 s). It also shows
  overfitting most clearly: after point A the training error goes to ~10⁻¹⁵ while the test error goes up.
* Learning-rate trade-off: η = 0.001 is too slow, η = 0.05 is good, η = 5 oscillates
  (`figures/diagnosis_eta_tradeoff.png`).
* Single layer vs MLP: with only 2 features (I₂/I₁, BRB sideband) the healthy class sits in a corner and the
  faulty region is an OR of two half-planes. A single line cannot draw that shape: Perceptron 86.7 %, ADALINE 85.0 %,
  MLP 92.5 % (`figures/nonlinear_separability.png`). With all 19 features the classes are already almost
  linearly separable, which is why the Perceptron still does well there. Good features matter as much as the network.
* RBF (fixed centres, σ = d/√(2M)) trains in 10 ms, but it is the least accurate. Its result depends a lot
  on M and on which centres are drawn at random (`figures/diagnosis_rbf_vs_M.png`): in 19 dimensions, a few
  random centres and one common σ cannot cover the input space.

**Key figures:** `motor_start_up.png`, `spectrum_*.png`, `park_vector.png`, `feature_boxplots.png`,
`diagnosis_learning_curves.png`, `diagnosis_mlp_bp_point_A.png`, `diagnosis_confusion_matrices.png`,
`diagnosis_comparison.png`, `nonlinear_separability.png`, `diagnosis_eta_tradeoff.png`,
`diagnosis_momentum_effect.png`, `diagnosis_adaptive_eta.png`, `diagnosis_lm_mu.png`, `diagnosis_rbf_vs_M.png`.

## Reliability experiments (experiments.py)
* **Repeated runs (10×):** MLP-BP 94.0 ± 1.5 % is the best on average; Perceptron 93.3 ± 1.7 %, MLP-LM 92.3 ± 2.1 %,
  ADALINE 89.6 ± 1.7 %, RBF 77.5 ± 3.9 %. Lesson: one run can exaggerate differences – always report mean ± std.
* **Early faults:** on 500 fresh motors MLP-BP finds severity-1 faults 89 % of the time, severity 2–3 100 %.
* **Noise:** ≈ 95 % up to 1 % noise, 90 % at 2 %, but only 31–40 % (MLPs) at 5 % noise, which is outside the training
  range – ADALINE degrades more gracefully (65 %). Train with the noise of the real sensors.
* **Network size:** under-fitting with 1–2 hidden neurons, plateau from H ≈ 4; bigger networks only over-fit more
  (handled by early stopping).

## Likely professor questions

**1. Why do you need an MLP if the Perceptron already gets ~93 %?**
The 19 features were engineered with motor physics, so the classes are almost linearly separable.
When the inputs are fewer or less informative, the faulty region is no longer one half-plane (an OR of
conditions, like XOR). Then the single layer fails: 86.7 % vs 92.5 % for the MLP in the 2-feature experiment.
On the full problem the MLP is still the best on average (94.0 ± 1.5 % vs 93.3 ± 1.7 % over 10 runs), but the gap is small – honest answer: with good features a linear model is already strong.

**2. What is the difference between your test set and your validation set?**
Following the course: the test set is used *during* learning. Learning is paused periodically, the test
error is measured, and the weights at its minimum (point A) are kept. So the test set has influenced the
model. The validation set is never used before the end, so its score is unbiased. All reported numbers
are validation numbers.

**3. Why does Levenberg-Marquardt converge in so few iterations?**
Δw = −(JᵀJ + μI)⁻¹Jᵀe uses second-order (Gauss-Newton) information. When μ is large it behaves like
gradient descent with η = 1/μ (safe far from the minimum). When μ is small it becomes Gauss-Newton (fast
near the minimum). μ is decreased after every successful step. The cost: building and inverting a
P×P matrix (P = 380 weights here) at each iteration, so it suits small networks.

**4. Why is the RBF network worse than the MLP here?**
Gaussian neurons give a *local* response. The fixed-centres method takes centres at random and uses one σ
fixed by the largest distance d. In a 19-dimensional space, a few dozen random centres cannot cover all
the classes, and d is inflated by outliers. The MLP uses *global* sigmoid hyperplanes, which suit this
data better. Improvements: supervised selection of centres (gradient on t_j, σ_j, w_j), k-means centres,
or fewer, better-chosen inputs.

**5. How did you simulate a broken bar, and why does it give (1 ± 2s)f_s sidebands?**
A broken bar makes the rotor electrically asymmetric. In the model the rotor resistance on the rotor q-axis
is increased by k = 3n/(N−3n). An asymmetric rotor creates a backward-rotating field at −s f_s relative
to the rotor, which appears at (1−2s)f_s in the stator. The resulting torque/speed ripple at 2s f_s
gives the (1+2s)f_s component. The FFT of the simulated current shows exactly these lines.

**6. Will it work on a real motor?**
Not directly. The model is idealised: no saturation, no slot harmonics, a simplified short-circuit and
bearing model, and noise/spread that is only estimated. The method transfers (same features, same networks),
but the networks must be re-trained or fine-tuned on measured data, ideally from a test bench with seeded faults.
The features were chosen because they are standard MCSA indicators that also exist on real data.

**7. (bonus) Why normalise the features, and with which data?**
The features have very different scales (A, %, dB). Without normalisation, the large inputs saturate the
tanh neurons (φ' ≈ 0, so learning stops) and dominate the RBF distances. The mean/std are computed on the
training set only. Otherwise information from the test/validation sets would leak into training.
