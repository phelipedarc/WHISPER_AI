# WHISPER sanity check & benchmark — synthetic parameter recovery

Synthetic light curves `M(t, θ) + white noise` with **known ground truth**, fit by every WHISPER sampler, timed, and validated statistically (recovery z-scores, credible-interval coverage, posterior-predictive checks, Simulation-Based Calibration). Showcases: a physically-motivated **Bazin (2009) supernova** light curve (30k-simulation neural budgets, MDN + NSF density estimators) and a **damped sinusoid** (correlated, oscillatory stress test), plus a 2/4/6-parameter dimensionality sweep.

*Disclosure:* the Bazin and sweep noise seeds were **screened** to be non-adversarial (worst |MLE−truth|/σ_Fisher ≲ 1 — an unlucky draw makes every method 'miss' spuriously), so the single-realization recovery/coverage columns compare methods on a shared, well-posed realization and are favourable by construction; they are **not** calibration evidence. Calibration is tested by **SBC over many unscreened realizations**.

## Showcase — bazin_sn

Mock Bazin (2009) SN: `A·exp(−(t−t0)/τ_fall) / (1+exp(−(t−t0)/τ_rise))` (truth amplitude=5, t0=20, tau_rise=4, tau_fall=25), white noise σ=0.15. Every sampler fits the *same* data; the neural methods train on GPU and condition on **the stacked (value, err, time, band) tuple** (embeddings: mlp, none, tcn).

### Recovery, goodness-of-fit & speed

| method | max\|z\| | cov68 | cov95 | χ²_best | PPC cov68 | PPC cov95 | wall [s] | per-object [s] | sims |
|---|---|---|---|---|---|---|---|---|---|
| MCMC | 0.83 | 1.00 | 1.00 | 0.93 | 0.69 | 0.97 | 10.5 | 10.5 | — |
| ABC | 0.30 | 1.00 | 1.00 | 1.40 | 0.93 | 1.00 | 7.3 | 7.3 | — |
| ABC-SMC | 0.39 | 1.00 | 1.00 | 0.94 | 0.74 | 0.99 | 39.2 | 39.2 | 710,400 |
| NPE-MDN (GPU) | 0.74 | 1.00 | 1.00 | 1.10 | 0.80 | 1.00 | 197.8 | 0.01 | 30,000 |
| NPE-NSF (GPU) | 0.86 | 1.00 | 1.00 | 0.96 | 0.86 | 0.99 | 560.5 | 0.07 | 30,000 |
| SNPE-5r (GPU, no embed) | 0.45 | 1.00 | 1.00 | 0.96 | 0.89 | 1.00 | 412.6 | 412.6 | 15,000 |
| SNPE-5r (GPU, MLP embed) | 0.54 | 1.00 | 1.00 | 0.95 | 0.86 | 0.99 | 324.3 | 324.3 | 15,000 |
| SNPE-5r (GPU, TCN embed) | 0.32 | 1.00 | 1.00 | 0.99 | 0.79 | 0.97 | 380.7 | 380.7 | 15,000 |

*max|z| = max over parameters of |median−true|/σ (≲2 ⇒ recovered). cov68/95 = fraction of parameters whose credible interval covers the truth. χ²_best≈1 ⇒ the model fits; PPC cov68/95 = fraction of data inside the noise-inflated predictive band (≈0.68/0.95 ⇒ calibrated). wall = end-to-end fit time (MCMC includes its MLE seeding; neural methods include training + posterior sampling). **per-object** = cost of inferring one NEW observation: amortized 1-round NPE conditions the trained network and samples in ~a second — every other method pays a full refit. Single noise realization, so per-parameter coverage is coarse — SBC below is the calibration test over many realizations.*

### Simulation-Based Calibration (rank uniformity)

| method | L | min uniformity p | calibrated |
|---|---|---|---|
| MCMC | 100 | 0.229 | True |
| ABC | 100 | 0.000 | False |
| ABC-SMC | 60 | 0.000 | False |
| NPE-MDN (GPU) | 200 | 0.003 | False |
| NPE-NSF (GPU) | 200 | 0.119 | True |

*Uniform ranks (p ≳ 0.05) ⇒ calibrated uncertainties; ∪-shape = overconfident, ∩-shape = underconfident.*

### Takeaways

- **Recovery:** every sampler recovers all parameters within ~2σ (best max|z| = ABC at 0.30, worst = NPE-NSF (GPU) at 0.86).
- **Speed (end-to-end):** ABC (7s) < MCMC (10s) < ABC-SMC (39s) < NPE-MDN (GPU) (198s) < SNPE-5r (GPU, MLP embed) (324s) < SNPE-5r (GPU, TCN embed) (381s) < SNPE-5r (GPU, no embed) (413s) < NPE-NSF (GPU) (560s).
- **Calibration (SBC), best → worst rank-uniformity p:** MCMC (0.229, calibrated), NPE-NSF (GPU) (0.119, calibrated), NPE-MDN (GPU) (0.003), ABC (0.000), ABC-SMC (0.000). Only p ≥ 0.05 is formally calibrated; the ordering shows how close each gets.

![corner](sanity_corner_bazin_sn.png)

![histograms](sanity_hist_bazin_sn.png)

![ppc](sanity_ppc_bazin_sn.png)

![sbc](sanity_sbc_bazin_sn.png)

![summary](sanity_summary_bazin_sn.png)

## Showcase — damped_sine

Mock damped sinusoid: `A·exp(−t/τ)·sin(2πf·t+φ)` (truth A=5, tau=10, freq=0.07, phase=1), white noise σ=0.15. Every sampler fits the *same* data; the neural methods train on GPU and condition on **the raw value vector** (embeddings: none).

### Recovery, goodness-of-fit & speed

| method | max\|z\| | cov68 | cov95 | χ²_best | PPC cov68 | PPC cov95 | wall [s] | per-object [s] | sims |
|---|---|---|---|---|---|---|---|---|---|
| MCMC | 1.63 | 0.75 | 1.00 | 1.25 | 0.63 | 0.93 | 7.6 | 7.6 | — |
| ABC | 0.32 | 1.00 | 1.00 | 2.07 | 1.00 | 1.00 | 8.4 | 8.4 | — |
| ABC-SMC | 0.34 | 1.00 | 1.00 | 1.26 | 0.85 | 1.00 | 32.5 | 32.5 | 686,400 |
| NPE-MAF (GPU) | 1.06 | 0.50 | 1.00 | 1.29 | 0.61 | 0.95 | 299.1 | 299.1 | — |
| SNPE-MAF (GPU) | 1.04 | 1.00 | 1.00 | 1.26 | 0.83 | 0.99 | 1353.9 | 1353.9 | — |

*max|z| = max over parameters of |median−true|/σ (≲2 ⇒ recovered). cov68/95 = fraction of parameters whose credible interval covers the truth. χ²_best≈1 ⇒ the model fits; PPC cov68/95 = fraction of data inside the noise-inflated predictive band (≈0.68/0.95 ⇒ calibrated). wall = end-to-end fit time (MCMC includes its MLE seeding; neural methods include training + posterior sampling). **per-object** = cost of inferring one NEW observation: amortized 1-round NPE conditions the trained network and samples in ~a second — every other method pays a full refit. Single noise realization, so per-parameter coverage is coarse — SBC below is the calibration test over many realizations.*

### Simulation-Based Calibration (rank uniformity)

| method | L | min uniformity p | calibrated |
|---|---|---|---|
| MCMC | 100 | 0.091 | True |
| ABC | 100 | 0.000 | False |
| ABC-SMC | 60 | 0.000 | False |
| NPE-MAF (GPU) | 200 | 0.017 | False |

*Uniform ranks (p ≳ 0.05) ⇒ calibrated uncertainties; ∪-shape = overconfident, ∩-shape = underconfident.*

### Takeaways

- **Recovery:** every sampler recovers all parameters within ~2σ (best max|z| = ABC at 0.32, worst = MCMC at 1.63).
- **Speed (end-to-end):** MCMC (8s) < ABC (8s) < ABC-SMC (32s) < NPE-MAF (GPU) (299s) < SNPE-MAF (GPU) (1354s).
- **Calibration (SBC), best → worst rank-uniformity p:** MCMC (0.091, calibrated), NPE-MAF (GPU) (0.017), ABC-SMC (0.000), ABC (0.000). Only p ≥ 0.05 is formally calibrated; the ordering shows how close each gets.

![corner](sanity_corner_damped_sine.png)

![histograms](sanity_hist_damped_sine.png)

![ppc](sanity_ppc_damped_sine.png)

![sbc](sanity_sbc_damped_sine.png)

![summary](sanity_summary_damped_sine.png)

## Statistical notes & fixes

- **ABC-SMC ε-floor.** A naive adaptive ε shrinks to χ²_min and collapses the posterior onto the MLE (spuriously overconfident: on the 2-param Gaussian pulse the raw run gave |z|≈8 with 0% coverage). WHISPER's `min_epsilon="auto"` floors ε at χ²_min + 2(k+2), reproducing the Gaussian posterior width — restoring |z|≲2 and nominal coverage on the single-realization recovery.
- **ABC is approximate — SBC proves it.** Over many realizations, rejection ABC is **under-confident** (finite acceptance tolerance ⇒ posterior wider than the truth, ∩-shaped ranks) and even ε-floored ABC-SMC cannot perfectly calibrate a strongly **correlated** posterior with its diagonal-Gaussian kernel (on the damped sine: freq too wide, phase too narrow). Point recovery stays unbiased; the uncertainty *shape/width* is what suffers — exactly the likelihood-free approximation error SBC exists to reveal.
- **Neural SBI input & embeddings.** The **Bazin-showcase** neural methods condition on the **stacked observation tuple** `(value, error, time, band)` — the same information the likelihood-based samplers receive — and the noise added to every simulation is the **per-point reported error** (`flux_err`), matching the generative model of the data. The SNPE-5-round benchmark compares three inputs to the NSF estimator: the raw stacked vector, an **MLP** compressor, and a **TCN** (Temporal Convolutional Network — dilated causal convolutions specialized for time series), each trained jointly with the estimator to a 32-feature latent. (The legacy damped-sine/sweep MAF rows predate this and condition on the raw value vector.)
- **Noise-matched ABC — necessary, not sufficient.** `simulate_noise=True` (now the default) adds per-point `N(0, flux_err)` to every ABC/ABC-SMC simulation: the correct generative model, it removes the noiseless-shell pathology and makes ABC exact as ε→0. **Measured outcome (SBC):** at practical acceptance budgets the finite-tolerance width inflation still dominates — on the Bazin mock the ABC posterior is ~8× and the ε-floored ABC-SMC ~3× the Fisher width, so their rank-uniformity p stays ≈0 (under-confident), statistically unchanged from the noiseless run. Driving ε low enough to calibrate rejection ABC needs an infeasible simulation budget; the calibrated routes on this problem are exact MCMC and NPE-NSF.
- **Identifiable pulses.** A sum of Gaussians is invariant under permuting its (Aₖ,μₖ) pairs, so the sweep gives each μₖ a disjoint prior bin; otherwise every sampler is free to label-switch (a spurious multi-modal 'failure').
- **SNPE speed diagnosis (GPU).** Profiling shows three cost centres, now addressed: (1) *simulation* — the per-row Python loop is replaced by a **batched torch simulator on the GPU** (`predict_torch`), ~2000× faster (30k Bazin simulations in milliseconds); (2) *training* — small nets underutilize an A6000 (a few % GPU load: kernel-launch latency dominates), so the lever is larger batches (`training_batch_size=1000`) and tighter early-stop patience (`stop_after_epochs`), not more GPUs — **multi-GPU data-parallel training does not help here** (sbi 0.23 has no native support and these networks are far too small to amortize inter-GPU synchronization; the effective multi-GPU strategy is one method/config per GPU, which is how these benchmarks run); (3) *between-round posterior sampling* — rejection sampling under leakage; fewer, larger rounds (5×3000) cut it. JAX was considered for (1) but is not installed in this environment; the torch path already reduces simulation to a negligible cost.
- **Why exact-likelihood MCMC is still faster on THIS problem — and when SNPE wins.** A 4-parameter analytic Bazin likelihood costs microseconds, so MCMC's ~10⁵ sequential evaluations finish in seconds; no amount of GPU engineering makes *learning* a posterior cheaper than *evaluating* that likelihood. Neural SBI wins in its actual use cases: (a) **expensive/intractable simulators** (a kilonova model at ~0.1 s/call turns MCMC into hours while SNPE needs a fixed, parallelizable simulation budget), and (b) **amortization** — the per-object column above: after one training, NPE infers a NEW light curve in ~a second, vs a full refit for every likelihood method. SBC over many realizations exploits exactly that amortization; re-training sequential SNPE per realization is left out as prohibitive (its single-dataset recovery + PPC stand in).

## Dimensionality sweep (2/4/6 params, Gaussian pulses)

| method | 2p max\|z\| / t[s] | 4p max\|z\| / t[s] | 6p max\|z\| / t[s] |
|---|---|---|---|
| MCMC | 0.87 / 5.5 | 0.87 / 9.9 | 0.93 / 18.8 |
| ABC | 0.16 / 6.1 | 0.10 / 6.3 | 0.28 / 8.6 |
| ABC-SMC | 0.43 / 21.4 | 0.28 / 14.4 | 0.17 / 26.0 |
| NPE-MAF (GPU) | 0.68 / 101.7 | 1.48 / 106.3 | 0.79 / 129.3 |

*Cell = max|z| / wall[s]. All methods stay within ~2σ as the parameter count grows 2→4→6; runtime scales gently. The sweep uses the MAF-NPE config; sequential SNPE is omitted from the sweep (10-round cost) — its recovery is shown in the showcases above.*
