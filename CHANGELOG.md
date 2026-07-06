# Changelog

All notable changes to Whisper (`whisper_labia`). The project is in early development; nothing is
released to PyPI yet — install from GitHub (`pip install git+https://github.com/phelipedarc/WHISPER_AI.git`).

## [Unreleased] — 0.0.1.dev0

### Full UV–optical–NIR AT2017GFO application + SNPE robustness against pathological real-data conditioning
- **Full-UVOIR data.** `scripts/fetch_at2017gfo_full.py` pulls the complete AT2017GFO photometry from
  the Open Astronomy Catalog (607 detections, 18 bands, 0.5–25 d, Vega→AB converted where needed) into
  `tests/data/at2017gfo_full.csv`. `two_component_kilonova._redback_band` now maps any band redback's
  filter table knows (`H`/`J`/`Ks`/`Y` → 2MASS, `U`/`B`/`V`/`R`/`I` → Bessell, `uvot::*` → Swift UVOT),
  not just optical grizy, so the model can be fit to the full UV→optical→NIR range.
  `docs/PLAN_at2017gfo_fullband.md` records the acquisition plan and cost budget.
- **Physical ejecta-velocity prior.** `at2017gfo_villar.py`'s real-data prior tightened
  `vej_1`/`vej_2` from `Uniform(0.01, 0.7)` to `Uniform(0.05, 0.3)` (physical kilonova range) — the old
  bound let the MAP rail to an unphysical 0.7 c, a genuine-but-unphysical optimum caused by the
  permissive prior combined with g/r/i-only data underconstraining the red component. With the full
  UVOIR data the red component (κ_red, v_red, T_red) now moves off the prior edges toward
  Villar+2017-consistent values — the payoff of the NIR coverage g/r/i alone cannot supply.
- **Flux-vs-magnitude comparison.** `space="flux"|"magnitude"` now routes through the whole real-data
  application (`VILLAR_SPACE` env var): data, simulator noise, likelihood and the σ scatter prior
  (`LogUniform` bounds are space-appropriate — Jy for flux, mag for magnitude) are all consistent in
  the chosen space. Fixed a units bug where MCMC's ABC-seeded σ warm-start was hardcoded to a
  magnitude-scale value (0.2) regardless of space, silently placing flux-space MCMC's σ walker outside
  its own `LogUniform(1e-8, 1e-3)` Jy prior.
- **SNPE input normalisation (`standardize_x="asinh"|"zscore"|"none"`).** Flux-space light curves span
  ~6 orders of magnitude; sbi's built-in z-scoring fixes scale but not skew, which left flux-space
  neural SBI poorly conditioned. `asinh(x/scale)` (per-channel `scale` fit on round-0 simulations,
  applied identically to simulations, the observation and `result.format_x`) variance-stabilises the
  input, making flux as well-conditioned as magnitude space.
- **Band removed as a stacked-input channel.** Every SNPE simulation is drawn on the identical
  `(time, band)` grid as the observation, so band identity is already encoded positionally — a constant
  per-position band channel added no information and empirically hurt flux-space fits (removing it
  improved recovered κ_red from 2.34 to 3.37 cm²/g, closer to Villar+17's 3.65).
- **Robust post-fit max-likelihood scan.** The scan re-runs the forward model on up to `max_logl_scan`
  posterior draws to get the exact best-fit/AIC/BIC; a single pathological draw calling an expensive
  model (e.g. redback, ~0.2–0.3 s/call) could hang the entire fit for hours. Now runs in a process pool
  with a wall-clock `scan_timeout` (default 300 s), degrading to a smaller scored subset on timeout
  rather than hanging (root-caused a 10.8 h flux-space SNPE hang; fixed to ~200 s).
  Confirmed **not** a GPU or simulation-parallelism issue (both were already healthy).
- **Robust final posterior sampling against real-data pathologies.** Conditioning a trained density
  estimator on the *real* observation (rather than the simulated placeholders seen in training) can
  expose two failure modes sbi's default rejection sampling does not handle gracefully: a near-zero
  acceptance rate (technically not an error, just impractically slow — sbi's own diagnosis loop is
  itself just as slow, since it estimates the correction factor by accept-reject sampling until enough
  are accepted) and a numerically degenerate spline coefficient in the flow's inverse transform
  (`AssertionError` deep in `nflows`, for flexible estimators like NSF). `SNPESampler.fit` now probes
  the acceptance rate with a single bounded, hang-proof sample (drawn directly from the density
  estimator, cost independent of the acceptance rate) and catches the `AssertionError`, falling back to
  MCMC-based posterior sampling (conditions via the numerically-stable forward `log_prob` instead of the
  flow's inverse) in either case. The same acceptance factor is pre-cached on each round's posterior so
  sequential/truncated SNPE's internal proposal-restriction step (`get_density_thresholder`) reuses it
  instead of re-deriving it the slow way. `result.info["final_sample_method"]`
  (`"rejection"`/`"mcmc_fallback"`) and `["final_sample_acceptance_rate"]` record which path was taken.
  Also fixed the same exposure in `scripts/at2017gfo_villar.py`'s amortized-resample timing benchmark.
- **PPC plot readability.** Taller per-band posterior-predictive panels (shared y-axis, range set from
  data + model medians rather than σ-inflated tails) so bands separate cleanly. New supplementary
  **`villar_ppc_grid.png`**: the same check zoomed to the first 10 days, laid out as one square panel
  per method with larger labels and bolder, edge-outlined data markers, for a closer read of band-by-band
  structure where the two components pull apart fastest.
- **Rail/σ reporting fixes** in `at2017gfo_villar_plots.py`: prior-rail detection now judges proximity
  to the bound *value* (not fraction of prior range, which misreported well-constrained values under
  wide priors like κ∈[1,30]) and is keyed on the MCMC reference posterior rather than any single
  (possibly broad neural) method; the reported σ is MCMC's own value with space-aware units (mag vs Jy)
  rather than a cross-method median dragged up by broad neural posteriors.

### Villar+2017 AT2017GFO application: free-scatter likelihood, ABC comparison space, parallel MCMC
- **`GaussianLikelihoodWithScatter`** (`kind="gaussian_scatter"`/`"villar"`): Gaussian likelihood with
  a **free extra-scatter term σ added in quadrature** to the reported errors — Villar et al. 2017
  (ApJL 851 L21) Eq. 4 in its correctly normalized form, `lnL = −½Σ[(O−M)²/(σᵢ²+σ²) +
  ln(2π(σᵢ²+σ²))]`. σ is a *likelihood* parameter sampled with the rest: MCMC routes a prior
  parameter named `sigma` via `likelihood="gaussian_scatter"`; ABC/ABC-SMC/SNPE take
  `scatter_param="sigma"` and fold it into their **generative simulation noise**
  (`N(0, √(σᵢ²+σ²))` per draw). Verified on synthetic data with mis-reported errors: MCMC and the
  exact likelihood recover the injected scatter; neural SBI constrains it weakly (a noise level is a
  distributional feature of one realization); and a plain χ² rejection distance **cannot** identify
  it (monotonically penalised by extra noise — its ABC posterior collapses to zero), so the ABC
  family fits the physical parameters only, documented in the report.
- **ABC/ABC-SMC `space="auto"|"flux"|"magnitude"`**: the comparison space now routes the acceptance
  itself — data, simulations, noise and distance all live in the chosen space (previously flux-only).
  Sampled parameters are now `prior.names` (a prior may carry non-model parameters).
- **MCMC `n_jobs`**: emcee walker likelihoods in a process pool — makes an 8-D fit with the ~0.1 s
  redback kilonova likelihood run in minutes.
- **`scripts/at2017gfo_villar.py` + `_plots.py`**: the real-world application — the
  `two_component_kilonova` with **κ_blue = 0.5 fixed**, z fixed, κ_red + both temperature floors
  free + σ, fit in magnitude space by 7 methods (MCMC, ABC, ABC-SMC, NPE-MDN, NPE-NSF, SNPE-5r-NSF,
  SNPE-5r-NSF+TCN), rendered to `docs/REPORT_at2017gfo_villar.md` with annotated posterior
  histograms, an all-method corner, magnitude-space posterior-predictive light curves and a
  parameter/runtime summary.

### Calibrated likelihood-free inference + neural-SBI performance upgrade
- **Noise-matched ABC/ABC-SMC (`simulate_noise=True`, new default).** Every simulation now adds
  per-point white noise from the reported errors (`N(0, flux_err)`), matching the generative model of
  the data — this is what makes ABC exact as ε→0 and its posterior **width calibrated** (a noiseless
  simulator under a hard cut targets a likelihood shell). Reproducibility across `n_jobs` is preserved
  (noise drawn from each simulation's own RNG stream). **Scale note:** distances now include the
  simulation noise (`E[D] ≈ χ² + n_points`), so fixed `threshold`/`epsilon_schedule`/float
  `min_epsilon` values from the noiseless era must be re-derived (adaptive quantiles handle it).
  `best_params` (and AIC/BIC there) are now selected by the **exact Gaussian log-likelihood** over
  accepted draws — never by the noisy distance, whose argmin is the luckiest noise realization.
- **`whisper_labia.embeddings` (new):** `MLPEmbedding` and `TCNEmbedding` (Temporal Convolutional
  Network — dilated causal 1-D convolutions with residual blocks, avg+max-pool head), buildable via
  `fit_SNPE(..., embedding_net="mlp"|"tcn", embedding_latent=...)` and trained jointly with the
  density estimator.
- **`fit_SNPE` input & speed:** `x_format="stacked"` conditions the network on the full observation
  tuple `(value, error, time, band)` — the same information the likelihood-based samplers get;
  `predict_torch=` accepts a batched torch forward model and replaces the per-row Python simulator
  with a single on-device call (**~2000× faster simulation**; 30k Bazin simulations in milliseconds);
  `result.format_x` maps a raw vector to the network input for amortized reuse (same observing grid).
  Fixed a CUDA device mismatch when truncated SNPE (`proposal_mode="restricted"`) is combined with
  `predict_torch` (sbi's `RestrictedPrior` samples on CPU by default). Embedding weights are now
  initialized after `torch.manual_seed`, making embedding benchmarks run-to-run reproducible.
  Multi-GPU data-parallel training was evaluated and rejected: sbi 0.23 has no native support and
  these networks are far too small to amortize synchronization — the effective multi-GPU strategy is
  one method/config per GPU (how the benchmarks run).

### Inference-validation tools + synthetic-recovery sanity check
- **`whisper_labia.validation`** — reusable, sampler-agnostic checks that a fit *recovered the truth*
  with *reliable uncertainties*: `recovery_metrics` (per-parameter bias, standardized z-score, 68/95%
  credible-interval coverage vs a known truth), `posterior_predictive_check` (predictive band, best-fit
  reduced χ², noise-inflated predictive coverage, Bayesian p-value) and `sbc_rank` / `sbc_ranks`
  (Simulation-Based Calibration — Talts 2018 / Säilynoja 2022; rank uniformity with a χ²-of-uniformity
  p-value flags over-/under-confidence). Exposed as `wp.recovery_metrics` etc.
- **ABC-SMC `min_epsilon` (ε floor).** The adaptive schedule could drive ε all the way to χ²_min,
  collapsing the posterior onto the MLE — spuriously **overconfident** (on the synthetic 2-param recovery
  the raw run gave |z|≈8 with 0% coverage). `min_epsilon="auto"` floors ε at **χ²_min + 2(k+2)** (k =
  #parameters), which reproduces the Gaussian posterior width and restores |z|≲2 with nominal coverage;
  a float sets a fixed floor. Default `None` keeps the old behavior.
- **`scripts/sanity_check.py` + `scripts/sanity_check_plots.py`** — end-to-end recovery benchmark on
  synthetic data with known ground truth: fits a physically-motivated **Bazin (2009) supernova** light
  curve (headline showcase), a 4-param damped sinusoid (correlated/oscillatory stress test) and a
  2/4/6-param Gaussian-pulse sweep, timing every sampler, and renders posterior histograms, an
  all-sampler corner, posterior-predictive checks, SBC rank histograms, a recovery/speed/scaling summary
  and `REPORT.md` into `docs/figures/sanity_check/`. Showcase noise seeds are screened non-adversarial
  (worst |MLE−truth|/σ_Fisher ≲ 1) and the choice is disclosed in the report — single-realization tables
  compare methods; **SBC over many unscreened realizations is the calibration evidence**. On the damped
  sine, exact MCMC calibrates, NPE-MAF trails mildly over-confident, and the ABC family shows its
  width errors (rejection ABC under-confident; the diagonal-kernel ABC-SMC can't fully capture the
  correlated posterior) — exactly the approximation error SBC is designed to expose.
- **Neural SBI at scale: MDN + NSF, GPU-parallel, no embedding net.** The Bazin showcase fits
  `npe_mdn`/`npe_nsf` (amortized, 1 round × 30k simulations) and `snpe_mdn`/`snpe_nsf` (sequential,
  10 rounds × 3k) with the density estimators conditioning **directly on the raw light-curve vector**;
  each method trains on its own GPU, so all four neural fits run in parallel. Results: **every method
  recovers the truth within 1σ** (reduced χ² ≈ 1); **NPE-NSF is formally SBC-calibrated** (min
  rank-uniformity p = 0.057; three of four parameters p > 0.5) alongside exact MCMC (p = 0.229);
  NPE-MDN is the fastest accurate neural method (371 s end-to-end, best point recovery 0.24σ) but
  marginally over-confident (p = 0.036). Note: sbi 0.23.3's SNPE-C **non-atomic MoG loss has a CUDA
  device-mismatch bug** (triggered when the proposal is an MDN posterior), so `snpe_mdn` runs the
  **truncated sequential scheme** (`proposal_mode="restricted"`) — documented in the script.

### Scientific-review fixes (physical + Bayesian correctness)
A systematic, adversarially-verified review (physical consistency, Bayesian rigor, numerics) fixed:
- **Cross-sampler AIC/BIC now comparable.** ABC/ABC-SMC previously reported `-2 lnL` from the bare
  flux χ² (Gaussian normalisation dropped, flux-only), so their AIC/BIC were on a different scale than
  MCMC/SNPE — invalidating any model-selection table that mixed samplers. They now evaluate the **exact
  Gaussian log-likelihood at the best fit** in the data's natural space (`info['likelihood_space']`).
- **ABC-SMC is now importance-weighted** (Beaumont 2009 / Toni 2009): weighted resampling, an adaptive
  diagonal-Gaussian kernel (2× weighted population variance), and weights
  `w_i ∝ π(θ_i)/Σ_j w_j K(θ_i|θ_j)`; the equal-weight resampled posterior is returned and per-round
  effective sample size is recorded. The old **unweighted** variant biased the posterior.
- **`mck19` blackbody was missing a factor of π** (`F = πB(R/D)²`) — every predicted flux was π (~1.24
  mag) too faint (an error inherited from the reference). Fixed.
- **WAIC now drops non-finite *draws*, not data *points*** — a single bad draw no longer removes an
  entire data point, which had made ΔWAIC between models incomparable (different `n_data`); returns
  `n_draws_dropped` and `subsampled` and warns on both.
- **MCMC AIC/BIC use the max-*likelihood* draw**, not the max-*posterior* draw (these differ under a
  LogUniform prior); plus a convergence warning when `nsteps < 50·τ` (`info['converged']`).
- **CCM89 extinction clamps** out-of-range bands to the nearest regime (was silently `A=0`, e.g. for
  JWST/mid-IR) with a warning; `flux_density_to_mag` returns `NaN` + warns on non-positive flux (was a
  silent NaN with a sign-flipped error); SNPE guards an all-non-finite likelihood scan; ABC warns on an
  empty (0-accepted) posterior; the magnitude-space flux floor is documented.

### `plot_corner` + WAIC
- **`wp.plot_corner(posteriors, ...)`** — a built-in, documented corner plot for **overlaying a list of
  posteriors** (SamplerResults, DataFrames, dicts, or arrays) on one publication-ready figure: shared
  per-parameter ranges so panels align, a **dark, distinct palette** (`CORNER_PALETTE`), contour lines
  (not filled) so overlaps stay legible, `log_params=` for log axes, `truths=`, and a colour→label
  legend. `scripts/corner_kilonova_benchmark.py` uses it for the AT2017GFO benchmark posteriors
  (`docs/figures/at2017gfo_corner_flux.png`) — the corner shows whether the samplers are *compatible*
  (full posteriors + uncertainties), which the point-estimate table cannot.
- **`wp.waic(posterior, lc, model, ...)`** — the **Widely Applicable Information Criterion** (Watanabe
  2010), a fully-Bayesian fit score (lower is better) that uses the *whole* posterior: returns `waic`,
  `lppd`, `p_waic` (effective #params), `se`, with a `fixed=` hook for pinned parameters. Backed by a
  new pointwise log-likelihood (`GaussianLikelihood[WithUpperLimits].log_likelihood_pointwise`).

### SNPE on GPU
- **SNPE can now train on a GPU.** `fit_SNPE(..., device=...)` accepts `'cpu'` (default), `'cuda'` /
  `'gpu'` / `'cuda:N'`, or **`'auto'`** (CUDA when available, else CPU). The torch prior and observed
  data are now placed on the chosen device (fixing the sbi *"prior device must match training device"*
  error), and requesting a GPU without one **warns and falls back to CPU** instead of crashing.
- The GPU accelerates the neural-network **training**, not the (CPU) simulator — so it helps most with
  many simulations / large networks. `scripts/benchmark_snpe_device.py` measures GPU-vs-CPU runtime
  across a ladder of simulation counts (estimating each tier's time before running it, with a
  `--budget` guard) and saves a log-log plot (`docs/figures/snpe_device_benchmark.png`).
- Tested: `_resolve_device` mapping/fallback (no GPU needed) + a slow on-GPU recovery test
  (`skipif` no CUDA).

### Kilonova flux-vs-magnitude benchmark
- New **timed benchmark + sanity check** ([`docs/BENCHMARK.md`](docs/BENCHMARK.md),
  `scripts/benchmark_kilonova_modes.py`): `two_component_kilonova` fit to AT2017GFO (g/r/i) in
  **flux** vs **magnitude** space with ABC / MCMC / SNPE (6 configs). Records per-config **runtime**,
  AIC, RMS and posterior size; each config writes its own result file so the six run in parallel.
- **Publication-quality report figure** (`docs/figures/kilonova_benchmark_report.png`): larger fonts,
  colourblind-safe (Okabe–Ito) band colours, line-style per sampler, inward ticks, clear unit-labelled
  axes, and an **embedded table of the best-fit ejecta parameters + metrics per configuration**.
- Tested (`tests/test_benchmark_kilonova.py`): `setup` + magnitude-space distance (no redback) + a
  slow, guarded end-to-end fit→report.

### `two_component_kilonova` — first redback-backed model
- New built-in **`two_component_kilonova`** (`whisper_labia/models/two_component_kilonova.py`): a
  blue (low-κ) + red (high-κ) kilonova via the optional **redback** package (`[models]` extra), wrapping
  redback's `two_component_kilonova_model`. Parameters: `mej_1/2`, `vej_1/2`, `kappa_1/2`,
  `temperature_floor_1/2`, `redshift` (default prior follows the Darc kilonova-simulation setup).
- **redback is imported lazily** — WHISPER and `list_models()` work without it; only `predict` needs the
  extra (clear `ImportError` pointing to `pip install 'whisper-labia[models]'` otherwise). This is the
  template for a planned series of redback-backed models.
- Band-dependent: WHISPER bands → redback LSST filters (`g→lsstg`…); redback's band-integrated AB
  magnitude is converted to WHISPER's canonical **flux density (Jy)** as an exact (machine-precision)
  round-trip, so the shared likelihood/samplers treat it like any other model. Expensive simulator
  (~50 ms/call) → **SNPE** is the natural sampler; `predict` is module-level (parallel-ABC safe).
- `scripts/demo_kilonova.py` (light curve) and `scripts/fit_kilonova_at2017gfo.py` (ABC/MCMC/SNPE fit of
  AT2017GFO — which, being a real kilonova, this model fits well, unlike `mck19`).

### `mck19` physical model — BBH merger in an AGN disk
- New built-in **`mck19`** model (`whisper_labia/models/mck19.py`): the optical flare from a
  binary-black-hole merger embedded in an AGN accretion disk. A GW-recoil-kicked remnant shocks a
  bound-gas **hotspot** that radiates as a blackbody — a `sin²` rise to the ram-pressure delay `t_ram`,
  then exponential decay back to the disk baseline. McKernan et al. 2019
  ([ApJL 884, L50](https://iopscience.iop.org/article/10.3847/2041-8213/ab4886)); implementation of
  [Darc 2025](https://arxiv.org/abs/2506.02224). Parameters: `v_kick`, `M_smbh`, `M_bh`, `r_bh`,
  `redshift` (default prior spans the Darc 2025 grid).
- **First band-dependent built-in:** returns flux density (Jy) per `(time, band)`, evaluating the
  hotspot + disk blackbody at each band's effective wavelength (via Whisper's band system) and the
  source redshift (Planck18 luminosity distance + time dilation). Self-contained — astropy
  constants/cosmology only (no `speclite`/`extinction`); the AB magnitude is the monochromatic-at-`λ_eff`
  approximation to the original LSST filter integration. Fits with every sampler through the shared
  likelihood (an MCMC recovery test confirms data-mode-consistent magnitude-space fitting).
- `scripts/demo_mck19.py` renders the g/r/i light curve (`docs/figures/mck19_lightcurve.png`).

### MCMC sampler (emcee)
- New **`MCMCSampler`** (`mcmc`, `fit_MCMC`) — affine-invariant ensemble MCMC via `emcee` (a core
  dependency). The log-posterior is the Whisper prior + the **shared likelihood layer**
  (`make_likelihood`), so MCMC uses the *same physically consistent, `data_mode`-aware likelihood* as
  ABC/ABC-SMC/SNPE (flux data → flux space, magnitude data → magnitude space). Walkers init from the
  prior (or a given `initial_guess`); sampling is **seeded/reproducible**; exact Gaussian AIC/BIC; the
  `emcee.EnsembleSampler` is attached as `result.emcee_sampler`.
- `scripts/compare_samplers.py` — sanity check that ABC / ABC-SMC / MCMC / SNPE converge to the **same
  posterior** on `gaussian_rise`, with an overlaid corner plot.

### Production-readiness pass
- **Reproducibility (scientific bug fix):** ABC and ABC-SMC results are now **independent of `n_jobs`**.
  Each simulation/attempt owns an RNG stream derived from its *global index* (not the worker), so a fixed
  `seed` gives identical posteriors regardless of core count. SNPE seeds the simulator's observational
  noise per parameter row (same reason — a shared RNG made `num_workers>1` add identical noise). Locked
  by new determinism tests.
- **Four pluggable axes, all symmetric:** added `register_likelihood`/`list_likelihoods` and
  `register_distance`/`get_distance`/`list_distances` to match `register_model`/`register_sampler`;
  samplers now accept `distance=` by name. `register_manual_band` documents its process-global scope and
  gains `unregister_manual_band`/`clear_manual_bands`.
- **Packaging / open-source-ready:** added a `LICENSE` file (GPL-3.0), a `py.typed` marker (shipped via
  package-data) + `Typing :: Typed` classifier, single-sourced the version (`dynamic`), Python-version
  classifiers, and Documentation/Issues/Changelog URLs. Return-type hints on the public entry points.
- **Docs:** new `docs/DESIGN.md` (design rationale + known limitations), `CONTRIBUTING.md`, `CITATION.cff`;
  ABC/ABC-SMC `fit` docstrings; magic-number guards promoted to named constants; doc accuracy fixes.

### LightCurve is now an `astropy.table.Table`
- `LightCurve` **subclasses `astropy.table.Table`**: per-point quantities are columns, scalar metadata
  lives in `.meta`, and full table semantics work — `lc['new'] = lc['flux'] + 5`, boolean-mask slicing
  (`lc[lc['magnitude'] < 18]`, which keeps the subclass + `.meta`), `sort` / `group_by`, etc. `lc()`
  returns the table itself. The common quantities stay available as attributes (`lc.time`, `lc.flux`,
  `lc.redshift`, …) so the samplers/likelihood/plotting are unchanged. `to_dataframe()` → `to_pandas()`.
- New methods: **`where(**constraints)`** (`col` / `col_min` / `col_max` / `col_not`, list = OR);
  **`calc_phase(reference=, redshift=, peak=, hours=)`** — rest-frame phase `(t − ref)/(1+z)`;
  **`calc_absmag(dm=, redshift=, ebv=, rv=3.1, extinction=)`** — distance modulus (from `z` /
  `luminosity_distance`) + Milky-Way extinction (CCM89 from `ebv`/`rv`, or an explicit `{band: A_mag}`
  dict) → an `absmag` column. Inspired by `lightcurve_fitting.LC`.

### Survey-CSV ingestion: data_mode, redshift, astropy units, SVO band fallback
- `LightCurve.data_mode` is now a stored attribute in `{flux_density, magnitude, flux}` (default
  inferred from the columns; explicit override) with an `output_format` property (`magnitude` /
  `flux_density`) — the forward-model comparison space (the optional redback backend uses the same).
- **Redshift** is resolved `redshift=` argument > `redshift` column > unknown. Unknown does *not* fail:
  the curve records `redshift_known=False` + a default `redshift_prior` and warns that `z` will be
  sampled. Validation: `z >= 0`; `z == 0` requires an explicit `luminosity_distance`; negative/NaN is fatal.
- **astropy.units throughout** (`io/units.py`): `flux_density` accepts F_nu (Jy/mJy/µJy) or F_lambda
  (erg/s/cm²/Å) and stores Jy canonically, converting F_lambda via `u.spectral_density(lambda_eff)`
  (errors clearly when the per-point effective wavelength is missing); `magnitude` must be dimensionless
  AB; band-integrated `flux` validates erg/s/cm² dimensionality. No-unit columns warn and apply a
  documented per-mode default.
- **Bands** resolve to per-point effective wavelength + zero point via `FILTER_LOOKUP` anchored to the
  LSST ugrizy zero points (`resolve_band` / `resolve_bands`, `LSST_BAND_INFO`). A miss warns and falls
  back to the **SVO Filter Profile Service** (`io/svo.py`, `astroquery.svo_fps`) — results cached by
  filter ID (offline-safe; corrupt/unusable cache entries are ignored, never crash a load), graceful
  degradation on network failure with a `register_manual_band` override path. astroquery is an optional
  `[svo]` extra; all SVO/network calls are mocked in tests.
- Note: `add_flux`/`add_mag` keep the constant **AB 3631 Jy** zero point so the modelling flux the
  samplers/likelihood consume stays on one zero point across bands; pass `zeropoint_jy=lc.zero_point` to
  opt into the per-band (LSST/SVO) zero points.

### Data ingestion & plotting
- `LightCurve` canonical container with pandas-style `select_*` / `add_*` methods, an `snr` property,
  `set_explosion_date`, and upper-limit support.
- `load_lightcurve` — flexible CSV loader: column auto-detection, band normalization + `FILTER_LOOKUP`
  broadband grouping, mag↔flux conversion, quality cuts, SNR cut (`min_snr`), time window, explosion date.
- `plot_light_curve` — report (magnitude + flux) and per-band grid layouts; markers for detections
  (circles), SNR<3 (△) and upper limits (▽); apparent/absolute magnitude or flux.

### Inference (pluggable models + samplers)
- Priors: `Uniform`, `LogUniform`, `Prior` (sample / log_prob / rescale; picklable for parallel ABC).
- Models: registry (`register_model` / `list_models`) + built-ins **`flare`**, **`bazin`**,
  **`gaussian_rise`**.
- Distance: `chi2_distance` (pluggable; any `f(obs, err, sim, bands) -> float`).
- Samplers: registry (`register_sampler` / `list_samplers`) + **`ABCSampler`** (parallel rejection),
  **`ABCSMCSampler`** (sequential, adaptive or explicit epsilon), and **`SNPESampler`** — **Sequential
  Neural Posterior Estimation** (`snpe`/`npe`, `fit_SNPE`) via `sbi`: the simulator is Whisper's forward
  model + Gaussian noise, the prior is auto-adapted to torch (`Uniform`→`BoxUniform`, mixed→
  `MultipleIndependent`), `num_rounds>1` is sequential, and the trained sbi posterior is attached as
  `result.posterior`. **Flexible:** a custom `embedding_net` (`torch.nn.Module`), a custom
  `density_estimator` (or `posterior_nn` `hidden_features`/`num_transforms`/`num_bins`), parallel
  `num_workers`, and `proposal_mode="restricted"` for truncated SNPE (`RestrictedPrior` +
  `get_density_thresholder`, with `support_samples` capping the otherwise-1e6 support estimate). Lazy
  optional `[sbi]` extra. `SamplerResult` with posterior summary, best-fit, AIC / BIC / max-log-likelihood
  (χ² ≙ −2 ln L; SNPE uses the exact Gaussian value), `to_json`.
- Notebook quick-start: `examples/at2017gfo_quickstart.ipynb` — load AT2017GFO, register a **custom
  model**, fit with ABC, compare models, and run SNPE.
- Likelihoods (`likelihood.py`): `GaussianLikelihood`, `GaussianLikelihoodWithUpperLimits`,
  `MixtureGaussianLikelihood`, `make_likelihood` — **flux or apparent-magnitude space**, default by
  data type, upper limits in flux/mag. _Standalone + tested; sampler integration pending._

### Fixed
- `flare` returns 0 before the explosion (was negative for `t<0`); `bazin` is evaluated in log-space
  (numerically stable — no early-time plateau for `tau_rise < tau_fall`).
- SNPE: the trained `result.posterior` now has its default `x` set every round, so
  `result.posterior.sample()` works without re-passing `x` (regressed in the truncated-proposal refactor
  for `num_rounds=1`).

### Packaging & docs
- pip-installable from GitHub; relaxed dependency pins (no forced numpy/scipy downgrade); redback is an
  optional `[models]` extra — Phase-1 data + plotting + ABC run with no redback and no compiler.
- Tutorial, API reference, design rationale, extensibility + contributing guides, an AT2017GFO
  model-comparison report, and a quick-start notebook. LICENSE (GPL-3.0), CITATION.cff, py.typed. 189 tests (redback-backed model + benchmark tests skip without the [models] extra).
