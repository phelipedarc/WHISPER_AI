# Changelog

All notable changes to Whisper (`whisper_labia`). The project is in early development; nothing is
released to PyPI yet — install from GitHub (`pip install git+https://github.com/phelipedarc/WHISPER_AI.git`).

## [Unreleased] — 0.0.1.dev0

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
