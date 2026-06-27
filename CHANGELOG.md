# Changelog

All notable changes to Whisper (`whisper_labia`). The project is in early development; nothing is
released to PyPI yet — install from GitHub (`pip install git+https://github.com/phelipedarc/WHISPER_AI.git`).

## [Unreleased] — 0.0.1.dev0

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
  model-comparison report, and a quick-start notebook. LICENSE (GPL-3.0), CITATION.cff, py.typed. 153 tests.
