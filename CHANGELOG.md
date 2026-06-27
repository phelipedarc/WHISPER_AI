# Changelog

All notable changes to Whisper (`whisper_labia`). The project is in early development; nothing is
released to PyPI yet — install from GitHub (`pip install git+https://github.com/phelipedarc/WHISPER_AI.git`).

## [Unreleased] — 0.0.1.dev0

### Survey-CSV ingestion: data_mode, redshift, astropy units, SVO band fallback
- `LightCurve.data_mode` is now a stored attribute in `{flux_density, magnitude, flux}` (default
  inferred from the columns; explicit override) with an `output_format` property (`magnitude` /
  `flux_density`) — the forward-model comparison space (the optional redback backend uses the same).
  Calling the object
  (`lc()` / `to_dataframe()`) returns the **enriched** dataframe with both flux and magnitude filled in
  from the per-band zero point.
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
- Note: the enriched dataframe (`lc()`) uses the **per-band** zero point; the `add_flux`/`add_mag`
  helpers keep the constant **AB 3631 Jy** zero point so the modelling flux the samplers/likelihood
  consume stays on one zero point across bands (backward-compatible).

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

### Packaging & docs
- pip-installable from GitHub; relaxed dependency pins (no forced numpy/scipy downgrade); redback is an
  optional `[models]` extra — Phase-1 data + plotting + ABC run with no redback and no compiler.
- Tutorial, API reference, extensibility guide, an AT2017GFO model-comparison report, and a quick-start
  notebook. 141 tests.
