# Whisper (`whisper_labia`) — API Reference

Generated for **v0.0.1.dev0**. Covers **Phase 1 (data ingestion + plotting)** and the **ABC inference
layer** (pluggable models, priors, distance, samplers).

- **Environment:** Docker container `phe_sbi`, Python 3.11.
- **Run tests:** `docker exec phe_sbi bash -lc 'cd /tf/astrodados2/phelipedata2/WHISPER/whisper-labia && python -m pytest tests -q'` (133 tests).

## Package map

```
whisper_labia/
  __init__.py          # public API exports
  plotting.py          # plot_light_curve
  priors.py            # Uniform, LogUniform, Prior
  distance.py          # chi2_distance
  likelihood.py        # GaussianLikelihood, ...WithUpperLimits, Mixture..., make_likelihood
  models/              # Model + register/get/list + built-ins flare/bazin/gaussian_rise
  samplers/            # BaseSampler, SamplerResult, ABCSampler, ABCSMCSampler, fit_ABC(_SMC), fit
  io/                  # LightCurve, load_lightcurve, bands, photometry, units, svo
```

Top-level (`import whisper_labia as wp`): `LightCurve`, `load_lightcurve`, `plot_light_curve`,
`group_bands`, `FILTER_LOOKUP`, `resolve_band`, `resolve_bands`, `LSST_BAND_INFO`,
`register_manual_band`, `SvoUnavailable`, `Prior`, `Uniform`, `LogUniform`, `Model`, `register_model`,
`get_model`, `list_models`, `chi2_distance`, `GaussianLikelihood`, `GaussianLikelihoodWithUpperLimits`,
`MixtureGaussianLikelihood`, `make_likelihood`, `fit_ABC`, `fit_ABC_SMC`, `fit`, `SamplerResult`,
`register_sampler`, `list_samplers`.

---

## 1. `LightCurve`  (`io.schema`)

Constructor fields: `time`, `band` (required); `magnitude`, `magnitude_err`, `flux`, `flux_err`,
`upper_limit`, `system`, `lambda_eff`, `zero_point` (optional per-point); `name`, `redshift`,
`luminosity_distance`, `data_mode`, `redshift_prior`, `meta` (scalar). Requires `band` and at least one
of `magnitude`/`flux`. `flux` holds **flux density in Jy** (the canonical internal unit).

- **`data_mode`** ∈ `{flux_density, magnitude, flux}` — stored; inferred from the columns when omitted
  (mag-only → `magnitude`, else `flux_density`). Invalid values raise.
- **Redshift** is validated in `__post_init__`: finite & `≥ 0`; `z == 0` requires `luminosity_distance`
  (Mpc); negative/NaN raises. When `redshift is None` the curve is *unknown*: `redshift_known` is
  `False` and a default `redshift_prior` is attached (the loader emits the one-time warning).

Properties: `n_points`, `bands`, `data_mode`, `output_format` (the forward-model comparison space,
`'magnitude'`/`'flux_density'`), `redshift_known`, `snr` (`flux/flux_err` or `(2.5/ln10)/mag_err`).

Methods (each returns a new `LightCurve` unless noted): `select_bands`, `select_time_window`,
`select_snr(min_snr=5.0)`, `add_flux(zeropoint_jy=3631.0)` / `add_mag(...)` (constant **AB** zero point;
raise for `data_mode='flux'`), `resolve_bands(svo_fallback=True)` (fill `lambda_eff`/`zero_point`),
`set_explosion_date`, `to_dataframe()`. **`lc()` (`__call__`)** returns the enriched dataframe: for
`flux_density`/`magnitude` the missing column is derived from the **per-band** zero point (`lambda_eff`,
`zero_point` included); band-integrated `flux` keeps only the flux column.

## 2. `load_lightcurve(path, *, ...)` → `LightCurve`  (`io.loader`)

Auto-maps columns; key options: `column_map`, `band_lookup` (broadband grouping), `default_band`,
`quality_cuts`, `flag_filters={'catflags':0}`, `time_min/time_max`, `bands`, `min_snr`,
`explosion_date`, `delimiter`. Detects an `upper_limit` column. Raises `ValueError` for a missing
time / band / measurement column.

Ingestion options:

| Argument | Default | Description |
|---|---|---|
| `redshift` | `None` | Explicit redshift; otherwise a `redshift` column is used, else unknown (warns). |
| `luminosity_distance` | `None` | Mpc; required when `redshift == 0`. |
| `data_mode` | `None` | `flux_density`/`magnitude`/`flux`; inferred from the columns when omitted. |
| `flux_unit` | `None` | astropy unit of the flux column — F_ν (Jy/mJy/µJy) or F_λ (erg/s/cm²/Å). `None` → warn + assume Jy. |
| `magnitude_unit` | `None` | Must be dimensionless AB; a flux unit raises. `None` → warn + assume dimensionless. |
| `resolve_band_info` | `True` | Fill per-point `lambda_eff` + `zero_point` from the bands. |
| `svo_fallback` | `True` | Query SVO for bands missing from `FILTER_LOOKUP`. |

## 3. Bands & SVO resolution  (`io.bands`, `io.svo`)
`normalize_band(s)`, `group_bands(bands, lookup=FILTER_LOOKUP)`, `unmapped_bands`, plus
`DEFAULT_BAND_ALIASES` and `FILTER_LOOKUP` (88 labels → 10 effective bands).

- **`resolve_band(band, *, lookup=None, svo_fallback=True, lambda_eff_hint=None, warn=True)`** →
  `{group, lambda_eff (Å), zero_point (Jy), filter_id, source}`. Order: `FILTER_LOOKUP` group →
  `LSST_BAND_INFO` (optical anchored to **LSST ugrizy**; NIR documented) → SVO fallback → unresolved
  (warn). `source` ∈ `{lsst, documented, svo, manual, unresolved}`.
- **`resolve_bands(bands, *, lookup=None, svo_fallback=True, warn=True)`** → `(lambda_eff, zero_point,
  info)` arrays (NaN where unresolved); resolves each distinct band once.
- **`LSST_BAND_INFO`** — effective wavelength + zero point per effective band.

**SVO** (`io.svo`, needs the `[svo]` extra → `astroquery`): `resolve_band_svo(band, *,
lambda_eff_hint=None)`, `get_filter_metadata(filter_id)` (cached by ID; offline-safe disk cache under
`$WHISPER_SVO_CACHE`), `get_transmission_data(filter_id)` (Phase-2 spectral integration),
`find_filter_id`, `register_manual_band(band, lambda_eff, zero_point)`, `clear_cache`. Network failure /
unknown filter raises `SvoUnavailable`, which `resolve_band` turns into a warning + the manual-override
path. Corrupt/unusable cache entries are ignored, never crash a load. All network access goes through
`_svo_fetch_metadata` / `_svo_fetch_index` / `_svo_fetch_transmission` (the points the tests mock).

## 4. Photometry & units  (`io.photometry`, `io.units`)
`mag_to_flux_density`, `flux_density_to_mag`, `mag_err_to_snr`; `AB_ZEROPOINT_JY=3631.0`,
`POGSON=2.5/ln10`.

`io.units` stores **one canonical unit per mode**: `flux_density` → Jy, `flux` → erg/s/cm², `magnitude`
→ dimensionless AB. `to_canonical(values, unit, data_mode, *, lambda_eff=None, warn_default=True)`
validates + converts; `to_flux_density_jy` accepts F_ν directly and F_λ via `u.spectral_density(λ_eff)`
(**requires** a per-point wavelength — errors clearly, naming the offending points, when missing or
NaN); `check_magnitude_unit` rejects a flux unit on a magnitude column. A no-unit column warns and
applies the documented per-mode default.

## 5. Plotting — `plot_light_curve(lc, *, layout="report", quantity="apparent_mag", bands=None, ncols=3, figsize=None, title=None, save=None)`
`layout`: `"report"` (mag + flux panels) or `"grid"` (per band). `quantity`: `apparent_mag` /
`absolute_mag` (needs redshift) / `flux`. Markers: detections = circles, SNR<3 = △, upper limits = ▽.

---

## 6. Inference: priors, models, distance, samplers

### 6.1 Priors  (`whisper_labia.priors`)
Small, **picklable** distributions (so they cross process boundaries in parallel ABC).

| Class | Constructor | Methods |
|---|---|---|
| `Uniform` | `(low, high, name=None)` | `sample(rng)`, `log_prob(x)`, `rescale(u)`, `bounds` |
| `LogUniform` | `(low, high, name=None)` (`low>0`) | same |
| `Prior` | `(distributions: dict)` | `sample(rng=None) -> dict`, `log_prob(params)`, `rescale(unit_cube)`, `names`, `bounds` |

```python
prior = wp.Prior({"amplitude": wp.Uniform(0, 10), "rise_time": wp.Uniform(1, 10)})
prior.sample(np.random.default_rng(0))   # {'amplitude': ..., 'rise_time': ...}
```

### 6.2 Models  (`whisper_labia.models`)
A model maps parameters to predicted flux: `predict(params: dict, times: np.ndarray, bands) -> np.ndarray`.

| Function | Signature | Description |
|---|---|---|
| `register_model` | `(name, predict, parameters, prior=None, description="", *, overwrite=False)` | Register a model by name. |
| `get_model` | `(model)` | Resolve a name (or pass a `Model`). |
| `list_models` | `()` | Sorted registered model names. |
| `Model` | dataclass: `name, predict, parameters, default_prior, description` | Callable: `model(params, times, bands)`. |

Built-in models (band-independent, vectorized; `t` = days since explosion; scale `amplitude` to your
flux units for real data):

| Name | Form | Parameters |
|---|---|---|
| `flare` | `A·(1 − e^(−t/t_rise))·e^(−t/t_decay)` | amplitude, rise_time, decay_time |
| `bazin` | `A·e^(−(t−t0)/τ_fall) / (1 + e^(−(t−t0)/τ_rise))` | amplitude, t0, tau_rise, tau_fall |
| `gaussian_rise` | Gaussian rise to peak at `t0`, then exp decay | amplitude, t0, sigma_rise, tau_decay |

> For **parallel** ABC (`n_jobs>1`) the `predict` function must be picklable (module-level, not a
> closure/lambda). Closures work with `n_jobs=1`.

### 6.3 Distance  (`whisper_labia.distance`)
`chi2_distance(obs_flux, obs_flux_err, sim_flux, bands=None) -> float` = `sum(((obs-sim)/err)**2)`.
Any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float` can be passed as a custom distance.

### 6.4 Samplers  (`whisper_labia.samplers`)

`fit_ABC(lc, model="flare", ...)` and `fit_ABC_SMC(lc, model="flare", ...)` → `SamplerResult`.
`fit(lc, model, sampler="abc"|"abc_smc", **kwargs)` is the generic dispatcher. `list_samplers()` →
`['abc', 'abc_smc']`; `register_sampler(name, cls)` adds your own.

**`ABCSampler.fit(lc, model, prior=None, *, ...)`**:

| Argument | Default | Description |
|---|---|---|
| `n_simulations` | `10000` | Total prior draws / simulations. |
| `quantile` | `0.01` | Accept the best fraction by distance (robust default). |
| `threshold` | `None` | Fixed acceptance distance ε (overrides `quantile` if set). |
| `distance` | `chi2_distance` | Distance function. |
| `n_jobs` | `None` (→ min(cpu, 8)) | Processes for parallel simulation. |
| `seed` | `0` | RNG seed (independent streams per worker via `SeedSequence`). |

**`ABCSMCSampler.fit(lc, model, prior=None, *, n_particles=500, n_rounds=5, epsilon_schedule=None,
quantile=0.5, perturbation_scale=0.1, distance=chi2_distance, n_jobs=None, seed=0)`** — sequential
rejection: round 0 draws from the prior; later rounds resample + Gaussian-perturb accepted particles
under a shrinking epsilon (explicit `epsilon_schedule`, or adaptive `quantile` of the previous round's
distances). Perturbs only parameters and rejects proposals outside the prior; `info` carries per-round
epsilon / acceptance / `total_simulations`.

**`SamplerResult`** fields: `sampler`, `model`, `parameters`, `samples` (DataFrame of accepted draws
+ `distance`), `summary` (median/ci16/ci84/mean/std per param), `best_params`, `n_data`, `n_params`,
`runtime_s`, `info` (n_simulations, n_accepted, acceptance_rate, epsilon, quantile, n_jobs),
`min_distance`, `max_log_likelihood`, `aic`, `bic`. Methods: `n_samples`, `to_dict()`, `to_json(path=None)`.

> Metrics note: for the χ² distance, `chi2 = -2 ln L` (Gaussian), so `max_log_likelihood = -0.5·χ²_min`,
> `AIC = χ²_min + 2k`, `BIC = χ²_min + k·ln(n)`.

```python
res = wp.fit_ABC(r_band_lc, "flare", prior=prior, n_simulations=200_000, quantile=0.005, n_jobs=16)
res.summary["amplitude"]    # {'median':..., 'ci16':..., 'ci84':...}
res.best_params; res.aic; res.bic
res.to_json("fit.json")
```

---

### 6.5 Likelihoods  (`whisper_labia.likelihood`)

Models predict **flux**; a likelihood compares it to the data in a chosen **space** —
`space='flux'` (residuals/errors in Jy; upper limits usable), `space='magnitude'` (model flux → AB
mag vs observed mag/err), or `space='auto'` (magnitude data → magnitude space, flux data → flux space;
the correct default). Each exposes `log_likelihood(model_flux) -> float` and is picklable.

| Class / function | Purpose |
|---|---|
| `GaussianLikelihood(lc, space="auto")` | Independent Gaussian in the chosen space. |
| `GaussianLikelihoodWithUpperLimits(lc, space="auto", upper_limit_sigma=3.0)` | Gaussian for detections + a CDF (flux: P(true<limit)) / survival (mag: P(true>limit)) term for upper limits. |
| `MixtureGaussianLikelihood(lc, space="auto", alpha=0.9, sigma_out_scale=10.0)` | Outlier-robust two-component mixture (α, σ_out fixed). |
| `make_likelihood(lc, kind="auto", space="auto", **kw)` | Build the data-appropriate likelihood (auto-selects upper-limits when present). |

> Status: implemented + tested but **not yet wired into the samplers** (ABC/ABC-SMC currently score
> with `chi2_distance`). Adding `likelihood=` / `space=` to the samplers is the next step.

## Notes & limitations (review findings)

- **Metrics vs likelihood:** ABC/ABC-SMC score with the χ² distance, so `AIC`/`BIC`/
  `max_log_likelihood` are χ²-based — exact for **model comparison on the same data**, but offset by
  the Gaussian normalization constant in absolute terms. The likelihood layer makes them exact.
- **ABC-SMC is unweighted** (uniform parent resampling, no importance weights): best-fit + an
  approximate posterior are reliable; rigorous posterior weights are a planned option.
- **ABC posteriors are approximate** (broadened by the acceptance ε); tighten ε / use SMC for sharper ones.
- **Toy models** are band-independent analytic forms: `flare` = 0 before explosion; `bazin` computed
  stably in log-space; `gaussian_rise` has a derivative kink at the peak. Physical, band-dependent
  models can optionally be supplied by the external redback `[models]` extra (a models/priors source only).
- **SNR(magnitude)** uses `(2.5/ln10)/σ_m`, valid for small magnitude errors.

## 7. Internals
`io.loader._resolve_columns`, `schema.LightCurve._subset/_copy`, `plotting._categories/_scatter`,
`samplers.abc._simulate_batch/_worker`, `samplers.base.summarize_posterior`,
`io.units.to_canonical`, `io.svo._svo_fetch_metadata/_svo_fetch_index/_svo_fetch_transmission`
(network boundary), `scripts/{phase0_smoke,demo_abc_at2017gfo,demo_ingestion}.py`.

## 8. Test coverage (133 tests, all passing)

| File | Tests | Focus |
|---|---|---|
| `test_photometry.py` | 5 | AB zeropoint, mag↔flux, error propagation, SNR. |
| `test_bands.py` | 11 | Aliases, case-sensitivity, `group_bands`/`FILTER_LOOKUP`, `unmapped_bands`. |
| `test_schema.py` | 11 | Validation, subsetting, `add_*`, `snr`/`select_snr`, `set_explosion_date`, upper limits. |
| `test_loader.py` | 12 | AT2017GFO load, window/subset, grouping, `min_snr`, `explosion_date`, upper limits. |
| `test_plotting.py` | 5 | report/grid layouts, flux/absolute-mag, redshift guard, upper-limit markers. |
| `test_priors.py` | 6 | Uniform/LogUniform/Prior sampling, log_prob, rescale, picklability. |
| `test_models.py` | 8 | flare/bazin/gaussian_rise (incl. flare pre-explosion=0, bazin tail→0), custom model, errors. |
| `test_distance.py` | 2 | chi² zero/known value. |
| `test_abc.py` | 6 | parameter recovery, serial+parallel, acceptance count, JSON, dispatch, custom model. |
| `test_abc_smc.py` | 6 | SMC registration, recovery, epsilon tightening, explicit schedule, dispatch, model-agnostic. |
| `test_likelihood.py` | 9 | Gaussian flux/mag, space-auto, upper-limit CDF, mixture, make_likelihood, picklable. |
| `test_units.py` | 12 | F_ν/F_λ→Jy, per-point λ, NaN-λ error, mag rejects flux unit, no-unit default, flux dimensionality. |
| `test_svo.py` | 15 | FILTER_LOOKUP→SVO, mocked metadata/index, cache hit + disk cache, corrupt-cache (×4 params), graceful degrade, manual override (no spurious warn), transmission, ambiguity. |
| `test_ingestion.py` | 25 | data_mode/output_format, `__call__`, redshift (arg/column/unknown/0/neg/NaN/all-NaN), units, band-info, subset preservation, backward-compat ZP, flux-mode. |

Fixtures: `tests/data/at2017gfo.csv`, `tests/data/ztf18aarlhfw.csv`. Figures + ABC JSON in `docs/figures/`.
All SVO/network calls are **mocked** — no live network in CI. See
[`REPORT_ingestion_upgrade.md`](REPORT_ingestion_upgrade.md) for the full per-test results.

## 9. End-to-end example

```python
import whisper_labia as wp

lc = wp.load_lightcurve("at2017gfo.csv", explosion_date=57982.0, min_snr=3)
r  = lc.select_bands("r")
fmax = r.add_flux().flux.max()
prior = wp.Prior({"amplitude": wp.Uniform(0, 10*fmax),
                  "rise_time": wp.Uniform(0.05, 10), "decay_time": wp.Uniform(0.5, 40)})
res = wp.fit_ABC(r, "flare", prior=prior, n_simulations=200_000, quantile=0.005, n_jobs=16)
print(res, res.best_params)
```
