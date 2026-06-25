# Whisper (`whisper_labia`) — API Reference

Generated for **v0.0.1.dev0**. Covers **Phase 1 (data ingestion + plotting)** and the **ABC inference
layer** (pluggable models, priors, distance, samplers).

- **Environment:** Docker container `phe_sbi`, Python 3.11.
- **Run tests:** `docker exec phe_sbi bash -lc 'cd /tf/astrodados2/phelipedata2/WHISPER/whisper-labia && python -m pytest tests -q'` (70 tests).

## Package map

```
whisper_labia/
  __init__.py          # public API exports
  plotting.py          # plot_light_curve
  priors.py            # Uniform, LogUniform, Prior
  distance.py          # chi2_distance
  models/              # Model, register_model/get_model/list_models + built-in `flare`
  samplers/            # BaseSampler, SamplerResult, ABCSampler, fit_ABC, fit, register/list_samplers
  io/                  # LightCurve, load_lightcurve, bands, photometry
```

Top-level (`import whisper_labia as wp`): `LightCurve`, `load_lightcurve`, `plot_light_curve`,
`group_bands`, `FILTER_LOOKUP`, `Prior`, `Uniform`, `LogUniform`, `Model`, `register_model`,
`get_model`, `list_models`, `chi2_distance`, `fit_ABC`, `fit_ABC_SMC`, `fit`, `SamplerResult`, `register_sampler`,
`list_samplers`.

---

## 1. `LightCurve`  (`io.schema`)

Constructor fields: `time`, `band` (required); `magnitude`, `magnitude_err`, `flux`, `flux_err`,
`upper_limit`, `system` (optional per-point); `name`, `redshift`, `meta` (scalar). Requires `band`
and at least one of `magnitude`/`flux`.

Properties: `n_points`, `bands`, `data_mode`, `snr` (`flux/flux_err` or `(2.5/ln10)/mag_err`).

Methods (each returns a new `LightCurve`): `select_bands(bands)`, `select_time_window(time_min,
time_max)`, `select_snr(min_snr=5.0)`, `add_flux(zeropoint_jy=3631.0)`, `add_mag(...)`,
`set_explosion_date(explosion_date)` (time → days since explosion), `to_dataframe()`.

## 2. `load_lightcurve(path, *, ...)` → `LightCurve`  (`io.loader`)

Auto-maps columns; key options: `column_map`, `band_lookup` (broadband grouping), `default_band`,
`quality_cuts`, `flag_filters={'catflags':0}`, `time_min/time_max`, `bands`, `min_snr`,
`explosion_date`, `delimiter`. Detects an `upper_limit` column. Raises `ValueError` for a missing
time / band / measurement column.

## 3. Band utilities  (`io.bands`)
`normalize_band(s)`, `group_bands(bands, lookup=FILTER_LOOKUP)`, `unmapped_bands`, plus
`DEFAULT_BAND_ALIASES` and `FILTER_LOOKUP` (88 labels → 10 effective bands).

## 4. Photometry  (`io.photometry`)
`mag_to_flux_density`, `flux_density_to_mag`, `mag_err_to_snr`; `AB_ZEROPOINT_JY=3631.0`,
`POGSON=2.5/ln10`.

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

## 7. Internals
`io.loader._resolve_columns`, `schema.LightCurve._subset/_copy`, `plotting._categories/_scatter`,
`samplers.abc._simulate_batch/_worker`, `samplers.base.summarize_posterior`,
`scripts/{phase0_smoke,demo_abc_at2017gfo}.py`.

## 8. Test coverage (70 tests, all passing)

| File | Tests | Focus |
|---|---|---|
| `test_photometry.py` | 5 | AB zeropoint, mag↔flux, error propagation, SNR. |
| `test_bands.py` | 11 | Aliases, case-sensitivity, `group_bands`/`FILTER_LOOKUP`, `unmapped_bands`. |
| `test_schema.py` | 11 | Validation, subsetting, `add_*`, `snr`/`select_snr`, `set_explosion_date`, upper limits. |
| `test_loader.py` | 12 | AT2017GFO load, window/subset, grouping, `min_snr`, `explosion_date`, upper limits. |
| `test_plotting.py` | 5 | report/grid layouts, flux/absolute-mag, redshift guard, upper-limit markers. |
| `test_priors.py` | 6 | Uniform/LogUniform/Prior sampling, log_prob, rescale, picklability. |
| `test_models.py` | 6 | flare/bazin/gaussian_rise vectorization, custom model, unknown-model error. |
| `test_distance.py` | 2 | chi² zero/known value. |
| `test_abc.py` | 6 | parameter recovery, serial+parallel, acceptance count, JSON, dispatch, custom model. |
| `test_abc_smc.py` | 6 | SMC registration, recovery, epsilon tightening, explicit schedule, dispatch, model-agnostic. |

Fixtures: `tests/data/at2017gfo.csv`, `tests/data/ztf18aarlhfw.csv`. Figures + ABC JSON in `docs/figures/`.

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
