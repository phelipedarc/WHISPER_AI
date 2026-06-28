# Whisper (`whisper_labia`) — API Reference

Generated for **v0.0.1.dev0**. Covers data ingestion + plotting and the inference layer — four pluggable
axes (models, samplers, likelihoods, distances) with the ABC / ABC-SMC / MCMC / SNPE samplers.

- **Environment:** Docker container `phe_sbi`, Python 3.11.
- **Run tests:** `docker exec phe_sbi bash -lc 'cd /tf/astrodados2/phelipedata2/WHISPER/whisper-labia && python -m pytest tests -q'` (183 tests; `-m "not slow"` skips the SNPE training/GPU + mck19 / kilonova fit + benchmark tests).

## Package map

```
whisper_labia/
  __init__.py          # public API exports
  plotting.py          # plot_light_curve, plot_corner
  metrics.py           # waic (Widely Applicable Information Criterion)
  priors.py            # Uniform, LogUniform, Prior
  distance.py          # chi2_distance
  likelihood.py        # GaussianLikelihood, ...WithUpperLimits, Mixture..., make_likelihood
  models/              # Model + register/get/list + built-ins flare/bazin/gaussian_rise + physical mck19 + redback two_component_kilonova
  samplers/            # BaseSampler, SamplerResult, ABCSampler, ABCSMCSampler, MCMCSampler, SNPESampler, fit_ABC(_SMC)/fit_MCMC/fit_SNPE, fit
  io/                  # LightCurve, load_lightcurve, bands, photometry, units, svo
```

Top-level (`import whisper_labia as wp`): `LightCurve`, `load_lightcurve`, `plot_light_curve`,
`plot_corner`, `CORNER_PALETTE`, `waic`,
`group_bands`, `FILTER_LOOKUP`, `resolve_band`, `resolve_bands`, `LSST_BAND_INFO`, `SvoUnavailable`,
`register_manual_band`, `unregister_manual_band`, `clear_manual_bands`, `Prior`, `Uniform`, `LogUniform`,
`Model`, `register_model`, `get_model`, `list_models`, `chi2_distance`, `register_distance`,
`get_distance`, `list_distances`, `GaussianLikelihood`, `GaussianLikelihoodWithUpperLimits`,
`MixtureGaussianLikelihood`, `make_likelihood`, `register_likelihood`, `list_likelihoods`, `fit_ABC`,
`fit_ABC_SMC`, `fit_MCMC`, `fit_SNPE`, `fit`, `SamplerResult`, `register_sampler`, `list_samplers`.

---

## 1. `LightCurve`  (`io.schema`)

**`LightCurve` is a subclass of `astropy.table.Table`.** Per-point quantities are **columns** (`time`,
`band`, `magnitude`, `magnitude_err`, `flux`, `flux_err`, `upper_limit`, `system`, `lambda_eff`,
`zero_point`, plus any you add) and scalar metadata lives in **`.meta`** (`name`, `redshift`,
`data_mode`, `luminosity_distance`, `redshift_prior`, `dm`, `refmjd`, …). So full table semantics work:

```python
lc['absmag_shift'] = lc['magnitude'] + 5        # add / compute columns
bright = lc[lc['magnitude'] < 18]               # boolean-mask slicing (keeps the subclass + .meta)
lc.sort('time'); lc.group_by('band')            # any astropy Table method
lc()                                            # __call__ -> the table itself
```

Construct from arrays — `LightCurve(time=, band=, magnitude=|flux=, …, name=, redshift=, data_mode=)`
(requires `band` + at least one of `magnitude`/`flux`; `flux` is flux density in Jy) — or from anything
`Table` accepts. `data_mode` ∈ `{flux_density, magnitude, flux}` (inferred from the columns when
omitted). Redshift is validated at construction: finite & `≥ 0`; `z == 0` requires `luminosity_distance`
(Mpc); negative/NaN raises; `None` → *unknown* (`redshift_known=False` + a default `redshift_prior`).

For convenience the common quantities are **also** attributes: `lc.time` / `lc.flux` / `lc.band` … return
the column data (`None` if absent; settable); `lc.redshift` / `lc.data_mode` / `lc.name` /
`lc.redshift_known` / `lc.luminosity_distance` / `lc.output_format` read `.meta`; `lc.n_points`,
`lc.bands`, `lc.snr` are derived.

Methods (return a new `LightCurve` unless noted): **`where(**constraints)`** (`col` / `col_min` /
`col_max` / `col_not`, list = OR — `lc.where(band='r', time_min=58000, upper_limit=False)`);
`select_bands` / `select_time_window` / `select_snr`; `add_flux(zeropoint_jy=3631.0)` / `add_mag(...)`
(constant **AB** zero point — pass `zeropoint_jy=lc.zero_point` to opt into per-band; raise for
`data_mode='flux'`); `resolve_bands(svo_fallback=True)` (fill `lambda_eff`/`zero_point`);
`set_explosion_date(mjd)` (observer-frame days since explosion); **`calc_phase(reference=, redshift=,
peak=, hours=)`** (rest-frame phase `(t − ref)/(1+z)`); **`calc_absmag(dm=, redshift=, ebv=, rv=3.1,
extinction=)`** (distance modulus from `z`/`luminosity_distance` + Milky-Way extinction via CCM89 or an
explicit `{band: A_mag}` dict → `absmag` column); `to_dataframe()` (→ `to_pandas()`); `lc()` returns the
table itself.

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

## 5. Plotting

**`plot_light_curve(lc, *, layout="report", quantity="apparent_mag", bands=None, ncols=3, figsize=None, title=None, save=None)`**
— `layout`: `"report"` (mag + flux panels) or `"grid"` (per band). `quantity`: `apparent_mag` /
`absolute_mag` (needs redshift) / `flux`. Markers: detections = circles, SNR<3 = △, upper limits = ▽.

**`plot_corner(posteriors, *, labels=None, parameters=None, colors=None, truths=None, bins=30,
levels=(0.39, 0.86), smooth=1.0, log_params=None, title=None, legend_loc="upper right", save=None, **corner_kwargs)`**
— overlay a **list of posteriors** on one publication-ready corner plot. Each posterior is a
`SamplerResult`, a `DataFrame`, a `{name: array}` dict, or a 2-D array (then pass `parameters`).
Shared per-parameter ranges align the panels; each posterior gets a distinct dark colour
(`CORNER_PALETTE`); 2-D panels are contour **lines** (default `levels` ≈ 1σ/2σ) and the diagonals are
step histograms, so several posteriors stay readable overlaid. `parameters` defaults to the columns
common to all inputs; `log_params` puts those on a `log10` axis; `truths` (dict or list) draws dashed
reference lines; a colour→label legend is added. Returns the `Figure`. Ideal for comparing samplers on
the same data — the posteriors (with uncertainties) show whether methods are *compatible*, which a table
of point estimates cannot.

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

Built-in **toy** models (band-independent, vectorized; `t` = days since explosion; scale `amplitude` to
your flux units for real data):

| Name | Form | Parameters |
|---|---|---|
| `flare` | `A·(1 − e^(−t/t_rise))·e^(−t/t_decay)` | amplitude, rise_time, decay_time |
| `bazin` | `A·e^(−(t−t0)/τ_fall) / (1 + e^(−(t−t0)/τ_rise))` | amplitude, t0, tau_rise, tau_fall |
| `gaussian_rise` | Gaussian rise to peak at `t0`, then exp decay | amplitude, t0, sigma_rise, tau_decay |

Built-in **physical** models:

| Name | Physics | Parameters |
|---|---|---|
| `mck19` | EM flare from a **BBH merger in an AGN disk** — a GW-recoil-kicked remnant shocks a bound-gas **hotspot** that radiates as a blackbody: a `sin²` rise to the ram-pressure delay `t_ram`, then exponential decay back to the disk baseline. McKernan et al. 2019 ([ApJL 884, L50](https://iopscience.iop.org/article/10.3847/2041-8213/ab4886)), implementation of [Darc 2025](https://arxiv.org/abs/2506.02224). | v_kick, M_smbh, M_bh, r_bh, redshift |
| `two_component_kilonova` | **Kilonova** (NS–NS merger) with a blue (low-κ) + red (high-κ) ejecta component, via the optional **redback** backend (`[models]` extra). Band-integrated AB magnitudes (redshift-aware) → flux density (Jy). | mej_1, vej_1, kappa_1, temperature_floor_1, mej_2, vej_2, kappa_2, temperature_floor_2, redshift |

`two_component_kilonova` is the **first redback-backed model** — redback is imported lazily, so WHISPER
and `list_models()` work without it; only `predict` needs the `[models]` extra. It wraps redback's
`two_component_kilonova_model`, mapping WHISPER bands → redback LSST filters and converting redback's AB
magnitude to flux density (a machine-precision round-trip). It is an **expensive** simulator (~50 ms per
band-call), so **SNPE** (amortized) is the natural sampler; ABC/MCMC want modest budgets. See
`scripts/demo_kilonova.py` and `scripts/fit_kilonova_at2017gfo.py`.

`mck19` is **band-dependent** — it returns flux density (Jy) at each `(time, band)`, evaluating the
hotspot + disk blackbody at the band's effective wavelength (via Whisper's band system) and the redshift
(Planck18 luminosity distance + time dilation). `t = 0` is the merger; the flare peaks at the
observer-frame delay `t_ram`. Self-contained (astropy constants/cosmology only — no `speclite`); the AB
magnitude is the monochromatic-at-`λ_eff` approximation to the original LSST filter integration. Fixed
disk constants (`mdot=0.05` Edd, `alpha=0.1`) do not enter the light-curve *shape*. See
`scripts/demo_mck19.py` for the g/r/i light curve.

> For **parallel** ABC (`n_jobs>1`) the `predict` function must be picklable (module-level, not a
> closure/lambda). Closures work with `n_jobs=1`.

### 6.3 Distance  (`whisper_labia.distance`)
`chi2_distance(obs_flux, obs_flux_err, sim_flux, bands=None) -> float` = `sum(((obs-sim)/err)**2)`.
Any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float` can be passed as a custom distance.

### 6.4 Samplers  (`whisper_labia.samplers`)

`fit_ABC(lc, model="flare", ...)`, `fit_ABC_SMC(lc, model="flare", ...)`, `fit_MCMC(lc, model="flare",
...)`, and `fit_SNPE(lc, model="flare", ...)` → `SamplerResult`. `fit(lc, model,
sampler="abc"|"abc_smc"|"mcmc"|"snpe", **kwargs)` is the generic dispatcher. `list_samplers()` →
`['abc', 'abc_smc', 'mcmc', 'npe', 'snpe']` (`npe` is an alias of `snpe`); `register_sampler(name, cls)`
adds your own. **ABC/ABC-SMC** use the χ² distance; **MCMC/SNPE** use the shared likelihood layer, so all
four reach the same posterior on the same data (see `scripts/compare_samplers.py`).

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

**`MCMCSampler.fit(lc, model, prior=None, *, nwalkers=None, nsteps=5000, burnin=1000, thin=10,
initial_guess=None, initial_scatter=1e-3, space="auto", likelihood="auto", seed=0, progress=False,
moves=None)`** — affine-invariant ensemble MCMC via `emcee` (`samplers.mcmc`; emcee is a **core**
dependency). The log-posterior is the Whisper prior + the **shared likelihood layer**
(`make_likelihood(lc, kind=likelihood, space=space)`), so MCMC uses the same physically consistent,
`data_mode`-aware likelihood as the others (flux data → flux space, magnitude data → magnitude space).
`nwalkers` defaults to `max(2·ndim+2, 4·ndim)` (forced even); walkers init from the prior unless
`initial_guess` is given. Sampling is **seeded/reproducible**. `best_params` is the max-posterior draw;
`max_log_likelihood`/`AIC`/`BIC` are exact Gaussian values; `info` carries acceptance fraction +
autocorrelation time; the `emcee.EnsembleSampler` is attached as `result.emcee_sampler`. `fit_MCMC(...)`
is the convenience wrapper.

**`SNPESampler.fit(lc, model, prior=None, *, num_rounds=2, num_simulations=1000, space="auto",
density_estimator="maf", embedding_net=None, hidden_features=None, num_transforms=None, num_bins=None,
proposal_mode="posterior", truncate_quantile=1e-4, support_samples=10000, num_samples=10000,
device="cpu", seed=0, show_progress=False, num_workers=1, max_logl_scan=2000, **train_kwargs)`** —
Sequential Neural Posterior Estimation via `sbi` (`samplers.snpe`). Needs the optional **`[sbi]`** extra
(sbi + torch; imported lazily). The simulator is Whisper's forward model + Gaussian noise from the data
errors; the prior is adapted automatically (`Uniform`→`BoxUniform`, mixed `Uniform`/`LogUniform`→
`MultipleIndependent`). `num_rounds=1` is amortized NPE, `>1` is sequential; `num_simulations` is per
round; `space` ('auto'|'flux'|'magnitude') matches the likelihood; `num_workers` parallelizes simulation.

- **Density estimator (flexible):** `density_estimator` is an estimator name **or** a pre-built
  `posterior_nn(...)` factory; pass an **`embedding_net`** (`torch.nn.Module`; input dim = number of
  light-curve points) and/or `hidden_features` / `num_transforms` / `num_bins` to build a custom
  architecture.
- **Sequential scheme:** `proposal_mode='posterior'` (SNPE-C, default) or `'restricted'` (truncated SNPE
  via `RestrictedPrior` + `get_density_thresholder(quantile=truncate_quantile)`; support estimated from
  `support_samples` draws — kept modest, since sbi's default 1e6 can take hours; rejection sampling makes
  it compute-heavy).

- **Device (GPU):** `device` = `'cpu'` (default), `'cuda'`/`'gpu'`/`'cuda:N'`, or `'auto'` (CUDA when
  available, else CPU). The torch prior + observed data are placed on the device; a GPU request without
  one warns and falls back to CPU. The GPU accelerates *training* (not the CPU simulator), so it helps
  most with many simulations / large nets — see `scripts/benchmark_snpe_device.py` (`docs/figures/snpe_device_benchmark.png`).

Extra kwargs pass to `NPE.train` (e.g. `max_num_epochs`, `training_batch_size`, `stop_after_epochs`).
`max_log_likelihood`/`AIC`/`BIC` are the exact Gaussian values at the best posterior draw. The trained
sbi posterior is attached as `result.posterior` (and `result.posteriors` per round) for resampling /
`sbi.analysis.pairplot`. `fit_SNPE(...)` is the convenience wrapper; `"snpe"` and `"npe"` both dispatch here.

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

Each Gaussian likelihood also exposes **`log_likelihood_pointwise(model_flux) -> array`** (the per-data-
point log-likelihood, summing to `log_likelihood`) — the ingredient WAIC needs.

### 6.6 Metrics  (`whisper_labia.metrics`)

**`waic(posterior, lc, model=None, *, space="auto", likelihood="auto", fixed=None, max_samples=2000, seed=0)`**
— the **Widely Applicable Information Criterion** (Watanabe 2010; Gelman et al. 2014): a fully-Bayesian
fit score that, unlike AIC/BIC (which use one best-fit point), uses the **whole posterior**. It evaluates
the model's *pointwise* log-likelihood across the posterior draws and returns a dict with `waic`
(`= -2(lppd - p_waic)`, **lower is better**), `lppd`, `p_waic` (effective # parameters), `se` (standard
error), `n_samples`, `n_data`. `posterior` may be a `SamplerResult` (uses `.samples`/`.model`), a
`DataFrame`, or an array; `fixed=` supplies values for parameters pinned during the fit (so absent from
the posterior columns); `max_samples` caps the per-draw model evaluations (matters for slow simulators).
Note `p_waic` (and hence WAIC) inflates for posteriors much broader than the likelihood — e.g. ABC's
tolerance posterior or an under-converged SNPE run — which is itself a useful diagnostic; magnitude space
is numerically gentler than flux space (whose tiny errors make the likelihood very sharp).

## Notes & limitations (review findings)

- **Metrics vs likelihood:** ABC/ABC-SMC score with the χ² distance, so `AIC`/`BIC`/
  `max_log_likelihood` are χ²-based — exact for **model comparison on the same data**, but offset by
  the Gaussian normalization constant in absolute terms. The likelihood layer makes them exact.
- **ABC-SMC is unweighted** (uniform parent resampling, no importance weights): best-fit + an
  approximate posterior are reliable; rigorous posterior weights are a planned option.
- **ABC posteriors are approximate** (broadened by the acceptance ε); tighten ε / use SMC for sharper ones.
- **Toy models** are band-independent analytic forms: `flare` = 0 before explosion; `bazin` computed
  stably in log-space; `gaussian_rise` has a derivative kink at the peak. **`mck19`** is a built-in
  *physical*, **band-dependent** model (blackbody hotspot + AGN disk, redshift-aware). Further physical,
  band-dependent models can optionally be supplied by the external redback `[models]` extra (a
  models/priors source only).
- **SNR(magnitude)** uses `(2.5/ln10)/σ_m`, valid for small magnitude errors.

## 7. Internals
`io.loader._resolve_columns`, `schema.LightCurve._subset/_copy`, `plotting._categories/_scatter`,
`samplers.abc._simulate_batch/_worker`, `samplers.base.summarize_posterior`,
`io.units.to_canonical`, `io.svo._svo_fetch_metadata/_svo_fetch_index/_svo_fetch_transmission`
(network boundary), `scripts/{phase0_smoke,demo_abc_at2017gfo,demo_ingestion}.py`.

## 8. Test coverage (183 tests, all passing)

| File | Tests | Focus |
|---|---|---|
| `test_photometry.py` | 5 | AB zeropoint, mag↔flux, error propagation, SNR. |
| `test_bands.py` | 11 | Aliases, case-sensitivity, `group_bands`/`FILTER_LOOKUP`, `unmapped_bands`. |
| `test_schema.py` | 11 | Validation, subsetting, `add_*`, `snr`/`select_snr`, `set_explosion_date`, upper limits. |
| `test_loader.py` | 12 | AT2017GFO load, window/subset, grouping, `min_snr`, `explosion_date`, upper limits. |
| `test_plotting.py` | 7 | report/grid layouts, flux/absolute-mag, redshift guard, upper-limit markers; `plot_corner` overlay/legend, common-params + log axes + array/empty errors. |
| `test_metrics.py` | 4 | WAIC keys + finite + better-fit-lower ordering, `fixed=`/subsampling, and pointwise log-lik (Gaussian + upper-limits) summing to the total. |
| `test_priors.py` | 6 | Uniform/LogUniform/Prior sampling, log_prob, rescale, picklability. |
| `test_models.py` | 8 | flare/bazin/gaussian_rise (incl. flare pre-explosion=0, bazin tail→0), custom model, errors. |
| `test_distance.py` | 2 | chi² zero/known value. |
| `test_abc.py` | 6 | parameter recovery, serial+parallel, acceptance count, JSON, dispatch, custom model. |
| `test_abc_smc.py` | 6 | SMC registration, recovery, epsilon tightening, explicit schedule, dispatch, model-agnostic. |
| `test_likelihood.py` | 9 | Gaussian flux/mag, space-auto, upper-limit CDF, mixture, make_likelihood, picklable. |
| `test_units.py` | 12 | F_ν/F_λ→Jy, per-point λ, NaN-λ error, mag rejects flux unit, no-unit default, flux dimensionality. |
| `test_svo.py` | 15 | FILTER_LOOKUP→SVO, mocked metadata/index, cache hit + disk cache, corrupt-cache (×4 params), graceful degrade, manual override (no spurious warn), transmission, ambiguity. |
| `test_ingestion.py` | 25 | data_mode/output_format, `__call__`, redshift (arg/column/unknown/0/neg/NaN/all-NaN), units, band-info, subset preservation, backward-compat ZP, flux-mode. |
| `test_snpe.py` | 8 | registry/alias + proposal-mode validation (no sbi), torch-prior adapter, density-estimator dispatch; SNPE fit end-to-end / multi-round / embedding-net + custom estimator (`slow`; need `[sbi]`). |
| `test_table_lc.py` | 7 | LightCurve as astropy.Table: column ops + native methods, property setters, slicing keeps subclass/meta, `where()`, rest-frame `calc_phase`, `calc_absmag` (distance modulus + CCM89/explicit extinction). |
| `test_registries.py` | 5 | likelihood + distance registries (`register_*`/`list_*`/`get_distance`, distance-by-name in `fit`), manual-band register/unregister/clear. |
| `test_mcmc.py` | 4 | MCMC recovery + reproducibility, data-mode-consistent likelihood space, and ABC↔MCMC posterior agreement. |
| `test_mck19.py` | 7 | `mck19` AGN-disk BBH flare: registration, finite/positive flux, band-dependence, peak at `t_ram`, redshift dimming, no-bands warning, and end-to-end MCMC recovery in magnitude space (`slow`). |
| `test_two_component_kilonova.py` | 8 | redback kilonova: registry/prior + band-mapping + no-bands error (no redback needed); flux finite/positive, band-dependence, machine-precision redback round-trip, mixed-band predict, and SNPE recovery (`slow`) — all guarded by `importorskip("redback")`. |
| `test_benchmark_kilonova.py` | 3 | flux-vs-magnitude benchmark: `setup` (data + prior) and the magnitude-space χ² distance (no redback), plus a `slow` end-to-end fit→publication-report render. |

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
