# Whisper Tutorial — working with transient light curves

A hands-on tour of what Whisper can do today (**Phase 1: data ingestion + plotting**). Everything
runs inside the `phe_sbi` container. Import once:

```python
import whisper_labia as wp
```

---

## 1. Load a light curve

`load_lightcurve` reads almost any photometry CSV and works out the columns for you.

```python
lc = wp.load_lightcurve("tests/data/at2017gfo.csv")
print(lc)
# LightCurve(name='at2017gfo', n_points=645, bands=[...29 labels...], mode='magnitude')
```

It auto-detects `time/MJD`, `magnitude`/`flux`, their errors, `band` and `system` columns (override
with `column_map={'time': 'MJD', ...}`), sniffs comma- vs semicolon-separated files, and drops
obviously bad rows (non-finite values, non-positive errors).

## 2. Inspect it

```python
lc.n_points              # 645
lc.bands                 # sorted unique band labels
lc.data_mode             # 'flux_density' | 'magnitude' | 'flux'  (stored attribute)
lc.output_format         # forward-model comparison space: 'magnitude' | 'flux_density'
lc.redshift_known        # True/False — False means a redshift prior must be sampled
lc.snr                   # per-point signal-to-noise (computed from the errors)
lc()                     # the ENRICHED dataframe (both flux & magnitude); == lc.to_dataframe()
```

## 3. Clean & shape — chainable, each call returns a **new** `LightCurve`

```python
lc = (wp.load_lightcurve("tests/data/at2017gfo.csv", band_lookup=True)  # group bands
        .select_snr(min_snr=5)               # keep SNR >= 5
        .select_time_window(time_max=57990)  # MJD window
        .set_explosion_date(57982.0))        # time -> days since explosion (day 0)
```

…or do it all at load time:

```python
lc = wp.load_lightcurve("tests/data/at2017gfo.csv",
                        band_lookup=True, min_snr=5, time_max=57990, explosion_date=57982.0)
```

### Band grouping
Surveys label filters inconsistently. `band_lookup=True` collapses them into an effective ladder
(`U/g/r/i/z/J/H/K-band` + JWST) using `wp.FILTER_LOOKUP` — e.g. `B→g-band`, `V→r-band`, `Ks→K-band`,
`F606W→r-band`. Clear/white-light bands (`C`, `W`, `w`) are kept as-is. AT2017GFO's 29 raw labels
collapse to 11 effective bands this way.

### Signal-to-noise cut
```python
wp.load_lightcurve("tests/data/at2017gfo.csv").n_points              # 645
wp.load_lightcurve("tests/data/at2017gfo.csv", min_snr=3).n_points   # 632
wp.load_lightcurve("tests/data/at2017gfo.csv", min_snr=5).n_points   # 578
```

## 4. Convert magnitude ↔ flux

```python
lc.add_flux()      # AB magnitudes -> flux density (Jy), with error propagation
flux_lc.add_mag()  # flux -> AB magnitudes
```

`add_flux`/`add_mag` use the constant **AB 3631 Jy** zero point (so the modelling flux the samplers see
stays on one zero point). For the **per-band** physical conversion (LSST/SVO zero points) just call the
curve — `lc()` returns a dataframe with both columns filled in (see §5).

## 5. Data mode, redshift, units & band resolution

These four are wired into `load_lightcurve` and the `LightCurve` itself.

### Data mode and the enriched dataframe
`data_mode` is `flux_density` (canonical unit Jy, default), `magnitude` (dimensionless AB), or `flux`
(band-integrated erg/s/cm²). It is inferred from the columns but you can set it explicitly. **Calling**
the curve returns the enriched dataframe — the missing one of flux/magnitude is derived from each band's
zero point:

```python
df = lc()                # both 'flux' and 'magnitude' columns (+ lambda_eff, zero_point, snr)
lc.output_format         # 'flux_density' / 'magnitude' — the forward-model comparison space
```

### Redshift — argument > column > unknown (never silently assumed)
```python
wp.load_lightcurve("sn.csv", redshift=0.034)        # explicit
wp.load_lightcurve("sn.csv")                        # a 'redshift' column is picked up automatically
```
If neither is present the load **does not fail** — the curve is flagged `redshift_known=False`, carries a
default `redshift_prior` you can override, and warns that *z will be sampled, not assumed*. Validation:
`z ≥ 0`; `z == 0` needs an explicit `luminosity_distance=` (Mpc); negative/NaN is a hard error.

```python
lc.redshift_known        # False
lc.redshift_prior        # {'type': 'Uniform', 'low': 0.001, 'high': 1.0, 'name': 'redshift'}
wp.load_lightcurve("z0.csv", redshift=0.0, luminosity_distance=40.0)   # z=0 case
```

### Units (astropy) — F_ν or F_λ in, canonical Jy out
Flux density may arrive as **F_ν** (Jy/mJy/µJy) or **F_λ** (erg/s/cm²/Å). Pass the unit and Whisper
stores Jy internally; the F_λ→F_ν conversion uses each band's effective wavelength
(`u.spectral_density`):

```python
wp.load_lightcurve("fnu.csv",  flux_unit="mJy")                  # F_nu
wp.load_lightcurve("flam.csv", flux_unit="erg/(s cm2 AA)")       # F_lambda -> Jy via band wavelength
```
Magnitudes must be dimensionless AB (a flux unit on a magnitude column is a clear error). A flux column
with **no** unit warns and assumes the documented default (Jy) — pass `flux_unit=` to silence it.

### Bands — FILTER_LOOKUP, then SVO fallback
Each band resolves to an effective wavelength + zero point. Known filters come from `FILTER_LOOKUP`
(optical bands anchored to LSST ugrizy); an unknown band warns and falls back to the **SVO Filter
Profile Service**:

```python
wp.resolve_band("g")                 # {'source':'lsst', 'lambda_eff':4866.0, 'zero_point':3631.0, ...}
wp.resolve_band("PAN-STARRS/PS1.w")  # warns, then queries SVO (cached by filter ID; offline-safe)
```
SVO results are cached locally, so re-runs never re-query. If SVO is unavailable (no network /
astroquery), the load degrades gracefully and you can supply the band by hand:

```python
wp.register_manual_band("my_filter", lambda_eff=9000.0, zero_point=3631.0)
```
> SVO needs the optional `[svo]` extra (`pip install 'whisper-labia[svo]'`, which adds `astroquery`).
> A runnable, **offline** tour of all four features is in [`scripts/demo_ingestion.py`](../scripts/demo_ingestion.py).

## 6. Plot

### Report (overview): apparent magnitude **and** flux, all bands overlaid
```python
wp.plot_light_curve(lc, layout="report")
```
![report](figures/at2017gfo_report.png)

### Per-band grid: one box per band, choose the quantity
```python
wp.plot_light_curve(lc, layout="grid", quantity="apparent_mag", ncols=4)
```
![grid](figures/at2017gfo_grid_mag.png)

`quantity` can be `"apparent_mag"`, `"flux"`, or `"absolute_mag"` (the latter needs `redshift=` set on
the curve, e.g. `load_lightcurve(..., redshift=0.0099)`).

**Marker conventions:** each band gets a distinct color; **detections** are circles with a black edge;
**SNR < 3** points are up-triangles (△); **upper limits** are down-triangles (▽); magnitude axes are
inverted (brighter = up).

### Any survey format works
The same one-liner on raw ZTF photometry (`zg`/`zr` → `ztfg`/`ztfr`, with the `catflags` quality flag):
```python
ztf = wp.load_lightcurve("tests/data/ztf18aarlhfw.csv", flag_filters={"catflags": 0})
wp.plot_light_curve(ztf, layout="report")
```
![ztf](figures/ztf18aarlhfw_report.png)

---

## 7. Fit a model with ABC

Whisper has two pluggable axes — **models** and **samplers**:

```python
wp.list_models()     # ['flare']  (+ your own via register_model)
wp.list_samplers()   # ['abc']    (+ your own via register_sampler)
```

The built-in **flare** model is `flux = A·(1 − e^(−t/t_rise))·e^(−t/t_decay)`. Fit it to AT2017GFO's
r-band with **Approximate Bayesian Computation** (parallel rejection sampling):

```python
lc = wp.load_lightcurve("at2017gfo.csv", explosion_date=57982.0, min_snr=3)
r  = lc.select_bands("r")

fmax  = r.add_flux().flux.max()                       # scale the amplitude prior to the data
prior = wp.Prior({"amplitude":  wp.Uniform(0, 10*fmax),
                  "rise_time":  wp.Uniform(0.05, 10),
                  "decay_time": wp.Uniform(0.5, 40)})

res = wp.fit_ABC(r, "flare", prior=prior, n_simulations=200_000, quantile=0.005, n_jobs=16)
print(res)                 # SamplerResult(sampler='abc', model='flare', n_samples=1000, AIC=..., runtime=1.2s)
res.summary["amplitude"]   # {'median':..., 'ci16':..., 'ci84':...}
res.best_params            # best-fit parameter dict
res.to_json("fit.json")    # AIC, BIC, max-likelihood, posterior summary, diagnostics
```

![ABC flare fit](figures/at2017gfo_abc_flare_r.png)

The flare model tracks the r-band decline well. (Its reduced χ² is high only because the photometry
is high-SNR — tiny error bars magnify any model imperfection; physically-motivated models come later.)

- **Acceptance** is by `quantile` (keep the best fraction — robust default) or a fixed `threshold`.
- **Metrics:** the χ² distance equals −2 ln L for a Gaussian likelihood, so ABC reports
  `max_log_likelihood`, `AIC` and `BIC` for model comparison.

### It runs in parallel
Simulations are split across processes (`n_jobs`). On this machine (200k simulations):

| n_jobs | time | sims/s |
|---|---|---|
| 1 | 7.25 s | 27,600 |
| 8 | 1.90 s | 105,000 |
| 32 | 1.73 s | 115,000 |

(~4× here; for expensive physical models the speedup scales further.)

### Bring your own model / distance
```python
import numpy as np
def my_model(params, times, bands=None):
    return params["a"] * np.exp(-times / params["tau"])

wp.register_model("expdecay", my_model, ["a", "tau"],
                  prior=wp.Prior({"a": wp.Uniform(0, 1), "tau": wp.Uniform(1, 50)}))
res = wp.fit_ABC(lc, "expdecay", n_jobs=1)   # n_jobs=1 for closures; module-level fns run in parallel
```
A custom distance is any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float` passed as `distance=`.

### ABC-SMC and more models

There's also **ABC-SMC** (sequential rejection over rounds of shrinking threshold) and two more
built-in models — **`bazin`** (SN rise+fall) and **`gaussian_rise`** (Gaussian rise + exp decay):

```python
wp.fit_ABC_SMC(r, "bazin", prior=prior, n_particles=1000, n_rounds=8, quantile=0.4, n_jobs=32)
```

A full **model-comparison report** (3 models × both samplers on AT2017GFO) is in
[`REPORT_at2017gfo.md`](REPORT_at2017gfo.md) — there ABC-SMC matches flat ABC's fit with **~4× fewer
simulations**.

![model comparison](figures/at2017gfo_model_comparison.png)

### Likelihoods & space (flux vs magnitude)

All inference can run in **flux** or **apparent-magnitude** space, with Gaussian, upper-limit, and
mixture (outlier-robust) likelihoods (`whisper_labia.likelihood`). The default is chosen by data type;
override for edge cases (non-detections, outliers):

```python
from whisper_labia.likelihood import make_likelihood, GaussianLikelihoodWithUpperLimits
lik = make_likelihood(lc, space="magnitude")                 # default by data type; override space/kind
lik = GaussianLikelihoodWithUpperLimits(lc, space="flux")    # use non-detections in flux space
```

(Implemented and tested; wiring them into the samplers — `fit_ABC(..., space=..., likelihood=...)` —
is the next step.)

## What's next

Next: wire likelihoods into the samplers (flux/magnitude space + upper limits in inference), then a
**likelihood-based sampler** (MCMC / Dynesty). Physical models + priors can optionally be plugged in
from the external redback package (the `[models]` extra) — an auxiliary source of models and priors only.
