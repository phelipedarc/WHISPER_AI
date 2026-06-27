# Whisper — design rationale & architecture

This document explains the **why** behind Whisper's structure so a contributor can understand, trust,
extend, and maintain it without reverse-engineering the source. For the *how-to*, see
[`TUTORIAL.md`](TUTORIAL.md), [`API_REFERENCE.md`](API_REFERENCE.md), and [`EXTENDING.md`](EXTENDING.md).

## 1. Goals & philosophy

Whisper answers one question — *"which model best fits this transient light curve, and what are the
posteriors?"* — and is built for use in real transient-characterization pipelines. Design priorities, in
order: **clarity > cleverness; explicit > hidden magic; reproducibility > convenience; extensibility >
hardcoding; user experience > developer convenience; long-term maintainability > short-term speed.**

## 2. Architecture at a glance

```
load_lightcurve ──► LightCurve (astropy.Table)
                         │
        ┌────────────────┼─────────────────────────────┐
     models           samplers          likelihoods / distances
   (register_model) (register_sampler)  (register_likelihood / register_distance)
        │                │                         │
        └──────► fit(lc, model, sampler) ──► SamplerResult (posterior, AIC/BIC, JSON)
```

**Four pluggable axes**, each a small name registry with `register_*` / `list_*` / `get_*` helpers:
**models**, **samplers**, **likelihoods**, **distances**. "Easy to add a new one" is a core goal —
adding any axis member is ~1 function and needs no edits elsewhere.

## 3. Key decisions (and why)

- **`LightCurve` *is* an `astropy.table.Table`.** Astronomy-native (units, masked columns, `group_by`,
  ascii/FITS I/O), and lets users compute on columns directly (`lc['x'] = lc['flux'] + 5`). The common
  quantities are mirrored as attributes (`lc.flux`, `lc.time`, `lc.redshift`) so consumers and
  beginners keep a simple interface. Scalar metadata lives in `.meta`, which astropy propagates through
  slicing — so subsetting preserves redshift/data_mode for free.
- **redback is optional and auxiliary.** Everything (ingestion, samplers, likelihoods, plots) is
  Whisper's own and runs standalone. redback (the `[models]` extra) is used *only* as a source of
  physical models + priors — never as a runtime dependency of the core.
- **Flux is the canonical model output; likelihoods choose the space.** Models predict flux density;
  the likelihood compares in `flux` or `magnitude` space. This keeps models space-agnostic and makes
  the comparison space a user choice (`space='auto'|'flux'|'magnitude'`).
- **χ² distance ≙ −2 ln L (Gaussian),** so ABC can report `max_log_likelihood`, `AIC`, `BIC` for model
  comparison without an explicit likelihood evaluation.
- **Reproducibility is seed-based and `n_jobs`-independent.** Each simulation/attempt owns an RNG stream
  derived from its *global index* (not the worker), so a fixed `seed` gives identical posteriors
  regardless of core count or `n_jobs`. SNPE seeds the simulator's noise per parameter row for the same
  reason. Parallelism affects speed only, never the science.
- **Picklability constraint.** For parallel ABC, model `predict`, priors, and distances must be
  picklable (module-level, not closures) — this shapes those APIs (small picklable classes/functions).
- **SVO band fallback behind one mockable boundary.** All network access goes through three private
  `_svo_fetch_*` functions, cached by filter ID (offline-safe). Tests mock that boundary, so CI never
  hits the network and the package imports without `astroquery`.

## 4. Known limitations

- **ABC posteriors are approximate** (broadened by the acceptance ε); tighten ε or use ABC-SMC.
- **ABC-SMC is unweighted** (uniform parent resampling, no importance weights): best-fit + an
  approximate posterior are reliable; rigorously weighted posteriors are planned.
- **Likelihoods are not yet wired into the samplers' acceptance.** ABC/ABC-SMC score with the χ²
  distance (flux space, requires errors); SNPE uses a Gaussian noise model. Selecting
  `GaussianLikelihoodWithUpperLimits` / `MixtureGaussianLikelihood` per fit is a planned `likelihood=`
  parameter (see §5).
- **Metrics are χ²-based for ABC:** `AIC`/`BIC`/`max_log_likelihood` are exact for *model comparison on
  the same data* but offset by the Gaussian normalization constant in absolute terms.
- **Built-in models are analytic and band-independent** (`flare`, `bazin`, `gaussian_rise`); physical,
  band-dependent models come from redback via the `[models]` extra.
- **`calc_absmag` extinction** uses the CCM89 law and applies no Vega→AB offsets for NIR bands; host
  extinction beyond an explicit `extinction` dict is not modeled.
- **SVO mapping** uses a ±5% wavelength window and documented default filter IDs; ambiguous matches warn
  and pick the closest, which may not be the intended filter — override with `register_manual_band`.

## 5. Planned extensions

- `likelihood=` / `space=` on every sampler (route `make_likelihood` into acceptance; exact −2 ln L).
- Unified sampler keyword names across ABC/ABC-SMC/SNPE (with deprecation aliases).
- Likelihood-based samplers: MCMC (emcee) and nested sampling (Dynesty).
- Full type-hint coverage across all internal modules (the public surface is annotated and `py.typed`).
- Importance-weighted ABC-SMC; redback model adapters; a `fit_all` grid (transient × model × sampler).
