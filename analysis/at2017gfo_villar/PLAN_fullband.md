# Plan — full UV–optical–NIR AT2017GFO analysis

## Motivation
The current AT2017GFO fit uses **g/r/i only** (`tests/data/at2017gfo.csv`). Those optical bands are
dominated by the **blue** (lanthanide-poor) kilonova component; the **red** (lanthanide-rich) component
radiates mostly in the **NIR**, which g/r/i barely sees. That is exactly why our fits leave the red
parameters (κ_red, v_ej^red, T_floor^red) poorly constrained and railing against prior edges, while the
blue parameters recover cleanly near Villar+2017. Villar+2017 avoided this by fitting the full
**UV → optical → NIR (UVOIR)** light curve. This plan pulls that full dataset so both components are
constrained.

## Groundwork already done (verified, in the repo)
- **Fetcher** `fetch_at2017gfo_full.py` → `data/at2017gfo_full.csv`. Pulls the complete
  photometry from the Open Astronomy Catalog (`api.astrocats.space`, fetched with `verify=False` — the
  container lacks the CA cert; data is public/read-only), converts the ~18 Vega points to AB, keeps
  post-merger detections with errors, and writes WHISPER's `event,time,magnitude,e_magnitude,band,system`
  format.
- **Result: 607 detections, 0.5–25 d, 18 bands** spanning the full UVOIR range:
  - **UV:** `U`, `u`, `uvot::uvw1` (Swift UVOT), `uvot::white`
  - **Optical:** `B`, `g`, `V`, `r`, `R`, `i`, `I`, `z`, `y`
  - **NIR:** `Y`, `J`, `H`, `K`, `Ks`
- **redback supports every band natively** (`redback/tables/filters.csv`, 264 filters): `2massj/h/ks`,
  bare `H/J/K/Ks/Y`, `uvot::uvw1/uvw2/uvm2/white`, `UVW1/2`, `bessell*`, `sdss/lsst/ps1` grizy. So the
  observed band names map to redback filters with **no translation** for most, and `similar` filters for
  the rest (e.g. Swift `W`→`uvot::white`).
- **WHISPER ingests it:** `load_lightcurve("at2017gfo_full.csv", ...)` loads all 598 points / 18 bands.

## The one code gap (small, well-scoped)
`whisper_labia/models/two_component_kilonova.py::_redback_band` currently maps **only the 6 LSST optical
bands** and rejects NIR/UV names. It must **pass through any band redback's filter table knows**:

1. Add a lazy `_redback_known_bands()` that reads the `bands` column of `redback/tables/filters.csv`
   into a set (cached).
2. In `_redback_band`: keep the bare-`grizy`→LSST mapping (consistency with the existing analysis),
   then **pass through** any `band` (or a normalized form) that is in the redback set — so `H`, `J`,
   `Ks`, `uvot::uvw1`, `uvot::white`, `Y`, `U`, `B`, `V`, `R`, `I` all resolve directly.
3. Test: `test_two_component_kilonova.py` gains a case asserting NIR/UV bands map + predict is finite.

## Analysis
Parametrize `villar.py` to take a **data file + band list** (default stays g/r/i for
reproducibility; `--full` uses `at2017gfo_full.csv` with all 18 bands). Everything else is unchanged:
the Villar+17 two-component model (κ_blue = 0.5 fixed, z fixed), the physical velocity prior (0.05–0.3 c),
the free scatter σ, magnitude space, and the same **7 samplers** (MCMC, ABC, ABC-SMC, NPE-MDN, NPE-NSF,
SNPE-5r-NSF, SNPE-5r-NSF+TCN) with the same figures + report.

### Cost — the one thing to budget for
redback is called **once per unique band** per model evaluation, so going 3→18 bands makes each
`predict` ~5–6× slower (~0.07 s → ~0.4 s). Mitigations, in order of preference:
1. **Trim to ~10–12 well-sampled bands** spanning UVOIR (e.g. `U, B, g, V, r, i, z, Y, J, H, Ks`) — full
   wavelength leverage at ~half the cost. (`fetch_at2017gfo_full.py` already takes a band list arg.)
2. Scale down sim budgets: ABC 60k→30k, SNPE 25k→15k, MCMC keep 12k steps (n_jobs=48). Expected
   wall-clock: ABC ~5 min, MCMC ~30–45 min, each neural fit ~15–25 min (4 in parallel on 4 GPUs).
3. A torch/JAX Bazin-style `predict_torch` is **not** available for redback (semi-analytic, CPU-only),
   so neural simulation stays CPU-parallel (`num_workers`).

## Expected outcome
With NIR coverage the **red component becomes identifiable**: κ_red, v_ej^red and T_floor^red should
pull away from the prior edges toward physical, Villar+17-consistent values (κ_red ≈ 3–4 cm²/g,
v_red ≈ 0.15 c, and a *constrained* red temperature floor rather than the g/r/i-only spread to the
prior floor). The MCMC-vs-SBI tension should shrink further, and the χ²/dof (with σ) should stay ≈1
with the full data. This turns the analysis from a "blue-only, red-unconstrained" demo into a genuine
UVOIR two-component kilonova fit comparable to the literature.

## Caveats (state in the report)
- **Filter approximations:** heterogeneous AT2017GFO photometry (many telescopes) is fit with redback's
  nearest standard bandpasses; small color offsets are absorbed by the fitted scatter σ.
- **Upper limits** are dropped (detections-only); a future step could use
  `GaussianLikelihoodWithUpperLimits`.
- **Vega→AB** applied to ~18 points via standard offsets; 97% of the data is already AB.
- **HST 1–2-point filters** (F336W, F475W, …) are dropped by default (sparse, instrument-specific).

## Checklist
- [x] Fetcher + full-band CSV (`fetch_at2017gfo_full.py`, `data/at2017gfo_full.csv`)
- [x] Verify redback filter support + WHISPER ingestion
- [x] Extend `_redback_band` for NIR/UV pass-through via sncosmo names (+ test); **end-to-end predict
      over all 18 bands verified finite** (lowercase grizy→LSST, uppercase UBVRI→Bessell, JHKKs→2MASS,
      `uvot::*`→Swift UVOT)
- [ ] `--full` / band-list option in `villar.py`
- [ ] Run 7 methods on the full-band data; render figures + report
- [ ] Report: red-component now constrained; compare g/r/i vs full-UVOIR side by side
