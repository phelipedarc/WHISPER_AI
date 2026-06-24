# Whisper (`whisper_labia`)

**Easy Bayesian model comparison of astronomical transient light curves.**

Give Whisper a light-curve CSV; it will tell you **which transient model best fits** and return the
**posteriors + model-selection metrics** (AIC, BIC, evidence, max-likelihood) for each model. It uses
[redback](https://github.com/nikhil-sarin/redback) for the physical *models* and *priors*; the data
handling, samplers, plots and outputs are Whisper's own — built to be **simple and extensible**.

> **Status:** early development. **Phase 1 — data ingestion + plotting — is done and tested
> (44 unit tests).** The samplers (MCMC, Dynesty, …) come next.

## Install (Docker)

Everything runs inside the `phe_sbi` container — nothing is installed on the host.

```bash
docker exec phe_sbi bash -lc 'cd /tf/astrodados2/phelipedata2/WHISPER/whisper-labia && pip install -e .'
```

## 30-second quickstart

```python
import whisper_labia as wp

lc = wp.load_lightcurve("at2017gfo.csv")     # flexible CSV  ->  LightCurve
lc = lc.select_snr(min_snr=5)                # keep good detections (SNR >= 5)
wp.plot_light_curve(lc, layout="report")     # apparent-mag + flux overview, all bands
```

![report plot](docs/figures/at2017gfo_report.png)

## What you can do today

- **Load** messy CSVs — auto-detects columns (time/mag/flux/err/band/system), comma or semicolon.
- **Group bands** — collapse heterogeneous filters (`B→g`, `V→r`, `Ks→K`, HST/JWST…) via `FILTER_LOOKUP`.
- **Quality-control** — drop bad rows, cut by **SNR** (`min_snr=3` or `5`), pick time windows / bands.
- **Convert & derive** — magnitude ↔ flux, per-point **SNR**, set the **explosion date** (day 0).
- **Plot** — a report (mag + flux) or a per-band grid (apparent / absolute mag, or flux), with clear
  marker conventions (detections = circles, SNR<3 = △, upper limits = ▽).

## Learn more

- 📘 **[Tutorial](docs/TUTORIAL.md)** — a hands-on tour of every feature, with plots.
- 📑 **[API reference](docs/API_REFERENCE.md)** — every function and its arguments.

## License

GPLv3 (inherited from the redback dependency).
