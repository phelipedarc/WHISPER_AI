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
lc.data_mode             # 'magnitude' or 'flux_density'
lc.snr                   # per-point signal-to-noise (computed from the errors)
lc.to_dataframe().head() # a pandas view (includes an 'snr' column)
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
lc.add_flux()    # AB magnitudes -> flux density (Jy), with error propagation
flux_lc.add_mag()  # flux -> AB magnitudes
```

## 5. Plot

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

## What's next

Phase 2 wires up the **models** (redback's library + your own via `register_model`) and the
**forward model** that turns parameters into predicted observables; then the **samplers** (MCMC,
Dynesty) and the per-transient model-comparison JSON.
