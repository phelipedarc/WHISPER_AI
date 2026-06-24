# Whisper (`whisper_labia`) — API Reference

Generated for **v0.0.1.dev0**. Covers everything implemented through **Phase 1 (data ingestion +
plotting)**. Signatures are taken directly from the installed package.

- **Environment:** Docker container `phe_sbi`, Python 3.11.
- **Run tests:** `docker exec phe_sbi bash -lc 'cd /tf/astrodados2/phelipedata2/WHISPER/whisper-labia && python -m pytest tests -q'` (44 tests).

## Package map

```
whisper_labia/
  __init__.py          # public API exports
  plotting.py          # plot_light_curve
  io/
    schema.py          # LightCurve
    loader.py          # load_lightcurve + CANONICAL_SYNONYMS
    bands.py           # normalize_band(s), group_bands, unmapped_bands, DEFAULT_BAND_ALIASES, FILTER_LOOKUP
    photometry.py      # mag_to_flux_density, flux_density_to_mag, mag_err_to_snr, AB_ZEROPOINT_JY, POGSON
```

Top-level (`import whisper_labia as wp`): `__version__`, `LightCurve`, `load_lightcurve`,
`plot_light_curve`, `group_bands`, `FILTER_LOOKUP`.

Naming convention: subsetting → `select_*`, derived columns → `add_*`.

---

## 1. `LightCurve` — canonical light-curve container
`whisper_labia.io.schema.LightCurve` (dataclass)

### Constructor fields
| Field | Type | Default | Description |
|---|---|---|---|
| `time` | `np.ndarray` | — (required) | Epochs (MJD). Coerced to `float`. |
| `band` | `np.ndarray` | — (required) | Band labels. Coerced to `str`. |
| `magnitude` | `np.ndarray \| None` | `None` | Apparent magnitudes. |
| `magnitude_err` | `np.ndarray \| None` | `None` | Magnitude 1σ errors. |
| `flux` | `np.ndarray \| None` | `None` | Flux density (Jy). |
| `flux_err` | `np.ndarray \| None` | `None` | Flux-density 1σ errors. |
| `upper_limit` | `np.ndarray \| None` | `None` | Boolean non-detection flag per point. |
| `system` | `np.ndarray \| None` | `None` | Magnitude system per point (`'AB'`/`'Vega'`/`'unknown'`). |
| `name` | `str \| None` | `None` | Object/event name. |
| `redshift` | `float \| None` | `None` | Redshift (needed for absolute magnitudes). |
| `meta` | `dict` | `{}` | Free-form metadata (`source_file`, `n_rows_raw`, `n_rows_kept`, `explosion_mjd`). |

Requires `band` and at least one of `magnitude`/`flux`; all per-point arrays must match `time` in
length, else `ValueError`.

### Properties
| Property | Returns | Description |
|---|---|---|
| `n_points` / `len(lc)` | `int` | Number of epochs. |
| `bands` | `list[str]` | Sorted unique band labels. |
| `data_mode` | `str` | `'magnitude'` or `'flux_density'`. |
| `snr` | `np.ndarray` | Per-point SNR: `flux/flux_err`, else `(2.5/ln10)/magnitude_err`. Raises if no errors. |

### Methods (each returns a **new** `LightCurve`)
| Method | Arguments | Description |
|---|---|---|
| `select_bands(bands)` | `str` or iterable | Keep only the given band(s). |
| `select_time_window(time_min=None, time_max=None)` | `float \| None` (MJD) | Keep `time_min ≤ t ≤ time_max`. |
| `select_snr(min_snr=5.0)` | `float` | Keep points with `snr ≥ min_snr` (e.g. 3 or 5). |
| `add_flux(zeropoint_jy=3631.0)` | `float` | Copy with `flux`(+`flux_err`) from magnitude (AB). |
| `add_mag(zeropoint_jy=3631.0)` | `float` | Copy with `magnitude`(+`magnitude_err`) from flux (AB). |
| `set_explosion_date(explosion_date)` | `float` (MJD) | Copy with time = days since explosion (day 0); records `meta['explosion_mjd']`. |
| `to_dataframe()` | — | `pandas.DataFrame` (includes `snr` when errors available). |

---

## 2. `load_lightcurve` — flexible CSV loader
`whisper_labia.load_lightcurve(path, *, ...)` → `LightCurve`

```python
load_lightcurve(path, *, name=None, redshift=None, column_map=None, band_aliases=None,
                band_lookup=None, normalize=True, default_band=None, quality_cuts=True,
                drop_nonfinite=True, flag_filters=None, time_min=None, time_max=None,
                bands=None, min_snr=None, explosion_date=None, delimiter=None)
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | CSV file path. |
| `name` | `str \| None` | `None` | Override object name (else from a name/event column). |
| `redshift` | `float \| None` | `None` | Stored on the `LightCurve`. |
| `column_map` | `dict \| None` | `None` | Explicit canonical→actual overrides, e.g. `{'time': 'MJD'}`. |
| `band_aliases` | `dict \| None` | `None` | Extra exact-match aliases for normalization. |
| `band_lookup` | `dict \| bool \| None` | `None` | Broadband grouping: `True` → `FILTER_LOOKUP`; a `dict` → use it. |
| `normalize` | `bool` | `True` | Light band normalization (`DEFAULT_BAND_ALIASES` + `band_aliases`). |
| `default_band` | `str \| None` | `None` | Band to assign when there is no band column. |
| `quality_cuts` | `bool` | `True` | Drop rows with non-finite/≤0 error (**upper limits are kept**). |
| `drop_nonfinite` | `bool` | `True` | Drop rows with non-finite time or primary measurement. |
| `flag_filters` | `dict \| None` | `None` | `{column: value \| callable}` keep-row filters, e.g. `{'catflags': 0}`. |
| `time_min`, `time_max` | `float \| None` | `None` | MJD window. |
| `bands` | `list \| None` | `None` | Keep only these (post-grouping) band labels. |
| `min_snr` | `float \| None` | `None` | Drop points below this SNR (e.g. 3 or 5). |
| `explosion_date` | `float \| None` | `None` | Express time as days since this MJD (day 0). |
| `delimiter` | `str \| None` | `None` | Force a delimiter; default auto-sniffs. |

**Returns** `LightCurve`. **Raises** `ValueError` if no time / band (and no `default_band`) /
magnitude-or-flux column is found.

Also auto-detects an `upper_limit` column (synonyms `upper_limit, upperlimit, islimit, nondetection,
ul`). **`CANONICAL_SYNONYMS`** lists the header synonyms for every canonical field.

---

## 3. Band utilities — `whisper_labia.io.bands`

| Function | Signature | Description |
|---|---|---|
| `normalize_band` | `(band, aliases=None, warn_unknown=False, known=None)` | Trim + exact-match alias (case-sensitive: `r`≠`R`). |
| `normalize_bands` | `(bands, aliases=None, warn_unknown=False, known=None)` | Vectorized `normalize_band`. |
| `group_bands` | `(bands, lookup=None, default=None, warn_unknown=False)` | Collapse to broadbands via `lookup` (default `FILTER_LOOKUP`). |
| `unmapped_bands` | `(bands, lookup=None)` | Sorted labels not covered by `lookup`. |

- `DEFAULT_BAND_ALIASES` — `zg→ztfg, zr→ztfr, zi→ztfi` (+ `ztf_*`).
- `FILTER_LOOKUP` — **88 labels → 10 effective bands** (`U/g/r/i/z/J/H/K-band` + JWST
  `F356W-band`, `F444W-band`); e.g. `B→g-band`, `V→r-band`, `Ks→K-band`, `F606W→r-band`. Clear bands
  `C/W/w` are left out (passthrough).

---

## 4. Photometry — `whisper_labia.io.photometry`

AB system; flux density in **Jy**, `AB_ZEROPOINT_JY = 3631.0`, `POGSON = 2.5/ln10 ≈ 1.0857`.

| Function | Signature | Returns |
|---|---|---|
| `mag_to_flux_density` | `(magnitude, magnitude_err=None, zeropoint_jy=3631.0)` | `flux` or `(flux, flux_err)` |
| `flux_density_to_mag` | `(flux, flux_err=None, zeropoint_jy=3631.0)` | `mag` or `(mag, mag_err)` |
| `mag_err_to_snr` | `(magnitude_err)` | `SNR = POGSON / magnitude_err` |

`f = ZP·10^(−0.4 m)`, `σ_f = 0.4·ln10·f·σ_m`; `SNR = f/σ_f = (2.5/ln10)/σ_m`.
(redback's `flux_density` is in mJy; reconciled in the Phase-2 forward model.)

---

## 5. Plotting — `whisper_labia.plot_light_curve`

```python
plot_light_curve(lc, *, layout="report", quantity="apparent_mag", bands=None,
                 ncols=3, figsize=None, title=None, save=None)  ->  matplotlib.figure.Figure
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `lc` | `LightCurve` | — | The light curve to plot. |
| `layout` | `str` | `"report"` | `"report"` (2 panels: apparent-mag + flux, all bands) or `"grid"` (one panel per band). |
| `quantity` | `str` | `"apparent_mag"` | Grid y-axis: `"apparent_mag"`, `"absolute_mag"` (needs `redshift`), or `"flux"`. |
| `bands` | `list \| None` | `None` | Restrict to these bands. |
| `ncols` | `int` | `3` | Columns in the grid layout. |
| `figsize` | `tuple \| None` | `None` | Figure size. |
| `title` | `str \| None` | `None` | Title (defaults to the object name). |
| `save` | `str \| None` | `None` | Path to save a PNG. |

**Marker conventions:** distinct color per band; detections = circles (black edge); SNR<3 =
up-triangles; upper limits = down-triangles; magnitude axes inverted. The x-axis is "days since
explosion" when `set_explosion_date` / `explosion_date=` was used, else MJD.

---

## 6. Internals (developer reference)

- `loader._read_table`, `loader._resolve_columns`, `loader._to_bool` — CSV read + header/bool parsing.
- `schema.LightCurve._subset`, `._copy`, `.__post_init__` — array coercion/validation.
- `plotting._band_colors`, `_categories`, `_scatter`, `_quantity` — plot helpers.
- `scripts/phase0_smoke.py::main()` — Phase-0 smoke test.

## 7. Test coverage (44 tests, all passing)

| File | Tests | Focus |
|---|---|---|
| `tests/test_photometry.py` | 5 | AB zeropoint, mag↔flux round-trip, error propagation, mag-err→SNR. |
| `tests/test_bands.py` | 11 | Aliases, case-sensitivity, `group_bands`, `FILTER_LOOKUP`, `unmapped_bands`. |
| `tests/test_schema.py` | 11 | Validation, subsetting, `add_flux`/`add_mag`, `snr`/`select_snr`, `set_explosion_date`, upper limits. |
| `tests/test_loader.py` | 12 | AT2017GFO load, window/subset, grouping, `min_snr`, `explosion_date`, upper limits, errors. |
| `tests/test_plotting.py` | 5 | report/grid layouts, flux/absolute-mag, redshift guard, upper-limit markers. |

Fixtures: `tests/data/at2017gfo.csv` (kilonova, 645 pts, 29 bands) and `tests/data/ztf18aarlhfw.csv`
(ZTF, 2891 pts). Example figures: `docs/figures/`.

## 8. End-to-end example

```python
import whisper_labia as wp

lc = wp.load_lightcurve("at2017gfo.csv", band_lookup=True, min_snr=5,
                        time_max=57990.0, explosion_date=57982.0)
print(lc)                                   # LightCurve(name='at2017gfo', n_points=..., ...)
wp.plot_light_curve(lc, layout="report", save="report.png")
wp.plot_light_curve(lc, layout="grid", quantity="flux", bands=["g-band", "r-band", "i-band"])
```
