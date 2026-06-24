"""Flexible CSV -> :class:`LightCurve` loader."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .bands import FILTER_LOOKUP, group_bands, normalize_bands
from .schema import LightCurve

# Canonical field -> accepted header synonyms (matched case-insensitively).
CANONICAL_SYNONYMS = {
    "time": ["time", "mjd", "jd", "hjd", "t", "date"],
    "magnitude": ["magnitude", "mag", "apparent_mag", "app_mag", "appmag"],
    "magnitude_err": ["e_magnitude", "magnitude_err", "magnitude_error", "magerr",
                      "mag_err", "e_mag", "emag", "apparent_magerr", "dmag", "sigma_mag"],
    "flux": ["flux", "flux_density", "fluxdensity", "forcediffimflux", "fnu"],
    "flux_err": ["flux_err", "fluxerr", "flux_error", "e_flux", "flux_density_err",
                 "forcediffimfluxunc", "sigma_flux"],
    "band": ["band", "filter", "filtercode", "filtername", "passband", "bandpass"],
    "system": ["system", "magsystem", "magsys", "photsystem"],
    "name": ["event", "name", "object", "objectid", "oid", "iau", "transient", "sn"],
    "upper_limit": ["upper_limit", "upperlimit", "islimit", "is_limit", "nondetection", "ul"],
}

_TRUE = {"1", "true", "t", "yes", "y"}


def _read_table(path, delimiter=None):
    if delimiter is not None:
        return pd.read_csv(path, delimiter=delimiter)
    try:  # auto-sniff comma/semicolon/whitespace
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path)


def _norm(header):
    return str(header).strip().lower()


def _to_bool(series):
    return np.array([str(v).strip().lower() in _TRUE for v in series])


def _resolve_columns(columns, column_map=None):
    by_norm = {}
    for c in columns:
        by_norm.setdefault(_norm(c), c)
    resolved = dict(column_map or {})
    for canon, syns in CANONICAL_SYNONYMS.items():
        if canon in resolved:
            continue
        for s in syns:
            if s in by_norm:
                resolved[canon] = by_norm[s]
                break
    return resolved


def load_lightcurve(path, *, name=None, redshift=None, column_map=None, band_aliases=None,
                    band_lookup=None, normalize=True, default_band=None, quality_cuts=True,
                    drop_nonfinite=True, flag_filters=None, time_min=None, time_max=None,
                    bands=None, min_snr=None, explosion_date=None, delimiter=None):
    """Load a light-curve CSV into a canonical :class:`LightCurve`.

    Auto-detects columns (case-insensitive synonyms; override with ``column_map``), normalizes band
    names, drops bad rows (non-finite values, non-positive errors -- upper limits are kept), and
    optionally filters by band/time/SNR. Supports magnitude- or flux-based inputs.

    Key options: ``column_map={'time': 'MJD', ...}`` to force a mapping; ``default_band='ztfg'`` when
    there is no band column; ``band_lookup=True`` for broadband grouping; ``flag_filters={'catflags': 0}``
    to keep rows by a quality flag; ``time_min/time_max`` (MJD), ``bands=[...]`` and ``min_snr`` (e.g.
    3 or 5) to subset; ``explosion_date=<MJD>`` to express time as days since explosion (day 0).
    """
    path = Path(path)
    df = _read_table(path, delimiter)
    cols = _resolve_columns(df.columns, column_map)

    if "time" not in cols:
        raise ValueError(
            f"No time column found (looked for {CANONICAL_SYNONYMS['time']}). "
            f"Available columns: {list(df.columns)}. Pass column_map={{'time': '<column>'}}.")
    time = pd.to_numeric(df[cols["time"]], errors="coerce").to_numpy(dtype=float)

    if "band" in cols:
        band = np.array([str(b) for b in df[cols["band"]].to_numpy()])
    elif default_band is not None:
        band = np.array([str(default_band)] * len(df))
    else:
        raise ValueError(
            f"No band/filter column found (looked for {CANONICAL_SYNONYMS['band']}). "
            f"Available columns: {list(df.columns)}. "
            f"Pass default_band='...' or column_map={{'band': '<column>'}}.")

    def num(field):
        return (pd.to_numeric(df[cols[field]], errors="coerce").to_numpy(dtype=float)
                if field in cols else None)

    magnitude, magnitude_err = num("magnitude"), num("magnitude_err")
    flux, flux_err = num("flux"), num("flux_err")
    if magnitude is None and flux is None:
        raise ValueError(
            f"No magnitude or flux column found. Available columns: {list(df.columns)}.")

    upper_limit = _to_bool(df[cols["upper_limit"]]) if "upper_limit" in cols else None

    system = None
    if "system" in cols:
        system = np.array([
            "unknown" if str(s).strip().lower() in ("nan", "none", "") else str(s).strip()
            for s in df[cols["system"]].to_numpy()
        ])

    if name is None and "name" in cols and len(df):
        name = str(df[cols["name"]].iloc[0])

    if normalize:
        band = normalize_bands(band, aliases=band_aliases)
    if band_lookup is not None and band_lookup is not False:
        lookup = FILTER_LOOKUP if band_lookup is True else band_lookup
        band = group_bands(band, lookup=lookup)

    # --- row mask: quality cuts (upper limits are exempt from the error cut) ---
    ul_mask = upper_limit if upper_limit is not None else np.zeros(len(df), dtype=bool)
    mask = np.ones(len(df), dtype=bool)
    primary = magnitude if magnitude is not None else flux
    primary_err = magnitude_err if magnitude is not None else flux_err
    if drop_nonfinite:
        mask &= np.isfinite(time)
        mask &= np.isfinite(primary)
    if quality_cuts and primary_err is not None:
        mask &= (np.isfinite(primary_err) & (primary_err > 0)) | ul_mask
    if flag_filters:
        for fcol, cond in flag_filters.items():
            if fcol not in df.columns:
                raise ValueError(f"flag_filters column {fcol!r} not in {list(df.columns)}")
            values = df[fcol].to_numpy()
            mask &= (np.array([bool(cond(v)) for v in values]) if callable(cond)
                     else (values == cond))

    def m(v):
        return None if v is None else v[mask]

    lc = LightCurve(
        time=time[mask], band=band[mask],
        magnitude=m(magnitude), magnitude_err=m(magnitude_err),
        flux=m(flux), flux_err=m(flux_err), upper_limit=m(upper_limit), system=m(system),
        name=name, redshift=redshift,
        meta={"source_file": str(path), "n_rows_raw": int(len(df)),
              "n_rows_kept": int(mask.sum())},
    )

    if bands is not None:
        lc = lc.select_bands(bands)
    if time_min is not None or time_max is not None:
        lc = lc.select_time_window(time_min, time_max)
    if min_snr is not None:
        lc = lc.select_snr(min_snr)
    if explosion_date is not None:
        lc = lc.set_explosion_date(explosion_date)
    return lc
