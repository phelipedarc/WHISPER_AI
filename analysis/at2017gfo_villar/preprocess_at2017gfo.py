#!/usr/bin/env python
"""Preprocess the full AT2017GFO UVOIR photometry (``at2017gfo_full.csv``) into a cleaner,
lower-noise reduction for the Villar+2017-style two-component fit.

Three cuts, in order:

1. **Drop ``uvot::uvw1``** — the sole Swift-UVOT band in the analysis; its handful of points sit at
   the very edge of the model's validity (in the WORST-modelled part of the ejecta), earn the least
   redback-filter-mapping confidence, and are the noisiest points in the set (see
   ``PLAN_fullband.md``).
2. **SNR > 5** (was >= 3) — a stricter photometric-quality cut, using WHISPER's own
   ``mag_err_to_snr`` (Pogson: ``SNR = (2.5/ln10) / sigma_mag``) so it matches exactly what
   ``LightCurve.select_snr``/``load_lightcurve(min_snr=...)`` would compute downstream.
3. **One observation per (band, round(MJD, 2))** — near-simultaneous multi-telescope/instrument
   points in the same band (common in heterogeneous, many-telescope AT2017GFO photometry) would
   otherwise over-weight that instant in the likelihood; keep the single most precise
   (smallest-error) point per band per ~14-minute (0.01 d) bin.

Output columns match ``load_lightcurve``: ``event,time,magnitude,e_magnitude,band,system``.

    python analysis/at2017gfo_villar/preprocess_at2017gfo.py   # data/at2017gfo_full.csv -> ..._preprocessed.csv
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from whisper_labia.io.photometry import mag_err_to_snr

SELF = os.path.dirname(os.path.abspath(__file__))            # analysis/at2017gfo_villar/
IN = os.path.join(SELF, "data", "at2017gfo_full.csv")
OUT = os.path.join(SELF, "data", "at2017gfo_full_preprocessed.csv")
DROP_BANDS = ["uvot::uvw1"]
MIN_SNR = 5.0


def preprocess(df, drop_bands=DROP_BANDS, min_snr=MIN_SNR, verbose=False):
    d = df[~df["band"].isin(drop_bands)].copy()
    if verbose:
        print(f"drop {drop_bands}: {len(df)} -> {len(d)}")

    snr = mag_err_to_snr(d["e_magnitude"].to_numpy(float))
    n_before = len(d)
    d = d[snr > min_snr].copy()
    if verbose:
        print(f"SNR > {min_snr:g}: {n_before} -> {len(d)}")

    d["_epoch"] = np.round(d["time"].to_numpy(float), 2)
    n_before = len(d)
    d = d.sort_values("e_magnitude").drop_duplicates(subset=["band", "_epoch"], keep="first")
    if verbose:
        print(f"one obs per (band, round(MJD,2)): {n_before} -> {len(d)}")

    return d.drop(columns="_epoch").sort_values("time").reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv(IN)
    out = preprocess(df, verbose=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)}/{len(df)} detections -> {OUT}")
    inv = out.groupby("band").agg(n=("magnitude", "size"),
                                  tmin=("time", "min"), tmax=("time", "max"))
    print(inv.sort_values("tmin").to_string())
