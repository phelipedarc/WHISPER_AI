#!/usr/bin/env python
"""Fetch the COMPLETE AT2017GFO (GW170817) photometry from the Open Astronomy Catalog and write a
clean, WHISPER-format CSV spanning UV -> optical -> NIR.

The repo's ``tests/data/at2017gfo.csv`` was reduced to g/r/i only, which underconstrains the red
(lanthanide-rich, NIR-dominated) kilonova component. This pulls the full multi-band light curve so the
Villar+2017-style two-component fit can constrain BOTH components (as Villar+17 did with UVOIR data).

Output columns match ``load_lightcurve``: ``event,time,magnitude,e_magnitude,band,system``. Every band
name is one redback (``redback/tables/filters.csv``) natively understands, so the model needs no filter
translation. Requires network; the OAC API cert isn't in the container trust store, so we fetch with
``verify=False`` (data is public, read-only).

    python analysis/at2017gfo_villar/fetch_at2017gfo_full.py    # -> data/at2017gfo_full.csv
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings()

SELF = os.path.dirname(os.path.abspath(__file__))            # analysis/at2017gfo_villar/
OUT = os.path.join(SELF, "data", "at2017gfo_full.csv")
T_MERGER = 57982.529            # GW170817 merger, MJD (Abbott+2017)
OAC = ("https://api.astrocats.space/AT2017gfo/photometry/"
       "time+magnitude+e_magnitude+band+system+instrument+telescope?format=csv")

# Vega -> AB offsets (m_AB = m_Vega + off); redback works in AB. Only ~18 non-AB points exist, mostly V.
VEGA_AB = {"U": 0.79, "B": -0.09, "V": 0.02, "R": 0.21, "I": 0.45,
           "J": 0.91, "H": 1.39, "K": 1.85, "Ks": 1.83, "Y": 0.63}
# OAC band -> redback filter name (identity for all but a couple of aliases).
BAND_ALIAS = {"W": "uvot::white", "w": "uvot::uvw1"}   # Swift white / UV; keep everything else as-is
# well-sampled bands that span UV->NIR (drop 1-2-point HST/rare filters by default)
KEEP_DEFAULT = ["U", "u", "uvot::uvw1", "uvot::white", "B", "g", "V", "r", "R", "i", "I",
                "z", "y", "Y", "J", "H", "K", "Ks"]


def fetch():
    r = requests.get(OAC, timeout=60, verify=False, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    df["e_magnitude"] = pd.to_numeric(df["e_magnitude"], errors="coerce")
    return df


def clean(df, keep=None, tmax_days=40.0):
    keep = keep or KEEP_DEFAULT
    m = ((df["time"] > T_MERGER) & (df["time"] < T_MERGER + tmax_days)
         & df["magnitude"].notna() & df["e_magnitude"].notna() & (df["e_magnitude"] > 0))
    d = df[m].copy()
    d["band"] = d["band"].map(lambda b: BAND_ALIAS.get(b, b))
    # Vega -> AB (uses the ORIGINAL band letter for the offset, pre-alias)
    is_vega = d["system"].astype(str).str.lower().eq("vega")
    d.loc[is_vega, "magnitude"] += d.loc[is_vega, "band"].map(VEGA_AB).fillna(0.0)
    d["system"] = "AB"
    d = d[d["band"].isin(keep)]
    d["event"] = "AT2017gfo"
    out = d[["event", "time", "magnitude", "e_magnitude", "band", "system"]].sort_values("time")
    return out.reset_index(drop=True)


if __name__ == "__main__":
    keep = sys.argv[1:] or None
    df = fetch()
    out = clean(df, keep=keep)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"wrote {len(out)} detections -> {OUT}")
    print(f"time span: {out['time'].min() - T_MERGER:.2f} to {out['time'].max() - T_MERGER:.2f} d")
    inv = out.groupby("band").agg(n=("magnitude", "size"),
                                  tmin=("time", lambda x: round(x.min() - T_MERGER, 1)),
                                  tmax=("time", lambda x: round(x.max() - T_MERGER, 1)))
    print(inv.sort_values("tmin").to_string())
