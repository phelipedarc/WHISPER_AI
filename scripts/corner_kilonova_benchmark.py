#!/usr/bin/env python
"""Publication corner plot — AT2017GFO posteriors for every benchmark sampler, annotated with WAIC.

Overlays the ABC / MCMC / SNPE (CPU) / SNPE (GPU) posteriors from the kilonova benchmark
(`benchmark_kilonova_modes.py`) on a single corner plot via the built-in `wp.plot_corner` (dark
palette), and annotates each method with its **WAIC** (`wp.waic`). Unlike a table of medians, the
corner shows the full posteriors + uncertainties, so whether the methods are *compatible* is visible
directly. WAIC (lower = better) adds a fully-Bayesian model-fit score that uses the whole posterior.

    python scripts/corner_kilonova_benchmark.py [flux|magnitude]
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import whisper_labia as wp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "docs", "figures")
DATA = os.path.join(HERE, "tests", "data", "at2017gfo.csv")
Z_AT, T_FLOOR = 0.00984, 2500.0
BANDS = ["g", "r", "i"]
FREE = ["mej_1", "vej_1", "kappa_1", "mej_2", "vej_2", "kappa_2"]
SAMPLERS = [("abc", "ABC"), ("mcmc", "MCMC"), ("snpe", "SNPE (CPU)"), ("snpe_gpu", "SNPE (GPU)")]
FIXED = dict(redshift=Z_AT, temperature_floor_1=T_FLOOR, temperature_floor_2=T_FLOOR)


def main(mode):
    posteriors = []
    for key, label in SAMPLERS:
        npz = os.path.join(FIGDIR, f"kilonova_bench_{mode}_{key}.npz")
        if not os.path.exists(npz):
            print("skip (missing):", os.path.basename(npz))
            continue
        d = np.load(npz, allow_pickle=True)
        posteriors.append([label, pd.DataFrame(d["samples"], columns=list(d["free"]))])

    if not posteriors:
        raise SystemExit("no benchmark posteriors found; run benchmark_kilonova_modes.py first")

    # WAIC per method (uses the full redback model with redshift + temperature floors held fixed).
    lc = wp.load_lightcurve(DATA, explosion_date=57982.0, min_snr=3, bands=BANDS,
                            redshift=Z_AT).add_flux()
    # WAIC per method (reported to the console; the legend stays clean — WAIC is numerically unstable
    # for the broad ABC/SNPE posteriors here, where p_waic >> #params, so it is a diagnostic, not a
    # clean ranking. The corner itself answers the compatibility question.)
    print(f"\n{'method':12s}{'WAIC':>14s}{'p_waic':>12s}{'lppd':>11s}{'se':>11s}")
    for label, df in posteriors:
        w = wp.waic(df, lc, "two_component_kilonova", space=mode, fixed=FIXED, max_samples=400, seed=0)
        print(f"{label:12s}{w['waic']:>14.0f}{w['p_waic']:>12.0f}{w['lppd']:>11.1f}{w['se']:>11.0f}")

    # redback's import (via waic) turns matplotlib LaTeX on; force it off for rendering.
    matplotlib.rcParams["text.usetex"] = False
    out = os.path.join(FIGDIR, f"at2017gfo_corner_{mode}.png")
    wp.plot_corner(
        [df for _, df in posteriors],
        labels=[label for label, _ in posteriors],
        parameters=FREE, log_params=["mej_1", "mej_2"],
        title=f"AT2017GFO — two_component_kilonova posteriors ({mode}-space), all samplers",
        save=out)
    print("saved", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "flux")
