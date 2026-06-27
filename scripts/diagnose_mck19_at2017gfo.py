#!/usr/bin/env python
"""Quantitative diagnostics for the mck19-on-AT2017GFO ABC & MCMC fits (prints JSON).

Reports, per fitted parameter, the ABC and MCMC posterior median + 16/84 CI and a standardized
difference; convergence diagnostics (ABC effective n, MCMC acceptance + autocorrelation); the best-fit
RMS magnitude residual per band; and whether parameters rail against the (physical) prior bounds.
"""
from __future__ import annotations

import json
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np

from whisper_labia.models import get_model

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "docs", "figures")
NPZ = {s: os.path.join(FIGDIR, f"mck19_at2017gfo_{s}.npz") for s in ("abc", "mcmc")}
FREE = ["v_kick", "M_smbh", "M_bh", "r_bh"]
BOUNDS = {"v_kick": (100.0, 800.0), "M_smbh": (1e6, 1e9), "M_bh": (20.0, 160.0), "r_bh": (500.0, 10000.0)}
LOG = {"M_smbh", "r_bh"}  # rail-checked in log space


def _info(d):
    return {k: v for k, v in zip(list(d["info_keys"]), list(d["info_vals"]))}


def _pct(samples, names, p):
    idx = list(names).index(p)
    return np.percentile(samples[:, idx], [16, 50, 84])


def _rail(p, med):
    lo, hi = BOUNDS[p]
    if p in LOG:
        lo, hi, med = np.log10(lo), np.log10(hi), np.log10(med)
    span = hi - lo
    near_lo = (med - lo) / span < 0.02
    near_hi = (hi - med) / span < 0.02
    return "low" if near_lo else ("high" if near_hi else "interior")


def main():
    d_abc = np.load(NPZ["abc"], allow_pickle=True)
    d_mcmc = np.load(NPZ["mcmc"], allow_pickle=True)
    out = {"params": {}, "convergence": {}, "metrics": {}, "residual_mag": {}}

    for p in FREE:
        a16, a50, a84 = _pct(d_abc["samples"], d_abc["names"], p)
        m16, m50, m84 = _pct(d_mcmc["samples"], d_mcmc["names"], p)
        # standardized difference of medians vs combined 1-sigma widths
        wa, wm = (a84 - a16) / 2, (m84 - m16) / 2
        z = abs(a50 - m50) / np.hypot(wa, wm) if (wa + wm) > 0 else float("nan")
        out["params"][p] = {
            "abc": {"median": float(a50), "ci16": float(a16), "ci84": float(a84), "rail": _rail(p, a50)},
            "mcmc": {"median": float(m50), "ci16": float(m16), "ci84": float(m84), "rail": _rail(p, m50)},
            "median_diff_sigma": float(z),
        }

    out["convergence"]["abc"] = {"n_samples": int(d_abc["n_samples"]), **_info(d_abc)}
    out["convergence"]["mcmc"] = {"n_samples": int(d_mcmc["n_samples"]), **_info(d_mcmc)}
    out["metrics"] = {
        "abc": {"aic": float(d_abc["aic"]), "bic": float(d_abc["bic"]), "max_loglik": float(d_abc["max_loglik"])},
        "mcmc": {"aic": float(d_mcmc["aic"]), "bic": float(d_mcmc["bic"]), "max_loglik": float(d_mcmc["max_loglik"])},
    }

    # best-fit RMS magnitude residual per band (uses each sampler's best_params)
    m = get_model("mck19")
    t, band, mag = d_mcmc["time"], d_mcmc["band"], d_mcmc["mag"]
    for tag, d in [("abc", d_abc), ("mcmc", d_mcmc)]:
        best = {n: float(v) for n, v in zip(list(d["names"]), d["best"])}
        pmag = -2.5 * np.log10(m.predict(best, t, band) / 3631.0)
        per_band = {}
        for b in ["g", "r", "i"]:
            sel = band == b
            per_band[b] = float(np.sqrt(np.nanmean((pmag[sel] - mag[sel]) ** 2)))
        per_band["all"] = float(np.sqrt(np.nanmean((pmag - mag) ** 2)))
        out["residual_mag"][tag] = per_band

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
