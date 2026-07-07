#!/usr/bin/env python
"""Sanity check: fit AT2017GFO with mck19 in FLUX mode vs MAGNITUDE mode, across ABC / MCMC / SNPE.

Tests Whisper's data_mode / likelihood machinery — the *same* AT2017GFO data, represented as flux
(flux + flux_err) or as magnitude (magnitude + mag_err), should be fit consistently by every sampler:

* MCMC / SNPE switch space via ``space="flux"|"magnitude"`` (the shared GaussianLikelihood).
* ABC compares via χ² on flux by default; for magnitude mode we pass a magnitude-space χ² distance.

mck19 is the wrong physical model for a kilonova, so flux- and magnitude-space (which weight residuals
differently) need not give identical posteriors; the machinery running cleanly and the two modes landing
in a comparable region across all six configurations is the check. (The clean version of this test will
come with a kilonova model that actually fits AT2017GFO.)

    python sanity_check/sanity_mck19_modes.py        # fit all six configs, then plot
    python sanity_check/sanity_mck19_modes.py plot    # re-plot from saved results
"""
from __future__ import annotations

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import whisper_labia as wp
from whisper_labia.models import get_model
from whisper_labia.priors import LogUniform, Prior, Uniform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "sanity_check", "figures")
DATA = os.path.join(HERE, "tests", "data", "at2017gfo.csv")
RESULTS = os.path.join(FIGDIR, "mck19_modes_results.json")
DATANPZ = os.path.join(FIGDIR, "mck19_modes_data.npz")
Z_AT = 0.00984
BANDS = ["g", "r", "i"]
FREE = ["v_kick", "M_smbh", "M_bh", "r_bh"]
AB = 3631.0
MODES = ["flux", "magnitude"]
SAMPLERS = ["abc", "mcmc", "snpe"]
COL = {"abc": "#1f77b4", "mcmc": "#d62728", "snpe": "#9467bd"}
LOGPAR = {"M_smbh", "r_bh"}

# Fast configurations (SNPE explicitly fast: single-round NPE, few simulations/epochs).
CONFIG = {
    "abc": dict(n_simulations=150_000, quantile=0.003, n_jobs=8, seed=0),
    "mcmc": dict(nsteps=3000, burnin=800, thin=4, seed=0),
    "snpe": dict(num_rounds=1, num_simulations=2000, num_samples=4000, seed=0),
}


def mag_chi2(obs_flux, obs_flux_err, sim_flux, bands=None):
    """Magnitude-space χ² distance for ABC (converts the flux it is handed back to AB mag)."""
    of = np.asarray(obs_flux, float)
    sf = np.maximum(np.asarray(sim_flux, float), 1e-300)
    om = -2.5 * np.log10(of / AB)
    sm = -2.5 * np.log10(sf / AB)
    me = (2.5 / np.log(10.0)) * np.asarray(obs_flux_err, float) / of
    return float(np.sum(((om - sm) / me) ** 2))


def setup():
    lc = wp.load_lightcurve(DATA, explosion_date=57982.0, min_snr=3, bands=BANDS,
                            redshift=Z_AT).add_flux()
    prior = Prior({
        "v_kick": Uniform(100.0, 800.0),
        "M_smbh": LogUniform(1e6, 1e9),
        "M_bh": Uniform(20.0, 160.0),
        "r_bh": LogUniform(500.0, 10000.0),
        "redshift": Uniform(Z_AT - 1e-4, Z_AT + 1e-4),
    })
    return lc, prior


def fit_one(lc, prior, sampler, mode):
    if sampler == "abc":
        kw = dict(CONFIG["abc"])
        if mode == "magnitude":
            kw["distance"] = mag_chi2
        return wp.fit_ABC(lc, "mck19", prior=prior, **kw)
    if sampler == "mcmc":
        return wp.fit_MCMC(lc, "mck19", prior=prior, space=mode, **CONFIG["mcmc"])
    return wp.fit_SNPE(lc, "mck19", prior=prior, space=mode, **CONFIG["snpe"])


def run():
    lc, prior = setup()
    m = get_model("mck19")
    t = np.asarray(lc.time, float)
    band = np.asarray(lc.band).astype(str)
    mag = np.asarray(lc.magnitude, float)
    mag_err = np.asarray(lc.magnitude_err, float)
    np.savez(DATANPZ, time=t, band=band, mag=mag, mag_err=mag_err)

    results = {}
    for mode in MODES:
        for sampler in SAMPLERS:
            key = f"{mode}:{sampler}"
            try:
                res = fit_one(lc, prior, sampler, mode)
                best = {k: float(res.best_params[k]) for k in prior.names}
                pmag = -2.5 * np.log10(np.maximum(m.predict(best, t, band), 1e-300) / AB)
                rms = float(np.sqrt(np.nanmean((pmag - mag) ** 2)))
                results[key] = {
                    "best": best,
                    "summary": {k: {kk: float(vv) for kk, vv in res.summary[k].items()} for k in FREE},
                    "aic": float(res.aic), "max_loglik": float(res.max_log_likelihood),
                    "n_samples": int(res.n_samples), "rms_mag": rms, "ok": True,
                }
                print(f"[{mode:9s} {sampler:4s}] AIC={res.aic:11.1f}  rms={rms:.3f} mag  n={res.n_samples}")
            except Exception as exc:  # keep going; mark the failure
                results[key] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                print(f"[{mode:9s} {sampler:4s}] FAILED: {type(exc).__name__}: {exc}")
    with open(RESULTS, "w") as fh:
        json.dump(results, fh, indent=2)
    print("saved", RESULTS)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    results = json.load(open(RESULTS))
    d = np.load(DATANPZ, allow_pickle=True)
    t, band, mag, mag_err = d["time"], d["band"].astype(str), d["mag"], d["mag_err"]
    m = get_model("mck19")
    tgrid = np.linspace(max(1e-2, t.min()), t.max(), 400)

    # ---- Figure A: per-sampler, flux-mode (solid) vs magnitude-mode (dashed) best-fit on r-band -----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
    rsel = band == "r"
    for ax, sampler in zip(axes, SAMPLERS):
        ax.errorbar(t[rsel], mag[rsel], yerr=mag_err[rsel], fmt="o", ms=4, color="0.35",
                    alpha=0.6, lw=0.7, capsize=0, label="AT2017GFO r")
        for mode, ls in [("flux", "-"), ("magnitude", "--")]:
            r = results.get(f"{mode}:{sampler}", {})
            if not r.get("ok"):
                continue
            pred = m.predict(r["best"], tgrid, np.array(["r"] * tgrid.size))
            ax.plot(tgrid, -2.5 * np.log10(pred / AB), ls, color=COL[sampler], lw=1.9,
                    label=f"{mode} mode")
        ax.set_title(sampler.upper())
        ax.set_xlabel("days since explosion")
        ax.legend(frameon=False, fontsize=8)
    axes[0].invert_yaxis()                         # single invert (axes share y)
    axes[0].set_ylabel("apparent AB magnitude (r)")
    fig.suptitle("AT2017GFO / mck19 — flux-mode vs magnitude-mode best fit (per sampler)", y=1.00)
    fig.tight_layout()
    outA = os.path.join(FIGDIR, "mck19_modes_fit.png")
    fig.savefig(outA, dpi=130, bbox_inches="tight")
    print("saved", outA)

    # ---- Figure B: posterior parameter comparison across the six configs ------------------------
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0))
    xpos = {f"{mode}:{s}": i for i, (mode, s) in enumerate(
        [(mo, s) for mo in MODES for s in SAMPLERS])}
    for ax, par in zip(axes, FREE):
        for key, x in xpos.items():
            mode, sampler = key.split(":")
            r = results.get(key, {})
            if not r.get("ok"):
                continue
            s = r["summary"][par]
            med, lo, hi = s["median"], s["ci16"], s["ci84"]
            mk = "o" if mode == "flux" else "s"
            ax.errorbar([x], [med], yerr=[[med - lo], [hi - med]], fmt=mk, color=COL[sampler],
                        ms=7, capsize=3, mfc=(COL[sampler] if mode == "flux" else "white"))
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels([k.replace("flux", "F").replace("magnitude", "M").replace(":", "\n")
                            for k in xpos], fontsize=7)
        ax.set_title(par)
        if par in LOGPAR:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
    handles = [Line2D([], [], marker="o", color="0.3", ls="", label="flux mode (filled ●)"),
               Line2D([], [], marker="s", color="0.3", ls="", mfc="white", label="magnitude mode (open ■)")]
    fig.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)
    fig.suptitle("AT2017GFO / mck19 — posterior medians (16–84%) by mode × sampler", y=1.02)
    fig.tight_layout()
    outB = os.path.join(FIGDIR, "mck19_modes_params.png")
    fig.savefig(outB, dpi=130, bbox_inches="tight")
    print("saved", outB)

    # ---- console summary table -------------------------------------------------------------------
    print("\n%-18s %12s %10s %9s" % ("config", "AIC", "RMS[mag]", "n"))
    for mode in MODES:
        for sampler in SAMPLERS:
            r = results.get(f"{mode}:{sampler}", {})
            if r.get("ok"):
                print("%-18s %12.1f %10.3f %9d" % (f"{mode}:{sampler}", r["aic"], r["rms_mag"], r["n_samples"]))
            else:
                print("%-18s   FAILED (%s)" % (f"{mode}:{sampler}", r.get("error", "?")))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        run()
        plot()
