#!/usr/bin/env python
"""Fit the redback ``two_component_kilonova`` to AT2017GFO with ABC, MCMC and SNPE, and plot.

AT2017GFO *is* a kilonova (GW170817), so — unlike mck19 — this model should fit well. The expensive
redback simulator (~50 ms/call) makes the full 9-D fit costly, so we pin redshift (known, z=0.00984)
and the two temperature floors to narrow priors and fit the **6 core ejecta parameters** (mej/vej/kappa
for the blue + red components). The package-level model is used directly (picklable -> parallel ABC).

Usage (run the three samplers in parallel, then plot)::

    python scripts/fit_kilonova_at2017gfo.py abc &
    python scripts/fit_kilonova_at2017gfo.py mcmc &
    python scripts/fit_kilonova_at2017gfo.py snpe &
    wait
    python scripts/fit_kilonova_at2017gfo.py plot
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import whisper_labia as wp
from whisper_labia.models import get_model
from whisper_labia.priors import Prior, Uniform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "docs", "figures")
DATA = os.path.join(HERE, "tests", "data", "at2017gfo.csv")
Z_AT = 0.00984
T_FLOOR = 2500.0
BANDS = ["g", "r", "i"]
MODEL = "two_component_kilonova"
FREE = ["mej_1", "vej_1", "kappa_1", "mej_2", "vej_2", "kappa_2"]   # the 6 parameters we vary + plot
LOGPAR = {"mej_1", "mej_2"}
NPZ = {s: os.path.join(FIGDIR, f"kilonova_at2017gfo_{s}.npz") for s in ("abc", "mcmc", "snpe")}


def setup():
    lc = wp.load_lightcurve(DATA, explosion_date=57982.0, min_snr=3, bands=BANDS,
                            redshift=Z_AT).add_flux()
    # 6 free ejecta parameters + redshift / temperature floors pinned to narrow ("fixed") priors.
    prior = Prior({
        "mej_1": Uniform(1e-4, 0.1), "vej_1": Uniform(0.01, 0.7), "kappa_1": Uniform(0.1, 0.5),
        "mej_2": Uniform(1e-4, 0.1), "vej_2": Uniform(0.01, 0.7), "kappa_2": Uniform(1.0, 30.0),
        "temperature_floor_1": Uniform(T_FLOOR - 1.0, T_FLOOR + 1.0),
        "temperature_floor_2": Uniform(T_FLOOR - 1.0, T_FLOOR + 1.0),
        "redshift": Uniform(Z_AT - 1e-5, Z_AT + 1e-5),
    })
    return lc, prior


def run(sampler):
    lc, prior = setup()
    if sampler == "abc":
        res = wp.fit_ABC(lc, MODEL, prior=prior, n_simulations=40_000, quantile=0.005,
                         n_jobs=8, seed=0)
    elif sampler == "mcmc":
        res = wp.fit_MCMC(lc, MODEL, prior=prior, nsteps=400, burnin=150, thin=2,
                          space="flux", seed=0)
    elif sampler == "snpe":
        res = wp.fit_SNPE(lc, MODEL, prior=prior, num_rounds=1, num_simulations=3000,
                          num_samples=5000, space="flux", seed=0)
    else:
        raise SystemExit(f"unknown sampler {sampler!r}")

    allnames = list(prior.names)
    np.savez(NPZ[sampler],
             free=np.array(FREE), samples=res.samples[FREE].to_numpy(dtype=float),
             all_names=np.array(allnames), best=np.array([res.best_params[k] for k in allnames]),
             time=np.asarray(lc.time, float), band=np.asarray(lc.band).astype(str),
             mag=np.asarray(lc.magnitude, float), mag_err=np.asarray(lc.magnitude_err, float),
             aic=res.aic, bic=res.bic, max_loglik=res.max_log_likelihood,
             n_samples=res.n_samples, runtime=res.runtime_s,
             info_keys=np.array(list(res.info.keys())),
             info_vals=np.array([str(res.info[k]) for k in res.info]))
    print(f"[{sampler}] AIC={res.aic:.1f} n={res.n_samples} runtime={res.runtime_s:.1f}s")
    for k in FREE:
        s = res.summary[k]
        print(f"   {k:8s} median={s['median']:.4g}  [{s['ci16']:.4g}, {s['ci84']:.4g}]")
    print(f"[{sampler}] saved {NPZ[sampler]}")


def _best_full(d):
    return {n: float(v) for n, v in zip(list(d["all_names"]), d["best"])}


def _safe_range(x, pad=0.05):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9 * max(1.0, abs(hi)):
        s = max(abs(hi), 1.0) * 0.01
        return (lo - s, hi + s)
    dd = (hi - lo) * pad
    return (lo - dd, hi + dd)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    data = {s: np.load(NPZ[s], allow_pickle=True) for s in NPZ if os.path.exists(NPZ[s])}
    if not data:
        raise SystemExit("no result npz found; run abc/mcmc/snpe first")
    ref = data[next(iter(data))]
    t, band, mag, err = ref["time"], ref["band"].astype(str), ref["mag"], ref["mag_err"]
    tgrid = np.linspace(max(0.2, t.min()), t.max(), 300)
    col = {"abc": "#1f77b4", "mcmc": "#d62728", "snpe": "#9467bd"}
    band_col = {"g": "#2ca02c", "r": "#d62728", "i": "#8c564b"}

    m = get_model(MODEL)
    # Pre-compute every model curve (redback toggles matplotlib LaTeX) BEFORE building any figure.
    curves = {s: {b: -2.5 * np.log10(m.predict(_best_full(d), tgrid, np.array([b] * tgrid.size)) / 3631.0)
                  for b in BANDS} for s, d in data.items()}
    matplotlib.rcParams["text.usetex"] = False

    # ---- Figure A: best-fit models over the data ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    for b in BANDS:
        sel = band == b
        ax.errorbar(t[sel], mag[sel], yerr=err[sel], fmt="o", ms=4, color=band_col[b],
                    alpha=0.6, lw=0.7, capsize=0, label=f"{b} (data)")
    style = {"abc": "--", "mcmc": "-", "snpe": ":"}
    for s in data:
        for b in BANDS:
            ax.plot(tgrid, curves[s][b], style[s], color=band_col[b], lw=1.7,
                    label=(f"{s.upper()} fit" if b == "g" else None))
    ax.invert_yaxis()
    ax.set_xlabel("days since explosion (MJD 57982)")
    ax.set_ylabel("apparent AB magnitude")
    ax.set_title("two_component_kilonova fit to AT2017GFO (redshift + T_floor fixed)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    outA = os.path.join(FIGDIR, "kilonova_at2017gfo_fit.png")
    fig.savefig(outA, dpi=130, bbox_inches="tight")
    print("saved", outA)

    # ---- Figure B: posterior corner (ABC / MCMC / SNPE overlaid) ---------------------------------
    def transform(d):
        s = d["samples"]
        idx = {n: i for i, n in enumerate(list(d["free"]))}
        return np.column_stack([np.log10(s[:, idx[p]]) if p in LOGPAR else s[:, idx[p]] for p in FREE])

    labels = [(r"$\log_{10}$ " + p) if p in LOGPAR else p for p in FREE]
    X = {s: transform(d) for s, d in data.items()}
    allX = np.vstack(list(X.values()))
    rng = [_safe_range(allX[:, i]) for i in range(allX.shape[1])]
    fig = None
    for s, Xs in X.items():
        fig = corner.corner(Xs, fig=fig, color=col[s], labels=labels, range=rng, bins=22,
                            smooth=1.0, plot_datapoints=False, plot_density=False, fill_contours=False,
                            levels=(0.39, 0.86), contour_kwargs=dict(colors=col[s]),
                            hist_kwargs=dict(density=True, color=col[s], lw=1.5))
    fig.legend(handles=[Patch(color=col[s], label=s.upper()) for s in X],
               loc="upper right", frameon=True, fontsize=12, title="sampler")
    fig.suptitle("two_component_kilonova on AT2017GFO — ABC / MCMC / SNPE posterior", y=1.02)
    outB = os.path.join(FIGDIR, "kilonova_at2017gfo_corner.png")
    fig.savefig(outB, dpi=130, bbox_inches="tight")
    print("saved", outB)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if mode in ("abc", "mcmc", "snpe"):
        run(mode)
    elif mode == "plot":
        plot()
    else:
        raise SystemExit(__doc__)
