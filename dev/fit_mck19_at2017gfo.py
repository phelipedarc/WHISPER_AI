#!/usr/bin/env python
"""Fit the ``mck19`` model (BBH-in-AGN-disk flare) to AT2017GFO with ABC and MCMC, and plot.

AT2017GFO is the GW170817 **kilonova** (a neutron-star merger), *not* a BBH-in-AGN-disk event, so
``mck19`` is physically the wrong model here — this is a demonstration of running the new model through
Whisper's samplers on real data. As expected, the fit is poor and parameters rail against the (physical)
prior bounds; that mismatch (high AIC) is exactly the signal Whisper's model comparison surfaces.

Usage (run the two fits in parallel, then plot)::

    python dev/fit_mck19_at2017gfo.py abc &
    python dev/fit_mck19_at2017gfo.py mcmc &
    wait
    python dev/fit_mck19_at2017gfo.py plot
"""
from __future__ import annotations

import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import whisper_labia as wp
from whisper_labia.models import get_model
from whisper_labia.priors import LogUniform, Prior, Uniform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "dev", "figures")
DATA = os.path.join(HERE, "tests", "data", "at2017gfo.csv")
Z_AT = 0.00984                       # GW170817 / NGC 4993 redshift (known -> fixed)
BANDS = ["g", "r", "i"]
NPZ = {s: os.path.join(FIGDIR, f"mck19_at2017gfo_{s}.npz") for s in ("abc", "mcmc")}
# the four *fitted* physical parameters (redshift is fixed at the known value)
FREE = ["v_kick", "M_smbh", "M_bh", "r_bh"]


def setup():
    # add_flux gives a flux column (+ propagated errors) so ABC (χ² on flux) and MCMC (space="flux")
    # compare in the SAME space -> a fair sampler-vs-sampler comparison.
    lc = wp.load_lightcurve(DATA, explosion_date=57982.0, min_snr=3, bands=BANDS,
                            redshift=Z_AT).add_flux()
    prior = Prior({
        "v_kick": Uniform(100.0, 800.0),
        "M_smbh": LogUniform(1e6, 1e9),
        "M_bh": Uniform(20.0, 160.0),
        "r_bh": LogUniform(500.0, 10000.0),
        "redshift": Uniform(Z_AT - 1e-4, Z_AT + 1e-4),   # fixed (narrow) at the known z
    })
    return lc, prior


def run(sampler):
    lc, prior = setup()
    if sampler == "abc":
        res = wp.fit_ABC(lc, "mck19", prior=prior, n_simulations=300_000, quantile=0.002,
                         n_jobs=8, seed=0)
    elif sampler == "mcmc":
        res = wp.fit_MCMC(lc, "mck19", prior=prior, nsteps=6000, burnin=1500, thin=5,
                          space="flux", seed=0)
    else:
        raise SystemExit(f"unknown sampler {sampler!r}")

    samp = res.samples[list(prior.names)].to_numpy(dtype=float)
    np.savez(
        NPZ[sampler],
        names=np.array(list(prior.names)), samples=samp,
        best=np.array([res.best_params[k] for k in prior.names]),
        time=np.asarray(lc.time, float), band=np.asarray(lc.band).astype(str),
        mag=np.asarray(lc.magnitude, float), mag_err=np.asarray(lc.magnitude_err, float),
        aic=res.aic, bic=res.bic, max_loglik=res.max_log_likelihood,
        n_samples=res.n_samples, runtime=res.runtime_s,
        info_keys=np.array(list(res.info.keys())),
        info_vals=np.array([str(res.info[k]) for k in res.info]),
    )
    print(f"[{sampler}] AIC={res.aic:.1f} n_samples={res.n_samples} runtime={res.runtime_s:.1f}s")
    for k in FREE:
        s = res.summary[k]
        print(f"   {k:8s} median={s['median']:.4g}  [{s['ci16']:.4g}, {s['ci84']:.4g}]")
    print(f"[{sampler}] saved {NPZ[sampler]}")


def _best_dict(d):
    names = list(d["names"])
    return {n: float(v) for n, v in zip(names, d["best"])}


def _safe_range(x, pad=0.05):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-9 * max(1.0, abs(hi)):
        span = max(abs(hi), 1.0) * 0.01
        lo, hi = lo - span, hi + span
    else:
        d = (hi - lo) * pad
        lo, hi = lo - d, hi + d
    return (lo, hi)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import corner
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d_abc, d_mcmc = np.load(NPZ["abc"], allow_pickle=True), np.load(NPZ["mcmc"], allow_pickle=True)
    m = get_model("mck19")
    band_color = {"g": "#2ca02c", "r": "#d62728", "i": "#8c564b"}

    # ---- Figure A: best-fit model over the data --------------------------------------------------
    t = d_mcmc["time"]; band = d_mcmc["band"]; mag = d_mcmc["mag"]; err = d_mcmc["mag_err"]
    tgrid = np.linspace(max(1e-2, t.min()), t.max(), 400)
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for b in BANDS:
        sel = band == b
        ax.errorbar(t[sel], mag[sel], yerr=err[sel], fmt="o", ms=4, color=band_color[b],
                    alpha=0.65, lw=0.8, capsize=0, label=f"{b} (data)")
    for d, ls, lbl in [(d_mcmc, "-", "MCMC best fit"), (d_abc, "--", "ABC best fit")]:
        best = _best_dict(d)
        for b in BANDS:
            pred = m.predict(best, tgrid, np.array([b] * tgrid.size))
            ax.plot(tgrid, -2.5 * np.log10(pred / 3631.0), ls, color=band_color[b], lw=1.8,
                    label=(lbl if b == "g" else None))
    ax.invert_yaxis()
    ax.set_xlabel("days since explosion (MJD 57982)")
    ax.set_ylabel("apparent AB magnitude")
    ax.set_title("mck19 fit to AT2017GFO  —  ABC & MCMC")
    ax.text(0.5, 1.06, "caveat: AT2017GFO is a kilonova (NS–NS merger); mck19 models a BBH-in-AGN flare "
            "—\nthe model is misspecified here (poor fit / railed parameters by design).",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8, color="0.35")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    outA = os.path.join(FIGDIR, "mck19_at2017gfo_fit.png")
    fig.savefig(outA, dpi=130, bbox_inches="tight")
    print("saved", outA)

    # ---- Figure B: overlaid ABC + MCMC posteriors ------------------------------------------------
    def transform(d):
        names = list(d["names"]); s = d["samples"]
        idx = {n: i for i, n in enumerate(names)}
        cols = [s[:, idx["v_kick"]], np.log10(s[:, idx["M_smbh"]]),
                s[:, idx["M_bh"]], np.log10(s[:, idx["r_bh"]])]
        return np.column_stack(cols)

    labels = [r"$v_{\rm kick}$", r"$\log_{10} M_\bullet$", r"$M_{\rm BH}$", r"$\log_{10} r$"]
    X_abc, X_mcmc = transform(d_abc), transform(d_mcmc)
    allX = np.vstack([X_abc, X_mcmc])
    rng = [_safe_range(allX[:, i]) for i in range(allX.shape[1])]

    fig = None
    for X, color in [(X_abc, "#1f77b4"), (X_mcmc, "#d62728")]:
        fig = corner.corner(X, fig=fig, color=color, labels=labels, range=rng, bins=24,
                            smooth=1.0, plot_datapoints=False, plot_density=False,
                            fill_contours=False, levels=(0.39, 0.86),
                            contour_kwargs=dict(colors=color),
                            hist_kwargs=dict(density=True, color=color, lw=1.6))
    fig.legend(handles=[Patch(color="#1f77b4", label="ABC"), Patch(color="#d62728", label="MCMC")],
               loc="upper right", frameon=True, fontsize=12, title="sampler")
    fig.suptitle("mck19 on AT2017GFO: ABC vs MCMC posterior (redshift fixed at z=0.00984)", y=1.02)
    outB = os.path.join(FIGDIR, "mck19_at2017gfo_corner.png")
    fig.savefig(outB, dpi=130, bbox_inches="tight")
    print("saved", outB)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "plot"
    if mode in ("abc", "mcmc"):
        run(mode)
    elif mode == "plot":
        plot()
    else:
        raise SystemExit(__doc__)
