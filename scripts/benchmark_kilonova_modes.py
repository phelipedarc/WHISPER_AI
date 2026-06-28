#!/usr/bin/env python
"""WHISPER benchmark — ``two_component_kilonova`` on AT2017GFO, flux-space vs magnitude-space.

A timed, reproducible sanity check / benchmark: fit the redback two-component kilonova to AT2017GFO
(g/r/i) in **flux** mode (flux + flux_err) and **magnitude** mode, with each of the three samplers
(ABC, MCMC, SNPE). Because AT2017GFO is a real kilonova the model fits well, so the two likelihood
spaces should agree — the sanity check — while the per-sampler **runtime** gives WHISPER a benchmark.

* MCMC / SNPE switch space via ``space=``; ABC compares χ² on flux by default and on magnitude via the
  module-level ``mag_chi2`` distance (so all six configs use a matched comparison space).
* 6 core ejecta parameters are fit; redshift (z=0.00984) and the temperature floors are pinned.

Each ``fit <mode> <sampler>`` run writes its own result + samples, so the six can run in parallel::

    for m in flux magnitude; do for s in abc mcmc snpe; do
        python scripts/benchmark_kilonova_modes.py fit $m $s & ; done; done; wait
    python scripts/benchmark_kilonova_modes.py plot

Note: runtimes are wall-clock under the given budgets / `n_jobs` on a shared host — relative comparison
is robust; absolute numbers depend on machine load.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import whisper_labia as wp
from whisper_labia.models import get_model
from whisper_labia.priors import Prior, Uniform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "docs", "figures")
DATA = os.path.join(HERE, "tests", "data", "at2017gfo.csv")
Z_AT, T_FLOOR = 0.00984, 2500.0
BANDS = ["g", "r", "i"]
MODEL = "two_component_kilonova"
FREE = ["mej_1", "vej_1", "kappa_1", "mej_2", "vej_2", "kappa_2"]
LOGPAR = {"mej_1", "mej_2"}
MODES, SAMPLERS = ["flux", "magnitude"], ["abc", "mcmc", "snpe", "snpe_gpu"]
AB = 3631.0
# display labels for the table / legend
LABEL = {"abc": "ABC", "mcmc": "MCMC", "snpe": "SNPE (CPU)", "snpe_gpu": "SNPE (GPU)"}

CONFIG = {
    "abc": dict(n_simulations=20_000, quantile=0.01, n_jobs=8, seed=0),
    "mcmc": dict(nsteps=300, burnin=100, thin=2, seed=0),
    "snpe": dict(num_rounds=1, num_simulations=2000, num_samples=4000, seed=0),
    # SNPE on GPU, two sequential rounds (the simulator injects white Gaussian noise per draw).
    # Truncated/restricted SNPE: with redshift + temperature floors pinned to razor-thin prior dims,
    # vanilla 2-round SNPE-C leaks proposal samples outside the box (NaN prior eval); the restricted
    # scheme trains on in-support draws each round and avoids that.
    "snpe_gpu": dict(num_rounds=2, num_simulations=2000, num_samples=4000, seed=0, device="cuda",
                     proposal_mode="restricted", support_samples=5000),
}


def _paths(mode, sampler):
    base = os.path.join(FIGDIR, f"kilonova_bench_{mode}_{sampler}")
    return base + ".json", base + ".npz"


def mag_chi2(obs_flux, obs_flux_err, sim_flux, bands=None):
    """Magnitude-space χ² distance for ABC (converts the flux it is handed back to AB mag)."""
    of = np.asarray(obs_flux, float)
    sf = np.maximum(np.asarray(sim_flux, float), 1e-300)
    me = (2.5 / np.log(10.0)) * np.asarray(obs_flux_err, float) / of
    return float(np.sum(((-2.5 * np.log10(of / AB) + 2.5 * np.log10(sf / AB)) / me) ** 2))


def setup():
    lc = wp.load_lightcurve(DATA, explosion_date=57982.0, min_snr=3, bands=BANDS,
                            redshift=Z_AT).add_flux()
    prior = Prior({
        "mej_1": Uniform(1e-4, 0.1), "vej_1": Uniform(0.01, 0.7), "kappa_1": Uniform(0.1, 0.5),
        "mej_2": Uniform(1e-4, 0.1), "vej_2": Uniform(0.01, 0.7), "kappa_2": Uniform(1.0, 30.0),
        "temperature_floor_1": Uniform(T_FLOOR - 1.0, T_FLOOR + 1.0),
        "temperature_floor_2": Uniform(T_FLOOR - 1.0, T_FLOOR + 1.0),
        "redshift": Uniform(Z_AT - 1e-5, Z_AT + 1e-5),
    })
    return lc, prior


def fit(mode, sampler):
    lc, prior = setup()
    t0 = time.time()
    if sampler == "abc":
        kw = dict(CONFIG["abc"])
        if mode == "magnitude":
            kw["distance"] = mag_chi2
        res = wp.fit_ABC(lc, MODEL, prior=prior, **kw)
    elif sampler == "mcmc":
        res = wp.fit_MCMC(lc, MODEL, prior=prior, space=mode, **CONFIG["mcmc"])
    else:                                                   # snpe (CPU) or snpe_gpu (CUDA, 2 rounds)
        res = wp.fit_SNPE(lc, MODEL, prior=prior, space=mode, **CONFIG[sampler])
    wall = time.time() - t0

    m = get_model(MODEL)
    best = {k: float(res.best_params[k]) for k in list(prior.names)}
    t, band = np.asarray(lc.time, float), np.asarray(lc.band).astype(str)
    mag = np.asarray(lc.magnitude, float)
    pmag = -2.5 * np.log10(np.maximum(m.predict(best, t, band), 1e-300) / AB)
    rms = float(np.sqrt(np.nanmean((pmag - mag) ** 2)))

    jpath, npath = _paths(mode, sampler)
    json.dump({
        "mode": mode, "sampler": sampler, "best": best,
        "summary": {k: {kk: float(vv) for kk, vv in res.summary[k].items()} for k in FREE},
        "aic": float(res.aic), "bic": float(res.bic), "max_loglik": float(res.max_log_likelihood),
        "rms_mag": rms, "n_samples": int(res.n_samples),
        "runtime_s": float(res.runtime_s), "wall_s": float(wall),
    }, open(jpath, "w"), indent=2)
    np.savez(npath, free=np.array(FREE), samples=res.samples[FREE].to_numpy(dtype=float),
             time=t, band=band, mag=mag, mag_err=np.asarray(lc.magnitude_err, float))
    print(f"[{mode:9s} {sampler:4s}] AIC={res.aic:8.1f}  RMS={rms:.3f} mag  "
          f"runtime={res.runtime_s:6.1f}s  n={res.n_samples}")


# ----------------------------------------------------------------------------- publication report ---
_PUB_RC = {
    "font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16, "figure.titlesize": 18,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "axes.linewidth": 1.2, "lines.linewidth": 2.0, "text.usetex": False,
    "mathtext.fontset": "dejavusans", "font.family": "DejaVu Sans", "savefig.dpi": 200,
}
SAMPLER_LS = {"abc": (0, (6, 4)), "mcmc": "-", "snpe": (0, (1, 1.4)), "snpe_gpu": (0, (4, 1.5, 1, 1.5))}
BAND_COL = {"g": "#009E73", "r": "#D55E00", "i": "#785EF0"}   # Okabe–Ito (colourblind-safe)


def _load_all():
    out = {}
    for mode in MODES:
        for sampler in SAMPLERS:
            jpath, npath = _paths(mode, sampler)
            if os.path.exists(jpath):
                d = json.load(open(jpath))
                d["_npz"] = np.load(npath, allow_pickle=True) if os.path.exists(npath) else None
                out[(mode, sampler)] = d
    return out


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = _load_all()
    if not res:
        raise SystemExit("no benchmark results found; run `fit <mode> <sampler>` first")
    ref = next(iter(res.values()))["_npz"]
    t, band, mag, err = ref["time"], ref["band"].astype(str), ref["mag"], ref["mag_err"]
    tgrid = np.linspace(max(0.2, t.min()), t.max(), 300)
    m = get_model(MODEL)

    # Pre-compute every model curve (redback toggles matplotlib LaTeX) BEFORE styling/plotting.
    curves = {}
    for key, d in res.items():
        curves[key] = {b: -2.5 * np.log10(m.predict(d["best"], tgrid, np.array([b] * tgrid.size)) / AB)
                       for b in BANDS}

    with plt.rc_context(_PUB_RC):
        fig = plt.figure(figsize=(15.5, 11))
        gs = fig.add_gridspec(2, 2, height_ratios=[2.05, 1.0], hspace=0.28, wspace=0.07)
        ax_f = fig.add_subplot(gs[0, 0])
        ax_m = fig.add_subplot(gs[0, 1], sharey=ax_f, sharex=ax_f)

        for ax, mode, title in [(ax_f, "flux", "Flux-space likelihood"),
                                (ax_m, "magnitude", "Magnitude-space likelihood")]:
            for b in BANDS:
                sel = band == b
                ax.errorbar(t[sel], mag[sel], yerr=err[sel], fmt="o", ms=5, mfc="white",
                            mec=BAND_COL[b], ecolor=BAND_COL[b], elinewidth=1.0, capsize=2,
                            mew=1.2, zorder=3, label=f"{b}-band data")
            for sampler in SAMPLERS:
                if (mode, sampler) not in res:
                    continue
                for b in BANDS:
                    ax.plot(tgrid, curves[(mode, sampler)][b], ls=SAMPLER_LS[sampler],
                            color=BAND_COL[b], lw=2.0, alpha=0.95, zorder=2)
            ax.set_title(title, pad=8)
            ax.set_xlabel("Time since explosion  [days]")
            ax.grid(alpha=0.25, lw=0.6)

        ax_f.invert_yaxis()                                  # single invert (shared y)
        ax_f.set_ylabel("Apparent AB magnitude")
        plt.setp(ax_m.get_yticklabels(), visible=False)

        # two clear legends: bands (colour) and samplers (line style)
        from matplotlib.lines import Line2D
        band_handles = [Line2D([], [], color=BAND_COL[b], marker="o", mfc="white", mew=1.2, ls="",
                               label=f"{b} band") for b in BANDS]
        samp_handles = [Line2D([], [], color="0.25", ls=SAMPLER_LS[s], lw=2.2, label=LABEL[s])
                        for s in SAMPLERS]
        leg1 = ax_f.legend(handles=band_handles, loc="upper right", frameon=True, framealpha=0.9,
                           title="Photometry")
        ax_f.add_artist(leg1)
        ax_f.legend(handles=samp_handles, loc="lower left", frameon=True, framealpha=0.9,
                    title="Sampler (best fit)")

        # ---- metrics + best-parameter table ----
        ax_t = fig.add_subplot(gs[1, :])
        ax_t.axis("off")
        col = ["Mode", "Sampler", r"$M_{\rm ej,1}$", r"$v_{\rm ej,1}$", r"$\kappa_1$",
               r"$M_{\rm ej,2}$", r"$v_{\rm ej,2}$", r"$\kappa_2$",
               "AIC", "RMS [mag]", "runtime [s]", r"$N_{\rm post}$"]
        rows, cell_colours = [], []
        for mode in MODES:
            for sampler in SAMPLERS:
                d = res.get((mode, sampler))
                if d is None:
                    continue
                b = d["best"]
                rows.append([
                    mode, LABEL[sampler],
                    f"{b['mej_1']:.3f}", f"{b['vej_1']:.3f}", f"{b['kappa_1']:.2f}",
                    f"{b['mej_2']:.3f}", f"{b['vej_2']:.3f}", f"{b['kappa_2']:.2f}",
                    f"{d['aic']:.0f}", f"{d['rms_mag']:.3f}", f"{d['runtime_s']:.1f}",
                    f"{d['n_samples']}"])
                shade = "#eaf2f8" if mode == "flux" else "#fdf2e9"
                cell_colours.append([shade] * len(col))
        tbl = ax_t.table(cellText=rows, colLabels=col, cellColours=cell_colours,
                         loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#b0b0b0")
            if r == 0:
                cell.set_facecolor("#34495e")
                cell.set_text_props(color="white", weight="bold")
        ax_t.set_title("Best-fit ejecta parameters and metrics per configuration",
                       fontsize=15, pad=14, weight="bold")

        fig.suptitle("WHISPER benchmark — two_component_kilonova on AT2017GFO: "
                     "flux- vs magnitude-space inference", y=0.97, weight="bold")
        out = os.path.join(FIGDIR, "kilonova_benchmark_report.png")
        fig.savefig(out, bbox_inches="tight")
        print("saved", out)

    # console benchmark table
    print(f"\n{'config':18s}{'AIC':>10s}{'RMS[mag]':>10s}{'runtime[s]':>12s}{'N':>8s}")
    for mode in MODES:
        for sampler in SAMPLERS:
            d = res.get((mode, sampler))
            if d:
                print(f"{mode+':'+sampler:18s}{d['aic']:>10.1f}{d['rms_mag']:>10.3f}"
                      f"{d['runtime_s']:>12.1f}{d['n_samples']:>8d}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "fit" and len(args) == 3:
        fit(args[1], args[2])
    elif args and args[0] == "plot":
        plot()
    else:
        raise SystemExit("usage: benchmark_kilonova_modes.py fit <flux|magnitude> <abc|mcmc|snpe> | plot")
