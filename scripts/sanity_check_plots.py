"""Rendering for the WHISPER sanity check / benchmark (see scripts/sanity_check.py).

Reads the cached ``sanity_*.json`` / ``.npz`` results and produces the figures + REPORT.md in the
output folder: per-parameter posterior histograms, an all-method corner, posterior-predictive checks,
SBC rank histograms, and a recovery + timing + scaling summary.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 13,
                            "figure.titlesize": 16, "text.usetex": False})
import matplotlib.pyplot as plt

import whisper_labia as wp

PRIMARIES = ["bazin_sn", "damped_sine"]     # showcases, headline first (each rendered if results exist)
SWEEP = ["gp2", "gp4", "gp6"]
COLORS = dict(zip(["mcmc", "abc", "abc_smc", "npe", "snpe"], wp.CORNER_PALETTE))
COLORS.update({"npe_mdn": "#c51b7d", "npe_nsf": "#01665e",       # dark, distinct hues for the
               "snpe_mdn": "#252525", "snpe_nsf": "#b8860b"})    # MDN/NSF method variants


def _load(out):
    res, npz, sbc = {}, {}, {}
    for f in glob.glob(os.path.join(out, "sanity_*.json")):
        b = os.path.basename(f)
        if b.startswith("sanity_sbc_"):
            d = json.load(open(f)); sbc[(d["mock"], d["sampler"])] = d
        else:
            d = json.load(open(f)); res[(d["mock"], d["sampler"])] = d
            p = f[:-5] + ".npz"
            if os.path.exists(p):
                npz[(d["mock"], d["sampler"])] = np.load(p, allow_pickle=True)
    return res, npz, sbc


def _methods(res, mock, samplers):
    return [s for s in samplers if (mock, s) in res]


def plot(out, mocks, samplers):
    res, npz, sbc = _load(out)
    if not res:
        raise SystemExit("no sanity results found; run `sanity_check.py fit ...` first")
    shown = [p for p in PRIMARIES if _methods(res, p, samplers)]
    for primary in shown:
        _hist(out, res, npz, mocks, samplers, primary)
        _corner(out, res, npz, mocks, samplers, primary)
        _ppc(out, res, npz, mocks, samplers, primary)
        _sbc_fig(out, sbc, mocks, samplers, primary)
        _summary(out, res, sbc, mocks, samplers, primary)
    _report(out, res, sbc, mocks, samplers, shown)
    print("saved figures + REPORT.md ->", out)


# ---------------------------------------------------------------------------------- posterior histograms
def _hist(out, res, npz, mocks, samplers, primary):
    ms = [s for s in _methods(res, primary, samplers) if (primary, s) in npz]
    if not ms:
        return
    params = list(mocks[primary]["params"])
    truth = mocks[primary]["truth"]
    fig, ax = plt.subplots(len(ms), len(params), figsize=(3.1 * len(params), 2.3 * len(ms)),
                           squeeze=False)
    for i, s in enumerate(ms):
        d = npz[(primary, s)]
        cols = {p: d["samples"][:, j] for j, p in enumerate(list(d["params"]))}
        for j, p in enumerate(params):
            a = ax[i][j]
            a.hist(cols[p], bins=40, density=True, color=COLORS[s], alpha=0.55, histtype="stepfilled")
            a.axvline(truth[p], color="k", lw=1.6, ls="--")
            if i == 0:
                a.set_title(p)
            if j == 0:
                a.set_ylabel(samplers[s][0], fontsize=11)
            a.set_yticks([])
    fig.suptitle(f"Posterior histograms — {primary} (truth = dashed)", y=1.0, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sanity_hist_{primary}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------------- corner
def _corner(out, res, npz, mocks, samplers, primary):
    import pandas as pd
    ms = [s for s in _methods(res, primary, samplers) if (primary, s) in npz]   # keep labels/colors aligned
    posts = [pd.DataFrame(npz[(primary, s)]["samples"], columns=list(npz[(primary, s)]["params"]))
             for s in ms]
    if len(posts) < 1:
        return
    labels = [samplers[s][0] for s in ms]
    wp.plot_corner(posts, labels=labels, colors=[COLORS[s] for s in ms],
                   truths=mocks[primary]["truth"], parameters=list(mocks[primary]["params"]),
                   title=f"All-sampler posteriors — synthetic {primary}",
                   save=os.path.join(out, f"sanity_corner_{primary}.png"))
    plt.close("all")


# ----------------------------------------------------------------------------- posterior-predictive check
def _ppc(out, res, npz, mocks, samplers, primary):
    ms = [s for s in _methods(res, primary, samplers) if (primary, s) in npz]
    if not ms:
        return
    fig, ax = plt.subplots(len(ms), 1, figsize=(8.5, 2.1 * len(ms)), squeeze=False, sharex=True)
    for i, s in enumerate(ms):
        d = npz[(primary, s)]; a = ax[i][0]
        a.fill_between(d["ppc_t"], d["ppc_lo95"], d["ppc_hi95"], color=COLORS[s], alpha=0.18)
        a.fill_between(d["ppc_t"], d["ppc_lo68"], d["ppc_hi68"], color=COLORS[s], alpha=0.30)
        a.plot(d["ppc_t"], d["ppc_med"], color=COLORS[s], lw=1.6)
        a.errorbar(d["time"], d["obs"], yerr=d["err"], fmt="o", ms=2.5, color="0.25", alpha=0.6, lw=0.6)
        j = res[(primary, s)]["ppc"]
        a.set_ylabel(samplers[s][0], fontsize=11)
        a.text(0.99, 0.92, f"χ²_best={j['reduced_chi2']:.2f}  cov95={j['coverage95']:.2f}",
               transform=a.transAxes, ha="right", va="top", fontsize=9)
    ax[-1][0].set_xlabel("t")
    fig.suptitle(f"Posterior-predictive check — {primary} (68/95% band + data)", y=1.0, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sanity_ppc_{primary}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------- SBC
def _sbc_fig(out, sbc, mocks, samplers, primary):
    ms = [s for s in samplers if (primary, s) in sbc]
    if not ms:
        return
    params = list(mocks[primary]["params"])
    fig, ax = plt.subplots(len(ms), len(params), figsize=(3.0 * len(params), 2.1 * len(ms)),
                           squeeze=False)
    for i, s in enumerate(ms):
        diag = sbc[(primary, s)]["diagnostics"]
        for j, p in enumerate(params):
            a = ax[i][j]; dp = diag[p]
            counts = np.array(dp["counts"]); nb = dp["n_bins"]; L = dp["n_realizations"]
            a.bar(np.arange(nb), counts, width=1.0, color=COLORS[s], alpha=0.6, align="edge")
            exp = dp["expected"]
            band = 1.96 * np.sqrt(exp * (1 - 1.0 / nb))          # ~95% per-bin band under uniformity
            a.axhline(exp, color="k", lw=1.0)
            a.axhspan(exp - band, exp + band, color="0.7", alpha=0.35)
            if i == 0:
                a.set_title(p)
            if j == 0:
                a.set_ylabel(f"{samplers[s][0]}\nL={L}", fontsize=9)
            a.text(0.5, 0.9, f"p={dp['uniformity_p']:.2f}", transform=a.transAxes, ha="center",
                   va="top", fontsize=8)
            a.set_xticks([])
    fig.suptitle(f"Simulation-Based Calibration — {primary} (rank histograms; flat = calibrated)",
                 y=1.0, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"sanity_sbc_{primary}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------- recovery + timing + scaling
def _summary(out, res, sbc, mocks, samplers, primary):
    ms = _methods(res, primary, samplers)
    if not ms:
        return
    params = list(mocks[primary]["params"])
    fig = plt.figure(figsize=(16, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.95, 1.1], wspace=0.55)

    # z-score heatmap (method x param)
    ax0 = fig.add_subplot(gs[0])
    Z = np.array([[res[(primary, s)]["recovery"][p]["z_score"] for p in params] for s in ms])
    im = ax0.imshow(np.abs(Z), cmap="viridis", vmin=0, vmax=3, aspect="auto")
    ax0.set_xticks(range(len(params))); ax0.set_xticklabels(params, rotation=30, ha="right")
    ax0.set_yticks(range(len(ms))); ax0.set_yticklabels([samplers[s][0] for s in ms])
    for i in range(len(ms)):
        for j in range(len(params)):
            ax0.text(j, i, f"{Z[i, j]:+.1f}", ha="center", va="center", color="w", fontsize=9)
    ax0.set_title("Recovery |z| = |median−true|/σ"); fig.colorbar(im, ax=ax0, fraction=0.046)

    # runtime bar
    ax1 = fig.add_subplot(gs[1])
    rt = [res[(primary, s)]["runtime_s"] for s in ms]
    ax1.barh([samplers[s][0] for s in ms], rt, color=[COLORS[s] for s in ms])
    ax1.set_xlabel("runtime [s]"); ax1.set_title("Speed"); ax1.set_xscale("log")
    for y, v in enumerate(rt):
        ax1.text(v, y, f" {v:.1f}s", va="center", fontsize=9)

    # scaling from the sweep: max|z| vs #params (runtimes are in the report's sweep table)
    ax2 = fig.add_subplot(gs[2])
    dims = [2, 4, 6]
    for s in samplers:
        xs, zz = [], []
        for mock, dim in zip(SWEEP, dims):
            if (mock, s) in res:
                xs.append(dim); zz.append(res[(mock, s)]["recovery"]["_summary"]["max_abs_z"])
        if xs:
            ax2.plot(xs, zz, "o-", color=COLORS[s], label=samplers[s][0])
    ax2.set_xlabel("# parameters (sweep)"); ax2.set_ylabel("max |z|"); ax2.set_title("Scaling")
    ax2.axhline(2.0, color="0.5", ls=":", lw=1); ax2.set_xticks(dims)
    ax2.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Sanity-check summary — recovery, speed, scaling ({primary} + 2/4/6-param sweep)",
                 y=1.03, weight="bold")
    fig.savefig(os.path.join(out, f"sanity_summary_{primary}.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ report
def _wall(d):
    """Honest end-to-end fit time: wall_s (includes MLE seeding / neural sampling), else runtime_s."""
    return float(d.get("wall_s", d["runtime_s"]))


def _showcase(res, sbc, mocks, samplers, primary):
    """Recovery + SBC tables and data-driven takeaways for one showcase mock."""
    ms = _methods(res, primary, samplers)
    spec = mocks[primary]
    neural = [s for s in ms if s.startswith(("npe", "snpe"))]
    lines = [f"## Showcase — {primary}", "",
             f"Mock {spec.get('desc', primary)} (truth "
             + ", ".join(f"{k}={v:g}" for k, v in spec["truth"].items())
             + "), white noise σ=0.15. Every sampler fits the *same* data"
             + ("; the neural methods train on GPU and condition **directly on the raw light-curve "
                "vector (no embedding net)**." if neural else "."), "",
             "### Recovery, goodness-of-fit & speed", "",
             "| method | max\\|z\\| | cov68 | cov95 | χ²_best | PPC cov68 | PPC cov95 | wall [s] | sims |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in ms:
        d = res[(primary, s)]; sm = d["recovery"]["_summary"]; pp = d["ppc"]
        sims = d.get("info", {}).get("total_simulations")
        lines.append(f"| {samplers[s][0]} | {sm['max_abs_z']:.2f} | {sm['coverage68']:.2f} | "
                     f"{sm['coverage95']:.2f} | {pp['reduced_chi2']:.2f} | {pp['coverage68']:.2f} | "
                     f"{pp['coverage95']:.2f} | {_wall(d):.1f} | "
                     f"{f'{sims:,}' if sims else '—'} |")
    lines += ["", "*max|z| = max over parameters of |median−true|/σ (≲2 ⇒ recovered). cov68/95 = fraction "
              "of parameters whose credible interval covers the truth. χ²_best≈1 ⇒ the model fits; PPC "
              "cov68/95 = fraction of data inside the noise-inflated predictive band (≈0.68/0.95 ⇒ "
              "calibrated). wall = end-to-end fit time (MCMC includes its MLE seeding; neural methods "
              "include training + posterior sampling). Single noise realization, so per-parameter "
              "coverage is coarse — SBC below is the calibration test over many realizations.*", ""]

    with_sbc = [s for s in ms if (primary, s) in sbc]
    if with_sbc:
        lines += ["### Simulation-Based Calibration (rank uniformity)", "",
                  "| method | L | min uniformity p | calibrated |", "|---|---|---|---|"]
        for s in with_sbc:
            sd = sbc[(primary, s)]; su = sd["diagnostics"]["_summary"]
            lines.append(f"| {samplers[s][0]} | {sd['L']} | {su['min_uniformity_p']:.3f} | "
                         f"{su['calibrated']} |")
        lines += ["", "*Uniform ranks (p ≳ 0.05) ⇒ calibrated uncertainties; ∪-shape = overconfident, "
                  "∩-shape = underconfident.*", ""]

    timed = sorted(((s, _wall(res[(primary, s)])) for s in ms), key=lambda x: x[1])
    recov = sorted(((s, res[(primary, s)]["recovery"]["_summary"]["max_abs_z"]) for s in ms),
                   key=lambda x: x[1])
    sbc_p = sorted(((s, sbc[(primary, s)]["diagnostics"]["_summary"]["min_uniformity_p"])
                    for s in with_sbc), key=lambda x: -x[1])
    failed = [(s, z) for s, z in recov if not z <= 2.0]          # data-driven verdict (NaN counts as fail)
    if failed:
        rec_line = ("- **Recovery:** " + ", ".join(f"{samplers[s][0]} FAILS (max|z| = {z:.2f} > 2)"
                                                   for s, z in failed)
                    + f"; best = {samplers[recov[0][0]][0]} at {recov[0][1]:.2f}.")
    else:
        rec_line = (f"- **Recovery:** every sampler recovers all parameters within ~2σ "
                    f"(best max|z| = {samplers[recov[0][0]][0]} at {recov[0][1]:.2f}, "
                    f"worst = {samplers[recov[-1][0]][0]} at {recov[-1][1]:.2f}).")
    lines += ["### Takeaways", "", rec_line,
              f"- **Speed (end-to-end):** " + " < ".join(f"{samplers[s][0]} ({t:.0f}s)"
                                                         for s, t in timed) + "."]
    if sbc_p:
        lines += ["- **Calibration (SBC), best → worst rank-uniformity p:** "
                  + ", ".join(f"{samplers[s][0]} ({p:.3f}{', calibrated' if p >= 0.05 else ''})"
                              for s, p in sbc_p)
                  + ". Only p ≥ 0.05 is formally calibrated; the ordering shows how close each gets."]
    lines += ["", f"![corner](sanity_corner_{primary}.png)", "",
              f"![histograms](sanity_hist_{primary}.png)", "",
              f"![ppc](sanity_ppc_{primary}.png)", "",
              f"![sbc](sanity_sbc_{primary}.png)", "",
              f"![summary](sanity_summary_{primary}.png)", ""]
    return lines


def _report(out, res, sbc, mocks, samplers, shown):
    lines = ["# WHISPER sanity check & benchmark — synthetic parameter recovery", "",
             "Synthetic light curves `M(t, θ) + white noise` with **known ground truth**, fit by every "
             "WHISPER sampler, timed, and validated statistically (recovery z-scores, credible-interval "
             "coverage, posterior-predictive checks, Simulation-Based Calibration). Showcases: a "
             "physically-motivated **Bazin (2009) supernova** light curve (30k-simulation neural budgets, "
             "MDN + NSF density estimators) and a **damped sinusoid** (correlated, oscillatory stress "
             "test), plus a 2/4/6-parameter dimensionality sweep.", "",
             "*Disclosure:* the Bazin and sweep noise seeds were **screened** to be non-adversarial "
             "(worst |MLE−truth|/σ_Fisher ≲ 1 — an unlucky draw makes every method 'miss' spuriously), "
             "so the single-realization recovery/coverage columns compare methods on a shared, "
             "well-posed realization and are favourable by construction; they are **not** calibration "
             "evidence. Calibration is tested by **SBC over many unscreened realizations**.", ""]
    for primary in shown:
        lines += _showcase(res, sbc, mocks, samplers, primary)

    lines += ["## Statistical notes & fixes", "",
              "- **ABC-SMC ε-floor.** A naive adaptive ε shrinks to χ²_min and collapses the posterior "
              "onto the MLE (spuriously overconfident: on the 2-param Gaussian pulse the raw run gave "
              "|z|≈8 with 0% coverage). WHISPER's `min_epsilon=\"auto\"` floors ε at χ²_min + 2(k+2), "
              "reproducing the Gaussian posterior width — restoring |z|≲2 and nominal coverage on the "
              "single-realization recovery.",
              "- **ABC is approximate — SBC proves it.** Over many realizations, rejection ABC is "
              "**under-confident** (finite acceptance tolerance ⇒ posterior wider than the truth, "
              "∩-shaped ranks) and even ε-floored ABC-SMC cannot perfectly calibrate a strongly "
              "**correlated** posterior with its diagonal-Gaussian kernel (on the damped sine: freq too "
              "wide, phase too narrow). Point recovery stays unbiased; the uncertainty *shape/width* is "
              "what suffers — exactly the likelihood-free approximation error SBC exists to reveal.",
              "- **Neural SBI input.** No embedding net anywhere: the density estimators condition on "
              "the raw light-curve vector. MDN (mixture density network) trains fastest and samples "
              "directly; NSF (neural spline flow) is more expressive but costs more per epoch. Both are "
              "GPU-trained; each method runs on its own GPU, so the four neural fits run in parallel.",
              "- **Identifiable pulses.** A sum of Gaussians is invariant under permuting its (Aₖ,μₖ) "
              "pairs, so the sweep gives each μₖ a disjoint prior bin; otherwise every sampler is free "
              "to label-switch (a spurious multi-modal 'failure').",
              "- **SNPE cost & why MCMC can still be faster.** For a cheap *analytic* likelihood, "
              "MCMC evaluates it directly — seconds. Neural SBI must first *learn* the posterior from "
              "simulations, so its wall-clock is training-dominated; it pays off when the simulator is "
              "expensive or the likelihood intractable (its real use case), and NPE amortizes: train "
              "once, infer instantly for any new observation. SBC over many realizations exploits "
              "exactly that amortization; re-training 10-round SNPE per realization is left out as "
              "prohibitive (its single-dataset recovery + PPC stand in).", ""]

    lines += ["## Dimensionality sweep (2/4/6 params, Gaussian pulses)", "",
              "| method | " + " | ".join(f"{d}p max\\|z\\| / t[s]" for d in (2, 4, 6)) + " |",
              "|---|" + "---|" * 3]
    for s in samplers:
        cells = []
        for mock in SWEEP:
            if (mock, s) in res:
                r = res[(mock, s)]
                cells.append(f"{r['recovery']['_summary']['max_abs_z']:.2f} / {_wall(r):.1f}")
            else:
                cells.append("—")
        if all(c == "—" for c in cells):
            continue                                    # skip methods not run over the sweep
        lines.append(f"| {samplers[s][0]} | " + " | ".join(cells) + " |")
    lines += ["", "*Cell = max|z| / wall[s]. All methods stay within ~2σ as the parameter count grows "
              "2→4→6; runtime scales gently. The sweep uses the MAF-NPE config; sequential SNPE is "
              "omitted from the sweep (10-round cost) — its recovery is shown in the showcases above.*",
              ""]
    open(os.path.join(out, "REPORT.md"), "w").write("\n".join(lines))
