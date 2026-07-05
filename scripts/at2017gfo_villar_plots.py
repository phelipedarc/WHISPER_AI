"""Rendering for the Villar+17 AT2017GFO analysis (see scripts/at2017gfo_villar.py).

Reads the cached ``villar_*.json`` / ``.npz`` results and produces: per-parameter posterior
histograms annotated with median ± CI, an all-method corner, per-method posterior-predictive light
curves (magnitude space, 3 bands), a parameter/runtime summary figure, and
``docs/REPORT_at2017gfo_villar.md``.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                            "figure.titlesize": 15, "text.usetex": False})
import matplotlib.pyplot as plt

import whisper_labia as wp

_BAND_FIXED = {"g": "#2ca02c", "r": "#d62728", "i": "#8c564b"}


def _band_colors(bands):
    """Per-band colours: the g/r/i palette for the optical-only run, else a UV->NIR gradient
    (bands are ordered blue->red, so a spectral colormap reads as increasing wavelength)."""
    if all(b in _BAND_FIXED for b in bands):
        return {b: _BAND_FIXED[b] for b in bands}
    import matplotlib.cm as cm
    cmap = cm.get_cmap("turbo")
    n = max(len(bands) - 1, 1)
    return {b: cmap(0.04 + 0.92 * k / n) for k, b in enumerate(bands)}
COLORS = dict(zip(["mcmc", "abc", "abc_smc", "npe_mdn", "npe_nsf", "snpe5_nsf", "snpe5_tcn"],
                  ["#08306b", "#a50026", "#006d2c", "#c51b7d", "#01665e", "#4d004b", "#00441b"]))
LOGPAR = {"mej_1", "mej_2", "temperature_floor_1", "temperature_floor_2", "sigma"}
SHORT = {"mcmc": "MCMC", "abc": "ABC", "abc_smc": "ABC-SMC", "npe_mdn": "NPE-MDN",
         "npe_nsf": "NPE-NSF", "snpe5_nsf": "SNPE-5r", "snpe5_tcn": "SNPE-5r+TCN"}
VILLAR17 = {  # Villar et al. 2017 (ApJL 851, L21) Table 2, "2-Comp" fit (kappa_blue=0.5 fixed) —
    # blue = low-opacity component 1, red = high-opacity component 2 (matches this setup exactly).
    "mej_1": 0.023, "vej_1": 0.256, "temperature_floor_1": 3983,
    "mej_2": 0.050, "vej_2": 0.149, "kappa_2": 3.65, "temperature_floor_2": 1151,
    "sigma": 0.256}


def _load(out):
    res, npz = {}, {}
    for f in sorted(glob.glob(os.path.join(out, "villar_*.json"))):
        d = json.load(open(f))
        res[d["method"]] = d
        p = f[:-5] + ".npz"
        if os.path.exists(p):
            npz[d["method"]] = np.load(p, allow_pickle=True)
    return res, npz


def _fmt(v, lo, hi):
    """median +err/-err with sensible sig figs."""
    err = max(hi - v, v - lo)
    dec = max(0, 2 - int(np.floor(np.log10(err))) if err > 0 else 3)
    return f"{v:.{dec}f}^{{+{hi - v:.{dec}f}}}_{{-{v - lo:.{dec}f}}}"


def plot(out, samplers, params, labels, bands):
    res, npz = _load(out)
    if not res:
        raise SystemExit("no villar results found; run `at2017gfo_villar.py fit ...` first")
    ms = [m for m in samplers if m in res]
    allp = params + ["sigma"]

    # ---------------- histograms: rows = parameters, columns = methods, annotated median ± CI ----
    from matplotlib.ticker import LogLocator, MaxNLocator
    # log x-axis for scale parameters: temperature floors always, and σ too (it spans orders of
    # magnitude in flux space [Jy] and has a LogUniform prior). Temp floors also get the physical
    # LogUniform(100, 6000) range clip; σ is not clipped.
    tfloor = {"temperature_floor_1", "temperature_floor_2"}
    logx = tfloor | {"sigma"}

    # ONE shared x-range per physical variable (union of the methods' posterior bulk), with matching
    # bin edges, so every row is directly comparable across methods.
    xr, xbins = {}, {}
    for p in allp:
        pooled = np.concatenate([npz[m]["samples"][:, list(npz[m]["params"]).index(p)]
                                 for m in ms if p in list(npz[m]["params"])])
        if p in logx:
            lo, hi = np.percentile(pooled[pooled > 0], [1.0, 99.0])
            if p in tfloor:                       # clip to the physical prior range (neural tails leak)
                lo, hi = max(lo, 90.0), min(hi, 6200.0)
            xr[p] = (lo * 0.85, hi * 1.15)
            xbins[p] = np.logspace(np.log10(xr[p][0]), np.log10(xr[p][1]), 37)
        else:
            lo, hi = np.percentile(pooled, [1.0, 99.0])
            pad = (hi - lo) * 0.04 or max(abs(hi) * 0.02, 1e-3)
            xr[p] = (lo - pad, hi + pad)
            xbins[p] = np.linspace(xr[p][0], xr[p][1], 37)

    fig, ax = plt.subplots(len(allp), len(ms), figsize=(3.5 * len(ms), 2.4 * len(allp)),
                           squeeze=False)
    for j, m in enumerate(ms):
        d = npz[m]
        cols = {p: d["samples"][:, i] for i, p in enumerate(list(d["params"]))}
        for i, p in enumerate(allp):
            a = ax[i][j]
            islog = p in logx
            if islog:
                a.set_xscale("log")
            a.set_xlim(*xr[p])
            a.set_yticks([])
            if p not in cols:
                a.text(0.5, 0.5, "not fitted", transform=a.transAxes, ha="center", va="center",
                       fontsize=13, color="0.55", style="italic")
            else:
                x = cols[p]
                a.hist(x, bins=xbins[p], density=True, color=COLORS[m], alpha=0.68,
                       histtype="stepfilled")
                lo, med, hi = np.percentile(x, [16, 50, 84])
                a.axvline(med, color="k", lw=1.6)
                a.axvline(lo, color="k", lw=1.1, ls=":")
                a.axvline(hi, color="k", lw=1.1, ls=":")
                a.set_title(rf"${_fmt(med, lo, hi)}$", fontsize=12.5, pad=4)
            if islog:                                # a few clean decade ticks, no minor clutter
                a.xaxis.set_major_locator(LogLocator(numticks=5))
                a.xaxis.set_minor_locator(LogLocator(subs="auto", numticks=12))
            else:                                    # more ticks, edges pruned so neighbours don't touch
                a.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            a.tick_params(axis="x", labelsize=12.5, length=5)
            if i == 0:
                a.annotate(SHORT.get(m, samplers[m][0]), xy=(0.5, 1.46), xycoords="axes fraction",
                           ha="center", fontsize=15, weight="bold")
            if j == 0:
                a.set_ylabel(labels.get(p, p), fontsize=18)
    fig.suptitle("AT2017GFO — two-component kilonova posteriors (median ± 68% CI; shared x-range "
                 "per variable)", y=1.003, weight="bold", fontsize=18)
    fig.tight_layout(h_pad=2.8, w_pad=1.0)
    fig.savefig(os.path.join(out, "villar_hist.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- corner (log10 masses; all methods overlaid) ---------------------------------
    import pandas as pd
    posts, labs, cols_used = [], [], []
    show = [p for p in allp]
    for m in ms:
        d = npz[m]
        names = list(d["params"])
        df = pd.DataFrame(d["samples"], columns=names)
        for p in ("mej_1", "mej_2"):
            if p in df:
                df[p] = np.log10(df[p])
        keep = [p for p in show if p in df.columns]
        posts.append(df[keep])
        labs.append(samplers[m][0]); cols_used.append(COLORS[m])
    common = [p for p in show if all(p in df.columns for df in posts)]
    wp.plot_corner([df[common] for df in posts], labels=labs, colors=cols_used,
                   parameters=common,
                   title="AT2017GFO — all-sampler posterior (log10 ejecta masses)",
                   save=os.path.join(out, "villar_corner.png"))
    plt.close("all")

    # ---------------- PPC: one panel per method, 3-band model bands + data -----------------------
    # sharey=True -> ONE common magnitude axis for every panel; its range is set from the data (+the
    # median model curves) so the σ-inflated 95% band tails simply clip at the edges instead of
    # blowing the scale up to 30-40 mag and squashing the informative region.
    fig, ax = plt.subplots(len(ms), 1, figsize=(10.5, 3.6 * len(ms)), squeeze=False,
                           sharex=True, sharey=True)
    band_col = _band_colors(bands)                # g/r/i palette or a UV->NIR gradient
    ref = npz[ms[0]]
    t, band = ref["time"], ref["band"].astype(str)
    mag, err = ref["mag"], ref["mag_err"]
    # y-range from the DATA + median curves (not the σ-inflated tails), and tightened toward the data
    # so the many bands spread out vertically and are easy to tell apart; taller panels add resolution.
    med_all = np.concatenate([npz[m][f"curve_{b}"][1] for m in ms for b in bands])
    y_bright = min(float(np.min(mag - err)), float(np.percentile(med_all, 2))) - 0.4
    y_faint = max(float(np.max(mag + err)), float(np.percentile(med_all, 92))) + 0.4
    for i, m in enumerate(ms):
        a = ax[i][0]; d = npz[m]
        for b in bands:
            lo, med, hi = d[f"curve_{b}"]
            a.fill_between(d["tgrid"], lo, hi, color=band_col[b], alpha=0.20)
            a.plot(d["tgrid"], med, color=band_col[b], lw=1.3, label=b if i == 0 else None)
            sel = band == b
            a.errorbar(t[sel], mag[sel], yerr=err[sel], fmt="o", ms=2.6, color=band_col[b],
                       alpha=0.75, lw=0.7)
        j = res[m]["ppc"]
        a.set_ylim(y_faint, y_bright)             # inverted (faint at bottom); shared across panels
        a.set_ylabel(samplers[m][0], fontsize=10)
        a.text(0.99, 0.05, f"χ²/dof={j['chi2_reported']:.1f}  (+σ: {j['chi2_scatter']:.2f})  "
               f"cov95={j['cov95']:.2f}", transform=a.transAxes, ha="right", va="bottom", fontsize=9)
    ax[0][0].legend(loc="upper right", ncol=min(len(bands), 6), fontsize=8, frameon=False,
                    title="band (blue→red)", columnspacing=0.9, handletextpad=0.4)
    ax[-1][0].set_xlabel("days since merger (MJD 57982.53)")
    _bandlabel = "g/r/i" if all(b in _BAND_FIXED for b in bands) else "UV→optical→NIR (11 bands)"
    fig.suptitle(f"AT2017GFO posterior-predictive light curves — {_bandlabel} (95% model band + data)",
                 y=1.0, weight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "villar_ppc.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # ---------------- summary: medians vs methods + runtime ---------------------------------------
    # short method tags (module-level SHORT) keep the runtime-bar labels off the left panel
    fig = plt.figure(figsize=(15.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 0.85], wspace=0.30)
    ax0 = fig.add_subplot(gs[0])
    # physical params compare across spaces; σ is space-specific (mag vs Jy) so drop it from the
    # Villar-normalised panel for the flux run.
    show6 = ["mej_1", "vej_1", "mej_2", "vej_2", "kappa_2"]
    if not os.path.basename(out.rstrip("/")).endswith("_flux"):
        show6 = show6 + ["sigma"]
    xs = np.arange(len(show6))
    for k, m in enumerate(ms):
        d = res[m]["summary"]
        med = [d[p]["median"] if p in d else np.nan for p in show6]
        lo = [d[p]["median"] - d[p]["ci16"] if p in d else 0 for p in show6]
        hi = [d[p]["ci84"] - d[p]["median"] if p in d else 0 for p in show6]
        norm = [VILLAR17.get(p, np.nan) for p in show6]
        val = [m0 / n if np.isfinite(n) else m0 for m0, n in zip(med, norm)]
        e_lo = [l / n if np.isfinite(n) else l for l, n in zip(lo, norm)]
        e_hi = [h / n if np.isfinite(n) else h for h, n in zip(hi, norm)]
        ax0.errorbar(xs + 0.1 * (k - len(ms) / 2), val, yerr=[e_lo, e_hi], fmt="o", ms=4,
                     color=COLORS[m], label=SHORT.get(m, samplers[m][0]), lw=1.2, capsize=2)
    ax0.axhline(1.0, color="0.4", ls="--", lw=1)
    ax0.set_xticks(xs)
    ax0.set_xticklabels([labels.get(p, p) + ("\n(/V17)" if p in VILLAR17 else "") for p in show6])
    ax0.set_ylabel("median (÷ Villar+17 where available)")
    ax0.set_title("Parameter medians ± 68% CI across methods")
    # legend spans the bottom of the whole figure -> never overlaps either panel's data
    ax0.legend(loc="upper center", bbox_to_anchor=(0.72, -0.16), ncol=7, fontsize=8.5,
               frameon=False, handletextpad=0.3, columnspacing=1.1)

    ax1 = fig.add_subplot(gs[1])
    rt = [res[m]["wall_s"] for m in ms]
    y = np.arange(len(ms))
    ax1.barh(y, rt, color=[COLORS[m] for m in ms])
    ax1.set_yticks([])                               # no y-tick labels -> nothing can overflow left
    ax1.set_xlabel("wall [s]"); ax1.set_xscale("log"); ax1.set_title("End-to-end fit time")
    ax1.set_xlim(min(rt) * 0.6, max(rt) * 9)         # headroom for the name+time label past each bar
    for yi, m in zip(y, ms):                          # name + runtime together, just past the bar end
        ax1.text(rt[yi] * 1.18, yi, f"{SHORT.get(m, m)} · {rt[yi]:.0f}s", va="center", ha="left",
                 fontsize=8.5)
    fig.suptitle("AT2017GFO Villar+17 kilonova — summary", y=1.0, weight="bold")
    fig.savefig(os.path.join(out, "villar_summary.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    _report(out, res, samplers, params, labels, ms)
    print("saved figures + REPORT ->", out)


def _report(out, res, samplers, params, labels, ms):
    allp = params + ["sigma"]
    base = os.path.basename(out.rstrip("/"))
    full = "_full" in base
    space = "flux" if base.endswith("_flux") else "magnitude"
    space_desc = ("**flux space** (additive-flux scatter σ [Jy])" if space == "flux"
                  else "**apparent-magnitude space** (Villar+17; σ ≈ fractional-flux scatter [mag])")
    data_desc = ("**full UV → optical → NIR photometry** (11 bands, Swift-UV `uvw1` through 2MASS "
                 "`Ks`, SNR ≥ 3, 0–30 d)" if full else "g/r/i photometry (SNR ≥ 3)")
    lines = ["# AT2017GFO — Villar+2017-style two-component kilonova with WHISPER"
             + (" (full UVOIR" + (", flux space)" if space == "flux" else ")") if full else ""), "",
             "Real-data application: the redback `two_component_kilonova` model with "
             "**κ_blue = 0.5 cm²/g fixed**, redshift fixed (z = 0.00984), **κ_red and both "
             f"temperature floors free**, fit to the AT2017GFO {data_desc} in "
             f"{space_desc}. The likelihood-based and neural methods also fit the "
             "**Villar+17 extra-scatter term σ** (added in quadrature to the reported errors):", "",
             "$$\\ln\\mathcal{L} = -\\tfrac{1}{2}\\sum_i\\left[\\frac{(O_i-M_i)^2}"
             "{\\sigma_i^2+\\sigma^2} + \\ln\\big(2\\pi(\\sigma_i^2+\\sigma^2)\\big)\\right]$$", "",
             "*(the correctly normalized form of Villar et al. 2017, Eq. 4, as implemented in "
             "MOSFiT). The distance-based ABC family fits the 7 physical parameters only: a χ² "
             "rejection distance is monotonically penalised by extra simulation noise, so a "
             "noise-level parameter is not identifiable by distance-based ABC — verified on "
             "synthetic data.*", "",
             "## Posterior medians ± 68% CI", ""]
    hdr = "| parameter | " + " | ".join(samplers[m][0] for m in ms) + " |"
    lines += [hdr, "|" + "---|" * (len(ms) + 1)]
    for p in allp:
        row = [labels.get(p, p).replace("$", "")]
        for m in ms:
            s = res[m]["summary"].get(p)
            row.append("—" if s is None else
                       f"{s['median']:.4g} [+{s['ci84'] - s['median']:.2g} "
                       f"−{s['median'] - s['ci16']:.2g}]")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "*Reference — **Villar et al. 2017 (ApJL 851 L21), Table 2, 2-component fit** "
              "(κ_blue = 0.5 fixed, matching this setup): M_ej^blue = 0.023 M☉, v^blue = 0.256 c, "
              "T^blue = 3983 K, M_ej^red = 0.050 M☉, v^red = 0.149 c, κ_red = 3.65 cm²/g, "
              "T^red = 1151 K, σ = 0.256 mag (WAIC = −1030). Villar+17 fit a much larger UV–optical–NIR "
              "dataset with a radiative-transfer-calibrated model, so the absolute values are a "
              "literature anchor, not ground truth. The medians ÷ Villar+17 are compared in the "
              "summary figure below.*", "",
              "## Goodness-of-fit & cost", "",
              "| method | χ²/dof (reported σᵢ) | χ²/dof (σᵢ ⊕ σ) | PPC cov95 | wall [s] "
              "| per-object [s] | AIC |",
              "|---|---|---|---|---|---|---|"]
    for m in ms:
        d = res[m]; pp = d["ppc"]
        amort = d.get("amortized_s")
        per = (f"{amort:.2f}" if amort is not None and d.get("info", {}).get("num_rounds") == 1
               else f"{d['wall_s']:.0f}")
        lines.append(f"| {samplers[m][0]} | {pp['chi2_reported']:.1f} | {pp['chi2_scatter']:.2f} | "
                     f"{pp['cov95']:.2f} | {d['wall_s']:.0f} | {per} | {d['aic']:.0f} |")
    lines += ["", "*χ²/dof against the reported errors is ≫1 for every method — high-SNR kilonova "
              "photometry always carries model systematics beyond the measurement errors; that is "
              "exactly what σ absorbs: with the fitted scatter the χ²/dof (σᵢ ⊕ σ) is ≈1 and the "
              "predictive coverage is nominal. AIC values are comparable only among methods fitting "
              "the same parameter set (the ABC family omits σ).*", ""]

    # Data-driven interpretation. Rail detection: the posterior piles against a bound — judged by
    # proximity to the bound VALUE (not fraction of the prior range, which is misleading for wide
    # priors like κ∈[1,30] where κ=2 is a genuine constraint), and keyed on the reference exact
    # method (MCMC) rather than any single broad neural posterior.
    PRIOR_BOUNDS = {"mej_1": (1e-4, 0.1), "vej_1": (0.05, 0.3), "temperature_floor_1": (100, 6000),
                    "mej_2": (1e-4, 0.1), "vej_2": (0.05, 0.3), "kappa_2": (1.0, 30.0),
                    "temperature_floor_2": (100, 6000)}
    mcmc = res.get("mcmc", {}).get("summary", {})

    def _rails(p, summ=None):
        summ = summ if summ is not None else mcmc
        s = summ.get(p)
        if not s or p not in PRIOR_BOUNDS:
            return False
        lo, hi = PRIOR_BOUNDS[p]
        med = s["median"]
        return med <= lo * 1.15 or med >= hi * 0.92     # within ~15%/8% of the bound value

    red_railers = sorted(p for p in ("kappa_2", "vej_2", "temperature_floor_2") if _rails(p))
    blue_railers = sorted(p for p in ("vej_1", "mej_1", "temperature_floor_1") if _rails(p))
    vb = mcmc.get("vej_1", {}).get("median", float("nan"))
    kr = mcmc.get("kappa_2", {}).get("median", float("nan"))
    sig_mcmc = mcmc.get("sigma", {}).get("median", float("nan"))     # MCMC constrains σ best
    if space == "flux":
        sig_line = (f"- **The scatter term works.** MCMC recovers a flux-space extra scatter "
                    f"**σ ≈ {sig_mcmc:.2g} Jy** (∼ the per-point flux errors); the neural σ posteriors "
                    "are broader (a single light curve weakly constrains a noise level). Folding it in "
                    "quadrature turns the ")
    else:
        sig_line = (f"- **The scatter term works.** MCMC recovers an extra scatter "
                    f"**σ ≈ {sig_mcmc:.2f} mag**, in the ballpark of **Villar+2017's σ = "
                    f"{VILLAR17['sigma']:.3f} mag** (the neural σ posteriors run broader — a single "
                    "light curve weakly constrains a noise level). Folding it in quadrature turns the ")

    lines += ["## Interpretation", "",
              sig_line +
              "χ²/dof (vs reported errors) into ≈1 with nominal 95% predictive coverage — the excess is "
              "model systematics (a semi-analytic two-component kilonova can't capture every spectral "
              "feature), exactly what Villar+17 introduced σ to absorb.",
              "- **Blue component.** With κ_blue fixed at 0.5 the blue component is well-specified in "
              f"regime; MCMC gives v_ej^blue ≈ {vb:.2f} c"
              + (" — pushed to the fast edge of the physical prior (the optical decline wants fast blue "
                 "ejecta; the degeneracy only fully breaks with NIR)"
                 if "vej_1" in blue_railers else
                 f", consistent with Villar+2017 (v^blue = {VILLAR17['vej_1']:.3f} c, "
                 f"T^blue = {VILLAR17['temperature_floor_1']:.0f} K)") + ".",
              ("- **Red component — " + ("still edge-limited" if red_railers else "now constrained") +
               ".** " + (
                   ("κ_red is *free* and the lanthanide-rich red ejecta radiate mostly in the NIR; with "
                    f"the full UV–optical–NIR data the red parameters pull off the prior edges toward "
                    f"physical values (MCMC κ_red ≈ {kr:.1f} cm²/g vs Villar+2017's "
                    f"{VILLAR17['kappa_2']:.2f}). This is the payoff of adding the NIR bands the "
                    "optical-only fit lacked.")
                   if not red_railers else
                   ("κ_red is *free* and the red ejecta radiate mostly in the **NIR**, which this band "
                    f"set constrains weakly — so {', '.join(red_railers) or 'the red parameters'} rail "
                    "against their prior edges. Adding NIR coverage (the full-UVOIR run) is what "
                    "identifies them."))),
              f"- **MCMC vs simulation-based inference.** MCMC finds the sharp maximum-likelihood mode "
              f"(χ²/dof = {res['mcmc']['ppc']['chi2_reported']:.0f} vs reported errors, lowest AIC); "
              "the amortized/rejection samplers report a broader posterior bulk. They agree on the "
              "well-constrained quantities (blue ejecta, σ) and diverge where the data are least "
              "informative — the honest signature of a real-data fit.",
              "- **Amortized inference.** Once trained, NPE conditions a *new* AT2017GFO-like light "
              "curve in ~10–80 ms (the per-object column) versus a full refit for MCMC — the payoff of "
              "neural SBI when many objects share one model.", ""]
    fig_dir = "figures/" + os.path.basename(out.rstrip("/"))   # at2017gfo_villar[_full]
    lines += ["## Figures", "",
              "### Posterior histograms", "",
              "Per-parameter marginal posteriors (rows) for every method (columns), each annotated "
              "with its median ± 68% CI; each variable shares one x-range across methods for direct "
              "comparison. σ is *not fitted* by the distance-based ABC family.", "",
              f"![posterior histograms]({fig_dir}/villar_hist.png)", "",
              "### Corner plot", "",
              "Joint posteriors of all fitted parameters (ejecta masses shown as log₁₀), every method "
              "overlaid. The neural and ABC methods overlap in a broad central region while MCMC (dark "
              "blue) sits apart in its sharp, prior-edge MAP — the mode tension made visual, including "
              "the parameter correlations (e.g. M_ej^red–v_ej^red, κ_red–T_floor^red).", "",
              f"![corner plot]({fig_dir}/villar_corner.png)", "",
              "### Posterior-predictive light curves", "",
              "Each method's 95% posterior-predictive model band in g/r/i (coloured) over the "
              "AT2017GFO photometry, with the per-panel χ²/dof (vs reported errors and vs errors ⊕ σ) "
              "and 95% coverage. MCMC gives the tightest, best-tracking band; the neural methods carry "
              "wider bands reflecting the marginal σ uncertainty.", "",
              f"![posterior-predictive light curves]({fig_dir}/villar_ppc.png)", "",
              "### Summary — medians & runtime", "",
              "Parameter medians ± 68% CI across methods, each normalised to the Villar+2017 value "
              "where available (dashed line = Villar+17), and the end-to-end wall time per method.", "",
              f"![summary]({fig_dir}/villar_summary.png)", ""]
    report_name = "REPORT_" + os.path.basename(out.rstrip("/")) + ".md"   # ..._villar[_full].md
    path = os.path.join(os.path.dirname(out.rstrip("/")), "..", report_name)
    path = os.path.abspath(path)
    open(path, "w").write("\n".join(lines))
