"""Model-comparison report: ABC vs ABC-SMC for three simple models on AT2017GFO (r-band).

Fits {flare, bazin, gaussian_rise} x {ABC, ABC-SMC}, ranks by AIC, and writes
docs/REPORT_at2017gfo.md + a comparison figure + per-fit JSONs.

Run: docker exec phe_sbi python /tf/astrodados2/phelipedata2/WHISPER/whisper-labia/scripts/report_at2017gfo.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import whisper_labia as wp  # noqa: E402
from whisper_labia.models import get_model  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data" / "at2017gfo.csv"
FIGS = ROOT / "docs" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / "docs" / "REPORT_at2017gfo.md"

MODELS = ["flare", "bazin", "gaussian_rise"]


def make_prior(model, fmax):
    amp = wp.Uniform(0.0, 10 * fmax)
    if model == "flare":
        return wp.Prior({"amplitude": amp, "rise_time": wp.Uniform(0.05, 10),
                         "decay_time": wp.Uniform(0.5, 40)})
    if model == "bazin":
        return wp.Prior({"amplitude": amp, "t0": wp.Uniform(-5, 10),
                         "tau_rise": wp.Uniform(0.1, 10), "tau_fall": wp.Uniform(0.5, 40)})
    return wp.Prior({"amplitude": amp, "t0": wp.Uniform(0, 10),
                     "sigma_rise": wp.Uniform(0.05, 10), "tau_decay": wp.Uniform(0.5, 40)})


def main():
    lc = wp.load_lightcurve(DATA, explosion_date=57982.0, min_snr=3)
    r = lc.select_bands("r").add_flux()
    fmax = float(np.nanmax(r.flux))
    print(f"AT2017GFO r-band: {r.n_points} pts, t={r.time.min():.2f}..{r.time.max():.2f} d")

    results, fits = [], {}
    for model in MODELS:
        prior = make_prior(model, fmax)
        res_abc = wp.fit_ABC(r, model, prior=prior, n_simulations=200_000, quantile=0.01,
                             n_jobs=32, seed=0)
        res_smc = wp.fit_ABC_SMC(r, model, prior=prior, n_particles=1000, n_rounds=8,
                                 quantile=0.4, n_jobs=32, seed=0)
        for sampler, res in [("ABC", res_abc), ("ABC-SMC", res_smc)]:
            res.to_json(FIGS / f"at2017gfo_{model}_{sampler.lower().replace('-', '')}.json")
            dof = max(res.n_data - res.n_params, 1)
            sims = res.info.get("n_simulations") or res.info.get("total_simulations")
            results.append({"model": model, "sampler": sampler, "k": res.n_params,
                            "redchi2": res.min_distance / dof, "aic": res.aic, "bic": res.bic,
                            "runtime": res.runtime_s, "sims": int(sims)})
            fits[(model, sampler)] = res

    results.sort(key=lambda d: d["aic"])
    best = results[0]

    # --- comparison figure: data + ABC best (dashed) + SMC best (solid), per model ---
    tgrid = np.linspace(max(r.time.min(), 1e-3), r.time.max(), 300)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, model in zip(axes, MODELS):
        ax.errorbar(r.time, r.flux, yerr=r.flux_err, fmt="o", ms=4, color="0.4",
                    label="AT2017GFO r", zorder=1)
        fn = get_model(model)
        for sampler, ls, col in [("ABC", "--", "tab:orange"), ("ABC-SMC", "-", "tab:red")]:
            res = fits[(model, sampler)]
            dof = max(res.n_data - res.n_params, 1)
            ax.plot(tgrid, fn(res.best_params, tgrid), ls, color=col, lw=2,
                    label=f"{sampler} (chi2/dof={res.min_distance / dof:.0f})")
        ax.set_title(model)
        ax.set_xlabel("days since explosion")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("flux density [Jy]")
    fig.suptitle("AT2017GFO r-band - model & sampler comparison")
    fig.tight_layout()
    fig.savefig(FIGS / "at2017gfo_model_comparison.png", dpi=130, bbox_inches="tight")
    print("saved comparison figure")

    # --- markdown report ---
    L = []
    L.append("# Report - AT2017GFO: ABC vs ABC-SMC across simple models\n")
    L.append(f"Test case: **AT2017GFO** (GW170817 kilonova), r-band, {r.n_points} points after "
             f"`min_snr=3`; time = days since explosion (MJD 57982).\n")
    L.append("Models: `flare`, `bazin` (SN rise+fall), `gaussian_rise` (Gaussian rise + exp decay). "
             "Samplers: `ABC` (flat rejection, 200k sims, best 1%) and `ABC-SMC` (1000 particles, "
             "8 adaptive rounds, quantile 0.4). The chi-square distance gives `AIC = chi2_min + 2k`, "
             "`BIC = chi2_min + k*ln(n)`.\n")
    L.append("## Results (ranked by AIC)\n")
    L.append("| Model | Sampler | k | chi2/dof | AIC | BIC | runtime (s) | simulations |")
    L.append("|---|---|---|---|---|---|---|---|")
    for d in results:
        L.append(f"| {d['model']} | {d['sampler']} | {d['k']} | {d['redchi2']:.1f} | "
                 f"{d['aic']:.0f} | {d['bic']:.0f} | {d['runtime']:.2f} | {d['sims']:,} |")
    L.append("")
    L.append(f"**Best (lowest AIC): `{best['model']}` via {best['sampler']}** "
             f"(AIC={best['aic']:.0f}, chi2/dof={best['redchi2']:.1f}).\n")
    L.append("![comparison](figures/at2017gfo_model_comparison.png)\n")
    L.append("## Best-fit parameters (ABC-SMC)\n")
    for model in MODELS:
        res = fits[(model, "ABC-SMC")]
        ps = ", ".join(f"{p}={res.best_params[p]:.3g}" for p in res.parameters)
        L.append(f"- **{model}**: {ps}")
    L.append("")
    L.append("## Notes\n")
    L.append("- chi2/dof is high for all models because AT2017GFO's r-band is high-SNR (tiny error "
             "bars), so any imperfection inflates chi2. The *relative* AIC ranking still identifies "
             "the best simple model.")
    L.append("- Read chi2 alongside the **simulations** column. Flat ABC draws every simulation from "
             "the prior (simplest, embarrassingly parallel); ABC-SMC focuses simulations near good "
             "regions over rounds. SMC's value is reaching a tight threshold with *fewer* simulations "
             "-- decisive when each simulation is expensive (e.g. physical models); for these "
             "microsecond toy models a large flat-ABC budget is already very effective.")
    L.append("- These are analytic toy models; physically-motivated models + priors can optionally be "
             "plugged in from the external redback package via the `[models]` extra.")
    REPORT.write_text("\n".join(L))
    print("wrote", REPORT)

    print("\nRanked by AIC:")
    for d in results:
        print(f"  {d['model']:14s} {d['sampler']:8s} chi2/dof={d['redchi2']:7.1f} "
              f"AIC={d['aic']:8.0f} runtime={d['runtime']:5.2f}s sims={d['sims']:,}")


if __name__ == "__main__":
    main()
