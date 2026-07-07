"""Demo: ABC fit of the generic flare model to AT2017GFO (r-band), with a fit plot and timing.

Run inside the container:
    docker exec phe_sbi python /tf/astrodados2/phelipedata2/WHISPER/whisper-labia/dev/demo_abc_at2017gfo.py
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import whisper_labia as wp  # noqa: E402
from whisper_labia.models.flare import flare_flux  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "tests" / "data" / "at2017gfo.csv"
FIGS = ROOT / "dev" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)


def main():
    lc = wp.load_lightcurve(DATA, explosion_date=57982.0, min_snr=3)
    r = lc.select_bands("r").add_flux()
    fmax = float(np.nanmax(r.flux))
    print(f"AT2017GFO r-band: {r.n_points} pts, t={r.time.min():.2f}..{r.time.max():.2f} d, "
          f"flux_max={fmax:.3e} Jy")

    prior = wp.Prior({
        "amplitude": wp.Uniform(0.0, 10 * fmax),
        "rise_time": wp.Uniform(0.05, 10.0),
        "decay_time": wp.Uniform(0.5, 40.0),
    })

    res = wp.fit_ABC(r, "flare", prior=prior, n_simulations=200_000, quantile=0.005, n_jobs=16, seed=0)
    print(res)
    for p, s in res.summary.items():
        print(f"  {p:10s} median={s['median']:.4g}  [{s['ci16']:.4g}, {s['ci84']:.4g}]")
    print(f"  AIC={res.aic:.1f}  BIC={res.bic:.1f}  maxLogL={res.max_log_likelihood:.1f}  "
          f"accepted={res.info['n_accepted']}")
    res.to_json(FIGS / "at2017gfo_abc_flare_r.json")

    # --- fit plot: data + best fit + posterior spread ---
    tgrid = np.linspace(max(r.time.min(), 1e-3), r.time.max(), 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    draws = res.samples.sample(min(80, len(res.samples)), random_state=0)
    for _, s in draws.iterrows():
        ax.plot(tgrid, flare_flux({k: s[k] for k in res.parameters}, tgrid), color="0.6", alpha=0.08)
    ax.plot(tgrid, flare_flux(res.best_params, tgrid), "r-", lw=2, label="ABC best-fit flare")
    ax.errorbar(r.time, r.flux, yerr=r.flux_err, fmt="o", mec="black", ms=5,
                color="tab:blue", label="AT2017GFO  r")
    ax.set_xlabel("days since explosion")
    ax.set_ylabel("flux density [Jy]")
    ax.set_title("AT2017GFO r-band - ABC flare fit")
    ax.legend()
    fig.savefig(FIGS / "at2017gfo_abc_flare_r.png", dpi=130, bbox_inches="tight")
    print("saved:", FIGS / "at2017gfo_abc_flare_r.png")

    # --- timing: serial vs parallel ---
    print("\nTiming (n_simulations=200000; the flare model is tiny -> Python-overhead bound):")
    for n_jobs in (1, 8, 32):
        rr = wp.fit_ABC(r, "flare", prior=prior, n_simulations=200_000, quantile=0.01,
                        n_jobs=n_jobs, seed=1)
        print(f"  n_jobs={n_jobs:>2}: {rr.runtime_s:6.3f}s  ({200_000 / rr.runtime_s:8.0f} sims/s)")


if __name__ == "__main__":
    main()
