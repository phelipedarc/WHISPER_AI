#!/usr/bin/env python
"""Benchmark: SNPE training on **GPU vs CPU**, scaling simulations (and rounds).

The GPU accelerates the *neural-network training*, not the simulator, so this uses the cheap
``gaussian_rise`` model (negligible simulation cost) to isolate the training speed-up. It runs a ladder
of increasingly heavy configurations, **estimating each tier's time from the previous one before
running it** and skipping any tier whose CPU estimate exceeds ``--budget`` seconds. Outputs a table and
a log-log runtime plot (``sanity_check/figures/snpe_device_benchmark.png``).

    python sanity_check/benchmark_snpe_device.py [--budget 240]

GPU selection honours ``CUDA_VISIBLE_DEVICES``. Runtimes are wall-clock and depend on machine load; the
CPU/GPU *ratio* and the crossover point are the robust takeaways.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import whisper_labia as wp
from whisper_labia.models import get_model

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "sanity_check", "figures")
TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}
# (num_simulations, num_rounds), light -> heavy
TIERS = [(500, 1), (2_000, 1), (8_000, 1), (20_000, 1), (50_000, 1), (20_000, 2)]


def _make_lc():
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30.0, 60)
    flux = m.predict(TRUE, t, None)
    obs = flux + np.random.default_rng(0).normal(0.0, 0.1, flux.shape)
    return wp.LightCurve(time=t, band=["r"] * 60, flux=obs, flux_err=np.full_like(flux, 0.1), name="syn")


def _run(lc, device, num_simulations, num_rounds):
    t0 = time.time()
    res = wp.fit_SNPE(lc, "gaussian_rise", prior=get_model("gaussian_rise").default_prior,
                      num_rounds=num_rounds, num_simulations=num_simulations, num_samples=1000,
                      space="flux", device=device, seed=0, max_num_epochs=200)
    wall = time.time() - t0
    err = float(np.mean([abs(res.summary[p]["median"] - TRUE[p]) / TRUE[p] for p in TRUE]))
    return {"device": res.info["device"], "wall_s": wall, "runtime_s": float(res.runtime_s),
            "param_frac_err": err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0,
                    help="measure CPU while its estimate <= this (s); estimate it beyond (GPU always runs)")
    args = ap.parse_args()

    lc = _make_lc()
    # Warm up CUDA so the first GPU tier isn't charged the one-off context-init cost.
    try:
        _run(lc, "cuda", 200, 1)
        print("(GPU warmed up)")
    except Exception as e:
        print(f"(GPU warmup failed: {type(e).__name__}: {e})")

    rows, last_cpu = [], None
    print(f"{'sims':>7} {'rnds':>4} | {'CPU s':>10} {'GPU s':>8} {'speedup':>8} {'recov err':>9}")
    for num_sim, num_rnd in TIERS:
        total = num_sim * num_rnd
        gpu = _run(lc, "cuda", num_sim, num_rnd)                 # GPU is cheap -> always measured
        cpu_est = last_cpu["runtime_s"] * total / last_cpu["total"] if last_cpu else 0.0
        if last_cpu is None or cpu_est <= args.budget:           # measure CPU while affordable
            cpu = _run(lc, "cpu", num_sim, num_rnd)
            cpu_rt, estimated = cpu["runtime_s"], False
            last_cpu = {"runtime_s": cpu_rt, "total": total}
            rec = max(cpu["param_frac_err"], gpu["param_frac_err"])
        else:                                                    # estimate CPU (too slow to run here)
            cpu, cpu_rt, estimated = None, cpu_est, True
            rec = gpu["param_frac_err"]
        speed = cpu_rt / gpu["runtime_s"] if gpu["runtime_s"] > 0 else float("nan")
        tag = "~" if estimated else " "
        print(f"{num_sim:>7} {num_rnd:>4} | {tag}{cpu_rt:>8.1f}s {gpu['runtime_s']:>7.1f}s "
              f"{speed:>7.2f}x {rec:>8.1%}")
        rows.append({"num_sim": num_sim, "num_rounds": num_rnd, "total": total,
                     "cpu_runtime_s": cpu_rt, "cpu_estimated": estimated,
                     "gpu_runtime_s": gpu["runtime_s"], "gpu_device": gpu["device"],
                     "speedup": speed, "recov_err": rec})

    os.makedirs(FIGDIR, exist_ok=True)
    json.dump(rows, open(os.path.join(FIGDIR, "snpe_device_benchmark.json"), "w"), indent=2)
    _plot(rows)


def _plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 13, "axes.labelsize": 15, "axes.titlesize": 16,
                                "text.usetex": False, "figure.dpi": 130})
    import matplotlib.pyplot as plt

    done = [r for r in rows if r.get("num_rounds") == 1]
    if not done:
        print("no single-round tiers to plot")
        return
    x = [r["total"] for r in done]
    gpu = [r["gpu_runtime_s"] for r in done]
    cpu = [r["cpu_runtime_s"] for r in done]
    cpu_meas = [(r["total"], r["cpu_runtime_s"]) for r in done if not r["cpu_estimated"]]
    cpu_est = [(r["total"], r["cpu_runtime_s"]) for r in done if r["cpu_estimated"]]

    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.plot(x, gpu, "s-", color="#D55E00", lw=2.4, ms=9, label="GPU (CUDA), measured", zorder=3)
    ax.plot(x, cpu, "-", color="#1f77b4", lw=1.4, alpha=0.5, zorder=1)            # connecting guide
    if cpu_meas:
        ax.plot(*zip(*cpu_meas), "o", color="#1f77b4", ms=9, label="CPU, measured", zorder=3)
    if cpu_est:
        ax.plot(*zip(*cpu_est), "o", mfc="white", mec="#1f77b4", mew=1.8, ms=9,
                label="CPU, estimated (∝ N)", zorder=3)
    for xi, c, g in zip(x, cpu, gpu):
        ax.annotate(f"{c/g:.1f}×", (xi, max(c, g)), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=10, color="0.25", weight="bold")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Total simulations (training-set size)")
    ax.set_ylabel("SNPE runtime  [s]")
    ax.set_title("SNPE training: GPU vs CPU (gaussian_rise; CUDA speed-up annotated)")
    ax.grid(alpha=0.3, which="both", lw=0.5)
    ax.legend(frameon=True, fontsize=11, loc="upper left")
    fig.tight_layout()
    out = os.path.join(FIGDIR, "snpe_device_benchmark.png")
    fig.savefig(out, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
