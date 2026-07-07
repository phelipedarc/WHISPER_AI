#!/usr/bin/env python
"""Standalone end-to-end wall-clock benchmark for one Villar+17 AT2017GFO fit variant.

Reads each method's ``villar_<method>.json`` (the ``wall_s`` field) from a figures directory and
renders a single horizontal bar chart, sorted fastest -> slowest, styled to match ``villar_plots.py``
(shared ``COLORS`` / ``SHORT``). Default variant is the preprocessed UVOIR reduction.

    python analysis/at2017gfo_villar/runtime_plot.py [figures_subdir]
        # default: at2017gfo_villar_full_preprocessed
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

SELF = os.path.dirname(os.path.abspath(__file__))
COLORS = dict(zip(["mcmc", "abc", "abc_smc", "npe_mdn", "npe_nsf", "snpe5_nsf", "snpe5_tcn"],
                  ["#08306b", "#a50026", "#006d2c", "#c51b7d", "#01665e", "#4d004b", "#00441b"]))
SHORT = {"mcmc": "MCMC", "abc": "ABC", "abc_smc": "ABC-SMC", "npe_mdn": "NPE-MDN",
         "npe_nsf": "NPE-NSF", "snpe5_nsf": "SNPE-5r", "snpe5_tcn": "SNPE-5r+TCN"}
ORDER = ["mcmc", "abc", "abc_smc", "npe_mdn", "npe_nsf", "snpe5_nsf", "snpe5_tcn"]


def make(subdir="at2017gfo_villar_full_preprocessed"):
    figdir = os.path.join(SELF, "figures", subdir)
    rows = []
    for m in ORDER:
        p = os.path.join(figdir, f"villar_{m}.json")
        if os.path.exists(p):
            rows.append((m, float(json.load(open(p))["wall_s"])))
    rows.sort(key=lambda kv: kv[1])                  # fastest -> slowest
    methods, walls = [m for m, _ in rows], [w for _, w in rows]

    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(rows) + 1.2))
    y = np.arange(len(rows))
    ax.barh(y, walls, color=[COLORS[m] for m in methods], height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels([SHORT.get(m, m) for m in methods], fontsize=12)
    ax.set_xscale("log")
    ax.set_xlabel("end-to-end wall-clock time [s]  (log scale)", fontsize=12)
    ax.set_xlim(min(walls) * 0.7, max(walls) * 2.2)
    for yi, w in zip(y, walls):                      # absolute time + minutes past each bar end
        ax.text(w * 1.06, yi, f"{w:.0f} s  ({w/60:.1f} min)", va="center", ha="left", fontsize=11)
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.set_axisbelow(True)
    ax.set_title("AT2017GFO two-component kilonova — fit time per method\n"
                 "(preprocessed UVOIR: 10 bands, SNR > 5, deduped)", fontsize=13, weight="bold")
    fig.tight_layout()
    out = os.path.join(figdir, "villar_runtime.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    for m, w in rows:
        print(f"  {SHORT.get(m, m):12s} {w:8.0f} s")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "at2017gfo_villar_full_preprocessed")
