"""Demo: the redback-backed ``two_component_kilonova`` light curve in LSST g/r/i/z.

Shows how the blue (low-κ) + red (high-κ) components combine, for two ejecta configurations, in apparent
AB magnitude at AT2017GFO's distance (z=0.00984). Saves ``dev/figures/kilonova_lightcurve.png``.
"""
import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from whisper_labia.models import get_model

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "dev", "figures", "kilonova_lightcurve.png")

BANDS = ["g", "r", "i", "z"]
COLORS = {"g": "#2ca02c", "r": "#d62728", "i": "#8c564b", "z": "#7f7f7f"}
BASE = dict(temperature_floor_1=2500.0, temperature_floor_2=2500.0, redshift=0.00984)
CASES = [
    ("more blue ejecta", dict(mej_1=0.05, vej_1=0.30, kappa_1=0.3, mej_2=0.01, vej_2=0.15, kappa_2=10.0)),
    ("more red ejecta",  dict(mej_1=0.01, vej_1=0.30, kappa_1=0.3, mej_2=0.05, vej_2=0.15, kappa_2=10.0)),
]

m = get_model("two_component_kilonova")
t = np.linspace(0.1, 12, 240)

# Compute all light curves FIRST. redback's import (inside predict) turns on matplotlib LaTeX, so we do
# every redback call before building any figure, then force usetex off (no latex in the container).
data = []
for label, case in CASES:
    params = dict(BASE, **case)
    mags = {b: -2.5 * np.log10(m.predict(params, t, np.array([b] * t.size)) / 3631.0) for b in BANDS}
    data.append((label, case, mags))

matplotlib.rcParams["text.usetex"] = False
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
for ax, (label, case, mags) in zip(axes, data):
    for b in BANDS:
        ax.plot(t, mags[b], color=COLORS[b], label=f"{b}-band")
    ax.set_title(f"{label}\n" + r"$M_{ej,1}$=%.2f, $M_{ej,2}$=%.2f $M_\odot$" % (case["mej_1"], case["mej_2"]),
                 fontsize=10)
    ax.set_xlabel("days since merger (observer frame)")
    ax.legend(frameon=False, fontsize=9)

# Invert ONCE (panels share the y-axis): brighter = lower mag = top.
axes[0].invert_yaxis()
axes[0].set_ylabel("apparent AB magnitude")
fig.suptitle("two_component_kilonova (redback) — blue + red ejecta at z=0.00984", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.97))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("Saved", OUT)
