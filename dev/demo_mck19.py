"""Demo: the ``mck19`` AGN-disk BBH-merger flare light curve in LSST g/r/i.

Renders the kicked-hotspot flare (sin^2 rise to the ram-pressure delay, then exponential decay) for two
kick velocities, in apparent AB magnitude. Saves ``dev/figures/mck19_lightcurve.png``.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from whisper_labia.models import get_model, mck19

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "dev", "figures", "mck19_lightcurve.png")

BANDS = ["g", "r", "i"]
COLORS = {"g": "#2ca02c", "r": "#d62728", "i": "#8c564b"}
BASE = {"v_kick": 300.0, "M_smbh": 1e8, "M_bh": 80.0, "r_bh": 700.0, "redshift": 0.28}

m = get_model("mck19")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

for ax, v_kick in zip(axes, (250.0, 400.0)):
    params = dict(BASE, v_kick=v_kick)
    t = np.linspace(0, 90, 1000)
    for band in BANDS:
        flux = m.predict(params, t, np.array([band] * t.size))
        mag = -2.5 * np.log10(flux / 3631.0)
        ax.plot(t, mag, color=COLORS[band], label=f"{band}-band")
    t_ram = mck19._t_ram_days(params["M_bh"], v_kick, params["redshift"])
    ax.axvline(t_ram, ls=":", color="0.5", lw=1)
    ax.text(t_ram, 0.04, r"  $t_{\rm ram}$", transform=ax.get_xaxis_transform(),
            color="0.4", va="bottom", fontsize=9)
    ax.set_title(rf"$v_{{\rm kick}}={v_kick:.0f}$ km/s  ($M_\bullet=10^8\,M_\odot$, $r=700\,R_g$, $z=0.28$)")
    ax.set_xlabel("time since merger [observer-frame days]")
    ax.legend(frameon=False, fontsize=9)

# Invert ONCE: the panels share the y-axis, so inverting each would cancel out (brighter = lower mag = top).
axes[0].invert_yaxis()
axes[0].set_ylabel("apparent AB magnitude")
fig.suptitle("mck19 — BBH-merger flare in an AGN disk (McKernan 2019 / Darc 2025)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=130)
print("Saved", OUT)
