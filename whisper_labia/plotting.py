"""Light-curve report plots.

``plot_light_curve`` renders a transient's photometry with consistent, explainable styling:

* each band gets a distinct color;
* **detections** (SNR >= 3) are circles with a black edge;
* **low-SNR** points (SNR < 3) are up-triangles;
* **upper limits** are down-triangles;
* magnitude axes are inverted (brighter = up).

Two layouts:

* ``layout='report'`` -- one figure, two stacked panels (apparent magnitude vs time and flux density
  vs time), all selected bands overlaid.
* ``layout='grid'`` -- one panel per band, with the y-axis chosen by ``quantity``
  (``'apparent_mag'`` | ``'absolute_mag'`` (needs redshift) | ``'flux'``).
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

_MARKER = {"det": "o", "lowsnr": "^", "ul": "v"}
_SNR_LOW = 3.0


def _band_colors(bands):
    cmap = plt.get_cmap("turbo")
    n = max(len(bands), 1)
    return {b: cmap((i + 0.5) / n) for i, b in enumerate(bands)}


def _categories(lc):
    """Classify each point as 'det', 'lowsnr', or 'ul'."""
    n = lc.n_points
    ul = lc.upper_limit if lc.upper_limit is not None else np.zeros(n, dtype=bool)
    try:
        snr = lc.snr
    except ValueError:
        snr = np.full(n, np.inf)
    return np.where(ul, "ul", np.where(snr < _SNR_LOW, "lowsnr", "det")).astype(object)


def _time_label(lc):
    if "explosion_mjd" in lc.meta:
        return f"days since explosion (MJD {lc.meta['explosion_mjd']:.3f})"
    return "time [MJD]"


def _scatter(ax, time, y, yerr, cat, color, label=None):
    """Plot one band's points, split by marker category."""
    labeled = False
    for c in ("det", "lowsnr", "ul"):
        m = cat == c
        if not m.any():
            continue
        lab = label if (label and not labeled) else None
        style = dict(color=color, marker=_MARKER[c], linestyle="none",
                     markeredgecolor="black", markeredgewidth=0.6, markersize=6, label=lab)
        if c == "ul" or yerr is None:
            ax.plot(time[m], y[m], **style)
        else:
            ax.errorbar(time[m], y[m], yerr=yerr[m], ecolor=color, elinewidth=0.8,
                        capsize=0, **style)
        labeled = labeled or (lab is not None)


def _quantity(full, quantity):
    """Return (y, yerr, ylabel, invert_axis) for the requested quantity."""
    q = quantity.lower()
    if q in ("flux", "flux_density"):
        return full.flux, full.flux_err, "flux density [Jy]", False
    if q in ("absolute_mag", "absolute", "abs_mag"):
        if full.redshift is None:
            raise ValueError("quantity='absolute_mag' requires the light curve's redshift.")
        from astropy.cosmology import Planck18
        mu = float(Planck18.distmod(full.redshift).value)
        return full.magnitude - mu, full.magnitude_err, "absolute magnitude (AB)", True
    return full.magnitude, full.magnitude_err, "apparent magnitude (AB)", True


def plot_light_curve(lc, *, layout="report", quantity="apparent_mag", bands=None,
                     ncols=3, figsize=None, title=None, save=None):
    """Report plot of a light curve. Returns the matplotlib ``Figure``.

    See the module docstring for layouts, the ``quantity`` options, and the marker conventions.
    """
    if bands is not None:
        lc = lc.select_bands(bands)
    full = lc.add_flux().add_mag()      # ensure both magnitude and flux are available
    band_list = full.bands
    colors = _band_colors(band_list)
    cat = _categories(full)

    if layout == "report":
        fig, (ax_m, ax_f) = plt.subplots(2, 1, sharex=True, figsize=figsize or (9, 8))
        for b in band_list:
            m = full.band == b
            merr = None if full.magnitude_err is None else full.magnitude_err
            ferr = None if full.flux_err is None else full.flux_err
            _scatter(ax_m, full.time[m], full.magnitude[m],
                     None if merr is None else merr[m], cat[m], colors[b], label=b)
            _scatter(ax_f, full.time[m], full.flux[m],
                     None if ferr is None else ferr[m], cat[m], colors[b])
        ax_m.invert_yaxis()
        ax_m.set_ylabel("apparent magnitude (AB)")
        ax_f.set_ylabel("flux density [Jy]")
        ax_f.set_xlabel(_time_label(full))
        for ax in (ax_m, ax_f):
            ax.grid(alpha=0.3)
        ax_m.legend(ncol=4, fontsize=8, loc="best")

    elif layout == "grid":
        y, yerr, ylabel, invert = _quantity(full, quantity)
        nb = len(band_list)
        ncols = min(ncols, nb) or 1
        nrows = int(np.ceil(nb / ncols))
        fig, axes = plt.subplots(nrows, ncols, sharex=True,
                                 figsize=figsize or (4 * ncols, 2.6 * nrows), squeeze=False)
        axflat = axes.ravel()
        for ax, b in zip(axflat, band_list):
            m = full.band == b
            _scatter(ax, full.time[m], y[m], None if yerr is None else yerr[m],
                     cat[m], colors[b])
            if invert:
                ax.invert_yaxis()
            ax.set_title(b, fontsize=9)
            ax.grid(alpha=0.3)
        for ax in axflat[nb:]:
            ax.set_visible(False)
        fig.supxlabel(_time_label(full))
        fig.supylabel(ylabel)

    else:
        raise ValueError(f"Unknown layout {layout!r} (use 'report' or 'grid').")

    fig.suptitle(title or (full.name or "light curve"))
    fig.tight_layout()
    if save is not None:
        fig.savefig(save, dpi=130, bbox_inches="tight")
    return fig
