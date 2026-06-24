"""Whisper (``whisper_labia``): easy Bayesian model comparison of transient light curves.

redback provides only the built-in transient *models* (parameters -> predicted observables) and
the *priors*. Everything else -- data ingestion, likelihood, samplers, metrics, plots and outputs --
is Whisper's own, behind a small pluggable-model / pluggable-sampler API.

The public API is wired up phase by phase; see ``WHISPER_PLAN.md``.
"""

__version__ = "0.0.1.dev0"

from .io import FILTER_LOOKUP, LightCurve, group_bands, load_lightcurve
from .plotting import plot_light_curve

__all__ = [
    "__version__",
    "LightCurve",
    "load_lightcurve",
    "plot_light_curve",
    "group_bands",
    "FILTER_LOOKUP",
]
