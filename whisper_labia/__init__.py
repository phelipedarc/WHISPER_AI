"""Whisper (``whisper_labia``): easy Bayesian model comparison of transient light curves.

Two axes are pluggable: **models** (``register_model``; built-in + custom) and **samplers**
(``register_sampler``; ABC now, MCMC/Dynesty/SNPE coming). redback provides physical models + priors
(optional ``[models]`` extra); the data handling, samplers, plots and outputs are Whisper's own.
"""

__version__ = "0.0.1.dev0"

from .io import FILTER_LOOKUP, LightCurve, group_bands, load_lightcurve
from .plotting import plot_light_curve
from .priors import LogUniform, Prior, Uniform
from .distance import chi2_distance
from .models import Model, get_model, list_models, register_model
from .samplers import (
    SamplerResult,
    fit,
    fit_ABC,
    list_samplers,
    register_sampler,
)

__all__ = [
    "__version__",
    # data + plotting
    "LightCurve", "load_lightcurve", "plot_light_curve", "group_bands", "FILTER_LOOKUP",
    # priors / models / distance
    "Prior", "Uniform", "LogUniform",
    "Model", "register_model", "get_model", "list_models",
    "chi2_distance",
    # samplers
    "fit_ABC", "fit", "SamplerResult", "register_sampler", "list_samplers",
]
