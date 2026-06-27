"""Whisper (``whisper_labia``): easy Bayesian model comparison of transient light curves.

Two axes are pluggable: **models** (``register_model``; built-in + custom) and **samplers**
(``register_sampler``; ABC now, MCMC/Dynesty/SNPE coming). The data ingestion, samplers, likelihoods,
plots and outputs are Whisper's own and run standalone. Physical models + priors can optionally be
supplied by the external redback package (the ``[models]`` extra), which Whisper uses only as a source
of models and priors.
"""

__version__ = "0.0.1.dev0"

from .io import (
    FILTER_LOOKUP,
    LSST_BAND_INFO,
    LightCurve,
    SvoUnavailable,
    group_bands,
    load_lightcurve,
    register_manual_band,
    resolve_band,
    resolve_bands,
)
from .plotting import plot_light_curve
from .priors import LogUniform, Prior, Uniform
from .distance import chi2_distance
from .likelihood import (
    GaussianLikelihood,
    GaussianLikelihoodWithUpperLimits,
    MixtureGaussianLikelihood,
    make_likelihood,
)
from .models import Model, get_model, list_models, register_model
from .samplers import (
    SamplerResult,
    fit,
    fit_ABC,
    fit_ABC_SMC,
    list_samplers,
    register_sampler,
)

__all__ = [
    "__version__",
    # data + plotting
    "LightCurve", "load_lightcurve", "plot_light_curve", "group_bands", "FILTER_LOOKUP",
    "resolve_band", "resolve_bands", "LSST_BAND_INFO", "register_manual_band", "SvoUnavailable",
    # priors / models / distance
    "Prior", "Uniform", "LogUniform",
    "Model", "register_model", "get_model", "list_models",
    "chi2_distance",
    # likelihoods
    "GaussianLikelihood", "GaussianLikelihoodWithUpperLimits", "MixtureGaussianLikelihood",
    "make_likelihood",
    # samplers
    "fit_ABC", "fit_ABC_SMC", "fit", "SamplerResult", "register_sampler", "list_samplers",
]
