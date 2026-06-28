"""Bayesian model-selection metrics computed from a posterior sample.

**WAIC** — the Widely Applicable Information Criterion (Watanabe 2010; Gelman, Hwang & Vehtari 2014) —
is a fully-Bayesian alternative to AIC/BIC. Where AIC/BIC penalise a single best-fit point, WAIC uses
the **pointwise** log-likelihood averaged over the *whole posterior*, penalised by its posterior
variance (the effective number of parameters ``p_waic``). It therefore needs the posterior draws, not
just a point estimate — which is exactly why it complements (and is more honest than) a table of medians.

Lower WAIC is better. WAIC ``= -2 (lppd - p_waic)`` with
``lppd = Σ_i log mean_s p(y_i | θ_s)`` and ``p_waic = Σ_i Var_s log p(y_i | θ_s)``.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from .likelihood import make_likelihood
from .models import get_model


def _resolve_samples(posterior, model):
    """Return (samples ndarray, column names or None, model name or None) from varied inputs."""
    import pandas as pd

    model_name = getattr(posterior, "model", None)
    if hasattr(posterior, "samples"):                          # SamplerResult
        df = posterior.samples
        return df.to_numpy(dtype=float), list(df.columns), model_name
    if isinstance(posterior, pd.DataFrame):
        return posterior.to_numpy(dtype=float), list(posterior.columns), model_name
    return np.asarray(posterior, dtype=float), None, model_name   # ndarray -> names from the model


def waic(posterior, lc, model=None, *, space="auto", likelihood="auto", fixed=None,
         max_samples=2000, seed=0):
    """Widely Applicable Information Criterion (WAIC) from a posterior sample.

    Parameters
    ----------
    posterior : SamplerResult | pandas.DataFrame | numpy.ndarray
        Posterior draws. A :class:`SamplerResult` uses its ``.samples`` (and ``.model`` when ``model``
        is omitted); a ``DataFrame`` uses its columns as parameter names; a bare array takes names from
        the model's ``parameters`` (so the column order must match).
    lc : LightCurve
        Data the model is scored against.
    model : str | Model, optional
        Model name/object; defaults to ``posterior.model``.
    space, likelihood : str
        Forwarded to :func:`~whisper_labia.likelihood.make_likelihood` (data space + likelihood kind).
    fixed : dict, optional
        Parameter values to merge into every draw — for parameters held fixed during the fit and so not
        present in the posterior columns (e.g. a pinned ``redshift``).
    max_samples : int
        Cap on posterior draws evaluated (one model call per draw — matters for slow simulators).
    seed : int
        Seed for subsampling when the posterior has more than ``max_samples`` draws.

    Returns
    -------
    dict
        ``waic`` (lower is better), ``lppd``, ``p_waic`` (effective #parameters), ``se`` (standard
        error of WAIC), ``n_samples``, ``n_data``.
    """
    samples, names, model_name = _resolve_samples(posterior, model)
    model = get_model(model if model is not None else model_name)
    if model is None:
        raise ValueError("No model given and the posterior has no .model; pass model=...")
    names = list(model.parameters) if names is None else names

    if samples.shape[0] > max_samples:
        sel = np.random.default_rng(seed).choice(samples.shape[0], size=max_samples, replace=False)
        samples = samples[sel]

    lik = make_likelihood(lc, kind=likelihood, space=space)
    if not hasattr(lik, "log_likelihood_pointwise"):
        raise TypeError(f"{type(lik).__name__} has no log_likelihood_pointwise(); WAIC needs the "
                        "pointwise log-likelihood. Use a Gaussian (with/without upper limits) likelihood.")
    times, bands = np.asarray(lc.time, float), np.asarray(lc.band)
    extra = dict(fixed or {})

    ll = np.vstack([
        np.asarray(lik.log_likelihood_pointwise(
            model.predict({**extra, **{nm: float(v) for nm, v in zip(names, row)}}, times, bands)), float)
        for row in samples])                                   # (n_samples, n_data)
    keep = np.all(np.isfinite(ll), axis=0)                     # drop points any draw made non-finite
    ll = ll[:, keep]
    if ll.size == 0:
        raise ValueError("All pointwise log-likelihoods are non-finite; cannot compute WAIC.")

    n_samp = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(n_samp)            # log mean-likelihood per point
    p_waic_i = np.var(ll, axis=0, ddof=1)                      # posterior var of log-lik per point
    elpd_i = lppd_i - p_waic_i
    n_data = ll.shape[1]
    se = float(np.sqrt(n_data * np.var(-2.0 * elpd_i, ddof=1))) if n_data > 1 else float("nan")
    return {"waic": float(-2.0 * np.sum(elpd_i)), "lppd": float(np.sum(lppd_i)),
            "p_waic": float(np.sum(p_waic_i)), "se": se, "n_samples": int(n_samp), "n_data": int(n_data)}
