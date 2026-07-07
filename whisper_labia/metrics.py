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

import warnings

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

    n_total = samples.shape[0]
    subsampled = n_total > max_samples
    if subsampled:
        sel = np.random.default_rng(seed).choice(n_total, size=max_samples, replace=False)
        samples = samples[sel]
        warnings.warn(f"WAIC: posterior has {n_total} draws; evaluated a random {max_samples}-draw "
                      "subsample (raise max_samples for the full set).", stacklevel=2)

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
    # Drop non-finite *draws* (rows), NOT data points (columns): WAIC is a sum over data points, so
    # every model in a comparison must be scored on the SAME points. Dropping a column when a single
    # bad draw makes it non-finite would shrink the dataset model-dependently and invalidate ΔWAIC.
    good = np.all(np.isfinite(ll), axis=1)
    n_dropped = int((~good).sum())
    if n_dropped:
        warnings.warn(f"WAIC: dropped {n_dropped}/{ll.shape[0]} posterior draws with a non-finite "
                      f"pointwise log-likelihood (all {ll.shape[1]} data points retained).", stacklevel=2)
    ll = ll[good]
    if ll.shape[0] < 2:
        raise ValueError("WAIC needs >= 2 posterior draws with finite pointwise log-likelihoods; "
                         f"only {ll.shape[0]} of {good.size} qualified.")

    n_samp = ll.shape[0]
    lppd_i = logsumexp(ll, axis=0) - np.log(n_samp)            # log mean-likelihood per point
    p_waic_i = np.var(ll, axis=0, ddof=1)                      # posterior var of log-lik per point
    elpd_i = lppd_i - p_waic_i
    n_data = ll.shape[1]
    se = float(np.sqrt(n_data * np.var(-2.0 * elpd_i, ddof=1))) if n_data > 1 else float("nan")
    return {"waic": float(-2.0 * np.sum(elpd_i)), "lppd": float(np.sum(lppd_i)),
            "p_waic": float(np.sum(p_waic_i)), "se": se, "n_samples": int(n_samp), "n_data": int(n_data),
            "n_draws_dropped": n_dropped, "subsampled": bool(subsampled)}


def per_band_metrics(lc, model, params, *, space="auto", fixed=None):
    """Per-band goodness-of-fit residual metrics at a single parameter set (e.g. the best fit).

    Evaluates ``model`` at ``params`` on the observed ``(time, band)`` grid and, **per band**, reports
    the mean-squared error (MSE), root-mean-squared error (RMSE) and mean-absolute error (MAE) of the
    residuals ``observed - model``, computed in the Gaussian likelihood's **comparison space** — flux
    density [Jy] for a flux fit, apparent magnitude [mag] for a magnitude fit — so the metric matches
    the space the fit actually optimised. This is the deterministic point-estimate complement to the
    distributional :func:`waic` / posterior-predictive checks.

    Parameters
    ----------
    lc : LightCurve
        The observed data.
    model : str | Model
        Model name or object (its ``predict`` is called once at ``params``).
    params : dict
        Parameter values — typically ``result.best_params``. Only the model's own parameters are used;
        extra keys (e.g. a likelihood scatter term) are ignored.
    space : str
        ``'auto'`` | ``'flux'`` | ``'magnitude'`` — the residual space (default follows the data mode).
    fixed : dict, optional
        Parameters held fixed during the fit and absent from ``params`` (merged in before predicting).

    Returns
    -------
    dict
        ``{"space", "unit", "bands": {band: {"mse","rmse","mae","n"}}, "overall": {...}}``.
        ``unit`` is ``"Jy"`` (flux) or ``"mag"`` (magnitude).
    """
    from .likelihood import GaussianLikelihood

    m = get_model(model)
    if m is None:
        raise ValueError(f"Unknown model {model!r}.")
    lik = GaussianLikelihood(lc, space=space)
    times = np.asarray(lc.time, dtype=float)
    bands = np.asarray(lc.band).astype(str)
    p = dict(fixed or {})
    p.update({k: float(params[k]) for k in m.parameters if k in params})     # model params only
    model_flux = np.asarray(m.predict(p, times, bands), dtype=float)
    resid = np.asarray(lik.y, float) - np.asarray(lik.model_in_space(model_flux), float)

    def _stats(r):
        r = r[np.isfinite(r)]
        if r.size == 0:
            return {"mse": float("nan"), "rmse": float("nan"), "mae": float("nan"), "n": 0}
        mse = float(np.mean(r ** 2))
        return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(r))), "n": int(r.size)}

    per_band = {str(b): _stats(resid[bands == b]) for b in np.unique(bands)}
    unit = "mag" if lik.space == "magnitude" else "Jy"
    return {"space": lik.space, "unit": unit, "bands": per_band, "overall": _stats(resid)}
