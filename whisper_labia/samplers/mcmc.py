"""Markov Chain Monte Carlo (MCMC) sampling via ``emcee`` -- likelihood-based posterior inference.

Same pluggable contract as every other sampler: ``fit(lc, model, prior=None, ...) -> SamplerResult``.
The log-posterior is

    log P(theta | data) = log prior(theta) + log L(theta)

where the prior is Whisper's :class:`~whisper_labia.priors.Prior` and **the likelihood is Whisper's own
likelihood layer** (:func:`~whisper_labia.likelihood.make_likelihood` /
:class:`~whisper_labia.likelihood.GaussianLikelihood`). So MCMC uses the *same physically consistent
likelihood* as ABC / ABC-SMC / SNPE and automatically respects the light curve's ``data_mode`` — it
compares in **flux** space for flux data and **magnitude** space for magnitude data (the model always
predicts flux; the likelihood converts). All four samplers should therefore converge to the same
posterior.

``emcee`` is a core dependency (no extra needed). Sampling is seeded for reproducibility.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..likelihood import make_likelihood
from ..models import get_model
from .base import BaseSampler, SamplerResult, summarize_posterior


def _log_prob(theta, names, prior, predict, times, bands, likelihood):
    """log-posterior for one parameter vector ``theta`` (ordered like ``names``)."""
    params = {nm: float(v) for nm, v in zip(names, theta)}
    lp = prior.log_prob(params)
    if not np.isfinite(lp):                       # outside the prior support -> forbidden
        return -np.inf
    model_flux = np.asarray(predict(params, times, bands), dtype=float)
    ll = likelihood.log_likelihood(model_flux)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


class MCMCSampler(BaseSampler):
    """Affine-invariant ensemble MCMC (``emcee``). See the module docstring."""

    name = "mcmc"

    def fit(self, lc, model, prior=None, *, nwalkers=None, nsteps=5000, burnin=1000, thin=10,
            initial_guess=None, initial_scatter=1e-3, space="auto", likelihood="auto",
            seed=0, progress=False, moves=None) -> SamplerResult:
        """Fit ``lc`` with ``model`` via emcee MCMC, returning a :class:`SamplerResult`.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry errors (``flux_err`` / ``magnitude_err``) for the likelihood.
        model : str or Model
            Registered model name or a :class:`~whisper_labia.models.Model`.
        prior : Prior, optional
            Parameter prior; defaults to the model's ``default_prior`` (``ValueError`` if neither).
        nwalkers : int, optional
            Number of ensemble walkers (default ``max(2*ndim+2, 4*ndim)``, forced even; must be ``>= 2*ndim``).
        nsteps : int, default 5000
            MCMC iterations per walker.
        burnin : int, default 1000
            Initial steps discarded before convergence.
        thin : int, default 10
            Keep every ``thin``-th sample (reduces autocorrelation).
        initial_guess : dict or array, optional
            Starting point; walkers are perturbed around it by ``initial_scatter``. If ``None``, walkers
            are initialised by sampling the prior (no guess needed -- like ABC/SNPE).
        initial_scatter : float, default 1e-3
            Gaussian spread of the initial walker cloud around ``initial_guess``.
        space : {'auto', 'flux', 'magnitude'}, default 'auto'
            Comparison space passed to the likelihood (``'auto'`` follows the data's ``data_mode``).
        likelihood : str, default 'auto'
            Likelihood ``kind`` for :func:`make_likelihood` (e.g. ``'gaussian'``,
            ``'gaussian_upper_limits'``, ``'mixture'``); ``'auto'`` picks by data.
        seed : int, default 0
            RNG seed; the chain is reproducible given the seed (walker init + emcee proposals are seeded).
        progress : bool, default False
            Show the emcee progress bar.
        moves : optional
            An ``emcee`` move (or list of (move, weight)); default is emcee's stretch move.

        Returns
        -------
        SamplerResult
            Posterior samples + summary, ``best_params`` (max-posterior draw), exact Gaussian
            ``max_log_likelihood`` / ``aic`` / ``bic``, and diagnostics in ``info`` (acceptance fraction,
            autocorrelation time). The trained ``emcee.EnsembleSampler`` is attached as
            ``result.emcee_sampler``.
        """
        import emcee

        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")
        lik = make_likelihood(lc, kind=likelihood, space=space)   # reuse the shared likelihood layer
        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        predict = model.predict
        names = list(prior.names)
        ndim = len(names)
        k, n = ndim, int(len(times))

        nwalkers = int(nwalkers) if nwalkers else max(2 * ndim + 2, 4 * ndim)
        nwalkers += nwalkers % 2                                   # emcee's red-blue split needs even
        if nwalkers < 2 * ndim:
            raise ValueError(f"nwalkers ({nwalkers}) must be >= 2*ndim ({2 * ndim}).")

        rng = np.random.default_rng(int(seed))
        if initial_guess is not None:
            g = np.array([float(initial_guess[nm]) if isinstance(initial_guess, dict)
                          else float(initial_guess[i]) for i, nm in enumerate(names)], dtype=float)
            p0 = g + initial_scatter * rng.standard_normal((nwalkers, ndim))
        else:                                                     # spread walkers across the prior
            p0 = np.array([[prior.distributions[nm].sample(rng) for nm in names]
                           for _ in range(nwalkers)], dtype=float)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, _log_prob, moves=moves,
            args=(names, prior, predict, times, bands, lik))
        sampler._random.seed(int(seed))                           # seed emcee's proposal RNG

        t0 = time.perf_counter()
        sampler.run_mcmc(p0, int(nsteps), progress=progress)
        runtime = time.perf_counter() - t0

        flat = sampler.get_chain(discard=int(burnin), thin=int(thin), flat=True)
        flat_lp = sampler.get_log_prob(discard=int(burnin), thin=int(thin), flat=True)
        samples = pd.DataFrame(flat, columns=names)

        best_idx = int(np.argmax(flat_lp))
        best_params = {nm: float(flat[best_idx, j]) for j, nm in enumerate(names)}
        max_log_likelihood = float(
            lik.log_likelihood(np.asarray(predict(best_params, times, bands), dtype=float)))

        try:
            autocorr = float(np.nanmean(sampler.get_autocorr_time(tol=0)))
        except Exception:                                         # pragma: no cover - short chains
            autocorr = float("nan")
        info = {
            "nwalkers": int(nwalkers), "nsteps": int(nsteps), "burnin": int(burnin), "thin": int(thin),
            "space": lik.space, "likelihood": type(lik).__name__,
            "mean_acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "mean_autocorr_time": autocorr, "n_samples_per_walker": int(flat.shape[0] // nwalkers),
            "seed": int(seed),
        }
        result = SamplerResult(
            sampler="mcmc", model=model.name, parameters=list(names), samples=samples,
            summary=summarize_posterior(samples, names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )
        result.emcee_sampler = sampler
        return result


def fit_MCMC(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via emcee MCMC. See :meth:`MCMCSampler.fit` for options."""
    return MCMCSampler().fit(lc, model, prior=prior, **kwargs)
