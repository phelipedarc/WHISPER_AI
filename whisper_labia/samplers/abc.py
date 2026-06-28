"""Approximate Bayesian Computation -- parallel rejection sampler.

Workflow: sample prior -> simulate light curve -> compute distance to data -> accept the closest.
Acceptance is by ``quantile`` (keep the best fraction; robust, default) or a fixed ``threshold``.
Simulations are split across processes (``n_jobs``); the model ``predict``, ``prior`` and
``distance`` must be picklable for ``n_jobs > 1`` (use ``n_jobs=1`` for closures/lambdas).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import warnings

from ..distance import chi2_distance, get_distance
from ..likelihood import make_likelihood
from ..models import get_model
from .base import BaseSampler, SamplerResult, summarize_posterior


def _simulate_batch(predict, prior, distance, times, bands, obs_flux, obs_err, indices, seed):
    """Simulate the given **global** simulation indices.

    Each index ``i`` draws from its own RNG stream ``default_rng([seed, i])``, so the set of draws is
    identical no matter how the indices are chunked across workers. This makes the result fully
    reproducible for a fixed ``seed`` **regardless of ``n_jobs``** (the machine's core count).
    """
    thetas = []
    distances = np.empty(len(indices), dtype=float)
    for j, idx in enumerate(indices):
        rng = np.random.default_rng([int(seed), int(idx)])
        theta = prior.sample(rng)
        sim = np.asarray(predict(theta, times, bands), dtype=float)
        distances[j] = distance(obs_flux, obs_err, sim, bands)
        thetas.append(theta)
    return thetas, distances


def _worker(args):
    return _simulate_batch(*args)


class ABCSampler(BaseSampler):
    """Approximate Bayesian Computation by parallel rejection (see the module docstring)."""

    name = "abc"

    def fit(self, lc, model, prior=None, *, n_simulations=10000, quantile=0.01, threshold=None,
            distance=chi2_distance, n_jobs=None, seed=0):
        """Fit ``lc`` with ``model`` by rejection ABC, returning a :class:`SamplerResult`.

        Parameters
        ----------
        lc : LightCurve
            Observed light curve; must carry flux errors (``flux_err``) for the chi-square distance.
        model : str or Model
            A registered model name or a :class:`~whisper_labia.models.Model`.
        prior : Prior, optional
            Parameter prior; defaults to the model's ``default_prior`` (``ValueError`` if neither).
        n_simulations : int, default 10000
            Number of prior draws / simulations.
        quantile : float, default 0.01
            Acceptance fraction — keep the closest ``quantile`` of draws (robust default).
        threshold : float, optional
            Fixed acceptance distance epsilon; overrides ``quantile`` when given.
        distance : callable, default :func:`chi2_distance`
            ``f(obs_flux, obs_err, sim_flux, bands) -> float``. Must be picklable for ``n_jobs > 1``.
        n_jobs : int, optional
            Worker processes (default ``min(os.cpu_count(), 8)``). **The result is independent of
            ``n_jobs``** for a fixed ``seed`` — parallelism affects speed only, not the science.
        seed : int, default 0
            Base RNG seed; the full posterior is reproducible given ``(seed, n_simulations)``.

        Returns
        -------
        SamplerResult
            Posterior samples + summary, ``best_params``, and ``max_log_likelihood``/``aic``/``bic``
            (chi-square = -2 ln L for a Gaussian likelihood).
        """
        model = get_model(model)
        distance = get_distance(distance)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")

        lc = lc.add_flux()
        if lc.flux_err is None:
            raise ValueError("ABC chi-square distance needs flux errors (flux_err).")
        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        obs_flux = np.asarray(lc.flux, dtype=float)
        obs_err = np.asarray(lc.flux_err, dtype=float)
        predict = model.predict

        n_jobs = n_jobs or min(os.cpu_count() or 1, 8)
        n_jobs = max(1, min(int(n_jobs), n_simulations))
        # Split the GLOBAL simulation indices into contiguous chunks; each index keeps its own RNG
        # stream, so the union of draws (and their order) is identical for any n_jobs.
        index_chunks = [c for c in np.array_split(np.arange(n_simulations), n_jobs) if len(c)]

        t0 = time.perf_counter()
        thetas = []
        dist_parts = []
        if n_jobs == 1:
            th, ds = _simulate_batch(predict, prior, distance, times, bands,
                                     obs_flux, obs_err, index_chunks[0], seed)
            thetas.extend(th)
            dist_parts.append(ds)
        else:
            args = [(predict, prior, distance, times, bands, obs_flux, obs_err, idx, seed)
                    for idx in index_chunks]
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for th, ds in executor.map(_worker, args):
                    thetas.extend(th)
                    dist_parts.append(ds)
        runtime = time.perf_counter() - t0

        distances = np.concatenate(dist_parts)
        epsilon = float(np.quantile(distances, quantile)) if threshold is None else float(threshold)
        keep = distances <= epsilon
        if not keep.any():
            warnings.warn(
                f"ABC accepted 0 of {n_simulations} draws at epsilon={epsilon:g}; the posterior is "
                "empty. best_params / AIC / BIC reflect the single closest draw only.", stacklevel=2)
        accepted = [thetas[i] for i in np.nonzero(keep)[0]]
        samples = pd.DataFrame(accepted, columns=model.parameters)
        samples["distance"] = distances[keep]

        # Model-selection metrics. The chi-square distance drops the Gaussian normalisation and is
        # always in flux space; to make AIC/BIC **comparable across samplers** (MCMC/SNPE use the exact
        # Gaussian log-likelihood in the data's natural space), evaluate that same likelihood at the
        # best fit here too.
        chi2_min = float(distances.min())
        k, n = len(model.parameters), lc.n_points
        best = thetas[int(np.argmin(distances))]
        lik = make_likelihood(lc)
        max_log_likelihood = float(lik.log_likelihood(np.asarray(predict(best, times, bands), dtype=float)))
        info = {
            "n_simulations": int(n_simulations),
            "n_accepted": int(keep.sum()),
            "acceptance_rate": float(keep.mean()),
            "epsilon": epsilon,
            "quantile": None if threshold is not None else float(quantile),
            "n_jobs": int(n_jobs),
            "distance": getattr(distance, "__name__", str(distance)),
            "likelihood_space": lik.space,
        }
        return SamplerResult(
            sampler="abc", model=model.name, parameters=list(model.parameters),
            samples=samples, summary=summarize_posterior(samples, model.parameters),
            best_params=best, n_data=n, n_params=k, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )


def fit_ABC(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via rejection ABC. See :meth:`ABCSampler.fit` for options."""
    return ABCSampler().fit(lc, model, prior=prior, **kwargs)
