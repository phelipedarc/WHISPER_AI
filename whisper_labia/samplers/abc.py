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

from ..distance import chi2_distance
from ..models import get_model
from .base import BaseSampler, SamplerResult, summarize_posterior


def _simulate_batch(predict, prior, distance, times, bands, obs_flux, obs_err, n_sims, seed):
    rng = np.random.default_rng(seed)
    thetas = []
    distances = np.empty(n_sims, dtype=float)
    for i in range(n_sims):
        theta = prior.sample(rng)
        sim = np.asarray(predict(theta, times, bands), dtype=float)
        distances[i] = distance(obs_flux, obs_err, sim, bands)
        thetas.append(theta)
    return thetas, distances


def _worker(args):
    return _simulate_batch(*args)


class ABCSampler(BaseSampler):
    name = "abc"

    def fit(self, lc, model, prior=None, *, n_simulations=10000, quantile=0.01, threshold=None,
            distance=chi2_distance, n_jobs=None, seed=0):
        model = get_model(model)
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
        base, rem = divmod(n_simulations, n_jobs)
        chunks = [base + (1 if k < rem else 0) for k in range(n_jobs)]
        seeds = np.random.SeedSequence(seed).spawn(n_jobs)

        t0 = time.perf_counter()
        thetas = []
        dist_parts = []
        if n_jobs == 1:
            th, ds = _simulate_batch(predict, prior, distance, times, bands,
                                     obs_flux, obs_err, chunks[0], seeds[0])
            thetas.extend(th)
            dist_parts.append(ds)
        else:
            args = [(predict, prior, distance, times, bands, obs_flux, obs_err, chunks[k], seeds[k])
                    for k in range(n_jobs) if chunks[k] > 0]
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                for th, ds in executor.map(_worker, args):
                    thetas.extend(th)
                    dist_parts.append(ds)
        runtime = time.perf_counter() - t0

        distances = np.concatenate(dist_parts)
        epsilon = float(np.quantile(distances, quantile)) if threshold is None else float(threshold)
        keep = distances <= epsilon
        accepted = [thetas[i] for i in np.nonzero(keep)[0]]
        samples = pd.DataFrame(accepted, columns=model.parameters)
        samples["distance"] = distances[keep]

        # Model-selection metrics: chi-square distance == -2 ln L (Gaussian).
        chi2_min = float(distances.min())
        k, n = len(model.parameters), lc.n_points
        best = thetas[int(np.argmin(distances))]
        info = {
            "n_simulations": int(n_simulations),
            "n_accepted": int(keep.sum()),
            "acceptance_rate": float(keep.mean()),
            "epsilon": epsilon,
            "quantile": None if threshold is not None else float(quantile),
            "n_jobs": int(n_jobs),
            "distance": getattr(distance, "__name__", str(distance)),
        }
        return SamplerResult(
            sampler="abc", model=model.name, parameters=list(model.parameters),
            samples=samples, summary=summarize_posterior(samples, model.parameters),
            best_params=best, n_data=n, n_params=k, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=-0.5 * chi2_min,
            aic=float(chi2_min + 2 * k), bic=float(chi2_min + k * np.log(n)),
        )


def fit_ABC(lc, model="flare", prior=None, **kwargs):
    """Fit ``lc`` with ``model`` via rejection ABC. See :meth:`ABCSampler.fit` for options."""
    return ABCSampler().fit(lc, model, prior=prior, **kwargs)
