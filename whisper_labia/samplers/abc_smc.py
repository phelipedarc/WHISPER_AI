"""ABC-SMC (Sequential Monte Carlo).

Refine a particle population over rounds of shrinking acceptance threshold (epsilon):

* round 0 samples the prior;
* each later round resamples an accepted particle, perturbs it with a Gaussian kernel, and keeps it
  if it falls under that round's (tighter) epsilon.

Far more simulation-efficient than flat rejection at tight thresholds. Improvements over the textbook
sketch: only the *parameters* are perturbed (not the stored distance), perturbed particles outside the
prior's support are rejected, and epsilon can be adaptive.

Epsilon schedule: pass an explicit ``epsilon_schedule`` list, or leave it ``None`` for an **adaptive**
schedule where the next round's epsilon = the ``quantile`` (default 0.5) of the current round's
accepted distances.

Note: this is the **unweighted** SMC variant (parents are resampled uniformly; no importance
weights). The best fit and an approximate posterior are reliable; for a rigorously weighted posterior,
importance-weighted ABC-SMC is a planned option.
"""
from __future__ import annotations

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ..distance import chi2_distance
from ..models import get_model
from .base import BaseSampler, SamplerResult, summarize_posterior


def _perturb(particle, scale, rng):
    """Gaussian kernel: each parameter += N(0, scale*|value| + 1e-6)."""
    return {k: v + rng.normal(0.0, scale * abs(v) + 1e-6) for k, v in particle.items()}


def _smc_batch(args):
    (round_idx, parents, prior, perturb_scale, predict, distance,
     times, bands, obs_flux, obs_err, epsilon, batch_size, seed) = args
    rng = np.random.default_rng(seed)
    accepted = []
    attempts = 0
    for _ in range(batch_size):
        attempts += 1
        if round_idx == 0:
            theta = prior.sample(rng)
        else:
            theta = _perturb(parents[int(rng.integers(len(parents)))], perturb_scale, rng)
            if not np.isfinite(prior.log_prob(theta)):   # outside prior support -> reject
                continue
        sim = np.asarray(predict(theta, times, bands), dtype=float)
        d = distance(obs_flux, obs_err, sim, bands)
        if d < epsilon:
            rec = dict(theta)
            rec["distance"] = d
            accepted.append(rec)
    return accepted, attempts


class ABCSMCSampler(BaseSampler):
    name = "abc_smc"

    def fit(self, lc, model, prior=None, *, n_particles=500, n_rounds=5, epsilon_schedule=None,
            quantile=0.5, perturbation_scale=0.1, distance=chi2_distance, n_jobs=None, seed=0,
            max_attempts_per_round=None):
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
        params = list(model.parameters)

        if epsilon_schedule is not None:
            epsilon_schedule = [float(e) for e in epsilon_schedule]
            n_rounds = len(epsilon_schedule)
        n_jobs = max(1, int(n_jobs or min(os.cpu_count() or 1, 8)))
        if max_attempts_per_round is None:
            max_attempts_per_round = max(200 * n_particles, 200_000)
        batch = max(n_particles // n_jobs, 200)
        ss = np.random.SeedSequence(seed)

        t0 = time.perf_counter()
        parents = None
        population = []
        round_info = []
        total_attempts = 0
        epsilon = epsilon_schedule[0] if epsilon_schedule is not None else np.inf
        pool = ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        try:
            for round_idx in range(n_rounds):
                if epsilon_schedule is not None:
                    epsilon = epsilon_schedule[round_idx]
                accepted, attempts = [], 0
                while len(accepted) < n_particles and attempts < max_attempts_per_round:
                    args = [(round_idx, parents, prior, perturbation_scale, predict, distance,
                             times, bands, obs_flux, obs_err, epsilon, batch, s)
                            for s in ss.spawn(n_jobs)]
                    results = [_smc_batch(args[0])] if pool is None else list(pool.map(_smc_batch, args))
                    for acc, att in results:
                        accepted.extend(acc)
                        attempts += att
                total_attempts += attempts
                if len(accepted) < n_particles:
                    warnings.warn(
                        f"ABC-SMC round {round_idx + 1}: only {len(accepted)}/{n_particles} accepted "
                        f"after {attempts} attempts (epsilon={epsilon:g}).", stacklevel=2)
                population = accepted[:n_particles] if len(accepted) >= n_particles else accepted
                dists = (np.array([p["distance"] for p in population]) if population
                         else np.array([np.inf]))
                round_info.append({
                    "round": round_idx + 1,
                    "epsilon": (float(epsilon) if np.isfinite(epsilon) else None),
                    "n_accepted": len(population),
                    "attempts": int(attempts),
                    "acceptance_rate": float(len(population) / attempts) if attempts else 0.0,
                    "best_distance": float(dists.min()),
                })
                parents = [{k: p[k] for k in params} for p in population]   # param-only for perturbation
                if epsilon_schedule is None and round_idx + 1 < n_rounds:
                    epsilon = float(np.quantile(dists, quantile))
        finally:
            if pool is not None:
                pool.shutdown()
        runtime = time.perf_counter() - t0

        samples = pd.DataFrame(population, columns=params + ["distance"])
        distances = samples["distance"].to_numpy(dtype=float) if len(samples) else np.array([np.inf])
        chi2_min = float(distances.min())
        k, n = len(params), lc.n_points
        best = ({p: float(samples.iloc[int(np.argmin(distances))][p]) for p in params}
                if len(samples) else {})
        info = {
            "n_particles": int(n_particles), "n_rounds": int(n_rounds),
            "rounds": round_info, "total_simulations": int(total_attempts),
            "perturbation_scale": float(perturbation_scale),
            "quantile": None if epsilon_schedule is not None else float(quantile),
            "n_jobs": int(n_jobs), "distance": getattr(distance, "__name__", str(distance)),
        }
        return SamplerResult(
            sampler="abc_smc", model=model.name, parameters=params,
            samples=samples, summary=summarize_posterior(samples, params), best_params=best,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            min_distance=chi2_min, max_log_likelihood=-0.5 * chi2_min,
            aic=float(chi2_min + 2 * k), bic=float(chi2_min + k * np.log(n)),
        )


def fit_ABC_SMC(lc, model="flare", prior=None, **kwargs):
    """Fit ``lc`` with ``model`` via ABC-SMC. See :meth:`ABCSMCSampler.fit` for options."""
    return ABCSMCSampler().fit(lc, model, prior=prior, **kwargs)
