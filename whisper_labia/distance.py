"""Distance metrics for ABC (and pseudo-likelihoods for model selection).

A distance has the signature ``f(obs_flux, obs_flux_err, sim_flux, bands) -> float`` and is treated
as a black box by the sampler -- plug in any custom metric.
"""
from __future__ import annotations

import numpy as np


def chi2_distance(obs_flux, obs_flux_err, sim_flux, bands=None):
    """Multi-band chi-square: ``sum(((obs - sim) / err)**2)``.

    Summing over all points is equivalent to summing per band and adding. Numerically this equals
    ``-2 ln L`` for an independent Gaussian likelihood, which is what lets ABC report AIC/BIC.
    """
    obs_flux = np.asarray(obs_flux, dtype=float)
    sim_flux = np.asarray(sim_flux, dtype=float)
    obs_flux_err = np.asarray(obs_flux_err, dtype=float)
    residual = (obs_flux - sim_flux) / obs_flux_err
    return float(np.sum(residual * residual))
