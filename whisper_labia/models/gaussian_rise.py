"""Gaussian-rise / exponential-decay model.

Rises as a Gaussian up to the peak at ``t0`` then decays exponentially -- continuous at the peak
(both branches equal ``amplitude`` there):

    flux(t) = A * exp(-(t - t0)^2 / (2 sigma_rise^2))   for t <  t0
    flux(t) = A * exp(-(t - t0) / tau_decay)            for t >= t0

Band-independent and vectorized. ``t`` is days since explosion.
"""
from __future__ import annotations

import numpy as np

from ..priors import Prior, Uniform

PARAMETERS = ["amplitude", "t0", "sigma_rise", "tau_decay"]
DESCRIPTION = "Gaussian rise to peak at t0 then exponential decay."

PRIOR = Prior({
    "amplitude": Uniform(0.0, 10.0),
    "t0": Uniform(0.0, 30.0),
    "sigma_rise": Uniform(0.1, 15.0),
    "tau_decay": Uniform(0.5, 60.0),
})


def gaussian_rise_flux(parameters, times, bands=None):
    amplitude = parameters["amplitude"]
    t0 = parameters["t0"]
    sigma_rise = parameters["sigma_rise"]
    tau_decay = parameters["tau_decay"]
    dt = np.asarray(times, dtype=float) - t0
    rise = amplitude * np.exp(-(dt ** 2) / (2.0 * sigma_rise ** 2))
    decay = amplitude * np.exp(np.clip(-dt / tau_decay, -50.0, 50.0))
    return np.where(dt < 0.0, rise, decay)
