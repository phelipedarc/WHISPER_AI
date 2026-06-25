"""Bazin (2009) supernova light-curve model -- the classic rise + fall.

``flux(t) = A * exp(-(t - t0) / tau_fall) / (1 + exp(-(t - t0) / tau_rise))``

Band-independent and vectorized. Exponent arguments are clipped to keep extreme prior draws finite
(so ABC always gets a usable distance). ``t`` is days since explosion.
"""
from __future__ import annotations

import numpy as np

from ..priors import Prior, Uniform

PARAMETERS = ["amplitude", "t0", "tau_rise", "tau_fall"]
DESCRIPTION = "Bazin (2009): A*exp(-(t-t0)/tau_fall) / (1 + exp(-(t-t0)/tau_rise))."

PRIOR = Prior({
    "amplitude": Uniform(0.0, 10.0),
    "t0": Uniform(-10.0, 30.0),
    "tau_rise": Uniform(0.1, 20.0),
    "tau_fall": Uniform(0.5, 60.0),
})


def bazin_flux(parameters, times, bands=None):
    amplitude = parameters["amplitude"]
    t0 = parameters["t0"]
    tau_rise = parameters["tau_rise"]
    tau_fall = parameters["tau_fall"]
    dt = np.asarray(times, dtype=float) - t0
    fall = np.exp(np.clip(-dt / tau_fall, -50.0, 50.0))
    rise = np.exp(np.clip(-dt / tau_rise, -50.0, 50.0))
    return amplitude * fall / (1.0 + rise)
