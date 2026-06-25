"""Generic single-component flare model (analytic; no redback needed).

``flux(t) = amplitude * (1 - exp(-t / rise_time)) * exp(-t / decay_time)``

Band-independent and vectorized over ``times``. ``t`` should be days since explosion (use
``LightCurve.set_explosion_date`` / ``load_lightcurve(..., explosion_date=...)``).
"""
from __future__ import annotations

import numpy as np

from ..priors import Prior, Uniform

PARAMETERS = ["amplitude", "rise_time", "decay_time"]
DESCRIPTION = "Generic flare: A*(1 - exp(-t/t_rise))*exp(-t/t_decay)."

# Default prior matches the example; scale `amplitude` to your flux units when fitting real data.
PRIOR = Prior({
    "amplitude": Uniform(0.0, 10.0),
    "rise_time": Uniform(1.0, 10.0),
    "decay_time": Uniform(5.0, 30.0),
})


def flare_flux(parameters, times, bands=None):
    """Predicted flux at ``times`` (bands ignored -- this toy model is band-independent)."""
    amplitude = parameters["amplitude"]
    rise_time = parameters["rise_time"]
    decay_time = parameters["decay_time"]
    t = np.asarray(times, dtype=float)
    return amplitude * (1.0 - np.exp(-t / rise_time)) * np.exp(-t / decay_time)
