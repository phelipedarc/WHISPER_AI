"""Tests for WAIC (`whisper_labia.metrics.waic`) and the pointwise log-likelihood it relies on."""
import numpy as np
import pandas as pd
import pytest

import whisper_labia as wp
from whisper_labia.likelihood import GaussianLikelihood, GaussianLikelihoodWithUpperLimits
from whisper_labia.models import get_model

TRUE = {"amplitude": 5.0, "t0": 8.0, "sigma_rise": 3.0, "tau_decay": 15.0}


def _lc():
    m = get_model("gaussian_rise")
    t = np.linspace(0.1, 30, 50)
    flux = m.predict(TRUE, t, None)
    obs = flux + np.random.default_rng(0).normal(0, 0.1, flux.shape)
    return wp.LightCurve(time=t, band=["r"] * 50, flux=obs, flux_err=np.full_like(flux, 0.1), name="syn")


def test_pointwise_sums_to_total_loglik():
    lc = _lc()
    lik = GaussianLikelihood(lc, space="flux")
    mf = get_model("gaussian_rise").predict(TRUE, np.asarray(lc.time), None)
    pw = lik.log_likelihood_pointwise(mf)
    assert pw.shape == (lc.n_points,)
    assert np.isclose(pw.sum(), lik.log_likelihood(mf))


def test_pointwise_upper_limits_sums_to_total():
    """The upper-limit likelihood's pointwise terms also sum to its total."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.1, 30, 20)
    flux = get_model("gaussian_rise").predict(TRUE, t, None)
    ul = np.zeros(20, dtype=bool); ul[::5] = True
    lc = wp.LightCurve(time=t, band=["r"] * 20, flux=flux + rng.normal(0, 0.1, 20),
                       flux_err=np.full(20, 0.1), upper_limit=ul, name="ul")
    lik = GaussianLikelihoodWithUpperLimits(lc, space="flux")
    mf = get_model("gaussian_rise").predict(TRUE, t, None)
    pw = lik.log_likelihood_pointwise(mf)
    assert pw.shape == (20,)
    assert np.isclose(pw.sum(), lik.log_likelihood(mf))


def test_waic_keys_finite_and_ordering():
    """WAIC returns the expected fields; a better-fitting posterior has the lower WAIC."""
    lc = _lc()
    rng = np.random.default_rng(1)
    names = list(TRUE)
    good = pd.DataFrame({n: rng.normal(TRUE[n], 0.02 * abs(TRUE[n]), 300) for n in names})
    bad = pd.DataFrame({n: rng.normal(TRUE[n] * 1.5, 0.02 * abs(TRUE[n]), 300) for n in names})
    wg = wp.waic(good, lc, "gaussian_rise", space="flux", max_samples=300, seed=0)
    wb = wp.waic(bad, lc, "gaussian_rise", space="flux", max_samples=300, seed=0)
    assert set(wg) >= {"waic", "lppd", "p_waic", "se", "n_samples", "n_data"}
    assert np.isfinite(wg["waic"]) and wg["p_waic"] > 0 and wg["n_data"] == lc.n_points
    assert wg["waic"] < wb["waic"]


def test_waic_fixed_parameters_and_subsampling():
    """`fixed=` supplies params absent from the posterior columns; `max_samples` caps the draws."""
    lc = _lc()
    rng = np.random.default_rng(2)
    df = pd.DataFrame({n: rng.normal(TRUE[n], 0.02 * abs(TRUE[n]), 500)
                       for n in ["amplitude", "t0", "sigma_rise"]})       # tau_decay omitted
    w = wp.waic(df, lc, "gaussian_rise", space="flux", fixed={"tau_decay": 15.0},
                max_samples=120, seed=0)
    assert np.isfinite(w["waic"]) and w["n_samples"] == 120
