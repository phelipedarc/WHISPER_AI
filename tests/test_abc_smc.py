import numpy as np

from whisper_labia import LightCurve, fit, fit_ABC_SMC, list_samplers
from whisper_labia.models.flare import flare_flux


def _synthetic(truth, n=50, noise_frac=0.01, seed=0):
    times = np.linspace(0.5, 30, n)
    bands = np.array(["r"] * n)
    flux = flare_flux(truth, times, bands)
    err = np.full_like(flux, noise_frac * flux.max()) + 1e-9
    noisy = flux + np.random.default_rng(seed).normal(0, err)
    return LightCurve(time=times, band=bands, flux=noisy, flux_err=err, name="synth")


def test_smc_registered():
    assert "abc_smc" in list_samplers()


def test_smc_reproducible_and_njobs_independent():
    """Fixed seed -> identical accepted population, independent of n_jobs."""
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=30)
    kw = dict(n_particles=150, n_rounds=3, quantile=0.5, seed=5)
    r1 = fit_ABC_SMC(lc, "flare", n_jobs=1, **kw)
    r1b = fit_ABC_SMC(lc, "flare", n_jobs=1, **kw)
    r4 = fit_ABC_SMC(lc, "flare", n_jobs=4, **kw)
    assert r1.samples.equals(r1b.samples) and r1.best_params == r1b.best_params   # determinism
    assert r1.samples.equals(r4.samples) and r1.best_params == r4.best_params      # n_jobs-independent


def test_smc_recovers_amplitude():
    lc = _synthetic({"amplitude": 5.0, "rise_time": 3.0, "decay_time": 15.0})
    res = fit_ABC_SMC(lc, "flare", n_particles=300, n_rounds=4, quantile=0.5, n_jobs=4, seed=1)
    assert res.n_samples == 300
    assert abs(res.summary["amplitude"]["median"] - 5.0) < 1.5
    assert abs(res.best_params["amplitude"] - 5.0) < 1.0
    assert np.isfinite(res.aic)


def test_smc_epsilon_tightens():
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=30)
    res = fit_ABC_SMC(lc, "flare", n_particles=200, n_rounds=4, quantile=0.5, n_jobs=2, seed=2)
    eps = [r["epsilon"] for r in res.info["rounds"] if r["epsilon"] is not None]
    assert len(eps) >= 2 and all(eps[i + 1] < eps[i] for i in range(len(eps) - 1))


def test_smc_explicit_schedule():
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=20)
    res = fit_ABC_SMC(lc, "flare", n_particles=100, epsilon_schedule=[1e10, 1e8, 1e6], n_jobs=2, seed=3)
    assert res.info["n_rounds"] == 3 and res.n_samples == 100


def test_smc_serial_and_dispatch():
    lc = _synthetic({"amplitude": 4.0, "rise_time": 2.0, "decay_time": 12.0}, n=20)
    res = fit(lc, "flare", sampler="abc_smc", n_particles=100, n_rounds=3, n_jobs=1, seed=4)
    assert res.sampler == "abc_smc" and res.n_samples == 100


def test_smc_on_bazin_model_agnostic():
    from whisper_labia.models.bazin import bazin_flux
    times = np.linspace(0, 40, 40)
    bands = np.array(["g"] * 40)
    truth = {"amplitude": 5.0, "t0": 3.0, "tau_rise": 3.0, "tau_fall": 15.0}
    flux = bazin_flux(truth, times, bands)
    err = np.full_like(flux, 0.02 * flux.max()) + 1e-9
    noisy = flux + np.random.default_rng(0).normal(0, err)
    lc = LightCurve(time=times, band=bands, flux=noisy, flux_err=err)
    res = fit_ABC_SMC(lc, "bazin", n_particles=200, n_rounds=4, quantile=0.5, n_jobs=4, seed=0)
    assert res.n_samples == 200 and res.n_params == 4
