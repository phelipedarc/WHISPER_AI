"""SNPE / NPE sampler (sbi). Training tests skip cleanly when the [sbi] extra is absent."""
import numpy as np
import pytest

import whisper_labia as wp
from whisper_labia.models import get_model


def test_snpe_registered_without_sbi():
    """Registry + import work even without sbi installed (sbi is imported lazily)."""
    assert "snpe" in wp.list_samplers()
    assert "npe" in wp.list_samplers()


# Everything below needs the optional [sbi] extra (sbi + torch).
sbi = pytest.importorskip("sbi")
from whisper_labia.samplers.snpe import _require_sbi, _to_torch_prior  # noqa: E402


def test_torch_prior_adapter_uniform_and_loguniform():
    sb = _require_sbi()
    # all-Uniform -> BoxUniform
    tp = _to_torch_prior(wp.Prior({"a": wp.Uniform(0, 2), "b": wp.Uniform(-1, 1)}), sb)
    s = tp.sample((200,))
    assert tuple(s.shape) == (200, 2)
    assert float(s[:, 0].min()) >= 0 and float(s[:, 0].max()) <= 2
    assert float(s[:, 1].min()) >= -1 and float(s[:, 1].max()) <= 1
    # mixed Uniform + LogUniform -> MultipleIndependent; log-param stays within (positive) bounds
    tp2 = _to_torch_prior(wp.Prior({"a": wp.Uniform(0, 1), "tau": wp.LogUniform(1, 100)}), sb)
    s2 = tp2.sample((200,))
    assert tuple(s2.shape) == (200, 2)
    assert float(s2[:, 1].min()) >= 1 and float(s2[:, 1].max()) <= 100


def test_torch_prior_adapter_rejects_unknown_dist():
    sb = _require_sbi()

    class Weird:
        pass

    p = wp.Prior({"a": wp.Uniform(0, 1)})
    p.distributions["a"] = Weird()
    with pytest.raises(TypeError, match="Uniform/LogUniform"):
        _to_torch_prior(p, sb)


def _synthetic_flare_lc(seed=0, n=40):
    m = get_model("flare")
    t = np.linspace(0.1, 10, n)
    flux = m.predict({"amplitude": 1.0, "rise_time": 1.0, "decay_time": 3.0}, t, None)
    err = np.full_like(flux, 0.05)
    obs = flux + np.random.default_rng(seed).normal(0, err)
    return wp.LightCurve(time=t, band=["r"] * n, flux=obs, flux_err=err, name="synth")


@pytest.mark.slow
def test_snpe_fit_end_to_end():
    """One NPE round on synthetic flare data: structural correctness + prior-support sanity.

    (Tight parameter recovery needs many more simulations/rounds; that's exercised in the demo, not
    in this fast CI test.)
    """
    lc = _synthetic_flare_lc()
    prior = wp.Prior({"amplitude": wp.Uniform(0, 5),
                      "rise_time": wp.Uniform(0.1, 5),
                      "decay_time": wp.Uniform(0.5, 10)})
    res = wp.fit_SNPE(lc, "flare", prior=prior, num_rounds=1, num_simulations=400,
                      num_samples=500, seed=0, max_num_epochs=40)

    assert res.sampler == "snpe" and res.model == "flare"
    assert res.samples.shape == (500, 3)
    assert list(res.samples.columns) == ["amplitude", "rise_time", "decay_time"]
    assert set(res.best_params) == {"amplitude", "rise_time", "decay_time"}
    assert np.isfinite(res.aic) and np.isfinite(res.bic) and np.isfinite(res.max_log_likelihood)
    assert res.n_data == len(lc) and res.n_params == 3
    # samples respect the prior support -> validates the prior adapter (bounds + ordering)
    assert res.samples["amplitude"].between(0, 5).all()
    assert res.samples["rise_time"].between(0.1, 5).all()
    assert res.samples["decay_time"].between(0.5, 10).all()
    # trained sbi posterior is attached for resampling / pairplot
    assert hasattr(res, "posterior") and len(res.posteriors) == 1
    extra = res.posterior.sample((50,), show_progress_bars=False)   # resample from the trained posterior
    assert tuple(extra.shape) == (50, 3)
    import json
    json.loads(res.to_json())   # JSON-serializable (no torch objects leak into to_dict)


@pytest.mark.slow
def test_snpe_dispatch_and_alias_multiround():
    lc = _synthetic_flare_lc(seed=1)
    prior = wp.Prior({"amplitude": wp.Uniform(0, 5),
                      "rise_time": wp.Uniform(0.1, 5),
                      "decay_time": wp.Uniform(0.5, 10)})
    res = wp.fit(lc, "flare", sampler="npe", prior=prior, num_rounds=2, num_simulations=250,
                 num_samples=300, seed=1, max_num_epochs=25)
    assert res.sampler == "snpe" and res.info["num_rounds"] == 2
    assert res.info["total_simulations"] == 500
    assert res.n_samples == 300
