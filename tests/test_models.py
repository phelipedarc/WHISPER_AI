import numpy as np
import pytest

from whisper_labia import Prior, Uniform, get_model, list_models, register_model
from whisper_labia.models.flare import flare_flux


def test_flare_registered():
    assert "flare" in list_models()
    m = get_model("flare")
    assert m.parameters == ["amplitude", "rise_time", "decay_time"]
    assert m.default_prior is not None


def test_flare_flux_vectorized():
    t = np.array([0.0, 1.0, 5.0, 10.0])
    f = flare_flux({"amplitude": 2.0, "rise_time": 2.0, "decay_time": 10.0}, t)
    assert f.shape == t.shape
    assert np.isclose(f[0], 0.0)     # (1 - exp(0)) == 0 at t=0
    assert np.all(f[1:] > 0)


def test_register_custom_model():
    def quad(params, times, bands=None):
        return params["a"] * np.asarray(times, float) ** 2

    register_model("quad_test", quad, ["a"], prior=Prior({"a": Uniform(0, 1)}), overwrite=True)
    assert "quad_test" in list_models()
    assert np.allclose(get_model("quad_test")({"a": 2.0}, [1, 2, 3]), [2, 8, 18])


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("does_not_exist")
