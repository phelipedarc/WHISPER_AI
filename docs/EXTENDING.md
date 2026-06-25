# Extending Whisper

Whisper has two pluggable axes — **models** and **samplers** — plus a pluggable **distance**. Each is
a tiny registry, so you can add your own without touching the core.

## Add a model

A model maps parameters to predicted flux: `predict(params: dict, times, bands) -> np.ndarray`.
Keep it **vectorized** (no per-point loops), and for **parallel** ABC define it at **module level**
(closures/lambdas only work with `n_jobs=1`, because parallel workers pickle the function).

```python
import numpy as np
import whisper_labia as wp

def powerlaw(params, times, bands=None):
    return params["amplitude"] * np.clip(times, 1e-6, None) ** (-params["index"])

wp.register_model(
    "powerlaw", powerlaw, ["amplitude", "index"],
    prior=wp.Prior({"amplitude": wp.Uniform(0, 10), "index": wp.Uniform(0, 3)}),
    description="Simple power-law decay.",
)

print("powerlaw" in wp.list_models())     # True
res = wp.fit_ABC(lc, "powerlaw")          # usable by name with any sampler
```

`bands` is passed for generality; ignore it for band-independent models. See
`whisper_labia/models/{flare,bazin,gaussian_rise}.py` for references.

## Add a sampler

Subclass `BaseSampler`, implement `fit(lc, model, prior=None, **kwargs) -> SamplerResult`, and register
it. Reuse the building blocks: `get_model`, the model's `predict`, the prior's `sample` / `log_prob` /
`rescale`, a distance (or your own likelihood), and `summarize_posterior`.

```python
import time
import numpy as np
import pandas as pd
import whisper_labia as wp
from whisper_labia.samplers import register_sampler
from whisper_labia.samplers.base import BaseSampler, SamplerResult, summarize_posterior
from whisper_labia.models import get_model
from whisper_labia.distance import chi2_distance

class RandomSearch(BaseSampler):
    name = "random_search"

    def fit(self, lc, model, prior=None, *, n=2000, keep=0.05, seed=0, **kw):
        model = get_model(model)
        prior = prior or model.default_prior
        lc = lc.add_flux()
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()

        rows = []
        for _ in range(n):
            theta = prior.sample(rng)
            sim = model.predict(theta, lc.time, lc.band)
            theta["distance"] = chi2_distance(lc.flux, lc.flux_err, sim)
            rows.append(theta)

        df = pd.DataFrame(rows).nsmallest(int(n * keep), "distance")
        chi2_min = float(df["distance"].min())
        k, ndata = len(model.parameters), lc.n_points
        best = df.iloc[0][model.parameters].to_dict()
        return SamplerResult(
            sampler=self.name, model=model.name, parameters=model.parameters,
            samples=df, summary=summarize_posterior(df, model.parameters),
            best_params={p: float(v) for p, v in best.items()},
            n_data=ndata, n_params=k, runtime_s=time.perf_counter() - t0, info={"n": n},
            min_distance=chi2_min, max_log_likelihood=-0.5 * chi2_min,
            aic=chi2_min + 2 * k, bic=chi2_min + k * np.log(ndata),
        )

register_sampler("random_search", RandomSearch)
res = wp.fit(lc, "flare", sampler="random_search", n=5000)   # dispatch by name
```

The full ABC and ABC-SMC implementations live in `whisper_labia/samplers/abc.py` and `abc_smc.py`.

## Add a distance

A distance is any `f(obs_flux, obs_flux_err, sim_flux, bands) -> float`. Pass it as `distance=` to a
sampler:

```python
def mae_distance(obs, err, sim, bands=None):
    return float(np.sum(np.abs((obs - sim) / err)))

wp.fit_ABC(lc, "flare", distance=mae_distance)
```
