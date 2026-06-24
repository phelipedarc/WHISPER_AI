"""Phase-0 smoke test (run inside the ``whisper_dev`` container).

Validates the two redback touch-points Whisper relies on, plus the beta samplers:

  1. redback model callable: parameters -> predicted observables
  2. ``redback.priors.get_priors`` -> bilby ``PriorDict``
  3. ``emcee`` + ``dynesty`` import
  4. ``import whisper_labia``

Exits non-zero on failure so it can gate CI.
"""
from __future__ import annotations

import inspect
import sys

import numpy as np


def main() -> int:
    import whisper_labia as wp
    print(f"[ok] whisper_labia {wp.__version__}")

    import emcee
    import dynesty
    print(f"[ok] emcee {emcee.__version__} | dynesty {dynesty.__version__}")

    import redback  # noqa: F401
    from redback.model_library import all_models_dict
    from redback.priors import get_priors

    model_name = "arnett"
    model = all_models_dict[model_name]
    print(f"[ok] redback model callable: {model_name} -> {getattr(model, '__name__', model)}")

    # (2) priors
    priors = get_priors(model=model_name)
    print(f"[ok] get_priors({model_name}): {len(priors)} params -> {sorted(priors.keys())}")

    # (1) sample a parameter set from the priors and evaluate the model at a few epochs.
    sample = priors.sample()
    sample.setdefault("redshift", 0.1)

    times = np.linspace(1.0, 40.0, 8)          # days since explosion
    bands = np.array(["ztfg"] * len(times))

    sig = inspect.signature(model)
    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )

    def accepts(name: str) -> bool:
        return accepts_var_kw or name in sig.parameters

    call_kwargs = {k: v for k, v in sample.items() if accepts(k)}
    if accepts("bands"):
        call_kwargs["bands"] = bands
    if accepts("output_format"):
        call_kwargs["output_format"] = "magnitude"

    out = np.atleast_1d(np.asarray(model(times, **call_kwargs), dtype=float))
    finite = np.isfinite(out)
    print(f"[ok] model evaluated: {out.shape} values, {int(finite.sum())}/{out.size} finite")
    if finite.any():
        print(f"     first finite magnitudes: {np.round(out[finite][:5], 3)}")

    assert out.size >= 1 and finite.any(), "model produced no finite observables"

    print("\nPHASE-0 SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
