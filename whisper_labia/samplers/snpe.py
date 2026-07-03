"""Sequential Neural Posterior Estimation (SNPE / NPE) via ``sbi``.

Simulation-based inference: instead of an explicit likelihood, SNPE trains a neural density estimator
``q(theta | x)`` on (parameters, simulated light curve) pairs drawn from the prior, then conditions it
on the **observed** light curve to get the posterior. Running it over several *rounds* (each proposing
from the latest posterior) focuses simulations on the good region — the "Sequential" in SNPE.

How it plugs into Whisper:

* **Simulator** = Whisper's forward model. For a parameter vector it calls ``model.predict`` at the
  observed times/bands, maps the prediction into the data space (flux or magnitude, like
  :class:`~whisper_labia.likelihood.GaussianLikelihood`), and adds Gaussian noise with the per-point
  data error — so the implicit likelihood matches Whisper's Gaussian likelihood.
* **Prior** = Whisper's :class:`~whisper_labia.priors.Prior`, adapted to a torch prior (``Uniform`` →
  ``BoxUniform``; mixed ``Uniform``/``LogUniform`` → ``MultipleIndependent``).
* **Result** = the same :class:`~whisper_labia.samplers.SamplerResult` every other sampler returns,
  with exact Gaussian ``max_log_likelihood`` / ``AIC`` / ``BIC`` evaluated at the best posterior draw.
  The trained sbi posterior is attached as ``result.posterior`` (and ``result.posteriors`` per round)
  for resampling / sbi ``pairplot``.

``sbi`` + ``torch`` are the optional ``[sbi]`` extra; they are imported lazily, so importing Whisper
never requires them. ``num_rounds=1`` gives amortized NPE; ``num_rounds>1`` is sequential SNPE.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..models import get_model
from .base import BaseSampler, SamplerResult, summarize_posterior


def _require_sbi():
    """Lazily import sbi/torch; raise a clear, actionable error if the ``[sbi]`` extra is missing."""
    try:
        import torch
        from sbi.inference import NPE, simulate_for_sbi
        from sbi.utils import (
            BoxUniform,
            MultipleIndependent,
            RestrictedPrior,
            get_density_thresholder,
        )
        from sbi.utils.user_input_checks import (
            check_sbi_inputs,
            process_prior,
            process_simulator,
        )
    except Exception as exc:  # pragma: no cover - exercised only without the extra installed
        raise ImportError(
            "SNPE needs the optional 'sbi' + 'torch' dependencies (the `[sbi]` extra). Install with "
            "`pip install 'whisper-labia[sbi]'` (or `pip install sbi torch`).") from exc
    return SimpleNamespace(
        torch=torch, NPE=NPE, simulate_for_sbi=simulate_for_sbi, BoxUniform=BoxUniform,
        MultipleIndependent=MultipleIndependent, RestrictedPrior=RestrictedPrior,
        get_density_thresholder=get_density_thresholder, process_prior=process_prior,
        process_simulator=process_simulator, check_sbi_inputs=check_sbi_inputs)


def _to_torch_prior(prior, sb, device="cpu"):
    """Adapt a Whisper :class:`Prior` to a torch prior sbi can use, on ``device``.

    All-``Uniform`` priors become a single ``BoxUniform`` (fast); a mix that includes ``LogUniform``
    becomes a ``MultipleIndependent`` of per-parameter 1-D distributions (``LogUniform`` = exp of a
    uniform in log-space). Unsupported distribution types raise a clear error. The prior tensors are
    created on ``device`` so it matches the (GPU) training device sbi requires.
    """
    from ..priors import LogUniform, Uniform

    torch = sb.torch
    lows, highs, comps, all_uniform = [], [], [], True
    for name in prior.names:
        d = prior.distributions[name]
        if isinstance(d, Uniform):
            lows.append(float(d.low))
            highs.append(float(d.high))
            comps.append(torch.distributions.Uniform(
                torch.tensor([float(d.low)], device=device), torch.tensor([float(d.high)], device=device)))
        elif isinstance(d, LogUniform):
            all_uniform = False
            base = torch.distributions.Uniform(
                torch.tensor([float(np.log(d.low))], device=device),
                torch.tensor([float(np.log(d.high))], device=device))
            comps.append(torch.distributions.TransformedDistribution(
                base, torch.distributions.ExpTransform()))
        else:
            raise TypeError(
                f"SNPE prior adapter supports Uniform/LogUniform; got {type(d).__name__} for "
                f"parameter {name!r}. Provide a Uniform/LogUniform prior or extend _to_torch_prior.")
    if all_uniform:
        return sb.BoxUniform(low=torch.tensor(lows, device=device), high=torch.tensor(highs, device=device))
    return sb.MultipleIndependent(comps)


def _build_simulator(predict, names, times, bands, model_in_space, sigma, torch, seed, extra=None):
    """A batched sbi simulator: parameter vector(s) -> noisy light curve in the data space.

    The observational noise for each parameter row is seeded from ``(seed, that row's values)``, so it
    is **reproducible** and **independent of how sbi chunks the batch across workers** — sharing a
    single RNG would make all ``num_workers>1`` processes add identical noise. When ``extra`` (a flat
    array of constant context channels: errors / times / band codes) is given, it is appended to every
    row — the ``x_format="stacked"`` layout.
    """
    import zlib

    n_obs = len(times)
    sig = np.asarray(sigma, dtype=float)
    sig = np.where(np.isfinite(sig) & (sig > 0), sig, 1.0)   # guard degenerate/zero errors
    base_seed = int(seed)
    extra = None if extra is None else np.asarray(extra, dtype=float).ravel()

    def simulator(theta):
        th = np.asarray(theta.detach().cpu().numpy(), dtype=float)
        single = th.ndim == 1
        if single:
            th = th[None, :]
        width = n_obs if extra is None else n_obs + len(extra)
        out = np.empty((th.shape[0], width), dtype=float)
        for i in range(th.shape[0]):
            params = {nm: float(th[i, j]) for j, nm in enumerate(names)}
            model_flux = np.asarray(predict(params, times, bands), dtype=float)
            mf = np.nan_to_num(model_in_space(model_flux), nan=0.0, posinf=0.0, neginf=0.0)
            row_seed = (base_seed + zlib.crc32(th[i].astype(np.float64).tobytes())) & 0xFFFFFFFF
            out[i, :n_obs] = mf + np.random.default_rng(row_seed).normal(0.0, sig)
            if extra is not None:
                out[i, n_obs:] = extra
        res = torch.as_tensor(out, dtype=torch.float32)
        return res[0] if single else res

    return simulator


def _torch_simulate(proposal, num_simulations, predict_torch, times_t, sig_t, extra_t, torch, seed,
                    show_progress=False):
    """GPU-vectorized simulation: one batched ``predict_torch`` call replaces the per-row Python loop.

    ``predict_torch(theta, times)`` maps a ``(B, D)`` parameter tensor and ``(n,)`` time tensor to a
    ``(B, n)`` flux tensor **on the same device** — so 30k simulations are a single kernel launch
    instead of 30k Python iterations. Per-point white noise ``N(0, sigma)`` (the reported errors) is
    added on-device with a seeded generator for reproducibility.
    """
    with torch.no_grad():
        try:
            theta = proposal.sample((int(num_simulations),), show_progress_bars=show_progress)
        except TypeError:                              # plain torch priors take no progress kwarg
            theta = proposal.sample((int(num_simulations),))
        theta = theta.to(times_t.device)               # e.g. RestrictedPrior samples on CPU (sbi 0.23)
        flux = predict_torch(theta, times_t)
        flux = torch.nan_to_num(flux.to(times_t.device), nan=0.0, posinf=0.0, neginf=0.0)
        gen = torch.Generator(device=flux.device.type)
        gen.manual_seed(int(seed) & 0x7FFFFFFF)
        x = flux + torch.randn(flux.shape, generator=gen, device=flux.device) * sig_t
        if extra_t is not None:
            x = torch.cat([x, extra_t.unsqueeze(0).expand(x.shape[0], -1)], dim=1)
        return theta.float(), x.float()


def _build_density_estimator(density_estimator, embedding_net, hidden_features,
                             num_transforms, num_bins):
    """Resolve the sbi density estimator.

    * a callable (an already-built ``posterior_nn(...)`` factory) is used as-is;
    * a string with no extra options is passed straight to ``NPE`` (sbi builds the default);
    * a string **plus** an ``embedding_net`` and/or hyperparameters is wrapped in ``posterior_nn`` so
      a custom feature-extractor / architecture can be used (essential for high-dimensional,
      multi-band light curves).
    """
    if callable(density_estimator) and not isinstance(density_estimator, str):
        return density_estimator
    if embedding_net is None and hidden_features is None and num_transforms is None \
            and num_bins is None:
        return density_estimator
    from sbi.neural_nets import posterior_nn
    kwargs = {}
    if embedding_net is not None:
        kwargs["embedding_net"] = embedding_net
    if hidden_features is not None:
        kwargs["hidden_features"] = hidden_features
    if num_transforms is not None:
        kwargs["num_transforms"] = num_transforms
    if num_bins is not None:
        kwargs["num_bins"] = num_bins
    return posterior_nn(model=density_estimator, **kwargs)


def _resolve_device(device, torch):
    """Resolve the requested SNPE training device, falling back to CPU when CUDA is unavailable.

    Accepts ``'cpu'``, ``'cuda'`` / ``'gpu'`` / ``'cuda:N'``, or ``'auto'`` (use CUDA when available,
    else CPU). Requesting a GPU without one available warns and uses CPU rather than crashing.
    """
    import warnings

    d = str(device).lower()
    cuda_ok = bool(torch.cuda.is_available())
    if d == "auto":
        return "cuda" if cuda_ok else "cpu"
    if d in ("gpu", "cuda") or d.startswith("cuda:"):
        if not cuda_ok:
            warnings.warn(f"SNPE device={device!r} requested but CUDA is unavailable; using CPU.",
                          stacklevel=3)
            return "cpu"
        return "cuda" if d == "gpu" else d
    return "cpu"


class SNPESampler(BaseSampler):
    """Sequential Neural Posterior Estimation (sbi ``NPE``). See the module docstring."""

    name = "snpe"

    def fit(self, lc, model, prior=None, *, num_rounds=2, num_simulations=1000, space="auto",
            density_estimator="maf", embedding_net=None, embedding_latent=32, x_format="value",
            predict_torch=None, hidden_features=None, num_transforms=None,
            num_bins=None, proposal_mode="posterior", truncate_quantile=1e-4,
            support_samples=10000, num_samples=10000, device="cpu", seed=0, show_progress=False,
            num_workers=1, max_logl_scan=2000, **train_kwargs):
        """Fit ``lc`` with ``model`` via SNPE/NPE.

        ``num_rounds=1`` is amortized NPE; ``num_rounds>1`` focuses simulations sequentially.
        ``num_simulations`` is *per round*; ``num_workers`` parallelizes simulation. ``space``
        ('auto'|'flux'|'magnitude') sets the data space for the simulator and the Gaussian noise model
        (errors required).

        **Input layout:** ``x_format="value"`` (default) conditions on the data-space values alone;
        ``x_format="stacked"`` conditions on the full observation tuple — four channels
        ``(value, error, time, band code)`` concatenated per point, exactly the information the
        likelihood-based samplers receive. The context channels are constant across simulations for a
        fixed observing grid, but give an embedding net the cadence/noise structure and make the
        network amortizable over observations.

        **Density estimator (flexible):** ``density_estimator`` is an sbi estimator name ('maf', 'nsf',
        'mdn', ...) **or** a pre-built ``posterior_nn(...)`` factory. ``embedding_net`` is ``None``
        (condition on the raw vector), a built-in name — ``"mlp"`` or ``"tcn"`` (Temporal Convolutional
        Network; see :mod:`whisper_labia.embeddings`), compressed to ``embedding_latent`` features —
        or any ``torch.nn.Module``. ``hidden_features`` / ``num_transforms`` / ``num_bins`` build a
        custom architecture via ``sbi``'s ``posterior_nn``.

        **GPU simulation:** pass ``predict_torch(theta, times) -> flux`` (a batched, device-agnostic
        torch implementation of the model: ``(B, D)`` parameters + ``(n,)`` times → ``(B, n)`` flux) to
        replace the per-row Python simulator with a single on-device batched call — per-point white
        noise ``N(0, err)`` is added on-device. Flux-space data only.

        **Sequential scheme:** ``proposal_mode='posterior'`` (default) is SNPE-C (propose from the latest
        posterior). ``proposal_mode='restricted'`` is **truncated SNPE** — each round restricts the prior
        to the high-density region with ``get_density_thresholder(quantile=truncate_quantile)`` +
        ``RestrictedPrior`` (often more robust); the support is estimated from ``support_samples`` draws
        (kept modest; sbi's default of 1e6 can take hours). Extra keyword args pass through to ``NPE.train``
        (e.g. ``max_num_epochs``, ``training_batch_size``, ``stop_after_epochs``). The trained sbi
        posterior is attached as ``result.posterior`` (per round in ``result.posteriors``).

        **Device:** ``device`` selects where the neural density estimator trains — ``'cpu'`` (default),
        ``'cuda'`` / ``'gpu'`` / ``'cuda:N'``, or ``'auto'`` (CUDA when available, else CPU). The torch
        prior and observed data are placed on the device automatically; requesting a GPU without one
        warns and falls back to CPU. The GPU accelerates *training*, not the (CPU) simulator — so it
        helps most with many simulations / large networks (see ``scripts/benchmark_snpe_device.py``).
        """
        if proposal_mode not in ("posterior", "restricted"):
            raise ValueError(f"proposal_mode must be 'posterior' or 'restricted'; got {proposal_mode!r}.")
        if x_format not in ("value", "stacked"):
            raise ValueError(f"x_format must be 'value' or 'stacked'; got {x_format!r}.")
        sb = _require_sbi()
        torch = sb.torch
        model = get_model(model)
        prior = prior if prior is not None else model.default_prior
        if prior is None:
            raise ValueError(f"No prior available for model {model.name!r}; pass prior=...")

        # Reuse the Gaussian likelihood for the data space, observation, errors, and exact metrics.
        from ..likelihood import GaussianLikelihood
        lik = GaussianLikelihood(lc, space=space)
        if predict_torch is not None and lik.space != "flux":
            raise ValueError("predict_torch (GPU simulation) currently supports flux-space data only; "
                             f"got space={lik.space!r}.")

        times = np.asarray(lc.time, dtype=float)
        bands = np.asarray(lc.band)
        predict = model.predict
        param_names = list(prior.names)
        k, n = len(param_names), int(len(times))

        # Constant context channels for the stacked layout: per-point error, time, integer band code.
        sig_clean = np.where(np.isfinite(np.asarray(lik.sigma, float)) & (np.asarray(lik.sigma) > 0),
                             np.asarray(lik.sigma, float), 1.0)
        if x_format == "stacked":
            band_codes = np.unique(bands, return_inverse=True)[1].astype(float)
            extra = np.concatenate([sig_clean, times, band_codes])
            n_channels = 4
        else:
            extra, n_channels = None, 1

        torch.manual_seed(int(seed))                       # BEFORE building nets: embedding init draws
        emb_spec = (embedding_net if isinstance(embedding_net, str)
                    else type(embedding_net).__name__ if embedding_net is not None else None)
        if isinstance(embedding_net, str):
            from ..embeddings import build_embedding
            embedding_net = build_embedding(embedding_net, n_points=n, n_channels=n_channels,
                                            latent_dim=int(embedding_latent))

        device = _resolve_device(device, torch)            # 'auto'/'gpu' -> 'cuda'/'cpu', with fallback
        torch_prior = _to_torch_prior(prior, sb, device)   # prior tensors must match the training device
        torch_prior, _, prior_returns_numpy = sb.process_prior(torch_prior)
        if predict_torch is None:
            simulator = _build_simulator(
                predict, param_names, times, bands, lik.model_in_space, lik.sigma, torch, seed, extra)
            simulator = sb.process_simulator(simulator, torch_prior, prior_returns_numpy)
            sb.check_sbi_inputs(simulator, torch_prior)
            times_t = sig_t = extra_t = None
        else:
            times_t = torch.as_tensor(times, dtype=torch.float32, device=device)
            sig_t = torch.as_tensor(sig_clean, dtype=torch.float32, device=device)
            extra_t = (None if extra is None
                       else torch.as_tensor(extra, dtype=torch.float32, device=device))

        x_np = np.asarray(lik.y, dtype=float)
        if extra is not None:
            x_np = np.concatenate([x_np, extra])
        x_o = torch.as_tensor(x_np, dtype=torch.float32).to(device)
        de_builder = _build_density_estimator(
            density_estimator, embedding_net, hidden_features, num_transforms, num_bins)
        inference = sb.NPE(prior=torch_prior, density_estimator=de_builder,
                           device=device, show_progress_bars=show_progress)
        restricted = proposal_mode == "restricted"

        t0 = time.perf_counter()
        proposal = torch_prior
        posteriors = []
        for r in range(int(num_rounds)):
            if predict_torch is not None:
                theta, x = _torch_simulate(proposal, num_simulations, predict_torch, times_t, sig_t,
                                           extra_t, torch, int(seed) + 7919 * r, show_progress)
            else:
                theta, x = sb.simulate_for_sbi(
                    simulator, proposal, num_simulations=int(num_simulations),
                    num_workers=num_workers, seed=int(seed) + r, show_progress_bar=show_progress)
            if restricted:
                # Truncated SNPE: the proposal is a RestrictedPrior (not a posterior) -> first-round loss.
                de_net = inference.append_simulations(theta, x, exclude_invalid_x=True).train(
                    force_first_round_loss=True, show_train_summary=False, **train_kwargs)
            else:
                de_net = inference.append_simulations(
                    theta, x, proposal=proposal, exclude_invalid_x=True).train(
                    show_train_summary=False, **train_kwargs)
            posterior = inference.build_posterior(de_net)
            posterior.set_default_x(x_o)                      # so result.posterior.sample() needs no x
            posteriors.append(posterior)
            if r < int(num_rounds) - 1:                      # update proposal for the next round
                if restricted:
                    # num_samples_to_estimate_support defaults to 1e6 in sbi (hours of sampling);
                    # cap it to keep truncated SNPE practical.
                    accept_reject_fn = sb.get_density_thresholder(
                        posterior, quantile=float(truncate_quantile),
                        num_samples_to_estimate_support=int(support_samples))
                    proposal = sb.RestrictedPrior(
                        torch_prior, accept_reject_fn, sample_with="rejection", device=device)
                else:
                    proposal = posterior
        runtime = time.perf_counter() - t0

        samples_np = np.asarray(
            posterior.sample((int(num_samples),), x=x_o, show_progress_bars=show_progress)
            .detach().cpu().numpy(), dtype=float)
        samples = pd.DataFrame(samples_np, columns=param_names)

        # Exact Gaussian metrics at the best (max-likelihood) posterior draw.
        scan = samples_np if len(samples_np) <= max_logl_scan else samples_np[:max_logl_scan]
        logls = np.array([
            lik.log_likelihood(predict({nm: float(v) for nm, v in zip(param_names, row)}, times, bands))
            for row in scan], dtype=float)
        if np.any(np.isfinite(logls)):
            best_idx = int(np.nanargmax(logls))
            max_log_likelihood = float(logls[best_idx])
        else:                                       # every scanned draw gave a non-finite likelihood
            import warnings
            warnings.warn("SNPE: all scanned posterior draws have non-finite log-likelihood; "
                          "AIC/BIC are -inf and best_params is the first draw.", stacklevel=2)
            best_idx, max_log_likelihood = 0, float("-inf")
        best_params = {nm: float(scan[best_idx][j]) for j, nm in enumerate(param_names)}

        info = {
            "num_rounds": int(num_rounds), "num_simulations": int(num_simulations),
            "total_simulations": int(num_rounds) * int(num_simulations),
            "density_estimator": density_estimator if isinstance(density_estimator, str) else "custom",
            "embedding_net": emb_spec,
            "x_format": x_format,
            "sim_backend": "torch" if predict_torch is not None else "numpy",
            "proposal_mode": proposal_mode,
            "truncate_quantile": float(truncate_quantile) if proposal_mode == "restricted" else None,
            "space": lik.space, "num_samples": int(num_samples), "device": str(device),
            "seed": int(seed), "num_workers": int(num_workers),
        }
        result = SamplerResult(
            sampler="snpe", model=model.name, parameters=list(param_names), samples=samples,
            summary=summarize_posterior(samples, param_names), best_params=best_params,
            n_data=n, n_params=k, runtime_s=runtime, info=info,
            max_log_likelihood=max_log_likelihood,
            aic=float(-2.0 * max_log_likelihood + 2 * k),
            bic=float(-2.0 * max_log_likelihood + k * np.log(n)),
        )
        # Attach the trained sbi objects for resampling / pairplot (not part of to_json).
        result.posterior = posterior
        result.posteriors = posteriors

        def format_x(values):
            """Map a raw data-space vector (n,) to the network's conditioning input on ``device``,
            appending THIS fit's context channels (errors/times/bands) when ``x_format='stacked'``.
            Use it to condition the amortized posterior on a new observation **taken on the same
            observing grid** (same cadence, errors and bands — e.g. simulation-based calibration
            realizations). An observation with a different grid needs a re-trained network — do not
            feed it through this closure, and note the input length is checked only by the network."""
            v = np.asarray(values, dtype=float).ravel()
            if extra is not None:
                v = np.concatenate([v, extra])
            return torch.as_tensor(v, dtype=torch.float32).to(device)

        result.format_x = format_x
        return result


def fit_SNPE(lc, model="flare", prior=None, **kwargs) -> SamplerResult:
    """Fit ``lc`` with ``model`` via Sequential Neural Posterior Estimation.

    See :meth:`SNPESampler.fit` for options. Requires the optional ``[sbi]`` extra (sbi + torch).
    """
    return SNPESampler().fit(lc, model, prior=prior, **kwargs)
