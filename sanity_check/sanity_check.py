#!/usr/bin/env python
"""WHISPER inference-recovery sanity check & benchmark on synthetic data with KNOWN ground truth.

Generate mock light curves ``M(t, θ) + white noise``, fit them with every sampler (MCMC, ABC, ABC-SMC,
and GPU neural SBI), and check each **recovers the truth** with **calibrated uncertainties** — while
timing everything. Mocks: a physically-motivated **Bazin (2009) supernova light curve** (headline
showcase; fit with MDN and NSF density estimators at a 30k-simulation budget), a 4-parameter damped
sinusoid (stress test: correlated, oscillatory), and a 2/4/6-parameter Gaussian-pulse family
(dimensionality sweep). The Bazin neural methods condition on the **stacked (value, err, time, band)
tuple** with GPU-vectorized simulation; the SNPE-5-round benchmark compares no embedding vs an MLP vs a
TCN embedding. ABC/ABC-SMC simulations are **noise-matched** (per-point ``flux_err`` white noise).
Statistics: per-parameter z-score + credible-interval coverage, posterior-predictive checks (reduced χ²
+ predictive coverage), and Simulation-Based Calibration (rank uniformity). Outputs land in
``sanity_check/figures/``.

    # one config at a time (parallel-friendly; give each GPU method its own GPU), then render:
    CUDA_VISIBLE_DEVICES=0 python sanity_check/sanity_check.py fit bazin_sn npe_mdn
    python sanity_check/sanity_check.py sbc bazin_sn mcmc
    python sanity_check/sanity_check.py plot
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
import numpy as np

import whisper_labia as wp
from whisper_labia.models import get_model, register_model
from whisper_labia.models.bazin import bazin_flux            # physically-motivated SN mock (picklable)
from whisper_labia.priors import Prior, Uniform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "sanity_check", "figures")
NOISE = 0.15                       # white-noise sigma (flux units); damped-sine peak ~5 -> SNR ~ 33
# Noise seeds. DATA_SEED is arbitrary (unscreened). SWEEP_SEED/BAZIN_SEED were SCREENED with
# dev/_scan_seed.py to be non-adversarial (worst |MLE-truth|/sigma_Fisher <~ 1): an unlucky draw
# (e.g. a 2.4-sigma noise excursion) makes every method "miss" the truth spuriously, which defeats a
# cross-method comparison table. The screening makes the single-realization recovery/coverage columns
# favourable by construction — they compare methods on a shared, well-posed realization and are NOT
# calibration evidence; calibration is tested by SBC over many unscreened realizations. Disclosed in
# the generated REPORT.md.
DATA_SEED = 1234                   # damped_sine (unscreened)
SWEEP_SEED = 6                     # gp2/gp4/gp6 sweep (screened; worst |z_fisher| ~ 0.9)
BAZIN_SEED = 0                     # Bazin showcase (screened; worst |z_fisher| ~ 0.88)

# ------------------------------------------------------------------ mock models (module-level = picklable)
_PULSE_SIGMA = 1.5                 # fixed Gaussian-pulse width for the sweep family


def bazin_flux_torch(theta, times):
    """Batched torch Bazin: theta (B, 4) = [amplitude, t0, tau_rise, tau_fall], times (n,) -> (B, n).

    Device-agnostic (runs on whatever device the tensors live on) — one kernel launch simulates the
    whole batch, replacing the per-row Python loop for GPU simulation (``predict_torch``)."""
    import torch
    A, t0, tr, tf = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3], theta[:, 3:4]
    dt = times.unsqueeze(0) - t0
    log_flux = -dt / tf - torch.logaddexp(torch.zeros_like(dt), -dt / tr)   # matches models.bazin
    return A * torch.exp(torch.clamp(log_flux, -80.0, 80.0))               # float32-safe clip


def mock_damped_sine_flux(parameters, times, bands=None):
    """M = A*exp(-t/tau)*sin(2*pi*f*t + phi) — 4 correlated, non-linear parameters."""
    t = np.asarray(times, dtype=float)
    return (float(parameters["A"]) * np.exp(-t / float(parameters["tau"]))
            * np.sin(2.0 * np.pi * float(parameters["freq"]) * t + float(parameters["phase"])))


def mock_gauss_pulses_flux(parameters, times, bands=None):
    """M = sum_k A_k * exp(-(t - mu_k)^2 / 2 sigma0^2); reads however many (A_k, mu_k) are present."""
    t = np.asarray(times, dtype=float)
    out = np.zeros_like(t)
    k = 1
    while f"A{k}" in parameters:
        out += float(parameters[f"A{k}"]) * np.exp(
            -(t - float(parameters[f"mu{k}"])) ** 2 / (2.0 * _PULSE_SIGMA ** 2))
        k += 1
    return out


def _gp_spec(k):
    """Build the (params, prior, truth) for a K-pulse Gaussian mock (2K parameters).

    Each pulse position ``mu_k`` gets a **disjoint** prior bin so the pulses are *identifiable*: a sum of
    Gaussians is invariant under permuting the (A_k, mu_k) pairs, so sharing one wide mu prior across
    pulses makes the posterior exchangeable and every sampler is free to label-switch (a spurious
    multi-modal 'failure'). Confining each mu to its own bin breaks that degeneracy — the standard
    ordering constraint for mixture-like models — leaving a clean recovery test of dimensionality.
    """
    params, prior, truth = [], {}, {}
    edges = np.linspace(4.0, 20.0, k + 1)              # k disjoint mu bins -> identifiable pulses
    mus = 0.5 * (edges[:-1] + edges[1:])               # each pulse centred in its bin (off the boundary)
    for i in range(1, k + 1):
        params += [f"A{i}", f"mu{i}"]
        prior[f"A{i}"] = Uniform(0.5, 8.0)
        prior[f"mu{i}"] = Uniform(float(edges[i - 1]), float(edges[i]))
        truth[f"A{i}"] = 2.0 + 1.3 * i
        truth[f"mu{i}"] = float(mus[i - 1])
    return params, Prior(prior), truth


# mock name -> {predict, params, prior, truth, tmax, npts, seed, desc}
MOCKS = {
    "bazin_sn": {
        # Bazin (2009) — THE standard parametric supernova light curve (rise + radioactive-decay fall);
        # peak flux ~3.2 at t~26.6 d -> peak SNR ~21 with sigma=0.15; ~0.9 d cadence over 70 d.
        "predict": bazin_flux,
        "params": ["amplitude", "t0", "tau_rise", "tau_fall"],
        "prior": Prior({"amplitude": Uniform(0.5, 15.0), "t0": Uniform(5.0, 40.0),
                        "tau_rise": Uniform(0.5, 12.0), "tau_fall": Uniform(5.0, 60.0)}),
        "truth": {"amplitude": 5.0, "t0": 20.0, "tau_rise": 4.0, "tau_fall": 25.0},
        "tmax": 70.0, "npts": 80, "seed": BAZIN_SEED, "predict_torch": bazin_flux_torch,
        "desc": "Bazin (2009) SN: `A·exp(−(t−t0)/τ_fall) / (1+exp(−(t−t0)/τ_rise))`", "log": [],
    },
    "damped_sine": {
        "predict": mock_damped_sine_flux,
        "params": ["A", "tau", "freq", "phase"],
        "prior": Prior({"A": Uniform(1.0, 10.0), "tau": Uniform(3.0, 30.0),
                        "freq": Uniform(0.03, 0.40), "phase": Uniform(0.0, 2.0 * np.pi)}),
        "truth": {"A": 5.0, "tau": 10.0, "freq": 0.07, "phase": 1.0},   # ~1.1 periods over t<=16 (mild)
        "tmax": 16.0, "npts": 100, "seed": DATA_SEED,
        "desc": "damped sinusoid: `A·exp(−t/τ)·sin(2πf·t+φ)`", "log": [],
    },
}
for _k in (1, 2, 3):
    _p, _pr, _tr = _gp_spec(_k)
    MOCKS[f"gp{2 * _k}"] = {"predict": mock_gauss_pulses_flux, "params": _p, "prior": _pr,
                            "truth": _tr, "tmax": 24.0, "npts": 90, "seed": SWEEP_SEED, "log": []}

# sampler key -> (label, fit function, kwargs). NPE = 1 round (amortized); SNPE = sequential rounds.
# Bazin neural configs: GPU training + GPU-vectorized simulation, stacked (value, err, time, band)
# input; density estimator MDN (fast mixture) or NSF (expressive spline flow); the snpe5_* trio
# benchmarks no-embedding vs MLP vs TCN. The legacy npe/snpe MAF configs (damped_sine + sweep results)
# keep the raw value-vector input.
SAMPLERS = {
    "mcmc": ("MCMC", wp.fit_MCMC, dict(nsteps=6000, burnin=1500, thin=4, space="flux", seed=0)),
    "abc": ("ABC", wp.fit_ABC, dict(n_simulations=300_000, quantile=0.002, n_jobs=8, seed=0)),
    "abc_smc": ("ABC-SMC", wp.fit_ABC_SMC,
                dict(n_particles=1500, n_rounds=16, quantile=0.5, min_epsilon="auto", n_jobs=8, seed=0)),
    # legacy MAF configs (the damped_sine + gp-sweep results were produced with these)
    "npe": ("NPE-MAF (GPU)", wp.fit_SNPE,
            dict(num_rounds=1, num_simulations=5000, num_samples=6000, space="flux",
                 device="cuda", seed=0, max_num_epochs=200)),
    "snpe": ("SNPE-MAF (GPU)", wp.fit_SNPE,
             dict(num_rounds=10, num_simulations=1000, num_samples=2000, space="flux",
                  device="cuda", seed=0, max_num_epochs=80)),           # SNPE-C, 10 sequential rounds
    # Bazin-showcase NPE: 30k sims, GPU-vectorized simulation (use_torch_sim -> spec["predict_torch"]),
    # stacked (value, err, time, band) input, large batches + tighter early-stop patience for speed.
    "npe_mdn": ("NPE-MDN (GPU)", wp.fit_SNPE,
                dict(num_rounds=1, num_simulations=30_000, num_samples=10_000, space="flux",
                     density_estimator="mdn", x_format="stacked", use_torch_sim=True,
                     device="cuda", seed=0,
                     max_num_epochs=300, training_batch_size=1000, stop_after_epochs=15)),
    "npe_nsf": ("NPE-NSF (GPU)", wp.fit_SNPE,
                dict(num_rounds=1, num_simulations=30_000, num_samples=10_000, space="flux",
                     density_estimator="nsf", x_format="stacked", use_torch_sim=True,
                     device="cuda", seed=0,
                     max_num_epochs=300, training_batch_size=1000, stop_after_epochs=15)),
    # SNPE embedding benchmark: 5 sequential rounds x 3000 sims (NSF), identical except the embedding:
    # none (raw stacked vector) vs MLP compressor vs Temporal Convolutional Network (see embeddings.py).
    # Runs the TRUNCATED sequential scheme (proposal_mode="restricted", TSNPE): SNPE-C's between-round
    # posterior sampling reliably hits nflows spline assertions (leakage) at these budgets.
    "snpe5_plain": ("SNPE-5r (GPU, no embed)", wp.fit_SNPE,
                    dict(num_rounds=5, num_simulations=3000, num_samples=10_000, space="flux",
                         density_estimator="nsf", x_format="stacked", use_torch_sim=True,
                         device="cuda", seed=0, proposal_mode="restricted", support_samples=5000,
                         max_num_epochs=120, training_batch_size=1000, stop_after_epochs=12)),
    "snpe5_mlp": ("SNPE-5r (GPU, MLP embed)", wp.fit_SNPE,
                  dict(num_rounds=5, num_simulations=3000, num_samples=10_000, space="flux",
                       density_estimator="nsf", embedding_net="mlp", embedding_latent=32,
                       x_format="stacked", use_torch_sim=True, device="cuda", seed=0,
                       proposal_mode="restricted", support_samples=5000,
                       max_num_epochs=120, training_batch_size=1000, stop_after_epochs=12)),
    "snpe5_tcn": ("SNPE-5r (GPU, TCN embed)", wp.fit_SNPE,
                  dict(num_rounds=5, num_simulations=3000, num_samples=10_000, space="flux",
                       density_estimator="nsf", embedding_net="tcn", embedding_latent=32,
                       x_format="stacked", use_torch_sim=True, device="cuda", seed=0,
                       proposal_mode="restricted", support_samples=5000,
                       max_num_epochs=120, training_batch_size=1000, stop_after_epochs=12)),
}
# lighter per-fit configs for the many-fit SBC loop (kept close to production so the calibration
# verdict is fair; ABC keeps the production quantile=0.002 tolerance, only trimming the sim budget).
# NPE variants are amortized: train ONCE at full budget, then rank many realisations by conditioning.
SBC_KW = {
    "mcmc": dict(nsteps=2500, burnin=600, thin=3, space="flux"),
    "abc": dict(n_simulations=150_000, quantile=0.002, n_jobs=8),
    "abc_smc": dict(n_particles=800, n_rounds=11, quantile=0.5, min_epsilon="auto", n_jobs=8),
    "npe": dict(num_rounds=1, num_simulations=5000, num_samples=2000, space="flux", device="cuda",
                max_num_epochs=200),
    "npe_mdn": dict(num_rounds=1, num_simulations=30_000, num_samples=2000, space="flux",
                    density_estimator="mdn", x_format="stacked", use_torch_sim=True, device="cuda",
                    max_num_epochs=300, training_batch_size=1000, stop_after_epochs=15),
    "npe_nsf": dict(num_rounds=1, num_simulations=30_000, num_samples=2000, space="flux",
                    density_estimator="nsf", x_format="stacked", use_torch_sim=True, device="cuda",
                    max_num_epochs=300, training_batch_size=1000, stop_after_epochs=15),
}
# per-method realisations (10-round SNPE re-trains per realisation -> prohibitive; NPE covers neural SBC)
SBC_L = {"mcmc": 100, "abc": 100, "abc_smc": 60, "npe": 200, "npe_mdn": 200, "npe_nsf": 200}


def _register(mock):
    spec = MOCKS[mock]
    name = f"mock_{mock}"
    if name not in __import__("whisper_labia.models", fromlist=["list_models"]).list_models():
        register_model(name, spec["predict"], spec["params"], prior=spec["prior"],
                       description=f"sanity-check mock: {mock}")
    return name


def _make_lc(mock, truth, seed):
    spec = MOCKS[mock]
    t = np.linspace(0.5, spec["tmax"], spec["npts"])
    clean = spec["predict"](truth, t, None)
    obs = clean + np.random.default_rng(seed).normal(0.0, NOISE, t.shape)
    return wp.LightCurve(time=t, band=["x"] * len(t), flux=obs,
                         flux_err=np.full_like(t, NOISE), name=f"mock_{mock}")


# --------------------------------------------------------------------------------- one timed fit + metrics
def _resolve_kw(kw, spec):
    """Expand config flags that need the mock spec: use_torch_sim -> the mock's predict_torch."""
    kw = dict(kw)
    if kw.pop("use_torch_sim", False):
        kw["predict_torch"] = spec.get("predict_torch")   # None -> fit_SNPE falls back to numpy loop
    return kw


def fit(mock, sampler):
    name = _register(mock)
    spec = MOCKS[mock]
    lc = _make_lc(mock, spec["truth"], spec["seed"])
    label, fn, kw = SAMPLERS[sampler]
    kw = _resolve_kw(kw, spec)

    t0 = time.perf_counter()
    res = _robust_fit(fn, lc, name, spec["prior"], sampler, kw)
    wall = time.perf_counter() - t0

    # Amortized per-object cost: for a TRAINED network, conditioning on a new observation and drawing
    # 2000 posterior samples is the entire per-object inference (vs a full refit for MCMC/ABC). Timed
    # on a FRESH noise realization (an unseen "new object" on the same grid), not the training x_o.
    amortized = None
    if hasattr(res, "posterior") and hasattr(res, "format_x"):
        t = np.asarray(lc.time, float)
        fresh = (spec["predict"](spec["truth"], t, None)
                 + np.random.default_rng([spec["seed"], 424242]).normal(0.0, NOISE, t.shape))
        xt = res.format_x(fresh)
        t1 = time.perf_counter()
        res.posterior.sample((2000,), x=xt, show_progress_bars=False)
        amortized = time.perf_counter() - t1

    rec = wp.recovery_metrics(res, spec["truth"])
    ppc = wp.posterior_predictive_check(res, lc, name, n_draws=400, seed=0)
    try:
        w = wp.waic(res, lc, name, space="flux", max_samples=1500)["waic"]
    except Exception:
        w = float("nan")

    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"sanity_{mock}_{sampler}.npz"),
             params=np.array(spec["params"]), samples=res.samples[spec["params"]].to_numpy(float),
             time=np.asarray(lc.time, float), obs=np.asarray(lc.flux, float),
             err=np.asarray(lc.flux_err, float),
             ppc_t=ppc["time_grid"], ppc_med=ppc["median"], ppc_lo68=ppc["lo68"], ppc_hi68=ppc["hi68"],
             ppc_lo95=ppc["lo95"], ppc_hi95=ppc["hi95"])
    json.dump({
        "mock": mock, "sampler": sampler, "label": label, "params": spec["params"],
        "truth": spec["truth"], "recovery": rec,
        "ppc": {"reduced_chi2": ppc["reduced_chi2"], "coverage68": ppc["ppc_coverage68"],
                "coverage95": ppc["ppc_coverage95"], "bayesian_p_value": ppc["bayesian_p_value"],
                "dof": ppc["dof"]},
        "runtime_s": float(res.runtime_s), "wall_s": float(wall), "n_samples": int(res.n_samples),
        "amortized_s": (float(amortized) if amortized is not None else None),
        "aic": float(res.aic), "bic": float(res.bic), "waic": float(w),
        "info": {k: res.info.get(k) for k in ("converged", "mean_acceptance_fraction",
                                              "mean_autocorr_time", "device", "density_estimator",
                                              "embedding_net", "num_rounds", "total_simulations",
                                              "x_format", "sim_backend", "simulate_noise")
                 if k in res.info},
    }, open(os.path.join(OUT, f"sanity_{mock}_{sampler}.json"), "w"), indent=2, default=float)
    s = rec["_summary"]
    print(f"[{mock:11s} {sampler:7s}] runtime={res.runtime_s:7.1f}s  max|z|={s['max_abs_z']:.2f}  "
          f"cov95={s['coverage95']:.2f}  chi2_best={ppc['reduced_chi2']:.2f}  "
          f"ppc95={ppc['ppc_coverage95']:.2f}")


def _mle_guess(model, lc, prior, params):
    """Cheap global MLE (differential evolution on the χ²) — a fair, standard MCMC starting point."""
    from scipy.optimize import differential_evolution
    m = get_model(model)
    t, b = np.asarray(lc.time, float), np.asarray(lc.band)
    obs, err = np.asarray(lc.flux, float), np.asarray(lc.flux_err, float)
    bounds = [(prior.distributions[p].low, prior.distributions[p].high) for p in params]

    def nll(x):
        return 0.5 * np.sum(((obs - np.asarray(m.predict(dict(zip(params, x)), t, b), float)) / err) ** 2)

    r = differential_evolution(nll, bounds, seed=0, maxiter=200, tol=1e-8, polish=True)
    return dict(zip(params, [float(v) for v in r.x]))


def _robust_fit(fn, lc, model, prior, sampler, kw):
    """Run a fit. MCMC gets an MLE-based walker start (emcee from a broad prior can't find a narrow
    likelihood ridge); 10-round SNPE falls back to truncated (restricted) SNPE if SNPE-C leaks (NaN)."""
    kw = dict(kw)
    if sampler == "mcmc" and "initial_guess" not in kw:
        kw["initial_guess"] = _mle_guess(model, lc, prior, get_model(model).parameters)
    try:
        return fn(lc, model, prior=prior, **kw)
    except AssertionError:
        if sampler.startswith("snpe"):
            warnings.warn("SNPE-C hit a leakage assertion; retrying with proposal_mode='restricted'.")
            return fn(lc, model, prior=prior, proposal_mode="restricted", support_samples=5000, **kw)
        raise


# --------------------------------------------------------------------------------- simulation-based calib.
def sbc(mock, sampler):
    if sampler not in SBC_L:
        raise SystemExit(f"SBC not run for {sampler!r} (e.g. 10-round SNPE re-training per realisation "
                         "is prohibitive; NPE's amortized SBC covers neural-SBI calibration).")
    name = _register(mock)
    spec = MOCKS[mock]
    prior, params, predict = spec["prior"], spec["params"], spec["predict"]
    t = np.linspace(0.5, spec["tmax"], spec["npts"])
    L = SBC_L[sampler]
    label, fn = SAMPLERS[sampler][0], SAMPLERS[sampler][1]
    kw = _resolve_kw(SBC_KW[sampler], spec)
    ranks = {p: [] for p in params}
    t0 = time.perf_counter()

    if sampler.startswith("npe"):
        # amortized: train ONCE on a reference realisation, then rank many by conditioning on new x.
        ref = _make_lc(mock, spec["truth"], spec["seed"])
        res = _robust_fit(fn, ref, name, prior, sampler, dict(kw, seed=0))
        posterior = res.posterior
        for i in range(L):
            theta = prior.sample(np.random.default_rng([DATA_SEED, i]))
            x = predict(theta, t, None) + np.random.default_rng([DATA_SEED, i, 7]).normal(0, NOISE, t.shape)
            draws = posterior.sample((2000,), x=res.format_x(x),
                                     show_progress_bars=False).detach().cpu().numpy()
            for j, p in enumerate(params):
                ranks[p].append(wp.sbc_rank(draws[:, j], theta[p]))
    else:
        for i in range(L):
            theta = prior.sample(np.random.default_rng([DATA_SEED, i]))
            obs = predict(theta, t, None) + np.random.default_rng([DATA_SEED, i, 7]).normal(0, NOISE, t.shape)
            lc = wp.LightCurve(time=t, band=["x"] * len(t), flux=obs, flux_err=np.full_like(t, NOISE))
            res = _robust_fit(fn, lc, name, prior, sampler, dict(kw, seed=i))   # own chain per realisation
            for p in params:
                ranks[p].append(wp.sbc_rank(np.asarray(res.samples[p], float), theta[p]))

    diag = wp.sbc_ranks({p: np.array(r) for p, r in ranks.items()})
    os.makedirs(OUT, exist_ok=True)
    json.dump({"mock": mock, "sampler": sampler, "label": label, "L": L,
               "runtime_s": time.perf_counter() - t0, "diagnostics": diag},
              open(os.path.join(OUT, f"sanity_sbc_{mock}_{sampler}.json"), "w"), indent=2, default=float)
    print(f"[SBC {mock:11s} {sampler:7s}] L={L}  min uniformity p={diag['_summary']['min_uniformity_p']:.3f}"
          f"  calibrated={diag['_summary']['calibrated']}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "fit":
        fit(a[1], a[2])
    elif len(a) == 3 and a[0] == "sbc":
        sbc(a[1], a[2])
    elif a and a[0] == "plot":
        from sanity_check_plots import plot          # rendering lives in a sibling module
        plot(OUT, MOCKS, SAMPLERS)
    else:
        raise SystemExit(__doc__)
