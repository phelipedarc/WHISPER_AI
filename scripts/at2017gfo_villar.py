#!/usr/bin/env python
"""Villar+2017-style two-component kilonova fit of AT2017GFO with every WHISPER sampler.

The real-world companion of ``scripts/sanity_check.py``: the redback ``two_component_kilonova``
model with **kappa_blue fixed at 0.5 cm^2/g** (lanthanide-poor wind) and redshift fixed at the known
z=0.00984, leaving 7 free physical parameters — M_ej, v_ej and the temperature floor of each
component plus kappa_red — and, for the likelihood-based and neural methods, the **free extra-scatter
term sigma of Villar et al. 2017 (ApJL 851, L21)**, added in quadrature to the reported magnitude
errors (``GaussianLikelihoodWithScatter``):

    ln L = -1/2 sum_i [ (O_i - M_i)^2 / (sigma_i^2 + sigma^2) + ln(2 pi (sigma_i^2 + sigma^2)) ]

Everything is fit in **apparent-magnitude space** (g, r, i; SNR>=3), like Villar et al. The
distance-based ABC family fits the 7 physical parameters only: a chi-square rejection distance is
monotonically penalised by extra simulation noise, so a noise-level parameter is not identifiable by
distance-based ABC (verified on synthetic data — its posterior collapses toward zero scatter).

    python scripts/at2017gfo_villar.py fit mcmc          # one method at a time (parallel-friendly)
    python scripts/at2017gfo_villar.py plot              # figures + REPORT_at2017gfo_villar.md
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
from whisper_labia.models import register_model
from whisper_labia.models.two_component_kilonova import two_component_kilonova_flux
from whisper_labia.priors import LogUniform, Prior, Uniform

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# VILLAR_FULL=1 -> full UV-optical-NIR dataset (11 bands spanning Swift-UV to 2MASS-Ks); default is the
# g/r/i-only reduction. TMAX_DAYS restricts the light curve to the early kilonova (0-30 d rest of decay).
FULL = os.environ.get("VILLAR_FULL") == "1"
# Comparison space for the likelihood/distance: "magnitude" (Villar+17; σ ≈ fractional-flux scatter)
# or "flux" (additive-flux scatter). Set VILLAR_SPACE=flux for the flux-space comparison.
SPACE = os.environ.get("VILLAR_SPACE", "magnitude")
_SPACE_SUFFIX = "_flux" if SPACE == "flux" else ""            # magnitude = no suffix
DATA = os.path.join(HERE, "tests", "data",
                    "at2017gfo_full.csv" if FULL else "at2017gfo.csv")
OUT = os.path.join(HERE, "docs", "figures",
                   ("at2017gfo_villar_full" if FULL else "at2017gfo_villar") + _SPACE_SUFFIX)
BANDS = (["uvot::uvw1", "B", "g", "V", "r", "i", "z", "Y", "J", "H", "Ks"]  # UV -> optical -> NIR
         if FULL else ["g", "r", "i"])
EXPLOSION = 57982.529 if FULL else 57982.0     # GW170817 merger (MJD); g/r/i run used 57982.0
TMAX_DAYS = 30.0
Z_AT = 0.00984                      # GW170817 host (NGC 4993)
KAPPA_BLUE = 0.5                    # cm^2/g, fixed (Villar+17 blue component)
MODEL = "villar17_kilonova"
PARAMS = ["mej_1", "vej_1", "temperature_floor_1",          # blue component
          "mej_2", "vej_2", "kappa_2", "temperature_floor_2"]  # red component (kappa free)
LABELS = {"mej_1": r"$M_{ej}^{blue}$", "vej_1": r"$v_{ej}^{blue}$",
          "temperature_floor_1": r"$T_{floor}^{blue}$", "mej_2": r"$M_{ej}^{red}$",
          "vej_2": r"$v_{ej}^{red}$", "kappa_2": r"$\kappa_{red}$",
          "temperature_floor_2": r"$T_{floor}^{red}$", "sigma": r"$\sigma$"}


def villar17_kilonova_flux(parameters, times, bands=None):
    """Two-component kilonova with kappa_blue and redshift FIXED (module-level -> picklable).

    Fills ``kappa_1`` = 0.5 cm^2/g and ``redshift`` = 0.00984 into the parameter dict and delegates to
    the package model; extra keys (e.g. the likelihood scatter ``sigma``) are ignored downstream."""
    kw = {k: float(parameters[k]) for k in PARAMS}
    kw["kappa_1"] = KAPPA_BLUE
    kw["redshift"] = Z_AT
    return two_component_kilonova_flux(kw, times, bands)


def _register():
    if MODEL not in wp.list_models():
        register_model(MODEL, villar17_kilonova_flux, PARAMS,
                       description="Villar+17 two-component kilonova (kappa_blue=0.5, z fixed)")
    return MODEL


# 7 free physical parameters; sigma = Villar+17 extra scatter (magnitudes), fit by MCMC + neural SBI.
# Ejecta velocity prior is restricted to the PHYSICAL kilonova range 0.05-0.3 c: the wide 0.01-0.7 c
# prior let the exact-likelihood MAP rail to an unphysical 0.7 c blue component (confirmed as a genuine
# but unphysical global likelihood maximum by a from-scratch global optimizer — the g/r/i-only data +
# semi-analytic model prefer a fast, hot blue component that chases the rapid optical decline). Villar+17
# find v_blue = 0.256 c, v_red = 0.149 c, both inside this range.
V_PHYS = Uniform(0.05, 0.3)
PRIOR_PHYS = {
    "mej_1": Uniform(1e-4, 0.1), "vej_1": V_PHYS,
    "temperature_floor_1": LogUniform(100.0, 6000.0),
    "mej_2": Uniform(1e-4, 0.1), "vej_2": V_PHYS,
    "kappa_2": Uniform(1.0, 30.0),
    "temperature_floor_2": LogUniform(100.0, 6000.0),
}
# Extra-scatter prior is space-specific: magnitudes for magnitude space, Jy for flux space (the
# AT2017gfo fluxes span ~1e-6..7e-4 Jy, so a mag-scale prior would swamp the data).
SIGMA_PRIOR = LogUniform(1e-8, 1e-3) if SPACE == "flux" else LogUniform(0.01, 2.0)
PRIOR_FULL = Prior({**PRIOR_PHYS, "sigma": SIGMA_PRIOR})             # + scatter (mag or Jy)
PRIOR_ABC = Prior(dict(PRIOR_PHYS))                                  # distance-based: no scatter dim


def setup():
    lc = wp.load_lightcurve(DATA, explosion_date=EXPLOSION, min_snr=3, bands=BANDS,
                            redshift=Z_AT)
    tp = np.asarray(lc.time, float)                       # phase (days since explosion)
    keep = (tp >= 0.0) & (tp <= TMAX_DAYS)                # early kilonova window
    return lc[keep] if not keep.all() else lc


# method -> (label, fn, prior, kwargs). space set by SPACE; neural = GPU + stacked input.
NEURAL = dict(space=SPACE, scatter_param="sigma", x_format="stacked", device="cuda",
              seed=0, training_batch_size=1000, num_workers=24)
SAMPLERS = {
    "mcmc": ("MCMC", wp.fit_MCMC, PRIOR_FULL,
             # 11-band UVOIR data is ~2x costlier per predict but far more constraining, so fewer steps
             # converge; g/r/i keeps the longer chain. Both sized past 50*tau for converged=True.
             dict(nsteps=8000 if FULL else 12000, burnin=2500 if FULL else 4000, thin=4,
                  nwalkers=32 if FULL else 40, space=SPACE,
                  likelihood="gaussian_scatter", n_jobs=48, seed=0)),
    "abc": ("ABC", wp.fit_ABC, PRIOR_ABC,
            dict(n_simulations=60_000, quantile=0.005, space=SPACE, n_jobs=48, seed=0)),
    "abc_smc": ("ABC-SMC", wp.fit_ABC_SMC, PRIOR_ABC,
                dict(n_particles=800, n_rounds=8, quantile=0.5, min_epsilon="auto",
                     space=SPACE, n_jobs=48, seed=0)),
    "npe_mdn": ("NPE-MDN (GPU)", wp.fit_SNPE, PRIOR_FULL,
                dict(num_rounds=1, num_simulations=25_000, num_samples=10_000,
                     density_estimator="mdn", max_num_epochs=300, stop_after_epochs=15, **NEURAL)),
    "npe_nsf": ("NPE-NSF (GPU)", wp.fit_SNPE, PRIOR_FULL,
                dict(num_rounds=1, num_simulations=25_000, num_samples=10_000,
                     density_estimator="nsf", max_num_epochs=300, stop_after_epochs=15, **NEURAL)),
    "snpe5_nsf": ("SNPE-5r NSF (GPU, no embed)", wp.fit_SNPE, PRIOR_FULL,
                  dict(num_rounds=5, num_simulations=5000, num_samples=10_000,
                       density_estimator="nsf", proposal_mode="restricted", support_samples=5000,
                       max_num_epochs=120, stop_after_epochs=12, **NEURAL)),
    "snpe5_tcn": ("SNPE-5r NSF (GPU, TCN embed)", wp.fit_SNPE, PRIOR_FULL,
                  dict(num_rounds=5, num_simulations=5000, num_samples=10_000,
                       density_estimator="nsf", embedding_net="tcn", embedding_latent=32,
                       proposal_mode="restricted", support_samples=5000,
                       max_num_epochs=120, stop_after_epochs=12, **NEURAL)),
}


def _ppc_arrays(res, lc, model, n_draws=200, seed=0):
    """Posterior-predictive check. Smooth per-band model bands are drawn in **magnitude** for a common
    display across the flux- and magnitude-space fits; the χ²/dof and noise-inflated predictive coverage
    are computed in the **fit's own space** (``SPACE``) so the fitted scatter σ (magnitudes vs Jy) is
    applied consistently."""
    from whisper_labia import GaussianLikelihood
    from whisper_labia.models import get_model
    m = get_model(model)
    rng = np.random.default_rng(seed)
    t = np.asarray(lc.time, float)
    bands = np.asarray(lc.band).astype(str)
    lik = GaussianLikelihood(lc, space=SPACE)                 # observation + errors in the fit space
    y_obs, err = np.asarray(lik.y, float), np.asarray(lik.sigma, float)
    names = [p for p in res.samples.columns if p != "distance"]
    S = res.samples[names].to_numpy(float)
    idx = rng.choice(len(S), size=min(n_draws, len(S)), replace=len(S) < n_draws)

    # redback needs strictly increasing times per band call: keep the smooth grid and the data-time
    # evaluations as two separate predicts, with the data sorted in time (unsorted afterwards).
    tg = np.linspace(max(0.3, t.min()), t.max(), 120)
    grid_t = np.concatenate([tg for _ in BANDS])
    grid_b = np.concatenate([np.array([b] * len(tg)) for b in BANDS])
    order = np.argsort(t, kind="stable")
    inv = np.argsort(order, kind="stable")
    t_sorted, b_sorted = t[order], bands[order]
    curves = np.empty((len(idx), len(tg) * len(BANDS)))       # magnitude, for display
    preds = np.empty((len(idx), len(t)))                      # model in fit space, for coverage
    sig_draw = np.zeros(len(idx))
    for i, j in enumerate(idx):
        th = dict(zip(names, S[j]))
        fg = np.asarray(m.predict(th, grid_t, grid_b), float)
        fd = np.asarray(m.predict(th, t_sorted, b_sorted), float)[inv]
        curves[i] = -2.5 * np.log10(np.clip(fg, 1e-300, None) / 3631.0)
        preds[i] = np.asarray(lik.model_in_space(fd), float)
        sig_draw[i] = float(th.get("sigma", 0.0))
    y_rep = preds + rng.normal(0.0, np.sqrt(err ** 2 + sig_draw[:, None] ** 2))
    lo95, lo68, hi68, hi95 = np.percentile(y_rep, [2.5, 16, 84, 97.5], axis=0)
    inside = lambda lo, hi: float(np.mean((y_obs >= np.minimum(lo, hi)) & (y_obs <= np.maximum(lo, hi))))
    cov68, cov95 = inside(lo68, hi68), inside(lo95, hi95)
    band_curves = {b: np.percentile(curves[:, k * len(tg):(k + 1) * len(tg)], [2.5, 50, 97.5], axis=0)
                   for k, b in enumerate(BANDS)}
    # chi2 at the best draw (in the fit space): vs reported errors alone, and vs errors + fitted scatter
    best = res.best_params
    mfit = np.asarray(lik.model_in_space(np.asarray(m.predict(best, t_sorted, b_sorted), float)[inv]),
                      float)
    dof = max(len(y_obs) - len(names), 1)
    chi2_rep = float(np.sum(((y_obs - mfit) / err) ** 2) / dof)
    s_best = float(best.get("sigma", 0.0))
    chi2_sc = float(np.sum((y_obs - mfit) ** 2 / (err ** 2 + s_best ** 2)) / dof)
    return dict(tgrid=tg, band_curves=band_curves, cov68=cov68, cov95=cov95,
                chi2_reported=chi2_rep, chi2_scatter=chi2_sc, dof=dof, space=SPACE)


def fit(method):
    name = _register()
    lc = setup()
    label, fn, prior, kw = SAMPLERS[method]
    kw = dict(kw)
    if method == "mcmc":                       # seed walkers from the (cheap) ABC best fit if present
        f = os.path.join(OUT, "villar_abc.json")
        if os.path.exists(f):
            best = json.load(open(f))["best_params"]
            best["sigma"] = 0.2                # scatter start: typical Villar+17 posterior scale
            kw["initial_guess"] = best
            kw["initial_scatter"] = 1e-2

    t0 = time.perf_counter()
    res = fn(lc, name, prior=prior, **kw)
    wall = time.perf_counter() - t0

    amortized = None
    if hasattr(res, "posterior") and hasattr(res, "format_x"):
        xt = res.format_x(np.asarray(lc.magnitude, float))
        t1 = time.perf_counter()
        res.posterior.sample((2000,), x=xt, show_progress_bars=False)
        amortized = time.perf_counter() - t1

    ppc = _ppc_arrays(res, lc, name, n_draws=200, seed=0)
    names = [p for p in res.samples.columns if p != "distance"]
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, f"villar_{method}.npz"),
             params=np.array(names), samples=res.samples[names].to_numpy(float),
             time=np.asarray(lc.time, float), band=np.asarray(lc.band).astype(str),
             mag=np.asarray(lc.magnitude, float), mag_err=np.asarray(lc.magnitude_err, float),
             tgrid=ppc["tgrid"],
             **{f"curve_{b}": ppc["band_curves"][b] for b in BANDS})
    json.dump({
        "method": method, "label": label, "params": names,
        "summary": {p: res.summary[p] for p in names},
        "best_params": res.best_params,
        "ppc": {k: ppc[k] for k in ("cov68", "cov95", "chi2_reported", "chi2_scatter", "dof", "space")},
        "runtime_s": float(res.runtime_s), "wall_s": float(wall),
        "amortized_s": (float(amortized) if amortized is not None else None),
        "n_samples": int(res.n_samples), "aic": float(res.aic), "bic": float(res.bic),
        "max_log_likelihood": float(res.max_log_likelihood),
        "info": {k: v for k, v in res.info.items()
                 if isinstance(v, (int, float, str, bool, type(None)))},
    }, open(os.path.join(OUT, f"villar_{method}.json"), "w"), indent=2, default=float)
    s = res.summary
    sig = (" sigma=%.3f" % s["sigma"]["median"]) if "sigma" in s else ""
    print(f"[villar {method:10s}] wall={wall:7.1f}s  chi2_rep={ppc['chi2_reported']:.2f}  "
          f"chi2+sc={ppc['chi2_scatter']:.2f}  cov95={ppc['cov95']:.2f}{sig}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 2 and a[0] == "fit":
        fit(a[1])
    elif a and a[0] == "plot":
        from at2017gfo_villar_plots import plot
        plot(OUT, SAMPLERS, PARAMS, LABELS, BANDS)
    else:
        raise SystemExit(__doc__)
