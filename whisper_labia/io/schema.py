"""Canonical light-curve container shared across Whisper."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .photometry import (
    AB_ZEROPOINT_JY,
    flux_density_to_mag,
    mag_err_to_snr,
    mag_to_flux_density,
)

#: Allowed values for :attr:`LightCurve.data_mode`.
VALID_DATA_MODES = ("flux_density", "magnitude", "flux")

#: data_mode -> the forward-model comparison space (``magnitude`` | ``flux_density``); band-integrated
#: ``flux`` maps onto ``flux_density``. (Optional physical-model backends such as redback expose this
#: same two-value ``output_format``, so the two line up.)
_OUTPUT_FORMAT = {"flux_density": "flux_density", "magnitude": "magnitude", "flux": "flux_density"}

#: Default prior spec attached when the redshift is unknown (override per object). The actual
#: sampling lives in Phase 2 (priors.py / PriorSet); here it is just a serialisable hint.
DEFAULT_REDSHIFT_PRIOR = {"type": "Uniform", "low": 0.001, "high": 1.0, "name": "redshift"}


def _str_array(x):
    return np.array([str(v) for v in x])


@dataclass
class LightCurve:
    """A multi-band transient light curve.

    Per-point arrays (all the same length): ``time`` (MJD), ``band``, and at least one of
    ``magnitude`` / ``flux`` (with optional ``*_err``). ``system`` records the magnitude system
    per point ('AB'/'Vega'/'unknown'); ``upper_limit`` flags non-detections; ``lambda_eff`` /
    ``zero_point`` carry the per-point effective wavelength (Angstrom) and zero point (Jy) once the
    bands are resolved (FILTER_LOOKUP/SVO). Scalar metadata: ``name``, ``redshift``,
    ``luminosity_distance`` (Mpc; required when ``redshift == 0``), ``data_mode``, ``meta``.

    ``data_mode`` is one of :data:`VALID_DATA_MODES` -- ``"flux_density"`` (canonical unit Jy,
    default), ``"magnitude"`` (dimensionless AB), or ``"flux"`` (band-integrated erg/s/cm^2). When
    not given it is inferred from which columns are present. Use :attr:`output_format` for the
    forward-model comparison space (``magnitude`` | ``flux_density``).

    Subsetting methods (``select_*``) and column-adding methods (``add_*``) follow a pandas-like
    naming convention and each return a **new** ``LightCurve``. Calling the object (``lc()``) returns
    the enriched :class:`pandas.DataFrame`: for ``flux_density`` / ``magnitude`` data the missing one
    of flux/magnitude is derived from the per-band zero point; band-integrated ``flux`` mode keeps
    only the flux column (no density<->magnitude mapping applies).
    """

    time: np.ndarray
    band: np.ndarray
    magnitude: Optional[np.ndarray] = None
    magnitude_err: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    flux_err: Optional[np.ndarray] = None
    upper_limit: Optional[np.ndarray] = None
    system: Optional[np.ndarray] = None
    name: Optional[str] = None
    redshift: Optional[float] = None
    meta: dict = field(default_factory=dict)
    data_mode: Optional[str] = None
    lambda_eff: Optional[np.ndarray] = None
    zero_point: Optional[np.ndarray] = None
    luminosity_distance: Optional[float] = None
    redshift_prior: Optional[dict] = None

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        n = self.time.size
        if self.band is None:
            raise ValueError("LightCurve requires a 'band' array.")
        self.band = _str_array(self.band)
        if self.band.size != n:
            raise ValueError(f"band length {self.band.size} != time length {n}")
        for attr in ("magnitude", "magnitude_err", "flux", "flux_err", "lambda_eff", "zero_point"):
            v = getattr(self, attr)
            if v is not None:
                v = np.asarray(v, dtype=float)
                if v.size != n:
                    raise ValueError(f"{attr} length {v.size} != time length {n}")
                setattr(self, attr, v)
        if self.upper_limit is not None:
            self.upper_limit = np.asarray(self.upper_limit, dtype=bool)
            if self.upper_limit.size != n:
                raise ValueError(f"upper_limit length {self.upper_limit.size} != time length {n}")
        if self.system is not None:
            self.system = _str_array(self.system)
            if self.system.size != n:
                raise ValueError(f"system length {self.system.size} != time length {n}")
        if self.magnitude is None and self.flux is None:
            raise ValueError("LightCurve requires either 'magnitude' or 'flux'.")

        self.data_mode = self._resolve_data_mode(self.data_mode)
        self._validate_redshift()
        if not self.redshift_known and self.redshift_prior is None:
            self.redshift_prior = dict(DEFAULT_REDSHIFT_PRIOR)

    # --- data mode ---
    def _resolve_data_mode(self, data_mode):
        if data_mode is not None:
            if data_mode not in VALID_DATA_MODES:
                raise ValueError(
                    f"data_mode={data_mode!r} invalid; expected one of {VALID_DATA_MODES}.")
            return data_mode
        # Infer from the columns present (keeps historical behaviour: mag data -> 'magnitude').
        if self.magnitude is not None:
            return "magnitude"
        return "flux_density"

    @property
    def output_format(self):
        """Forward-model comparison space for this data mode (``'magnitude'`` | ``'flux_density'``).

        Optional physical-model backends (e.g. redback) expose this same two-value convention.
        """
        return _OUTPUT_FORMAT[self.data_mode]

    # --- redshift ---
    def _validate_redshift(self):
        z = self.redshift
        if z is None:
            return  # unknown -- handled by redshift_prior; loader warns once.
        z = float(z)
        if not np.isfinite(z):
            raise ValueError(f"redshift must be finite; got {self.redshift!r}.")
        if z < 0:
            raise ValueError(f"redshift must be >= 0; got {z}.")
        if z == 0 and self.luminosity_distance is None:
            raise ValueError(
                "redshift == 0 leaves the luminosity distance undefined; pass an explicit "
                "luminosity_distance (Mpc) instead of z=0.")
        self.redshift = z

    @property
    def redshift_known(self):
        """``True`` when a redshift was supplied; ``False`` means a prior must be sampled."""
        return self.redshift is not None

    # --- basic info ---
    def __len__(self):
        return int(self.time.size)

    @property
    def n_points(self):
        return int(self.time.size)

    @property
    def bands(self):
        return sorted(set(self.band.tolist()))

    @property
    def snr(self):
        """Per-point signal-to-noise ratio.

        Uses ``flux / flux_err`` when flux errors are present, else the magnitude-error relation
        ``SNR = (2.5 / ln 10) / magnitude_err``. Raises ``ValueError`` if no errors are available.
        """
        if self.flux is not None and self.flux_err is not None:
            return np.abs(self.flux) / self.flux_err
        if self.magnitude_err is not None:
            return mag_err_to_snr(self.magnitude_err)
        raise ValueError("Cannot compute SNR: need flux_err or magnitude_err.")

    # --- copy / subset (return new LightCurves) ---
    def _subset(self, mask):
        def sub(v):
            return None if v is None else v[mask]

        return LightCurve(
            time=self.time[mask], band=self.band[mask],
            magnitude=sub(self.magnitude), magnitude_err=sub(self.magnitude_err),
            flux=sub(self.flux), flux_err=sub(self.flux_err),
            upper_limit=sub(self.upper_limit), system=sub(self.system),
            lambda_eff=sub(self.lambda_eff), zero_point=sub(self.zero_point),
            name=self.name, redshift=self.redshift, data_mode=self.data_mode,
            luminosity_distance=self.luminosity_distance,
            redshift_prior=None if self.redshift_prior is None else dict(self.redshift_prior),
            meta=dict(self.meta),
        )

    def _copy(self):
        return self._subset(np.ones(self.n_points, dtype=bool))

    def select_bands(self, bands):
        """Keep only the given band(s) (str or iterable)."""
        bands = [bands] if isinstance(bands, str) else list(bands)
        return self._subset(np.isin(self.band, bands))

    def select_time_window(self, time_min=None, time_max=None):
        """Keep points with ``time_min <= time <= time_max``."""
        mask = np.ones(self.n_points, dtype=bool)
        if time_min is not None:
            mask &= self.time >= time_min
        if time_max is not None:
            mask &= self.time <= time_max
        return self._subset(mask)

    def select_snr(self, min_snr=5.0):
        """Keep points with signal-to-noise ratio >= ``min_snr`` (e.g. 3 or 5)."""
        return self._subset(self.snr >= min_snr)

    # --- band resolution (effective wavelength + zero point) ---
    def _resolved_zero_point(self, svo_fallback=False):
        """Per-point zero point (Jy): stored values if present, else resolve bands locally.

        Unresolved bands (NaN) fall back to the AB 3631 Jy zero point so conversions stay finite.
        """
        if self.zero_point is not None:
            zp = self.zero_point
        else:
            from .bands import resolve_bands
            _, zp, _ = resolve_bands(self.band, svo_fallback=svo_fallback, warn=False)
        zp = np.asarray(zp, dtype=float)
        return np.where(np.isfinite(zp), zp, AB_ZEROPOINT_JY)

    def resolve_bands(self, *, svo_fallback=True):
        """Copy with ``lambda_eff`` / ``zero_point`` filled from FILTER_LOOKUP (+ SVO fallback)."""
        from .bands import resolve_bands as _resolve
        lam, zp, _ = _resolve(self.band, svo_fallback=svo_fallback)
        out = self._copy()
        out.lambda_eff = lam
        out.zero_point = zp
        return out

    # --- derived columns ---
    def add_flux(self, zeropoint_jy=AB_ZEROPOINT_JY):
        """Copy with ``flux`` (+ ``flux_err``) from magnitude (AB). No-op if flux present.

        Uses the constant AB ``zeropoint_jy`` (3631 Jy) so the modelling flux stays on one zero point
        for all bands -- this matches what the samplers / likelihood / plotting consume. For the
        physical per-band conversion (LSST/SVO zero points) use the enriched dataframe from
        :meth:`__call__` / :meth:`to_dataframe` instead. Raises for band-integrated ``flux`` data.
        """
        if self.data_mode == "flux":
            raise ValueError(
                "add_flux is for flux-density data; data_mode='flux' is band-integrated energy "
                "flux (erg/s/cm^2) and has no magnitude<->flux-density mapping.")
        if self.flux is not None:
            return self
        if self.magnitude is None:
            raise ValueError("Cannot add flux: no magnitude available.")
        out = self._copy()
        if self.magnitude_err is None:
            out.flux = mag_to_flux_density(self.magnitude, None, zeropoint_jy)
        else:
            out.flux, out.flux_err = mag_to_flux_density(
                self.magnitude, self.magnitude_err, zeropoint_jy)
        return out

    def add_mag(self, zeropoint_jy=AB_ZEROPOINT_JY):
        """Copy with ``magnitude`` (+ ``magnitude_err``) from flux (AB). No-op if magnitude present.

        Uses the constant AB ``zeropoint_jy`` (3631 Jy); see :meth:`add_flux` for why, and use the
        enriched dataframe for per-band zero points. Raises for band-integrated ``flux`` data.
        """
        if self.data_mode == "flux":
            raise ValueError(
                "add_mag is for flux-density data; data_mode='flux' is band-integrated energy "
                "flux (erg/s/cm^2) and has no flux-density<->magnitude mapping.")
        if self.magnitude is not None:
            return self
        if self.flux is None:
            raise ValueError("Cannot add magnitude: no flux available.")
        out = self._copy()
        if self.flux_err is None:
            out.magnitude = flux_density_to_mag(self.flux, None, zeropoint_jy)
        else:
            out.magnitude, out.magnitude_err = flux_density_to_mag(
                self.flux, self.flux_err, zeropoint_jy)
        return out

    def set_explosion_date(self, explosion_date):
        """Copy with time measured in **days since ``explosion_date`` (MJD)**; day 0 = explosion.

        Records ``meta['explosion_mjd']``. Epochs before the explosion become negative.
        """
        out = self._copy()
        out.time = out.time - float(explosion_date)
        out.meta["explosion_mjd"] = float(explosion_date)
        return out

    # --- export ---
    def _enriched(self):
        """Return (magnitude, flux) arrays with the missing one filled via per-band zero points.

        Band-integrated ``flux`` data is left as-is (no density<->magnitude conversion applies).
        Points whose band could not be resolved (zero point NaN) keep ``NaN`` in the derived column.
        """
        magnitude, flux = self.magnitude, self.flux
        if self.data_mode == "flux":
            return magnitude, flux
        zp = self._resolved_zero_point()
        if magnitude is None and flux is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                magnitude = flux_density_to_mag(flux, None, zp)
        elif flux is None and magnitude is not None:
            with np.errstate(invalid="ignore"):
                flux = mag_to_flux_density(magnitude, None, zp)
        return magnitude, flux

    def to_dataframe(self):
        """Tabular view (``pandas.DataFrame``).

        Includes both ``magnitude`` and ``flux`` (the missing one derived from the per-band zero
        point), ``lambda_eff`` / ``zero_point`` when known, and ``snr`` when errors are available.
        """
        magnitude, flux = self._enriched()
        data = {"time": self.time, "band": self.band}
        derived = {"magnitude": magnitude, "magnitude_err": self.magnitude_err,
                   "flux": flux, "flux_err": self.flux_err,
                   "upper_limit": self.upper_limit, "system": self.system,
                   "lambda_eff": self.lambda_eff, "zero_point": self.zero_point}
        for key, v in derived.items():
            if v is not None:
                data[key] = v
        try:
            data["snr"] = self.snr
        except ValueError:
            pass
        return pd.DataFrame(data)

    def __call__(self):
        """Return the enriched :class:`pandas.DataFrame` (alias for :meth:`to_dataframe`)."""
        return self.to_dataframe()

    def __repr__(self):
        z = "unknown" if self.redshift is None else self.redshift
        return (f"LightCurve(name={self.name!r}, n_points={self.n_points}, "
                f"bands={self.bands}, mode={self.data_mode!r}, z={z})")
