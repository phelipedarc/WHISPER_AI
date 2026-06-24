"""Canonical light-curve container shared across Whisper."""
from __future__ import annotations

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


def _str_array(x):
    return np.array([str(v) for v in x])


@dataclass
class LightCurve:
    """A multi-band transient light curve.

    Per-point arrays (all the same length): ``time`` (MJD), ``band``, and at least one of
    ``magnitude`` / ``flux`` (with optional ``*_err``). ``system`` records the magnitude system
    per point ('AB'/'Vega'/'unknown'); ``upper_limit`` flags non-detections. Scalar metadata:
    ``name``, ``redshift``, ``meta``.

    Subsetting methods (``select_*``) and column-adding methods (``add_*``) follow a pandas-like
    naming convention and each return a **new** ``LightCurve``.
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

    def __post_init__(self):
        self.time = np.asarray(self.time, dtype=float)
        n = self.time.size
        if self.band is None:
            raise ValueError("LightCurve requires a 'band' array.")
        self.band = _str_array(self.band)
        if self.band.size != n:
            raise ValueError(f"band length {self.band.size} != time length {n}")
        for attr in ("magnitude", "magnitude_err", "flux", "flux_err"):
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
    def data_mode(self):
        return "magnitude" if self.magnitude is not None else "flux_density"

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
            name=self.name, redshift=self.redshift, meta=dict(self.meta),
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

    # --- derived columns ---
    def add_flux(self, zeropoint_jy=AB_ZEROPOINT_JY):
        """Copy with ``flux`` (+ ``flux_err``) from magnitude (AB). No-op if flux present."""
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
        """Copy with ``magnitude`` (+ ``magnitude_err``) from flux (AB). No-op if magnitude present."""
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
    def to_dataframe(self):
        """Tabular view (``pandas.DataFrame``); includes ``snr`` when errors are available."""
        data = {"time": self.time, "band": self.band}
        for attr in ("magnitude", "magnitude_err", "flux", "flux_err", "upper_limit", "system"):
            v = getattr(self, attr)
            if v is not None:
                data[attr] = v
        try:
            data["snr"] = self.snr
        except ValueError:
            pass
        return pd.DataFrame(data)

    def __repr__(self):
        return (f"LightCurve(name={self.name!r}, n_points={self.n_points}, "
                f"bands={self.bands}, mode={self.data_mode!r})")
