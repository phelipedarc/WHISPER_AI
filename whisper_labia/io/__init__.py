"""Data ingestion: canonical light-curve container, CSV loader, band/photometry helpers."""
from .bands import (
    DEFAULT_BAND_ALIASES,
    FILTER_LOOKUP,
    group_bands,
    normalize_band,
    normalize_bands,
    unmapped_bands,
)
from .loader import load_lightcurve
from .photometry import (
    AB_ZEROPOINT_JY,
    flux_density_to_mag,
    mag_err_to_snr,
    mag_to_flux_density,
)
from .schema import LightCurve

__all__ = [
    "LightCurve",
    "load_lightcurve",
    "normalize_band",
    "normalize_bands",
    "group_bands",
    "unmapped_bands",
    "DEFAULT_BAND_ALIASES",
    "FILTER_LOOKUP",
    "mag_to_flux_density",
    "flux_density_to_mag",
    "mag_err_to_snr",
    "AB_ZEROPOINT_JY",
]
