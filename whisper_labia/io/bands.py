"""Band-name normalization and grouping.

Two composable layers:

1. :func:`normalize_band` / :func:`normalize_bands` -- light, case-sensitive formatting of raw
   survey codes via :data:`DEFAULT_BAND_ALIASES` (e.g. 'zg' -> 'ztfg'). Unknown labels pass through.

2. :func:`group_bands` -- collapse many heterogeneous filters into a small effective-band ladder
   via :data:`FILTER_LOOKUP` (e.g. 'B' -> 'g-band', 'V' -> 'r-band', 'Ks' -> 'K-band',
   'F606W' -> 'r-band'). Band labels are case-sensitive in astronomy ('r' is SDSS r, 'R' is
   Cousins R), so the lookup lists case variants explicitly.

They compose: e.g. 'zg' --normalize--> 'ztfg' --group--> 'g-band'.
"""
from __future__ import annotations

import warnings

import numpy as np

# Light survey-code normalization (extend per-call with the ``aliases`` argument).
DEFAULT_BAND_ALIASES = {
    "zg": "ztfg", "zr": "ztfr", "zi": "ztfi",
    "ztf_g": "ztfg", "ztf_r": "ztfr", "ztf_i": "ztfi",
}

# Broadband grouping supplied by the user: collapse heterogeneous filters into an effective-band
# ladder. Comments give the rough wavelength-order index of each group.
FILTER_LOOKUP = {
    # U-group (+2)
    'u': 'U-band', 'U': 'U-band', "u'": 'U-band',
    'uvw2': 'U-band', 'uvm2': 'U-band', 'uvw1': 'U-band', 'uvot_u': 'U-band',
    'F336W': 'U-band', 'f336w': 'U-band',
    # g-group (+1)
    'g': 'g-band', 'G': 'g-band',
    'lsstg': 'g-band', 'g-ztf': 'g-band', 'ztfg': 'g-band', 'g-p1': 'g-band',
    'B': 'g-band', 'b': 'g-band',
    'F475W': 'g-band', 'f475w': 'g-band',
    # r-group (0)
    'r': 'r-band', 'R': 'r-band', "r'": 'r-band',
    'lsstr': 'r-band', 'r-ztf': 'r-band', 'ztfr': 'r-band', 'r-p1': 'r-band',
    'V': 'r-band', 'v': 'r-band', 'Vc': 'r-band',
    'F606W': 'r-band', 'F070W': 'r-band',
    'F625W': 'r-band', 'f625w': 'r-band',
    # i-group (-1)
    'i': 'i-band', 'I': 'i-band', 'Ic': 'i-band', "i'": 'i-band',
    'lssti': 'i-band', 'i-ztf': 'i-band', 'ztfi': 'i-band', 'i-p1': 'i-band',
    'F814W': 'i-band',
    'F775W': 'i-band', 'f775w': 'i-band',
    # z-group (-2)
    'z': 'z-band', 'Z': 'z-band', "z'": 'z-band',
    'lsstz': 'z-band', 'z-ztf': 'z-band', 'ztfz': 'z-band', 'z-p1': 'z-band',
    'y': 'z-band', 'Y': 'z-band', 'lssty': 'z-band', 'y-p1': 'z-band',
    'F090W': 'z-band',
    'F850W': 'z-band', 'f850w': 'z-band',
    # J-group (-3)
    'J': 'J-band', 'j': 'J-band', '2massj': 'J-band',
    'F110W': 'J-band', 'f110w': 'J-band',
    'F115W': 'J-band', 'f115w': 'J-band',
    'F125W': 'J-band', 'f125w': 'J-band',
    'J1': 'J-band', 'j1': 'J-band',
    # H-group (-4)  -- NOTE: original '2massh' value was 'H-Band' (capital B); fixed to 'H-band'.
    'H': 'H-band', 'h': 'H-band', '2massh': 'H-band',
    'F150W': 'H-band', 'f150w': 'H-band',
    'F160W': 'H-band', 'f160w': 'H-band',
    # K-group (-5)
    'K': 'K-band', 'Ks': 'K-band', 'k': 'K-band', '2massks': 'K-band',
    'F200W': 'K-band', 'f200W': 'K-band',
    'F277W': 'K-band', 'f277W': 'K-band',
    # JWST sim bands (-6)
    'F356W': 'F356W-band', 'f356w': 'F356W-band',
    'F444W': 'F444W-band', 'f444w': 'F444W-band',
}


def normalize_band(band, aliases=None, warn_unknown=False, known=None):
    """Normalize a single band label (trim whitespace, apply exact-match alias map)."""
    table = DEFAULT_BAND_ALIASES if aliases is None else {**DEFAULT_BAND_ALIASES, **aliases}
    raw = str(band).strip()
    if raw in table:
        return table[raw]
    if warn_unknown and known is not None and raw not in known:
        warnings.warn(f"Unrecognized band {raw!r} (left unchanged).", stacklevel=2)
    return raw


def normalize_bands(bands, aliases=None, warn_unknown=False, known=None):
    """Vectorized :func:`normalize_band` -> ``np.ndarray`` of normalized labels."""
    return np.array([
        normalize_band(b, aliases=aliases, warn_unknown=warn_unknown, known=known)
        for b in bands
    ])


def group_bands(bands, lookup=None, default=None, warn_unknown=False):
    """Map each band to its effective broadband via ``lookup`` (default :data:`FILTER_LOOKUP`).

    Labels absent from the lookup are left unchanged (or set to ``default``); set
    ``warn_unknown=True`` to warn about them.
    """
    lookup = FILTER_LOOKUP if lookup is None else lookup
    out, unknown = [], []
    for b in bands:
        key = str(b).strip()
        if key in lookup:
            out.append(lookup[key])
        else:
            unknown.append(key)
            out.append(key if default is None else default)
    if warn_unknown and unknown:
        warnings.warn(
            f"{len(set(unknown))} band(s) not in lookup, left ungrouped: {sorted(set(unknown))}",
            stacklevel=2)
    return np.array(out)


def unmapped_bands(bands, lookup=None):
    """Return the sorted distinct labels in ``bands`` not covered by ``lookup``."""
    lookup = FILTER_LOOKUP if lookup is None else lookup
    return sorted({str(b).strip() for b in bands if str(b).strip() not in lookup})
