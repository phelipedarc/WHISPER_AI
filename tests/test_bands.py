from whisper_labia.io.bands import (
    group_bands,
    normalize_band,
    normalize_bands,
    unmapped_bands,
)


def test_ztf_aliases():
    assert normalize_band("zg") == "ztfg"
    assert normalize_band("zr") == "ztfr"


def test_passthrough_and_case_sensitive():
    assert normalize_band("g") == "g"
    assert normalize_band("R") == "R"   # Cousins R, not SDSS r
    assert normalize_band("r") == "r"


def test_whitespace_trimmed():
    assert normalize_band(" zg ") == "ztfg"


def test_custom_alias_override():
    assert normalize_band("g", aliases={"g": "sdssg"}) == "sdssg"


def test_normalize_bands_array():
    out = normalize_bands(["zg", "zr", "g", "Ks"])
    assert list(out) == ["ztfg", "ztfr", "g", "Ks"]


# --- broadband grouping (FILTER_LOOKUP) ---

def test_group_bands_basic():
    out = group_bands(["B", "V", "y", "Ks", "F606W", "F110W", "F160W"])
    assert list(out) == ["g-band", "r-band", "z-band", "K-band", "r-band", "J-band", "H-band"]


def test_group_bands_2massh_typo_fixed():
    assert group_bands(["2massh"])[0] == "H-band"   # not 'H-Band'


def test_group_bands_passthrough_unknown():
    out = group_bands(["g", "C", "W"])   # C, W are absent from the lookup
    assert list(out) == ["g-band", "C", "W"]


def test_group_bands_compose_with_normalize():
    # 'zg' -> 'ztfg' (normalize) -> 'g-band' (group)
    assert list(group_bands(normalize_bands(["zg", "zr"]))) == ["g-band", "r-band"]


def test_group_bands_hst_extensions():
    out = group_bands(["F336W", "F475W", "F625W", "F775W", "F850W", "J1"])
    assert list(out) == ["U-band", "g-band", "r-band", "i-band", "z-band", "J-band"]


def test_unmapped_bands():
    # C and W (clear/white-light) are intentionally left out of the lookup.
    assert unmapped_bands(["g", "r", "C", "W"]) == ["C", "W"]
