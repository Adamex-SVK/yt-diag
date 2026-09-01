"""Checks for the version-3 colour-temperature computation.

The tests cover the sRGB conversion, known illuminants, the supported
Planckian-locus range, signed Duv, off-locus rejection, and the aggregation
definition shared by collection and recomputation.

    cd 02_Data && ../.venv/bin/python tests/test_cct.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collect_and_extract import (CCT_VALID_K, MAX_DUV,  # noqa: E402
                                 _nearest_planckian_cct_duv_uv,
                                 _planckian_lut, _planckian_xy, _uv_1960,
                                 correlated_color_temp,
                                 correlated_color_temp_and_duv,
                                 population_std, srgb_to_linear)


def test_srgb_transfer_function_matches_the_standard():
    """Reference points of the IEC 61966-2-1 curve."""
    assert srgb_to_linear(0.0) == 0.0
    assert abs(srgb_to_linear(1.0) - 1.0) < 1e-12
    # mid-grey 0.5 encoded is ~0.2140 linear -- the whole point of gamma
    assert abs(srgb_to_linear(0.5) - 0.21404) < 1e-4
    # below the knee the curve is linear, not a power
    assert abs(srgb_to_linear(0.04) - 0.04 / 12.92) < 1e-12
    assert srgb_to_linear(0.25) < 0.25  # encoded values overstate intensity


def test_srgb_white_is_d65():
    """Equal linear sRGB is the D65 white point, CCT approximately 6504 K."""
    cct = correlated_color_temp((1.0, 1.0, 1.0))
    assert cct is not None and abs(cct - 6504) < 30, cct
    # brightness must not change the temperature -- CCT is a chromaticity
    for level in (0.05, 0.2, 0.5, 0.9):
        grey = correlated_color_temp((level, level, level))
        assert grey is not None and abs(grey - cct) < 1e-6, (level, grey)


def test_warm_and_cool_land_on_the_right_sides():
    warm = correlated_color_temp((1.0, 0.6, 0.3))   # tungsten-ish
    cool = correlated_color_temp((0.6, 0.75, 1.0))  # overcast-ish
    assert warm is not None and cool is not None
    assert warm < 5000 < cool, (warm, cool)


def test_known_planckian_points_across_the_supported_range():
    """Every reference point on the locus maps back to its own temperature.

    The 4000 K piecewise-polynomial join is the least exact point, hence the
    2 K tolerance; all other checked points are much closer.
    """
    for expected in (1667, 2000, 2856, 4000, 6500, 10000, 15000, 20000, 25000):
        x, y = _planckian_xy(expected)
        result = _nearest_planckian_cct_duv_uv(*_uv_1960(x, y))
        assert result is not None
        got, duv = result
        assert abs(got - expected) < 2.0, (expected, got)
        assert abs(duv) < 3e-6, (expected, duv)


def test_duv_is_signed_and_d65_is_near_the_locus():
    x, y = _planckian_xy(6500)
    u, v = _uv_1960(x, y)
    above = _nearest_planckian_cct_duv_uv(u, v + 0.005)
    below = _nearest_planckian_cct_duv_uv(u, v - 0.005)
    assert above is not None and above[1] > 0
    assert below is not None and below[1] < 0
    d65 = correlated_color_temp_and_duv((1.0, 1.0, 1.0))
    assert d65 is not None
    _, d65_duv = d65
    assert 0 < d65_duv < MAX_DUV


def test_temperatures_beyond_supported_locus_are_not_clipped_to_endpoints():
    locus = _planckian_lut()
    _, u0, v0 = locus[0]
    _, u1, v1 = locus[1]
    _, up, vp = locus[-2]
    _, un, vn = locus[-1]
    warm_beyond = (u0 - 0.1 * (u1 - u0), v0 - 0.1 * (v1 - v0))
    cool_beyond = (un + 0.1 * (un - up), vn + 0.1 * (vn - vp))
    assert _nearest_planckian_cct_duv_uv(*warm_beyond) is None
    assert _nearest_planckian_cct_duv_uv(*cool_beyond) is None


def test_degenerate_colours_return_none_rather_than_a_number():
    """The failure that produced negative Kelvin: an out-of-range result must
    be absent, not a number a model would treat as a temperature."""
    assert correlated_color_temp((0.0, 0.0, 0.0)) is None      # black
    assert correlated_color_temp((0.0, 1.0, 0.0)) is None      # pure green
    for rgb in [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.02, 0.0, 0.03)]:
        v = correlated_color_temp(rgb)
        assert v is None or CCT_VALID_K[0] <= v <= CCT_VALID_K[1], (rgb, v)


def test_no_input_can_produce_an_impossible_temperature():
    """Sweep the RGB cube: the old code returned up to 5.0e9 K and negative
    values. Nothing may now escape CCT_VALID_K."""
    step = 0.1
    vals = [i * step for i in range(11)]
    checked = 0
    for r in vals:
        for g in vals:
            for b in vals:
                v = correlated_color_temp((r, g, b))
                checked += 1
                assert v is None or CCT_VALID_K[0] <= v <= CCT_VALID_K[1], ((r, g, b), v)
    assert checked == 11 ** 3


def test_population_std_is_shared_and_well_defined_for_one_value():
    assert population_std([]) is None
    assert population_std([7.0]) == 0.0
    assert abs(population_std([1.0, 2.0, 3.0]) - (2.0 / 3.0) ** 0.5) < 1e-12


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name} OK")
    print("ALL CCT CHECKS PASSED")
