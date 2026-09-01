"""Checks for the corrected colour-temperature computation.

The original implementation was wrong in a way no test would have caught,
because there was no test: it set X=r, Y=g, Z=b and skipped the sRGB->XYZ
matrix, so its chromaticity was normalised RGB. The fix is only trustworthy
against KNOWN answers, so these check physical reference points rather than
"it returns a number".

    cd 02_Data && ../.venv/bin/python tests/test_cct.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collect_and_extract import (CCT_VALID_K, correlated_color_temp,  # noqa: E402
                                 srgb_to_linear)


def test_srgb_transfer_function_matches_the_standard():
    """Reference points of the IEC 61966-2-1 curve."""
    assert srgb_to_linear(0.0) == 0.0
    assert abs(srgb_to_linear(1.0) - 1.0) < 1e-12
    # mid-grey 0.5 encoded is ~0.2140 linear -- the whole point of gamma
    assert abs(srgb_to_linear(0.5) - 0.21404) < 1e-4
    # below the knee the curve is linear, not a power
    assert abs(srgb_to_linear(0.04) - 0.04 / 12.92) < 1e-12
    assert srgb_to_linear(0.25) < 0.25  # encoded values overstate intensity


def test_equal_energy_white_is_d65():
    """Equal linear RGB is the sRGB white point, D65 ~ 6504 K. This is the
    single check that would have caught the original bug: the broken version
    returned 5520 K for white (its n collapsed to ~0 for any neutral colour)."""
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


def test_near_the_old_singularity_is_handled():
    """g/(r+g+b) ~ 0.1858 is where the ORIGINAL formula blew up. In real CIE
    space that input is unremarkable, and must either give a sane temperature
    or None -- never a divergence."""
    for r in [0.3, 0.5, 0.7]:
        for b in [0.3, 0.5, 0.7]:
            g = 0.1858 * (r + b) / (1 - 0.1858)  # makes the RGB fraction exactly 0.1858
            v = correlated_color_temp((r, g, b))
            assert v is None or CCT_VALID_K[0] <= v <= CCT_VALID_K[1], ((r, g, b), v)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"{name} OK")
    print("ALL CCT CHECKS PASSED")
