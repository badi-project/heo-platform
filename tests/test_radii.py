"""The ionic-radius table (data/reference/shannon_radii.json, src/radii.py)."""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import radii  # noqa: E402

# values read in Shannon (1976), Table 1, effective ionic radii — the transcription must
# give them back exactly
SHANNON = [
    ("Mg", 2, 6, 0.72), ("Mg", 2, 4, 0.57), ("Fe", 2, 6, 0.78), ("Fe", 3, 6, 0.645),
    ("Fe", 3, 4, 0.49), ("Co", 2, 6, 0.745), ("Co", 3, 6, 0.545), ("Ni", 2, 6, 0.69),
    ("Cu", 2, 6, 0.73), ("Zn", 2, 6, 0.74), ("Zn", 2, 4, 0.60), ("Mn", 2, 6, 0.83),
    ("Mn", 3, 6, 0.645), ("Mn", 4, 6, 0.53), ("Cr", 3, 6, 0.615), ("Al", 3, 6, 0.535),
    ("Ti", 4, 6, 0.605), ("Ce", 4, 8, 0.97), ("Ce", 3, 8, 1.143), ("Zr", 4, 8, 0.84),
    ("Hf", 4, 8, 0.83), ("Y", 3, 8, 1.019), ("La", 3, 12, 1.36), ("Sr", 2, 12, 1.44),
    ("Ba", 2, 12, 1.61), ("Gd", 3, 8, 1.053), ("Sn", 4, 6, 0.69), ("Li", 1, 6, 0.76),
]


@pytest.mark.parametrize("el, ox, cn, r", SHANNON)
def test_transcription_matches_shannon(el, ox, cn, r):
    assert radii.radius(el, ox, cn) == (r, "exact")


def test_oxygen():
    assert (radii.oxygen(2), radii.oxygen(4), radii.oxygen(6), radii.oxygen(8)) == (1.35, 1.38, 1.40, 1.42)


def test_spin_convention():
    tab = radii.table()
    assert radii.radius("Co", 3, 6)[0] == tab["Co"]["3"]["VI"]["ls"] == 0.545
    assert radii.radius("Ni", 3, 6)[0] == tab["Ni"]["3"]["VI"]["ls"] == 0.56
    assert radii.radius("Fe", 2, 6)[0] == tab["Fe"]["2"]["VI"]["hs"] == 0.78
    assert radii.radius("Mn", 3, 6)[0] == tab["Mn"]["3"]["VI"]["hs"] == 0.645


def test_missing_coordination_is_flagged():
    r, how = radii.radius("Pr", 3, 12)        # tabulated VI, VIII, IX
    assert how == "extrapolated" and 1.27 < r < 1.34
    r, how = radii.radius("La", 3, 11)        # between X and XII
    assert how == "interpolated" and 1.27 < r < 1.36
    assert radii.radius("Fe", 5, 6) == (None, "missing")


def test_every_platform_cation_has_a_radius():
    elements = json.load(open(os.path.join(ROOT, "data", "reference",
                                           "element_properties.json"), encoding="utf-8"))
    for el in elements:
        assert radii.states(el), el
        assert any(radii.radius(el, ox, 6)[0] for ox in radii.states(el)), el


def test_source_is_recorded():
    src = radii.load()["_source"]
    assert src["doi"] == "10.1107/S0567739476001551" and "pymatgen" in src["transcription"]


def test_extrapolated_twelvefold_radii_follow_the_lanthanide_contraction():
    """Regression: a straight line through the last two tabulated points made Pr3+ XII
    larger than Ce3+ XII and Eu3+ XII larger than Sm3+ XII."""
    order = ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
    r = [radii.radius(el, 3, 12)[0] for el in order]
    assert all(a >= b for a, b in zip(r, r[1:])), dict(zip(order, r))
    # Y3+ sits with Ho3+ / Dy3+ at every tabulated coordination; it must stay there
    y = radii.radius("Y", 3, 12)[0]
    assert radii.radius("Ho", 3, 12)[0] - 0.02 <= y <= radii.radius("Dy", 3, 12)[0] + 0.02
