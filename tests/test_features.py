"""Unit tests for the descriptor specification (src/features.py)."""
import json
import math
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402

BUNDLE = json.load(open(os.path.join(ROOT, "app", "heo_model.json"), encoding="utf-8"))
ELEMENTS = features.load_elements()
ONEHOT = BUNDLE["onehot"]


def vec(frac):
    names, vals = features.featurize(frac, ELEMENTS, ONEHOT)
    return dict(zip(names, vals)), names, vals


def test_load_elements_works_from_any_directory(tmp_path):
    """Regression: the default path used to be relative to the working directory."""
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {os.path.join(ROOT, 'src')!r}); "
         "import features; print(len(features.load_elements()))"],
        cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "58"


def test_element_table_has_every_property_used():
    """Regression: r_shannon_VIII and r_shannon_XII were missing from the shipped table."""
    needed = set(features.PROPS)
    for symbol, props in ELEMENTS.items():
        assert needed <= set(props), f"{symbol} is missing {sorted(needed - set(props))}"


def test_vector_length_and_order_match_the_model():
    _, names, vals = vec(features.equimolar("Mg", "Co", "Ni", "Cu", "Zn"))
    assert len(vals) == 134
    assert names == BUNDLE["featnames"]


def test_configurational_entropy_is_ideal_mixing():
    d, _, _ = vec(features.equimolar("Mg", "Co", "Ni", "Cu", "Zn"))
    assert d["s_config"] == pytest.approx(math.log(5.0), abs=1e-12)
    assert d["n_cat"] == 5.0


def test_goldschmidt_factor_of_strontium_titanate():
    """SrTiO3 is the textbook t ~ 1.00 perovskite."""
    d, _, _ = vec({"Sr": 0.5, "Ti": 0.5})
    assert d["goldschmidt_t"] == pytest.approx(1.002, abs=0.005)


def test_mean_octahedral_radius_of_the_canonical_rocksalt():
    """(MgCoNiCuZn)O: 0.71 A is deduced from the measured lattice parameter."""
    d, _, _ = vec(features.equimolar("Mg", "Co", "Ni", "Cu", "Zn"))
    assert d["r_shannon_VI_mean"] == pytest.approx(0.692, abs=0.01)


def test_ab_split_normalises_each_sublattice():
    A, B = features.ab_split(features.equimolar("La", "Sr", "Co", "Fe", "Mn"), ELEMENTS)
    assert sum(A.values()) == pytest.approx(1.0, abs=1e-12)
    assert sum(B.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(ELEMENTS[e]["r_shannon_XII"] for e in A) >= \
        max(ELEMENTS[e]["r_shannon_XII"] for e in B)


def test_onehot_block_is_binary_and_consistent():
    d, _, _ = vec(features.equimolar("Mg", "Co", "Ni", "Cu", "Zn"))
    flags = {k[4:]: v for k, v in d.items() if k.startswith("has_")}
    assert len(flags) == 30
    assert set(flags.values()) <= {0.0, 1.0}
    assert {e for e, v in flags.items() if v} == {"Mg", "Co", "Ni", "Cu", "Zn"}


def test_family_fractions_sum_to_one():
    d, _, _ = vec(features.equimolar("La", "Sr", "Co", "Fe", "Mn"))
    fams = [v for k, v in d.items() if k.startswith("frac_")]
    assert sum(fams) == pytest.approx(1.0, abs=1e-12)
