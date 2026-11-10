"""Measured lattice parameters (data/reference/lattice_measured.csv)."""
import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lattice_dataset as ld  # noqa: E402
import predict  # noqa: E402

SOURCE = re.compile(r"^(10\.\d{4,9}/\S+|arXiv:\d{4}\.\d{4,5}|https?://\S+|lab:\S+)$")


@pytest.fixture(scope="module")
def P():
    return predict.Predictor()


@pytest.fixture(scope="module")
def rows():
    return ld.load()


def test_every_line_is_valid(P, rows):
    assert ld.validate(rows, P) == []


def test_every_value_has_a_traceable_source(rows):
    for r in rows:
        assert SOURCE.match(r["source"]), (r["id"], r["source"])
        assert r["source_detail"], r["id"]


def test_five_families_covered_in_both_groups(P, rows):
    comps = ld.unique_compositions(rows, P)
    for f in ld.FAMILIES:
        ref = sum(1 for c in comps.values() if c["proto"] == f and c["group"] == "reference")
        heo = sum(1 for c in comps.values() if c["proto"] == f and c["group"] == "heo")
        assert ref >= 4, (f, ref)
        assert heo >= 9, (f, heo)


def test_pseudocubic_parameter_of_an_orthorhombic_cell():
    row = {"a_ang": "", "abc_ang": "5.55531;5.5602;7.85114"}       # LaFeO3, Pbnm
    a = ld.a_measured(row)
    assert a == pytest.approx((5.55531 * 5.5602 * 7.85114 / 4) ** (1 / 3))
    assert 3.92 < a < 3.94
    assert ld.a_measured({"a_ang": "4.2368", "abc_ang": ""}) == 4.2368


def test_repeated_composition_is_one_unit_with_median(P, rows):
    comps = ld.unique_compositions(rows, P)
    mgo = [c for c in comps.values() if c["proto"] == "rocksalt" and c["frac"] == {"Mg": 1.0}]
    assert len(mgo) == 1 and len(mgo[0]["rows"]) == 2
    assert mgo[0]["a"] == pytest.approx((4.211 + 4.217) / 2)
    # written in another order, (CoCrFeMnNi)3O4 is the same composition
    s = [c for c in comps.values() if c["proto"] == "spinel" and len(c["frac"]) == 5
         and set(c["frac"]) == {"Co", "Cr", "Fe", "Mn", "Ni"}]
    assert len(s) == 1 and len(s[0]["rows"]) == 7


def test_impure_lines_left_out_of_the_main_metric(P, rows):
    comps = ld.unique_compositions(rows, P)
    used = {r["id"] for c in comps.values() for r in c["rows"]}
    for r in rows:
        assert (r["id"] in used) == (r["phase_note"] == "single"), r["id"]


def test_scatter_needs_two_origins(P, rows):
    comps = ld.unique_compositions(rows, P)
    sc = ld.scatter(comps)
    assert all(s["sources"] >= 2 for s in sc)
    # two sintering routes of one paper are not an inter-laboratory comparison
    assert not any("Y0.2Gd0.2Nd0.2Sm0.2)2Zr2O7" in s["formula"] for s in sc)
    # a compilation counts each paper it cites
    assert any(s["formula"].startswith("(Mg0.2Fe0.2Co0.2Ni0.2Cu0.2)Fe2O4") for s in sc)


def test_simple_oxides_are_read(P):
    assert ld.fractions("MgO", P.elements) == {"Mg": 1.0}
    f = ld.fractions("MgAl2O4", P.elements)
    assert f["Al"] == pytest.approx(2 / 3) and f["Mg"] == pytest.approx(1 / 3)
    with pytest.raises(ValueError):
        ld.fractions("O2", P.elements)
