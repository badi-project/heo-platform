"""Lattice model (src/lattice.py): relations, valences, sites, validation criteria."""
import json
import math
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lattice  # noqa: E402
import predict  # noqa: E402
import validate_lattice  # noqa: E402

P = predict.Predictor()
E = P.elements


def ions(proto, frac):
    return [(s, el, ox, round(x, 6)) for s, el, ox, x in lattice.assign(proto, frac, E)[0]]


# ---------------------------------------------------------------- hand calculations ----

def test_rocksalt_is_anion_cation_contact():
    # MgO: 2 (0.72 + 1.40) = 4.24 (measured 4.211-4.217)
    assert lattice.predict("rocksalt", {"Mg": 1.0}, E)["a"] == pytest.approx(4.24, abs=1e-12)
    a = lattice.predict("rocksalt", {e: 0.2 for e in ("Mg", "Co", "Ni", "Cu", "Zn")}, E)["a"]
    assert a == pytest.approx(2 * ((0.72 + 0.745 + 0.69 + 0.73 + 0.74) / 5 + 1.40), abs=1e-12)


def test_fluorite_without_vacancies_is_contact():
    # CeO2: 4/sqrt 3 (0.97 + 1.38) = 5.427 (measured 5.411-5.414)
    r = lattice.predict("fluorite", {"Ce": 1.0}, E)
    assert r["a"] == pytest.approx(4 / math.sqrt(3) * (0.97 + 1.38), abs=1e-12)
    assert r["vacancies"] is None


def test_perovskite_follows_ubic():
    # SrTiO3, sixfold radii: Sr2+ 1.18, Ti4+ 0.605, O 1.40 -> 3.9237 (measured 3.905, +0.5 %)
    a = lattice.predict("perovskite", {"Sr": 0.5, "Ti": 0.5}, E)["a"]
    assert a == pytest.approx(0.06741 + 0.49052 * (1.18 + 1.40) + 1.29212 * (0.605 + 1.40), abs=1e-12)
    assert a == pytest.approx(3.905, rel=0.006)


def test_spinel_follows_oneill_navrotsky():
    # MgAl2O4 normal: R_T = 0.57 + 1.38, R_O = 0.535 + 1.38 -> 8.09 (measured 8.083)
    RT, RO = 0.57 + 1.38, 0.535 + 1.38
    a = lattice.predict("spinel", {"Mg": 1 / 3, "Al": 2 / 3}, E)["a"]
    assert a == pytest.approx(8 / (11 * math.sqrt(3)) * (5 * RT + math.sqrt(33 * RO ** 2 - 8 * RT ** 2)),
                              abs=1e-9)
    assert a == pytest.approx(8.083, abs=0.02)


def test_pyrochlore_follows_mouta():
    # Gd2Zr2O7: R_A = 1.053 + 1.38, R_B = 0.72 + 1.38 -> 10.52 (measured 10.53)
    RA, RB = 1.053 + 1.38, 0.72 + 1.38
    a = lattice.predict("pyrochlore", {"Gd": 0.5, "Zr": 0.5}, E)["a"]
    assert a == pytest.approx(8 / math.sqrt(3) * (1.43373 * RA - 0.42931 * RA ** 2 / RB), abs=1e-9)
    assert a == pytest.approx(10.53, abs=0.03)


# ------------------------------------------------------------- valences and sites ----

def test_spinel_charge_balance_oxidises_iron_then_manganese_then_cobalt():
    # Fe3O4: one Fe2+ and two Fe3+, inverse (Fe3+ on the tetrahedral site)
    got = ions("spinel", {"Fe": 1.0})
    assert ("T", "Fe", 3, 1.0) in got and ("O", "Fe", 2, 0.5) in got and ("O", "Fe", 3, 0.5) in got
    # Co3O4: normal, Co2+ tetrahedral, low-spin Co3+ octahedral
    assert ions("spinel", {"Co": 1.0}) == [("O", "Co", 3, 1.0), ("T", "Co", 2, 1.0)]
    # (CoCrFeMnNi)3O4: Fe and Mn 3+, the last 0.2 charge taken by a third of the cobalt
    got = dict(((s, el, ox), x) for s, el, ox, x in ions("spinel", {e: 0.2 for e in
                                                               ("Co", "Cr", "Fe", "Mn", "Ni")}))
    assert got[("T", "Fe", 3)] == pytest.approx(0.6)                 # Fe3+ fills T first
    assert ("T", "Cr", 3) not in got and ("T", "Ni", 2) not in got   # octahedral preference
    charge = sum(ox * x * (1 if s == "T" else 2) for (s, _, ox), x in got.items())
    assert charge == pytest.approx(8.0)


@pytest.mark.parametrize("formula,tet", [
    ({"Mg": 1 / 3, "Al": 2 / 3}, "Mg"), ({"Zn": 1 / 3, "Fe": 2 / 3}, "Zn"),      # normal
    ({"Ni": 1 / 3, "Fe": 2 / 3}, "Fe"), ({"Mg": 1 / 3, "Fe": 2 / 3}, "Fe"),      # inverse
    ({"Ni": 1 / 3, "Al": 2 / 3}, "Al"), ({"Mg": 2 / 3, "Ti": 1 / 3}, "Mg"),      # inverse
])
def test_spinel_site_preference_gives_the_known_inversion(formula, tet):
    t = [i for i in ions("spinel", formula) if i[0] == "T"]
    assert [i[1] for i in t] == [tet] and t[0][3] == pytest.approx(1.0)


def test_rocksalt_compensates_lithium_by_oxidising_cobalt():
    got = ions("rocksalt", {"Mg": 0.18, "Co": 0.18, "Ni": 0.18, "Cu": 0.18, "Zn": 0.18, "Li": 0.1})
    co = {ox: x for _, el, ox, x in got if el == "Co"}
    assert co[3] == pytest.approx(0.1) and co[2] == pytest.approx(0.08)
    assert sum(ox * x for _, _, ox, x in got) == pytest.approx(2.0)


def test_perovskite_sites_and_valences():
    # La(CoCrFeMnNi)O3: every B cation 3+
    got = ions("perovskite", {"La": 0.5, **{e: 0.1 for e in ("Co", "Cr", "Fe", "Mn", "Ni")}})
    assert all(ox == 3 for _, _, ox, _ in got)
    assert [i[1] for i in got if i[0] == "A"] == ["La"]
    # Sr on A: the 3d cation is oxidised to 4+ to reach +6
    assert ("B", "Fe", 4, 1.0) in ions("perovskite", {"Sr": 0.5, "Fe": 0.5})
    # fixed-valence B cations short of charge: oxygen vacancies, reported
    r = lattice.predict("perovskite", {"Ba": 0.25, "Sr": 0.25, "Ga": 1 / 6, "In": 1 / 6, "Sn": 1 / 6}, E)
    assert r["vacancies"]["per_cation"] > 0 and any("lacunes d'oxygène" in n for n in r["notes"])


def test_fluorite_vacancies_shrink_the_cell():
    frac = {e: 0.2 for e in ("Ce", "La", "Pr", "Sm", "Y")}
    r = lattice.predict("fluorite", frac, E)
    contact = lattice.predict("fluorite", frac, E, vacancy_model=False)
    assert r["vacancies"]["per_cation"] == pytest.approx((4 - (4 + 3 + 11 / 3 + 3 + 3) / 5) / 2)
    assert 0.9 < r["vacancies"]["r_V"] < 1.38          # a vacancy is smaller than an oxide ion
    assert r["a"] < contact["a"]
    assert abs(r["a"] / 5.5071 - 1) < 0.005 < abs(contact["a"] / 5.5071 - 1)   # measured 5.5071


# -------------------------------------------------------------------- reporting ----

def test_estimated_radii_and_large_cations_are_reported():
    r = lattice.predict("pyrochlore", {"Pr": 0.5, "Sn": 0.5}, E)
    assert r["estimated"] == []
    r = lattice.predict("fluorite", {"Zr": 0.4, "Hf": 0.4, "Mn": 0.2}, E)
    assert r["estimated"] and any(n.startswith("rayon Mn") and "Å" in n for n in r["notes"])
    r = lattice.predict("rocksalt", {e: 0.2 for e in ("Cu", "Zn", "Ni", "Co", "Sr")}, E)
    assert any(n.startswith("cation volumineux (Sr)") for n in r["notes"])
    out = P.predict({e: 0.2 for e in ("Cu", "Zn", "Ni", "Co", "Sr")})
    if out["family"] == "rocksalt":
        assert any(w.startswith("cation volumineux") for w in out["warnings"])


def test_no_model_is_an_error_not_a_number():
    with pytest.raises(lattice.LatticeError):
        lattice.predict("layered", {"Li": 0.5, "Co": 0.5}, E)
    assert P.geometric_lattice("layered", {"Li": 0.5, "Co": 0.5}) is None
    # a rocksalt cation without a 2+ ion takes its usual valence (amendment A2)
    assert lattice.predict("rocksalt", {"Mg": 0.5, "Ni": 0.3, "Y": 0.2}, E)["a"] > 4.2
    # five 3d metals called perovskite: sites by size, and the prediction says so
    r = lattice.predict("perovskite", {e: 0.2 for e in ("Co", "Cr", "Fe", "Mn", "Ni")}, E)
    assert any("rayon décroissant" in n for n in r["notes"])


def test_prediction_uses_the_model_and_its_validated_error():
    val = json.load(open(os.path.join(ROOT, "data", "reference", "lattice_validation.json"),
                         encoding="utf-8"))
    out = P.predict("(LaNdSmEuGd)2Zr2O7")
    lat = out["lattice"]
    assert out["family"] == "pyrochlore" and lat["source"] == "geometric"
    assert abs(lat["a"] / 10.64 - 1) < 0.006                            # measured 10.639-10.644
    assert lat["uncertainty_pct"] == val["families"]["pyrochlore"]["uncertainty_pct"]
    # CHGNet keeps the spinels it covers: the validation prefers it there
    sp = P.predict("(FeCoNiCrMn)3O4")["lattice"]
    assert sp["source"] == "chgnet" and val["spinel_selection"]["retained"] == "chgnet"


# ------------------------------------------------------------------ validation ----

def test_validation_report_is_current_and_criteria_are_as_published():
    report, summary, ok = validate_lattice.build()
    assert open(validate_lattice.REPORT, encoding="utf-8").read() == report
    assert open(validate_lattice.SUMMARY, encoding="utf-8").read() == summary
    val = json.loads(summary)
    assert ok and val["C1_met"] and val["C3_met"]
    fam = val["families"]
    for f in ("spinel", "fluorite", "perovskite", "pyrochlore"):
        assert fam[f]["C2"] and fam[f]["heo"]["mae_pct"] <= 1.0
    # the miss is published: rocksalt is above 1 %, and the report says why
    assert not fam["rocksalt"]["C2"] and "NON ATTEINT" in report and "incorporation" in report
    # the amended family is declared non-blind, with the result of the protocol as written
    assert fam["fluorite"]["blind"] is False and fam["fluorite"]["heo_blind"]["mae_pct"] > 1.0
    assert all(fam[f]["blind"] for f in fam if f != "fluorite")
    assert val["heo"]["mae_pct"] < val["heo_protocol_initial"]["mae_pct"] < val["heo_v1"]["mae_pct"]
    for f in fam:
        assert fam[f]["heo"]["mae_pct"] < fam[f]["heo_v1"]["mae_pct"]


def test_protocol_records_the_amendments():
    text = open(os.path.join(ROOT, "data", "reference", "lattice_protocol.md"), encoding="utf-8").read()
    assert "A1, fluorine, après lecture des résidus" in text and "plus aveugle" in text
    assert "A2, rocksalt" in text


def test_check_mode_of_the_validation_script():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "validate_lattice.py"), "--check"],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
