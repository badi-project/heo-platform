"""Tests of the theoretical (X-ray) density (src/predict.py, tools/validate_density.py)."""
import csv
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import predict  # noqa: E402
import validate_density  # noqa: E402

P = predict.Predictor()
N_A = 6.02214076e23


def test_reference_oxides_within_one_percent():
    """Cell content: with the measured lattice parameter, the published density comes back."""
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "reference",
                                                 "density_references.csv"), encoding="utf-8")))
    assert {r["proto"] for r in rows} == {"rocksalt", "spinel", "perovskite", "fluorite"}
    for r in rows:
        assert r["source"].startswith("https://")
        rho = P.xray_density(r["proto"], json.loads(r["frac"]), float(r["a_ang"]))
        assert abs(rho / float(r["rho_ref_g_cm3"]) - 1) < 0.01, r["compound"]


def test_density_by_hand_for_the_founding_rocksalt():
    """(MgCoNiCuZn)O, Z = 4, at the lattice parameter measured by Rost et al."""
    els, a = ["Mg", "Co", "Ni", "Cu", "Zn"], 4.236
    molar = sum(P.elements[e]["mass"] for e in els) / 5 + 15.999
    by_hand = 4 * molar / (N_A * (a * 1e-8) ** 3)
    assert abs(P.xray_density("rocksalt", {e: 0.2 for e in els}, a) - by_hand) < 1e-9
    assert 6.0 < by_hand < 6.3


@pytest.mark.parametrize("proto, cations, oxygens", [
    ("rocksalt", 4, 4), ("fluorite", 4, 8), ("perovskite", 2, 3),
    ("spinel", 24, 32), ("pyrochlore", 32, 56),      # 8 AB2O4 ; 8 A2B2O7
])
def test_cell_content_of_each_prototype(proto, cations, oxygens):
    sites = P.xrd["prototypes"][proto]["nsites"]
    assert sum(n for t, n in sites.items() if t[0] != "O") == cations
    assert sum(n for t, n in sites.items() if t[0] == "O") == oxygens


def test_ordered_prototype_puts_large_cations_on_the_a_site():
    """SrTiO3: one Sr, one Ti and three O per cell, whatever the order of the keys."""
    a = 3.905
    expected = (87.62 + 47.867 + 3 * 15.999) / (N_A * (a * 1e-8) ** 3)
    for frac in ({"Sr": 0.5, "Ti": 0.5}, {"Ti": 0.5, "Sr": 0.5}):
        assert abs(P.xray_density("perovskite", frac, a) - expected) < 0.01


def test_density_scales_with_the_inverse_cube_of_the_lattice_parameter():
    frac = {e: 0.2 for e in ("Co", "Cr", "Fe", "Mn", "Ni")}
    r1, r2 = P.xray_density("spinel", frac, 8.30), P.xray_density("spinel", frac, 8.383)
    assert abs(r1 / r2 - (8.383 / 8.30) ** 3) < 1e-12
    assert P.xray_density("layered", frac, 8.3) is None and P.xray_density("spinel", frac, None) is None


def test_prediction_reports_density_with_an_honest_uncertainty():
    seen = P.lattice["rho"]
    sp = P.predict("(FeCoNiCrMn)3O4")["lattice"]
    assert sp["source"] == "chgnet" and 5.2 < sp["density_g_cm3"] < 5.45
    assert sp["density_uncertainty_pct"] == max(3 * sp["uncertainty_pct"], seen["chgnet"]["mae_pct"])
    rs = P.predict("(Mg0.2Co0.2Ni0.2Cu0.2Zn0.2)O")["lattice"]
    # the displayed uncertainty is the larger of three times the lattice uncertainty and
    # the density error observed for this lattice model on measured high-entropy oxides
    assert rs["source"] == "geometric" and seen["geo"]["met"] and seen["geo"]["n"] >= 40
    assert rs["density_uncertainty_pct"] == max(3 * rs["uncertainty_pct"], seen["geo"]["mae_pct"])
    assert rs["density_uncertainty_pct"] >= seen["geo"]["mae_pct"]
    assert abs(rs["density_uncertainty_g_cm3"]
               - rs["density_g_cm3"] * rs["density_uncertainty_pct"] / 100) < 1e-12


def test_validation_report_is_the_output_of_its_script():
    assert validate_density.main(check=True) == 0
    report = open(os.path.join(ROOT, "QC", "density_validation.txt"), encoding="utf-8").read()
    assert "ecart maximal" in report and "-> atteint" in report
    # the worst case of the geometric model is published, not hidden
    assert "Co-Cu-Ni-Sr-Zn" in report and "lattice_measured.csv" in report


def test_page_embeds_the_validation_summary():
    html = open(os.path.join(ROOT, "app", "heo_designer.html"), encoding="utf-8").read()
    start = html.index("{", html.index("const PAYLOAD = "))
    payload, _ = json.JSONDecoder().raw_decode(html, start)
    assert payload["LATTICE"] == json.loads(json.dumps(P.lattice))
    assert payload["LATTICE"]["rho"] == json.load(open(os.path.join(
        ROOT, "data", "reference", "density_validation.json"), encoding="utf-8"))
