"""Tests of the thermodynamics layer (thermo/).

CHGNet is not run here: the published results in data/thermo/ are checked for
consistency with their reports, the oxygen thermochemistry against NIST, and the
surrogates shipped to the web page against the raw calculations."""
import csv
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data", "thermo")


def rows(name):
    return list(csv.DictReader(open(os.path.join(D, name), encoding="utf-8")))


def report(name):
    return open(os.path.join(ROOT, "QC", name), encoding="utf-8").read()


def test_oxygen_thermochemistry_matches_nist():
    pytest.importorskip("pymatgen")
    import sys
    sys.path.insert(0, os.path.join(ROOT, "thermo"))
    import heo_thermo as ht
    dh, s = ht.o2_h_s(298.15)
    assert abs(dh) < 0.01 and abs(s - 205.15) < 0.05          # NIST: S°(298) = 205.15
    dh, s = ht.o2_h_s(1000)
    assert abs(dh - 22.707) < 0.02 and abs(s - 243.58) < 0.05  # JANAF O2 at 1000 K
    # Reuter & Scheffler: Delta mu_O(1000 K, 1 atm) = -1.10 eV, referenced to 0 K; here the
    # reference is the 298 K enthalpy (the MP elemental reference is fitted at 298 K), which
    # shifts it by -(H298 - H0)/2 = -8.683/2 kJ/mol = -0.045 eV
    assert abs(ht.delta_mu_O(1000, p_bar=1.01325) - (-1.10 - 0.045)) < 0.01


def test_rocksalt_benchmark_recovers_the_eight_outcomes():
    r = rows("rocksalt_benchmark.csv")
    assert len({x["system"] for x in r}) == 8
    assert all(sum(1 for x in r if x["system"] == s) == 3 for s in {x["system"] for x in r})
    assert "issues experimentales retrouvees : 8 / 8" in report("thermo_rocksalt_benchmark.txt")
    th = json.load(open(os.path.join(D, "rocksalt_thresholds.json")))
    # thresholds are the largest single-phase values: (MgCoNiCuZn)O sits on them
    h5 = [float(x["dH_mix_meV_atom"]) for x in r if x["system"] == "Co-Cu-Mg-Ni-Zn"]
    assert abs(sum(h5) / 3 - th["dH_mix_meV_atom"]) < 1e-6


def test_redox_corrections_come_from_mo_to_m3o4():
    c = json.load(open(os.path.join(D, "redox_corrections.json")))
    assert set(c["eV_per_electron"]) == {"Co", "Fe", "Mn"}
    rep = report("thermo_redox_validation.txt")
    m = re.search(r"Fe3O4 \+ O -> 3 Fe2O3 : ecart brut ([+-][0-9.]+) -> ([+-][0-9.]+) eV/O", rep)
    assert m and abs(float(m.group(2))) < 0.1 < abs(float(m.group(1)))


def test_oxidation_surrogate_reproduces_its_fit():
    np = pytest.importorskip("numpy")
    d = rows("oxidation_descriptor.csv")
    s = json.load(open(os.path.join(D, "oxidation_surrogate.json")))
    assert s["n_systems"] == len(d)
    els = s["elements"]
    X = np.array([[json.loads(r["frac"]).get(e, 0.0) for e in els] for r in d])
    y = np.array([float(r["dH_ox_corr_eV_per_O"]) for r in d])
    h, *_ = np.linalg.lstsq(X, y, rcond=None)
    assert np.allclose(h, s["h_eV_per_O"], atol=1e-3)
    assert s["auc_system"] > 0.85 and s["loo_r2"] > 0.8


def test_spinel_lattice_beats_the_geometric_model():
    rep = report("thermo_spinel_lattice.txt")
    m = re.search(r"ecart absolu moyen sur les spinelles HEO : ([0-9.]+) %", rep)
    assert m and float(m.group(1)) < 1.0      # v1.0 geometric model: 1.9 %


def test_rocksalt_benchmark_reports_its_leave_one_out_score():
    # the 8 / 8 uses thresholds read on the single-phase compositions themselves
    assert re.search(r"en un-contre-tous .* : [0-8] / 8", report("thermo_rocksalt_benchmark.txt"))


def test_vacancy_energy_tracks_measured_reducibility():
    scipy = pytest.importorskip("scipy.stats")
    b = rows("vacancy_binaries.csv")
    assert len(b) == 9
    y = [float(r["dH_red_exp_eV_per_O"]) for r in b]
    rho = scipy.spearmanr([float(r["E_vac_corr_eV"]) for r in b], y)[0]
    rho_raw = scipy.spearmanr([float(r["E_vac_raw_eV"]) for r in b], y)[0]
    assert rho >= 0.7 and rho_raw >= 0.7            # criterion fixed before the calculation
    m = re.search(r"correlation de rang \(Spearman\) : ([0-9.]+) corrigee", report("thermo_vacancy.txt"))
    assert m and abs(float(m.group(1)) - rho) < 0.005


def test_measured_reduction_enthalpies_come_from_the_sourced_table():
    hf = {r["compound"]: float(r["dfH_kJmol"]) for r in rows("reference_thermo.csv")
          if float(r["T_K"]) == 298.15}
    hf.update({"Co": 0.0, "Fe": 0.0, "Cu": 0.0, "Mn": 0.0})
    # "A -> n B + O": enthalpy per O atom released, O referenced to 1/2 O2
    reactions = {"Co3O4": ("Co3O4", 1, "CoO", 3), "CoO": ("CoO", 1, "Co", 1),
                 "Fe2O3": ("Fe2O3", 3, "Fe3O4", 2), "Fe3O4": ("Fe3O4", 1, "FeO", 3),
                 "FeO": ("FeO", 1, "Fe", 1), "CuO": ("CuO", 2, "Cu2O", 1),
                 "Cu2O": ("Cu2O", 1, "Cu", 2), "Mn3O4": ("Mn3O4", 1, "MnO", 3),
                 "MnO": ("MnO", 1, "Mn", 1)}
    for r in rows("vacancy_binaries.csv"):
        a, na, b, nb = reactions[r["oxide"]]
        dh = (nb * hf[b] - na * hf[a]) / 96.485332
        assert abs(dh - float(r["dH_red_exp_eV_per_O"])) < 0.002, r["oxide"]


def test_surrogate_coverage_excludes_undetermined_cations():
    s = json.load(open(os.path.join(D, "oxidation_surrogate.json")))
    n = dict(zip(s["elements"], s["n_systems_per_element"]))
    html = open(os.path.join(ROOT, "app", "heo_designer.html"), encoding="utf-8").read()
    start = html.index("{", html.index("const PAYLOAD = "))
    payload, _ = json.JSONDecoder().raw_decode(html, start)
    cov = payload["THERMO"]["ox"]["cov"]
    assert all(n[e] >= 2 for e in cov)
    assert all(e not in cov for e in s["not_covered"])


def test_page_carries_the_thermo_payload():
    html = open(os.path.join(ROOT, "app", "heo_designer.html"), encoding="utf-8").read()
    start = html.index("{", html.index("const PAYLOAD = "))
    payload, _ = json.JSONDecoder().raw_decode(html, start)
    th = payload["THERMO"]
    assert th and th["ox"]["elements"] and "Co-Cr-Fe-Mn-Ni" in th["systems"]
    assert th["systems"]["Co-Cu-Mg-Ni-Zn"]["rs"]["single"] is True
    assert 'id="thermocard"' in html
