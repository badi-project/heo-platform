"""Tests of the Python prediction engine (src/predict.py) and its command line
(tools/predict.py). Numerical agreement with the JavaScript engine on the whole database
is the job of tools/parity_check.py (tests/test_app.py runs it); these tests pin the
behaviour of the Python side on its own."""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import predict  # noqa: E402

P = predict.Predictor()
CLI = os.path.join(ROOT, "tools", "predict.py")


def run_cli(*args, cwd=None):
    return subprocess.run([sys.executable, CLI, *args], capture_output=True, text=True, cwd=cwd)


def test_exported_trees_reproduce_the_frozen_probe():
    """The probabilities scikit-learn froze at export time come back from predict()."""
    display = P.bundle["classes_order"]
    for case in P.bundle["probe"]:
        r = P.predict(case["frac"], pattern=False)
        got = [r["probabilities"][c] for c in display]
        assert max(abs(u - v) for u, v in zip(got, case["probs"])) < 1e-9


def test_reference_compositions():
    """The two founding compositions of the field land in their own family."""
    r = P.predict("(Mg0.2Co0.2Ni0.2Cu0.2Zn0.2)O")
    assert r["family"] == "rocksalt" and r["in_domain"] and r["known_entries"] > 50
    assert r["lattice"]["source"] == "geometric" and r["lattice"]["space_group"] == "Fm-3m"
    r = P.predict("(FeCoNiCrMn)3O4")
    assert r["family"] == "spinel" and r["in_domain"]
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-12
    assert list(r["probabilities"].values()) == sorted(r["probabilities"].values(), reverse=True)
    # covered spinel: the CHGNet lattice model replaces the geometric one, and the
    # oxidation enthalpy was computed directly for this very composition
    assert r["lattice"]["source"] == "chgnet" and 8.25 < r["lattice"]["a"] < 8.40
    assert r["thermo"]["covered"] and r["thermo"]["direct"] and r["thermo"]["verdict"] == "spinel"
    # the strongest spinel reflection is (311), near 35.7 degrees with Cu K-alpha
    top = max(r["pattern"], key=lambda p: p["intensity"])
    assert top["hkl"] == "311" and 35 < top["two_theta"] < 36.5 and top["intensity"] == 100


def test_formula_and_fractions_give_the_same_prediction():
    a = P.predict("(Na0.2Bi0.2Ba0.2Sr0.2Ca0.2)TiO3")
    b = P.predict({"Na": 1, "Bi": 1, "Ba": 1, "Sr": 1, "Ca": 1, "Ti": 5})
    assert a["family"] == b["family"] == "perovskite"
    assert a["probabilities"] == b["probabilities"]
    # key order of the dictionary must not matter
    c = P.predict({"Ti": 5, "Ca": 1, "Sr": 1, "Ba": 1, "Bi": 1, "Na": 1})
    assert c["probabilities"] == a["probabilities"] and c["pattern"] == a["pattern"]


def test_layered_family_has_no_cubic_prototype():
    r = P.predict("Li(NiCoMnAlFe)O2")
    assert r["family"] == "layered" and r["lattice"] is None and r["pattern"] == []
    assert r["thermo"] == {"covered": False, "missing": ["Li"]} or not r["thermo"]["covered"]


def test_out_of_domain_is_flagged():
    r = P.predict("(MoWNbTaV)O")
    assert not r["in_domain"] and r["ad_distance"] > r["ad_threshold"]
    assert any("hors du domaine" in w for w in r["warnings"])


@pytest.mark.parametrize("text, reason", [
    ("", "non lue"), ("hello", "non lue"), ("Fe3O4", "non lue"),
    ("BaTiO3", "au moins 4"), ("(UThCeZrHf)O2", "sans propriétés tabulées"),
])
def test_unusable_input_is_refused_with_its_reason(text, reason):
    with pytest.raises(predict.PredictionError, match=reason):
        P.predict(text)


def test_thermo_payload_is_what_the_page_embeds():
    """tools/build_app.py inlines predict.thermo_payload(): one source for both."""
    html = open(os.path.join(ROOT, "app", "heo_designer.html"), encoding="utf-8").read()
    start = html.index("{", html.index("const PAYLOAD = "))
    payload, _ = json.JSONDecoder().raw_decode(html, start)
    assert payload["THERMO"] == json.loads(json.dumps(P.thermo))
    assert payload["LATTICE"]["mae_pct"] == predict.LATTICE_MAE_PCT


def test_cli_single_formula_and_exit_status():
    out = run_cli("(FeCoNiCrMn)3O4", "--peaks")
    assert out.returncode == 0, out.stderr
    assert "spinelle" in out.stdout and "Fd-3m" in out.stdout and "311" in out.stdout
    out = run_cli("(FeCoNiCrMn)3O4", "--json")
    assert json.loads(out.stdout)["family"] == "spinel"
    out = run_cli("BaTiO3")
    assert out.returncode == 1 and "pas de prédiction" in out.stdout


def test_cli_works_from_any_directory(tmp_path):
    out = run_cli("(Mg0.2Co0.2Ni0.2Cu0.2Zn0.2)O", cwd=tmp_path)
    assert out.returncode == 0 and "rocksalt" in out.stdout


def test_cli_batch_keeps_columns_and_reports_bad_lines(tmp_path):
    src, dst = tmp_path / "in.csv", tmp_path / "out.csv"
    src.write_text("id,formule,note\n"
                   "A1,(FeCoNiCrMn)3O4,lot 1\n"
                   'A2,"(Mg,Co,Ni,Cu,Zn)O",\n'
                   "A3,Fe3O4,témoin\n", encoding="utf-8")
    out = run_cli("--csv", str(src), "--out", str(dst))
    assert out.returncode == 0, out.stderr
    assert "2 prédiction(s) sur 3 ligne(s)" in out.stderr
    rows = list(csv.DictReader(open(dst, encoding="utf-8")))
    assert [r["id"] for r in rows] == ["A1", "A2", "A3"] and rows[0]["note"] == "lot 1"
    assert rows[0]["proto"] == "spinel" and rows[0]["a_source"] == "chgnet"
    assert rows[1]["proto"] == "rocksalt" and rows[1]["cations"] == "Co-Cu-Mg-Ni-Zn"
    assert rows[2]["proto"] == "" and "non lue" in rows[2]["error"]


def test_cli_batch_reads_a_plain_list(tmp_path):
    src = tmp_path / "liste.txt"
    src.write_text("(FeCoNiCrMn)3O4\n\n(Ce0.2Zr0.2Hf0.2Sn0.2Ti0.2)O2\n", encoding="utf-8")
    out = run_cli("--csv", str(src))
    rows = list(csv.DictReader(out.stdout.splitlines()))
    assert [r["proto"] for r in rows] == ["spinel", "fluorite"]


def test_batch_on_the_database_agrees_with_the_training_labels(tmp_path):
    """Sanity check of the whole chain: scoring the database through the command line
    gives back the labels the model was fitted on (this is a training-set figure, not a
    validation — the validated accuracy is in QC/training_report.txt)."""
    dst = tmp_path / "db.csv"
    truth = list(csv.DictReader(open(os.path.join(ROOT, "data", "heo_database.csv"),
                                     encoding="utf-8")))
    out = run_cli("--csv", os.path.join(ROOT, "data", "heo_database.csv"), "--out", str(dst))
    assert out.returncode == 0, out.stderr
    pred = list(csv.DictReader(open(dst, encoding="utf-8")))
    assert len(pred) == len(truth)
    labelled = [(t["proto"], p["proto"]) for t, p in zip(truth, pred)
                if t["proto"] in P.model["classes"] and not p["error"]]
    agree = sum(a == b for a, b in labelled) / len(labelled)
    assert len(labelled) > 800 and agree > 0.95
