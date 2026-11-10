"""Tests of the Python prediction engine (src/predict.py) and its command line
(tools/heo_predict.py). Numerical agreement with the JavaScript engine on the whole database
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
CLI = os.path.join(ROOT, "tools", "heo_predict.py")


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
    assert payload["LATTICE"] == json.loads(json.dumps(P.lattice))
    val = json.load(open(os.path.join(ROOT, "data", "reference", "lattice_validation.json"),
                         encoding="utf-8"))
    for fam, pct in payload["LATTICE"]["mae_pct"].items():
        assert pct == val["families"][fam]["uncertainty_pct"]


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
    """No header: one formula per line, and the commas of a formula are not separators."""
    src = tmp_path / "liste.txt"
    src.write_text("(FeCoNiCrMn)3O4\n\n(Ce0.2Zr0.2Hf0.2Sn0.2Ti0.2)O2\n(Mg,Co,Ni,Cu,Zn)O\n",
                   encoding="utf-8")
    out = run_cli("--csv", str(src))
    rows = list(csv.DictReader(out.stdout.splitlines()))
    assert [r["proto"] for r in rows] == ["spinel", "fluorite", "rocksalt"]
    assert rows[2]["formula_text"] == "(Mg,Co,Ni,Cu,Zn)O"


def test_record_is_text_in_fixed_columns_and_never_raises():
    rec = P.record("(FeCoNiCrMn)3O4")
    assert list(rec) == predict.ROW_COLUMNS and all(isinstance(v, str) for v in rec.values())
    assert rec["proto"] == "spinel" and rec["a_source"] == "chgnet" and rec["error"] == ""
    assert rec["frac"] == '{"Co": 0.2, "Cr": 0.2, "Fe": 0.2, "Mn": 0.2, "Ni": 0.2}'
    bad = P.record("hello")
    assert bad["proto"] == "" and "non lue" in bad["error"]


def test_flags_do_not_depend_on_the_hash_seed():
    """Regression: the flags came out of a set in arbitrary order, so the same file could
    be exported differently from one run to the next."""
    text = "[Lix(BaSrCaMg)(1−x)/4]TiO3"
    seen = set()
    for seed in ("0", "1", "2", "3"):
        out = subprocess.run([sys.executable, CLI, text, "--json"], capture_output=True,
                             text=True, env={**os.environ, "PYTHONHASHSEED": seed})
        seen.add(tuple(json.loads(out.stdout)["flags"]))
    assert len(seen) == 1 and list(next(iter(seen))) == sorted(next(iter(seen)))


def test_profile_and_xy_export(tmp_path):
    r = P.predict("(FeCoNiCrMn)3O4")
    prof = P.profile(r["pattern"])
    assert len(prof) == 3501 and prof[0][0] == 10.0 and abs(prof[-1][0] - 80.0) < 1e-9
    top = max(prof, key=lambda xy: xy[1])
    strongest = max(r["pattern"], key=lambda p: p["intensity"])
    assert top[1] == 100 and abs(top[0] - strongest["two_theta"]) <= 0.02
    dst = tmp_path / "motif.xy"
    out = run_cli("(FeCoNiCrMn)3O4", "--xy", str(dst))
    assert out.returncode == 0, out.stderr
    lines = dst.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("# (FeCoNiCrMn)3O4") and "a = 8.33" in lines[0]
    data = [ln.split() for ln in lines if not ln.startswith("#")]
    assert len(data) == 3501 and all(len(d) == 2 for d in data)
    assert run_cli("Li(NiCoMnAlFe)O2", "--xy", str(dst)).returncode != 0   # no cubic prototype


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
