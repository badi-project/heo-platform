"""Tests on the data pipeline: cleaned database, exported model, regenerated page."""
import csv
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402
import formula  # noqa: E402

E = features.load_elements()


def rows():
    with open(os.path.join(ROOT, "data", "heo_database.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def bundle():
    with open(os.path.join(ROOT, "app", "heo_model.json"), encoding="utf-8") as fh:
        return json.load(fh)


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_database_is_the_output_of_the_cleaning_script():
    out = run(os.path.join("tools", "clean_database.py"), "--check")
    assert out.returncode == 0, out.stdout + out.stderr


def test_page_payload_is_up_to_date():
    out = run(os.path.join("tools", "build_app.py"), "--check")
    assert out.returncode == 0, out.stdout + out.stderr


def test_every_entry_has_real_fractions_over_at_least_five_tabulated_cations():
    for r in rows():
        fr = json.loads(r["frac"])
        cats = r["cations"].split("-")
        assert cats == sorted(fr), r["formula_text"]
        assert len(cats) == int(r["n_cat"]) >= 5
        assert set(cats) <= set(E)
        assert abs(sum(fr.values()) - 1) < 1e-5
        assert min(fr.values()) >= formula.MIN_CATION_FRACTION - 1e-6
        assert r["stoich_source"] in {"formula", "attested_set"}


def test_stored_fractions_are_those_of_the_written_formula():
    for r in rows():
        if r["stoich_source"] != "formula":
            continue
        fr = formula.cation_fractions(r["formula_text"], E)
        stored = json.loads(r["frac"])
        assert set(fr) == set(stored)
        assert all(abs(fr[e] - stored[e]) < 1e-6 for e in fr), r["formula_text"]


def test_no_perovskite_has_lost_its_b_site():
    """Regression: v1.0 stored (Na0.2Bi0.2Ba0.2Sr0.2Ca0.2)TiO3 without Ti — 38 of 240
    perovskites had no B-site cation at all, which made the Goldschmidt factor wrong."""
    def a_site_only(e):
        return 57 <= E[e]["Z"] <= 71 or E[e]["group"] in (1, 2) or e in ("Y", "Bi", "Pb")
    pv = [r for r in rows() if r["proto"] == "perovskite"]
    assert pv
    assert not [r["formula_text"] for r in pv
                if all(a_site_only(e) for e in r["cations"].split("-"))]


def test_equimolar_fallback_is_never_used_for_multi_site_structures():
    for r in rows():
        if r["stoich_source"] == "attested_set":
            assert r["proto"] in {"spinel", "rocksalt", "fluorite", "bixbyite", "rutile", "unk"}


def test_corpus_no_longer_redistributes_publisher_abstracts():
    with open(os.path.join(ROOT, "data", "heo_corpus.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            assert "abstract" not in rec
            assert set(rec) >= {"doi", "title", "year", "tier"}


def test_exported_model_carries_its_validation():
    b = bundle()
    m = b["metrics"]
    assert m["n_labeled"] == sum(m["class_counts"].values())
    assert b["model"]["temperature"] == m["cv"]["temperature"]
    for key in ("cv", "temporal", "leave_chemistry_out", "baselines", "perovskite_binary"):
        assert key in m
    # the claim of the README: the model beats every baseline on the grouped validation
    assert m["cv"]["accuracy"] > max(v["accuracy"] for v in m["baselines"].values())
    assert m["export"]["gap"] == 0.0
    assert len(b["probe"]) == 200


def test_candidates_are_unpublished_in_domain_and_capped_per_family():
    with open(os.path.join(ROOT, "data", "heo_candidats.csv"), encoding="utf-8") as fh:
        cands = list(csv.DictReader(fh))
    published = {r["sysk"] for r in rows()}
    per = {}
    for c in cands:
        assert c["cations"] not in published
        assert c["in_domain"] == "1" and c["novel"] == "1"
        per[c["proto"]] = per.get(c["proto"], 0) + 1
    assert max(per.values()) <= 700
