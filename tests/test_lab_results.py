"""Internal results (data/lab_results.csv) enter the database through the same parser."""
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "src"))
import clean_database as cd  # noqa: E402
import features  # noqa: E402


def test_lab_rows_are_parsed_and_dated(tmp_path, monkeypatch):
    lab = tmp_path / "lab_results.csv"
    lab.write_text(
        "id,date,formula_text,proto,phase,T_C,atm,a_ang,route,notes\n"
        "B-001,2026-07-15,(FeCoNiCrMn)3O4,spinel,single,900,air,8.328,sol-gel,lot 1\n"
        "B-002,2026-07-20,(FeCoNiCuZn)3O4,spinel,multi,700,air,,sol-gel,phase secondaire\n"
        "B-003,,(MgCoNiCuZn)O,rocksalt,single,1000,air,,solide,sans date\n",
        encoding="utf-8")
    monkeypatch.setattr(cd, "LAB", str(lab))
    fields = list(csv.DictReader(open(cd.RAW, encoding="utf-8")).fieldnames) + ["cations_llm"]
    rows = cd.lab_rows(fields)
    assert [r["doi"] for r in rows] == ["lab:B-001", "lab:B-002"]   # undated line skipped
    E = features.load_elements()
    out = [cd.clean_row(r, E)[0] for r in rows]
    assert out[0]["cations"] == "Co-Cr-Fe-Mn-Ni" and out[0]["tier"] == "lab"
    assert out[1]["phase"] == "multi" and out[1]["year"] == "2026"


def test_template_has_the_expected_columns():
    head = open(os.path.join(ROOT, "data", "lab_results_template.csv"), encoding="utf-8").readline()
    assert head.strip().split(",") == cd.LAB_FIELDS
