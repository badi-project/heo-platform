"""Tests on the shipped application and the published data files."""
import csv
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sync_app  # noqa: E402

HAS_NODE = shutil.which("node") is not None
needs_node = pytest.mark.skipif(not HAS_NODE, reason="node is not on PATH")


def read(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def test_inlined_javascript_matches_the_standalone_files():
    """heo_designer.html inlines heo_engine.js and heo_ui.js; they must not drift."""
    assert sync_app.sync(check=True) == 0


@needs_node
def test_python_javascript_parity():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "parity_check.py"),
                          "--check"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


@needs_node
def test_entries_without_doi_do_not_produce_a_link():
    """Regression: 58 preprint entries have no DOI and rendered as https://doi.org/NaN."""
    ui = read("app", "heo_ui.js")
    start, stop = ui.index("const esc ="), ui.index("const SCREEN_TOTAL")
    driver = ui[start:stop] + """
const cases = [NaN, null, undefined, "", "nan", "10.1016/j.ceramint.2021.11.041"];
console.log(JSON.stringify(cases.map(refCell)));
"""
    out = subprocess.run(["node", "--input-type=module", "-e", driver],
                         capture_output=True, text=True, check=True)
    cells = json.loads(out.stdout)
    for cell in cells[:5]:
        assert "doi.org" not in cell and "sans DOI" in cell
    assert 'href="https://doi.org/10.1016/j.ceramint.2021.11.041"' in cells[5]
    assert 'rel="noopener noreferrer"' in cells[5]


def test_every_external_link_sets_noreferrer():
    html = read("app", "heo_designer.html")
    assert html.count('target="_blank"') == html.count('rel="noopener noreferrer"')


def test_candidate_table_announces_truncation():
    ui = read("app", "heo_ui.js")
    assert "rows.slice(0,MAXROWS)" in ui
    assert "premières lignes du tri courant" in ui


def test_database_shape_and_missing_dois():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "heo_database.csv"),
                                    encoding="utf-8")))
    assert len(rows) == 1133
    assert sum(1 for r in rows if not r["doi"]) == 58
    assert {"cations", "n_cat", "phase", "proto", "a_ang", "T_C", "route", "doi"} \
        <= set(rows[0])
    assert all(int(r["n_cat"]) >= 5 for r in rows)


def test_candidate_file_matches_the_exported_subset():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "heo_candidats.csv"),
                                    encoding="utf-8")))
    assert len(rows) == 3589
