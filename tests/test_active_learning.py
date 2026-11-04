"""Step 3: the published benchmark and proposals are consistent with their rules."""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_proposals_follow_the_selection_rule():
    rows = list(csv.DictReader(open(os.path.join(ROOT, "data", "next_experiments.csv"),
                                    encoding="utf-8")))
    known = {r["sysk"] for r in csv.DictReader(open(os.path.join(ROOT, "data",
                                                                 "heo_database.csv"),
                                                    encoding="utf-8"))}
    lab = set("Mg Al Cr Mn Fe Co Ni Cu Zn Ga".split())
    assert 1 <= len(rows) <= 8
    for i, r in enumerate(rows):
        cats = set(r["system"].split("-"))
        assert cats <= lab and len(cats) in (5, 6)
        assert r["system"] not in known                     # unpublished
        assert float(r["family_entropy"]) >= 0.3            # genuinely uncertain
        for other in rows[:i]:                              # diverse
            assert len(cats & set(other["system"].split("-"))) < 4
    ent = [float(r["family_entropy"]) for r in rows]
    assert ent == sorted(ent, reverse=True)


def test_benchmark_states_its_criterion():
    rep = open(os.path.join(ROOT, "QC", "active_learning.txt"), encoding="utf-8").read()
    assert re.search(r"10 decoupages", rep)
    assert re.search(r"\(critere : >= 25 %.*-> (atteint|non atteint)", rep)
