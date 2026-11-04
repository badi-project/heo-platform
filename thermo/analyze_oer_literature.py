#!/usr/bin/env python3
"""Step 2, part 1 — can published OER overpotentials train a predictor at all?

Extracts eta_10 (OER overpotential at 10 mA cm-2) from the performance sentence of
the database entries, then asks how much of its variance is explained by *which
material* it is (between cation systems) versus *who measured it and how* (within the
same cation system: electrode, loading, substrate, electrolyte, iR correction — almost
never stated in abstracts).

If the same composition spreads as widely as different compositions, a model trained on
these numbers learns protocol noise: the intraclass correlation (ICC) gives the
ceiling of the fraction of variance any composition model could explain.

Writes QC/oer_literature_variance.txt and data/thermo/oer_eta10_literature.csv.
"""
import csv
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DB = os.path.join(ROOT, "data", "heo_database.csv")
OUT = os.path.join(ROOT, "data", "thermo", "oer_eta10_literature.csv")
REPORT = os.path.join(ROOT, "QC", "oer_literature_variance.txt")

# value in mV followed (closely) by "10 mA"; must sit in an OER clause, not HER / UOR
VAL = re.compile(r"(\d{2,3}(?:[.,]\d+)?)\s*mV(?:[^.;]{0,30}?)\b10\s*mA", re.I)
VAL2 = re.compile(r"(?:η\s*₁₀|η10|eta10|η\s*10)\s*(?:=|of|:)?\s*(\d{2,3}(?:[.,]\d+)?)\s*mV", re.I)
CLAUSE = re.compile(r"[;\n]|(?<=\.)\s")


def eta10(perf):
    """eta_10 for the OER from a performance sentence, or None."""
    for clause in CLAUSE.split(perf or ""):
        low = clause.lower()
        if any(k in low for k in ("her:", "her ", "hydrogen evolution", "uor", "urea",
                                  "orr", "nitrate")) and "oer" not in low:
            continue
        m = VAL.search(clause) or VAL2.search(clause)
        if m:
            v = float(m.group(1).replace(",", "."))
            if 100 <= v <= 700:
                return v
    return None


def main():
    rows = [r for r in csv.DictReader(open(DB, encoding="utf-8")) if r["app"] == "oer"]
    data = []
    for r in rows:
        v = eta10(r["perf"])
        if v is not None:
            data.append((r["sysk"], r["proto"], v, r["doi"], r["perf"]))
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "proto", "eta10_mV", "doi", "perf_text"])
        w.writerows(data)

    by = {}
    for s, p, v, *_ in data:
        by.setdefault(s, []).append(v)
    reps = {s: v for s, v in by.items() if len(v) >= 2}
    allv = np.array([v for _, _, v, *_ in data])
    # one-way random-effects ANOVA on systems reported at least twice
    groups = list(reps.values())
    k = len(groups)
    n = np.array([len(g) for g in groups])
    N = n.sum()
    grand = np.concatenate(groups).mean()
    ssb = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ssw = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in groups)
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    var_b = max((msb - msw) / n0, 0.0)
    icc = var_b / (var_b + msw)
    # bootstrap over systems
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(2000):
        gs = [groups[i] for i in rng.integers(0, k, k)]
        nn = np.array([len(g) for g in gs])
        NN = nn.sum()
        gm = np.concatenate(gs).mean()
        b = sum(len(g) * (np.mean(g) - gm) ** 2 for g in gs) / (k - 1)
        wv = sum(((np.array(g) - np.mean(g)) ** 2).sum() for g in gs) / (NN - k)
        nz = (NN - (nn ** 2).sum() / NN) / (k - 1)
        vb = max((b - wv) / nz, 0.0)
        boots.append(vb / (vb + wv) if vb + wv > 0 else 0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    lines = [
        "Etape 2 - les surtensions OER publiees peuvent-elles entrainer un predicteur ?",
        f"entrees 'oer' de la base : {len(rows)} ; eta_10 lisible dans la phrase de performance : "
        f"{len(data)} ({len(by)} systemes cationiques)",
        f"eta_10 publies : mediane {np.median(allv):.0f} mV, etendue {allv.min():.0f}-{allv.max():.0f} mV",
        f"systemes rapportes au moins deux fois : {k} ({N} valeurs)",
        f"  ecart-type a composition egale (meme systeme, autres labos) : {np.sqrt(msw):.0f} mV",
        f"  ecart-type entre systemes : {np.sqrt(var_b):.0f} mV",
        f"  correlation intraclasse (part de variance due a la composition) : {icc:.2f} "
        f"[IC 95 % {lo:.2f}-{hi:.2f}]",
        "",
        "exemples (meme systeme cationique, valeurs publiees en mV) :"]
    for s, v in sorted(reps.items(), key=lambda t: -len(t[1]))[:6]:
        lines.append(f"  {s:<24s} " + ", ".join(f"{x:.0f}" for x in sorted(v)))
    lines += ["",
              "lecture : une correlation intraclasse de x signifie qu'un modele parfait de la "
              "composition n'expliquerait au mieux que x de la variance des valeurs publiees ;",
              "le reste est l'electrode et le protocole, absents des resumes.",
              "",
              "genere par thermo/analyze_oer_literature.py"]
    text = "\n".join(lines) + "\n"
    open(REPORT, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
