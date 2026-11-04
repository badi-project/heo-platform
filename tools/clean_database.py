#!/usr/bin/env python3
"""Rebuild data/heo_database.csv from the written formulas.

Version 1.0 of the database computed every descriptor on an equimolar mixture of a
cation set chosen by the extraction language model. An independent check showed that
the set regularly lost a whole crystallographic site — (Na0.2Bi0.2Ba0.2Sr0.2Ca0.2)TiO3
was stored without Ti — and that some title-only entries carried cations that the title
never names. This script replaces the language-model set by what the paper writes:

1. ``formula_text`` is parsed by src/formula.py into real cation fractions;
2. if the formula yields fewer than five cations (prose such as "spinel oxide"), the
   language-model set is kept only for single-cation-site families, and only the
   cations attested in the formula text, the evidence sentence or the title (symbol
   or English name) — never for perovskite, pyrochlore and layered entries, whose
   stoichiometry cannot be guessed;
3. entries with fewer than five cations at >= 1 % of the cation sum are dropped;
4. exact duplicates (same DOI, same fractions, same labels) are merged.

The input is data/heo_database_raw.csv — the v1.0 database exactly as published,
never edited — so that a fix to the parser can bring back an entry dropped earlier.
The original language-model set is kept in ``cations_llm`` for traceability.

    python tools/clean_database.py            # rewrite data/heo_database.csv
    python tools/clean_database.py --check    # exit 1 if the file is not up to date
"""
import argparse
import csv
import io
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402
import formula  # noqa: E402

RAW = os.path.join(ROOT, "data", "heo_database_raw.csv")   # v1.0, never edited
DB = os.path.join(ROOT, "data", "heo_database.csv")
REPORT = os.path.join(ROOT, "QC", "database_cleaning.txt")
MIN_CATIONS = 5
SINGLE_SITE_OK = {"spinel", "rocksalt", "fluorite", "bixbyite", "rutile", "unk"}

NAMES = {
    "Li": "lithium", "Na": "sodium", "K": "potassium", "Mg": "magnesium",
    "Ca": "calcium", "Sr": "strontium", "Ba": "barium", "Al": "alumin", "Ga": "gallium",
    "In": "indium", "Sc": "scandium", "Y": "yttri", "La": "lanthan", "Ce": "cer",
    "Pr": "praseodym", "Nd": "neodym", "Sm": "samari", "Eu": "europ", "Gd": "gadolin",
    "Tb": "terbi", "Dy": "dysprosi", "Ho": "holmi", "Er": "erbi", "Tm": "thuli",
    "Yb": "ytterbi", "Lu": "luteti", "Ti": "titan", "Zr": "zircon", "Hf": "hafni",
    "V": "vanad", "Nb": "niob", "Ta": "tantal", "Cr": "chrom", "Mo": "molybd",
    "W": "tungst", "Mn": "mangan", "Fe": "iron", "Co": "cobalt", "Ni": "nickel",
    "Cu": "copper", "Zn": "zinc", "Cd": "cadmi", "Ag": "silver", "Pd": "pallad",
    "Pt": "platin", "Ru": "ruthen", "Ir": "iridi", "Rh": "rhodi", "Sn": "tin",
    "Pb": "lead", "Bi": "bismuth", "Sb": "antimon", "Te": "tellur", "Ge": "german",
    "Si": "silic", "As": "arsen", "Tl": "thalli", "Hg": "mercur",
}
# symbols that are also ordinary English words at the start of a sentence
AMBIGUOUS_SYMBOLS = {"In", "As", "I", "W", "V", "Y", "K"}
# entries rejected by the manual audits (QC/audit_stoechiometrie*.csv): no single
# composition can be recovered from what the extraction kept
MANUAL_REJECT = {
    "10.1021/acsami.0c17446": "serie decrite en prose (Ce + 4 metaux parmi 9)",
    "10.1016/j.materresbull.2026.114124": "indices perdus a l'extraction "
                                          "(La Sr1-(CoCrFeMnNi)O3 : site A illisible)",
}
# "Co-rich", "Fe-rich": the text says the composition is NOT equimolar
_RICH = re.compile(r"(?<![A-Za-z])[A-Z][a-z]?-rich(?![a-z])")


def attested(symbol, text):
    """Is the cation named in the text, as a formula symbol or by its English name?"""
    # 'Cu' in 'CaCu3Ti4O12' follows a lowercase letter; 'Co' in 'Coordination' is
    # excluded by the lookahead
    pat = r"(?<![A-Z])" + re.escape(symbol) + r"(?![a-z])"
    hits = [m for m in re.finditer(pat, text)]
    if symbol in AMBIGUOUS_SYMBOLS:
        # 'In this work', 'W cm-2', 'Y-doped' is fine; require a formula-like context
        hits = [m for m in hits if not text[m.end():m.end() + 1].isspace()]
    if hits:
        return True
    name = NAMES.get(symbol)
    return bool(name) and name in text.lower()


def clean_row(row, elements):
    llm = row.get("cations_llm") or row["cations"]
    llm_set = [e for e in llm.split("-") if e]
    text = row.get("formula_text") or ""
    if row["doi"] in MANUAL_REJECT:
        return None, "rejet manuel apres audit (" + MANUAL_REJECT[row["doi"]] + ")"
    parsed = formula.parse(text, elements)
    fr = formula.cation_fractions(text, elements)
    flags = set(parsed.flags)
    if len(fr) >= MIN_CATIONS:
        source = "formula"
    elif len(parsed.amounts) >= 2:
        # a real formula was read: it is authoritative, even with fewer than five
        # cations (CaCu3Ti3.85Mn0.05Sn0.1O12, a FeCoNiOx-MnCrOx heterojunction)
        return None, "formule lisible a moins de cinq cations"
    else:
        context = " ".join([text, row.get("evidence") or "", row.get("title") or ""])
        kept = [e for e in llm_set if e in elements and attested(e, context)]
        if row["proto"] not in SINGLE_SITE_OK:
            return None, f"stoechiometrie illisible pour une structure a plusieurs sites " \
                         f"({row['proto']})"
        if len(kept) < MIN_CATIONS:
            missing = sorted(set(llm_set) - set(kept))
            return None, f"cations non attestes dans le texte source ({'-'.join(missing)})" \
                if missing else "moins de cinq cations"
        if _RICH.search(text):
            return None, "composition non equimolaire non chiffree (X-rich)"
        fr = {e: 1.0 / len(kept) for e in sorted(kept)}
        source = "attested_set"
        flags.add("equimolar_fallback")
    unknown = [e for e in fr if e not in elements]
    if unknown:
        return None, f"cation absent de la table elementaire ({'-'.join(unknown)})"
    cats = sorted(fr)
    out = dict(row)
    out["cations_llm"] = llm
    out["cations"] = "-".join(cats)
    out["sysk"] = "-".join(cats)
    out["n_cat"] = str(len(cats))
    out["frac"] = json.dumps({e: round(fr[e], 6) for e in cats})
    out["equimolar_used"] = str(source == "attested_set")
    out["stoich_source"] = source
    out["parse_flags"] = ";".join(sorted(flags))
    return out, None


LAB = os.path.join(ROOT, "data", "lab_results.csv")
LAB_FIELDS = ["id", "date", "formula_text", "proto", "phase", "T_C", "atm", "a_ang", "route",
              "notes"]


def lab_rows(fields):
    """Internal results (data/lab_results.csv), written by the laboratory: one line per
    synthesis, failures included — a multiphase batch is a negative label the literature
    almost never provides. They become database rows with tier 'lab' and doi 'lab:<id>',
    and go through the same formula parser as the literature."""
    if not os.path.exists(LAB):
        return []
    out = []
    for r in csv.DictReader(open(LAB, encoding="utf-8", newline="")):
        if not (r.get("id") or "").strip() or not (r.get("formula_text") or "").strip():
            continue
        if not re.match(r"^\d{4}", (r.get("date") or "").strip()):
            print(f"lab_results.csv : ligne {r['id']} ignoree, date (AAAA-MM-JJ) manquante")
            continue
        row = {k: "" for k in fields}
        row.update({"doi": "lab:" + r["id"].strip(), "title": r.get("notes", ""),
                    "journal": "laboratoire", "year": (r.get("date") or "")[:4],
                    "tier": "lab", "formula_text": r["formula_text"].strip(),
                    "proto": r["proto"].strip() or "unk",
                    "phase": r["phase"].strip() or "unk", "T_C": r.get("T_C", ""),
                    "atm": r.get("atm", "") or "unk", "a_ang": r.get("a_ang", ""),
                    "route": r.get("route", "") or "unk", "evidence": r.get("notes", ""),
                    "cations": "", "cations_llm": ""})
        out.append(row)
    return out


def main(check=False):
    elements = features.load_elements()
    with open(RAW, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    fields = list(rows[0].keys())
    rows += lab_rows(fields + ["cations_llm"])
    for extra in ("cations_llm", "stoich_source", "parse_flags"):
        if extra not in fields:
            fields.append(extra)

    kept, dropped, seen = [], Counter(), set()
    changed = Counter()
    for row in rows:
        out, why = clean_row(row, elements)
        if out is None:
            dropped[why.split(" (")[0]] += 1
            continue
        key = (out["doi"], out["frac"], out["proto"], out["phase"], out["formula_text"])
        if key in seen:
            dropped["doublon exact"] += 1
            continue
        seen.add(key)
        llm = set(out["cations_llm"].split("-"))
        new = set(out["cations"].split("-"))
        if out["tier"] == "lab":
            changed["resultats internes (data/lab_results.csv)"] += 1
        elif new != llm:
            changed["ensemble de cations corrige"] += 1
            if new - llm:
                changed["  dont cations ajoutes (site oublie)"] += 1
            if llm - new:
                changed["  dont cations retires (dopant, non atteste)"] += 1
        kept.append(out)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in kept:
        w.writerow({k: r.get(k, "") for k in fields})
    text = buf.getvalue()

    src = Counter(r["stoich_source"] for r in kept)
    lines = [
        f"entrees en entree : {len(rows)}",
        f"entrees conservees : {len(kept)}",
        f"  stoechiometrie lue dans la formule : {src['formula']}",
        f"  ensemble attesté, repli equimolaire : {src['attested_set']}",
        f"entrees ecartees : {sum(dropped.values())}",
    ] + [f"  {k} : {v}" for k, v in dropped.most_common()] + [
        "corrections par rapport a l'ensemble du modele de langue :",
    ] + [f"  {k} : {v}" for k, v in changed.items()] + [
        f"systemes cationiques distincts : {len(set(r['sysk'] for r in kept))}",
        f"cations distincts : {len(set(e for r in kept for e in r['cations'].split('-')))}",
        "",
        "genere par tools/clean_database.py",
    ]
    report = "\n".join(lines) + "\n"
    print(report, end="")

    with open(DB, encoding="utf-8", newline="") as fh:
        current = fh.read()
    if check:
        if current != text:
            print("data/heo_database.csv is not the output of tools/clean_database.py")
            return 1
        return 0
    with open(DB, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    with open(REPORT, "w", encoding="utf-8") as fh:
        fh.write(report)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, do not rewrite")
    sys.exit(main(check=ap.parse_args().check))
