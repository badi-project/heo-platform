#!/usr/bin/env python3
"""Validate the theoretical (X-ray) density returned by src/predict.py.

The density is not learned: it is the mass of the prototype cell over its volume. Two
things can still be wrong — the cell content (sites per cell, molar masses, A/B split)
and the lattice parameter it inherits. They are checked separately, and the acceptance
criteria below were written before the numbers were computed.

A. Cell content. With the *measured* lattice parameter of reference oxides, one per
   cubic prototype, the density must fall within 1 % of the published value
   (data/reference/density_references.csv, every line with its source).
   Criterion: every line within 1 %.

B. Inherited lattice error. On the high-entropy oxides of measured lattice parameter
   (data/reference/lattice_measured.csv, single-phase, one line per composition), the
   density computed from the *predicted* lattice parameter is compared with the one
   computed from the measured parameter. This is not independent of the lattice
   validation (QC/lattice_validation.txt): a relative error e on a gives 3e on the
   density. Criterion: mean absolute error at most 2 % for spinels (CHGNet model) and
   6 % for the geometric model — three times the lattice errors published with v1.0.

Part A is a gate: a wrong cell content is a bug. Part B is a measurement: its outcome,
met or not, is written to the report and to data/reference/density_validation.json, from
which the page and src/predict.py take the uncertainty they display — the larger of the
propagated lattice uncertainty and the error actually observed.

    python tools/validate_density.py            # rewrite the report and the JSON summary
    python tools/validate_density.py --check    # exit 1 if they are not up to date or if
                                                # the cell-content criterion is not met
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402
import predict  # noqa: E402

TOL_CONTENT_PCT = 1.0
TOL_MAE_PCT = {"chgnet": 2.0, "geometric": 6.0}
REPORT = os.path.join(ROOT, "QC", "density_validation.txt")
SUMMARY = os.path.join(ROOT, "data", "reference", "density_validation.json")


def read_csv(*parts):
    with open(os.path.join(ROOT, *parts), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def measured_heo(P):
    """(system, prototype, fractions, measured a, origin) for every high-entropy oxide of
    data/reference/lattice_measured.csv declared single-phase, each composition once at
    the median of its measurements (tools/lattice_dataset.py)."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import lattice_dataset as ld
    out = []
    for c in ld.unique_compositions(ld.load(), P).values():
        if c["group"] == "heo":
            out.append(("-".join(c["frac"]), c["proto"], c["frac"], c["a"],
                        ", ".join(sorted({r["source"] for r in c["rows"]}))))
    return out


def build():
    P = predict.Predictor()
    lines = ["Densite theorique (rayons X) : masse de la maille du prototype / volume",
             "",
             "A. Contenu de la maille : densite calculee avec la maille MESUREE d'oxydes de reference",
             f"   critere ecrit avant le calcul : chaque ligne a moins de {TOL_CONTENT_PCT:.0f} % de la valeur publiee",
             "",
             "   compose              prototype    a (A)   calcule  publie  ecart   nature de la valeur publiee"]
    worst_a = 0.0
    for r in read_csv("data", "reference", "density_references.csv"):
        rho = P.xray_density(r["proto"], json.loads(r["frac"]), float(r["a_ang"]))
        ref = float(r["rho_ref_g_cm3"])
        err = 100 * (rho - ref) / ref
        worst_a = max(worst_a, abs(err))
        lines.append(f"   {r['compound']:<20s} {r['proto']:<11s} {float(r['a_ang']):7.4f}  {rho:7.3f} "
                     f"{ref:7.3f}  {err:+5.2f} %  {r['kind']}")
    ok_a = worst_a <= TOL_CONTENT_PCT
    lines += ["",
              f"   ecart maximal {worst_a:.2f} % -> {'atteint' if ok_a else 'NON ATTEINT'}",
              "   pyrochlore : pas de densite publiee sourcee ; le contenu de la maille "
              "(16 A + 16 B + 56 O, soit 8 A2B2O7) est verifie par tests/test_density.py",
              "",
              "B. Erreur heritee de la maille : oxydes a haute entropie de maille mesuree",
              "   critere ecrit avant le calcul : ecart absolu moyen <= "
              f"{TOL_MAE_PCT['chgnet']:.0f} % (spinelles, modele CHGNet), "
              f"<= {TOL_MAE_PCT['geometric']:.0f} % (modele geometrique)",
              "",
              "   systeme                        prototype   a mesure  a predit  modele      rho(a mesure)  rho(a predit)  ecart"]
    errs = {"chgnet": [], "geometric": []}
    for sysk, proto, frac, a_exp, origin in measured_heo(P):
        a_th = P.thermo_lattice(proto, frac)
        a_pred, src = (a_th, "chgnet") if a_th is not None else \
            (P.geometric_lattice(proto, frac), "geometric")
        rho_exp, rho_pred = P.xray_density(proto, frac, a_exp), P.xray_density(proto, frac, a_pred)
        err = 100 * (rho_pred - rho_exp) / rho_exp
        errs[src].append(abs(err))
        lines.append(f"   {sysk:<30s} {proto:<10s} {a_exp:8.4f}  {a_pred:8.4f}  {src:<10s} "
                     f"{rho_exp:10.3f}  {rho_pred:13.3f}   {err:+5.2f} %")
    lines.append("")
    summary = {"reference": {"n": len(read_csv("data", "reference", "density_references.csv")),
                             "max_pct": round(worst_a, 2), "criterion_pct": TOL_CONTENT_PCT,
                             "met": ok_a}}
    for src, label in (("chgnet", "spinelles, modele CHGNet"), ("geometric", "modele geometrique")):
        if not errs[src]:
            continue
        mae = features.nsum(errs[src]) / len(errs[src])
        ok = mae <= TOL_MAE_PCT[src]
        # keys are the model names of the page (aSrc in heo_engine.js)
        summary["geo" if src == "geometric" else src] = {
            "n": len(errs[src]), "mae_pct": round(mae, 2), "max_pct": round(max(errs[src]), 2),
            "criterion_pct": TOL_MAE_PCT[src], "met": ok}
        lines.append(f"   {label} : ecart absolu moyen {mae:.2f} % sur {len(errs[src])} systeme(s), "
                     f"maximum {max(errs[src]):.2f} % -> {'atteint' if ok else 'NON ATTEINT'}")
    lines += ["",
              "Lecture : avec une maille mesuree, la densite est exacte au contenu de la maille pres",
              "(partie A) ; c'est la valeur a utiliser pour la densite relative d'une pastille frittee.",
              "Avec une maille predite, l'incertitude affichee est la plus grande de deux valeurs :",
              "trois fois l'incertitude de la maille, ou l'ecart moyen observe ci-dessus pour ce modele.",
              "Mailles mesurees : data/reference/lattice_measured.csv (sources par ligne) ; erreur de",
              "maille par famille : QC/lattice_validation.txt.",
              "Limite : le sous-reseau d'oxygene est suppose complet. Une fluorine riche en cations",
              "trivalents contient des lacunes d'oxygene et est plus legere que la valeur affichee",
              "(jusqu'a environ 4 % pour quatre cations trivalents sur cinq).",
              "",
              "genere par tools/validate_density.py"]
    return "\n".join(lines) + "\n", json.dumps(summary, indent=1) + "\n", ok_a


def main(check=False):
    report, summary, ok = build()
    if check:
        stale = [os.path.relpath(p, ROOT) for p, text in ((REPORT, report), (SUMMARY, summary))
                 if not os.path.exists(p) or open(p, encoding="utf-8").read() != text]
        if stale:
            print(", ".join(stale) + " not up to date: run tools/validate_density.py")
            return 1
        print("QC/density_validation.txt is up to date"
              + ("" if ok else " — but the cell-content criterion is NOT met"))
        return 0 if ok else 1
    for path, text in ((REPORT, report), (SUMMARY, summary)):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    print(report, end="")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify only, do not rewrite")
    sys.exit(main(check=ap.parse_args().check))
