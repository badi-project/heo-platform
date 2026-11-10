#!/usr/bin/env python3
"""Predict from the command line, without the browser.

    python tools/heo_predict.py "(FeCoNiCrMn)3O4"                 # one formula, readable report
    python tools/heo_predict.py "(FeCoNiCrMn)3O4" --peaks         # ... with the reflection list
    python tools/heo_predict.py "La(CrMnFeCoNi)O3" --json         # machine-readable
    python tools/heo_predict.py "(FeCoNiCrMn)3O4" --xy motif.xy   # simulated profile, 2 columns
    python tools/heo_predict.py --csv formules.csv --out predictions.csv    # a whole file

The CSV file holds one formula per line, written as in a paper, in a column named
``formula_text``, ``formula``, ``formule`` or ``composition`` (or chosen with --column);
a file without such a header is read as a plain list, one formula per line. Every other column
is copied to the output. A line that cannot be predicted keeps its reason in the
``error`` column instead of stopping the run.

Uses src/predict.py — standard library only, same numbers as the web page
(tools/parity_check.py).
"""
import argparse
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from predict import ROW_COLUMNS, PredictionError, Predictor  # noqa: E402

PROTO_FR = {"spinel": "spinelle", "perovskite": "pérovskite", "rocksalt": "rocksalt (NaCl)",
            "fluorite": "fluorine", "pyrochlore": "pyrochlore", "layered": "lamellaire"}
VERDICT_FR = {"rocksalt": "plutôt rocksalt", "spinel": "plutôt spinelle",
              "boundary": "zone frontière"}
SOURCE_FR = {"chgnet": "potentiel CHGNet calibré sur spinelles binaires",
             "geometric": "modèle géométrique, rayons de Shannon"}
FORMULA_COLUMNS = ("formula_text", "formula", "formule", "composition")
OUT_COLUMNS = ROW_COLUMNS


def fr(v, n=3):
    return f"{v:.{n}f}".replace(".", ",")


def report(r, peaks=False):
    """Readable summary of one prediction, in the wording of the page."""
    out = [r["formula"] or "-".join(r["fractions"])]
    out.append("  lu : " + " · ".join(f"{e} {fr(100 * x, 1)} %" for e, x in r["fractions"].items()))
    order = list(r["probabilities"].items())
    out.append(f"  famille structurale : {PROTO_FR[r['family']]}, probabilité "
               f"{fr(100 * r['probability'], 0)} %  (puis {PROTO_FR[order[1][0]]} "
               f"{fr(100 * order[1][1], 0)} %)")
    out.append("  domaine d'applicabilité : "
               + ("dans le domaine" if r["in_domain"] else "HORS DOMAINE")
               + f" (distance {fr(r['ad_distance'], 2)}, seuil {fr(r['ad_threshold'], 2)})")
    n = r["known_entries"]
    out.append("  base : " + (f"déjà publié ({n} entrée{'s' if n > 1 else ''})" if n
                              else "combinaison absente de la base"))
    lat = r["lattice"]
    if lat:
        out.append(f"  maille : a = {fr(lat['a'])} ± {fr(lat['uncertainty_A'], 2)} Å "
                   f"({lat['space_group']} ; {SOURCE_FR[lat['source']]})")
        out.append(f"  densité théorique : {fr(lat['density_g_cm3'], 2)} ± "
                   f"{fr(lat['density_uncertainty_g_cm3'], 2)} g/cm³ (sous-réseau d'oxygène complet)")
    else:
        out.append("  maille : pas de prototype cubique pour cette famille")
    th = r["thermo"]
    if th and th["covered"]:
        how = ("calcul direct" if th["direct"]
               else f"modèle additif, ± {fr(th['uncertainty_eV_per_O'], 2)}")
        out.append(f"  oxydation rocksalt → spinelle : {fr(th['dh_ox_eV_per_O'], 2)} eV/O ({how}) ; "
                   f"dans l'air : {VERDICT_FR[th['verdict']]} "
                   f"(rocksalt {fr(100 * th['p_rocksalt'], 0)} %)")
    elif th:
        out.append("  thermodynamique calculée : cation(s) non couvert(s) : "
                   + ", ".join(th["missing"]))
    if peaks and r["pattern"]:
        out.append("  réflexions (Cu Kα1) :  2θ (°)   I rel.   hkl")
        out += [f"    {p['two_theta']:9.3f} {p['intensity']:8.1f}   {p['hkl']}"
                for p in r["pattern"]]
    out += ["  ! " + w for w in r["warnings"]]
    return "\n".join(out)


def write_xy(predictor, r, path):
    """Simulated profile as a two-column text file, the format diffraction software reads."""
    if not r["pattern"]:
        raise SystemExit("pas de diffractogramme simulé : cette famille n'a pas de prototype cubique")
    lat = r["lattice"]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# {r['formula'] or '-'.join(r['fractions'])} — {r['family']} {lat['space_group']}, "
                 f"a = {lat['a']:.4f} A ({lat['source']})\n"
                 f"# profil simule, Cu K-alpha1 {predictor.xrd['wavelength']} A, pseudo-Voigt "
                 f"de largeur {predictor_profile()['fwhm']} deg ; colonnes : 2theta (deg), intensite (max 100)\n")
        for x, y in predictor.profile(r["pattern"]):
            fh.write(f"{x:.2f} {y:.3f}\n")
    print(f"diffractogramme simulé -> {path}", file=sys.stderr)


def predictor_profile():
    from predict import PROFILE
    return PROFILE


def read_table(path, column=None):
    """Rows of the input file as dictionaries, and the name of the formula column."""
    with open(path, encoding="utf-8-sig", newline="") as fh:
        text = fh.read()
    lines = [ln for ln in csv.reader(text.splitlines()) if ln and any(c.strip() for c in ln)]
    if not lines:
        raise SystemExit(f"{path} : fichier vide")
    header = [h.strip() for h in lines[0]]
    if column is None:
        column = next((c for c in FORMULA_COLUMNS if c in header), None)
    if column is None:
        # plain list, no header: the whole line is the formula — its commas belong to it,
        # as in (Mg,Co,Ni,Cu,Zn)O
        return [{"formula_text": ln.strip().strip('"')} for ln in text.splitlines()
                if ln.strip()], "formula_text"
    if column not in header:
        raise SystemExit(f"{path} : pas de colonne « {column} » (colonnes : {', '.join(header)})")
    return [dict(zip(header, ln)) for ln in lines[1:]], column


def batch(predictor, path, out_path, column=None):
    rows, column = read_table(path, column)
    kept = [c for c in rows[0] if c not in OUT_COLUMNS]
    n_ok = 0
    out = open(out_path, "w", encoding="utf-8", newline="") if out_path else sys.stdout
    try:
        w = csv.DictWriter(out, fieldnames=kept + OUT_COLUMNS, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for src in rows:
            rec = {c: src.get(c, "") for c in kept}
            rec.update(predictor.record(src.get(column, "")))
            n_ok += not rec["error"]
            w.writerow(rec)
    finally:
        if out_path:
            out.close()
    print(f"{n_ok} prédiction(s) sur {len(rows)} ligne(s)"
          + (f" -> {out_path}" if out_path else ""), file=sys.stderr)
    return 0 if n_ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("formulas", nargs="*", help="formule(s) écrite(s) comme dans un article")
    ap.add_argument("--csv", metavar="FICHIER", help="fichier de formules à traiter en lot")
    ap.add_argument("--column", help="nom de la colonne des formules dans ce fichier")
    ap.add_argument("--out", metavar="FICHIER", help="fichier CSV de sortie (défaut : écran)")
    ap.add_argument("--json", action="store_true", help="sortie JSON complète")
    ap.add_argument("--peaks", action="store_true", help="liste les réflexions simulées")
    ap.add_argument("--xy", metavar="FICHIER",
                    help="écrit le diffractogramme simulé (2θ, intensité) d'une formule")
    args = ap.parse_args(argv)
    if not args.formulas and not args.csv:
        ap.error("donnez une formule ou --csv FICHIER")
    predictor = Predictor()
    if args.csv:
        return batch(predictor, args.csv, args.out, args.column)
    if args.xy and len(args.formulas) != 1:
        ap.error("--xy s'emploie avec une seule formule")
    status, results = 0, []
    for text in args.formulas:
        try:
            r = predictor.predict(text)
            results.append(r)
            if args.xy:
                write_xy(predictor, r, args.xy)
            if not args.json:
                print(report(r, peaks=args.peaks))
        except PredictionError as exc:
            status = 1
            results.append({"formula": text, "error": str(exc)})
            if not args.json:
                print(f"{text}\n  ! pas de prédiction : {exc}")
    if args.json:
        print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False,
                         indent=1))
    return status


if __name__ == "__main__":
    sys.exit(main())
