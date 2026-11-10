#!/usr/bin/env python3
"""Validate the lattice model (src/lattice.py) on measured lattice parameters.

The protocol (data/reference/lattice_protocol.md) was written before the model was
computed: data, split, rules and criteria. This script applies it to
data/reference/lattice_measured.csv and writes

- QC/lattice_validation.txt — criteria C1 (references, a gate), C2 (high-entropy oxides,
  per family), C3 (better than the v1.0 model), the choice between CHGNet and the
  geometric model for spinels, and the error of every high-entropy oxide;
- data/reference/lattice_validation.json — the same numbers; the platform shows the
  error per family found here as the uncertainty of its predicted lattice parameter.

The fluorite model was amended after the first reading of the residuals (amendment A1,
oxygen vacancies): the report gives the result of the model as first written next to
the amended one, and declares the fluorite validation non-blind.

    python tools/validate_lattice.py            # rewrite the report and the summary
    python tools/validate_lattice.py --check    # exit 1 if they are not up to date or if
                                                # the reference gate C1 is not met
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import features  # noqa: E402
import lattice  # noqa: E402
import lattice_dataset as ld  # noqa: E402
import predict  # noqa: E402

REPORT = os.path.join(ROOT, "QC", "lattice_validation.txt")
SUMMARY = os.path.join(ROOT, "data", "reference", "lattice_validation.json")
C_REF = 1.0       # C1: mean absolute error on the references, per family (%)
C_HEO = 1.0       # C2: mean absolute error on the high-entropy oxides, per family (%)


def err_pct(pred, meas):
    return (pred - meas) / meas * 100


def stats(errs):
    if not errs:
        return None
    n = len(errs)
    return {"n": n, "mae_pct": features.nsum([abs(e) for e in errs]) / n,
            "bias_pct": features.nsum(errs) / n, "max_pct": max(abs(e) for e in errs)}


def fmt(s):
    return (f"{s['n']:>3d}  {s['mae_pct']:5.2f} %  {s['bias_pct']:+6.2f} %  {s['max_pct']:5.2f} %"
            if s else "  -")


def build():
    P = predict.Predictor()
    E = P.elements
    rows = ld.load()
    comps = list(ld.unique_compositions(rows, P).values())
    floor = json.load(open(ld.SUMMARY, encoding="utf-8"))["floor"]

    res = []          # one line per unique composition
    for c in comps:
        r = lattice.predict(c["proto"], c["frac"], E)
        blind = lattice.predict(c["proto"], c["frac"], E, vacancy_model=False)["a"]
        v1 = ld.v1_geometric(c["proto"], c["frac"], E)
        th = P.thermo_lattice(c["proto"], c["frac"])
        res.append({**c, "pred": r["a"], "blind": blind, "v1": v1, "chgnet": th, "notes": r["notes"],
                    "e": err_pct(r["a"], c["a"]), "e_blind": err_pct(blind, c["a"]),
                    "e_v1": err_pct(v1, c["a"]),
                    "e_chgnet": err_pct(th, c["a"]) if th is not None else None})

    def pick(group, fam, key):
        return [x[key] for x in res if x["group"] == group and x["proto"] == fam]

    fam = {}
    for f in ld.FAMILIES:
        fam[f] = {"reference": stats(pick("reference", f, "e")),
                  "reference_blind": stats(pick("reference", f, "e_blind")),
                  "heo": stats(pick("heo", f, "e")),
                  "heo_blind": stats(pick("heo", f, "e_blind")),
                  "heo_v1": stats(pick("heo", f, "e_v1")),
                  "floor_pct": floor[f]["mad_pct"]}
        fam[f]["C1"] = fam[f]["reference"]["mae_pct"] <= C_REF
        fam[f]["C2"] = fam[f]["heo"]["mae_pct"] <= C_HEO
        fam[f]["C3"] = fam[f]["heo"]["mae_pct"] <= fam[f]["heo_v1"]["mae_pct"]
        fam[f]["uncertainty_pct"] = round(fam[f]["heo"]["mae_pct"], 2)
        fam[f]["blind"] = f != "fluorite"
    heo_all = stats([x["e"] for x in res if x["group"] == "heo"])
    heo_blind = stats([x["e_blind"] for x in res if x["group"] == "heo"])
    heo_v1 = stats([x["e_v1"] for x in res if x["group"] == "heo"])
    c1 = all(fam[f]["C1"] for f in ld.FAMILIES)
    c3 = heo_all["mae_pct"] < heo_v1["mae_pct"] and all(fam[f]["C3"] for f in ld.FAMILIES)

    cov = [x for x in res if x["group"] == "heo" and x["proto"] == "spinel" and x["chgnet"] is not None]
    sel = {"n": len(cov), "geometric": stats([x["e"] for x in cov]),
           "chgnet": stats([x["e_chgnet"] for x in cov])}
    sel["retained"] = ("chgnet" if sel["chgnet"]["mae_pct"] < sel["geometric"]["mae_pct"]
                       else "geometric")

    L = ["Modele de maille (src/lattice.py) : validation sur des mailles mesurees",
         "protocole ecrit avant le calcul : data/reference/lattice_protocol.md",
         "regles : data/reference/lattice_rules.json ; donnees : data/reference/lattice_measured.csv",
         "unite : composition unique (mediane de ses mesures), phase pure declaree par la source",
         "ecart = (predit - mesure) / mesure ; colonnes : n, ecart absolu moyen, biais moyen, ecart maximal",
         "",
         f"C1 (porte). References : ecart absolu moyen <= {C_REF:.1f} % dans chaque famille",
         "",
         "   famille       modele final                       protocole initial (sans A1)"]
    for f in ld.FAMILIES:
        L.append(f"   {f:<12s} {fmt(fam[f]['reference'])}   "
                 f"{'atteint' if fam[f]['C1'] else 'NON ATTEINT':<11s} {fmt(fam[f]['reference_blind'])}")
    L += ["", f"   -> C1 {'atteint' if c1 else 'NON ATTEINT'}", "",
          f"C2. Oxydes a haute entropie : ecart absolu moyen <= {C_HEO:.1f} % dans chaque famille",
          "",
          "   famille       modele final                       aveugle  protocole initial (sans A1)        plancher"]
    for f in ld.FAMILIES:
        fl = fam[f]["floor_pct"]
        L.append(f"   {f:<12s} {fmt(fam[f]['heo'])}   {'atteint' if fam[f]['C2'] else 'NON ATTEINT':<11s}"
                 f" {'oui' if fam[f]['blind'] else 'non':<7s}  {fmt(fam[f]['heo_blind'])}   "
                 + (f"{fl:.2f} %" if fl is not None else "-"))
    L += ["",
          "   -> C2 atteint pour : " + ", ".join(f for f in ld.FAMILIES if fam[f]["C2"])
          + " ; non atteint pour : " + (", ".join(f for f in ld.FAMILIES if not fam[f]["C2"]) or "aucune"),
          "",
          "C3. Meilleur que le modele geometrique de la v1.0, globalement et dans chaque famille",
          "",
          "   famille       v1.0                               modele final"]
    for f in ld.FAMILIES:
        L.append(f"   {f:<12s} {fmt(fam[f]['heo_v1'])}   {fmt(fam[f]['heo'])}   "
                 f"{'mieux' if fam[f]['C3'] else 'MOINS BIEN'}")
    L += [f"   {'toutes':<12s} {fmt(heo_v1)}   {fmt(heo_all)}",
          f"   (protocole initial, sans A1 : {fmt(heo_blind)})",
          "",
          f"   -> C3 {'atteint' if c3 else 'NON ATTEINT'}",
          "",
          "Spinelles : CHGNet ou modele geometrique (selection de modele declaree, protocole section 5)",
          f"   {sel['n']} spinelles de validation dont tous les cations sont couverts par CHGNet",
          f"   geometrique {fmt(sel['geometric'])}",
          f"   CHGNet      {fmt(sel['chgnet'])}",
          f"   -> retenu pour ces spinelles : {sel['retained']} ; geometrique pour les autres",
          "",
          "Amendement A1 (fluorine). Ecrit apres la premiere lecture des residus : la relation de",
          "contact surestimait toutes les fluorines lacunaires, references comprises (CeO2 +0,25 %, mais",
          "La0.5Ce0.5O1.75 +1,32 %, Ce-La-Pr +1,59 %). Les lacunes d'oxygene sont traitees en spheres",
          "dures de rayon r_V donne par l'hote (Chatzichristodoulou et al. 2015, tableau 1, aucune",
          "constante ajustee ici). La validation de la fluorine n'est donc plus aveugle ; le resultat du",
          "protocole initial est donne a cote.",
          "",
          "Detail, oxydes a haute entropie",
          "",
          "   famille      composition                                    mesure   predit   ecart    v1.0     notes"]
    for x in res:
        if x["group"] != "heo":
            continue
        keep = [n for n in x["notes"] if not n.startswith(("rayon ", "lacunes d'oxyg"))]
        est = [n.split(" ")[1] for n in x["notes"] if n.startswith("rayon ")]
        if est:
            keep.append("rayon estime : " + ", ".join(est))
        notes = "; ".join(k.split(" : ")[0] for k in keep)
        L.append(f"   {x['proto']:<11s}  {x['formula'][:45]:<45s} {x['a']:7.4f}  {x['pred']:7.4f}  "
                 f"{x['e']:+5.2f} %  {x['e_v1']:+5.2f} %  {notes[:110]}")
    big = [x for x in res if x["group"] == "heo" and x["proto"] == "rocksalt"
           and any(n.startswith("cation volumineux") for n in x["notes"])]
    rest = stats([x["e"] for x in res if x["group"] == "heo" and x["proto"] == "rocksalt" and x not in big])
    L += ["",
          "Lecture",
          f"   Rocksalt : l'echec de C2 vient de {len(big)} composition(s) a cation volumineux (Sr2+, K+, Na+) dont",
          "   la maille mesuree est celle de la matrice sans ce cation ; la relation de contact, comme la loi",
          "   de Vegard, predit une dilatation que la mesure ne montre pas, ce qui suggere une incorporation",
          "   partielle. Sans elles (information, pas critere) : " + fmt(rest) + ".",
          "   La plateforme le signale pour tout rocksalt contenant un cation de rayon VI > 1,0 A.",
          "   Fluorine : les compositions a Mn sont sous-estimees (-0,6 %) : la valence de Mn y est fixee",
          "   a celle de Mn3O4, non mesuree.",
          "   Spinelle : le plus grand ecart, (MnFeCoNiZr)3O4 synthetise a 500 C, suppose Zr4+ dans le",
          "   reseau ; une partie peut etre restee en ZrO2 amorphe.",
          "",
          "genere par tools/validate_lattice.py"]

    def rnd(v):
        if isinstance(v, float):
            return round(v, 6)      # six decimals, so that quoting two never rounds twice
        if isinstance(v, dict):
            return {k: rnd(w) for k, w in v.items()}
        return v
    summary = {"criteria_pct": {"C1": C_REF, "C2": C_HEO},
               "C1_met": c1, "C2_met": [f for f in ld.FAMILIES if fam[f]["C2"]], "C3_met": c3,
               "families": rnd(fam), "heo": rnd(heo_all), "heo_protocol_initial": rnd(heo_blind),
               "heo_v1": rnd(heo_v1), "spinel_selection": rnd(sel)}
    return "\n".join(L) + "\n", json.dumps(summary, indent=1) + "\n", c1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    report, summary, ok = build()
    if args.check:
        stale = [p for p, txt in ((REPORT, report), (SUMMARY, summary))
                 if not os.path.exists(p) or open(p, encoding="utf-8").read() != txt]
        if stale:
            print("pas a jour : " + ", ".join(os.path.relpath(p, ROOT) for p in stale)
                  + " (python tools/validate_lattice.py)")
        if not ok:
            print("critere C1 (references) non atteint")
        sys.exit(1 if stale or not ok else 0)
    for p, txt in ((REPORT, report), (SUMMARY, summary)):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
