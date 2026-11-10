#!/usr/bin/env python3
"""Validate the indexing and lattice refinement of a measured pattern (src/xrdfit.py).

No raw diffraction file exists in the repository, so the refinement cannot yet be judged
on a real measurement. It is judged on what can be controlled: laboratory-like patterns
of *known* lattice parameter, built with K-alpha1 and K-alpha2 lines, a pseudo-Voigt
width, a zero shift, a decaying background and counting noise from a fixed-seed
generator (xrdfit.synthetic_pattern). The criteria were written before any case was run.

Two sets of cases, neither used while the algorithm was first written:
- set 1 was run once and missed two criteria narrowly (accuracy 0.051 % for 0.05 %; two
  genuine reflections reported as unindexed). Three causes were found and corrected: the
  K-alpha2 line pulled every peak top towards high angles (it is now detected and
  removed, Rachinger correction); reflections within 0.3 degree of the ends of the scan,
  or just under the intensity threshold of the fit, were not credited; a bump of the
  noise on the tail of a strong peak passed for a peak (a prominence test now rejects it).
  Set 1 therefore no longer counts as untouched.
- set 2 was written after these corrections and run without any further change. It adds
  what set 1 lacked: data without K-alpha2 (monochromator), a coarse step, cobalt
  radiation. It is the one that tells what to expect. One correction came later, from a
  different file (a noise bump on the flank of a strong peak, at the detection threshold:
  the prominence test now scales with the local counting noise); set 2 was run again
  and every figure of its table came out identical.

1. Clean patterns (5 000 counts at the strongest line): the refined lattice parameter is
   within 0.05 % of the true one, for every case.
2. Noisy patterns (at most 800 counts over a background of at least 400): within 0.25 %.
3. The true prototype comes first of the five cubic prototypes, for every clean case.
4. Single-phase clean patterns: no peak reported as unindexed (no false alarm).
   Patterns with lines of a foreign phase, at least 5 % of the strongest line and clear
   of the host reflections: every foreign line reported as unindexed.
5. Starting value 5 % away from the truth, both sides: same accuracy as criterion 1.

    python tools/validate_xrd_fit.py            # rewrite the report and its JSON summary
    python tools/validate_xrd_fit.py --check    # exit 1 if the report is not up to date
                                                # or a criterion is not met
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import predict  # noqa: E402
import xrdfit  # noqa: E402

REPORT = os.path.join(ROOT, "QC", "xrd_fit_validation.txt")
SUMMARY = os.path.join(ROOT, "data", "reference", "xrd_fit_validation.json")
TOL_CLEAN_PCT, TOL_NOISY_PCT = 0.05, 0.25
NOISY = {"counts": 800.0, "background": 400.0, "fwhm": 0.3, "step": 0.04}


def eq(*els):
    return {e: 1 / len(els) for e in sorted(els)}


def ab(a_site, b_site):
    """Equimolar A site and equimolar B site, half of the cations each."""
    d = {e: 0.5 / len(a_site) for e in a_site}
    d.update({e: 0.5 / len(b_site) for e in b_site})
    return {e: d[e] for e in sorted(d)}


# name, prototype, cation fractions, true a (angstrom), pattern options
CASES = [
    ("(CoCuFeMnNi)3O4", "spinel", eq("Co", "Cu", "Fe", "Mn", "Ni"), 8.3602,
     {"zero": -0.05, "fwhm": 0.12, "seed": 7}),
    ("(AlCoCrFeMgNi)3O4", "spinel", eq("Al", "Co", "Cr", "Fe", "Mg", "Ni"), 8.19,
     {"fwhm": 0.20, "seed": 11}),
    ("(CoCuLiMgNiZn)O", "rocksalt", eq("Co", "Cu", "Li", "Mg", "Ni", "Zn"), 4.21,
     {"zero": 0.08, "seed": 3}),
    ("(CoMgMnNiZn)O", "rocksalt", eq("Co", "Mg", "Mn", "Ni", "Zn"), 4.27, {"seed": 5}),
    ("(CeLaPrSmY)O2", "fluorite", eq("Ce", "La", "Pr", "Sm", "Y"), 5.48,
     {"zero": -0.03, "fwhm": 0.22, "seed": 13}),
    ("(CeGdHfYZr)O2", "fluorite", eq("Ce", "Gd", "Hf", "Y", "Zr"), 5.29, {"seed": 17}),
    ("(GdLaNdSmY)CoO3", "perovskite", ab(["Gd", "La", "Nd", "Sm", "Y"], ["Co"]), 3.80,
     {"fwhm": 0.10, "seed": 19}),
    ("Ba(HfNbSnTiZr)O3", "perovskite", ab(["Ba"], ["Hf", "Nb", "Sn", "Ti", "Zr"]), 4.12,
     {"zero": 0.04, "seed": 23}),
    ("(DyErGdHoY)2Ti2O7", "pyrochlore", ab(["Dy", "Er", "Gd", "Ho", "Y"], ["Ti"]), 10.10,
     {"fwhm": 0.18, "seed": 29}),
    ("(CeEuLaNdSm)2Sn2O7", "pyrochlore", ab(["Ce", "Eu", "La", "Nd", "Sm"], ["Sn"]), 10.60,
     {"zero": -0.06, "seed": 31}),
]
# foreign lines: hematite (104) and (024) in a spinel, tenorite (11-1) and (111) in a rocksalt
FOREIGN = [
    ("(AlCoCrFeMgNi)3O4 + Fe2O3", "spinel", eq("Al", "Co", "Cr", "Fe", "Mg", "Ni"), 8.19,
     {"seed": 37, "extra": [(33.15, 10.0), (49.48, 6.0)]}),
    ("(CoMgMnNiZn)O + CuO", "rocksalt", eq("Co", "Mg", "Mn", "Ni", "Zn"), 4.27,
     {"seed": 41, "extra": [(35.54, 8.0), (38.71, 8.0)]}),
]


# ---- set 2: written after the corrections, run once ----
CASES2 = [
    ("(CoCrFeMnZn)3O4", "spinel", eq("Co", "Cr", "Fe", "Mn", "Zn"), 8.37,
     {"zero": 0.03, "fwhm": 0.16, "seed": 43}),
    ("(CoFeGaMgNi)3O4 mono.", "spinel", eq("Co", "Fe", "Ga", "Mg", "Ni"), 8.31,
     {"kalpha2": False, "seed": 47}),
    ("(CoCuMgNiZn)O mono.", "rocksalt", eq("Co", "Cu", "Mg", "Ni", "Zn"), 4.24,
     {"kalpha2": False, "zero": -0.04, "seed": 53}),
    ("(CeLaNdPrY)O2 pas 0,05", "fluorite", eq("Ce", "La", "Nd", "Pr", "Y"), 5.50,
     {"step": 0.05, "fwhm": 0.28, "seed": 59}),
    ("(BaCaLaNaSr)TiO3 Co", "perovskite", ab(["Ba", "Ca", "La", "Na", "Sr"], ["Ti"]), 3.91,
     {"radiation": "Co", "tt_min": 20.0, "tt_max": 90.0, "seed": 61}),
    ("(GdLaNdSmY)2Zr2O7", "pyrochlore", ab(["Gd", "La", "Nd", "Sm", "Y"], ["Zr"]), 10.55,
     {"zero": 0.07, "seed": 67}),
]
FOREIGN2 = [
    ("(CeLaNdPrY)O2 + raies etrangeres", "fluorite", eq("Ce", "La", "Nd", "Pr", "Y"), 5.50,
     {"seed": 71, "extra": [(30.0, 10.0), (39.5, 7.0)]}),
    ("(GdLaNdSmY)2Zr2O7 + raies etr.", "pyrochlore", ab(["Gd", "La", "Nd", "Sm", "Y"], ["Zr"]),
     10.55, {"seed": 73, "extra": [(31.6, 9.0), (45.3, 6.0)]}),
]


def run(P, proto, frac, a_true, kw):
    tt, counts = xrdfit.synthetic_pattern(P, proto, frac, a_true, **kw)
    found, fits = xrdfit.compare_measured(P, tt, counts, frac, kw.get("radiation", "Cu"))
    mine = next(f for f in fits if f["proto"] == proto)
    return tt, counts, found, fits, mine


def evaluate(P, L, label, cases, foreign, covered):
    """Run the five criteria on one set of cases; returns {criterion: met}. ``covered``
    accumulates [cases whose quoted uncertainty contains the true error, cases, signal-to-
    noise ratios of the clean patterns, of the noisy ones]."""
    ok = {}

    def table(title, tol, noisy=False):
        L.append(title)
        L.append("   compose                 prototype    a vrai   a depart   a affine    ecart   +/-affiche  "
                 "zero vrai/trouve  reflexions  1er prototype  non indexes  K-alpha2 retiree")
        errs, firsts, alarms = [], [], []
        for name, proto, frac, a_true, kw in cases:
            kw = {**kw, **(NOISY if noisy else {})}
            _tt, _c, found, fits, f = run(P, proto, frac, a_true, kw)
            err = 100 * (f["a"] - a_true) / a_true if f["a"] else float("nan")
            errs.append(abs(err))
            firsts.append(fits[0]["proto"] == proto)
            alarms.append(len(f.get("unindexed") or []))
            covered[0] += abs(f["a"] - a_true) <= f["a_uncertainty"]
            covered[1] += 1
            covered[3 if noisy else 2].append(found["snr"])
            L.append(f"   {name:<23s} {proto:<11s} {a_true:7.4f}  {f['a0']:8.4f}  {f['a']:9.4f}  "
                     f"{err:+6.3f} %  {f['a_uncertainty']:8.4f}   {kw.get('zero', 0):+5.2f} / {f['zero']:+5.2f}"
                     f"     {f['n']:2d} / {f['n_pred']:<2d}     {fits[0]['proto']:<11s}      {alarms[-1]}"
                     f"        {'oui' if found['stripped'] else 'non'}"
                     f"{'' if found['stripped'] == kw.get('kalpha2', True) else '  (a tort)'}")
        L.append(f"   ecart maximal {max(errs):.3f} % (critere {tol} %) -> "
                 f"{'atteint' if max(errs) <= tol else 'NON ATTEINT'}")
        return errs, firsts, alarms

    L.append(f"===== {label} =====")
    errs, firsts, alarms = table("1. Motifs propres (5 000 coups au pic le plus fort)", TOL_CLEAN_PCT)
    worst_clean = max(errs)
    ok["1 exactitude, motifs propres"] = max(errs) <= TOL_CLEAN_PCT
    ok["3 prototype vrai en tete"] = all(firsts)
    ok["4a aucune fausse alerte"] = sum(alarms) == 0
    L += [f"   3. prototype vrai classe premier : {sum(firsts)} / {len(firsts)} -> "
          f"{'atteint' if all(firsts) else 'NON ATTEINT'}",
          f"   4. pics signales a tort comme non indexes : {sum(alarms)} -> "
          f"{'atteint' if sum(alarms) == 0 else 'NON ATTEINT'}", ""]

    errs, firsts_n, _ = table("2. Motifs bruites (800 coups sur un fond de 400, largeur 0,3 deg, pas 0,04 deg)",
                              TOL_NOISY_PCT, noisy=True)
    worst_noisy = max(errs)
    ok["2 exactitude, motifs bruites"] = max(errs) <= TOL_NOISY_PCT
    L += [f"   prototype vrai classe premier : {sum(firsts_n)} / {len(firsts_n)} (hors critere : sous le "
          "bruit, les raies de surstructure d'un pyrochlore peuvent disparaitre, et la fluorine l'explique alors aussi bien)",
          ""]

    L.append("4. Phase etrangere : raies ajoutees au motif d'une phase unique")
    hits = total = 0
    for name, proto, frac, a_true, kw in foreign:
        _tt, _c, _found, _fits, f = run(P, proto, frac, a_true, kw)
        seen = [round(u["tt"], 2) for u in f["unindexed"]]
        got = sum(1 for t, _i in kw["extra"] if any(abs(t - s) <= 0.1 for s in seen))
        hits += got
        total += len(kw["extra"])
        L.append(f"   {name:<32s} raies ajoutees a {', '.join(f'{t:.2f}' for t, _ in kw['extra'])} deg -> "
                 f"signalees non indexees : {', '.join(f'{s:.2f}' for s in seen) or 'aucune'} ; "
                 f"a affine {f['a']:.4f} (vrai {a_true})")
    ok["4b raies etrangeres signalees"] = hits == total
    L += [f"   raies etrangeres retrouvees : {hits} / {total} -> "
          f"{'atteint' if hits == total else 'NON ATTEINT'}", ""]

    L.append("5. Valeur de depart a 5 % de la verite")
    worst = 0.0
    for name, proto, frac, a_true, kw in cases[::3]:
        tt, counts = xrdfit.synthetic_pattern(P, proto, frac, a_true, **kw)
        found = xrdfit.find_peaks(tt, counts, kw.get("radiation", "Cu"))
        for k in (0.95, 1.05):
            f = xrdfit.refine_lattice(P, found, proto, frac, a_true * k, tt[0], tt[-1],
                                      kw.get("radiation", "Cu"))
            err = 100 * (f["a"] - a_true) / a_true if f and f["a"] else float("nan")
            worst = max(worst, abs(err)) if err == err else float("inf")
            L.append(f"   {name:<23s} depart {a_true * k:8.4f} ({100 * (k - 1):+.0f} %) -> "
                     f"{f['a']:9.4f}  ecart {err:+6.3f} %")
    ok["5 depart eloigne"] = worst <= TOL_CLEAN_PCT
    L += [f"   ecart maximal {worst:.3f} % (critere {TOL_CLEAN_PCT} %) -> "
          f"{'atteint' if worst <= TOL_CLEAN_PCT else 'NON ATTEINT'}", "",
          f"Bilan {label} : " + " ; ".join(f"{k} : {'atteint' if v else 'NON ATTEINT'}"
                                             for k, v in ok.items()),
          ""]
    return ok, {"n": len(cases), "clean_max_pct": round(worst_clean, 3),
                "noisy_max_pct": round(worst_noisy, 3), "foreign_found": hits,
                "foreign_total": total, "false_alarms": sum(alarms),
                "true_prototype_first": sum(firsts), "met": all(ok.values())}


def build():
    P = predict.Predictor()
    L = ["Diffractogramme mesure : indexation sur un prototype cubique et affinement de la maille",
         "motifs synthetiques de maille connue (K-alpha1 + K-alpha2, largeur, decalage de zero, fond, bruit)",
         "criteres ecrits avant toute execution",
         "",
         "Jeu 1 : execute une premiere fois, il a manque deux criteres de peu (exactitude 0,051 % pour 0,05 % ;",
         "deux reflexions reelles signalees non indexees). Trois causes corrigees : la raie K-alpha2 tirait",
         "chaque sommet vers les grands angles (elle est maintenant detectee et retiree) ; les reflexions",
         "proches des bords du balayage ou juste sous le seuil d'intensite n'etaient pas creditees ; une bosse",
         "du bruit sur le pied d'un pic fort passait pour un pic. Ce jeu ne vaut donc plus comme jeu vierge.",
         "Jeu 2 : ecrit apres ces corrections, execute sans aucune retouche. C'est lui qui dit a quoi s'attendre.",
         "Une derniere correction est venue ensuite d'un autre fichier (bosse du bruit sur le flanc d'un pic fort,",
         "au seuil de detection) ; le jeu 2, reexecute, a rendu exactement les memes chiffres.",
         ""]
    covered = [0, 0, [], []]
    ok1, sum1 = evaluate(P, L, "jeu 1 (a servi aux corrections)", CASES, FOREIGN, covered)
    ok2, sum2 = evaluate(P, L, "jeu 2 (vierge)", CASES2, FOREIGN2, covered)
    L += ["Lecture : l'incertitude affichee est la plus grande de l'erreur statistique de l'ajustement et de",
          f"l'exactitude etablie ici : {100 * xrdfit.ACCURACY_CLEAN:.2f} % de a quand le rapport signal sur bruit du pic "
          f"le plus fort atteint {xrdfit.SNR_CLEAN},",
          f"{100 * xrdfit.ACCURACY_NOISY:.2f} % en dessous. Les motifs propres ci-dessus ont un rapport de "
          f"{min(covered[2]):.0f} a {max(covered[2]):.0f}, les motifs bruites de {min(covered[3]):.0f} a "
          f"{max(covered[3]):.0f} ;",
          f"sous {xrdfit.SNR_MIN}, les donnees sont plus bruitees que tout ce qui a ete verifie et la page le signale.",
          f"Incertitude affichee >= ecart reel : {covered[0]} cas sur {covered[1]} (les deux jeux, propres et bruites).",
          "Limites : positions des pics seulement (pas un affinement de Rietveld), prototypes cubiques,",
          "une phase a la fois ; une distorsion quadratique ou orthorhombique dedouble des pics que ce",
          "modele signale alors comme non indexes. Les motifs synthetiques ont des pics symetriques et un",
          "rapport K-alpha2 / K-alpha1 de 0,5 exactement : rien ici ne remplace l'essai sur un diffractogramme reel.",
          "",
          "genere par tools/validate_xrd_fit.py"]
    summary = {"criteria_pct": {"clean": TOL_CLEAN_PCT, "noisy": TOL_NOISY_PCT},
               "set1_used_for_corrections": sum1, "set2_untouched": sum2,
               "uncertainty_covers_error": covered[:2]}
    return ("\n".join(L) + "\n", json.dumps(summary, indent=1) + "\n",
            all(ok1.values()) and all(ok2.values()))


def main(check=False):
    report, summary, ok = build()
    if check:
        stale = [os.path.relpath(p, ROOT) for p, text in ((REPORT, report), (SUMMARY, summary))
                 if not os.path.exists(p) or open(p, encoding="utf-8").read() != text]
        if stale:
            print(", ".join(stale) + " not up to date: run tools/validate_xrd_fit.py")
            return 1
        print("QC/xrd_fit_validation.txt is up to date"
              + ("" if ok else " — but a criterion is NOT met"))
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
