#!/usr/bin/env python3
"""Step 4 of the lattice protocol: does a learned correction beat the geometric model?

A regressor of the residual of the geometric lattice model (src/lattice.py) is trained on
the measured lattice parameters of data/reference/lattice_measured.csv and judged out of
sample, one group of sources left out at a time, exactly as written in
data/reference/lattice_protocol.md before the first run: target, groups, descriptors,
candidates and their hyperparameters, nested choice of the candidate, and the criteria
for integrating the correction into the platform.

Writes QC/lattice_correction.txt and data/reference/lattice_correction.json. The verdict
is computed, not typed: "integrated" is true only if both criteria are met.

    python tools/learn_lattice_correction.py            # rewrite the report
    python tools/learn_lattice_correction.py --check    # exit 1 if it is not up to date

Requires numpy and scikit-learn (requirements-train.txt).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import lattice  # noqa: E402
import lattice_dataset as ld  # noqa: E402
import predict  # noqa: E402

REPORT = os.path.join(ROOT, "QC", "lattice_correction.txt")
SUMMARY = os.path.join(ROOT, "data", "reference", "lattice_correction.json")
VARIABLE_VALENCE = {"Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ce", "Pr", "Tb"}
NUMERIC = ["charge moyenne", "desaccord de taille", "lacunes par cation", "part de rayons estimes",
           "part de cations a valence variable", "entropie de configuration"]
CANDIDATES = ["M0 aucune correction", "M1 biais par famille", "M2 ridge", "M3 gradient boosting"]
MIN_GAIN_POINT = 0.1      # criteria of the protocol, section 5
MIN_GAIN_REL = 0.15
MAX_FAMILY_LOSS = 0.1


def descriptors(proto, frac, model):
    """The six numeric descriptors of the protocol, from the geometric model output."""
    sites = {}
    for u in model["radii"]:
        sites.setdefault(u["site"], []).append(u)
    mult = {"T": 1, "O": 2}                       # cations per formula unit on each site
    q_num = q_den = 0.0
    for s, _el, ox, x in model["ions"]:
        w = mult.get(s, 1)
        q_num += ox * x * w
        q_den += x * w
    mism = []
    for s, us in sites.items():
        tot = sum(u["amount"] for u in us)
        mean = sum(u["r"] * u["amount"] for u in us) / tot
        var = sum(u["amount"] * (u["r"] - mean) ** 2 for u in us) / tot
        mism.append((math.sqrt(var) / mean, mult.get(s, 1)))
    used = sum(u["amount"] for u in model["radii"])
    est = sum(u["amount"] for u in model["estimated"])
    return [q_num / q_den,
            sum(m * w for m, w in mism) / sum(w for _, w in mism),
            model["vacancies"]["per_cation"] if model["vacancies"] else 0.0,
            est / used,
            sum(x for e, x in frac.items() if e in VARIABLE_VALENCE),
            -sum(x * math.log(x) for x in frac.values() if x > 0)]


def groups_of(comps):
    """Connected components: two compositions measured by a same origin share a group."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for k, c in comps.items():
        for r in c["rows"]:
            parent[find("c:" + k)] = find("o:" + ld.origin(r))
    roots = {}
    return [roots.setdefault(find("c:" + k), len(roots)) for k in comps]


def dataset():
    P = predict.Predictor()
    comps = ld.unique_compositions(ld.load(), P)
    rows = []
    for (key, c), g in zip(comps.items(), groups_of(comps)):
        m = lattice.predict(c["proto"], c["frac"], P.elements)
        rows.append({"key": key, "formula": c["formula"], "proto": c["proto"], "heo": c["group"] == "heo",
                     "group": g, "a": c["a"], "pred": m["a"],
                     "y": (c["a"] - m["a"]) / m["a"] * 100,
                     "x": descriptors(c["proto"], c["frac"], m)})
    return rows


def design(rows):
    fam = np.array([[1.0 if r["proto"] == f else 0.0 for f in ld.FAMILIES] for r in rows])
    return fam, np.array([r["x"] for r in rows]), np.array([r["y"] for r in rows])


def fit_predict(name, train, test):
    """Residual (%) estimated for the test rows by one candidate trained on train."""
    fam_tr, x_tr, y_tr = design(train)
    fam_te, x_te, _ = design(test)
    if name.startswith("M0"):
        return np.zeros(len(test))
    if name.startswith("M1"):
        out = np.zeros(len(test))
        for j in range(len(ld.FAMILIES)):
            if fam_tr[:, j].sum() > 0:
                out[fam_te[:, j] == 1] = y_tr[fam_tr[:, j] == 1].mean()
        return out
    mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
    sd[sd == 0] = 1.0
    X_tr = np.hstack([fam_tr, (x_tr - mu) / sd])
    X_te = np.hstack([fam_te, (x_te - mu) / sd])
    if name.startswith("M2"):
        model = Ridge(alpha=1.0)
    else:
        model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                          random_state=0)
    return model.fit(X_tr, y_tr).predict(X_te)


def abs_err(row, residual_hat):
    """|corrected - measured| / measured, in %."""
    return abs(row["pred"] * (1 + residual_hat / 100) - row["a"]) / row["a"] * 100


def loo_errors(name, rows):
    """{key: out-of-sample absolute error} for the high-entropy oxides of rows, one group
    left out at a time; training on everything else (references included)."""
    out = {}
    for g in sorted({r["group"] for r in rows if r["heo"]}):
        test = [r for r in rows if r["group"] == g and r["heo"]]
        train = [r for r in rows if r["group"] != g]
        for r, h in zip(test, fit_predict(name, train, test)):
            out[r["key"]] = abs_err(r, float(h))
    return out


def nested(rows):
    """Outer leave-one-group-out; in each fold the candidate is chosen by an inner
    leave-one-group-out on the training rows. Returns errors and the choices made."""
    out, chosen = {}, {}
    for g in sorted({r["group"] for r in rows if r["heo"]}):
        test = [r for r in rows if r["group"] == g and r["heo"]]
        train = [r for r in rows if r["group"] != g]
        score = {}
        for name in CANDIDATES:
            e = loo_errors(name, train)
            score[name] = sum(e.values()) / len(e)
        best = min(CANDIDATES, key=lambda n: (round(score[n], 9), CANDIDATES.index(n)))
        chosen[g] = best
        for r, h in zip(test, fit_predict(best, train, test)):
            out[r["key"]] = abs_err(r, float(h))
    return out, chosen


def mae(errors, rows, fam=None):
    v = [errors[r["key"]] for r in rows if r["heo"] and (fam is None or r["proto"] == fam)]
    return sum(v) / len(v)


def build():
    rows = dataset()
    heo = [r for r in rows if r["heo"]]
    n_groups = len({r["group"] for r in heo})
    single = {name: loo_errors(name, rows) for name in CANDIDATES}
    nest, chosen = nested(rows)
    base = single[CANDIDATES[0]]

    def table(errors):
        return {"all": mae(errors, rows), **{f: mae(errors, rows, f) for f in ld.FAMILIES}}
    t_base, t_nest = table(base), table(nest)
    gain = t_base["all"] - t_nest["all"]
    rel = gain / t_base["all"]
    worst = max(t_nest[f] - t_base[f] for f in ld.FAMILIES)
    c_gain = gain >= MIN_GAIN_POINT and rel >= MIN_GAIN_REL
    c_fam = worst <= MAX_FAMILY_LOSS
    integrated = c_gain and c_fam
    picks = {n: sum(1 for v in chosen.values() if v == n) for n in CANDIDATES}

    L = ["Correction apprise du modele de maille : bat-elle le modele geometrique hors echantillon ?",
         "protocole : data/reference/lattice_protocol.md, section 5, etape 4 (ecrit avant ce calcul)",
         f"donnees : {len(rows)} compositions uniques, dont {len(heo)} oxydes a haute entropie en {n_groups} groupes de sources",
         "cible : residu relatif du modele geometrique ; un groupe de sources retire a chaque tour",
         "ecart absolu moyen hors echantillon sur les oxydes a haute entropie, en %",
         "",
         "   " + f"{'':<34s}" + "".join(f"{c:>12s}" for c in ("toutes",) + ld.FAMILIES)]

    def line(label, t):
        return f"   {label:<34s}" + "".join(f"{t[c]:>10.2f} %" for c in ("all",) + ld.FAMILIES)
    L.append(line("M0 aucune correction (etape 3)", t_base))
    for name in CANDIDATES[1:]:
        L.append(line(name + " (information)", table(single[name])))
    L += [line("procedure jugee (choix interne)", t_nest),
          "",
          "   candidat retenu par la validation interne, sur " + str(len(chosen)) + " tours : "
          + " ; ".join(f"{n.split(' ', 1)[0]} {k}" for n, k in picks.items()),
          "",
          "Criteres",
          f"   gain : {gain:+.3f} point, {100 * rel:+.1f} % en relatif "
          f"(exige : au moins {MIN_GAIN_POINT} point et {100 * MIN_GAIN_REL:.0f} %) -> "
          + ("atteint" if c_gain else "NON ATTEINT"),
          f"   pire degradation d'une famille : {worst:+.3f} point (toleree : {MAX_FAMILY_LOSS}) -> "
          + ("atteint" if c_fam else "NON ATTEINT"),
          "",
          "Verdict : " + ("correction INTEGREE a la plateforme" if integrated
                          else "correction NON INTEGREE ; la plateforme garde le modele geometrique de l'etape 3"),
          ""]
    best_single = min(CANDIDATES[1:], key=lambda n: mae(single[n], rows))
    t_best = table(single[best_single])
    better = [f for f in ld.FAMILIES if t_nest[f] < t_base[f] - 0.005]
    worse = [f for f in ld.FAMILIES if t_nest[f] > t_base[f] + 0.005]
    L += ["Lecture",
          "   Procedure jugee : mieux pour " + (", ".join(better) or "aucune famille")
          + " ; moins bien pour " + (", ".join(worse) or "aucune famille") + ".",
          f"   Meilleur candidat pris seul : {best_single} ({t_best['all']:.2f} %, soit "
          f"{t_base['all'] - t_best['all']:+.3f} point). Il est rapporte pour information : le",
          "   retenir apres avoir vu ces chiffres serait une selection sur les donnees de test.",
          "   Ce que la correction apprend, c'est surtout un biais par famille (perovskite, pyrochlore,",
          "   rocksalt). Les modeles plus souples n'ajoutent rien de fiable : ils degradent la fluorine, que",
          "   le modele geometrique decrit deja pres du plancher.",
          f"   Les {len(heo)} oxydes a haute entropie viennent de {n_groups} groupes de sources ; perovskite et",
          "   pyrochlore reposent sur deux ou trois groupes. C'est trop peu pour apprendre davantage qu'un",
          "   biais. Plancher : pour une meme composition, deux laboratoires different de 0,02 a 0,14 % en",
          "   moyenne (QC/lattice_dataset.txt).",
          "",
          "genere par tools/learn_lattice_correction.py"]

    def rnd(t):
        return {k: round(v, 6) for k, v in t.items()}   # six decimals: quoting two never rounds twice
    summary = {"n_compositions": len(rows), "n_heo": len(heo), "n_groups": n_groups,
               "step3": rnd(t_base), "judged": rnd(t_nest),
               "candidates": {n: rnd(table(single[n])) for n in CANDIDATES[1:]},
               "chosen": picks, "gain_point": round(gain, 6), "gain_rel": round(rel, 6),
               "worst_family_loss": round(worst, 4),
               "criteria": {"gain": bool(c_gain), "family": bool(c_fam)}, "integrated": bool(integrated)}
    return "\n".join(L) + "\n", json.dumps(summary, indent=1) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    report, summary = build()
    if args.check:
        stale = [p for p, txt in ((REPORT, report), (SUMMARY, summary))
                 if not os.path.exists(p) or open(p, encoding="utf-8").read() != txt]
        if stale:
            print("pas a jour : " + ", ".join(os.path.relpath(p, ROOT) for p in stale))
        sys.exit(1 if stale else 0)
    for p, txt in ((REPORT, report), (SUMMARY, summary)):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(report)


if __name__ == "__main__":
    main()
