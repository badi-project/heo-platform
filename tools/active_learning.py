#!/usr/bin/env python3
"""Step 3 — which compositions should the lab make next?

A retrospective benchmark first, then a proposal.

Benchmark. The database is replayed as if each cation system had to be synthesised to
learn its structure family. A model starts from 20 systems; at each round it "makes"
10 more, chosen by a strategy, and is scored on held-out systems:
    random        the baseline: pick at random
    uncertainty   pick the systems whose predicted family is most uncertain (entropy)
    diversity     pick the systems farthest from everything already made
    hybrid        uncertainty weighted by distance (informative and new)
Acceptance (fixed before running): one of the informed strategies reaches the final
accuracy of random sampling (within 1 point) with at least 25 % fewer systems made,
averaged over 10 splits, and better than random on a majority of them (paired).

Proposal. With the full model, among unpublished compositions of the cations a 3d-oxide
laboratory handles (Mg Al Cr Mn Fe Co Ni Cu Zn Ga), rank by expected information:
uncertainty on the family, inside the applicability domain, and — where the CHGNet
surrogate exists — closeness to the rock-salt / spinel boundary.

    python tools/active_learning.py

Writes QC/active_learning.txt and data/next_experiments.csv.
"""
import csv
import itertools
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402

FEAT = os.path.join(ROOT, "data", "heo_features.csv")
REPORT = os.path.join(ROOT, "QC", "active_learning.txt")
NEXT = os.path.join(ROOT, "data", "next_experiments.csv")
MODEL = os.path.join(ROOT, "app", "heo_model.json")
SURR = os.path.join(ROOT, "data", "thermo", "oxidation_surrogate.json")
CLASSES = ["fluorite", "layered", "perovskite", "pyrochlore", "rocksalt", "spinel"]
START, BATCH, ROUNDS, SPLITS = 20, 10, 22, 10
LAB_CATIONS = ["Mg", "Al", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga"]


def model():
    return HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
                                          class_weight="balanced", random_state=0)


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, 1))).sum(axis=1)


def load():
    rows = [r for r in csv.DictReader(open(FEAT, encoding="utf-8")) if r["proto"] in CLASSES]
    names = [k for k in rows[0] if k not in ("doi", "sysk", "proto", "phase", "tier")]
    X = np.array([[float(r[n]) for n in names] for r in rows])
    y = np.array([r["proto"] for r in rows])
    g = np.array([r["sysk"] for r in rows])
    return X, y, g, names


def simulate(X, y, g, strategy, seed):
    rng = np.random.default_rng(seed)
    systems = np.array(sorted(set(g)))
    rng.shuffle(systems)
    n_test = len(systems) // 4
    test_sys, pool_sys = set(systems[:n_test]), list(systems[n_test:])
    te = np.array([s in test_sys for s in g])
    mu, sd = X[~te].mean(0), X[~te].std(0) + 1e-9
    Z = (X - mu) / sd
    sys_idx = {s: np.where(g == s)[0] for s in pool_sys}
    cent = {s: Z[sys_idx[s]].mean(0) for s in pool_sys}
    made = list(rng.choice(pool_sys, START, replace=False))
    left = [s for s in pool_sys if s not in made]
    curve = []
    for _ in range(ROUNDS + 1):
        tr = np.concatenate([sys_idx[s] for s in made])
        if len(set(y[tr])) < 2:
            clf = None
            acc = np.mean(y[te] == y[tr][0])
        else:
            clf = model().fit(X[tr], y[tr])
            acc = accuracy_score(y[te], clf.predict(X[te]))
        curve.append((len(made), acc))
        if not left:
            break
        k = min(BATCH, len(left))
        if strategy == "random" or clf is None:
            pick = list(rng.choice(left, k, replace=False))
        else:
            rows = np.concatenate([sys_idx[s] for s in left])
            ent = dict(zip(rows, entropy(clf.predict_proba(X[rows]))))
            unc = np.array([np.mean([ent[i] for i in sys_idx[s]]) for s in left])
            made_c = np.array([cent[s] for s in made])
            dist = np.array([np.min(np.linalg.norm(made_c - cent[s], axis=1)) for s in left])
            if strategy == "uncertainty":
                score = unc
            elif strategy == "diversity":
                score = dist
            else:
                score = unc / (unc.max() + 1e-12) + dist / (dist.max() + 1e-12)
            pick = [left[i] for i in np.argsort(-score)[:k]]
        made += pick
        left = [s for s in left if s not in pick]
    return curve


def systems_to_reach(curve, target):
    for n, a in curve:
        if a >= target:
            return n
    return None


def benchmark(X, y, g):
    strategies = ["random", "uncertainty", "diversity", "hybrid"]
    curves = {s: [simulate(X, y, g, s, seed) for seed in range(SPLITS)] for s in strategies}
    lines = ["Etape 3 - boucle active : rejouer la base comme si chaque systeme devait etre synthetise",
             f"depart {START} systemes, {BATCH} de plus par tour, test sur 25 % des systemes "
             f"tenus a l'ecart, {SPLITS} decoupages", ""]
    # random's final accuracy is the target, per split
    gains = {}
    lines.append(f"{'strategie':<13s}{'exactitude a 60 syst.':>22s}{'a 120':>8s}{'a 200':>8s}"
                 f"{'systemes pour la cible':>24s}")
    for s in strategies:
        need, at = [], {60: [], 120: [], 200: []}
        for seed in range(SPLITS):
            c = curves[s][seed]
            target = curves["random"][seed][-1][1] - 0.01
            need.append(systems_to_reach(c, target) or c[-1][0])
            for n in at:
                best = [a for m, a in c if m <= n]
                at[n].append(best[-1] if best else np.nan)
        gains[s] = np.mean(need)
        lines.append(f"{s:<13s}{np.nanmean(at[60]):>22.3f}{np.nanmean(at[120]):>8.3f}"
                     f"{np.nanmean(at[200]):>8.3f}{np.mean(need):>17.0f} ± {np.std(need):.0f}")
    base = gains["random"]
    best = min((s for s in strategies if s != "random"), key=lambda s: gains[s])
    saving = 1 - gains[best] / base
    per = {}
    for s in strategies:
        per[s] = []
        for seed in range(SPLITS):
            target = curves["random"][seed][-1][1] - 0.01
            c = curves[s][seed]
            per[s].append(systems_to_reach(c, target) or c[-1][0])
    wins = sum(a < b for a, b in zip(per[best], per["random"]))
    ties = sum(a == b for a, b in zip(per[best], per["random"]))
    # uncertainty of the saving: bootstrap over the splits (paired)
    rng = np.random.default_rng(0)
    pb, pr = np.array(per[best], float), np.array(per["random"], float)
    boot = []
    for _ in range(5000):
        i = rng.integers(0, SPLITS, SPLITS)
        boot.append(1 - pb[i].mean() / pr[i].mean())
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lines += ["",
              "cible = exactitude finale du tirage au hasard (moins 1 point), meme decoupage",
              f"economie de la meilleure strategie : IC 95 % {lo * 100:.0f} a {hi * 100:.0f} % "
              f"(bootstrap apparie sur les {SPLITS} decoupages) ; la meilleure des trois "
              f"strategies informees est choisie apres coup, ce qui flatte un peu le resultat",
              f"meilleure strategie : {best}, {saving * 100:.0f} % de syntheses en moins que le "
              f"hasard pour la meme exactitude (critere : >= 25 %) ; plus economique que le "
              f"hasard sur {wins} decoupages sur {SPLITS} ({ties} egalites) -> "
              f"{'atteint' if saving >= 0.25 and wins > SPLITS / 2 else 'non atteint'}", ""]
    return lines, best


def proposals(X, y, g, names, strategy):
    bundle = json.load(open(MODEL, encoding="utf-8"))
    E = features.load_elements()
    onehot = bundle["onehot"]
    clf = model().fit(X, y)
    ad = bundle["ad"]
    idx = ad["idx"]
    rows_z = np.array(ad["rows"])
    surr = json.load(open(SURR)) if os.path.exists(SURR) else None
    known = set(g)
    cands = []
    for k in (5, 6):
        for combo in itertools.combinations(LAB_CATIONS, k):
            sysk = "-".join(sorted(combo))
            if sysk in known:
                continue
            frac = {e: 1 / k for e in combo}
            _, x = features.featurize(frac, E, onehot)
            cands.append((sysk, frac, x))
    Xc = np.array([c[2] for c in cands])
    P = clf.predict_proba(Xc)
    unc = entropy(P)
    z = (Xc[:, idx] - np.array(ad["mean"])) / np.array(ad["scale"])
    d = np.array([np.sort(np.linalg.norm(rows_z - zi, axis=1))[:ad["k"]].mean() for zi in z])
    inside = d <= ad["threshold"]


    # cations the CHGNet surrogate covers (same rule as the web page, tools/build_app.py)
    cov = set()
    if surr:
        n_sys = surr.get("n_systems_per_element") or [2] * len(surr["elements"])
        cov = {e for e, n in zip(surr["elements"], n_sys)
               if n >= 2 and e not in (surr.get("not_covered") or {})}
        h = dict(zip(surr["elements"], surr["h_eV_per_O"]))
        lg = surr["logistic"]
    out = []
    for i, (sysk, frac, _) in enumerate(cands):
        if not inside[i]:
            continue
        dh = p_rs = None
        if surr and all(e in cov for e in frac):
            dh = sum(frac[e] * h[e] for e in frac)
            p_rs = 1 / (1 + np.exp(-(lg["intercept"] + lg["slope"] * dh)))
        top = np.argsort(-P[i])[:2]
        out.append({"system": sysk, "entropy": float(unc[i]),
                    "top": f"{clf.classes_[top[0]]} {P[i][top[0]]:.2f} / "
                           f"{clf.classes_[top[1]]} {P[i][top[1]]:.2f}",
                    "ad": float(d[i]), "dH_ox": dh, "p_rs": p_rs})
    # greedy: most uncertain first, but no two proposals sharing four cations, and only
    # compositions the model is genuinely unsure about (entropy >= 0.3 nat)
    out.sort(key=lambda r: -r["entropy"])
    chosen = []
    for r in out:
        if len(chosen) == 8 or r["entropy"] < 0.3:
            break
        s = set(r["system"].split("-"))
        if all(len(s & set(c["system"].split("-"))) < 4 for c in chosen):
            chosen.append(r)
    with open(NEXT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "family_entropy", "top_two_families", "ad_distance",
                    "dH_ox_surrogate_eV_per_O", "p_rocksalt_thermo"])
        for r in chosen:
            w.writerow([r["system"], f"{r['entropy']:.3f}", r["top"], f"{r['ad']:.2f}",
                        "" if r["dH_ox"] is None else f"{r['dH_ox']:.2f}",
                        "" if r["p_rs"] is None else f"{r['p_rs']:.2f}"])
    lines = [f"Propositions : {len(cands)} compositions equimolaires inedites a 5-6 cations parmi "
             + " ".join(LAB_CATIONS) + f", {int(inside.sum())} dans le domaine d'applicabilite ;",
             "les plus informatives (entropie >= 0,3 sur la famille, pas deux propositions "
             "partageant quatre cations) :"]
    for r in chosen:
        dh = "" if r["dH_ox"] is None else \
            f", dH_ox ~ {r['dH_ox']:+.2f} eV/O, P(rocksalt) thermo {r['p_rs']:.2f}"
        lines.append(f"  {r['system']:<26s} entropie {r['entropy']:.2f} ({r['top']}){dh}")
    lines.append("lecture : chaque synthese tranche une question que le modele ne sait pas "
                 "trancher ; un lot multiphase est aussi informatif qu'un monophase, a "
                 "condition d'etre saisi dans data/lab_results.csv avec T et atmosphere")
    lines += ["", "liste complete : data/next_experiments.csv"]
    return lines


def main():
    X, y, g, names = load()
    lines, best = benchmark(X, y, g)
    lines += proposals(X, y, g, names, best)
    lines += ["", "genere par tools/active_learning.py"]
    text = "\n".join(lines) + "\n"
    open(REPORT, "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
