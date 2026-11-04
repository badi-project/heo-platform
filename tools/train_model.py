#!/usr/bin/env python3
"""Train, validate and export the structure-family classifier.

Until v1.0 the training chain was not in the repository: the exported model could
not be regenerated and differed from its own frozen probe on 22 of 200 compositions.
This script is that chain. From data/heo_database.csv it

1. computes the 134 descriptors with src/features.py on the real cation fractions
   (written to data/heo_features.csv);
2. validates the classifier three ways, each stricter than the last:
     - 20-fold cross-validation grouped by cation system (no system on both sides),
     - temporal split: trained on papers up to 2024, tested on 2025-2027,
     - leave-chemistry-out: 20 clusters of cation sets held out one at a time;
   and compares it with simple baselines, so that the gain is measured, not assumed;
3. fits a temperature on the out-of-fold logits (probability calibration);
4. fits the final model on all data and exports it to app/heo_model.json with
   unrounded thresholds, checking that the exported trees reproduce scikit-learn's
   raw predictions exactly before writing anything;
5. writes QC/training_report.txt.

    python tools/train_model.py

Requires numpy and scikit-learn (requirements-train.txt pins the versions used for
the published model: tree thresholds depend on them).
"""
import csv
import json
import math
import os
import sys
from collections import Counter

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402

CLASSES = ["fluorite", "layered", "perovskite", "pyrochlore", "rocksalt", "spinel"]
CLASSES_ORDER = ["spinel", "perovskite", "rocksalt", "fluorite", "pyrochlore", "layered"]
N_ONEHOT = 30
SEED = 0
HGB_PARAMS = dict(max_iter=250, max_leaf_nodes=15, class_weight="balanced",
                  random_state=SEED)
AD_FEATS = ["goldschmidt_t", "octahedral_factor", "delta_r", "delta_X_rel", "r_A", "r_B",
            "mean_ox", "ox_spread", "redox_flex", "s_config", "n_cat",
            "r_shannon_VI_mean", "X_mean", "mass_mean", "melting_point_mean",
            "e_density_mean", "frac_RE", "frac_TM3d", "frac_AE", "frac_pblock"]
AD_K, AD_PCT = 5, 95
CLASSIC_9 = ["goldschmidt_t", "octahedral_factor", "delta_r", "delta_X_rel", "mean_ox",
             "s_config", "r_shannon_VI_mean", "X_mean", "n_cat"]
TEMPORAL_CUT = 2024


def hgb():
    return HistGradientBoostingClassifier(**HGB_PARAMS)


# ------------------------------------------------------------------ data -------
def load_rows():
    with open(os.path.join(ROOT, "data", "heo_database.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def onehot_elements(rows):
    cnt = Counter(e for r in rows for e in r["cations"].split("-"))
    return sorted(sorted(cnt, key=lambda e: (-cnt[e], e))[:N_ONEHOT])


def featurize_rows(rows, elements, onehot):
    X = []
    for r in rows:
        fr = json.loads(r["frac"])
        tot = sum(fr.values())
        names, vals = features.featurize({e: v / tot for e, v in fr.items()}, elements,
                                         onehot)
        X.append(vals)
    return names, np.array(X, dtype=np.float64)


def stoich_pattern(rows, width=10):
    """Cation fractions sorted in decreasing order, zero-padded: the shape of the
    written formula with every chemical identity removed."""
    out = []
    for r in rows:
        v = sorted(json.loads(r["frac"]).values(), reverse=True)[:width]
        tot = sum(v)
        out.append([x / tot for x in v] + [0.0] * (width - len(v)) + [float(len(v))])
    return np.array(out)


def write_features(rows, names, X):
    path = os.path.join(ROOT, "data", "heo_features.csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["doi", "sysk"] + names + ["proto", "phase", "tier"])
        for r, x in zip(rows, X):
            w.writerow([r["doi"], r["sysk"]] + [repr(float(v)) for v in x]
                       + [r["proto"], r["phase"], r["tier"]])


# ------------------------------------------------------------- metrics ---------
def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(raw, y_idx, T):
    p = softmax(raw / T)
    return -np.mean(np.log(np.clip(p[np.arange(len(y_idx)), y_idx], 1e-15, 1)))


def fit_temperature(raw, y_idx):
    lo, hi = math.log(0.2), math.log(10.0)
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    for _ in range(80):
        if nll(raw, y_idx, math.exp(a)) < nll(raw, y_idx, math.exp(b)):
            hi = b
        else:
            lo = a
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    return math.exp((lo + hi) / 2)


def hash_half(s):
    """Deterministic 0/1 split of a cation-system key (no Python hash randomisation)."""
    return sum(s.encode("utf-8")) % 2


def ece(p, y_idx, bins=10):
    conf, pred = p.max(axis=1), p.argmax(axis=1)
    edges = np.linspace(0, 1, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return tot


def scores(y, pred):
    return accuracy_score(y, pred), f1_score(y, pred, average="macro", labels=CLASSES)


def oof(make, X, y, groups, n_splits=20):
    """Out-of-fold predictions and raw logits under a grouped split."""
    raw = np.zeros((len(y), len(CLASSES)))
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = make()
        clf.fit(X[tr], y[tr])
        pred[te] = clf.predict(X[te])
        if hasattr(clf, "decision_function"):
            d = clf.decision_function(X[te])
            # align to CLASSES (every fold sees all six classes in training)
            idx = [list(clf.classes_).index(c) for c in CLASSES]
            raw[te] = d[:, idx]
    return pred, raw


def nearest_system_baseline(cats, y, groups, n_splits=20):
    """Label of the most similar published cation set (Jaccard) — no learning at all."""
    pred = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=n_splits).split(cats, y, groups):
        for i in te:
            sims = np.array([len(cats[i] & cats[j]) / len(cats[i] | cats[j]) for j in tr])
            best = tr[sims == sims.max()]
            vals, cnt = np.unique(y[best], return_counts=True)
            pred[i] = vals[cnt.argmax()]
    return pred


def goldschmidt_rule(t, y):
    """Best perovskite window lo <= t <= hi, tuned on all data (optimistic baseline)."""
    is_p = y == "perovskite"
    best = (0, None)
    grid = np.unique(np.round(t, 3))
    for lo in grid[::3]:
        for hi in grid[::3]:
            if hi <= lo:
                continue
            acc = np.mean(((t >= lo) & (t <= hi)) == is_p)
            if acc > best[0]:
                best = (acc, (lo, hi))
    return best


# --------------------------------------------------------------- export --------
def export_trees(clf, featnames):
    """Trees as [feature | -1, threshold, left, right, leaf value, missing_go_to_left],
    thresholds unrounded (a 1-ulp rounding flips splits, see docs §8)."""
    out = []
    for it in clf._predictors:
        per_class = []
        for pred in it:
            nodes = []
            for n in pred.nodes:
                if n["is_leaf"]:
                    nodes.append([-1, 0.0, 0, 0, float(n["value"]), 0])
                else:
                    nodes.append([int(n["feature_idx"]), float(n["num_threshold"]),
                                  int(n["left"]), int(n["right"]), 0.0,
                                  int(n["missing_go_to_left"])])
            per_class.append(nodes)
        out.append(per_class)
    return out


def tree_value(nodes, x):
    i = 0
    while True:
        n = nodes[i]
        if n[0] < 0:
            return n[4]
        i = n[2] if x[n[0]] <= n[1] else n[3]


def raw_from_export(model, x):
    raw = list(model["baseline"])
    for it in model["trees"]:
        for k in range(len(model["classes"])):
            raw[k] += tree_value(it[k], x)
    return raw


# ----------------------------------------------------------------- main --------
def main():
    rows = [r for r in load_rows()]
    elements = features.load_elements()
    onehot = onehot_elements(rows)
    names, Xall = featurize_rows(rows, elements, onehot)
    write_features(rows, names, Xall)

    lab = np.array([r["proto"] in CLASSES for r in rows])
    X = Xall[lab]
    y = np.array([r["proto"] for r in rows])[lab]
    groups = np.array([r["sysk"] for r in rows])[lab]
    years = np.array([int(r["year"]) for r in rows])[lab]
    cats = np.array([set(r["cations"].split("-")) for r in rows], dtype=object)[lab]
    y_idx = np.array([CLASSES.index(c) for c in y])
    fi = {n: i for i, n in enumerate(names)}
    rep = []

    def line(s=""):
        rep.append(s)
        print(s)

    M = {"n_rows": len(rows), "n_labeled": int(len(y)), "n_systems": len(set(groups)),
         "class_counts": {c: int((y == c).sum()) for c in CLASSES_ORDER}}
    line(f"echantillons etiquetes (6 familles) : {len(y)} sur {len(rows)} ; "
         f"systemes cationiques distincts : {len(set(groups))}")
    line("repartition : " + ", ".join(f"{c} {int((y == c).sum())}" for c in CLASSES_ORDER))
    line(f"scikit-learn {sklearn.__version__}, numpy {np.__version__}, graine {SEED}")
    line()

    # 1. grouped cross-validation -------------------------------------------------
    line("1. Validation croisee en 20 plis groupes par systeme cationique")
    pred, raw = oof(hgb, X, y, groups)
    acc, f1 = scores(y, pred)
    # rounded once, here: the exported model, the frozen probe and every metric below
    # use this exact value
    T = round(fit_temperature(raw, y_idx), 6)
    p_raw, p_cal = softmax(raw), softmax(raw / T)
    line(f"   arbres boostes, 134 descripteurs : exactitude {acc:.3f}, F1 macro {f1:.3f}, "
         f"log-vraisemblance {nll(raw, y_idx, 1.0):.3f} -> {nll(raw, y_idx, T):.3f} "
         f"apres calibration (T = {T:.3f})")
    line(f"   erreur de calibration (ECE, 10 classes) : {ece(p_raw, y_idx):.3f} -> "
         f"{ece(p_cal, y_idx):.3f}")
    # the same T scored on systems it was not fitted on: fit on one half of the
    # cation systems, score on the other, and swap
    halves = np.array([hash_half(g) for g in groups])
    p_cross = np.zeros_like(p_cal)
    for h in (0, 1):
        Th = fit_temperature(raw[halves != h], y_idx[halves != h])
        p_cross[halves == h] = softmax(raw[halves == h] / Th)
    ece_cross = ece(p_cross, y_idx)
    line(f"   ECE avec T ajustee sur l'autre moitie des systemes (sans fuite) : {ece_cross:.3f}")
    per = f1_score(y, pred, average=None, labels=CLASSES_ORDER)
    line("   F1 par famille : " + ", ".join(f"{c} {v:.2f}" for c, v in zip(CLASSES_ORDER, per)))
    cm = confusion_matrix(y, pred, labels=CLASSES_ORDER)
    line("   matrice de confusion (lignes = vrai, colonnes = predit, ordre "
         + "/".join(CLASSES_ORDER) + ") :")
    for c, rowc in zip(CLASSES_ORDER, cm):
        line(f"     {c:<11s}" + "".join(f"{v:5d}" for v in rowc))
    cv_main = (acc, f1)
    conf, hit = p_cal.max(axis=1), p_cal.argmax(axis=1) == y_idx
    rel = {}
    for lo_, hi_ in ((0.9, 1.0), (0.6, 0.7)):
        m_ = (conf > lo_) & (conf <= hi_)
        rel[f"{lo_:.1f}-{hi_:.1f}"] = {"n": int(m_.sum()), "confidence": float(conf[m_].mean()),
                                      "accuracy": float(hit[m_].mean())}
    M["cv"] = {"accuracy": acc, "f1_macro": f1, "nll_raw": nll(raw, y_idx, 1.0),
               "nll_cal": nll(raw, y_idx, T), "temperature": T,
               "ece_raw": ece(p_raw, y_idx), "ece_cal": ece(p_cal, y_idx), "ece_cross": ece_cross,
               "f1_per_class": {c: float(v) for c, v in zip(CLASSES_ORDER, per)},
               "rocksalt_as_spinel": int(cm[2][0]), "spinel_as_rocksalt": int(cm[0][2]),
               "reliability": rel}

    line("   references :")
    maj = Counter(y).most_common(1)[0][0]
    line(f"     classe majoritaire ({maj}) : exactitude {np.mean(y == maj):.3f}")
    M["baselines"] = {"majority": {"class": maj, "accuracy": float(np.mean(y == maj))}}
    pn = nearest_system_baseline(cats, y, groups)
    a, f = scores(y, pn)
    line(f"     systeme publie le plus proche (Jaccard, sans apprentissage) : "
         f"exactitude {a:.3f}, F1 macro {f:.3f}")
    M["baselines"]["nearest_system"] = {"accuracy": a, "f1_macro": f}
    oh = [fi["has_" + e] for e in onehot]
    a, f = scores(y, oof(hgb, X[:, oh], y, groups)[0])
    line(f"     arbres boostes, presence des 30 cations seulement : exactitude {a:.3f}, "
         f"F1 macro {f:.3f}")
    M["baselines"]["cation_presence"] = {"accuracy": a, "f1_macro": f}
    c9 = [fi[n] for n in CLASSIC_9]
    a, f = scores(y, oof(hgb, X[:, c9], y, groups)[0])
    line(f"     arbres boostes, 9 descripteurs classiques : exactitude {a:.3f}, "
         f"F1 macro {f:.3f}")
    M["baselines"]["classic_9"] = {"accuracy": a, "f1_macro": f}
    # how much does the written cation ratio alone give away? (La(..)O3 always has one
    # cation at 0.5, (..)Fe2O4 one at 0.67): sorted fractions, no chemistry at all
    XS = stoich_pattern(rows)[lab]
    a, f = scores(y, oof(hgb, XS, y, groups)[0])
    line(f"     arbres boostes, motif stoechiometrique seul (fractions triees, sans "
         f"chimie) : exactitude {a:.3f}, F1 macro {f:.3f}")
    M["baselines"]["stoich_pattern"] = {"accuracy": a, "f1_macro": f}
    eqm = np.array([max(json.loads(r["frac"]).values()) - min(json.loads(r["frac"]).values())
                    < 1e-6 for r in rows])[lab]
    a_eq, f_eq = scores(y[eqm], pred[eqm])
    line(f"     modele complet restreint aux {int(eqm.sum())} compositions equimolaires "
         f"(aucun indice de stoechiometrie) : exactitude {a_eq:.3f}, F1 macro {f_eq:.3f}")
    M["cv"]["equimolar_subset"] = {"n": int(eqm.sum()), "accuracy": a_eq, "f1_macro": f_eq}
    lr = lambda: make_pipeline(StandardScaler(),  # noqa: E731
                               LogisticRegression(max_iter=4000, class_weight="balanced"))
    a, f = scores(y, oof(lr, X, y, groups)[0])
    line(f"     regression logistique, 134 descripteurs : exactitude {a:.3f}, "
         f"F1 macro {f:.3f}")
    M["baselines"]["logistic"] = {"accuracy": a, "f1_macro": f}
    gacc, (glo, ghi) = goldschmidt_rule(X[:, fi["goldschmidt_t"]], y)
    bin_acc = np.mean((pred == "perovskite") == (y == "perovskite"))
    line(f"   perovskite ou non : modele {bin_acc:.3f} ; regle de Goldschmidt "
         f"{glo:.3f} <= t <= {ghi:.3f} optimisee sur toute la base {gacc:.3f}")
    M["perovskite_binary"] = {"model": float(bin_acc), "goldschmidt": float(gacc),
                              "t_window": [float(glo), float(ghi)]}
    line()

    # 2. temporal validation ------------------------------------------------------
    line(f"2. Validation temporelle : apprentissage sur les articles <= {TEMPORAL_CUT}, "
         f"test sur {TEMPORAL_CUT + 1}-{years.max()}")
    tr, te = years <= TEMPORAL_CUT, years > TEMPORAL_CUT
    clf = hgb().fit(X[tr], y[tr])
    pt = clf.predict(X[te])
    a, f = scores(y[te], pt)
    seen = set(groups[tr])
    unseen = np.array([g not in seen for g in groups[te]])
    a2, f2 = scores(y[te][unseen], pt[unseen])
    line(f"   {int(tr.sum())} echantillons d'apprentissage, {int(te.sum())} de test : "
         f"exactitude {a:.3f}, F1 macro {f:.3f}")
    line(f"   dont {int(unseen.sum())} systemes jamais vus avant {TEMPORAL_CUT + 1} : "
         f"exactitude {a2:.3f}, F1 macro {f2:.3f}")
    clf_oh = hgb().fit(X[tr][:, oh], y[tr])
    a3, _ = scores(y[te][unseen], clf_oh.predict(X[te][:, oh])[unseen])
    line(f"   reference presence des cations, memes systemes nouveaux : exactitude {a3:.3f}")
    temporal = (a, f, a2, f2, int(unseen.sum()))
    M["temporal"] = {"cut": TEMPORAL_CUT, "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "accuracy": a, "f1_macro": f, "n_unseen": int(unseen.sum()),
                     "accuracy_unseen": a2, "f1_unseen": f2,
                     "cation_presence_unseen": a3}
    line()

    # 3. leave-chemistry-out ------------------------------------------------------
    km = KMeans(n_clusters=20, n_init=10, random_state=SEED).fit(X[:, oh])
    line("3. Chimie inedite : 20 groupes de systemes voisins (k-moyennes sur la presence "
         "des cations), chacun retire a son tour")
    a, f = scores(y, oof(hgb, X, y, km.labels_)[0])
    a0, f0 = scores(y, nearest_system_baseline(cats, y, km.labels_))
    a1, f1b = scores(y, oof(hgb, X[:, oh], y, km.labels_)[0])
    line(f"   arbres boostes, 134 descripteurs : exactitude {a:.3f}, F1 macro {f:.3f}")
    line(f"   presence des cations seulement : exactitude {a1:.3f}, F1 macro {f1b:.3f}")
    line(f"   systeme publie le plus proche : exactitude {a0:.3f}, F1 macro {f0:.3f}")
    chem = (a, f)
    M["leave_chemistry_out"] = {"accuracy": a, "f1_macro": f, "cation_presence": a1,
                                "nearest_system": a0}
    line()

    # 4. final model, applicability domain, export --------------------------------
    final = hgb().fit(X, y)
    assert list(final.classes_) == CLASSES
    model = {"classes": CLASSES,
             "baseline": [float(v) for v in np.ravel(final._baseline_prediction)],
             "temperature": T,
             "trees": export_trees(final, names)}

    ad_idx = [fi[n] for n in AD_FEATS]
    Z = X[:, ad_idx]
    mean, scale = Z.mean(axis=0), Z.std(axis=0)
    scale[scale == 0] = 1.0
    Zs = np.round((Z - mean) / scale, 5)
    d2 = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    knn = np.sort(np.sqrt(d2), axis=1)[:, :AD_K].mean(axis=1)
    ad = {"idx": ad_idx, "mean": mean.tolist(), "scale": scale.tolist(),
          "rows": Zs.tolist(), "k": AD_K, "threshold": float(np.percentile(knn, AD_PCT))}

    # probe: 100 database compositions + 100 random equimolar ones (out of domain too)
    rng = np.random.default_rng(SEED)
    probe_fr = []
    for i in rng.choice(len(rows), 100, replace=False):
        fr = json.loads(rows[i]["frac"])
        tot = sum(fr.values())
        probe_fr.append({e: v / tot for e, v in fr.items()})
    pool = sorted(set(e for r in rows for e in r["cations"].split("-")))
    for _ in range(100):
        els = rng.choice(pool, int(rng.integers(5, 8)), replace=False)
        probe_fr.append({str(e): 1.0 / len(els) for e in els})
    PX = np.array([features.featurize(f, elements, onehot)[1] for f in probe_fr])
    sk_raw = final._raw_predict(PX)
    ex_raw = np.array([raw_from_export(model, x) for x in PX])
    gap = float(np.abs(sk_raw - ex_raw).max())
    line(f"4. Export : {sum(len(t) for it in model['trees'] for t in it)} noeuds ; "
         f"ecart maximal arbres exportes / scikit-learn sur le probe : {gap:.1e}")
    if gap > 1e-12:
        raise SystemExit("exported trees do not reproduce scikit-learn: not writing")
    probs = softmax(sk_raw / T)
    order = [CLASSES.index(c) for c in CLASSES_ORDER]
    probe = [{"frac": f, "probs": [float(p[i]) for i in order]}
             for f, p in zip(probe_fr, probs)]
    line(f"   domaine d'applicabilite : {len(AD_FEATS)} descripteurs, k = {AD_K}, "
         f"seuil {ad['threshold']:.3f} ({AD_PCT}e centile)")
    M["export"] = {"nodes": sum(len(t) for it in model["trees"] for t in it), "gap": gap,
                   "ad_threshold": ad["threshold"]}

    bundle = {"model": model, "featnames": names, "onehot": onehot,
              "props": features.PROPS, "classes_order": CLASSES_ORDER, "probe": probe,
              "ad": ad,
              "metrics": M,
              "meta": {"n_train": int(len(y)), "sklearn": sklearn.__version__,
                       "cv_grouped": {"accuracy": round(cv_main[0], 4),
                                      "f1_macro": round(cv_main[1], 4)},
                       "temporal": {"cut": TEMPORAL_CUT, "accuracy": round(temporal[0], 4),
                                    "f1_macro": round(temporal[1], 4),
                                    "accuracy_unseen": round(temporal[2], 4),
                                    "n_unseen": temporal[4]},
                       "leave_chemistry_out": {"accuracy": round(chem[0], 4),
                                               "f1_macro": round(chem[1], 4)},
                       "temperature": round(T, 6)}}
    with open(os.path.join(ROOT, "app", "heo_model.json"), "w", encoding="utf-8") as fh:
        json.dump(bundle, fh, separators=(",", ":"))
    line()
    line("genere par tools/train_model.py")
    with open(os.path.join(ROOT, "QC", "training_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep) + "\n")


if __name__ == "__main__":
    main()
