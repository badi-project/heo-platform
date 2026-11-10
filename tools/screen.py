#!/usr/bin/env python3
"""Screen the compositional space with the exported classifier.

Enumerates every equimolar quinary of the cations documented at least MIN_COUNT times
in the database, then the senaries obtained by adding one cation to the N_SEED best
in-domain quinaries. Each composition gets its descriptors (src/features.py), the
calibrated class probabilities of the published model, its distance to the training set
and a novelty flag. Writes

    data/screening_all.csv.gz   every composition
    data/heo_candidats.csv      unpublished, in-domain candidates, at most CAP per family,
                                with the nearest published system and a predicted lattice
                                parameter

The model is refitted here from the same data and seed as tools/train_model.py, and the
script stops unless it reproduces the probabilities frozen in app/heo_model.json
exactly — so the screening is guaranteed to come from the published model.

    python tools/screen.py
"""
import csv
import gzip
import itertools
import json
import os
import sys
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import features  # noqa: E402
import lattice  # noqa: E402
import train_model as tm  # noqa: E402

MIN_COUNT = 20
N_SEED = 3000
CAP = 700
CHUNK = 20000


def lattice_parameter(proto, frac, E):
    """Geometric lattice model of the platform (src/lattice.py, predictA() in
    app/heo_engine.js); None for a family without a cubic prototype or an ion without
    a Shannon radius."""
    if proto not in lattice.rules()["families"]:
        return None
    try:
        return lattice.predict(proto, frac, E)["a"]
    except lattice.LatticeError:
        return None


def main():
    bundle = json.load(open(os.path.join(ROOT, "app", "heo_model.json"), encoding="utf-8"))
    E = features.load_elements()
    onehot, T, ad = bundle["onehot"], bundle["model"]["temperature"], bundle["ad"]

    # refit the published model and prove it is the same one
    rows = tm.load_rows()
    names, Xall = tm.featurize_rows(rows, E, onehot)
    lab = np.array([r["proto"] in tm.CLASSES for r in rows])
    y = np.array([r["proto"] for r in rows])[lab]
    clf = tm.hgb().fit(Xall[lab], y)
    PX = np.array([features.featurize(p["frac"], E, onehot)[1] for p in bundle["probe"]])
    order = [tm.CLASSES.index(c) for c in bundle["classes_order"]]
    probs = tm.softmax(clf._raw_predict(PX) / T)[:, order]
    frozen = np.array([p["probs"] for p in bundle["probe"]])
    if np.abs(probs - frozen).max() > 1e-12:
        raise SystemExit("refitted model differs from app/heo_model.json: retrain first")

    published = Counter(r["sysk"] for r in rows)
    cnt = Counter(e for r in rows for e in r["cations"].split("-"))
    pool = sorted(e for e, n in cnt.items() if n >= MIN_COUNT)
    mean, scale = np.array(ad["mean"]), np.array(ad["scale"])
    ref = np.array(ad["rows"])
    ref2 = (ref ** 2).sum(axis=1)

    def score(sets):
        out = []
        for i in range(0, len(sets), CHUNK):
            part = sets[i:i + CHUNK]
            X = np.array([features.featurize({e: 1.0 / len(s) for e in s}, E, onehot)[1]
                          for s in part])
            p = tm.softmax(clf._raw_predict(X) / T)
            Z = (X[:, ad["idx"]] - mean) / scale
            d2 = np.maximum((Z ** 2).sum(axis=1)[:, None] + ref2[None, :] - 2 * Z @ ref.T, 0)
            dist = np.sort(np.sqrt(d2), axis=1)[:, :ad["k"]].mean(axis=1)
            for s, pi, d, x in zip(part, p, dist, X):
                k = int(pi.argmax())
                out.append({"cations": "-".join(s), "proto": tm.CLASSES[k],
                            "p": float(pi[k]), "ad": float(d),
                            "novel": int("-".join(s) not in published),
                            "in_domain": int(d <= ad["threshold"]), "n_cat": len(s),
                            "gold_t": float(x[names.index("goldschmidt_t")]),
                            "delta_r": float(x[names.index("delta_r")]),
                            "s_config": float(x[names.index("s_config")])})
            print(f"  {min(i + CHUNK, len(sets))}/{len(sets)}", file=sys.stderr, end="\r")
        print(file=sys.stderr)
        return out

    quin = [tuple(c) for c in itertools.combinations(pool, 5)]
    res = score(quin)
    seeds = sorted((r for r in res if r["in_domain"]), key=lambda r: -r["p"])[:N_SEED]
    sen = sorted({tuple(sorted(set(r["cations"].split("-")) | {e}))
                  for r in seeds for e in pool if e not in r["cations"].split("-")})
    res += score(sen)

    with gzip.open(os.path.join(ROOT, "data", "screening_all.csv.gz"), "wt",
                   encoding="utf-8", newline="", compresslevel=9) as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["cations", "proto", "p", "ad", "novel", "in_domain", "n_cat"])
        for r in res:
            w.writerow([r["cations"], r["proto"], f"{r['p']:.6f}", f"{r['ad']:.4f}",
                        r["novel"], r["in_domain"], r["n_cat"]])

    systems = [set(s.split("-")) for s in published]
    sys_names = list(published)
    cands = []
    for proto in tm.CLASSES_ORDER:
        sel = sorted((r for r in res if r["proto"] == proto and r["novel"]
                      and r["in_domain"]), key=lambda r: -r["p"])[:CAP]
        for r in sel:
            s = set(r["cations"].split("-"))
            shared = [len(s & t) for t in systems]
            best = max(shared)
            near = min((n for n, k in zip(sys_names, shared) if k == best),
                       key=lambda n: (-published[n], n))
            a = lattice_parameter(proto, {e: 1.0 / len(s) for e in s}, E)
            cands.append({**r, "voisin_publie": near, "cations_partages": best,
                          "a_predit": a})
    fields = ["cations", "proto", "p", "ad", "gold_t", "delta_r", "s_config", "novel",
              "n_cat", "in_domain", "voisin_publie", "cations_partages", "a_predit"]
    with open(os.path.join(ROOT, "data", "heo_candidats.csv"), "w", encoding="utf-8",
              newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        for c in cands:
            w.writerow({**c, "a_predit": "" if c["a_predit"] is None else c["a_predit"]})

    n_in = sum(r["in_domain"] for r in res)
    fam = Counter(r["proto"] for r in res)
    fam_in = Counter(r["proto"] for r in res if r["in_domain"])
    lines = [
        f"vivier : {len(pool)} cations documentes au moins {MIN_COUNT} fois",
        f"compositions criblees : {len(res)} ({len(quin)} quinaires equimolaires, "
        f"{len(sen)} senaires issues des {N_SEED} meilleurs quinaires)",
        f"dans le domaine d'applicabilite : {n_in} ({100 * n_in / len(res):.0f} %)",
        f"dans le domaine et absentes de la base : "
        f"{sum(1 for r in res if r['in_domain'] and r['novel'])}",
        f"dans le domaine, p >= 0,7 : {sum(1 for r in res if r['in_domain'] and r['p'] >= .7)}",
        "famille predite, toutes compositions : " + ", ".join(
            f"{c} {100 * fam[c] / len(res):.0f} %" for c in tm.CLASSES_ORDER),
        "famille predite, dans le domaine : " + ", ".join(
            f"{c} {100 * fam_in[c] / max(n_in, 1):.0f} %" for c in tm.CLASSES_ORDER),
        f"candidats exportes : {len(cands)} (au plus {CAP} par famille)",
        "meilleur candidat non publie par famille :",
    ]
    for proto in tm.CLASSES_ORDER:
        top = [c for c in cands if c["proto"] == proto][:1]
        if top:
            c = top[0]
            a = f"{c['a_predit']:.3f} A" if c["a_predit"] else "-"
            lines.append(f"  {proto:<11s} {c['cations']:<22s} p = {c['p']:.3f}  a = {a}")
    lines += ["", "genere par tools/screen.py"]
    report = "\n".join(lines) + "\n"
    print(report, end="")
    with open(os.path.join(ROOT, "QC", "screening_report.txt"), "w", encoding="utf-8") as fh:
        fh.write(report)


if __name__ == "__main__":
    main()
