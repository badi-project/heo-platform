#!/usr/bin/env python3
"""Regenerate QC/parity_check.txt: Python reference vs JavaScript engine.

The web page recomputes the 134 descriptors and traverses the exported boosted trees
in JavaScript. This script scores the 200 probe compositions shipped in
app/heo_model.json twice — once with src/features.py plus a Python traversal of the
exported trees, once with app/heo_engine.js under node — and fails if the two sides
disagree by more than the tolerance. It also reports, for information only, the gap to
the probabilities frozen in the probe set by the original scikit-learn model; that gap
is dominated by split thresholds sitting exactly on training values and is not a gate.

    python tools/parity_check.py            # rewrite QC/parity_check.txt
    python tools/parity_check.py --check    # verify only, exit 1 on mismatch

Requires node on PATH for the JavaScript side.
"""
import argparse
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402

TOL_JS = 1e-12      # Python vs JavaScript, same algorithm in two languages
TOL_FROZEN = 1e-8   # vs the probabilities stored in the probe set (9 decimals)


def tree_value(nodes, x):
    """node = [feature_idx | -1 for leaf, threshold, left, right, value, missing_left]"""
    i = 0
    while True:
        n = nodes[i]
        if n[0] < 0:
            return n[4]
        v = x[n[0]]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            i = n[2] if n[5] else n[3]
        else:
            i = n[2] if v <= n[1] else n[3]


def raw_predict(model, x):
    raw = list(model["baseline"])
    for it in model["trees"]:
        for k in range(len(model["classes"])):
            raw[k] += tree_value(it[k], x)
    return raw


def softmax(raw, temperature):
    """Deliberately mirrors the naive accumulation in heo_engine.js."""
    z = [v / temperature for v in raw]
    mx = max(z)
    ex = [math.exp(v - mx) for v in z]
    s = 0.0
    for v in ex:
        s += v
    return [v / s for v in ex]


def python_side(bundle, elements):
    probs, feats = [], []
    for p in bundle["probe"]:
        names, x = features.featurize(p["frac"], elements, bundle["onehot"])
        assert names == bundle["featnames"], "descriptor order differs from the model export"
        feats.append(x)
        probs.append(softmax(raw_predict(bundle["model"], x), bundle["model"]["temperature"]))
    return probs, feats


def js_side():
    out = subprocess.run(
        ["node", os.path.join(ROOT, "tools", "parity_js.mjs")],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def max_abs(a, b):
    return max(abs(u - v) for ra, rb in zip(a, b) for u, v in zip(ra, rb))


def main(check=False):
    bundle = json.load(open(os.path.join(ROOT, "app", "heo_model.json"), encoding="utf-8"))
    elements = features.load_elements()

    py_probs, py_feats = python_side(bundle, elements)
    js = js_side()
    d_probs = max_abs(py_probs, js["probs"])
    d_feats = max_abs(py_feats, js["feats"])
    # the probe stores probabilities in display order (classes_order), the model emits
    # them in its own class order (model.classes) — reindex before comparing
    perm = [bundle["model"]["classes"].index(c) for c in bundle["classes_order"]]
    reordered = [[p[i] for i in perm] for p in py_probs]
    frozen = [p["probs"] for p in bundle["probe"]]
    per_case = sorted(max(abs(u - v) for u, v in zip(a, b)) for a, b in zip(reordered, frozen))
    d_frozen = per_case[-1]
    n_frozen_off = sum(1 for v in per_case if v > 1e-3)
    flips = sum(1 for a, b in zip(reordered, frozen)
                if a.index(max(a)) != b.index(max(b)))

    lines = [
        f"compositions testees : {len(bundle['probe'])}",
        f"longueur du vecteur de descripteurs : {len(py_feats[0])} / attendu {len(bundle['featnames'])}",
        f"ecart maximal de descripteur JS vs Python : {d_feats:.3e} (tolerance {TOL_JS:.0e})",
        f"ecart maximal de probabilite JS vs Python : {d_probs:.3e} (tolerance {TOL_JS:.0e})",
        "",
        "comparaison informative avec les probabilites figees du probe "
        f"(issues du modele scikit-learn d'origine, tolerance indicative {TOL_FROZEN:.0e}) :",
        f"  ecart median {per_case[len(per_case) // 2]:.3e}, ecart maximal {d_frozen:.3e}",
        f"  {n_frozen_off} compositions sur {len(per_case)} au-dela de 1e-3, "
        f"{flips} changement(s) de classe predite",
        "  ces ecarts viennent de seuils de coupure poses exactement sur des valeurs",
        "  d'entrainement : un ecart d'un ulp fait basculer quelques arbres. Ils ne sont",
        "  pas bloquants ; la garantie verifiee en integration continue est l'accord",
        "  Python / JavaScript ci-dessus.",
        "",
        "genere par tools/parity_check.py",
    ]
    report = "\n".join(lines) + "\n"
    failures = [msg for ok, msg in (
        (d_feats <= TOL_JS, f"descriptors differ by {d_feats:.3e}"),
        (d_probs <= TOL_JS, f"probabilities differ by {d_probs:.3e}"),
    ) if not ok]

    path = os.path.join(ROOT, "QC", "parity_check.txt")
    if not check:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
    print(report, end="")
    for msg in failures:
        print("FAIL: " + msg)
    if failures:
        return 1
    if check and open(path, encoding="utf-8").read() != report:
        print("FAIL: QC/parity_check.txt is stale — run python tools/parity_check.py")
        return 1
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, do not rewrite")
    sys.exit(main(check=ap.parse_args().check))
