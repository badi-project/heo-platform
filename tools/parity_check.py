#!/usr/bin/env python3
"""Regenerate QC/parity_check.txt: Python reference vs JavaScript engine.

The web page recomputes the 134 descriptors and traverses the exported boosted trees
in JavaScript. This script scores the 200 probe compositions shipped in
app/heo_model.json twice — once with src/features.py plus a Python traversal of the
exported trees, once with app/heo_engine.js under node — and fails if the two sides
disagree by more than the tolerance. It also fails if the exported trees do not
reproduce the probabilities frozen in the probe by scikit-learn at export time
(tools/train_model.py), and if the JavaScript formula parser (FX in heo_engine.js)
reads any database formula differently from src/formula.py.

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
import formula  # noqa: E402

TOL_JS = 1e-12      # Python vs JavaScript, same algorithm in two languages
TOL_FROZEN = 1e-9   # exported trees vs the scikit-learn probabilities frozen in the probe


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


def formula_cases():
    """Every formula of the database plus the notations of tests/test_formula.py."""
    import csv
    with open(os.path.join(ROOT, "data", "heo_database.csv"), encoding="utf-8") as fh:
        texts = [r["formula_text"] for r in csv.DictReader(fh)]
    extra = ["(Mg, Co, Ni, Cu, Zn)_{1-x}Li_xO", "LaBO3 (B = Sc, Al, Cr, Ni, Fe)",
             "(XCoNiCuZn)O (X = Li, Na, Mg)", "10La2O3-20TiO2-10Nb2O5-20WO3-20ZrO2",
             "La₁₋ₓ(Nd, Sm, Gd, Y)ₓ/₄CoO₃", "HEO (MgCoNiCuZn oxide with Rh nanoparticles)",
             "(FeCoNiMnCu)Ox/13X", "Mn2–xAlxZn0.2Ni0.6Mg0.2O4", "TiZrNbHfTaO<sub>x</sub>",
             "Co-rich (CoCrFeM nNi)", "spinel oxide", ""]
    return texts + extra


def formula_parity(elements):
    texts = formula_cases()
    py = [formula.cation_fractions(t, elements) for t in texts]
    out = subprocess.run(["node", os.path.join(ROOT, "tools", "parity_js.mjs"), "formulas"],
                         input=json.dumps(texts), capture_output=True, text=True, check=True)
    js = json.loads(out.stdout)
    worst, bad = 0.0, []
    for t, a, b in zip(texts, py, js):
        if set(a) != set(b):
            bad.append(t)
            continue
        for k in a:
            worst = max(worst, abs(a[k] - b[k]))
    return len(texts), bad, worst


def max_abs(a, b):
    return max(abs(u - v) for ra, rb in zip(a, b) for u, v in zip(ra, rb))


def main(check=False):
    bundle = json.load(open(os.path.join(ROOT, "app", "heo_model.json"), encoding="utf-8"))
    elements = features.load_elements()

    py_probs, py_feats = python_side(bundle, elements)
    js = js_side()
    n_formulas, formula_mismatch, d_formula = formula_parity(elements)
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
        f"probabilites figees du probe (modele scikit-learn exporte par tools/train_model.py) :",
        f"  ecart maximal {d_frozen:.3e} (tolerance {TOL_FROZEN:.0e}), "
        f"{flips} changement(s) de classe predite",
        f"analyseur de formules, {n_formulas} formules (base + cas de test) : "
        f"{len(formula_mismatch)} ensemble(s) de cations differents, "
        f"ecart maximal de fraction {d_formula:.3e} (tolerance {TOL_JS:.0e})",
        "",
        "genere par tools/parity_check.py",
    ]
    report = "\n".join(lines) + "\n"
    failures = [msg for ok, msg in (
        (d_feats <= TOL_JS, f"descriptors differ by {d_feats:.3e}"),
        (d_probs <= TOL_JS, f"probabilities differ by {d_probs:.3e}"),
        (d_frozen <= TOL_FROZEN, f"exported model differs from its frozen probe by "
                                 f"{d_frozen:.3e}: rerun tools/train_model.py"),
        (not formula_mismatch, "formula parser differs on: " + "; ".join(formula_mismatch[:5])),
        (d_formula <= TOL_JS, f"formula fractions differ by {d_formula:.3e}"),
    ) if not ok]

    path = os.path.join(ROOT, "QC", "parity_check.txt")
    if not check:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(report)
    print(report, end="")
    for msg in failures:
        print("FAIL: " + msg)
    # QC/parity_check.txt is not compared byte for byte: the reported magnitudes depend
    # on the platform libm at the last digit, so only the tolerances above are a gate.
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, do not rewrite")
    sys.exit(main(check=ap.parse_args().check))
