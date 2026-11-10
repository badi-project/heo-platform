#!/usr/bin/env python3
"""Regenerate QC/parity_check.txt: Python reference vs JavaScript engine.

The web page recomputes the 134 descriptors and traverses the exported boosted trees
in JavaScript. This script scores the 200 probe compositions shipped in
app/heo_model.json twice — once with src/features.py plus a Python traversal of the
exported trees, once with app/heo_engine.js under node — and fails if the two sides
disagree by more than the tolerance. It also fails if the exported trees do not
reproduce the probabilities frozen in the probe by scikit-learn at export time
(tools/train_model.py), and if the JavaScript formula parser (FX in heo_engine.js)
reads any database formula differently from src/formula.py. Finally it compares the
complete prediction — family, applicability domain, lattice parameter, powder pattern,
computed thermodynamics — between src/predict.py and predict() of heo_engine.js, on every
readable formula of the database: the command line and the page must give the same answer.

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
import predict  # noqa: E402
import xrdfit  # noqa: E402

TOL_JS = 1e-12      # Python vs JavaScript, same algorithm in two languages
TOL_FROZEN = 1e-9   # exported trees vs the scikit-learn probabilities frozen in the probe
TOL_PRED = 1e-9     # complete prediction (domain distance, lattice, pattern, thermo)


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


def prediction_parity():
    """Complete prediction, src/predict.py against predict() of heo_engine.js, on every
    readable formula of the database and on the probe compositions. Both engines receive
    the same data (Predictor.context()). Returns the number of compositions, the list of
    discrete disagreements and the largest numerical gap per quantity."""
    import csv
    P = predict.Predictor()
    fracs, texts = [], formula_cases()
    with open(os.path.join(ROOT, "data", "heo_database.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                fracs.append(P.read(r["formula_text"])[0])
            except predict.PredictionError:
                pass
    fracs += [p["frac"] for p in P.bundle["probe"]]
    profiled = fracs[::25]
    measured = measured_cases(P)
    out = subprocess.run(["node", os.path.join(ROOT, "tools", "parity_js.mjs"), "predict"],
                         input=json.dumps({"context": P.context(), "fracs": fracs,
                                           "texts": texts, "profiles": profiled,
                                           "measured": measured}),
                         capture_output=True, text=True, check=True)
    result = json.loads(out.stdout)
    classes = P.model["classes"]
    src = {"geometric": "geo", "chgnet": "chgnet"}
    gap = {"probs": 0.0, "ad": 0.0, "a": 0.0, "two_theta": 0.0, "intensity": 0.0, "dh": 0.0,
           "density": 0.0, "profile": 0.0, "fit": 0.0}
    bad = []
    # flat records (batch table, CSV export): text against text
    rows_off = [t for t, js in zip(texts, result["rows"]) if P.record(t) != js]
    for frac, js in zip(profiled, result["profiles"]):
        py = [y for _, y in P.profile(P.predict(frac)["pattern"])]
        if len(py) != len(js):
            bad.append("profile " + "-".join(frac))
        else:
            gap["profile"] = max([gap["profile"]] + [abs(u - v) for u, v in zip(py, js)])

    def worst(key, u, v):
        gap[key] = max(gap[key], abs(u - v))
    for frac, js in zip(fracs, result["full"]):
        py = P.predict(frac)
        name = "-".join(frac)
        for i, c in enumerate(classes):
            worst("probs", py["probabilities"][c], js["probs"][i])
        worst("ad", py["ad_distance"], js["ad"])
        lat, th = py["lattice"], py["thermo"]
        ok = (py["family"] == js["top"] and py["in_domain"] == js["inDomain"]
              and (lat is None) == (js["a"] is None)
              and len(py["pattern"]) == len(js["pattern"])
              and (th is None) == (js["thermo"] is None))
        if ok and lat:
            worst("a", lat["a"], js["a"])
            worst("density", lat["density_g_cm3"], js["rho"])
            ok = src[lat["source"]] == js["aSrc"]
        if ok:
            for u, v in zip(py["pattern"], js["pattern"]):
                worst("two_theta", u["two_theta"], v["tt"])
                worst("intensity", u["intensity"], v["I"])
                ok = ok and u["hkl"] == v["hkl"]
        if ok and th:
            t = js["thermo"]
            ok = th["covered"] == t["covered"] and th["missing"] == t["missing"]
            if ok and th["covered"]:
                worst("dh", th["dh_ox_eV_per_O"], t["dh"])
                worst("dh", th["p_rocksalt"], t["pRocksalt"])
                ok = th["direct"] == t["direct"] and th["verdict"] == t["verdict"]
        if not ok:
            bad.append(name)
    for case, js in zip(measured, result["measured"]):
        found, fits = xrdfit.compare_measured(P, case["tt"], case["I"], case["frac"],
                                              case["radiation"], case["zero"])
        name = "measured " + case["name"]
        ok = (len(found["peaks"]) == len(js["peaks"])
              and [f["proto"] for f in fits] == [f["proto"] for f in js["fits"]])
        if ok:
            for u, v in zip(found["peaks"], js["peaks"]):
                worst("fit", u["tt"], v["tt"])
            for f, g in zip(fits, js["fits"]):
                ok = ok and (f["a"] is None) == (g["a"] is None) and f["n"] == g["n"]
                if ok and f["a"] is not None:
                    for u, v in ((f["a"], g["a"]), (f["zero"], g["zero"]), (f["rms"], g["rms"]),
                                 (f["score"], g["score"]), (f["density_g_cm3"], g["rho"])):
                        worst("fit", u, v)
                    ok = ([m["hkl"] for m in f["matches"]] == g["hkl"]
                          and len(f["unindexed"]) == len(g["unindexed"]))
        if not ok:
            bad.append(name)
    return len(fracs), bad, gap, (len(texts), rows_off, len(profiled), len(measured))


def measured_cases(P):
    """Synthetic laboratory patterns of known lattice parameter, built on the Python side
    and handed to both engines: one per cubic prototype, with K-alpha2, background, noise
    and a zero shift, one with a foreign phase, one with cobalt radiation."""
    eq = lambda *els: {e: 1 / len(els) for e in sorted(els)}  # noqa: E731
    mix = lambda d: {e: v / sum(d.values()) for e, v in sorted(d.items())}  # noqa: E731
    specs = [
        ("spinelle", "spinel", eq("Co", "Cr", "Fe", "Mn", "Ni"), 8.328, {}),
        ("rocksalt", "rocksalt", eq("Co", "Cu", "Mg", "Ni", "Zn"), 4.236, {"zero": 0.06}),
        ("fluorine", "fluorite", eq("Ce", "Hf", "Sn", "Ti", "Zr"), 5.25, {"fwhm": 0.25}),
        ("pyrochlore", "pyrochlore",
         mix({"La": 1, "Nd": 1, "Sm": 1, "Eu": 1, "Gd": 1, "Zr": 5}), 10.62, {}),
        ("perovskite", "perovskite",
         mix({"La": 5, "Cr": 1, "Mn": 1, "Fe": 1, "Co": 1, "Ni": 1}), 3.89,
         {"counts": 800.0, "background": 400.0, "fwhm": 0.3, "step": 0.04}),
        ("spinelle + phase etrangere", "spinel", eq("Al", "Cu", "Fe", "Ni", "Zn"), 8.2453,
         {"extra": [(33.15, 12.0), (49.5, 8.0)]}),
        ("spinelle, cobalt", "spinel", eq("Co", "Cr", "Fe", "Mn", "Ni"), 8.328,
         {"radiation": "Co", "tt_min": 20.0, "tt_max": 90.0}),
    ]
    out = []
    for name, proto, frac, a, kw in specs:
        tt, counts = xrdfit.synthetic_pattern(P, proto, frac, a, **kw)
        out.append({"name": name, "tt": tt, "I": counts, "frac": frac,
                    "radiation": kw.get("radiation", "Cu"), "zero": True})
    return out


def max_abs(a, b):
    return max(abs(u - v) for ra, rb in zip(a, b) for u, v in zip(ra, rb))


def main(check=False):
    bundle = json.load(open(os.path.join(ROOT, "app", "heo_model.json"), encoding="utf-8"))
    elements = features.load_elements()

    py_probs, py_feats = python_side(bundle, elements)
    js = js_side()
    n_formulas, formula_mismatch, d_formula = formula_parity(elements)
    n_pred, pred_mismatch, gap, (n_rows, rows_off, n_prof, n_meas) = prediction_parity()
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
        f"prediction complete, src/predict.py vs heo_engine.js, {n_pred} compositions "
        f"(base + probe) : {len(pred_mismatch)} desaccord(s) de famille, de domaine, de "
        f"modele de maille ou d'indices hkl",
        f"  ecarts maximaux : probabilite {gap['probs']:.1e} (tolerance {TOL_JS:.0e}), "
        f"distance au domaine {gap['ad']:.1e}, maille {gap['a']:.1e} A, "
        f"densite {gap['density']:.1e} g/cm3,",
        f"  position des pics {gap['two_theta']:.1e} deg, intensite {gap['intensity']:.1e}, "
        f"thermodynamique {gap['dh']:.1e} (tolerance {TOL_PRED:.0e})",
        f"  ligne de tableau (export CSV de la page et de tools/heo_predict.py), {n_rows} formules : "
        f"{len(rows_off)} ligne(s) differente(s), caractere pour caractere",
        f"  profil continu du diffractogramme, {n_prof} compositions : ecart maximal "
        f"{gap['profile']:.1e} (tolerance {TOL_PRED:.0e})",
        f"  diffractogramme mesure (pics, indexation, maille affinee), {n_meas} motifs "
        f"synthetiques : ecart maximal {gap['fit']:.1e} (tolerance {TOL_PRED:.0e})",
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
        (not pred_mismatch, "src/predict.py and heo_engine.js disagree on: "
                            + "; ".join(pred_mismatch[:5])),
        (not rows_off, "flat records differ between src/predict.py and heo_engine.js on: "
                       + "; ".join(repr(t) for t in rows_off[:5])),
        (gap["probs"] <= TOL_JS, f"full-prediction probabilities differ by {gap['probs']:.3e}"),
        (max(gap[k] for k in gap if k != "probs") <= TOL_PRED,
         "full prediction differs: " + ", ".join(f"{k} {v:.3e}" for k, v in gap.items())),
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
