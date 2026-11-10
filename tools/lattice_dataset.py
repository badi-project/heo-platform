#!/usr/bin/env python3
"""Measured lattice parameters: dataset check, inter-laboratory scatter, baseline.

Reads data/reference/lattice_measured.csv (one measured lattice parameter per line,
each with its source) and the protocol data/reference/lattice_protocol.md, written
before any computation of the new model. It writes:

- QC/lattice_dataset.txt — counts per family, the scatter between laboratories that
  measured the same composition (the floor below which no model can be judged), and the
  error of the v1.0 lattice model (the baseline to beat, criterion C3 of the
  protocol), recomputed here so that it stays fixed when the platform changes model;
- data/reference/lattice_dataset.json — the same numbers, machine-readable.

    python tools/lattice_dataset.py            # rewrite the report and the summary
    python tools/lattice_dataset.py --check    # exit 1 if they are not up to date or if
                                               # a line of the dataset is invalid
"""
import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402
import formula  # noqa: E402
import predict  # noqa: E402

DATA = os.path.join(ROOT, "data", "reference", "lattice_measured.csv")
REPORT = os.path.join(ROOT, "QC", "lattice_dataset.txt")
SUMMARY = os.path.join(ROOT, "data", "reference", "lattice_dataset.json")
FAMILIES = ("rocksalt", "spinel", "fluorite", "perovskite", "pyrochlore")
GROUPS = ("reference", "heo")
COLUMNS = ["id", "group", "proto", "formula_text", "a_ang", "abc_ang", "space_group",
           "phase_note", "synthesis", "source", "source_detail"]


def load(path=DATA):
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == COLUMNS, f"columns {reader.fieldnames}"
        return list(reader)


def a_measured(row):
    """Cubic parameter, or pseudo-cubic (abc/4)^(1/3) of an orthorhombic Pbnm cell."""
    if row["a_ang"]:
        return float(row["a_ang"])
    a, b, c = (float(v) for v in row["abc_ang"].split(";"))
    return (a * b * c / 4) ** (1 / 3)


def fractions(text, elements):
    """Normalised cation fractions of a formula. Unlike Predictor.read, a simple oxide
    (one to three cations) is accepted: the references are simple oxides."""
    raw = formula.cation_fractions(text, elements)
    if not raw or any(e not in elements for e in raw):
        raise ValueError(f"formule non lue : {text}")
    tot = features.nsum(raw[e] for e in sorted(raw))
    return {e: raw[e] / tot for e in sorted(raw)}


def validate(rows, P):
    """List of problems; empty when every line is usable."""
    problems, seen = [], set()
    for r in rows:
        rid = r["id"]
        if rid in seen:
            problems.append(f"{rid}: identifiant en double")
        seen.add(rid)
        if r["group"] not in GROUPS:
            problems.append(f"{rid}: groupe inconnu {r['group']}")
        if r["proto"] not in FAMILIES:
            problems.append(f"{rid}: prototype inconnu {r['proto']}")
        if bool(r["a_ang"]) == bool(r["abc_ang"]):
            problems.append(f"{rid}: il faut a_ang ou abc_ang, pas les deux ni aucun")
        if r["abc_ang"] and len(r["abc_ang"].split(";")) != 3:
            problems.append(f"{rid}: abc_ang doit contenir a;b;c")
        if not r["source"] or not r["source_detail"]:
            problems.append(f"{rid}: source manquante")
        if not r["phase_note"]:
            problems.append(f"{rid}: phase non renseignee")
        try:
            frac = fractions(r["formula_text"], P.elements)
        except ValueError as exc:
            problems.append(f"{rid}: formule illisible ({exc})")
            continue
        a = a_measured(r)
        lo, hi = {"rocksalt": (3.9, 4.6), "spinel": (7.9, 8.8), "fluorite": (4.9, 5.7),
                  "perovskite": (3.7, 4.3), "pyrochlore": (9.9, 10.9)}[r["proto"]]
        if not lo <= a <= hi:
            problems.append(f"{rid}: maille {a:.4f} A hors de [{lo}, {hi}] pour {r['proto']}")
        n = len(frac)
        if r["group"] == "heo" and n < 4:
            problems.append(f"{rid}: {n} cations pour un oxyde a haute entropie")
        if r["group"] == "reference" and n > 3:
            problems.append(f"{rid}: {n} cations pour une reference")
    return problems


def comp_key(proto, frac):
    """Unique composition: prototype + cation fractions rounded to 0.001."""
    return proto + ":" + ",".join(f"{e}{frac[e]:.3f}" for e in sorted(frac))


def median(values):
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def unique_compositions(rows, P, single_only=True):
    """{key: {proto, group, frac, a (median), rows}} in first-appearance order."""
    out = {}
    for r in rows:
        if single_only and r["phase_note"] != "single":
            continue
        frac = fractions(r["formula_text"], P.elements)
        k = comp_key(r["proto"], frac)
        if k not in out:
            out[k] = {"proto": r["proto"], "group": r["group"], "frac": frac,
                      "formula": r["formula_text"], "rows": []}
        out[k]["rows"].append(r)
    for c in out.values():
        c["a"] = median([a_measured(r) for r in c["rows"]])
    return out


def origin(row):
    """Who measured the value. A compilation (one DOI, several cited papers) counts each
    cited paper once; two samples of the same paper count as one origin."""
    m = re.search(r"ref \[(\d+)\]", row["source_detail"])
    return row["source"] + (f" ref {m.group(1)}" if m else "")


def scatter(comps):
    """Inter-laboratory scatter: compositions measured by at least two origins."""
    out = []
    for c in comps.values():
        if len({origin(r) for r in c["rows"]}) < 2:
            continue
        vals = [a_measured(r) for r in c["rows"]]
        m = c["a"]
        dev = features.nsum([abs(v - m) for v in vals]) / len(vals) / m * 100
        out.append({"proto": c["proto"], "formula": c["formula"], "n": len(vals),
                    "sources": len({origin(r) for r in c["rows"]}), "median": m,
                    "min": min(vals), "max": max(vals),
                    "range_pct": (max(vals) - min(vals)) / m * 100, "mad_pct": dev})
    return out


_V1 = None


def v1_geometric(proto, frac, E):
    """The geometric lattice model of v1.0, frozen here as the baseline to beat: one
    Shannon radius per element (element_properties.json), mean over the relevant site,
    times a constant calibrated on one compound per prototype (lattice_model.json)."""
    global _V1
    if _V1 is None:
        with open(os.path.join(ROOT, "data", "reference", "lattice_model.json"),
                  encoding="utf-8") as fh:
            _V1 = json.load(fh)
    kind, r_o = _V1["geo"][proto]

    def mean(key, f):
        return features.wsum([E[e][key] for e in f], [f[e] for e in f])
    if kind == "mean_VI":
        r = mean("r_shannon_VI", frac)
    elif kind == "mean_IV":
        r = mean("r_shannon_IV", frac)
    elif kind == "mean_VIII":
        r = mean("r_shannon_VIII", frac)
    else:
        A, B = features.ab_split(frac, E)
        r = mean("r_shannon_VI", B) if kind == "rB_VI" else mean("r_shannon_VIII", A)
    return _V1["kcal"][proto] * (r + r_o)


def current_model(P, proto, frac):
    """(geometric a, platform a, platform source) of the v1.0 model: CHGNet for the
    covered spinels, the v1.0 geometric model otherwise."""
    geo = v1_geometric(proto, frac, P.elements)
    th = P.thermo_lattice(proto, frac)
    return geo, (th if th is not None else geo), ("chgnet" if th is not None else "geometric")


def stats(errs):
    if not errs:
        return None
    n = len(errs)
    return {"n": n, "mae_pct": features.nsum([abs(e) for e in errs]) / n,
            "bias_pct": features.nsum(errs) / n, "max_pct": max(abs(e) for e in errs)}


def build():
    P = predict.Predictor()
    rows = load()
    problems = validate(rows, P)
    comps = unique_compositions(rows, P)
    flagged = [r for r in rows if r["phase_note"] != "single"]
    lines = ["Mailles mesurees : jeu de donnees, dispersion entre laboratoires, modele actuel",
             "protocole (ecrit avant le calcul du nouveau modele) : data/reference/lattice_protocol.md",
             "donnees : data/reference/lattice_measured.csv",
             "",
             "1. Contenu",
             "",
             "   famille       references (lignes / compositions)   validation HEO (lignes / compositions)"]
    counts = {}
    for f in FAMILIES:
        c = {}
        for g in GROUPS:
            c[g] = {"rows": sum(1 for r in rows if r["proto"] == f and r["group"] == g),
                    "compositions": sum(1 for k in comps.values()
                                        if k["proto"] == f and k["group"] == g)}
        counts[f] = c
        lines.append(f"   {f:<12s}  {c['reference']['rows']:>5d} / {c['reference']['compositions']:<5d}"
                     f"                         {c['heo']['rows']:>5d} / {c['heo']['compositions']:<5d}")
    n_src = len({r["source"] for r in rows})
    lines += ["",
              f"   {len(rows)} lignes, {n_src} sources ; compositions uniques (une seule phase declaree) : "
              f"{len(comps)}",
              f"   lignes ecartees de la metrique principale (phase non pure) : "
              + ", ".join(f"{r['id']} ({r['phase_note']})" for r in flagged),
              ""]
    if problems:
        lines += ["   PROBLEMES :"] + [f"   - {p}" for p in problems] + [""]
    else:
        lines += ["   controle des lignes : aucune anomalie (formule lue, maille plausible, source,"
                  " groupe coherent avec le nombre de cations)", ""]

    sc = scatter(comps)
    lines += ["2. Dispersion entre laboratoires (compositions mesurees par au moins deux origines ; une",
              "   compilation compte chaque article cite, deux echantillons d'un meme article comptent une fois)",
              "",
              "   famille      composition                                    n  mediane   min      max      etendue  ecart moyen"]
    for s in sc:
        lines.append(f"   {s['proto']:<11s}  {s['formula'][:45]:<45s} {s['n']:>2d}  {s['median']:7.4f}  "
                     f"{s['min']:7.4f}  {s['max']:7.4f}  {s['range_pct']:5.2f} %   {s['mad_pct']:5.2f} %")
    floor = {}
    for f in FAMILIES:
        v = [s["mad_pct"] for s in sc if s["proto"] == f]
        floor[f] = {"n": len(v), "mad_pct": (features.nsum(v) / len(v)) if v else None}
    lines += ["",
              "   plancher par famille (moyenne de l'ecart moyen a la mediane) : "
              + " ; ".join(f"{f} {floor[f]['mad_pct']:.2f} % ({floor[f]['n']})" if floor[f]["n"]
                           else f"{f} -" for f in FAMILIES),
              "   lecture : sous ce plancher, l'erreur d'un modele n'est plus distinguable de l'ecart",
              "   entre deux syntheses de la meme composition (temperature, atmosphere, stoechiometrie",
              "   en oxygene, taille des cristallites).",
              ""]

    lines += ["3. Modele de maille de la v1.0 (reference a battre, critere C3)",
              "   geometrique : rayon moyen x constante calibree sur un compose par prototype",
              "   (fige dans data/reference/lattice_model.json, recalcule ici independamment de src/predict.py)",
              "   plateforme  : CHGNet pour les spinelles a cations couverts, geometrique sinon",
              "",
              "   groupe     famille      n   geometrique : moyen   biais    max     plateforme : moyen   biais    max"]
    base = {}
    detail = []
    for g in GROUPS:
        base[g] = {}
        for f in FAMILIES:
            eg, ep = [], []
            for c in comps.values():
                if c["proto"] != f or c["group"] != g:
                    continue
                geo, plat, src = current_model(P, f, c["frac"])
                e1 = (geo - c["a"]) / c["a"] * 100
                e2 = (plat - c["a"]) / c["a"] * 100
                eg.append(e1)
                ep.append(e2)
                if g == "heo":
                    detail.append((f, c["formula"], c["a"], geo, plat, src, e1, e2))
            sg, sp = stats(eg), stats(ep)
            base[g][f] = {"geometric": sg, "platform": sp}
            if sg:
                lines.append(f"   {g:<9s}  {f:<11s} {sg['n']:>2d}         {sg['mae_pct']:6.2f} %  {sg['bias_pct']:+6.2f} %  "
                             f"{sg['max_pct']:5.2f} %          {sp['mae_pct']:6.2f} %  {sp['bias_pct']:+6.2f} %  {sp['max_pct']:5.2f} %")
        allg = [e for c in comps.values() if c["group"] == g
                for e in [(current_model(P, c["proto"], c["frac"])[0] - c["a"]) / c["a"] * 100]]
        allp = [e for c in comps.values() if c["group"] == g
                for e in [(current_model(P, c["proto"], c["frac"])[1] - c["a"]) / c["a"] * 100]]
        base[g]["all"] = {"geometric": stats(allg), "platform": stats(allp)}
        lines.append(f"   {g:<9s}  {'toutes':<11s} {len(allg):>2d}         {stats(allg)['mae_pct']:6.2f} %  "
                     f"{stats(allg)['bias_pct']:+6.2f} %  {stats(allg)['max_pct']:5.2f} %          "
                     f"{stats(allp)['mae_pct']:6.2f} %  {stats(allp)['bias_pct']:+6.2f} %  {stats(allp)['max_pct']:5.2f} %")
    lines += ["",
              "   detail, validation HEO (maille mesuree = mediane des lignes de la composition) :",
              "   famille      composition                                    mesure   geometr.  plateforme  ecart geo  ecart plat."]
    for f, formula, a, geo, plat, src, e1, e2 in detail:
        lines.append(f"   {f:<11s}  {formula[:45]:<45s} {a:7.4f}  {geo:8.4f}  {plat:8.4f} {src[:3]}  "
                     f"{e1:+6.2f} %   {e2:+6.2f} %")
    lines += ["",
              "genere par tools/lattice_dataset.py"]

    def rnd(x):
        if isinstance(x, float):
            return round(x, 3)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    summary = {"n_rows": len(rows), "n_sources": n_src, "n_compositions": len(comps),
               "counts": counts, "floor": rnd(floor), "baseline": rnd(base),
               "excluded": [r["id"] for r in flagged], "problems": problems}
    return "\n".join(lines) + "\n", json.dumps(summary, indent=1) + "\n", not problems


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
                  + " (python tools/lattice_dataset.py)")
        if not ok:
            print("le jeu de donnees contient des lignes invalides (voir le rapport)")
        sys.exit(1 if stale or not ok else 0)
    for p, txt in ((REPORT, report), (SUMMARY, summary)):
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(txt)
    print(report)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
