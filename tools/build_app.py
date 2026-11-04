#!/usr/bin/env python3
"""Regenerate the data payload of app/heo_designer.html.

The page is a single self-contained file. Its JavaScript (heo_engine.js, heo_ui.js) is
re-inlined by tools/sync_app.py; this script rewrites everything else that depends on the
data: the payload (element table, model, applicability domain, database, candidates),
the header line, and the "Méthode & limites" text, whose numbers are filled from the
outputs of the pipeline instead of being typed by hand.

    python tools/build_app.py            # rewrite app/heo_designer.html
    python tools/build_app.py --check    # exit 1 if the page is not up to date

Run after tools/train_model.py and tools/screen.py.
"""
import argparse
import csv
import gzip
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import sync_app  # noqa: E402

HTML = os.path.join(ROOT, "app", "heo_designer.html")
LATTICE_MAE_PCT = 1.93
REPORT_FEATS = ["s_config", "delta_r", "delta_X_rel", "mean_ox", "redox_flex", "r_A", "r_B",
                "goldschmidt_t", "octahedral_factor", "r_shannon_VI_mean"]
DESC_FR = {
    "s_config": "entropie configurationnelle S/R",
    "delta_r": "désaccord de taille δr",
    "delta_X_rel": "désaccord d'électronégativité δχ/χ̄",
    "mean_ox": "état d'oxydation moyen",
    "redox_flex": "flexibilité rédox (états accessibles)",
    "r_A": "rayon moyen site A (CN XII, Å)",
    "r_B": "rayon moyen site B (CN VI, Å)",
    "goldschmidt_t": "facteur de tolérance t",
    "octahedral_factor": "facteur octaédrique r_B/r_O",
    "r_shannon_VI_mean": "rayon cationique moyen CN VI (Å)",
}
DESC_DEC = {"s_config": 3, "delta_r": 4, "delta_X_rel": 4, "mean_ox": 2, "redox_flex": 2,
            "r_A": 3, "r_B": 3, "goldschmidt_t": 3, "octahedral_factor": 3,
            "r_shannon_VI_mean": 3}
PROTO_FR = {"spinel": "spinelle", "perovskite": "pérovskite", "rocksalt": "rocksalt",
            "fluorite": "fluorine", "pyrochlore": "pyrochlore", "layered": "lamellaire"}


def read_json(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8") as fh:
        return json.load(fh)


def read_csv(*p):
    with open(os.path.join(ROOT, *p), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fr_num(v, n=3):
    return f"{v:.{n}f}".replace(".", ",")


def pct(v):
    return f"{100 * v:.0f} %"


def thousands(n):
    return f"{n:,}".replace(",", " ")


def screening_stats():
    total = quin = sen = novel_in = 0
    with gzip.open(os.path.join(ROOT, "data", "screening_all.csv.gz"), "rt",
                   encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            total += 1
            quin += r["n_cat"] == "5"
            sen += r["n_cat"] == "6"
            novel_in += r["novel"] == "1" and r["in_domain"] == "1"
    rep = open(os.path.join(ROOT, "QC", "screening_report.txt"), encoding="utf-8").read()
    pool = int(re.search(r"vivier : (\d+) cations", rep).group(1))
    cap = int(re.search(r"au plus (\d+) par famille", rep).group(1))
    return {"total": total, "quinaries": quin, "senaries": sen, "novel_in_domain": novel_in,
            "pool": pool, "cap": cap}


def thermo_payload():
    """Step-1/2 results for the page (thermo/): the per-cation surrogates and the values
    computed directly with CHGNet. None when thermo/ has not been run."""
    d = os.path.join(ROOT, "data", "thermo")
    if not os.path.exists(os.path.join(d, "oxidation_surrogate.json")):
        return None
    s = read_json("data", "thermo", "oxidation_surrogate.json")

    def covered(block):
        # a cation seen in a single calculated system has an undetermined weight; some
        # cations are outside the chemistry the descriptor describes (not_covered)
        n = block.get("n_systems_per_element") or [2] * len(block["elements"])
        out = block.get("not_covered") or {}
        return [e for e, k in zip(block["elements"], n) if k >= 2 and e not in out]
    out = {"ox": {"elements": s["elements"], "cov": covered(s), "h": s["h_eV_per_O"],
                  "mae": s["loo_mae"],
                  "b0": s["logistic"]["intercept"], "b1": s["logistic"]["slope"],
                  "switch": s["logistic"]["switch_eV_per_O"], "auc": s["auc_system"],
                  "n": s["n_systems"],
                  "lco_comp": s["leave_cation_out"]["composition"],
                  "lco_switch": s["leave_cation_out"]["aiguillage"]},
           "lat": None, "systems": {}}
    lat = s.get("spinel_lattice")
    if lat:
        out["lat"] = {"elements": lat["elements"], "cov": covered(lat), "a": lat["a_i_A"],
                      "k": lat["k"], "mae_pct": lat["measured_mae_pct"]}
    k = read_json("data", "thermo", "spinel_calibration.json")["k"]
    for r in read_csv("data", "thermo", "oxidation_descriptor.csv"):
        out["systems"][r["system"]] = {"dh": round(float(r["dH_ox_corr_eV_per_O"]), 3),
                                       "a": round(float(r["a_sp_A"]) * k, 3),
                                       "frac": json.loads(r["frac"])}
    bench = read_csv("data", "thermo", "rocksalt_benchmark.csv")
    if bench:
        agg = {}
        for r in bench:
            agg.setdefault(r["system"], []).append((float(r["dH_mix_meV_atom"]),
                                                    float(r["sigma_bonds_A"])))
        th = read_json("data", "thermo", "rocksalt_thresholds.json")
        th_h, th_s = th["dH_mix_meV_atom"], th["sigma_A"]
        out["rs_threshold"] = {"dh": th_h, "sigma": th_s}
        for sysk, v in agg.items():
            h_ = sum(x for x, _ in v) / len(v)
            s_ = sum(y for _, y in v) / len(v)
            e = out["systems"].setdefault(sysk, {})
            e["rs"] = {"dh": round(h_, 1), "sigma": round(s_, 3),
                       "single": bool(h_ <= th_h + 1e-9 and s_ <= th_s + 1e-9)}
    if os.path.exists(os.path.join(d, "vacancy_heo.csv")):
        for r in read_csv("data", "thermo", "vacancy_heo.csv"):
            e = out["systems"].setdefault(r["system"], {})
            e["ev"] = round(float(r["E_vac_mean_eV"]), 2)
            e["evmin"] = round(float(r["E_vac_min_eV"]), 2)
    return out


def thermo_fill(th):
    """Numbers of the thermodynamics / OER / active-learning paragraphs, read from the
    reports of thermo/ and tools/active_learning.py."""
    def grab(path, pat, cast=str):
        p = os.path.join(ROOT, *path)
        if not os.path.exists(p):
            return None
        m = re.search(pat, open(p, encoding="utf-8").read())
        return cast(m.group(1)) if m else None
    f = {k: "—" for k in ("th_elements", "th_n", "th_auc", "th_mae", "lat_mae", "rs_n", "rs_hits",
                          "rs_h5", "rs_cal", "oer_sd", "oer_icc", "al_saving", "rs_loo",
                          "lco_comp", "lco_switch", "vac_rho")}
    if th:
        f["th_elements"] = " ".join(th["ox"].get("cov") or th["ox"]["elements"])
        f["lco_comp"] = fr_num(100 * th["ox"]["lco_comp"], 1) + " %"
        f["lco_switch"] = fr_num(100 * th["ox"]["lco_switch"], 1) + " %"
        f["th_n"] = th["ox"]["n"]
        f["th_auc"] = fr_num(th["ox"]["auc"], 2)
        f["th_mae"] = fr_num(th["ox"]["mae"], 2)
        if th["lat"] and th["lat"]["mae_pct"] is not None:
            f["lat_mae"] = fr_num(th["lat"]["mae_pct"], 2)
    bench = ("QC", "thermo_rocksalt_benchmark.txt")
    hits = grab(bench, r"issues experimentales retrouvees : (\d+ / \d+)")
    if hits:
        f["rs_hits"] = hits.replace(" / ", " sur ")
        f["rs_n"] = hits.split(" / ")[1]
    loo = grab(bench, r"en un-contre-tous \(seuils lus sur les autres monophases\) : (\d+ / \d+)")
    if loo:
        f["rs_loo"] = loo.replace(" / ", " sur ")
    rho = grab(("QC", "thermo_vacancy.txt"), r"correlation de rang \(Spearman\) : ([0-9.]+) corrigee")
    if rho:
        f["vac_rho"] = fr_num(float(rho), 2)
    h5 = grab(bench, r"\(MgCoNiCuZn\)O : (\d+) meV/atome ici")
    cal = grab(bench, r"(\d+) mesure par calorimetrie")
    f["rs_h5"], f["rs_cal"] = h5 or "—", cal or "—"
    oer = ("QC", "oer_literature_variance.txt")
    sd = grab(oer, r"composition egale \(meme systeme, autres labos\) : (\d+) mV")
    icc = grab(oer, r"composition\) : ([0-9.]+ \[IC 95 % [0-9.]+-[0-9.]+\])")
    f["oer_sd"] = sd or "—"
    f["oer_icc"] = icc.replace(".", ",").replace("-", " à ") if icc else "—"
    sav = grab(("QC", "active_learning.txt"), r"(\d+) % de syntheses en moins")
    f["al_saving"] = f"{sav} %" if sav else "—"
    return f


def build(check=False):
    bundle = read_json("app", "heo_model.json")
    elem = read_json("data", "reference", "element_properties.json")
    latt = read_json("data", "reference", "lattice_model.json")
    xrd = read_json("data", "reference", "xrd_prototypes.json")
    db = read_csv("data", "heo_database.csv")
    cands = read_csv("data", "heo_candidats.csv")
    M = bundle["metrics"]
    scr = screening_stats()
    phases = Counter(r["phase"] for r in db)

    payload = {
        "ELEM": elem, "PROPS": bundle["props"], "ONEHOT": bundle["onehot"],
        "FEATNAMES": bundle["featnames"], "MODEL": bundle["model"], "AD": bundle["ad"],
        "LATTICE": {**latt, "mae_pct": LATTICE_MAE_PCT}, "XRD": xrd,
        "DB": [{"c": r["cations"], "p": r["proto"], "s": r["phase"],
                "d": r["doi"] if r["doi"] else None} for r in db],
        "DBSET": dict(Counter(r["sysk"] for r in db)),
        "DBCOUNT": dict(Counter(e for r in db for e in r["cations"].split("-"))),
        "CATIONS": sorted(set(e for r in db for e in r["cations"].split("-"))),
        "CANDS": [{"c": c["cations"], "t": c["proto"], "p": round(float(c["p"]), 4),
                   "ad": round(float(c["ad"]), 3), "gt": round(float(c["gold_t"]), 4),
                   "dr": round(float(c["delta_r"]), 4), "sc": round(float(c["s_config"]), 3),
                   "n": int(c["n_cat"]), "new": int(c["novel"]), "in": int(c["in_domain"]),
                   "a": round(float(c["a_predit"]), 3) if c["a_predit"] else None}
                  for c in cands],
        "REPORT_FEATS": REPORT_FEATS, "DESC_FR": DESC_FR, "DESC_DEC": DESC_DEC,
        "SCREEN": {"total": scr["total"], "cap": scr["cap"]},
        "PHASES": {"single": phases["single"], "multi": phases["multi"]},
        "THERMO": thermo_payload(),
    }

    clean = open(os.path.join(ROOT, "QC", "database_cleaning.txt"), encoding="utf-8").read()
    grab = lambda pat: int(re.search(pat, clean).group(1))  # noqa: E731
    cv, t, ch, b = M["cv"], M["temporal"], M["leave_chemistry_out"], M["baselines"]
    years = sorted(int(r["year"]) for r in db)
    rel = cv["reliability"]
    fill = {
        "n_rows": thousands(len(db)),
        "n_pubs": thousands(len(set(r["doi"] for r in db if r["doi"]))),
        "n_systems": thousands(len(set(r["sysk"] for r in db))),
        "n_cations": len(payload["CATIONS"]),
        "n_single": phases["single"], "n_multi": phases["multi"],
        "n_fixed": grab(r"ensemble de cations corrige : (\d+)"),
        "n_site": grab(r"dont cations ajoutes \(site oublie\) : (\d+)"),
        "n_dropped": grab(r"entrees ecartees : (\d+)"),
        "n_labeled": M["n_labeled"],
        "cv_acc": fr_num(cv["accuracy"]), "cv_f1": fr_num(cv["f1_macro"]),
        "t_cut": t["cut"], "t_test_years": f"{t['cut'] + 1}-{years[-1]}",
        "t_n_test": t["n_test"], "t_acc": fr_num(t["accuracy"]),
        "t_n_unseen": t["n_unseen"], "t_acc_unseen": fr_num(t["accuracy_unseen"]),
        "chem_acc": fr_num(ch["accuracy"]),
        "base_nearest": fr_num(b["nearest_system"]["accuracy"]),
        "base_presence": fr_num(b["cation_presence"]["accuracy"]),
        "base_classic": fr_num(b["classic_9"]["accuracy"]),
        "base_majority": fr_num(b["majority"]["accuracy"]),
        "bin_model": fr_num(M["perovskite_binary"]["model"]),
        "bin_gold": fr_num(M["perovskite_binary"]["goldschmidt"]),
        "f1_per_class": ", ".join(f"{PROTO_FR[c]} {fr_num(v, 2)}"
                                  for c, v in cv["f1_per_class"].items()),
        "rs_as_sp": cv["rocksalt_as_spinel"], "sp_as_rs": cv["spinel_as_rocksalt"],
        "temperature": fr_num(cv["temperature"], 2),
        "ece_raw": fr_num(cv["ece_raw"]), "ece_cal": fr_num(cv["ece_cal"]),
        "rel_hi": pct(rel["0.9-1.0"]["accuracy"]), "rel_mid": pct(rel["0.6-0.7"]["accuracy"]),
        "ad_threshold": fr_num(bundle["ad"]["threshold"], 2),
        "n_lattice": sum(1 for r in db if r["a_ang"]),
        "lattice_mae": fr_num(LATTICE_MAE_PCT, 1),
        "screen_total": thousands(scr["total"]), "screen_quin": thousands(scr["quinaries"]),
        "screen_sen": thousands(scr["senaries"]), "screen_pool": scr["pool"],
        "screen_novel_in": thousands(scr["novel_in_domain"]),
    }
    fill.update(thermo_fill(payload["THERMO"]))
    method = open(os.path.join(ROOT, "app", "method_template.html"), encoding="utf-8").read()
    for k, v in fill.items():
        method = method.replace("{{" + k + "}}", str(v))
    missing = re.findall(r"\{\{(\w+)\}\}", method)
    if missing:
        raise SystemExit("method_template.html: unfilled " + ", ".join(missing))

    html = open(HTML, encoding="utf-8").read()
    header = (f"Modèle entraîné sur {fill['n_rows']} échantillons extraits de {fill['n_pubs']} "
              "publications, stœchiométrie relue dans les formules. Prédit la famille "
              "structurale et le diffractogramme attendu — pas l'activité catalytique.")
    html = re.sub(r'(<p data-gen="header">).*?(</p>)', lambda m: m.group(1) + header + m.group(2),
                  html, count=1, flags=re.S)
    # the neighbour-table footnote quotes the publication count too
    html = re.sub(r"(n'apparaît pas dans les )[0-9  ]+( publications récoltées)",
                  lambda m: m.group(1) + fill["n_pubs"] + m.group(2), html, count=1)
    start = html.index("const PAYLOAD = ")
    end = html.index("const METHOD_HTML = ")
    end = html.index("\n", json.JSONDecoder().raw_decode(html, end + len("const METHOD_HTML = "))[1])
    names = list(payload)
    decl = ("const PAYLOAD = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            + ";\nconst " + ", ".join(f"{n}=PAYLOAD.{n}" for n in names) + ";\n"
            + "const METHOD_HTML = " + json.dumps(method, ensure_ascii=True) + ";")
    html = html[:start] + decl + html[end:]

    current = open(HTML, encoding="utf-8").read()
    if check:
        if current != html:
            print("app/heo_designer.html is not up to date: run tools/build_app.py")
            return 1
        print("app/heo_designer.html payload is up to date")
        return 0
    with open(HTML, "w", encoding="utf-8") as fh:
        fh.write(html)
    sync_app.sync()
    print(f"payload rewritten: {len(db)} entries, {len(cands)} candidates, "
          f"{os.path.getsize(HTML) / 1e6:.2f} MB")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only, do not rewrite")
    sys.exit(build(check=ap.parse_args().check))
