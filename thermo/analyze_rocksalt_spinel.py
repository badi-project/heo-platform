#!/usr/bin/env python3
"""Step 1b, part 3 — does the computed oxidation enthalpy settle rock salt vs spinel?

Three questions, answered on the database labels (never used to compute dH_ox):

1. Separation: does dH_ox rank the systems reported as rock salt above those reported
   as spinel (ROC AUC, system level and entry level)?
2. Usefulness: added to the composition classifier on the rock-salt / spinel entries,
   does it remove confusions on held-out cation systems (grouped cross-validation)?
3. Speed: can a per-cation additive surrogate, dH_ox ~ sum_i x_i h_i, reproduce the
   CHGNet value (leave-one-system-out), so that the web page computes it instantly
   for any composition of these cations?

Writes QC/thermo_rocksalt_spinel.txt and data/thermo/oxidation_surrogate.json.
"""
import csv
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DESC = os.path.join(ROOT, "data", "thermo", "oxidation_descriptor.csv")
FEAT = os.path.join(ROOT, "data", "heo_features.csv")
DB = os.path.join(ROOT, "data", "heo_database.csv")
REPORT = os.path.join(ROOT, "QC", "thermo_rocksalt_spinel.txt")
SURR = os.path.join(ROOT, "data", "thermo", "oxidation_surrogate.json")
LAT = os.path.join(ROOT, "data", "thermo", "spinel_calibration.json")
# cations for which 3 MO + O -> M3O4 does not describe the chemistry: the web page shows
# them as not covered. Li+ is monovalent, so its "MO" rock salt is not charge balanced;
# decided after the leave-one-cation-out test showed the failure (see the report)
NOT_COVERED = {"Li": "monovalent : la reaction 3 MO + O -> M3O4 ne s'applique pas"}
KB = 8.617333e-5


def auc_rs(y_is_rs, score):
    """AUC of 'higher score = rock salt'."""
    return roc_auc_score(y_is_rs, score)


def surrogate(desc, key):
    """Additive per-cation model y ~ sum_i x_i h_i, judged leave-one-system-out.

    A system holding a cation that appears in no other system cannot be predicted once it
    is left out (its h_i is then undetermined, and lstsq sets it to 0 — an error of
    x_i * |h_i|, i.e. 1.7 A on a lattice parameter). Such systems are left out of the
    score and counted; the web page treats the same cations as not covered."""
    els = sorted({e for d in desc for e in json.loads(d["frac"])})
    X = np.array([[json.loads(d["frac"]).get(e, 0.0) for e in els] for d in desc])
    y = np.array([float(d[key]) for d in desc])
    h, *_ = np.linalg.lstsq(X, y, rcond=None)
    n_sys = (X > 0).sum(0)
    scorable = [i for i in range(len(y)) if all(n_sys[X[i] > 0] >= 2)]
    loo = []
    for i in scorable:
        m = np.arange(len(y)) != i
        hi, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
        loo.append(X[i] @ hi - y[i])
    loo = np.array(loo)
    ys = y[scorable]
    r2 = 1 - (loo ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return {"els": els, "h": h, "mae": np.abs(loo).mean(), "r2": r2, "fit": X @ h,
            "n_sys": [int(v) for v in n_sys], "n_scored": len(scorable),
            "base_mae": np.abs(ys - ys.mean()).mean()}


class Average:
    """Mean of the rock-salt probabilities of the composition model and of the
    one-variable logistic model on dH_ox (last column). No fitted weight."""

    def __init__(self):
        self.a = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15,
                                                class_weight="balanced", random_state=0)
        self.b = LogisticRegression(class_weight="balanced")

    def fit(self, X, y):
        self.a.fit(X[:, :-1], y)
        self.b.fit(X[:, -1:], y)
        return self

    def predict_proba(self, X):
        return (self.a.predict_proba(X[:, :-1]) + self.b.predict_proba(X[:, -1:])) / 2

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


AD_FEATS = ["goldschmidt_t", "octahedral_factor", "delta_r", "delta_X_rel", "r_A", "r_B",
            "mean_ox", "ox_spread", "redox_flex", "s_config", "n_cat",
            "r_shannon_VI_mean", "X_mean", "mass_mean", "melting_point_mean",
            "e_density_mean", "frac_RE", "frac_TM3d", "frac_AE", "frac_pblock"]


class Switch:
    """Composition model inside its applicability domain, dH_ox outside it — the rule of
    the web page (same 20 descriptors, k = 5, 95th percentile as tools/train_model.py)."""

    def __init__(self, ad_idx):
        self.ad_idx = ad_idx
        self.a = HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15,
                                                class_weight="balanced", random_state=0)
        self.b = LogisticRegression(class_weight="balanced")

    def _d(self, Z):
        d = np.sqrt(((Z[:, None, :] - self.Ztr[None, :, :]) ** 2).sum(-1))
        return np.sort(d, axis=1)[:, :5].mean(1)

    def fit(self, X, y):
        self.a.fit(X[:, :-1], y)
        self.b.fit(X[:, -1:], y)
        Z = X[:, self.ad_idx]
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9
        self.Ztr = (Z - self.mu) / self.sd
        d = np.sqrt(((self.Ztr[:, None, :] - self.Ztr[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        self.th = np.percentile(np.sort(d, axis=1)[:, :5].mean(1), 95)
        return self

    def predict(self, X):
        out = self._d((X[:, self.ad_idx] - self.mu) / self.sd) > self.th
        self.last_out = out
        pa = self.a.predict(X[:, :-1])
        pb = self.b.predict(X[:, -1:])
        return np.where(out, pb, pa)


def main():
    desc = list(csv.DictReader(open(DESC)))
    by = {d["system"]: d for d in desc}
    lines = ["Etape 1b - l'enthalpie d'oxydation calculee departage-t-elle rocksalt et spinelle ?",
             f"{len(desc)} systemes de la base calcules (cations Mg Ca Mn Fe Co Ni Cu Zn Cr Al "
             "Ga Li Ti V) ; dH_ox en eV par O gagne, 3 MO (rocksalt) + O -> M3O4 (spinelle)",
             ""]
    M = {}
    for key, name in (("dH_ox_eV_per_O", "brut"), ("dH_ox_corr_eV_per_O", "corrige")):
        sys_rs = np.array([int(d["n_rocksalt"]) > int(d["n_spinel"]) for d in desc])
        score = np.array([float(d[key]) for d in desc])
        a_sys = auc_rs(sys_rs, score)
        # entry level: each database entry carries its system's value
        ent_y, ent_s = [], []
        for d in desc:
            ent_y += [1] * int(d["n_rocksalt"]) + [0] * int(d["n_spinel"])
            ent_s += [float(d[key])] * (int(d["n_rocksalt"]) + int(d["n_spinel"]))
        a_ent = auc_rs(np.array(ent_y), np.array(ent_s))
        both = [d for d in desc if int(d["n_rocksalt"]) and int(d["n_spinel"])]
        M[key] = (a_sys, a_ent)
        lines.append(f"dH_ox {name:<8s}: AUC rocksalt/spinelle {a_sys:.3f} par systeme "
                     f"({int(sys_rs.sum())} rocksalt majoritaires sur {len(desc)}), "
                     f"{a_ent:.3f} par entree")
    lines.append("")
    lines.append("valeurs (eV/O, corrigees) par etiquette majoritaire :")
    for lab, cond in (("rocksalt", lambda d: int(d["n_rocksalt"]) > int(d["n_spinel"])),
                      ("spinelle", lambda d: int(d["n_rocksalt"]) <= int(d["n_spinel"]))):
        v = np.array([float(d["dH_ox_corr_eV_per_O"]) for d in desc if cond(d)])
        lines.append(f"  {lab:<9s} n = {len(v):3d}  mediane {np.median(v):+.2f}, "
                     f"quartiles {np.percentile(v, 25):+.2f} / {np.percentile(v, 75):+.2f}")
    v = np.array([float(d["dH_ox_corr_eV_per_O"]) for d in both])
    lines.append(f"  les deux   n = {len(v):3d}  mediane {np.median(v):+.2f}")
    lines.append("")

    # 1b. numerical noise of the protocol: cheap (used everywhere) vs tighter relaxation,
    # longer Monte Carlo, other random decorations
    chk = os.path.join(ROOT, "data", "thermo", "oxidation_protocol_check.csv")
    if os.path.exists(chk):
        ck = list(csv.DictReader(open(chk)))
        lines.append("Bruit du protocole (raw dH_ox, eV/O) : protocole rapide contre deux tirages "
                     "d'un protocole serre (fmax 0.05, 400 pas, 200 pas de Monte-Carlo) :")
        spreads = []
        for s_ in sorted({r["system"] for r in ck}):
            cheap = [float(r["dH_ox_eV_per_O"]) for r in ck
                     if r["system"] == s_ and r["protocol"] == "cheap"]
            tight = [float(r["dH_ox_eV_per_O"]) for r in ck
                     if r["system"] == s_ and r["protocol"] == "tight"]
            if cheap and tight:
                spreads.append(abs(cheap[0] - np.mean(tight)))
                lines.append(f"  {s_:<18s} rapide {cheap[0]:+.3f} ; serre {', '.join(f'{t:+.3f}' for t in tight)}"
                             f" ; ecart {cheap[0] - np.mean(tight):+.3f}")
        if spreads:
            lines.append(f"  ecart moyen rapide / serre {np.mean(spreads):.3f} eV/O, a comparer a "
                         f"l'ecart des medianes rocksalt / spinelle")
        lines.append("")

    # 2. usefulness inside the classifier ------------------------------------------
    feats = list(csv.DictReader(open(FEAT, encoding="utf-8")))
    names = [k for k in feats[0] if k not in ("doi", "sysk", "proto", "phase", "tier")]
    rows = [(f, by[f["sysk"]]) for f in feats
            if f["proto"] in ("rocksalt", "spinel") and f["sysk"] in by]
    X = np.array([[float(f[n]) for n in names] for f, _ in rows])
    xd = np.array([[float(d["dH_ox_corr_eV_per_O"])] for _, d in rows])
    y = np.array([f["proto"] == "rocksalt" for f, _ in rows]).astype(int)
    g = np.array([f["sysk"] for f, _ in rows])
    n_split = min(10, len(set(g)))
    ad_idx = [names.index(n) for n in AD_FEATS]

    def cv(make, Xm):
        pred = np.zeros(len(y), dtype=int)
        for tr, te in GroupKFold(n_splits=n_split).split(Xm, y, g):
            pred[te] = make().fit(Xm[tr], y[tr]).predict(Xm[te])
        rs_as_sp = int(((y == 1) & (pred == 0)).sum())
        sp_as_rs = int(((y == 0) & (pred == 1)).sum())
        return (pred == y).mean(), rs_as_sp, sp_as_rs
    hgb = lambda: HistGradientBoostingClassifier(max_iter=250, max_leaf_nodes=15,  # noqa
                                                 class_weight="balanced", random_state=0)
    lr = lambda: LogisticRegression(class_weight="balanced")  # noqa: E731
    res = {"composition (134 descripteurs)": cv(hgb, X),
           "composition + dH_ox": cv(hgb, np.hstack([X, xd])),
           "dH_ox seul (logistique)": cv(lr, xd),
           "aiguillage (domaine d'applicabilite)": cv(lambda: Switch(ad_idx), np.hstack([X, xd]))}
    lines.append(f"Classifieur rocksalt / spinelle, {len(y)} entrees ({int(y.sum())} rocksalt), "
                 f"validation croisee groupee par systeme ({n_split} plis) :")
    for k, (acc, a, b) in res.items():
        lines.append(f"  {k:<32s} exactitude {acc:.3f} ; rocksalt -> spinelle {a:3d}, "
                     f"spinelle -> rocksalt {b:3d}")
    lines.append("")

    # 2b. where the statistical model has no data: leave one cation out ---------------
    # every system containing cation c is held out; dH_ox needs no label, only CHGNet
    cats = [set(s.split("-")) for s in g]
    lines.append("Cation inconnu du modele (tous les systemes qui le contiennent retires) :")
    variants = (("composition", hgb, X), ("composition + dH_ox", hgb, np.hstack([X, xd])),
                ("dH_ox seul", lr, xd), ("aiguillage", lambda: Switch(ad_idx), np.hstack([X, xd])))
    tot = {v[0]: [0, 0] for v in variants}
    for c in sorted({e for s in cats for e in s}):
        te = np.array([c in s for s in cats])
        if te.sum() < 8 or (~te).sum() < 30 or len(set(y[~te])) < 2:
            continue
        accs = []
        for name, make, Xm in variants:
            p = make().fit(Xm[~te], y[~te]).predict(Xm[te])
            ok = int((p == y[te]).sum())
            tot[name][0] += ok
            tot[name][1] += int(te.sum())
            accs.append(ok / te.sum())
        lines.append(f"  sans {c:<3s} ({int(te.sum()):3d} entrees) : composition {accs[0]:.3f}, "
                     f"+ dH_ox {accs[1]:.3f}, dH_ox seul {accs[2]:.3f}, aiguillage {accs[3]:.3f}")
    acc_lco = {k: v[0] / v[1] for k, v in tot.items()}
    lines.append("  ensemble : " + ", ".join(f"{k} {v:.3f}" for k, v in acc_lco.items()))
    M["leave_cation_out"] = acc_lco
    lines.append("")

    # 2c. synthesis temperature, when the abstract gives it ---------------------------
    db = {r["doi"] + r["sysk"]: r for r in csv.DictReader(open(DB, encoding="utf-8"))}
    tc_rows = []
    for (f, d) in rows:
        r = db.get(f["doi"] + f["sysk"])
        if r and r["T_C"]:
            try:
                tc_rows.append((float(d["dH_ox_corr_eV_per_O"]), float(r["T_C"]),
                                f["proto"] == "rocksalt", f["sysk"]))
            except ValueError:
                pass
    if len(tc_rows) >= 15:
        A = np.array([[a, t / 1000] for a, t, _, _ in tc_rows])
        yy = np.array([int(r) for _, _, r, _ in tc_rows])
        gg = np.array([s for _, _, _, s in tc_rows])
        if 0 < yy.sum() < len(yy):
            k = min(5, len(set(gg)))
            res2 = {}
            for name, cols in (("dH_ox", [0]), ("dH_ox + T synthese", [0, 1])):
                pr = np.zeros(len(yy), dtype=int)
                for tr, te in GroupKFold(n_splits=k).split(A, yy, gg):
                    if len(set(yy[tr])) < 2:
                        pr[te] = yy[tr][0]
                        continue
                    pr[te] = LogisticRegression(class_weight="balanced").fit(
                        A[tr][:, cols], yy[tr]).predict(A[te][:, cols])
                res2[name] = (pr == yy).mean()
            full = LogisticRegression(class_weight="balanced").fit(A, yy)
            lines.append(f"Entrees dont le resume donne la temperature de synthese : {len(yy)} "
                         f"({int(yy.sum())} rocksalt), validation groupee {k} plis :")
            for k2, v in res2.items():
                lines.append(f"  {k2:<20s} exactitude {v:.3f}")
            lines.append(f"  coefficient de la temperature (par 1000 C, ajustement complet) : "
                         f"{full.coef_[0][1]:+.2f} (positif = la chaleur favorise le rocksalt)")
            lines.append("")

    # 3. surrogate --------------------------------------------------------------------
    S = surrogate(desc, "dH_ox_corr_eV_per_O")
    els, h, mae, r2 = S["els"], S["h"], S["mae"], S["r2"]
    lines.append("Substitut additif dH_ox ~ somme x_i h_i (un contre tous, par systeme) : "
                 f"erreur absolue moyenne {mae:.3f} eV/O, R2 {r2:.2f} "
                 f"(reference 'moyenne' {S['base_mae']:.3f} ; {S['n_scored']} systemes notes sur "
                 f"{len(desc)}"
                 + (", les autres portent un cation present dans un seul systeme)"
                    if S["n_scored"] < len(desc) else ")"))
    rare = [e for e, n in zip(els, S["n_sys"]) if n < 2]
    if rare:
        lines.append(f"  cations presents dans un seul systeme (non couverts par la page) : "
                     f"{' '.join(rare)}")
    for e, why in NOT_COVERED.items():
        lines.append(f"  {e} non couvert par la page : {why}")
    lines.append("  propension a l'oxydation par cation h_i (eV/O ; negatif = s'oxyde) : "
                 + ", ".join(f"{e} {v:+.2f}" for e, v in sorted(zip(els, h), key=lambda t: t[1])))
    # calibrated probability "reported as rock salt" from dH_ox (system level)
    sys_rs = np.array([int(d["n_rocksalt"]) > int(d["n_spinel"]) for d in desc]).astype(int)
    xs = np.array([[float(d["dH_ox_corr_eV_per_O"])] for d in desc])
    lg = LogisticRegression(class_weight="balanced").fit(xs, sys_rs)
    b0, b1 = float(lg.intercept_[0]), float(lg.coef_[0][0])
    x50 = -b0 / b1
    lines.append(f"  P(rocksalt) = 1 / (1 + exp(-({b0:.2f} + {b1:.2f} dH_ox))) ; bascule a "
                 f"dH_ox = {x50:+.2f} eV/O (classes equilibrees)")
    # physical reading of the fitted switch: oxidation stops paying when dH_ox exceeds the
    # oxygen chemical potential of air, lowered by the configurational entropy the rock
    # salt keeps and an ordered spinel loses (at most 3 kT ln 5 per O gained, 5 cations)
    sys.path.insert(0, HERE)
    import heo_thermo as ht
    for t_c in (900, 1000):
        T = t_c + 273.15
        mu = ht.delta_mu_O(T)
        lines.append(f"  lecture physique a {t_c} C dans l'air : Delta mu_O = {mu:+.2f} eV/O ; "
                     f"avec l'entropie de configuration maximale du rocksalt (3 kT ln 5 = "
                     f"{3 * KB * T * np.log(5):.2f}) la bascule attendue est entre "
                     f"{mu - 3 * KB * T * np.log(5):+.2f} et {mu:+.2f} eV/O")
    # spinel lattice parameter: additive surrogate in CHGNet space, then calibration k
    lat = json.load(open(LAT)) if os.path.exists(LAT) else None
    a_block = None
    if lat:
        SA = surrogate(desc, "a_sp_A")
        els_a, ha, mae_a, r2_a = SA["els"], SA["h"], SA["mae"], SA["r2"]
        ha_d = dict(zip(els_a, ha))
        lines.append(f"Substitut additif du parametre de maille spinelle (CHGNet) : erreur un-contre-tous "
                     f"{mae_a:.3f} A ({mae_a / 8.3 * 100:.2f} %), R2 {r2_a:.2f} "
                     f"(reference 'moyenne' {SA['base_mae']:.3f} A ; {SA['n_scored']} systemes notes)")
        errs = []
        for system, a_exp in lat["measured"]:
            els_s = system.split("-")
            if all(e in ha_d for e in els_s):
                a_pred = lat["k"] * sum(ha_d[e] / len(els_s) for e in els_s)
                errs.append((system, a_exp, a_pred))
                lines.append(f"  {system:<22s} mesure {a_exp:.4f}  substitut calibre {a_pred:.4f} "
                             f"({(a_pred - a_exp) / a_exp * 100:+.2f} %)")
        if errs:
            m = np.mean([abs(p - a) / a * 100 for _, a, p in errs])
            lines.append(f"  ecart absolu moyen sur les spinelles HEO mesures : {m:.2f} %")
        a_block = {"elements": els_a, "a_i_A": [round(float(v), 5) for v in ha], "k": lat["k"],
                   "n_systems_per_element": SA["n_sys"],
                   "loo_mae_A": round(float(mae_a), 4), "loo_r2": round(float(r2_a), 4),
                   "measured_mae_pct": round(float(m), 3) if errs else None}
    json.dump({"elements": els, "h_eV_per_O": [round(float(v), 4) for v in h],
               "n_systems_per_element": S["n_sys"], "not_covered": NOT_COVERED,
               "loo_mae": round(float(mae), 4), "loo_r2": round(float(r2), 4),
               "logistic": {"intercept": round(b0, 4), "slope": round(b1, 4),
                            "switch_eV_per_O": round(x50, 3)},
               "auc_system": round(float(M["dH_ox_corr_eV_per_O"][0]), 3),
               "auc_entry": round(float(M["dH_ox_corr_eV_per_O"][1]), 3),
               "spinel_lattice": a_block,
               "leave_cation_out": {k: round(float(v), 3) for k, v in M["leave_cation_out"].items()},
               "n_systems": len(desc), "source": "thermo/analyze_rocksalt_spinel.py"},
              open(SURR, "w"), indent=1)
    lines += ["", "genere par thermo/analyze_rocksalt_spinel.py"]
    text = "\n".join(lines) + "\n"
    open(REPORT, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
