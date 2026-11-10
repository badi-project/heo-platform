"""Python prediction engine: the twin of ``predict()`` in app/heo_engine.js.

Until now the only complete predictor was the web page. The Python side could compute
descriptors (features.py) and read formulas (formula.py), but turning a formula into a
structural family, a lattice parameter and a powder pattern required opening the browser:
nothing could be scripted, and a list of two hundred formulas could not be scored.

This module does what the page does, with the standard library only and from the same
files (app/heo_model.json, data/reference/, data/thermo/):

    from predict import Predictor
    p = Predictor()
    r = p.predict("(FeCoNiCrMn)3O4")
    r["family"], r["probability"], r["lattice"]["a"], r["in_domain"]

``tools/predict.py`` is the command-line front end (one formula, or a CSV file).
``tools/parity_check.py`` compares every number returned here with the JavaScript engine.
"""
import csv
import json
import math
import os

import features
import formula

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# mean absolute error of the geometric lattice model on the measured values of the
# database, in percent (docs/RAPPORT_plateforme_HEO.md, section 5)
LATTICE_MAE_PCT = 1.93
MIN_CATIONS = 4          # the page refuses to predict below four cations; so does this
TRAINED_MIN_CATIONS = 5  # every training composition has at least five
TWO_THETA_MAX = 80.0


class PredictionError(ValueError):
    """The input cannot be turned into a prediction (unreadable formula, cation without
    tabulated properties, fewer than four cations)."""


def _read_json(root, *parts):
    with open(os.path.join(root, *parts), encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(root, *parts):
    with open(os.path.join(root, *parts), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ------------------------------------------------------------- thermo payload ----
def thermo_payload(root=ROOT):
    """Step-1/2 results of thermo/ in the form the page embeds (``THERMO``): the
    per-cation surrogates and the values computed directly with CHGNet. None when
    thermo/ has not been run. tools/build_app.py inlines exactly this object."""
    d = os.path.join(root, "data", "thermo")
    if not os.path.exists(os.path.join(d, "oxidation_surrogate.json")):
        return None
    s = _read_json(root, "data", "thermo", "oxidation_surrogate.json")

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
    k = _read_json(root, "data", "thermo", "spinel_calibration.json")["k"]
    for r in _read_csv(root, "data", "thermo", "oxidation_descriptor.csv"):
        out["systems"][r["system"]] = {"dh": round(float(r["dH_ox_corr_eV_per_O"]), 3),
                                       "a": round(float(r["a_sp_A"]) * k, 3),
                                       "frac": json.loads(r["frac"])}
    bench = _read_csv(root, "data", "thermo", "rocksalt_benchmark.csv")
    if bench:
        agg = {}
        for r in bench:
            agg.setdefault(r["system"], []).append((float(r["dH_mix_meV_atom"]),
                                                    float(r["sigma_bonds_A"])))
        th = _read_json(root, "data", "thermo", "rocksalt_thresholds.json")
        th_h, th_s = th["dH_mix_meV_atom"], th["sigma_A"]
        out["rs_threshold"] = {"dh": th_h, "sigma": th_s}
        for sysk, v in agg.items():
            h_ = sum(x for x, _ in v) / len(v)
            s_ = sum(y for _, y in v) / len(v)
            e = out["systems"].setdefault(sysk, {})
            e["rs"] = {"dh": round(h_, 1), "sigma": round(s_, 3),
                       "single": bool(h_ <= th_h + 1e-9 and s_ <= th_s + 1e-9)}
    if os.path.exists(os.path.join(d, "vacancy_heo.csv")):
        for r in _read_csv(root, "data", "thermo", "vacancy_heo.csv"):
            e = out["systems"].setdefault(r["system"], {})
            e["ev"] = round(float(r["E_vac_mean_eV"]), 2)
            e["evmin"] = round(float(r["E_vac_min_eV"]), 2)
    return out


# ------------------------------------------------------------------- trees -------
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
    """Deliberately the naive accumulation of heo_engine.js, not math.fsum."""
    z = [v / temperature for v in raw]
    mx = max(z)
    ex = [math.exp(v - mx) for v in z]
    s = 0.0
    for v in ex:
        s += v
    return [v / s for v in ex]


# ---------------------------------------------------------------- predictor ------
class Predictor:
    """Loads the published model once; ``predict`` is then a pure function of its input.

    ``context()`` returns the data the JavaScript engine reads as globals, so that the
    parity check feeds both engines the very same objects.
    """

    def __init__(self, root=ROOT):
        self.root = root
        bundle = _read_json(root, "app", "heo_model.json")
        self.bundle = bundle
        self.model = bundle["model"]
        self.featnames = bundle["featnames"]
        self.onehot = bundle["onehot"]
        self.ad = bundle["ad"]
        self.elements = _read_json(root, "data", "reference", "element_properties.json")
        self.lattice = {**_read_json(root, "data", "reference", "lattice_model.json"),
                        "mae_pct": LATTICE_MAE_PCT}
        self.xrd = _read_json(root, "data", "reference", "xrd_prototypes.json")
        self.thermo = thermo_payload(root)
        self._known = None

    def context(self):
        return {"ELEM": self.elements, "PROPS": self.bundle["props"], "ONEHOT": self.onehot,
                "FEATNAMES": self.featnames, "MODEL": self.model, "AD": self.ad,
                "LATTICE": self.lattice, "XRD": self.xrd, "THERMO": self.thermo,
                "REPORT_FEATS": []}

    # -- database lookup -----------------------------------------------------------
    def known_entries(self, cations):
        """Number of database entries with exactly this set of cations."""
        if self._known is None:
            self._known = {}
            for r in _read_csv(self.root, "data", "heo_database.csv"):
                self._known[r["sysk"]] = self._known.get(r["sysk"], 0) + 1
        return self._known.get("-".join(sorted(cations)), 0)

    # -- applicability domain ------------------------------------------------------
    def ad_distance(self, x):
        ad = self.ad
        z = [(x[j] - ad["mean"][i]) / ad["scale"][i] for i, j in enumerate(ad["idx"])]
        ds = []
        for row in ad["rows"]:
            d = 0.0
            for zi, ri in zip(z, row):
                d += (zi - ri) ** 2
            ds.append(d)
        ds.sort()
        acc = 0.0
        for d in ds[:ad["k"]]:
            acc += math.sqrt(d)
        return acc / ad["k"]

    # -- lattice parameter ---------------------------------------------------------
    def geometric_lattice(self, proto, frac):
        """Shannon-radius model recalibrated on one reference compound per prototype."""
        latt, E = self.lattice, self.elements
        if proto not in latt["geo"]:
            return None
        kind, r_o = latt["geo"][proto]

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
            if kind == "rB_VI":
                r = mean("r_shannon_VI", B)
            elif kind == "rA_VIII":
                r = mean("r_shannon_VIII", A)
            else:
                return None
        return latt["kcal"][proto] * (r + r_o)

    def thermo_lattice(self, proto, frac):
        """CHGNet additive model for spinels made of covered cations, else None."""
        th = self.thermo
        if proto != "spinel" or not th or not th.get("lat"):
            return None
        L = th["lat"]
        cov = L.get("cov") or L["elements"]
        els = sorted(frac)
        if not all(e in cov for e in els):
            return None
        return L["k"] * features.wsum([L["a"][L["elements"].index(e)] for e in els],
                                      [frac[e] for e in els])

    # -- oxidation enthalpy --------------------------------------------------------
    def thermo_ox(self, frac):
        """Oxidation enthalpy rocksalt -> spinel and the tendency in air it implies."""
        th = self.thermo
        if not th:
            return None
        ox = th["ox"]
        cov = ox.get("cov") or ox["elements"]
        els = sorted(frac)
        missing = [e for e in els if e not in cov]
        if missing:
            return {"covered": False, "missing": missing}
        dh_add = features.wsum([ox["h"][ox["elements"].index(e)] for e in els],
                               [frac[e] for e in els])
        exact = th["systems"].get("-".join(els))
        direct = bool(exact and "dh" in exact and exact.get("frac")
                      and all(abs(exact["frac"].get(e, 0) - frac[e]) < 0.02 for e in els))
        dh = exact["dh"] if direct else dh_add
        p = 1 / (1 + math.exp(-(ox["b0"] + ox["b1"] * dh)))
        return {"covered": True, "missing": [], "dh_ox_eV_per_O": dh, "direct": direct,
                "uncertainty_eV_per_O": None if direct else ox["mae"],
                "p_rocksalt": p,
                "verdict": "rocksalt" if p > 0.7 else "spinel" if p < 0.3 else "boundary",
                "system": exact}

    # -- powder pattern ------------------------------------------------------------
    def _f_atom(self, el, s):
        s2 = s * s
        g = 0.0
        for a, b in self.xrd["scattering"][el]:
            g += a * math.exp(-b * s2)
        return self.xrd["Z"][el] - 41.78214 * s2 * g

    def _site_occupancies(self, proto, frac):
        P = self.xrd["prototypes"][proto]
        occ = {}
        if P["ordered"]:
            A, B = features.ab_split(frac, self.elements)
            for t in P["nsites"]:
                occ[t] = A if t == "A" else B if t == "B" else {"O": 1}
        else:
            for t in P["nsites"]:
                occ[t] = {"O": 1} if t[0] == "O" else frac
        return occ

    def simulate_pattern(self, proto, frac, a, tt_max=TWO_THETA_MAX):
        """Kinematic powder pattern of the cubic prototype: positions from ``a``,
        intensities from the site occupancies. Mirrors simulatePattern() term for term."""
        P = self.xrd["prototypes"].get(proto)
        if not P:
            return []
        occ = self._site_occupancies(proto, frac)
        lam = self.xrd["wavelength"]
        s_max = math.sin((tt_max / 2) * math.pi / 180)
        acc, hkl = {}, {}
        for e in P["refl"]:
            st = lam * math.sqrt(e["q2"]) / (2 * a)
            if st > s_max or st >= 1:
                continue
            s = st / lam
            fr = fi = 0.0
            for t in e["G"]:
                f = 0.0
                for el in occ[t]:
                    f += occ[t][el] * self._f_atom(el, s)
                fr += f * e["G"][t][0]
                fi += f * e["G"][t][1]
            th = math.asin(st)
            lp = (1 + math.cos(2 * th) ** 2) / (math.sin(th) ** 2 * math.cos(th))
            inten = (fr * fr + fi * fi) * lp * e["n"]
            tt = f"{2 * th * 180 / math.pi:.3f}"
            acc[tt] = acc.get(tt, 0) + inten
            hkl.setdefault(tt, "".join(str(h) for h in e["hkl"]))
        peaks = [{"two_theta": float(k), "intensity": v, "hkl": hkl[k]} for k, v in acc.items()]
        mx = max([p["intensity"] for p in peaks] + [1e-30])
        for p in peaks:
            p["intensity"] = 100 * p["intensity"] / mx
        return sorted((p for p in peaks if p["intensity"] > 0.4), key=lambda p: p["two_theta"])

    # -- input -----------------------------------------------------------------------
    def read(self, composition):
        """Formula string or {element: amount} -> (fractions, flags, formula text)."""
        if isinstance(composition, dict):
            raw = {e: float(v) for e, v in composition.items() if float(v) > 0}
            text, flags = None, []
        else:
            text = str(composition).strip()
            raw = formula.cation_fractions(text, self.elements)
            flags = list(formula.parse(text, self.elements).flags)
        unknown = sorted(e for e in raw if e not in self.elements)
        if unknown:
            raise PredictionError("cation sans propriétés tabulées : " + ", ".join(unknown))
        if len(raw) < 2:
            raise PredictionError("formule non lue : écrivez-la comme dans un article, "
                                  "par exemple La(CrMnFeCoNi)O3")
        if len(raw) < MIN_CATIONS:
            raise PredictionError(f"{len(raw)} cations lus ({', '.join(sorted(raw))}) : "
                                  f"il en faut au moins {MIN_CATIONS}")
        tot = features.nsum(raw[e] for e in sorted(raw))
        return {e: raw[e] / tot for e in sorted(raw)}, flags, text

    # -- prediction ------------------------------------------------------------------
    def predict(self, composition, pattern=True):
        """Full prediction for a formula written as in a paper, or for a dictionary of
        cation amounts. Raises PredictionError when no prediction is possible."""
        frac, flags, text = self.read(composition)
        names, x = features.featurize(frac, self.elements, self.onehot)
        assert names == self.featnames, "descriptor order differs from the model export"
        probs = softmax(raw_predict(self.model, x), self.model["temperature"])
        classes = self.model["classes"]
        order = sorted(range(len(classes)), key=lambda i: -probs[i])
        top = classes[order[0]]
        d = self.ad_distance(x)
        in_domain = d <= self.ad["threshold"]

        a, pct, src = self.geometric_lattice(top, frac), self.lattice["mae_pct"], "geometric"
        a_th = self.thermo_lattice(top, frac)
        if a_th is not None:
            a, pct, src = a_th, self.thermo["lat"]["mae_pct"], "chgnet"
        lattice = None
        if a is not None:
            lattice = {"a": a, "uncertainty_pct": pct, "uncertainty_A": a * pct / 100,
                       "source": src, "space_group": self.xrd["prototypes"][top]["sg"]}

        warnings = []
        if not in_domain:
            warnings.append("hors du domaine d'applicabilité : le modèle extrapole, "
                            "la probabilité n'a pas de valeur")
        if len(frac) < TRAINED_MIN_CATIONS:
            warnings.append(f"{len(frac)} cations : le modèle est entraîné sur "
                            f"{TRAINED_MIN_CATIONS} cations ou plus")
        if probs[order[0]] - probs[order[1]] < 0.25:
            warnings.append(f"prédiction peu tranchée entre {top} et {classes[order[1]]}")

        return {
            "formula": text, "fractions": frac, "flags": flags,
            "family": top, "probability": probs[order[0]],
            "probabilities": {classes[i]: probs[i] for i in order},
            "margin": probs[order[0]] - probs[order[1]],
            "ad_distance": d, "ad_threshold": self.ad["threshold"], "in_domain": in_domain,
            "known_entries": self.known_entries(frac),
            "lattice": lattice,
            "thermo": self.thermo_ox(frac),
            "pattern": (self.simulate_pattern(top, frac, a) if (pattern and a) else []),
            "descriptors": dict(zip(names, x)),
            "warnings": warnings,
        }


if __name__ == "__main__":
    r = Predictor().predict("(FeCoNiCrMn)3O4")
    print(f"{r['formula']} — {r['family']} ({100 * r['probability']:.0f} %), "
          f"a = {r['lattice']['a']:.3f} Å, {len(r['pattern'])} réflexions")
