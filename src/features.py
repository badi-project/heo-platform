"""HEO descriptor specification.

Single source of truth for the feature vector. The JavaScript mirror in the web
interface implements exactly this function; parity is checked numerically
(parity_check.txt) before the interface ships.

Input  : frac = {element_symbol: cation_site_fraction}  (fractions sum to 1)
Output : (names, values) with values in the fixed order given by names
"""
import json
import os
from math import log, sqrt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ELEMENTS = os.path.join(_ROOT, "data", "reference", "element_properties.json")
DEFAULT_MODEL = os.path.join(_ROOT, "app", "heo_model.json")

R_OXYGEN_VI = 1.40      # Shannon radius of O2- in coordination VI
R_OXYGEN_IV = 1.38

PROPS = ["r_shannon_VI", "r_shannon_IV", "r_shannon_VIII", "r_shannon_XII", "X", "Z",
         "mass", "ox", "n_ox", "group", "row", "atomic_radius", "melting_point",
         "ie1", "electron_affinity", "molar_volume", "e_density"]


def _wstats(vals, ws):
    m = sum(v * w for v, w in zip(vals, ws))
    var = sum(w * (v - m) ** 2 for v, w in zip(vals, ws))
    return m, sqrt(max(var, 0.0)), min(vals), max(vals), max(vals) - min(vals)


def ab_split(frac, E):
    """Assign cations to A (large, 12-coordinate) and B (small, 6-coordinate) sites.

    Deterministic rule used for both the Goldschmidt factor and the ordered-site
    XRD prototypes: sort by Shannon XII radius descending and fill A until the
    cumulative site fraction reaches 0.5."""
    order = sorted(frac, key=lambda e: -E[e]["r_shannon_XII"])
    A, B, cum = {}, {}, 0.0
    for e in order:
        x = frac[e]
        if cum < 0.5 - 1e-9:
            take = min(x, 0.5 - cum)
            A[e] = take
            if x - take > 1e-9:
                B[e] = x - take
            cum += take
        else:
            B[e] = x
    sa, sb = sum(A.values()) or 1.0, sum(B.values()) or 1.0
    return {k: v / sa for k, v in A.items()}, {k: v / sb for k, v in B.items()}


def featurize(frac, E, onehot_elements):
    names, vals = [], []
    els = sorted(frac)
    ws = [frac[e] for e in els]

    for p in PROPS:
        xs = [E[e][p] for e in els]
        m, sd, lo, hi, rg = _wstats(xs, ws)
        for stat, v in (("mean", m), ("std", sd), ("min", lo), ("max", hi), ("range", rg)):
            names.append(f"{p}_{stat}"); vals.append(v)

    # configurational entropy (units of R)
    names.append("s_config"); vals.append(-sum(w * log(w) for w in ws if w > 0))
    names.append("n_cat"); vals.append(float(len(els)))

    # Hume-Rothery style size mismatch on the octahedral radius
    r6 = [E[e]["r_shannon_VI"] for e in els]
    r6m = sum(v * w for v, w in zip(r6, ws))
    names.append("delta_r"); vals.append(sqrt(sum(w * (1 - v / r6m) ** 2 for v, w in zip(r6, ws))))

    X = [E[e]["X"] for e in els]
    Xm = sum(v * w for v, w in zip(X, ws))
    names.append("delta_X_rel")
    vals.append(sqrt(sum(w * (v - Xm) ** 2 for v, w in zip(X, ws))) / Xm)

    ox = [E[e]["ox"] for e in els]
    names.append("mean_ox"); vals.append(sum(v * w for v, w in zip(ox, ws)))
    names.append("ox_spread"); vals.append(float(max(ox) - min(ox)))
    names.append("redox_flex"); vals.append(sum(E[e]["n_ox"] * frac[e] for e in els))
    # oxygen stoichiometry implied by charge balance: M O_(mean_ox/2)
    names.append("o_per_cation"); vals.append(sum(v * w for v, w in zip(ox, ws)) / 2.0)

    A, B = ab_split(frac, E)
    rA = sum(E[e]["r_shannon_XII"] * x for e, x in A.items())
    rB = sum(E[e]["r_shannon_VI"] * x for e, x in B.items())
    names.append("r_A"); vals.append(rA)
    names.append("r_B"); vals.append(rB)
    names.append("rA_over_rB"); vals.append(rA / rB)
    names.append("goldschmidt_t"); vals.append((rA + R_OXYGEN_VI) / (sqrt(2.0) * (rB + R_OXYGEN_VI)))
    names.append("octahedral_factor"); vals.append(rB / R_OXYGEN_VI)

    # chemical-family counts (fraction-weighted)
    fam = {"RE": 0.0, "AE": 0.0, "alkali": 0.0, "TM3d": 0.0, "TM4d5d": 0.0, "pblock": 0.0}
    for e in els:
        z, g, row, x = E[e]["Z"], E[e]["group"], E[e]["row"], frac[e]
        if 57 <= z <= 71 or e in ("Y", "Sc"):
            fam["RE"] += x
        elif g == 2:
            fam["AE"] += x
        elif g == 1:
            fam["alkali"] += x
        elif 3 <= g <= 12 and row == 4:
            fam["TM3d"] += x
        elif 3 <= g <= 12 and row in (5, 6):
            fam["TM4d5d"] += x
        elif g >= 13:
            fam["pblock"] += x
    for k in ("RE", "AE", "alkali", "TM3d", "TM4d5d", "pblock"):
        names.append("frac_" + k); vals.append(fam[k])

    for e in onehot_elements:
        names.append("has_" + e); vals.append(1.0 if e in frac else 0.0)

    return names, vals


def load_elements(path=None):
    """Load the elemental property table.

    Defaults to ``data/reference/element_properties.json`` resolved relative to this
    file, so the function works from any working directory and from an installed copy.
    """
    return json.load(open(path or DEFAULT_ELEMENTS, encoding="utf-8"))


def load_onehot(path=None):
    """Load the 30 one-hot cation symbols, in the exact order used by the model."""
    return json.load(open(path or DEFAULT_MODEL, encoding="utf-8"))["onehot"]


def equimolar(*elements):
    """Convenience constructor: ``equimolar("Mg", "Co", "Ni", "Cu", "Zn")``."""
    els = list(elements[0]) if len(elements) == 1 and not isinstance(elements[0], str) \
        else list(elements)
    return {e: 1.0 / len(els) for e in els}


if __name__ == "__main__":
    E, onehot = load_elements(), load_onehot()
    names, vals = featurize(equimolar("Mg", "Co", "Ni", "Cu", "Zn"), E, onehot)
    print(f"(MgCoNiCuZn)O — {len(vals)} descriptors")
    for n in ("s_config", "n_cat", "delta_r", "goldschmidt_t", "octahedral_factor", "mean_ox"):
        print(f"  {n:<20s} {vals[names.index(n)]:.6f}")
