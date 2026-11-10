"""Effective ionic radii of Shannon (1976) by oxidation state, coordination and spin.

The table (data/reference/shannon_radii.json) is the published one, transcribed from
pymatgen. This module only adds two documented conventions:

- spin state: high spin, except Co3+ and Ni3+, which are low spin in the oxides this
  platform deals with (LaCoO3 and Co3O4 at room temperature, LiNiO2); an ion tabulated in
  one spin state only gets that one;
- a coordination Shannon did not tabulate is estimated, and flagged so that a model can
  say how much of its input was not measured:
  * between two tabulated coordinations: straight line between them ("interpolated");
  * beyond them (Pr3+ or Y3+ in twelve-fold coordination, the A site of a perovskite):
    from each tabulated coordination c of the ion, r(c) times the mean ratio
    r(target) / r(c) of the ions of the same charge that Shannon tabulated at both; the
    estimates are averaged ("extrapolated"). Two simpler rules were tried and rejected
    because they broke the lanthanide contraction (tests/test_radii.py keeps that
    order): a straight line through the last two points (Pr3+ XII larger than Ce3+ XII),
    and the ratio from the nearest coordination only (Ho3+ XII larger than Dy3+ XII,
    through the isolated Ho3+ X value).

    import radii
    radii.radius("Fe", 3, 6)          # (0.645, 'exact')
    radii.radius("Pr", 3, 12)         # (1.304, 'extrapolated')

Mirrored by ionicRadius() in app/heo_engine.js.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CN_ROMAN = {"II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9,
            "X": 10, "XI": 11, "XII": 12}
ROMAN = {v: k for k, v in CN_ROMAN.items()}
LOW_SPIN = {("Co", 3), ("Ni", 3)}


def load(root=ROOT):
    with open(os.path.join(root, "data", "reference", "shannon_radii.json"), encoding="utf-8") as fh:
        return json.load(fh)


_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = load()["radii"]
    return _TABLE


def _pick(entry, el, ox):
    """Radius of one tabulated (ion, coordination) entry under the spin convention."""
    if (el, ox) in LOW_SPIN and "ls" in entry:
        return entry["ls"]
    return entry["hs"] if "hs" in entry else entry["ls"]


def states(el):
    """Oxidation states tabulated for an element, as integers, cations only."""
    return sorted(int(k) for k in table().get(el, {}) if k and int(k) > 0)


def radius(el, ox, cn, tab=None):
    """(radius in angstrom, how it was obtained) for ion el^ox+ in coordination cn,
    or (None, 'missing') when Shannon gives no radius at all for that oxidation state.
    how: 'exact' | 'interpolated' | 'extrapolated' | 'single' (one coordination only,
    used as is)."""
    tab = tab or table()
    by_cn = tab.get(el, {}).get(str(ox))
    if not by_cn:
        return None, "missing"
    pts = sorted((CN_ROMAN[k], _pick(v, el, ox)) for k, v in by_cn.items() if k in CN_ROMAN)
    if not pts:
        return None, "missing"
    for c, r in pts:
        if c == cn:
            return r, "exact"
    if len(pts) == 1:
        return pts[0][1], "single"
    lower = [p for p in pts if p[0] < cn]
    upper = [p for p in pts if p[0] > cn]
    if lower and upper:
        (c1, r1), (c2, r2) = lower[-1], upper[0]
        return round(r1 + (r2 - r1) * (cn - c1) / (c2 - c1), 4), "interpolated"
    estimates = []
    for c, r in pts:
        ratio = same_charge_ratio(ox, c, cn, tab)
        if ratio is not None:
            estimates.append(r * ratio)
    if not estimates:                 # no ion of this charge tabulated at both
        (c1, r1), (c2, r2) = (upper[0], upper[1]) if upper else (lower[-2], lower[-1])
        return round(r1 + (r2 - r1) * (cn - c1) / (c2 - c1), 4), "extrapolated"
    s = 0.0
    for v in estimates:
        s += v
    return round(s / len(estimates), 4), "extrapolated"


def same_charge_ratio(ox, cn_from, cn_to, tab=None):
    """Mean of r(cn_to) / r(cn_from) over the ions of charge ox tabulated at both
    coordinations, None if there is none. Elements in alphabetical order, plain sum, so
    that the JavaScript mirror gives the same value."""
    tab = tab or table()
    a, b = ROMAN[cn_from], ROMAN[cn_to]
    ratios = []
    for el in sorted(tab):
        by_cn = tab[el].get(str(ox))
        if by_cn and a in by_cn and b in by_cn:
            ratios.append(_pick(by_cn[b], el, ox) / _pick(by_cn[a], el, ox))
    if not ratios:
        return None
    s = 0.0
    for v in ratios:
        s += v
    return s / len(ratios)


def oxygen(cn):
    """Radius of O2- in coordination cn (tabulated II to VIII)."""
    by_cn = table()["O"]["-2"]
    pts = sorted((CN_ROMAN[k], v["hs"]) for k, v in by_cn.items())
    for c, r in pts:
        if c == cn:
            return r
    raise ValueError(f"O2- not tabulated in coordination {cn}")
