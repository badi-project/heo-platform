"""Lattice parameter of the five cubic (or pseudo-cubic) prototypes from Shannon radii.

The rules are those of data/reference/lattice_protocol.md, stored in
data/reference/lattice_rules.json so that this module and lattice() in
app/heo_engine.js read the very same tables: oxidation states by charge balance in a
fixed order, cation sites, and one published crystal-chemical relation per prototype.
No coefficient is fitted here.

- rocksalt   a = 2 (r_M^VI + r_O^VI)                                   anion-cation contact
- fluorite   a = (4/sqrt 3)(r_M^VIII + r_an), r_an = ((2 - d) r_O^IV + d r_V) / 2 for
             MO(2-d): oxygen vacancies as hard spheres of radius r_V, from the host
             (Chatzichristodoulou et al., J. Electroceram. 34 (2015) 100, table 1).
             Protocol amendment A1: without it (d = 0) the model overestimates every
             vacancy-bearing fluorite, references included.
- perovskite a_pc = 0.06741 + 0.49052 (r_A + r_O) + 1.29212 (r_B + r_O),
             sixfold radii                       Ubic, J. Am. Ceram. Soc. 90 (2007) 3326
- spinel     a = 8 / (11 sqrt 3) [5 R_T + sqrt(33 R_O^2 - 8 R_T^2)],
             R_T = r_T^IV + r_O^IV, R_O = r_O^VI + r_O^IV  (u eliminated between the
             two cation-anion distances)     O'Neill & Navrotsky, Am. Mineral. 68 (1983) 181
- pyrochlore a = (8/sqrt 3)[1.43373 R_A - 0.42931 R_A^2 / R_B],
             R_A = r_A^VIII + 1.38, R_B = r_B^VI + 1.38     Mouta, Silva & Paschoal (2013)

r is the mean radius over a site, weighted by the amount of each ion on it. Every sum is
a plain left-to-right loop over the ions in a fixed order, so that the JavaScript twin
gives the same value.

    import lattice
    lattice.predict("spinel", {"Co": .2, "Cr": .2, "Fe": .2, "Mn": .2, "Ni": .2}, elements)
"""
import json
import math
import os

import features
import radii

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPS = 1e-12
_RULES = None
HOW = {"interpolated": "interpolé", "extrapolated": "extrapolé",
       "single": "pris dans la seule coordinence tabulée"}


class LatticeError(ValueError):
    pass


def rules():
    global _RULES
    if _RULES is None:
        with open(os.path.join(ROOT, "data", "reference", "lattice_rules.json"),
                  encoding="utf-8") as fh:
            _RULES = json.load(fh)
    return _RULES


def _sum(values):
    s = 0.0
    for v in values:
        s += v
    return s


def _charge(ions):
    return _sum([i[2] * i[3] for i in ions])


def _shift(ions, order, need, step):
    """Change the state of the listed elements by step (+1 / -1), in order, until the
    charge has moved by need (> 0); the last ion changed may be split in two states.
    ions: list of [site, element, state, amount]. Returns the charge left to move."""
    for el in order:
        n = len(ions)
        for k in range(n):
            ion = ions[k]
            if ion[1] != el or need <= EPS:
                continue
            target = ion[2] + step
            if radii.radius(el, target, 6)[0] is None:       # state not in Shannon
                continue
            moved = min(ion[3], need)
            need -= moved
            if moved >= ion[3] - EPS:
                ion[2] = target
            else:
                ion[3] -= moved
                ions.append([ion[0], el, target, moved])
    return need


def _two_sites(frac, a_set, elements, notes):
    """A and B fractions by element class; by Shannon XII radius when a class is empty."""
    els = sorted(frac)
    A = [e for e in els if e in a_set]
    B = [e for e in els if e not in a_set]
    if A and B:
        ta, tb = _sum([frac[e] for e in A]), _sum([frac[e] for e in B])
        if abs(ta - tb) > 0.02:
            notes.append(f"rapport A/B = {ta / tb:.3f}, différent de 1")
        return {e: frac[e] / ta for e in A}, {e: frac[e] / tb for e in B}
    notes.append("aucun cation de la classe attendue sur l'un des sites : sites attribués "
                 "par rayon décroissant")
    return features.ab_split(frac, elements)


def assign(proto, frac, elements):
    """Ions of the composition: list of [site, element, oxidation state, amount], amounts
    summing to 1 on each site, and notes on what the rules could not do."""
    R = rules()
    D = R["default_ox"]
    els = sorted(frac)
    notes = []
    if proto == "rocksalt":
        d = D["rocksalt"]

        def ox0(e):
            # 2+ unless listed; an element without a tabulated 2+ ion (Y, Sc, Zr...) takes
            # the platform default (amendment A2, no validation composition concerned)
            if e in d:
                return d[e]
            return d["_"] if str(d["_"]) in radii.table().get(e, {}) else elements[e]["ox"]
        ions = [["M", e, ox0(e), frac[e]] for e in els]
        need = 2.0 - _charge(ions)
        if need > EPS:
            left = _shift(ions, R["oxidise"]["rocksalt"], need, +1)
            if left > 1e-9:
                notes.append(f"charge inférieure de {left:.3f} à +2 par cation : lacunes")
        elif need < -1e-9:
            notes.append(f"charge supérieure de {-need:.3f} à +2 par cation : lacunes")
        big = [e for e in els if radii.radius(e, ox0(e), 6)[0] is not None
               and radii.radius(e, ox0(e), 6)[0] > R["rocksalt"]["large_cation_A"]]
        if big:
            notes.append("cation volumineux (" + ", ".join(big) + ") : sur les rocksalt mesurés, "
                         "une maille inchangée suggère une incorporation partielle ; la maille "
                         "prédite peut être trop grande")
    elif proto == "spinel":
        d = D["spinel"]
        ions = [["?", e, d.get(e, elements[e]["ox"]), frac[e]] for e in els]
        need = 8 / 3 - _charge(ions)
        if need > EPS:
            left = _shift(ions, R["oxidise"]["spinel"], need, +1)
            if left > 1e-9:
                notes.append(f"charge inférieure de {3 * left:.3f} à +8 par AB2O4")
        elif need < -1e-9:
            notes.append(f"charge supérieure de {-3 * need:.3f} à +8 par AB2O4")
        ions = _spinel_sites(ions, R["tet_score"])
    elif proto == "fluorite":
        d, mixed = D["fluorite"], R["fluorite_mixed"]
        ions = []
        for e in els:
            if e in mixed:
                for ox, share in mixed[e]:
                    ions.append(["M", e, ox, frac[e] * share])
            else:
                ions.append(["M", e, d.get(e, elements[e]["ox"]), frac[e]])
    elif proto == "perovskite":
        A, B = _two_sites(frac, set(R["perovskite_A"]), elements, notes)
        dA, dB = D["perovskite_A"], D["perovskite_B"]
        ions = [["A", e, dA.get(e, elements[e]["ox"]), A[e]] for e in sorted(A)]
        ions += [["B", e, dB.get(e, elements[e]["ox"]), B[e]] for e in sorted(B)]
        need = 6.0 - _charge(ions)
        if need > EPS:
            left = _shift(ions, R["oxidise"]["perovskite"], need, +1)
            if left > 1e-9:
                notes.append(f"charge inférieure de {left:.3f} à +6 : lacunes d'oxygène")
        elif need < -EPS:
            left = _shift(ions, R["reduce_perovskite"], -need, -1)
            if left > 1e-9:
                notes.append(f"charge supérieure de {left:.3f} à +6")
    elif proto == "pyrochlore":
        A, B = _two_sites(frac, set(R["pyrochlore_A"]), elements, notes)
        dA, dB = D["pyrochlore_A"], D["pyrochlore_B"]
        ions = [["A", e, dA.get(e, elements[e]["ox"]), A[e]] for e in sorted(A)]
        ions += [["B", e, dB.get(e, elements[e]["ox"]), B[e]] for e in sorted(B)]
    else:
        raise LatticeError(f"prototype sans modèle de maille : {proto}")
    ions.sort(key=lambda i: (i[0], i[1], i[2]))
    return ions, notes


def _spinel_sites(ions, score):
    """Fill the tetrahedral site (one cation in three) by decreasing preference, the
    rest octahedral; ties broken by element then state. Amounts re-expressed per site."""
    order = sorted(ions, key=lambda i: (-score.get(f"{i[1]}{i[2]}", 0), i[1], i[2]))
    room = 1 / 3
    out = []
    for _, el, ox, x in order:
        t = min(x, room) if room > EPS else 0.0
        room -= t
        if t > EPS:
            out.append(["T", el, ox, t * 3])
        if x - t > EPS:
            out.append(["O", el, ox, (x - t) * 3 / 2])
    return out


def _mean_radius(ions, site, cn, used):
    num, den = 0.0, 0.0
    for s, el, ox, x in ions:
        if s != site:
            continue
        r, how = radii.radius(el, ox, cn)
        if r is None:
            raise LatticeError(f"pas de rayon de Shannon pour {el}{ox}+")
        used.append({"site": site, "ion": f"{el}{ox}+", "cn": cn, "r": r, "how": how,
                     "amount": x})
        num += r * x
        den += x
    return num / den


def vacancy_radius(ions):
    """(oxygen vacancies per cation d, vacancy radius r_V) of a fluorite MO(2-d)."""
    hosts = rules()["fluorite"]["vacancy_hosts"]
    q = _charge(ions)
    d = (4.0 - q) / 2
    if d <= 1e-9:
        return 0.0, None
    rh = {h: radii.radius(h, 4, 8)[0] for h in sorted(hosts)}
    num, den = 0.0, 0.0
    for _, el, ox, x in ions:
        if ox < 4:
            w = (4 - ox) * x
            num += radii.radius(el, ox, 8)[0] * w
            den += w
    rs = num / den
    rv, tot = 0.0, 0.0
    for _, el, ox, x in ions:
        if ox != 4:
            continue
        if el in hosts:
            h = el
        else:
            r = radii.radius(el, 4, 8)[0]
            h = min(sorted(hosts), key=lambda k: abs(rh[k] - r))
        c0, c1 = hosts[h]
        rv += (c0 - c1 * (rs - rh[h])) * x
        tot += x
    if tot <= EPS:
        raise LatticeError("fluorine lacunaire sans cation tétravalent hôte")
    return d, rv / tot


def predict(proto, frac, elements, vacancy_model=True):
    """{'a', 'ions', 'radii', 'notes', 'estimated', 'vacancies'} for a composition given
    as cation fractions; raises LatticeError when the prototype or an ion has no model.
    vacancy_model=False gives the fluorite model of the protocol as first written (plain
    contact, before amendment A1), kept for the validation report."""
    R = rules()
    ions, notes = assign(proto, frac, elements)
    used = []
    vac = None
    if proto == "rocksalt":
        a = 2 * (_mean_radius(ions, "M", 6, used) + radii.oxygen(6))
    elif proto == "fluorite":
        o4 = radii.oxygen(4)
        rc = _mean_radius(ions, "M", 8, used)
        d, rv = vacancy_radius(ions) if vacancy_model else (0.0, None)
        r_an = o4 if rv is None else ((2 - d) * o4 + d * rv) / 2
        if rv is not None:
            vac = {"per_cation": d, "r_V": rv}
            notes.append(f"lacunes d'oxygène : {d:.3f} par cation, rayon effectif {rv:.3f} Å "
                         "(sphères dures)")
        a = 4 / math.sqrt(3) * (rc + r_an)
    elif proto == "perovskite":
        c = R["perovskite"]["c"]
        o6 = radii.oxygen(6)
        rA = _mean_radius(ions, "A", 6, used)
        rB = _mean_radius(ions, "B", 6, used)
        a = c[0] + c[1] * (rA + o6) + c[2] * (rB + o6)
        q = _charge(ions)
        if q < 6 - 1e-9:
            vac = {"per_cation": (6 - q) / 4, "r_V": None}
    elif proto == "spinel":
        o4 = radii.oxygen(4)
        RT = _mean_radius(ions, "T", 4, used) + o4
        RO = _mean_radius(ions, "O", 6, used) + o4
        a = 8 / (11 * math.sqrt(3)) * (5 * RT + math.sqrt(33 * RO * RO - 8 * RT * RT))
    else:  # pyrochlore
        c = R["pyrochlore"]["c"]
        o4 = radii.oxygen(4)
        RA = _mean_radius(ions, "A", 8, used) + o4
        RB = _mean_radius(ions, "B", 6, used) + o4
        a = 8 / math.sqrt(3) * (c[0] * RA - c[1] * RA * RA / RB)
    estimated = [u for u in used if u["how"] != "exact"]
    for u in estimated:
        notes.append(f"rayon {u['ion']} en coordinence {u['cn']} {HOW[u['how']]} : "
                     f"{u['r']:.3f} Å")
    return {"a": a, "ions": ions, "radii": used, "notes": notes, "estimated": estimated,
            "vacancies": vac}
