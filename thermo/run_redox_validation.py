#!/usr/bin/env python3
"""Step 1b, part 1 — can CHGNet be trusted with oxidation energetics?

Before using the rock-salt -> spinel oxidation enthalpy as a descriptor of HEOs, check
it on binary redox couples whose thermochemistry is measured (NIST-JANAF; Jacob et al.
for the manganese oxides, see data/thermo/reference_thermo.csv):

    3 CoO   + O -> Co3O4        3 FeO + O -> Fe3O4        2 Fe3O4 + O -> 3 Fe2O3
    Cu2O    + O -> 2 CuO        2 Mn3O4 + O -> 3 Mn2O3

dH_ox (eV per O atom) = E(products) - E(reactants) - E_O_REF, with CHGNet energies
(MP2020-compatible) and the Materials Project O2 reference.

Acceptance (fixed before running): mean absolute error <= 0.30 eV/O and the
couples in the right order. First iteration: failed (0.51 eV/O). The errors are not
noise: both Fe2+ -> Fe3+ reactions are off by the same +0.68 eV/O, while the Mn error
belongs to the Mn2O3 phase (MnO -> Mn3O4 is off by only -0.06). Second iteration: a
per-element correction for the M2+ -> M3+ step, fitted on MO -> M3O4 (Co, Fe, Mn, the
reaction of a rock-salt -> spinel HEO), checked out of sample on Fe3O4 -> Fe2O3, and
written to data/thermo/redox_corrections.json for run_oxidation_descriptor.py. The transition temperature in air is also given, from
CHGNet enthalpies combined with the measured reaction entropy, to translate an enthalpy
error into kelvins.
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import heo_thermo as ht  # noqa: E402

OUT = os.path.join(ROOT, "data", "thermo", "redox_validation.csv")
REPORT = os.path.join(ROOT, "QC", "thermo_redox_validation.txt")
REF = os.path.join(ROOT, "data", "thermo", "reference_thermo.csv")
R_KJ = 8.314462618e-3
P_AIR_BAR = 0.2123

PHASES = {
    "CoO": lambda: ht.rocksalt(4.26, "Co"),
    "Co3O4": lambda: ht.primitive(ht.spinel(8.08, tet="Co", oct_="Co")),
    "FeO": lambda: ht.rocksalt(4.30, "Fe"),
    "Fe3O4": lambda: ht.primitive(ht.spinel(8.40, tet="Fe", oct_="Fe")),
    "Fe2O3": lambda: ht.primitive(ht.corundum("Fe")),
    "Cu2O": ht.cuprite,
    "CuO": ht.tenorite,
    "Mn3O4": lambda: ht.primitive(ht.spinel(8.20, tet="Mn", oct_="Mn")),
    "Mn2O3": lambda: ht.primitive(ht.bixbyite("Mn")),
    "MnO": lambda: ht.rocksalt(4.44, "Mn"),
}
# reaction: (reactants {phase: n}, products {phase: n}), one O atom gained
REACTIONS = {
    "3 CoO + O -> Co3O4": ({"CoO": 3}, {"Co3O4": 1}),
    "3 FeO + O -> Fe3O4": ({"FeO": 3}, {"Fe3O4": 1}),
    "2 Fe3O4 + O -> 3 Fe2O3": ({"Fe3O4": 2}, {"Fe2O3": 3}),
    "Cu2O + O -> 2 CuO": ({"Cu2O": 1}, {"CuO": 2}),
    "2 Mn3O4 + O -> 3 Mn2O3": ({"Mn3O4": 2}, {"Mn2O3": 3}),
    "3 MnO + O -> Mn3O4": ({"MnO": 3}, {"Mn3O4": 1}),
}
# the reactions that matter for a rock-salt -> spinel HEO: M2+ -> M3+ in MO -> M3O4
RS_TO_SP = {"Co": "3 CoO + O -> Co3O4", "Fe": "3 FeO + O -> Fe3O4",
            "Mn": "3 MnO + O -> Mn3O4"}
# independent check of the Fe correction: another Fe2+ -> Fe3+ oxidation
FE_CHECK = "2 Fe3O4 + O -> 3 Fe2O3"
FU_CATIONS = {"CoO": 1, "Co3O4": 3, "FeO": 1, "Fe3O4": 3, "Fe2O3": 2, "Cu2O": 2,
              "CuO": 1, "Mn3O4": 3, "Mn2O3": 2, "MnO": 1}
CORRECTIONS = os.path.join(ROOT, "data", "thermo", "redox_corrections.json")


def reference():
    H, G = {}, {}
    for r in csv.DictReader(open(REF)):
        T = float(r["T_K"])
        if T == 298.15:
            H[r["compound"]] = float(r["dfH_kJmol"])
        if r["dfG_kJmol"]:
            G.setdefault(r["compound"], {})[T] = float(r["dfG_kJmol"])
    return H, G


def energies():
    done = {}
    if os.path.exists(OUT):
        done = {r["phase"]: float(r["E_fu"]) for r in csv.DictReader(open(OUT))}
    rng = np.random.default_rng(0)
    for name, build in PHASES.items():
        if name in done:
            continue
        s = build()
        s.perturb(0.02, min_distance=0.0) if hasattr(s, "perturb") else None
        s, e = ht.relax(s, fmax=0.02, steps=600)
        n_cat = sum(1 for x in s if x.specie.symbol != "O")
        e_fu = e / n_cat * FU_CATIONS[name]
        done[name] = e_fu
        new = not os.path.exists(OUT)
        with open(OUT, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["phase", "space_group", "E_fu", "volume_per_cation"])
            w.writerow([name, s.get_space_group_info(symprec=0.1)[0], f"{e_fu:.6f}",
                        f"{s.volume / n_cat:.4f}"])
        print(f"  {name}: {e_fu:.4f} eV/fu", flush=True)
    del rng
    return done


def dG_exp_air(G, reac, prod, T):
    """Measured reaction Gibbs energy in air at T (kJ per O atom), None if not tabulated."""
    try:
        g = sum(n * G[p][T] for p, n in prod.items()) - sum(n * G[p][T] for p, n in reac.items())
    except KeyError:
        return None
    return g - 0.5 * R_KJ * T * np.log(P_AIR_BAR)


def crossing(f, temps):
    vals = [(T, f(T)) for T in temps]
    vals = [(T, v) for T, v in vals if v is not None]
    for (t1, v1), (t2, v2) in zip(vals, vals[1:]):
        if v1 <= 0 < v2:
            return t1 + (t2 - t1) * (-v1) / (v2 - v1)
    return None


def main():
    H, G = reference()
    E = energies()
    lines = ["Etape 1b - enthalpies d'oxydation de couples binaires : CHGNet contre experience",
             "reference experimentale : NIST-JANAF (Co, Fe, Cu), Jacob et al. (Mn), "
             "data/thermo/reference_thermo.csv",
             "",
             f"{'reaction':<26s}{'CHGNet':>9s}{'mesure':>9s}{'ecart':>8s}"
             f"{'T air mesuree':>15s}{'T air (H CHGNet)':>18s}"]
    errs, calc, meas = [], [], []
    rows = []
    for name, (reac, prod) in REACTIONS.items():
        dh_c = sum(n * E[p] for p, n in prod.items()) - sum(n * E[p] for p, n in reac.items()) \
            - ht.E_O_REF
        dh_m = (sum(n * H[p] for p, n in prod.items()) - sum(n * H[p] for p, n in reac.items())) \
            * ht.EV_PER_KJMOL
        temps = sorted(set.intersection(*[set(G.get(p, {})) for p in list(reac) + list(prod)])
                       - {298.15}) if all(p in G for p in list(reac) + list(prod)) else []
        t_exp = crossing(lambda T: dG_exp_air(G, reac, prod, T), temps) if temps else None

        def dg_mixed(T):
            g = dG_exp_air(G, reac, prod, T)
            if g is None:
                return None
            # same measured entropy, CHGNet enthalpy in place of the measured one
            return g + (dh_c - dh_m) * 96.485332
        t_mix = crossing(dg_mixed, temps) if temps else None
        errs.append(dh_c - dh_m)
        calc.append(dh_c)
        meas.append(dh_m)
        fmt = lambda t: f"{t:.0f} K" if t else ("stable" if temps else "-")  # noqa: E731
        lines.append(f"{name:<26s}{dh_c:>9.2f}{dh_m:>9.2f}{dh_c - dh_m:>+8.2f}"
                     f"{fmt(t_exp):>15s}{fmt(t_mix):>18s}")
        rows.append((name, dh_c, dh_m, t_exp, t_mix))
    # per-element correction, eV per electron of the M2+ -> M3+ step (2 per O)
    errd = {r[0]: r[1] - r[2] for r in rows}
    eps = {el: -errd[rx] / 2 for el, rx in RS_TO_SP.items()}
    json_out = {"eV_per_electron": {k: round(v, 4) for k, v in eps.items()},
                "fitted_on": RS_TO_SP, "source": "thermo/run_redox_validation.py"}
    import json
    open(CORRECTIONS, "w").write(json.dumps(json_out, indent=1) + "\n")
    fe_out = errd[FE_CHECK] + 2 * eps["Fe"]
    lines += ["",
              "correction par element de l'etape M2+ -> M3+ (ajustee sur MO -> M3O4) : "
              + ", ".join(f"{k} {v:+.3f} eV/electron" for k, v in eps.items()),
              f"controle hors echantillon, {FE_CHECK} : ecart brut {errd[FE_CHECK]:+.2f} -> "
              f"{fe_out:+.2f} eV/O apres correction du Fe",
              "Mn3O4 -> Mn2O3 (-0.68) n'est pas corrige : l'erreur porte sur la phase Mn2O3 "
              "(bixbyite), absente des spinelles HEO, pas sur l'element (MnO -> Mn3O4 : "
              f"{errd['3 MnO + O -> Mn3O4']:+.2f})"]
    errs = np.array(errs)
    order_ok = list(np.argsort(calc)) == list(np.argsort(meas))
    rho = np.corrcoef(np.argsort(np.argsort(calc)), np.argsort(np.argsort(meas)))[0, 1]
    lines += ["",
              "enthalpies en eV par atome O gagne ; 'stable' : pas de transition dans "
              "l'intervalle tabule",
              f"ecart absolu moyen : {np.abs(errs).mean():.2f} eV/O ; ecart moyen "
              f"{errs.mean():+.2f} eV/O",
              f"ordre des {len(rows)} couples : {'identique' if order_ok else 'different'} "
              f"(correlation de rang {rho:.2f})",
              "",
              "genere par thermo/run_redox_validation.py"]
    text = "\n".join(lines) + "\n"
    open(REPORT, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
