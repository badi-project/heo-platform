#!/usr/bin/env python3
"""Step 1c — lattice parameter of spinel HEOs from CHGNet, calibrated on binary spinels.

The v1.0 geometric model (Shannon radii, random cation distribution) overestimated the
lattice parameter of (FeCoNiCrMn)3O4 by 5.2 % and needed a bias correction fitted on
three literature points. Here:

1. 17 binary spinels of known lattice parameter and inversion are relaxed with CHGNet
   (conventional 56-atom cell, measured inversion imposed); a one-parameter linear
   calibration a_exp = k * a_CHGNet absorbs the systematic GGA+U overestimate. Its
   quality is measured by leave-one-out.
2. For each spinel HEO of measured lattice parameter, the tetrahedral / octahedral
   cation distribution is annealed by Metropolis Monte Carlo at the synthesis
   temperature (static CHGNet energies), the cell is relaxed, and the calibrated lattice
   parameter is predicted. None of the HEO values enter the calibration.

Acceptance (fixed before running): mean absolute error on the HEO spinels below the
1.9 % of the v1.0 geometric model, and below 1 %.
"""
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import heo_thermo as ht  # noqa: E402

OUT_BIN = os.path.join(ROOT, "data", "thermo", "spinel_binaries.csv")
OUT_HEO = os.path.join(ROOT, "data", "thermo", "spinel_heo.csv")
REPORT = os.path.join(ROOT, "QC", "thermo_spinel_lattice.txt")
HEO_CELL = [[0, 1, 2], [-2, -2, -2], [-2, 2, 1]]    # 10 primitive cells, 60 cations

# A B2 O4, room-temperature cubic lattice parameter (Å), inversion (fraction of the
# tetrahedral sites held by B). Usual handbook / ICSD values, +- 0.01 Å.
BINARIES = [
    ("Mg", "Al", 8.083, 0.0), ("Zn", "Al", 8.088, 0.0), ("Co", "Al", 8.106, 0.0),
    ("Fe", "Al", 8.155, 0.125), ("Ni", "Al", 8.048, 0.75),
    ("Mg", "Cr", 8.333, 0.0), ("Zn", "Cr", 8.327, 0.0), ("Co", "Cr", 8.332, 0.0),
    ("Mn", "Cr", 8.437, 0.0), ("Fe", "Cr", 8.378, 0.0),
    ("Zn", "Fe", 8.441, 0.0), ("Mg", "Fe", 8.383, 0.875), ("Ni", "Fe", 8.339, 1.0),
    ("Co", "Fe", 8.392, 0.75), ("Mn", "Fe", 8.499, 0.25),
    ("Fe", "Fe", 8.396, 1.0), ("Co", "Co", 8.084, 0.0),
]
# spinel HEOs with a measured lattice parameter: v1.0 database (3) and the
# (FeCoNiCrMn)3O4 sample calcined at 900 °C used as independent test in v1.0
HEO = [
    ("Co-Cr-Fe-Mn-Ni", 8.328, 1173, "echantillon interne, calcine 900 C"),
    ("Al-Co-Cu-Fe-Mn-Ni", 8.2697, 1173, "base, (AlCuCoFeMnNi)3O4"),
    ("Al-Cu-Fe-Ni-Zn", 8.2453, 1173, "base, (CuNiZnAlFe)3O4"),
    ("Co-Cu-Fe-Mn-Ni", 8.3602, 1173, "base, (CuNiFeMnCo)3O4"),
]
GEOMETRIC_V10 = {"Co-Cr-Fe-Mn-Ni": 8.763}      # v1.0 model before bias correction


def binary_spinel(a_el, b_el, a0, inv, rng):
    s = ht.spinel(a0, tet="Mg", oct_="Al")          # placeholders, then occupy
    tet = [i for i, x in enumerate(s) if x.properties["site"] == "tet"]
    octa = [i for i, x in enumerate(s) if x.properties["site"] == "oct"]
    n_swap = int(round(inv * len(tet)))
    tet_b = set(rng.choice(tet, n_swap, replace=False).tolist())
    oct_a = set(rng.choice(octa, n_swap, replace=False).tolist())
    for i in tet:
        s.replace(i, b_el if i in tet_b else a_el, properties=s[i].properties)
    for i in octa:
        s.replace(i, a_el if i in oct_a else b_el, properties=s[i].properties)
    return s


def anneal(s, T, n_steps, rng):
    """Metropolis swaps of unlike cations between a tetrahedral and an octahedral
    site, static energies on the current lattice. Returns structure, acceptance."""
    kT = ht.KB * T
    tet = [i for i, x in enumerate(s) if x.properties["site"] == "tet"]
    octa = [i for i, x in enumerate(s) if x.properties["site"] == "oct"]
    e = ht.static_energy(s)
    acc = 0
    for _ in range(n_steps):
        i, j = int(rng.choice(tet)), int(rng.choice(octa))
        a, b = s[i].specie.symbol, s[j].specie.symbol
        if a == b:
            continue
        t = s.copy()
        t.replace(i, b, properties=s[i].properties)
        t.replace(j, a, properties=s[j].properties)
        e2 = ht.static_energy(t)
        if e2 <= e or rng.random() < np.exp(-(e2 - e) / kT):
            s, e = t, e2
            acc += 1
    return s, acc / max(n_steps, 1)


def a_conventional(s):
    n_cat = sum(1 for x in s if x.specie.symbol != "O")
    return (s.volume * 24 / n_cat) ** (1 / 3)


def run_binaries():
    done = set()
    if os.path.exists(OUT_BIN):
        done = {r["spinel"] for r in csv.DictReader(open(OUT_BIN))}
    for a_el, b_el, a_exp, inv in BINARIES:
        name = f"{a_el}{b_el}2O4"
        if name in done:
            continue
        t0 = time.time()
        s = binary_spinel(a_el, b_el, a_exp, inv, np.random.default_rng(0))
        s, e = ht.relax_isotropic(s, fmax=0.03, steps=300)
        a_c = a_conventional(s)
        new = not os.path.exists(OUT_BIN)
        with open(OUT_BIN, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["spinel", "inversion", "a_exp_A", "a_chgnet_A", "E_eV", "seconds"])
            w.writerow([name, inv, a_exp, f"{a_c:.4f}", f"{e:.5f}", f"{time.time() - t0:.0f}"])
        print(f"  {name}: a CHGNet {a_c:.4f} vs {a_exp} A", flush=True)


def run_heo(n_mc=120, seeds=(0, 1)):
    done = set()
    if os.path.exists(OUT_HEO):
        done = {(r["system"], int(r["seed"])) for r in csv.DictReader(open(OUT_HEO))}
    prim = ht.primitive(ht.spinel(8.30))
    proto = prim.make_supercell(HEO_CELL, in_place=False)
    for system, a_exp, T, _ in HEO:
        els = system.split("-")
        frac = {e: 1 / len(els) for e in els}
        for seed in seeds:
            if (system, seed) in done:
                continue
            t0 = time.time()
            rng = np.random.default_rng(seed)
            s = ht.decorate(proto, frac, rng)
            s, _ = ht.relax_isotropic(s, fmax=0.1, steps=150, rounds=1)
            s, acc = anneal(s, T, n_mc, rng)
            s, e = ht.relax_isotropic(s, fmax=0.05, steps=300)
            a_c = a_conventional(s)
            tet_occ = {el: 0 for el in els}
            for x in s:
                if x.properties.get("site") == "tet":
                    tet_occ[x.specie.symbol] += 1
            new = not os.path.exists(OUT_HEO)
            with open(OUT_HEO, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["system", "seed", "a_exp_A", "a_chgnet_A", "E_eV",
                                "mc_acceptance", "tetrahedral_occupancy", "seconds"])
                w.writerow([system, seed, a_exp, f"{a_c:.4f}", f"{e:.5f}", f"{acc:.3f}",
                            " ".join(f"{k}:{v}" for k, v in sorted(tet_occ.items())),
                            f"{time.time() - t0:.0f}"])
            print(f"  {system} seed {seed}: a CHGNet {a_c:.4f} A, tet {tet_occ} "
                  f"({time.time() - t0:.0f} s)", flush=True)


def report():
    b = list(csv.DictReader(open(OUT_BIN)))
    x = np.array([float(r["a_chgnet_A"]) for r in b])
    y = np.array([float(r["a_exp_A"]) for r in b])
    k = float((x @ y) / (x @ x))
    loo = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        ki = (x[m] @ y[m]) / (x[m] @ x[m])
        loo.append(abs(ki * x[i] - y[i]) / y[i] * 100)
    raw = np.abs(x - y) / y * 100
    import json
    json.dump({"k": round(k, 6), "loo_mean_pct": round(float(np.mean(loo)), 3),
               "measured": [[s, a] for s, a, _, _ in HEO],
               "source": "thermo/run_spinel_lattice.py"},
              open(os.path.join(ROOT, "data", "thermo", "spinel_calibration.json"), "w"), indent=1)
    lines = ["Etape 1c - parametre de maille des spinelles : CHGNet calibre sur des binaires",
             f"calibration a_exp = k * a_CHGNet sur {len(x)} spinelles binaires : "
             f"k = {k:.4f} (CHGNet surestime de {(1 / k - 1) * 100:.2f} %)",
             f"  ecart brut moyen {raw.mean():.2f} % ; apres calibration, validation "
             f"croisee un-contre-tous : {np.mean(loo):.2f} % (max {np.max(loo):.2f} %)",
             ""]
    h = list(csv.DictReader(open(OUT_HEO)))
    lines.append(f"{'spinelle HEO':<22s}{'mesure':>9s}{'CHGNet calibre':>16s}{'ecart':>9s}"
                 f"{'v1.0 geom.':>12s}   occupation tetraedrique")
    errs = []
    for system, a_exp, _, src in HEO:
        rs = [r for r in h if r["system"] == system]
        if not rs:
            continue
        a = np.array([float(r["a_chgnet_A"]) for r in rs]) * k
        err = (a.mean() - a_exp) / a_exp * 100
        errs.append(err)
        g = GEOMETRIC_V10.get(system)
        gtxt = f"{(g - a_exp) / a_exp * 100:+.1f} %" if g else "-"
        lines.append(f"{system:<22s}{a_exp:>9.4f}{a.mean():>12.4f}±{a.std():.3f}{err:>+8.2f} %"
                     f"{gtxt:>12s}   {rs[0]['tetrahedral_occupancy']}")
    errs = np.array(errs)
    trivial = []
    for i, (system, a_exp, _, _) in enumerate(HEO):
        others = [o[1] for j, o in enumerate(HEO) if j != i]
        trivial.append(abs(np.mean(others) - a_exp) / a_exp * 100)
    lines += ["",
              f"ecart absolu moyen sur les spinelles HEO : {np.abs(errs).mean():.2f} % "
              f"(modele geometrique v1.0 : 1,9 % apres correction de biais ; reference "
              f"triviale 'moyenne des autres HEO' : {np.mean(trivial):.2f} %)",
              "",
              "genere par thermo/run_spinel_lattice.py"]
    text = "\n".join(lines) + "\n"
    open(REPORT, "w").write(text)
    print(text)


if __name__ == "__main__":
    if "--report" not in sys.argv:
        run_binaries()
        run_heo()
    report()
