#!/usr/bin/env python3
"""Step 1b, part 2 — rock salt or spinel? An oxidation descriptor for every rock-salt /
spinel system of the database.

The classifier's dominant confusion is rock salt <-> spinel, and 19 cation systems of
the database are reported in both structures: the same cations give MO or M3O4
depending on how oxidising the synthesis is. What the composition does fix is how
easily its cations accept the extra oxygen:

    3 MO (rock salt, HEO) + O -> M3O4 (spinel, HEO)        dH_ox, eV per O atom

computed with CHGNet on 60-cation cells of the actual composition (spinel cation
distribution annealed by Monte Carlo at 1173 K). A strongly negative dH_ox means the
cation set oxidises readily (spinel in air); a weakly negative or positive one means
it resists oxidation (rock salt survives).

Protocol kept cheap so that the 118 systems fit in a night on two CPU cores (one
decoration, relaxation to 0.1 eV/Å); its error is measured on a few systems against a
tighter protocol (see --check-protocol). Both cells are relaxed with
heo_thermo.relax_isotropic (positions, then volume): free-cell relaxation of a random
spinel ran away into a collapsed, higher-energy cell.

Results go to data/thermo/oxidation_descriptor.csv, appended as they come.
"""
import argparse
import collections
import csv
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import heo_thermo as ht  # noqa: E402
from run_spinel_lattice import HEO_CELL, anneal  # noqa: E402

DB = os.path.join(ROOT, "data", "heo_database.csv")
OUT = os.path.join(ROOT, "data", "thermo", "oxidation_descriptor.csv")
CHECK = os.path.join(ROOT, "data", "thermo", "oxidation_protocol_check.csv")
CORR = os.path.join(ROOT, "data", "thermo", "redox_corrections.json")
RS_CELL = [[1, -2, -2], [-2, -2, -1], [-2, 1, -1]]
SUPPORTED = set("Mg Ca Mn Fe Co Ni Cu Zn Cr Al Ga Li Ti V".split())
T_ANNEAL = 1173


def systems():
    """Rock-salt / spinel systems of the database made only of supported cations,
    most informative first: both labels, then rock-salt majority, then spinel."""
    rows = [r for r in csv.DictReader(open(DB, encoding="utf-8"))
            if r["proto"] in ("rocksalt", "spinel")
            and set(r["cations"].split("-")) <= SUPPORTED]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["sysk"]].append(r)
    out = []
    for sysk, rs in by.items():
        labels = collections.Counter(r["proto"] for r in rs)
        fracs = collections.Counter(
            json.dumps({k: round(v, 3) for k, v in sorted(json.loads(r["frac"]).items())})
            for r in rs)
        frac = json.loads(fracs.most_common(1)[0][0])
        both = len(labels) == 2
        major = labels.most_common(1)[0][0]
        prio = (0 if both else 1 if major == "rocksalt" else 2, -len(rs))
        out.append((prio, sysk, frac, labels))
    return [o[1:] for o in sorted(out)]


def valence(el, m):
    """Oxidation state from the CHGNet magnetic moment (thresholds read on the binary
    oxides: CoO 2.7 / Co3O4 oct 0.5; FeO 3.8 / Fe2O3 4.4; MnO 4.7 / Mn2O3 4.0;
    NiO 1.9; Cr2O3 3.0). None for cations whose valence is fixed or not tracked."""
    m = abs(m)
    if el == "Co":
        return 2 if m >= 1.8 else 3
    if el == "Fe":
        return 2 if m < 4.1 else 3
    if el == "Mn":
        return 2 if m > 4.35 else (3 if m > 3.5 else 4)
    if el == "Cr":
        return 2 if m > 3.4 else (3 if m > 2.4 else 4)
    if el == "Ni":
        return 2 if m > 1.4 else 3
    if el == "V":
        return 2 if m > 2.5 else (3 if m > 1.5 else 4)
    if el == "Ti":
        return 2 if m > 1.5 else (3 if m > 0.5 else 4)
    return None


def valence_counts(s):
    m = ht.model().predict_structure(s, task="em")["m"]
    out = {}
    for site, mm in zip(s, m):
        v = valence(site.specie.symbol, float(mm))
        if v is not None:
            out[site.specie.symbol] = out.get(site.specie.symbol, 0) + v
    return out


def rocksalt_energy(frac, seed, fmax, steps):
    s = ht.decorate(ht.rocksalt(4.25).make_supercell(RS_CELL, in_place=False), frac,
                    np.random.default_rng(seed))
    s, e = ht.relax_isotropic(s, fmax=fmax, steps=steps)
    return s, e


def spinel_energy(frac, seed, fmax, steps, n_mc):
    rng = np.random.default_rng(seed)
    proto = ht.primitive(ht.spinel(8.30)).make_supercell(HEO_CELL, in_place=False)
    s = ht.decorate(proto, frac, rng)
    s, _ = ht.relax_isotropic(s, fmax=0.15, steps=100, rounds=1)
    s, acc = anneal(s, T_ANNEAL, n_mc, rng)
    s, e = ht.relax_isotropic(s, fmax=fmax, steps=steps)
    return s, e, acc


def descriptor(frac, seed=0, fmax=0.1, steps=200, n_mc=80):
    t0 = time.time()
    srs, ers = rocksalt_energy(frac, seed, fmax, steps)
    ssp, esp, acc = spinel_energy(frac, seed, fmax, steps, n_mc)
    n_cat = sum(1 for x in srs if x.specie.symbol != "O")
    n_o_extra = sum(1 for x in ssp if x.specie.symbol == "O") - \
        sum(1 for x in srs if x.specie.symbol == "O")
    dh = (esp - ers - n_o_extra * ht.E_O_REF) / n_o_extra
    vrs, vsp = valence_counts(srs), valence_counts(ssp)
    electrons = {el: vsp.get(el, 0) - vrs.get(el, 0) for el in set(vrs) | set(vsp)}
    eps = json.load(open(CORR))["eV_per_electron"]
    dh_corr = dh + sum(eps.get(el, 0.0) * n for el, n in electrons.items()) / n_o_extra
    return {"E_rs": ers, "E_sp": esp, "dH_ox": dh, "dH_ox_corr": dh_corr,
            "electrons": electrons, "n_o_extra": n_o_extra, "n_cat": n_cat,
            "a_rs": (srs.volume / n_cat * 4) ** (1 / 3),
            "a_sp": (ssp.volume / n_cat * 24) ** (1 / 3), "mc_acc": acc,
            "seconds": time.time() - t0}


def append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerow(row)


def run(limit=None):
    done = set()
    if os.path.exists(OUT):
        done = {r["system"] for r in csv.DictReader(open(OUT))}
    todo = [s for s in systems() if s[0] not in done][:limit]
    print(f"{len(todo)} systemes a calculer", flush=True)
    for sysk, frac, labels in todo:
        d = descriptor(frac)
        append(OUT, ["system", "frac", "n_rocksalt", "n_spinel", "dH_ox_eV_per_O",
                     "dH_ox_corr_eV_per_O", "electrons_tracked", "n_O_extra",
                     "E_rs_eV", "E_sp_eV", "a_rs_A", "a_sp_A", "mc_acceptance", "seconds"],
               [sysk, json.dumps(frac), labels.get("rocksalt", 0), labels.get("spinel", 0),
                f"{d['dH_ox']:.4f}", f"{d['dH_ox_corr']:.4f}",
                json.dumps(d["electrons"], sort_keys=True), d["n_o_extra"],
                f"{d['E_rs']:.4f}", f"{d['E_sp']:.4f}",
                f"{d['a_rs']:.4f}", f"{d['a_sp']:.4f}", f"{d['mc_acc']:.3f}",
                f"{d['seconds']:.0f}"])
        print(f"  {sysk}: dH_ox = {d['dH_ox']:+.3f} (corr. {d['dH_ox_corr']:+.3f}) eV/O, "
              f"e- {d['electrons']}  (rs {labels.get('rocksalt', 0)}, "
              f"sp {labels.get('spinel', 0)}; {d['seconds']:.0f} s)", flush=True)


def check_protocol(names=("Co-Cu-Mg-Ni-Zn", "Co-Cr-Fe-Mn-Ni", "Co-Cu-Fe-Ni-Zn")):
    """Cheap protocol vs two seeds of a tighter one on a few systems."""
    sysd = {s[0]: s[1] for s in systems()}
    done = set()
    if os.path.exists(CHECK):
        done = {(r["system"], r["protocol"], r["seed"]) for r in csv.DictReader(open(CHECK))}
    for name in names:
        frac = sysd[name]
        for proto, kw in (("cheap", dict(seed=0)),
                          ("tight", dict(seed=1, fmax=0.05, steps=400, n_mc=200)),
                          ("tight", dict(seed=2, fmax=0.05, steps=400, n_mc=200))):
            if (name, proto, str(kw["seed"])) in done:
                continue
            d = descriptor(frac, **kw)
            append(CHECK, ["system", "protocol", "seed", "dH_ox_eV_per_O", "seconds"],
                   [name, proto, kw["seed"], f"{d['dH_ox']:.4f}", f"{d['seconds']:.0f}"])
            print(f"  {name} {proto} seed {kw['seed']}: {d['dH_ox']:+.3f} eV/O", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check-protocol", action="store_true")
    a = ap.parse_args()
    if a.check_protocol:
        check_protocol()
    else:
        run(a.limit)
