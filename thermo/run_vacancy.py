#!/usr/bin/env python3
"""Step 2, part 2 — oxygen-vacancy formation energy, a computable reducibility descriptor.

Published overpotentials cannot train a predictor (QC/oer_literature_variance.txt: the
same composition spreads over 160 mV between laboratories). The route kept instead is a
physical descriptor computed from the composition alone, to be confronted with
constant-protocol measurements: how easily the lattice gives up an oxygen atom.

    E_vac = E(cell with one O removed) - E(perfect cell) + E_O_REF        (eV)

1. Validation on binary oxides: E_vac must track the measured enthalpy of the lowest
   reduction step per O atom (NIST-JANAF, Jacob et al.), Spearman rho >= 0.7 fixed
   before running. The per-element redox corrections of step 1b are applied to the
   cations that the vacancy reduces (read from CHGNet magnetic moments).
2. Spinel HEOs: distribution of E_vac over 8 random oxygen sites of an annealed
   60-cation cell, for the spinel systems of the database with a published OER
   overpotential and for single substitutions of (CoCrFeMnNi)3O4.

Results: data/thermo/vacancy_binaries.csv, data/thermo/vacancy_heo.csv, report
QC/thermo_vacancy.txt.
"""
import csv
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import heo_thermo as ht  # noqa: E402
from run_oxidation_descriptor import SUPPORTED, valence_counts  # noqa: E402
from run_spinel_lattice import HEO_CELL, anneal  # noqa: E402

OUT_B = os.path.join(ROOT, "data", "thermo", "vacancy_binaries.csv")
OUT_H = os.path.join(ROOT, "data", "thermo", "vacancy_heo.csv")
OER = os.path.join(ROOT, "data", "thermo", "oer_eta10_literature.csv")
CORR = os.path.join(ROOT, "data", "thermo", "redox_corrections.json")
REPORT = os.path.join(ROOT, "QC", "thermo_vacancy.txt")
N_SITES = 8

# oxide in which the vacancy is made, builder, measured enthalpy of its lowest
# reduction step per O atom (eV) and the reaction it comes from
BINARIES = [
    ("Co3O4", lambda: ht.spinel(8.08, tet="Co", oct_="Co"), 2.040, "Co3O4 -> 3 CoO + O"),
    ("CoO", lambda: ht.rocksalt(4.26, "Co").make_supercell([2, 2, 2], in_place=False),
     2.464, "CoO -> Co + O"),
    ("Fe2O3", lambda: ht.corundum("Fe").make_supercell([2, 1, 1], in_place=False),
     2.433, "3 Fe2O3 -> 2 Fe3O4 + O"),
    ("Fe3O4", lambda: ht.spinel(8.40, tet="Fe", oct_="Fe"), 3.159, "Fe3O4 -> 3 FeO + O"),
    ("FeO", lambda: ht.rocksalt(4.30, "Fe").make_supercell([2, 2, 2], in_place=False),
     2.820, "FeO -> Fe + O"),
    ("CuO", lambda: ht.tenorite().make_supercell([2, 2, 2], in_place=False),
     1.466, "2 CuO -> Cu2O + O"),
    ("Cu2O", lambda: ht.cuprite().make_supercell([3, 3, 3], in_place=False),
     1.769, "Cu2O -> 2 Cu + O"),
    ("Mn3O4", lambda: ht.spinel(8.20, tet="Mn", oct_="Mn"), 2.390, "Mn3O4 -> 3 MnO + O"),
    ("MnO", lambda: ht.rocksalt(4.44, "Mn").make_supercell([2, 2, 2], in_place=False),
     3.992, "MnO -> Mn + O"),
]
# single substitutions of the internal sample (CoCrFeMnNi)3O4
BASE = ["Co", "Cr", "Fe", "Mn", "Ni"]
SUBS = ["Cu", "Zn", "Mg", "Al", "Ga"]


def corrected(e_raw, s_perf, s_def, eps):
    """Remove the per-element redox error: the vacancy reduces two cations."""
    vp, vd = valence_counts(s_perf), valence_counts(s_def)
    gained = {el: vp.get(el, 0) - vd.get(el, 0) for el in set(vp) | set(vd)}
    return e_raw - sum(eps.get(el, 0.0) * n for el, n in gained.items()), gained


def vacancies(s, e_perf, rng, n_sites, eps, steps=150):
    oxy = [i for i, x in enumerate(s) if x.specie.symbol == "O"]
    out = []
    for i in rng.choice(oxy, min(n_sites, len(oxy)), replace=False):
        d = s.copy()
        d.remove_sites([int(i)])
        d, e_def = ht.relax(d, fmax=0.05, steps=steps, relax_cell=False)
        raw = e_def - e_perf + ht.E_O_REF
        corr, gained = corrected(raw, s, d, eps)
        out.append((raw, corr, gained))
    return out


def append(path, header, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(header)
        w.writerow(row)


def run_binaries(eps):
    done = set()
    if os.path.exists(OUT_B):
        done = {r["oxide"] for r in csv.DictReader(open(OUT_B))}
    for name, build, dh, rx in BINARIES:
        if name in done:
            continue
        t0 = time.time()
        rng = np.random.default_rng(0)
        s, e = ht.relax(ht.rattle(build(), 0.02, rng), fmax=0.03, steps=400)
        v = vacancies(s, e, rng, 1, eps, steps=300)[0]
        append(OUT_B, ["oxide", "n_atoms", "E_vac_raw_eV", "E_vac_corr_eV",
                       "dH_red_exp_eV_per_O", "reaction", "seconds"],
               [name, len(s), f"{v[0]:.4f}", f"{v[1]:.4f}", dh, rx, f"{time.time() - t0:.0f}"])
        print(f"  {name}: E_vac {v[0]:.2f} (corr. {v[1]:.2f}) eV vs dH_red {dh:.2f}", flush=True)


def heo_systems():
    """The internal sample first, then its single substitutions (Cr replaced by each of
    SUBS; each other cation replaced by Cu), then the spinel systems with a published
    OER overpotential."""
    def swap(rep, sub):
        return "-".join(sorted([e for e in BASE if e != rep] + [sub]))
    out = ["-".join(sorted(BASE))]
    out += [swap("Cr", s) for s in SUBS]
    out += [swap(rep, "Cu") for rep in BASE if rep != "Cr"]
    for r in csv.DictReader(open(OER, encoding="utf-8")):
        if r["proto"] == "spinel" and set(r["system"].split("-")) <= SUPPORTED:
            out.append(r["system"])
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def run_heo(eps):
    done = set()
    if os.path.exists(OUT_H):
        done = {r["system"] for r in csv.DictReader(open(OUT_H))}
    proto = ht.primitive(ht.spinel(8.30)).make_supercell(HEO_CELL, in_place=False)
    for sysk in heo_systems():
        if sysk in done:
            continue
        t0 = time.time()
        els = sysk.split("-")
        rng = np.random.default_rng(0)
        s = ht.decorate(proto, {e: 1 / len(els) for e in els}, rng)
        s, _ = ht.relax_isotropic(s, fmax=0.15, steps=100, rounds=1)
        s, _ = anneal(s, 1173, 80, rng)
        s, e = ht.relax_isotropic(s, fmax=0.1, steps=200)
        v = vacancies(s, e, rng, N_SITES, eps)
        raw = np.array([x[0] for x in v])
        cor = np.array([x[1] for x in v])
        red = {}
        for x in v:
            for el, n in x[2].items():
                red[el] = red.get(el, 0) + n
        append(OUT_H, ["system", "E_vac_mean_eV", "E_vac_min_eV", "E_vac_std_eV",
                       "E_vac_raw_mean_eV", "reduced_cations", "seconds"],
               [sysk, f"{cor.mean():.4f}", f"{cor.min():.4f}", f"{cor.std():.4f}",
                f"{raw.mean():.4f}", json.dumps(red, sort_keys=True),
                f"{time.time() - t0:.0f}"])
        print(f"  {sysk}: E_vac {cor.mean():.2f} (min {cor.min():.2f}, sd {cor.std():.2f}) eV, "
              f"reduced {red} ({time.time() - t0:.0f} s)", flush=True)


def report():
    from scipy.stats import spearmanr
    b = list(csv.DictReader(open(OUT_B)))
    x = np.array([float(r["E_vac_corr_eV"]) for r in b])
    xr = np.array([float(r["E_vac_raw_eV"]) for r in b])
    y = np.array([float(r["dH_red_exp_eV_per_O"]) for r in b])
    rho, p = spearmanr(x, y)
    rho_r, _ = spearmanr(xr, y)
    lines = ["Etape 2 - energie de formation de lacune d'oxygene (CHGNet)",
             "",
             "1. Validation : la lacune suit-elle la reductibilite mesuree ?",
             f"{'oxyde':<8s}{'E_vac brute':>12s}{'corrigee':>10s}{'dH_red mesure':>15s}   reaction"]
    for r in b:
        lines.append(f"{r['oxide']:<8s}{float(r['E_vac_raw_eV']):>12.2f}"
                     f"{float(r['E_vac_corr_eV']):>10.2f}{float(r['dH_red_exp_eV_per_O']):>15.2f}"
                     f"   {r['reaction']}")
    lines += [f"correlation de rang (Spearman) : {rho:.2f} corrigee (p = {p:.3f}), "
              f"{rho_r:.2f} brute ; critere fixe avant calcul : >= 0,70", ""]
    if os.path.exists(OUT_H):
        h = list(csv.DictReader(open(OUT_H)))
        eta = {}
        for r in csv.DictReader(open(OER, encoding="utf-8")):
            eta.setdefault(r["system"], []).append(float(r["eta10_mV"]))
        lines.append("2. Spinelles HEO (8 sites O par composition, eV) :")
        lines.append(f"{'systeme':<26s}{'moyenne':>9s}{'min':>7s}{'e-t':>6s}   eta10 publies (mV)")
        for r in sorted(h, key=lambda r: float(r["E_vac_mean_eV"])):
            e = eta.get(r["system"])
            lines.append(f"{r['system']:<26s}{float(r['E_vac_mean_eV']):>9.2f}"
                         f"{float(r['E_vac_min_eV']):>7.2f}{float(r['E_vac_std_eV']):>6.2f}"
                         f"   {(', '.join(f'{v:.0f}' for v in e) if e else '-')}")
        # substitutions of the internal sample: a physical consistency check, the
        # expectation was written down before the Al and Ga results came in
        hd = {r["system"]: r for r in h}
        base = "-".join(sorted(BASE))

        def swap(rep, sub):
            return "-".join(sorted([e for e in BASE if e != rep] + [sub]))
        if base in hd:
            se = lambda r: float(r["E_vac_std_eV"]) / np.sqrt(N_SITES)  # noqa: E731
            lines.append("")
            lines.append(f"3. Substitutions de (CoCrFeMnNi)3O4 (reference "
                         f"{float(hd[base]['E_vac_mean_eV']):.2f} ± {se(hd[base]):.2f} eV, "
                         f"erreur-type sur {N_SITES} sites) :")
            div, tri = [], []
            for sub in SUBS:
                s_ = swap("Cr", sub)
                if s_ in hd:
                    v = float(hd[s_]["E_vac_mean_eV"])
                    (tri if sub in ("Al", "Ga") else div).append(v)
                    lines.append(f"  Cr -> {sub:<3s} {v:5.2f} ± {se(hd[s_]):.2f} eV")
            for rep in BASE:
                s_ = swap(rep, "Cu")
                if rep != "Cr" and s_ in hd:
                    lines.append(f"  {rep:<3s}-> Cu  {float(hd[s_]['E_vac_mean_eV']):5.2f} ± "
                                 f"{se(hd[s_]):.2f} eV")
            if div and tri:
                ok = min(tri) > max(div)
                lines.append(
                    "  attendu avant calcul : remplacer Cr3+ par un cation divalent (Cu, Zn, Mg) "
                    "force d'autres cations a l'etat 3+, donc reductibles -> E_vac baisse ; par un "
                    "trivalent non reductible (Al, Ga), non -> E_vac(Al, Ga) > E_vac(Cu, Zn, Mg) : "
                    + ("verifie" if ok else "non verifie")
                    + f" ({min(tri):.2f} contre {max(div):.2f} eV)")
        pairs = [(float(r["E_vac_mean_eV"]), np.median(eta[r["system"]])) for r in h
                 if r["system"] in eta]
        if len(pairs) >= 5:
            rr, pp = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
            single = sum(1 for r in h if r["system"] in eta and len(eta[r["system"]]) == 1)
            sign = ("une lacune plus difficile va avec un eta10 plus bas (plus actif)" if rr < 0
                    else "une lacune plus facile va avec un eta10 plus bas (plus actif)")
            lines.append(f"\nobservation (pas une validation) : correlation E_vac / eta10 publie "
                         f"(mediane par systeme), {len(pairs)} systemes : rho = {rr:+.2f} "
                         f"(p = {pp:.2f}) ; {sign}")
            oer_rep = os.path.join(ROOT, "QC", "oer_literature_variance.txt")
            txt = open(oer_rep, encoding="utf-8").read() if os.path.exists(oer_rep) else ""
            sd = re.search(r"autres labos\) : (\d+) mV", txt)
            icc = re.search(r"\(part de variance due a la composition\) : ([0-9.]+ \[IC 95 % "
                            r"[0-9.]+-[0-9.]+\])", txt)
            lines.append(f"  {single} des {len(pairs)} systemes n'ont qu'une mesure, d'un seul "
                         "laboratoire ; a composition egale, l'ecart-type entre laboratoires est de "
                         f"{sd.group(1) if sd else '?'} mV et la correlation intraclasse vaut "
                         f"{icc.group(1) if icc else '?'} (QC/oer_literature_variance.txt) : "
                         "le sens de la relation est a etablir a protocole constant")
    lines += ["", "genere par thermo/run_vacancy.py"]
    text = "\n".join(lines) + "\n"
    open(REPORT, "w").write(text)
    print(text)


if __name__ == "__main__":
    eps = json.load(open(CORR))["eV_per_electron"]
    if "--report" not in sys.argv:
        run_binaries(eps)
        run_heo(eps)
    report()
