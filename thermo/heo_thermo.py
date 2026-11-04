"""Thermodynamic descriptors of high-entropy oxides from a universal interatomic potential.

Step 1 of the roadmap: replace "which family does the literature report" by quantities
computed from the composition alone, with a machine-learned interatomic potential
(CHGNet, Deng et al., Nat. Mach. Intell. 2023) standing in for DFT.

What is computed
----------------
* ``mixing_rocksalt`` — mixing enthalpy of a rock-salt HEO with respect to the binary
  monoxides in their ground-state structures (MgO, CaO, MnO, FeO, CoO, NiO rock salt;
  CuO tenorite; ZnO wurtzite), and the spread of relaxed cation–oxygen bond lengths.
  This is the descriptor pair of Sivak et al. (Phys. Rev. Lett. 134, 216101, 2025),
  computed with the same potential, on smaller cells.
* ``oxidation_enthalpy`` — enthalpy per oxygen atom of 3 MO(rock salt) + O -> M3O4
  (spinel) for the same cation set. Rock-salt HEOs need cations that resist oxidation;
  the sign and size of this number is what separates (MgCoNiCuZn)O from (CrMnFeCoNi)3O4.
* ``spinel_lattice`` — relaxed lattice parameter of a spinel HEO after a Monte Carlo
  anneal of the tetrahedral / octahedral cation distribution.

Energies are CHGNet predictions, which are trained on Materials Project energies with the
MP2020 compatibility corrections applied; the oxygen reference is the Materials Project
elemental O2 energy (E_O_REF). Everything is checked against experiment in run_*.py.
"""
import math
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "2")

from pymatgen.core import Lattice, Structure  # noqa: E402

# --------------------------------------------------------------------- constants --
KB = 8.617333262e-5          # eV / K
EV_PER_KJMOL = 1 / 96.485332
E_O_REF = -4.948             # eV per O atom, Materials Project O2 (mp-12957), GGA
R_O = 2.6                    # Å, cut-off for first-neighbour cation-oxygen bonds

# ------------------------------------------------------------------- prototypes --


def rocksalt(a=4.20, el="Mg"):
    """Conventional NaCl cell, 4 cations + 4 O."""
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(a), [el, "O"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5]])


def tenorite():
    """CuO, C2/c (Åsbrink & Norrby 1970)."""
    lat = Lattice.monoclinic(4.6837, 3.4226, 5.1288, 99.54)
    return Structure.from_spacegroup("C2/c", lat, ["Cu", "O"],
                                     [[0.25, 0.25, 0.0], [0.0, 0.4184, 0.25]])


def wurtzite(el="Zn", a=3.2495, c=5.2069, u=0.3819):
    lat = Lattice.hexagonal(a, c)
    return Structure.from_spacegroup("P6_3mc", lat, [el, "O"],
                                     [[1 / 3, 2 / 3, 0.0], [1 / 3, 2 / 3, u]])


def corundum(el, a=5.035, c=13.747, z=0.3553, x=0.3059):
    lat = Lattice.hexagonal(a, c)
    return Structure.from_spacegroup("R-3c", lat, [el, "O"], [[0, 0, z], [x, 0, 0.25]])


def bixbyite(el="Mn", a=9.41, x=-0.0285, o=(0.3851, 0.1453, 0.3800)):
    """C-type sesquioxide (Mn2O3 at high temperature, In2O3), Ia-3, 80 atoms."""
    return Structure.from_spacegroup("Ia-3", Lattice.cubic(a), [el, el, "O"],
                                     [[0.25, 0.25, 0.25], [x, 0.0, 0.25], list(o)])


def rutile(el, a=4.40, c=2.87, x=0.305):
    return Structure.from_spacegroup("P4_2/mnm", Lattice.tetragonal(a, c), [el, "O"],
                                     [[0, 0, 0], [x, x, 0]])


def cuprite(a=4.2696):
    lat = Lattice.cubic(a)
    cu = [[0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]]
    return Structure(lat, ["O", "O"] + ["Cu"] * 4, [[0, 0, 0], [0.5, 0.5, 0.5]] + cu)


def spinel(a=8.30, x=0.2625, tet="Mg", oct_="Al"):
    """Conventional Fd-3m spinel (origin choice 2): 8 tetrahedral + 16 octahedral
    cations, 32 O. Site kind in site_properties['site']."""
    s = None
    for sg in ("Fd-3m:2", "Fd-3m"):
        try:
            s = Structure.from_spacegroup(sg, Lattice.cubic(a), [tet, oct_, "O"],
                                          [[0.125, 0.125, 0.125], [0.5, 0.5, 0.5],
                                           [x, x, x]])
        except Exception:
            continue
        if len(s) == 56:
            break
    if s is None or len(s) != 56:
        raise RuntimeError("spinel prototype: wrong setting")
    kinds = []
    for site in s:
        if site.specie.symbol == "O":
            kinds.append("O")
        else:
            kinds.append("tet" if site.specie.symbol == tet else "oct")
    s.add_site_property("site", kinds)
    return s


def primitive(struct):
    """Primitive standard cell, keeping the 'site' property when present."""
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    p = SpacegroupAnalyzer(struct, symprec=1e-3).get_primitive_standard_structure()
    if "site" in struct.site_properties:
        tag = {}
        for site in struct:
            tag[site.specie.symbol] = site.properties["site"]
        p.add_site_property("site", [tag[s.specie.symbol] for s in p])
    return p


def best_supercell(prim, ncells, span=2):
    """Integer matrix of determinant ``ncells`` (entries in [-span, span]) whose
    supercell has the largest shortest periodic-image distance: as isotropic as
    possible, so that a cation does not see its own image as a near neighbour."""
    import itertools as it
    lat = prim.lattice.matrix
    best, best_m = -1.0, None
    rng = range(-span, span + 1)
    for row0 in it.product(rng, repeat=3):
        if row0 < (0, 0, 0):
            continue
        for row1 in it.product(rng, repeat=3):
            for row2 in it.product(rng, repeat=3):
                m = np.array([row0, row1, row2])
                if round(abs(np.linalg.det(m))) != ncells:
                    continue
                v = m @ lat
                # shortest lattice vector among small combinations
                d = min(np.linalg.norm(a * v[0] + b * v[1] + c * v[2])
                        for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
                        if (a, b, c) != (0, 0, 0))
                if d > best + 1e-6:
                    best, best_m = d, m
    return best_m, best


# ------------------------------------------------------------------- decoration --


def rattle(struct, sigma, rng):
    """Gaussian displacement of every atom (Å): breaks the symmetry of an ideal cell so
    that a relaxation can find distortions (Jahn-Teller, magnetostriction)."""
    s = struct.copy()
    cart = s.cart_coords + rng.normal(0.0, sigma, (len(s), 3))
    props = {k: list(v) for k, v in s.site_properties.items()}
    return Structure(s.lattice, [x.specie for x in s], cart, coords_are_cartesian=True,
                     site_properties=props)


def integer_counts(frac, n):
    """Largest-remainder rounding of fractions to n sites."""
    els = sorted(frac)
    raw = np.array([frac[e] * n for e in els])
    base = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - base))[: n - base.sum()]:
        base[i] += 1
    return dict(zip(els, base.tolist()))


def decorate(struct, frac, rng, sites=None):
    """Random occupation of the cation sites (or of ``sites`` indices) by ``frac``."""
    s = struct.copy()
    idx = sites if sites is not None else [i for i, x in enumerate(s)
                                           if x.specie.symbol != "O"]
    counts = integer_counts(frac, len(idx))
    pool = [e for e, k in counts.items() for _ in range(k)]
    rng.shuffle(pool)
    for i, e in zip(idx, pool):
        s.replace(i, e, properties=s[i].properties)
    return s


# ---------------------------------------------------------------------- engine ---
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from chgnet.model import CHGNet
        _MODEL = CHGNet.load(verbose=False)
    return _MODEL


def static_energy(struct):
    """Total energy (eV) of a fixed structure."""
    pred = model().predict_structure(struct, task="e")
    return float(pred["e"]) * len(struct)


def release_memory():
    """Return freed heap to the system: after a 120-atom relaxation glibc keeps ~4.5 GB
    reserved, and the next relaxation of another size would exceed 7 GB."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def relax(struct, fmax=0.05, steps=400, relax_cell=True):
    """FIRE relaxation (positions and cell). Returns (structure, total energy eV)."""
    from chgnet.model import StructOptimizer
    opt = StructOptimizer(model=model(), optimizer_class="FIRE")
    with open(os.devnull, "w") as dn:
        import contextlib
        with contextlib.redirect_stdout(dn):
            r = opt.relax(struct, fmax=fmax, steps=steps, relax_cell=relax_cell,
                          verbose=False)
    final = r["final_structure"]
    e = float(r["trajectory"].energies[-1])
    del opt, r
    release_memory()
    # the ASE round trip drops custom site properties; atom order is preserved
    if "site" in struct.site_properties:
        final.add_site_property("site", list(struct.site_properties["site"]))
    return final, e


def relax_isotropic(struct, fmax=0.1, steps=200, rounds=2,
                    scales=(0.97, 0.99, 1.01, 1.03, 1.05)):
    """Robust relaxation for strongly disordered cells: positions at fixed cell, then an
    isotropic volume scan (static energies, parabola through the lowest points), twice.

    The free-cell FIRE relaxation (FrechetCellFilter) can run away on a random spinel
    HEO: starting from large stresses it fell into a collapsed cell 18 % denser and
    higher in energy than the starting point. Cell shape is kept; volume is optimised."""
    s, e = relax(struct, fmax=fmax, steps=steps, relax_cell=False)
    for _ in range(rounds):
        v0 = s.volume
        es = []
        for f in scales:
            t = s.copy()
            t.scale_lattice(v0 * f ** 3)
            es.append(static_energy(t))
        es = np.array(es)
        k = int(np.argmin(es))
        lo, hi = max(k - 1, 0), min(k + 2, len(scales))
        if hi - lo < 3:
            lo, hi = (0, 3) if k == 0 else (len(scales) - 3, len(scales))
        c = np.polyfit(np.array(scales[lo:hi]), es[lo:hi], 2)
        best = -c[1] / (2 * c[0]) if c[0] > 0 else scales[k]
        best = float(np.clip(best, scales[0], scales[-1]))
        s = s.copy()
        s.scale_lattice(v0 * best ** 3)
        s, e = relax(s, fmax=fmax, steps=steps, relax_cell=False)
    return s, e


def bond_lengths(struct, cutoff=R_O):
    """First-neighbour cation–oxygen distances."""
    out = []
    for i, site in enumerate(struct):
        if site.specie.symbol == "O":
            continue
        for nb in struct.get_neighbors(site, cutoff):
            if nb.specie.symbol == "O":
                out.append(nb.nn_distance)
    return np.array(out)


# --------------------------------------------------------------- thermochemistry --
# NIST-JANAF / Shomate parameters for O2(g), standard pressure 1 bar
_SHOMATE_O2 = [(100, 700, (31.32234, -20.23531, 57.86644, -36.50624, -0.007374,
                           -8.903471, 246.7945, 0.0)),
               (700, 2000, (30.03235, 8.772972, -3.988133, 0.788313, -0.741599,
                            -11.32468, 236.1663, 0.0))]


def o2_h_s(T):
    """H(T) - H(298.15) in kJ/mol and S(T) in J/(mol K) of O2 gas at 1 bar."""
    for lo, hi, (A, B, C, D, E, F, G, H) in _SHOMATE_O2:
        if lo <= T <= hi:
            break
    t = T / 1000
    dh = A * t + B * t ** 2 / 2 + C * t ** 3 / 3 + D * t ** 4 / 4 - E / t + F - H
    s = A * math.log(t) + B * t + C * t ** 2 / 2 + D * t ** 3 / 3 - E / (2 * t ** 2) + G
    return dh, s


def delta_mu_O(T, p_bar=0.2123):
    """mu_O(T, p) - mu_O(298 K, 1 bar enthalpy reference), eV per O atom (air by default)."""
    dh, s = o2_h_s(T)
    return (dh * 1000 - T * s) / 2 / 96485.332 + 0.5 * KB * T * math.log(p_bar)
