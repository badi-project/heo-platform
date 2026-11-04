"""Unit tests for the formula parser (src/formula.py).

Every case is a notation actually met in data/heo_database.csv; the expected cation
fractions are worked out by hand from the crystal chemistry of the written formula."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import features  # noqa: E402
import formula  # noqa: E402

E = features.load_elements()


def fr(text):
    return formula.cation_fractions(text, E)


def eq(*els):
    return {e: 1 / len(els) for e in sorted(els)}


def close(a, b, tol=1e-9):
    return set(a) == set(b) and all(abs(a[k] - b[k]) <= tol for k in a)


CASES = [
    # explicit fractions
    ("(Mg0.2Co0.2Ni0.2Cu0.2Zn0.2)O", eq("Mg", "Co", "Ni", "Cu", "Zn")),
    ("Mg₀.₂Co₀.₂Ni₀.₂Cu₀.₂Zn₀.₂O", eq("Mg", "Co", "Ni", "Cu", "Zn")),
    ("(Ce₀.₂Zr₀.₂Yb₀.₂Er₀.₂La₀.₂)O₂₋δ", eq("Ce", "Zr", "Yb", "Er", "La")),
    # no fractions
    ("(FeCoNiCrMn)3O4", eq("Fe", "Co", "Ni", "Cr", "Mn")),
    ("(Cr, Mn, Fe, Co, Ni)₃O₄", eq("Fe", "Co", "Ni", "Cr", "Mn")),
    ("MnCoNiZnCu", eq("Mn", "Co", "Ni", "Zn", "Cu")),
    ("CeZrLaPrYOx", eq("Ce", "Zr", "La", "Pr", "Y")),
    ("(MnFeCoNiCu)ₐOb", eq("Mn", "Fe", "Co", "Ni", "Cu")),
    # dash / prose lists
    ("Co–Mn–Fe–Cr–Ni–Zn–O", eq("Co", "Mn", "Fe", "Cr", "Ni", "Zn")),
    ("Fe─Co─Ni─Cr─Mn─O", eq("Fe", "Co", "Ni", "Cr", "Mn")),
    ("Co-Fe-Cr-Mn-Ni spinel", eq("Co", "Fe", "Cr", "Mn", "Ni")),
    ("Equimolar Mn, Fe, Mg, Zn, Ca rock salt HEO", eq("Mn", "Fe", "Mg", "Zn", "Ca")),
    ("HEO spinel (equimolar Co, Ni, Mn, Fe, Zn)", eq("Co", "Ni", "Mn", "Fe", "Zn")),
    ("CoNiFeZnCr-HEO", eq("Co", "Ni", "Fe", "Zn", "Cr")),
    # A/B site perovskites: the B cation must not be lost
    ("(Bi0.2Na0.2Ba0.2Sr0.2Ca0.2)TiO3",
     {"Bi": .1, "Na": .1, "Ba": .1, "Sr": .1, "Ca": .1, "Ti": .5}),
    ("(Pb1/6Ba1/6Sr1/6Ca1/6Na1/6Bi1/6)TiO3",
     {**{e: 1 / 12 for e in ("Pb", "Ba", "Sr", "Ca", "Na", "Bi")}, "Ti": .5}),
    ("La(Cr0.2Mn0.2Fe0.2Co0.2Ni0.2)O3",
     {"La": .5, **{e: .1 for e in ("Cr", "Mn", "Fe", "Co", "Ni")}}),
    ("La(CrMnFeCoNi)O3", {"La": .5, **{e: .1 for e in ("Cr", "Mn", "Fe", "Co", "Ni")}}),
    ("La(CrMnFeCo₀.₅Ni)O₃",
     {"La": .5, "Cr": .5 / 4.5, "Mn": .5 / 4.5, "Fe": .5 / 4.5, "Co": .25 / 4.5,
      "Ni": .5 / 4.5}),
    ("(Ba,Sr,Ca,Mg,Pb)(Ti,Zr,Hf,Sn)O3",
     {**{e: .1 for e in ("Ba", "Sr", "Ca", "Mg", "Pb")},
      **{e: .125 for e in ("Ti", "Zr", "Hf", "Sn")}}),
    ("(LaSmPrBaSr)0.2CoO3", {**{e: .1 for e in ("La", "Sm", "Pr", "Ba", "Sr")}, "Co": .5}),
    ("La₀.₂Pr₀.₂Nd₀.₂Ba₀.₂Sr₀.₂Co₀.₈Fe₀.₂O₃₋δ",
     {**{e: .1 for e in ("La", "Pr", "Nd", "Ba", "Sr")}, "Co": .4, "Fe": .1}),
    ("Eu(Sc/Cr/Fe/Ni/Al)O3", {"Eu": .5, **{e: .1 for e in ("Sc", "Cr", "Fe", "Ni", "Al")}}),
    # spinels with an explicit second site
    ("(CuNiMnCo)Fe2O4", {"Cu": 1 / 12, "Ni": 1 / 12, "Mn": 1 / 12, "Co": 1 / 12,
                         "Fe": 2 / 3}),
    ("(Ni0.2Mg0.2Cu0.2Zn0.2Co0.2)Cr2O4",
     {**{e: 1 / 15 for e in ("Ni", "Mg", "Cu", "Zn", "Co")}, "Cr": 2 / 3}),
    # pyrochlore A2B2O7 written with a bare B group: Hf1Zr1, not Hf0.5Zr0.5
    ("(Eu0.2Gd0.2Tb0.2Dy0.2Tm0.2)2(HfZr)O7",
     {**{e: .1 for e in ("Eu", "Gd", "Tb", "Dy", "Tm")}, "Hf": .25, "Zr": .25}),
    ("(Sm0.2Gd0.2Dy0.2Er0.2Yb0.2)2Zr2O7",
     {**{e: .1 for e in ("Sm", "Gd", "Dy", "Er", "Yb")}, "Zr": .5}),
    # variables evaluated at x = 0.1
    ("(Mg, Co, Ni, Cu, Zn)_{1-x}Li_xO",
     {**{e: .18 for e in ("Mg", "Co", "Ni", "Cu", "Zn")}, "Li": .1}),
    ("(La0.25Eu0.25Nd0.25Gd0.25)1-xDyxCrO3",
     {**{e: .1125 for e in ("La", "Eu", "Nd", "Gd")}, "Dy": .05, "Cr": .5}),
    # composites and heterostructures: the high-entropy component is kept
    ("Pt@(CrMnFeCoNi)3O4", eq("Cr", "Mn", "Fe", "Co", "Ni")),
    ("NiO/(NiZn)(MnCoFe)₂O₄", {"Ni": 1 / 6, "Zn": 1 / 6, "Mn": 2 / 9, "Co": 2 / 9,
                                "Fe": 2 / 9}),
    ("α-Fe₂O₃@LHEO (Fe, Co, Ni, Zn, Mn)", eq("Fe", "Co", "Ni", "Zn", "Mn")),
    ("(Bi0.2Na0.2Ba0.2Sr0.2Ca0.2)TiO3–Ba(Mg1/3Ta2/3)O3",
     {"Bi": .1, "Na": .1, "Ba": .1, "Sr": .1, "Ca": .1, "Ti": .5}),
    # dopant suffixes and charges
    ("(CrMnFeCoNi)₃O₄:P", eq("Cr", "Mn", "Fe", "Co", "Ni")),
    ("(FeCoNiCrMn)3O4 + Ce doping", eq("Fe", "Co", "Ni", "Cr", "Mn")),
    ("Ce₀.₂La₀.₂Gd₀.₂Y₀.₂Lu₀.₂O₁.₆:Eu³⁺", eq("Ce", "La", "Gd", "Y", "Lu")),
    # placeholder
    ("(M0.2La0.2Eu0.2Nd0.2Gd0.2)CrO3 (M = Dy, Ho, Er)",
     {**{e: .1 for e in ("Dy", "La", "Eu", "Nd", "Gd")}, "Cr": .5}),
    # nested groups
    ("[(Bi0.5Na0.5)0.25(La0.5Li0.5)0.25(Ce0.5K0.5)0.25Sr0.25Ca0.25]TiO3",
     # read literally the A site sums to 1.25 (3 x 0.25 + 0.25 + 0.25), so Ti = 1/2.25
     {"Bi": .125 / 2.25, "Na": .125 / 2.25, "La": .125 / 2.25, "Li": .125 / 2.25,
      "Ce": .125 / 2.25, "K": .125 / 2.25, "Sr": .25 / 2.25, "Ca": .25 / 2.25,
      "Ti": 1 / 2.25}),
]


@pytest.mark.parametrize("text,expected", CASES, ids=[c[0] for c in CASES])
def test_known_notations(text, expected):
    got = fr(text)
    assert close(got, expected, 1e-6), f"{text}: got {got}"


@pytest.mark.parametrize("text", [
    "high-entropy rare earth disilicate", "spinel oxide", "<UNKNOWN>",
    "High-entropy oxide perovskites", "Quinary HEO (derived from LDH)", "", None])
def test_prose_without_a_formula_yields_nothing_or_too_little(text):
    assert len(fr(text)) < 5


def test_prose_word_starting_with_a_symbol_is_not_read_as_an_element():
    assert "Co" not in fr("Coordination (MgNiCuZnFe)O")
    assert set(fr("Coordination (MgNiCuZnFe)O")) == {"Mg", "Ni", "Cu", "Zn", "Fe"}


def test_trace_dopants_are_not_counted():
    f = fr("Ba0.5Sr0.5Co0.79Fe0.2Ta0.005Ni0.003Zr0.002O3")
    assert set(f) == {"Ba", "Sr", "Co", "Fe"}


MORE = [
    ("TiZrNbHfTaO<sub>x</sub>", eq("Ti", "Zr", "Nb", "Hf", "Ta")),
    ("CrMnFeNiCuOX", eq("Cr", "Mn", "Fe", "Ni", "Cu")),
    ("LaBO3 (B = Sc, Al, Cr, Ni, Fe)", {"La": .5, **{e: .1 for e in ("Sc", "Al", "Cr", "Ni",
                                                                     "Fe")}}),
    ("(XCoNiCuZn)O (X = Li, Na, Mg)", eq("Li", "Co", "Ni", "Cu", "Zn")),
    ("(Mg0.25(1-x)CoxNi0.25(1-x)Cu0.25(1-x)Zn0.25(1-x))O",
     {"Mg": .225, "Co": .1, "Ni": .225, "Cu": .225, "Zn": .225}),
    ("(MgNiCoCu)(1-x)/4ZnxFe2O4",
     {"Mg": .075, "Ni": .075, "Co": .075, "Cu": .075, "Zn": .1 / 3, "Fe": 2 / 3}),
    ("[Lix(BaSrCaMg)(1−x)/4]TiO3",
     {"Li": .05, **{e: .1125 for e in ("Ba", "Sr", "Ca", "Mg")}, "Ti": .5}),
    ("Ti/V/Cr/Nb/Mo/W/Al/Cu oxide", eq("Ti", "V", "Cr", "Nb", "Mo", "W", "Al", "Cu")),
    ("La2O3+TiO2+Nb2O5+WO3+Ga2O3", {"La": .25, "Ti": .125, "Nb": .25, "W": .125, "Ga": .25}),
    ("10La2O3-20TiO2-10Nb2O5-20WO3-20ZrO2",
     {"La": .2, "Ti": .2, "Nb": .2, "W": .2, "Zr": .2}),
    ("Mn2–xAlxZn0.2Ni0.6Mg0.2O4",
     {"Mn": 1.9 / 3, "Al": .1 / 3, "Zn": .2 / 3, "Ni": .6 / 3, "Mg": .2 / 3}),
    ("Ca-doped (Gd,La,Nd,Pr,Sm)AlO3", {"Al": .5, **{e: .1 for e in ("Gd", "La", "Nd",
                                                                    "Pr", "Sm")}}),
    ("La₁₋ₓ(Nd, Sm, Gd, Y)ₓ/₄CoO₃",
     {"La": .45, **{e: .0125 for e in ("Nd", "Sm", "Gd", "Y")}, "Co": .5}),
    ("Al2O3/(Y0.2Er0.2Yb0.2Ho0.2Lu0.2)3Al5O12/ZrO2",
     {**{e: .075 for e in ("Y", "Er", "Yb", "Ho", "Lu")}, "Al": .625}),
    ("HEO (MgCoNiCuZn oxide with Rh nanoparticles)", eq("Mg", "Co", "Ni", "Cu", "Zn")),
    ("(FeCoNiMnCu)Ox/13X", eq("Fe", "Co", "Ni", "Mn", "Cu")),
    ("(PrBaSrCaMgSm)₁/₆CoO₃₋δ", {**{e: 1 / 12 for e in ("Pr", "Ba", "Sr", "Ca", "Mg",
                                                        "Sm")}, "Co": .5}),
    # ABO3 written without brackets: the A cation must not be diluted to 1/6
    ("S-LaNiFeCoCrMnO₃", {"La": .5, **{e: .1 for e in ("Ni", "Fe", "Co", "Cr", "Mn")}}),
    ("LaSrNiCoMnFeCuO₃", {"La": .25, "Sr": .25,
                          **{e: .1 for e in ("Ni", "Co", "Mn", "Fe", "Cu")}}),
    # ... but a bare run without O3 keeps equal amounts
    ("FeCoNiMnZnCrOx", eq("Fe", "Co", "Ni", "Mn", "Zn", "Cr")),
    ("NiLaMgAlCaCeOx", eq("Ni", "La", "Mg", "Al", "Ca", "Ce")),
    # "X-doped" in front of an element list names a dopant, not a sixth cation
    ("Zn-doped Co–Fe–Cr–Mn–Ni spinel", eq("Co", "Fe", "Cr", "Mn", "Ni")),
    ("La-doped (Co0.2Cr0.2Fe0.2Mn0.2Ni0.2)3O4", eq("Co", "Cr", "Fe", "Mn", "Ni")),
    ("Na-modified La(CrMnFeCoNi)1/5O3",
     {"La": .5, **{e: .1 for e in ("Cr", "Mn", "Fe", "Co", "Ni")}}),
]


@pytest.mark.parametrize("text,expected", MORE, ids=[c[0] for c in MORE])
def test_more_notations(text, expected):
    got = fr(text)
    assert close(got, expected, 1e-6), f"{text}: got {got}"
