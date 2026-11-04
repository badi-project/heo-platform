"""Chemical-formula parser for the HEO database.

Turns the formula as written in a paper (``formula_text`` in data/heo_database.csv)
into cation amounts, so that descriptors are computed on the real stoichiometry
instead of an equimolar guess over a cation set produced by a language model.

Notations met in the corpus and handled here:

    (Mg0.2Co0.2Ni0.2Cu0.2Zn0.2)O          explicit fractions
    (FeCoNiCrMn)3O4, MnCoNiZnCu           no fractions: equal amounts
    (Ba,Sr,Ca,Mg,Pb)(Ti,Zr,Hf,Sn)O3       comma / slash lists inside a group
    La(CrMnFeCo0.5Ni)O3                   group read as one crystallographic site
    (LaSmPrBaSr)0.2CoO3                   multiplier applying to each element
    (Pb1/6Ba1/6Sr1/6Ca1/6Na1/6Bi1/6)TiO3  rational fractions
    (Mg,Co,Ni,Cu,Zn)_{1-x}Li_xO           variables x, y, z evaluated at X_NOMINAL
    Mg0.25(1-x)Cox..., (MgNiCoCu)(1-x)/4  amounts as small expressions
    O3-δ, O4-x, O₂₋δ, Eu³⁺, OX            oxygen non-stoichiometry and charges ignored
    Co–Mn–Fe–Cr–Ni–O, Ti/V/Cr/Nb/Mo oxide  element lists
    La2O3+TiO2+Nb2O5+WO3+Ga2O3            sums of binary oxides
    Pt@(CrMnFeCoNi)3O4, NiO/(NiZn)(MnCoFe)2O4, A–B composites
                                          the component with most cations is kept
    (FeCoNiCrMn)3O4:P, Ca-doped (..)AlO3  dopants dropped
    (M0.2La0.2...)CrO3 (M = Dy, Ho, Er)   placeholder for variants: first value
    LaBO3 (B = Sc, Al, Cr, Ni, Fe)        placeholder for a mixed site: all values

A group whose elements carry no explicit amount is ambiguous: ``La(CrMnFeCoNi)O3``
means one B site shared by five cations, ``(Eu..)2(HfZr)O7`` means Hf1Zr1, and
``(LaSmPrBaSr)0.2CoO3`` means 0.2 of each. Both readings are evaluated and the one
closest to charge balance with the written oxygen count is kept (common oxidation
states from the element table). Without an oxygen count the site reading is used.

Standard library only, like the rest of src/. The JavaScript mirror used by the web
page (``parseFormula`` in app/heo_engine.js) is checked against this module on every
formula of the database by tools/parity_check.py.
"""
import itertools
import re

X_NOMINAL = 0.1              # value given to composition variables x, y, z
MIN_CATION_FRACTION = 0.01   # below this a cation is a trace dopant and is not counted

# the full periodic table, so that real elements absent from the property table are
# recognised (and reported) instead of being misread as prose
SYMBOLS = frozenset((
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At "
    "Rn Fr Ra Ac Th Pa U Np Pu").split())
# anions and light non-metals: never counted as cations; charge used for the balance
ANIONS = {"O": -2, "F": -1, "Cl": -1, "Br": -1, "I": -1, "N": -3, "S": -2, "Se": -2,
          "H": 0, "C": 0, "P": 0, "B": 0}
VARS = {"x": X_NOMINAL, "y": X_NOMINAL, "z": X_NOMINAL, "δ": 0.0, "a": 1.0, "b": 1.0}
PLACEHOLDERS = ["5TM", "TM", "RE", "Ln", "Me", "M", "A", "B", "X"]

_SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉₊₋ₐₓᵧ", "0123456789+-axy")
_SUPER = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
_DASHES = "–—─"              # en dash, em dash, box line
_ALT_SEPS = "@/+;|"           # heterostructure / composite separators
_LIST_SEPS = ",-" + _DASHES + " "
_ADDITIVE_WORDS = {"with", "plus", "doped", "decorated", "supported", "loaded", "on",
                   "containing", "coated"}
_NUM = re.compile(r"[0-9]+(?:\.[0-9]+)?|\.[0-9]+")
_PAREN_AMOUNT = re.compile(r"\(([0-9.xyzδ+\-/ ]*[0-9xyzδ][0-9.xyzδ+\-/ ]*)\)")
_BINARY_OXIDE = re.compile(
    r"^\s*([0-9]+(?:\.[0-9]+)?)?\s*([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?O([0-9]+(?:\.[0-9]+)?)?\s*$")


def _nsum(values):
    """Neumaier-compensated sum, as features.nsum and app/heo_engine.js: plain sum()
    only compensates from CPython 3.12 on."""
    s = c = 0.0
    for x in values:
        t = s + x
        c += (s - t) + x if abs(s) >= abs(x) else (x - t) + s
        s = t
    return s + c


# explicit character classes, mirrored in app/heo_engine.js (str.isupper() and friends
# follow Unicode tables that JavaScript does not share)
_LOWER = set("abcdefghijklmnopqrstuvwxyz") | set("αβγδεζηθικλμνξοπρστυφχψω")
_UPPER = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
_DIGIT = set("0123456789")


def _lo(c):
    return c in _LOWER


def _up(c):
    return c in _UPPER


def _dg(c):
    return c in _DIGIT


def _al(c):
    return c in _LOWER or c in _UPPER


class Parsed:
    """Result of :func:`parse`. ``amounts`` maps cation -> amount as written."""

    def __init__(self, amounts, flags, oxygen=None):
        self.amounts = amounts
        self.flags = flags
        self.oxygen = oxygen

    def fractions(self):
        tot = _nsum(self.amounts.values())
        return {e: v / tot for e, v in sorted(self.amounts.items())} if tot > 0 else {}

    def __repr__(self):
        return f"Parsed({self.amounts}, flags={sorted(self.flags)})"


# ------------------------------------------------------------- normalisation --
def normalize(text):
    t = re.sub(r"<[^>]*>", "", str(text))                    # <sub>x</sub>, <i>
    t = t.translate(_SUB)
    t = "".join(ch for ch in t if ch not in _SUPER)          # charges Eu³⁺
    t = t.replace("−", "-").replace("‐", "-").replace("‑", "-")
    t = re.sub(r"(?<=[0-9])[" + _DASHES + r"](?=[xyz0-9])", "-", t)   # Mn2–xAlx
    t = re.sub(r"_\{([^{}]*)\}", r"\1", t)                   # _{1-x}
    t = t.replace("±", "-").replace("_", "").replace("{", "(").replace("}", ")")
    t = t.replace("[", "(").replace("]", ")").replace(" ", " ").replace("·", "+")
    t = re.sub(r"O[X](?![a-z])", "Ox", t)                    # CrMnFeNiCuOX
    t = re.sub(r"[0-9]+(?:\.[0-9]+)?\s*(?:wt|at|mol)?\.?\s*%", "", t)   # 0.75% FeCoNi...
    return t.strip()


def _placeholder_variants(t):
    """'(M0.2La0.2..)CrO3 (M = Dy, Ho, Er)' -> variant reading and mixed-site reading."""
    pat = r"\(?\s*(" + "|".join(PLACEHOLDERS) + r")\s*=\s*([A-Z][a-z]?(?:\s*[,/]\s*" \
          r"[A-Z][a-z]?)*)\s*\)?"
    m = re.search(pat, t)
    if not m:
        return [(t, set())]
    ph, values = m.group(1), re.findall(r"[A-Z][a-z]?", m.group(2))
    base = (t[:m.start()] + " " + t[m.end():]).strip()
    # 'B' in 'LaBO3' follows the lowercase end of a symbol; 'M' in 'Mn' is excluded by
    # the lookahead, 'RE' inside 'PRE' by the lookbehind
    where = r"(?<![A-Z])" + re.escape(ph) + r"(?=[^a-z]|[xyz](?![a-z])|$)"
    first = re.sub(where, values[0], base)
    mixed = re.sub(where, "(" + ",".join(values) + ")", base)
    return [(first, {"placeholder_variant"}), (mixed, {"placeholder_mixed_site"})]


# ------------------------------------------------------------------- amounts --
def _var_at(s, j):
    return j < len(s) and s[j] in VARS and not (j + 1 < len(s) and _lo(s[j + 1]))


def _starts_amount(s, i):
    return i < len(s) and (_dg(s[i]) or s[i] == "." or _var_at(s, i)
                           or bool(_PAREN_AMOUNT.match(s, i)))


def _read_factor(s, j):
    m = _NUM.match(s, j)
    if m:
        return float(m.group(0)), False, m.end()
    if _var_at(s, j):
        return VARS[s[j]], s[j] in "xyz", j + 1
    m = _PAREN_AMOUNT.match(s, j)
    if m:
        inner = m.group(1).replace(" ", "")
        v, var, k = _read_sum(inner, 0)
        if v is not None and k == len(inner):
            return v, var, m.end()
    return None, False, j


def _read_term(s, j):
    v, var, j = _read_factor(s, j)
    if v is None:
        return None, False, j
    while j < len(s):
        if s[j] == "/":
            d, var2, k = _read_factor(s, j + 1)
            if d is None or d == 0:
                break
            v, var, j = v / d, var or var2, k
        elif s[j] == "(" and _PAREN_AMOUNT.match(s, j):      # 0.25(1-x)
            f, var2, k = _read_factor(s, j)
            if f is None:
                break
            v, var, j = v * f, var or var2, k
        else:
            break
    return v, var, j


def _read_sum(s, j):
    v, var, j = _read_term(s, j)
    if v is None:
        return None, False, j
    while j < len(s) and s[j] in "+-" and _starts_amount(s, j + 1):
        t, var2, k = _read_term(s, j + 1)
        if t is None:
            break
        v = v + t if s[j] == "+" else v - t
        var, j = var or var2, k
    return v, var, j


def _read_amount(s, j):
    """Amount after an element or a group: 0.2, 1/6, x, 1-x, 0.25(1-x), (1-x)/4, 3-δ.
    Returns (value or None, uses_variable, next index)."""
    return _read_sum(s, j)


# -------------------------------------------------------------------- tokens --
def _read_symbol(s, i):
    if i < len(s) and _up(s[i]):
        if i + 1 < len(s) and _lo(s[i + 1]) and s[i:i + 2] in SYMBOLS:
            return s[i:i + 2], i + 2
        if s[i] in SYMBOLS:
            return s[i], i + 1
    return None, i


def _parse_items(s, i, depth):
    """Parse elements and groups until ')' (depth > 0) or end of string.
    item = ("el", symbol, amount | None) | ("grp", items, amount | None).
    Returns (items, next index, ok, flags)."""
    items, flags = [], set()
    while i < len(s):
        c = s[i]
        if c == "(":
            sub, j, ok, f = _parse_items(s, i + 1, depth + 1)
            flags |= f
            if not ok:
                return items, j, False, flags
            amt, var, j = _read_amount(s, j)
            if var:
                flags.add("variable")
            items.append(("grp", sub, amt))
            i = j
        elif c == ")":
            return (items, i + 1, True, flags) if depth > 0 else (items, i, False, flags)
        elif _up(c):
            sym, j = _read_symbol(s, i)
            if sym is None or (j < len(s) and _lo(s[j]) and not _var_at(s, j)):
                return items, i, False, flags      # prose ("Coordination", "HEO")
            amt, var, j = _read_amount(s, j)
            if var:
                flags.add("variable")
            items.append(("el", sym, amt))
            i = j
        elif depth > 0 and (c in ",/ -" or c in _DASHES):
            i += 1
        elif depth > 0 and _lo(c):           # "(equimolar Co, Ni, ...)"
            j = i
            while j < len(s) and _al(s[j]):
                j += 1
            flags.add("prose_in_group")
            if s[i:j] in _ADDITIVE_WORDS:
                # "(MgCoNiCuZn oxide with Rh nanoparticles)": what follows is an
                # additive, not a lattice cation; skip to the closing bracket
                level = 0
                while j < len(s) and not (s[j] == ")" and level == 0):
                    level += (s[j] == "(") - (s[j] == ")")
                    j += 1
                flags.add("minor_component_dropped")
            i = j
        else:
            return items, i, False, flags
    return items, i, depth == 0, flags


# ---------------------------------------------------------------- evaluation --
def _is_cation_item(it):
    return it[0] == "grp" or it[1] not in ANIONS


def _ambiguous_groups(items, acc):
    """Groups holding a cation with no explicit amount, next to other cations."""
    siblings = [it for it in items if _is_cation_item(it)]
    for it in items:
        if it[0] != "grp":
            continue
        _ambiguous_groups(it[1], acc)
        bare = any(sub[0] == "el" and sub[1] not in ANIONS and sub[2] is None
                   for sub in it[1])
        if bare and len(siblings) > 1:
            acc.append(id(it))
    return acc


def _evaluate(items, mult, site_ids, cations, anions):
    for it in items:
        if it[0] == "el":
            amt = (1.0 if it[2] is None else it[2]) * mult
            bucket = anions if it[1] in ANIONS else cations
            bucket[it[1]] = bucket.get(it[1], 0.0) + amt
        else:
            gm = (1.0 if it[2] is None else it[2]) * mult
            if id(it) in site_ids:              # one site shared by the group members
                inner = {}
                _evaluate(it[1], 1.0, site_ids, inner, {})
                tot = _nsum(inner.values()) or 1.0
                for e, v in inner.items():
                    cations[e] = cations.get(e, 0.0) + v / tot * gm
            else:
                _evaluate(it[1], gm, site_ids, cations, anions)


def _charge_misfit(cations, anions, elements):
    if not elements or "O" not in anions:
        return None
    pos = _nsum([v * elements[e]["ox"] for e, v in cations.items() if e in elements])
    neg = _nsum([-v * ANIONS[a] for a, v in anions.items()])
    return abs(pos - neg) / max(neg, 1e-9)


def _evaluate_best(items, elements):
    amb = _ambiguous_groups(items, [])[:4]
    best = None
    for choice in itertools.product([True, False], repeat=len(amb)):
        site_ids = {g for g, c in zip(amb, choice) if c}
        cat, an = {}, {}
        _evaluate(items, 1.0, site_ids, cat, an)
        mis = _charge_misfit(cat, an, elements)
        key = 0.0 if mis is None else mis
        if best is None or key < best[0] - 1e-9:      # site reading (True) wins ties
            best = (key, cat, an, choice)
        if mis is None:
            break
    flags = set()
    if amb:
        flags.add("ambiguous_group")
        if not all(best[3]):
            flags.add("per_element_multiplier")
    return best[1], best[2], flags


# -------------------------------------------------------------------- driver --
def _fraction_end_ok(s, i):
    """After '/<digits>' a formula continues with a symbol, a bracket or nothing:
    'x/4CoO3' is a fraction, 'Ox/13X' (zeolite 13X) is not."""
    m = _NUM.match(s, i)
    k = m.end() if m else i
    return k >= len(s) or s[k] in "()-+ " or _read_symbol(s, k)[0] is not None


def _split_top(s, seps):
    """Split s outside parentheses on any char of seps. A '-' that starts an amount
    (1-x, O3-δ) and a '/' between digits (1/6) are not separators."""
    parts, depth, cur = [], 0, []
    for i, c in enumerate(s):
        if c == "(":
            depth += 1
        elif c == ")":
            depth = max(depth - 1, 0)
        frac_bar = c == "/" and 0 < i < len(s) - 1 and (
            _dg(s[i - 1]) or s[i - 1] == ")" or s[i - 1] in "xyz") \
            and _dg(s[i + 1]) and _fraction_end_ok(s, i + 1)
        if depth == 0 and c in seps and not frac_bar and \
                not (c == "-" and _starts_amount(s, i + 1)):
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


def _split_after_oxygen(s):
    """'...CoO3-δ-xCeO2' -> ['...CoO3-δ-x', 'CeO2']: a cation written after the
    oxygen count starts a second compound."""
    out, start, depth, i = [], 0, 0, 0
    while i < len(s):
        c = s[i]
        depth += (c == "(") - (c == ")")
        if depth == 0 and c == "O" and not (i + 1 < len(s) and _lo(s[i + 1])):
            _, _, j = _read_amount(s, i + 1)
            sym, _ = _read_symbol(s, j)
            if sym is not None and sym not in ANIONS:
                out.append(s[start:j])
                start = j
            i = max(j, i + 1)
            continue
        i += 1
    out.append(s[start:])
    return [p for p in out if p.strip()]


def _a_site_type(e, elements):
    """Large cations that take the 12-coordinate A site of a perovskite."""
    p = elements[e]
    return 57 <= p["Z"] <= 71 or e in ("Y", "Bi", "Pb") or p["group"] in (1, 2)


def _bare_abo3(items, elements):
    """'LaNiFeCoCrMnO3', 'LaSrNiCoMnFeCuO3': a run of symbols with no amount and no
    bracket, closed by O3, is the ABO3 notation with the brackets left out. Read the
    leading large cations as one A site and the rest as one B site (label-free: only
    the written oxygen count and the element table are used). Returns the cation
    amounts, or None when the pattern does not apply."""
    if not elements or any(it[0] != "el" for it in items):
        return None
    cats = [it for it in items if it[1] not in ANIONS]
    oxy = [it for it in items if it[1] == "O"]
    if len(cats) < 4 or len(oxy) != 1 or oxy[0][2] is None or abs(oxy[0][2] - 3) > 0.5:
        return None
    if any(it[2] is not None or it[1] not in elements for it in cats):
        return None
    syms = [it[1] for it in cats]
    n_a = 0
    while n_a < len(syms) and _a_site_type(syms[n_a], elements):
        n_a += 1
    if n_a == 0 or len(syms) - n_a < 2 or \
            any(_a_site_type(e, elements) for e in syms[n_a:]):
        return None
    out = {}
    for k, e in enumerate(syms):
        out[e] = out.get(e, 0.0) + 1.0 / (n_a if k < n_a else len(syms) - n_a)
    return out


def _parse_chunk(chunk, elements):
    """One chunk -> (cations, anions, flags), or None when it is not a formula."""
    chunk = chunk.strip().strip(".")
    m = re.match(r"^[0-9]+(?:\.[0-9]+)?(?=[A-Z(])", chunk)       # "0.5BaTiO3"
    if m:
        chunk = chunk[m.end():]
    if not chunk:
        return None
    items, j, ok, flags = _parse_items(chunk, 0, 0)
    if not ok or j != len(chunk) or not items:
        return None
    abo3 = _bare_abo3(items, elements)
    if abo3 is not None:
        an = {it[1]: it[2] for it in items if it[1] in ANIONS}
        return abo3, an, flags | {"abo3_bare_sequence"}
    cat, an, f2 = _evaluate_best(items, elements)
    cat = {e: v for e, v in cat.items() if v > 1e-12}
    return (cat, an, flags | f2) if cat else None


def _oxide_sum(s):
    """'La2O3+TiO2+Nb2O5+WO3+Ga2O3', '10La2O3-20TiO2-10Nb2O5-...' -> cation amounts."""
    pieces = re.split(r"\s*[+" + _DASHES + r"]\s*|\s*-\s*(?=[0-9]|[A-Z])", s)
    if len(pieces) < 3:
        return None
    out = {}
    for p in pieces:
        m = _BINARY_OXIDE.match(p)
        if not m or m.group(2) in ANIONS or m.group(2) not in SYMBOLS:
            return None
        coef = float(m.group(1) or 1.0)
        out[m.group(2)] = out.get(m.group(2), 0.0) + coef * float(m.group(3) or 1.0)
    return out


def _single(p):
    return len(p[0]) == 1 and not p[1] and list(p[0].values())[0] == 1.0


def _combine(pieces):
    """Combine the pieces of one list-separated run into candidate readings."""
    if len(pieces) == 1:
        return [pieces[0]]
    singles = [p for p in pieces if _single(p)]
    if len(singles) == len(pieces):
        # Co-Cr-Fe-Mn-Ni, "Mn, Fe, Mg, Zn, Ca": an element list, equal amounts
        return [({e: 1.0 for p in pieces for e in p[0]}, {}, {"element_list"})]
    if len(singles) == len(pieces) - 1:
        # one formula plus lone elements from prose ("Ca-doped (..)AlO3",
        # "Pt-(LaCe..)O"): the lone elements are dopants or supported metals
        main = [p for p in pieces if not _single(p)][0]
        return [(main[0], main[1], main[2] | {"minor_component_dropped"})]
    return pieces


_DOPANT_PREFIX = re.compile(
    r"(?<![A-Za-z])[A-Z][a-z]?-(?:doped|modified|decorated|substituted|loaded|supported"
    r"|coated|containing)(?![A-Za-z])\s*")


def _parse_string(s, elements):
    flags = set()
    if ":" in s:                                      # dopant suffix
        s = s.split(":")[0]
        flags.add("dopant_dropped")
    t = _DOPANT_PREFIX.sub("", s)                     # "Zn-doped Co-Fe-Cr-Mn-Ni spinel"
    if t != s:
        s = t.strip()
        flags.add("dopant_dropped")
    ox = _oxide_sum(s)
    if ox:
        return ox, {}, flags | {"oxide_sum"}
    alternatives = []
    for alt in _split_top(s, _ALT_SEPS):
        pieces = []
        for ch in _split_top(alt, _LIST_SEPS):
            for piece in _split_after_oxygen(ch) if "O" in ch else [ch]:
                r = _parse_chunk(piece, elements)
                if r is not None:
                    pieces.append(r)
        if pieces:
            alternatives.extend(_combine(pieces))
    if not alternatives:
        return {}, {}, flags | {"unparsed"}
    if len(alternatives) > 1:
        if all(_single(a) for a in alternatives) and len(alternatives) >= 3:
            # "Ti/V/Cr/Nb/Mo/W/Al/Cu oxide"
            return ({e: 1.0 for a in alternatives for e in a[0]}, {},
                    flags | {"element_list"})
        flags.add("composite_main_phase")
    cat, an, f = max(alternatives, key=lambda a: len(a[0]))
    return cat, an, flags | f


def parse(text, elements=None):
    """Parse a formula as written in a paper. ``amounts`` is empty when no cation
    could be read. ``elements`` (the property table) enables the charge-balance
    choice between readings of ambiguous groups."""
    if text is None or (isinstance(text, float) and text != text):
        return Parsed({}, {"empty"})
    best = None
    for s, f0 in _placeholder_variants(normalize(text)):
        cat, an, f = _parse_string(s, elements)
        n = sum(1 for v in cat.values() if v > 0)
        if best is None or (best[0] < 5 and n > best[0]):
            best = (n, Parsed(cat, f | f0, an.get("O")))
    return best[1]


def cation_fractions(text, elements=None):
    """Normalised cation fractions; trace cations (< MIN_CATION_FRACTION) removed and
    the remainder renormalised."""
    fr = parse(text, elements).fractions()
    keep = {e: v for e, v in fr.items() if v >= MIN_CATION_FRACTION}
    tot = _nsum(keep.values())
    return {e: v / tot for e, v in sorted(keep.items())} if tot else {}
