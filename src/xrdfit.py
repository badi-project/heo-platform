"""Confront a prediction with a measured powder pattern.

The prediction gives a prototype and a starting lattice parameter; a measured pattern
gives peak positions. Indexing the peaks on the cubic prototype and refining ``a`` by
least squares answers the experimental questions: is the predicted family the one
observed, what is the real lattice parameter, and which peaks does that family not
explain (a secondary phase leaves peaks unindexed).

    from predict import Predictor
    import xrdfit
    P = Predictor()
    tt, counts = xrdfit.read_pattern("scan.xy")
    r = P.predict("(FeCoNiCrMn)3O4")
    found, fits = xrdfit.compare_measured(P, tt, counts, r["fractions"])
    fits[0]["proto"], fits[0]["a"], fits[0]["unindexed"]

This is not a Rietveld refinement: peak positions only, cubic prototypes only, one phase
at a time. Mirrors findPeaks / refineLattice / compareMeasured of app/heo_engine.js, which
the page runs on a file dropped by the user; tools/parity_check.py compares the two.
"""
import math
import re

import features

# K-alpha1, K-alpha2, K-beta wavelengths in angstrom
RADIATION = {
    "Cu": (1.54056, 1.54439, 1.39222), "Co": (1.78897, 1.79285, 1.62079),
    "Mo": (0.70930, 0.71359, 0.63229), "Cr": (2.28970, 2.29361, 2.08487),
    "Fe": (1.93604, 1.93998, 1.75661),
}
# widths and tolerances in degrees 2-theta; window and grid_step relative to a
XFIT = {"smooth": 0.04, "background": 1.2, "isolate": 0.08, "n_sigma": 6, "min_rel": 0.02,
        "window": 0.06, "grid_step": 0.0005, "tol_search": 0.25, "tol_assign": 0.20,
        "tol_final": 0.10, "min_pred": 2, "strong": 10, "tol_line": 0.12, "edge": 0.3,
        "asymmetry": 0.12, "valley": 0.5, "prominence": 0.3}
# Accuracy of the refined lattice parameter established on synthetic patterns
# (QC/xrd_fit_validation.txt), by signal-to-noise ratio of the strongest peak: the quoted
# uncertainty is never below it, whatever the statistical error of the fit. Under SNR_MIN
# the data are noisier than anything validated.
SNR_CLEAN, ACCURACY_CLEAN, ACCURACY_NOISY, SNR_MIN = 500, 5e-4, 2.5e-3, 70
DEG = 180 / math.pi


def _round(x):
    """Math.round of JavaScript: halves go up (Python's round() sends them to even)."""
    return int(math.floor(x + 0.5))


def _median(values):
    v = sorted(values)
    m = len(v) >> 1
    if not v:
        return 0.0
    return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2


def _sample_at(tt, y, x):
    """Value of a regularly sampled curve at 2-theta x, by linear interpolation."""
    n = len(tt)
    pos = (x - tt[0]) / ((tt[n - 1] - tt[0]) / (n - 1))
    if pos <= 0:
        return y[0]
    if pos >= n - 1:
        return y[n - 1]
    i = int(math.floor(pos))
    return y[i] + (pos - i) * (y[i + 1] - y[i])


def _shift_line(x, lam1, lam2):
    """2-theta where wavelength lam2 puts the line that lam1 puts at x."""
    st = math.sin(x / (2 * DEG)) * lam2 / lam1
    return 2 * math.asin(st) * DEG if st < 1 else float("nan")


def kalpha2_asymmetry(tt, net, peaks, rad):
    """Is the K-alpha2 line in the data? A K-alpha1 peak alone is symmetric; with
    K-alpha2 it carries half its height again on the high-angle side. Median, over the
    strongest peaks above 40 degrees, of (signal at the K-alpha2 position - signal at the
    mirror position) over the peak height: about 0 without K-alpha2, 0.2 to 0.5 with it."""
    hmax = max([p["h"] for p in peaks] + [0])
    top = sorted((p for p in peaks if p["tt"] >= 40 and p["h"] >= 0.1 * hmax),
                 key=lambda p: (-p["h"], p["tt"]))[:8]
    r = []
    for p in top:
        d = _shift_line(p["tt"], rad[0], rad[1]) - p["tt"]
        if math.isnan(d) or p["tt"] + d > tt[-1]:
            continue
        r.append((_sample_at(tt, net, p["tt"] + d) - _sample_at(tt, net, p["tt"] - d)) / p["h"])
    return _median(r) if len(r) >= 2 else 0


def strip_kalpha2(tt, y, rad):
    """Rachinger correction: remove from each point half of the K-alpha1 signal found
    where K-alpha1 puts the line that K-alpha2 puts here. What remains is the K-alpha1
    pattern, whose peak tops are no longer pulled towards high angles."""
    n = len(tt)
    step = (tt[n - 1] - tt[0]) / (n - 1)
    out = [0.0] * n
    for i in range(n):
        pos = (_shift_line(tt[i], rad[1], rad[0]) - tt[0]) / step
        c = 0
        if pos > 0 and i > 0:
            k = min(int(math.floor(pos)), i - 1)
            f = min(pos - k, 1)
            c = out[k] + f * (out[k + 1] - out[k]) if k + 1 <= i - 1 else out[k]
        out[i] = y[i] - 0.5 * c
    return out


def _pick_peaks(tt, net, noise, level):
    """Local maxima of the net signal above the noise; position from the parabola through
    the three points at the top. ``level`` is the smoothed raw signal, background
    included."""
    n = len(tt)
    step = (tt[n - 1] - tt[0]) / (n - 1)
    mx = max([0.0] + net)
    # the net signal is mostly background: its median and spread measure what the noise
    # leaves after smoothing and background removal, better than the raw scatter does
    base = _median(net)
    spread = 1.4826 * _median(abs(v - base) for v in net)
    wp = max(1, _round(XFIT["isolate"] / step))
    wv = max(1, _round(XFIT["valley"] / step))
    floor = max(base + XFIT["n_sigma"] * max(spread, noise), XFIT["min_rel"] * mx)
    level_med = max(_median(level), 1e-300)
    peaks = []
    for i in range(1, n - 1):
        if net[i] < floor:
            continue
        top = True
        for j in range(max(0, i - wp), min(n - 1, i + wp) + 1):
            if net[j] > net[i] or (net[j] == net[i] and j < i):
                top = False
                break
        if not top:
            continue
        # prominence: a bump of the noise on the tail of a strong peak is a local maximum
        # too, but it does not rise above the valley that separates it from that peak
        valley = float("-inf")
        for direction in (-1, 1):
            low, j = net[i], i + direction
            while 0 <= j < n and abs(j - i) <= wv and net[j] <= net[i]:
                if net[j] < low:
                    low = net[j]
                j += direction
            if low > valley:
                valley = low
        # counting noise grows as the square root of the signal: on the flank of a strong
        # peak it is larger than on the background where ``noise`` was measured
        local = noise * math.sqrt(max(level[i], 0) / level_med)
        if net[i] - valley < max(XFIT["prominence"] * net[i], XFIT["n_sigma"] * local):
            continue
        y0, y1, y2 = net[i - 1], net[i], net[i + 1]
        den = y0 - 2 * y1 + y2
        dx = 0.5 * (y0 - y2) / den if den < 0 else 0
        peaks.append({"tt": tt[i] + dx * step, "h": y1 - 0.25 * (y0 - y2) * dx})
    return peaks, mx


def find_peaks(tt, counts, radiation="Cu"):
    """Peaks of a measured pattern: moving average, SNIP background, K-alpha2 removed
    when the data carry it, then the peak search.
    Returns {"peaks": [{"tt", "h"}], "net", "noise", "max", "asym", "stripped"}."""
    rad = RADIATION[radiation]
    n = len(tt)
    step = (tt[n - 1] - tt[0]) / (n - 1)
    ws = max(1, _round(XFIT["smooth"] / step))
    sm = [0.0] * n
    for i in range(n):
        acc, k = 0.0, 0
        for j in range(max(0, i - ws), min(n - 1, i + ws) + 1):
            acc += counts[j]
            k += 1
        sm[i] = acc / k
    noise = 1.4826 * _median(abs(counts[i] - sm[i]) for i in range(n)) / math.sqrt(2 * ws + 1)
    bg = list(sm)
    for p in range(max(1, _round(XFIT["background"] / step)), 0, -1):
        nb = list(bg)
        for i in range(p, n - p):
            nb[i] = min(bg[i], (bg[i - p] + bg[i + p]) / 2)
        bg = nb
    net = [sm[i] - bg[i] for i in range(n)]
    peaks, mx = _pick_peaks(tt, net, noise, sm)
    asym = kalpha2_asymmetry(tt, net, peaks, rad)
    stripped = asym > XFIT["asymmetry"]
    if stripped:
        net = strip_kalpha2(tt, net, rad)
        peaks, mx = _pick_peaks(tt, net, noise, sm)
    return {"peaks": peaks, "net": net, "noise": noise, "max": mx, "asym": asym,
            "stripped": stripped}


def tt_of(a, q2, lam):
    """2-theta in degrees of the reflection h2 + k2 + l2 = q2 of a cubic cell, or NaN."""
    st = lam * math.sqrt(q2) / (2 * a)
    return 2 * math.asin(st) * DEG if st < 1 else float("nan")


def _assign(peaks, refl, a, lam, zero, tol):
    """Nearest measured peak to each reflection within tol, each measured peak used once."""
    pairs = []
    for ri, r in enumerate(refl):
        tc = tt_of(a, r["q2"], lam) + zero
        if math.isnan(tc):
            continue
        best, bd = -1, tol
        for j, p in enumerate(peaks):
            d = abs(p["tt"] - tc)
            if d <= bd:
                bd, best = d, j
        if best >= 0:
            pairs.append((ri, best, bd))
    pairs.sort(key=lambda p: (p[2], p[0]))
    used, out = set(), []
    for p in pairs:
        if p[1] not in used:
            used.add(p[1])
            out.append(p)
    return sorted(out, key=lambda p: p[0])


def _fit(peaks, refl, pairs, a, lam, with_zero):
    """Least squares on 2-theta: lattice parameter and, if asked, a zero shift."""
    zero, cov = 0.0, 0.0
    fit_zero = bool(with_zero and len(pairs) >= 4)
    for _ in range(6):
        saa = saz = szz = ra = rz = 0.0
        for ri, j, _d in pairs:
            tc = tt_of(a, refl[ri]["q2"], lam)
            d = -2 * DEG * math.tan(tc / (2 * DEG)) / a        # d(2theta)/da
            res = peaks[j]["tt"] - zero - tc
            saa += d * d
            saz += d
            szz += 1
            ra += d * res
            rz += res
        if fit_zero:
            det = saa * szz - saz * saz
            a, zero = a + (ra * szz - rz * saz) / det, zero + (saa * rz - saz * ra) / det
            cov = szz / det
        else:
            a += ra / saa
            cov = 1 / saa
    ss = 0.0
    for ri, j, _d in pairs:
        ss += (peaks[j]["tt"] - zero - tt_of(a, refl[ri]["q2"], lam)) ** 2
    dof = len(pairs) - (2 if fit_zero else 1)
    return {"a": a, "zero": zero, "rms": math.sqrt(ss / len(pairs)),
            "sigma_a": math.sqrt(ss / dof * cov) if dof > 0 else None, "fit_zero": fit_zero}


def refine_lattice(P, found, proto, frac, a0, tt_min, tt_max, radiation="Cu", zero=True):
    """Index the measured peaks on one prototype and refine its lattice parameter.
    ``a0`` is the starting value (the prediction); the search explores a0 x (1 +/- window).
    Returns None when the prototype has no cubic model, a dictionary otherwise — with
    ``a`` None when fewer than two reflections could be matched."""
    rad = RADIATION[radiation]
    lam = rad[0]
    peaks = found["peaks"]
    if not peaks or not a0 or proto not in P.xrd["prototypes"]:
        return None
    # reflections that can fall in the measured range for any lattice parameter explored
    every = P.simulate_pattern(proto, frac, a0 * (1 - XFIT["window"]), 180, lam)
    fitted = [r for r in every if r["intensity"] >= XFIT["min_pred"]]

    def in_range(a, z):
        out = []
        for r in fitted:
            t = tt_of(a, r["q2"], lam) + z
            if tt_min <= t <= tt_max:
                out.append(r)
        return out
    hmax = max(p["h"] for p in peaks)
    best = None
    n_grid = _round(XFIT["window"] / XFIT["grid_step"])
    for g in range(-n_grid, n_grid + 1):
        a = a0 * (1 + g * XFIT["grid_step"])
        score = 0.0
        for r in in_range(a, 0):
            tc = tt_of(a, r["q2"], lam)
            bd, bh = XFIT["tol_search"], 0
            for p in peaks:
                d = abs(p["tt"] - tc)
                if d < bd:
                    bd, bh = d, p["h"]
            if bh > 0:
                score += (math.sqrt(r["intensity"]) * (1 - (bd / XFIT["tol_search"]) ** 2)
                          * math.sqrt(bh / hmax))
        if best is None or score > best[1] + 1e-12:
            best = (a, score)
    refl = in_range(best[0], 0)
    pairs = _assign(peaks, refl, best[0], lam, 0, XFIT["tol_assign"])
    if len(pairs) < 2:
        return {"proto": proto, "a": None, "n": len(pairs), "n_pred": len(refl)}
    fit = _fit(peaks, refl, pairs, best[0], lam, zero)
    refl = in_range(fit["a"], fit["zero"])
    pairs = _assign(peaks, refl, fit["a"], lam, fit["zero"], XFIT["tol_final"])
    if len(pairs) < 2:
        return {"proto": proto, "a": None, "n": len(pairs), "n_pred": len(refl)}
    fit = _fit(peaks, refl, pairs, fit["a"], lam, zero)

    matched = {j for _ri, j, _d in pairs}
    matches = []
    for ri, j, _d in pairs:
        r = refl[ri]
        tc = tt_of(fit["a"], r["q2"], lam) + fit["zero"]
        matches.append({"hkl": r["hkl"], "q2": r["q2"], "intensity": r["intensity"],
                        "tt_obs": peaks[j]["tt"], "tt_calc": tc,
                        "delta": peaks[j]["tt"] - tc, "h": 100 * peaks[j]["h"] / hmax})
    # a measured peak sitting where K-alpha2 or K-beta puts an indexed reflection belongs
    # to that reflection, not to another phase
    lines = []
    for m in matches:
        for k in (1, 2):
            t = tt_of(fit["a"], m["q2"], rad[k])
            if not math.isnan(t):
                lines.append(t + fit["zero"])
    # so does a peak sitting on a reflection too weak to take part in the fit
    for r in every:
        if r["intensity"] < XFIT["min_pred"]:
            t = tt_of(fit["a"], r["q2"], lam)
            if not math.isnan(t):
                lines.append(t + fit["zero"])
    unindexed, explained, total = [], 0.0, 0.0
    for j, p in enumerate(peaks):
        total += p["h"]
        if j in matched or any(abs(t - p["tt"]) <= XFIT["tol_line"] for t in lines):
            explained += p["h"]
        else:
            unindexed.append({"tt": p["tt"], "h": 100 * p["h"] / hmax})
    # a strong reflection is expected only clear of the ends of the scan
    strong = [r for r in refl if r["intensity"] >= XFIT["strong"]
              and tt_min + XFIT["edge"] <= tt_of(fit["a"], r["q2"], lam) + fit["zero"]
              <= tt_max - XFIT["edge"]]
    strong_found = sum(1 for r in strong if any(m["q2"] == r["q2"] for m in matches))
    share = explained / total if total > 0 else 0
    return {"proto": proto, "a": fit["a"], "sigma_a": fit["sigma_a"], "zero": fit["zero"],
            "fit_zero": fit["fit_zero"], "rms": fit["rms"], "n": len(pairs),
            "n_pred": len(refl), "n_strong": len(strong), "strong_found": strong_found,
            "stripped": found["stripped"],
            "explained": share, "matches": matches, "unindexed": unindexed,
            "score": (strong_found / len(strong) if strong else 0) * share}


def compare_measured(P, tt, counts, frac, radiation="Cu", zero=True):
    """The same measured pattern indexed on every cubic prototype, best explanation
    first: each one starts from the lattice parameter the platform predicts for it.
    Returns (peaks found, list of fits). ``found`` carries the signal-to-noise ratio and
    the quality class it implies. A fit carries the refined ``a``, its statistical error
    ``sigma_a``, the uncertainty to quote ``a_uncertainty`` (never below the accuracy
    validated for that quality class), the density at the refined parameter and the
    unindexed peaks."""
    found = find_peaks(tt, counts, radiation)
    found["snr"] = found["max"] / found["noise"] if found["noise"] > 0 else float("inf")
    found["quality"] = ("clean" if found["snr"] >= SNR_CLEAN
                        else "noisy" if found["snr"] >= SNR_MIN else "below")
    acc = ACCURACY_CLEAN if found["quality"] == "clean" else ACCURACY_NOISY
    fits = []
    for proto in P.model["classes"]:
        if proto not in P.xrd["prototypes"]:
            continue
        a_th = P.thermo_lattice(proto, frac)
        a0 = a_th if a_th is not None else P.geometric_lattice(proto, frac)
        r = refine_lattice(P, found, proto, frac, a0, tt[0], tt[-1], radiation, zero)
        if r:
            r["a0"] = a0
            r["density_g_cm3"] = P.xray_density(proto, frac, r["a"]) if r["a"] else None
            r["a_uncertainty"] = max(r["sigma_a"] or 0.0, acc * r["a"]) if r["a"] else None
            fits.append(r)
    fits.sort(key=lambda r: (-(r.get("score") or 0), r["proto"]))
    return found, fits


def parse_pattern(text):
    """Text of a diffractometer export -> (2-theta list, intensity list): two columns
    separated by spaces, tabs, commas or semicolons, decimal point or comma; or
    PANalytical .xrdml."""
    tt, counts = [], []
    if "<xrdMeasurement" in text:
        pos = re.search(r'<positions[^>]*axis="2Theta"[^>]*>\s*<startPosition>([^<]+)'
                        r'</startPosition>\s*<endPosition>([^<]+)</endPosition>', text)
        cnt = re.search(r"<(?:intensities|counts)[^>]*>([^<]+)</(?:intensities|counts)>", text)
        if pos and cnt:
            y = [float(v) for v in cnt.group(1).split()]
            a, b = float(pos.group(1)), float(pos.group(2))
            for i, v in enumerate(y):
                tt.append(a + (b - a) * i / (len(y) - 1))
                counts.append(v)
    else:
        for raw in text.splitlines():
            line = raw.strip()
            if not line or re.match(r"[#;!%'\"*a-zA-Z_]", line):
                continue
            if re.search(r"[;\t]", line):
                tok = [t.strip().replace(",", ".", 1) for t in re.split(r"[;\t]+", line)]
            elif re.search(r"\s", line):
                tok = [re.sub(r",$", "", t).replace(",", ".", 1) for t in line.split()]
            else:
                tok = line.split(",")
            try:
                x, y = float(tok[0]), float(tok[1])
            except (ValueError, IndexError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                tt.append(x)
                counts.append(y)
    order = sorted(range(len(tt)), key=lambda i: tt[i])
    return [tt[i] for i in order], [counts[i] for i in order]


def read_pattern(path):
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        return parse_pattern(fh.read())


def synthetic_pattern(P, proto, frac, a, radiation="Cu", zero=0.0, fwhm=0.15, step=0.02,
                      tt_min=10.0, tt_max=80.0, counts=5000.0, background=300.0, noise=True,
                      extra=(), seed=12345, kalpha2=True):
    """A laboratory-like pattern of known lattice parameter, for validation: K-alpha1 and
    K-alpha2 lines, pseudo-Voigt width, zero shift, decaying background, counting noise
    from a fixed-seed generator, and optional foreign lines ``extra`` = [(2-theta, I)].
    The refinement is judged on how well it gives ``a`` back."""
    rad = RADIATION[radiation]
    sticks = []
    for p in P.simulate_pattern(proto, frac, a, 100, rad[0]):
        sticks.append({"two_theta": p["two_theta"] + zero, "intensity": p["intensity"]})
        t2 = tt_of(a, p["q2"], rad[1])
        if kalpha2 and not math.isnan(t2):
            sticks.append({"two_theta": t2 + zero, "intensity": p["intensity"] / 2})
    for t, i in extra:
        sticks.append({"two_theta": t, "intensity": i})
    state = [seed]

    def rnd():
        state[0] = (state[0] * 1103515245 + 12345) % 2147483648
        return state[0] / 2147483648
    xs, ys = [], []
    for x, y in P.profile(sticks, fwhm=fwhm, step=step, tt_min=tt_min, tt_max=tt_max):
        c = counts * y / 100 + background * (0.4 + math.exp(-(x - 10) / 15))
        if noise:
            c += math.sqrt(c) * (features.nsum(rnd() for _ in range(12)) - 6)
        xs.append(x)
        ys.append(max(0.0, c))
    return xs, ys
