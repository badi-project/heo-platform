/* HEO designer — prediction engine.
   Mirrors features.py (descriptors) and the scikit-learn HistGradientBoosting
   raw-prediction path. Numerical parity with Python is checked in parity_check.txt. */

const R_O6 = 1.40, R_O4 = 1.38;

/* ---------- descriptors: exact mirror of features.py ---------- */
/* CPython's built-in sum() uses Neumaier compensated summation on floats (3.12+),
   so a naive accumulator in JS drifts ~1 ulp from features.py. That matters here:
   split thresholds can sit exactly on a training value, and a 1-ulp shift flipped
   89 of 1500 trees before this was fixed. nsum reproduces CPython's algorithm, and
   products are rounded into a Float64Array first so ARM64 cannot fuse the
   multiply-add. Verified bit-identical in parity_check.txt. */
function nsum(arr) {
  let s = 0.0, c = 0.0;
  for (let i = 0; i < arr.length; i++) {
    const x = arr[i], t = s + x;
    c += (Math.abs(s) >= Math.abs(x)) ? ((s - t) + x) : ((x - t) + s);
    s = t;
  }
  return s + c;
}

function wsum(vals, ws) {
  const n = vals.length, pr = new Float64Array(n);
  for (let i = 0; i < n; i++) pr[i] = vals[i] * ws[i];
  return nsum(pr);
}

function wstats(vals, ws) {
  const m = wsum(vals, ws);
  const n = vals.length, dev = new Float64Array(n);
  for (let i = 0; i < n; i++) dev[i] = (vals[i] - m) ** 2;
  const v = wsum(Array.from(dev), ws);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return [m, Math.sqrt(Math.max(v, 0)), lo, hi, hi - lo];
}

function abSplit(frac) {
  const order = Object.keys(frac).sort().sort(
    (a, b) => ELEM[b].r_shannon_XII - ELEM[a].r_shannon_XII);
  const A = {}, B = {}; let cum = 0;
  for (const e of order) {
    const x = frac[e];
    if (cum < 0.5 - 1e-9) {
      const take = Math.min(x, 0.5 - cum);
      A[e] = take;
      if (x - take > 1e-9) B[e] = x - take;
      cum += take;
    } else B[e] = x;
  }
  const sa = nsum(Object.values(A)) || 1;
  const sb = nsum(Object.values(B)) || 1;
  const An = {}, Bn = {};
  for (const k in A) An[k] = A[k] / sa;
  for (const k in B) Bn[k] = B[k] / sb;
  return [An, Bn];
}

function featurize(frac) {
  const els = Object.keys(frac).sort();
  const ws = els.map(e => frac[e]);
  const vals = [];
  for (const p of PROPS) {
    const xs = els.map(e => ELEM[e][p]);
    for (const v of wstats(xs, ws)) vals.push(v);
  }
  const pos = ws.filter(w => w > 0);
  vals.push(-wsum(pos.map(Math.log), pos));       // s_config
  vals.push(els.length);                          // n_cat

  const r6 = els.map(e => ELEM[e].r_shannon_VI);
  const r6m = wsum(r6, ws);
  vals.push(Math.sqrt(wsum(r6.map(v => (1 - v / r6m) ** 2), ws)));   // delta_r

  const X = els.map(e => ELEM[e].X);
  const Xm = wsum(X, ws);
  vals.push(Math.sqrt(wsum(X.map(v => (v - Xm) ** 2), ws)) / Xm);    // delta_X_rel

  const ox = els.map(e => ELEM[e].ox);
  const oxm = wsum(ox, ws);
  vals.push(oxm);                                 // mean_ox
  vals.push(Math.max(...ox) - Math.min(...ox));   // ox_spread
  vals.push(wsum(els.map(e => ELEM[e].n_ox), els.map(e => frac[e])));  // redox_flex
  vals.push(oxm / 2);                             // o_per_cation

  const [A, B] = abSplit(frac);
  const ka = Object.keys(A), kb = Object.keys(B);
  const rA = wsum(ka.map(e => ELEM[e].r_shannon_XII), ka.map(e => A[e]));
  const rB = wsum(kb.map(e => ELEM[e].r_shannon_VI), kb.map(e => B[e]));
  vals.push(rA); vals.push(rB); vals.push(rA / rB);
  vals.push((rA + R_O6) / (Math.SQRT2 * (rB + R_O6)));   // goldschmidt_t
  vals.push(rB / R_O6);                                   // octahedral_factor

  const fam = { RE: 0, AE: 0, alkali: 0, TM3d: 0, TM4d5d: 0, pblock: 0 };
  for (const e of els) {
    const z = ELEM[e].Z, g = ELEM[e].group, row = ELEM[e].row, x = frac[e];
    if ((z >= 57 && z <= 71) || e === "Y" || e === "Sc") fam.RE += x;
    else if (g === 2) fam.AE += x;
    else if (g === 1) fam.alkali += x;
    else if (g >= 3 && g <= 12 && row === 4) fam.TM3d += x;
    else if (g >= 3 && g <= 12 && (row === 5 || row === 6)) fam.TM4d5d += x;
    else if (g >= 13) fam.pblock += x;
  }
  for (const k of ["RE", "AE", "alkali", "TM3d", "TM4d5d", "pblock"]) vals.push(fam[k]);
  for (const e of ONEHOT) vals.push(e in frac ? 1 : 0);
  return vals;
}

/* ---------- HistGradientBoosting raw-prediction path ---------- */
/* node = [feature_idx | -1 for leaf, threshold, left, right, leaf_value, missing_go_to_left] */
function treeValue(nodes, x) {
  let i = 0;
  for (;;) {
    const n = nodes[i];
    if (n[0] < 0) return n[4];
    /* scikit-learn's HistGradientBoosting traverses on float64 input (X_DTYPE) and
       compares against float64 thresholds — keep full double precision here, and
       export thresholds unrounded, or splits flip (checked in parity_check.txt). */
    const v = x[n[0]];
    if (v === null || v === undefined || Number.isNaN(v)) i = n[5] ? n[2] : n[3];
    else i = (v <= n[1]) ? n[2] : n[3];
  }
}

function rawPredict(model, x) {
  const K = model.classes.length;
  const raw = model.baseline.slice();
  for (const it of model.trees)
    for (let k = 0; k < K; k++) raw[k] += treeValue(it[k], x);
  return raw;
}

function softmax(raw, temperature) {
  const T = temperature || 1.0;
  const z = raw.map(v => v / T);
  const mx = Math.max(...z);
  const ex = z.map(v => Math.exp(v - mx));
  const s = ex.reduce((p, c) => p + c, 0);
  return ex.map(v => v / s);
}

/* ---------- applicability domain ---------- */
function adDistance(x) {
  const z = AD.idx.map((j, i) => (x[j] - AD.mean[i]) / AD.scale[i]);
  const ds = [];
  for (const row of AD.rows) {
    let d = 0;
    for (let i = 0; i < z.length; i++) d += (z[i] - row[i]) ** 2;
    ds.push(d);
  }
  ds.sort((a, b) => a - b);
  let acc = 0; for (let i = 0; i < AD.k; i++) acc += Math.sqrt(ds[i]);
  return acc / AD.k;
}

/* ---------- lattice parameter (calibrated geometric model) ---------- */
function rEff(kind, frac) {
  const sum = (key, f) => {
    const k = Object.keys(f);
    return wsum(k.map(e => ELEM[e][key]), k.map(e => f[e]));
  };
  if (kind === "mean_VI") return sum("r_shannon_VI", frac);
  if (kind === "mean_IV") return sum("r_shannon_IV", frac);
  if (kind === "mean_VIII") return sum("r_shannon_VIII", frac);
  const [A, B] = abSplit(frac);
  if (kind === "rB_VI") return sum("r_shannon_VI", B);
  if (kind === "rA_VIII") return sum("r_shannon_VIII", A);
  return null;
}

function predictA(proto, frac) {
  if (!(proto in LATTICE.geo)) return null;
  const [kind, rO] = LATTICE.geo[proto];
  return LATTICE.kcal[proto] * (rEff(kind, frac) + rO);
}

/* ---------- simulated powder pattern ---------- */
function fAtom(el, s) {
  const s2 = s * s;
  let g = 0;
  for (const [a, b] of XRD.scattering[el]) g += a * Math.exp(-b * s2);
  return XRD.Z[el] - 41.78214 * s2 * g;
}

function siteOccupancies(proto, frac) {
  const P = XRD.prototypes[proto];
  const occ = {};
  if (P.ordered) {
    const [A, B] = abSplit(frac);
    for (const t in P.nsites) {
      if (t === "A") occ[t] = A;
      else if (t === "B") occ[t] = B;
      else occ[t] = { O: 1 };
    }
  } else {
    for (const t in P.nsites) occ[t] = (t[0] === "O") ? { O: 1 } : frac;
  }
  return occ;
}

function simulatePattern(proto, frac, a, ttMax) {
  const P = XRD.prototypes[proto];
  if (!P) return [];
  const occ = siteOccupancies(proto, frac);
  const lam = XRD.wavelength, sMax = Math.sin((ttMax / 2) * Math.PI / 180);
  const acc = {};
  for (const e of P.refl) {
    const st = lam * Math.sqrt(e.q2) / (2 * a);
    if (st > sMax || st >= 1) continue;
    const s = st / lam;
    let Fr = 0, Fi = 0;
    for (const t in e.G) {
      let f = 0;
      for (const el in occ[t]) f += occ[t][el] * fAtom(el, s);
      Fr += f * e.G[t][0]; Fi += f * e.G[t][1];
    }
    const th = Math.asin(st);
    const lp = (1 + Math.cos(2 * th) ** 2) / (Math.sin(th) ** 2 * Math.cos(th));
    const I = (Fr * Fr + Fi * Fi) * lp * e.n;
    const tt = (2 * th * 180 / Math.PI).toFixed(3);
    acc[tt] = (acc[tt] || 0) + I;
    if (!acc["hkl_" + tt]) acc["hkl_" + tt] = e.hkl.join("");
  }
  const peaks = Object.keys(acc).filter(k => !k.startsWith("hkl_"))
    .map(k => ({ tt: parseFloat(k), I: acc[k], hkl: acc["hkl_" + k] }));
  const mx = Math.max(...peaks.map(p => p.I), 1e-30);
  return peaks.map(p => ({ ...p, I: 100 * p.I / mx })).filter(p => p.I > 0.4)
    .sort((x, y) => x.tt - y.tt);
}

/* ---------- formula parser: mirror of src/formula.py ---------- */
/* Reads a formula as written in a paper — (Na0.2Bi0.2Ba0.2Sr0.2Ca0.2)TiO3,
   La(CrMnFeCoNi)O3, (FeCoNiCrMn)3O4, Co-Cr-Fe-Mn-Ni ... — into cation fractions, so
   that the page predicts on the same stoichiometry the model was trained on. Every
   database formula gives the same fractions here and in Python (tools/parity_check.py). */
const FX = (() => {
  const X_NOMINAL = 0.1, MIN_FRAC = 0.01;
  const SYMBOLS = new Set(("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr " +
    "Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I " +
    "Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb " +
    "Bi Po At Rn Fr Ra Ac Th Pa U Np Pu").split(" "));
  const ANIONS = { O: -2, F: -1, Cl: -1, Br: -1, I: -1, N: -3, S: -2, Se: -2, H: 0, C: 0,
    P: 0, B: 0 };
  const VARS = { x: X_NOMINAL, y: X_NOMINAL, z: X_NOMINAL, "δ": 0.0, a: 1.0, b: 1.0 };
  const PLACEHOLDERS = ["5TM", "TM", "RE", "Ln", "Me", "M", "A", "B", "X"];
  const SUB_FROM = "₀₁₂₃₄₅₆₇₈₉₊₋ₐₓᵧ", SUB_TO = "0123456789+-axy";
  const SUPER = new Set("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻");
  const DASHES = "–—─", ALT_SEPS = "@/+;|", LIST_SEPS = ",-" + DASHES + " ";
  const ADDITIVE = new Set(["with", "plus", "doped", "decorated", "supported", "loaded",
    "on", "containing", "coated"]);
  const LOWER = new Set("abcdefghijklmnopqrstuvwxyz" + "αβγδεζηθικλμνξοπρστυφχψω");
  const UPPER = new Set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), DIGIT = new Set("0123456789");
  const lo = c => LOWER.has(c), up = c => UPPER.has(c), dg = c => DIGIT.has(c);
  const al = c => LOWER.has(c) || UPPER.has(c);
  const NUM = /[0-9]+(?:\.[0-9]+)?|\.[0-9]+/y;
  const PAREN = /\(([0-9.xyzδ+\-/ ]*[0-9xyzδ][0-9.xyzδ+\-/ ]*)\)/y;
  const BINOX = /^\s*([0-9]+(?:\.[0-9]+)?)?\s*([A-Z][a-z]?)([0-9]+(?:\.[0-9]+)?)?O([0-9]+(?:\.[0-9]+)?)?\s*$/;
  const at = (re, s, j) => { re.lastIndex = j; return re.exec(s); };
  const nsumv = vals => nsum(Array.from(vals));
  const has = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

  function normalize(text) {
    let t = String(text).replace(/<[^>]*>/g, "");
    t = Array.from(t).map(ch => { const k = SUB_FROM.indexOf(ch); return k < 0 ? ch : SUB_TO[k]; })
      .filter(ch => !SUPER.has(ch)).join("");
    t = t.split("−").join("-").split("‐").join("-").split("‑").join("-");
    t = t.replace(new RegExp("(?<=[0-9])[" + DASHES + "](?=[xyz0-9])", "g"), "-");
    t = t.replace(/_\{([^{}]*)\}/g, "$1");
    t = t.split("±").join("-").split("_").join("").split("{").join("(").split("}").join(")");
    t = t.split("[").join("(").split("]").join(")").split(" ").join(" ").split("·").join("+");
    t = t.replace(/O[X](?![a-z])/g, "Ox");
    t = t.replace(/[0-9]+(?:\.[0-9]+)?\s*(?:wt|at|mol)?\.?\s*%/g, "");
    return t.trim();
  }

  function placeholderVariants(t) {
    const pat = new RegExp("\\(?\\s*(" + PLACEHOLDERS.join("|") +
      ")\\s*=\\s*([A-Z][a-z]?(?:\\s*[,/]\\s*[A-Z][a-z]?)*)\\s*\\)?");
    const m = pat.exec(t);
    if (!m) return [[t, []]];
    const ph = m[1], values = m[2].match(/[A-Z][a-z]?/g);
    const base = (t.slice(0, m.index) + " " + t.slice(m.index + m[0].length)).trim();
    const where = new RegExp("(?<![A-Z])" + ph + "(?=[^a-z]|[xyz](?![a-z])|$)", "g");
    return [[base.replace(where, values[0]), ["placeholder_variant"]],
            [base.replace(where, "(" + values.join(",") + ")"), ["placeholder_mixed_site"]]];
  }

  const varAt = (s, j) => j < s.length && has(VARS, s[j]) && !(j + 1 < s.length && lo(s[j + 1]));
  const startsAmount = (s, i) => i < s.length && (dg(s[i]) || s[i] === "." || varAt(s, i) ||
    !!at(PAREN, s, i));

  function readFactor(s, j) {
    let m = at(NUM, s, j);
    if (m) return [parseFloat(m[0]), false, j + m[0].length];
    if (varAt(s, j)) return [VARS[s[j]], "xyz".includes(s[j]), j + 1];
    m = at(PAREN, s, j);
    if (m) {
      const inner = m[1].split(" ").join("");
      const [v, vr, k] = readSum(inner, 0);
      if (v !== null && k === inner.length) return [v, vr, j + m[0].length];
    }
    return [null, false, j];
  }
  function readTerm(s, j) {
    let [v, vr, k] = readFactor(s, j);
    if (v === null) return [null, false, j];
    j = k;
    while (j < s.length) {
      if (s[j] === "/") {
        const [d, v2, k2] = readFactor(s, j + 1);
        if (d === null || d === 0) break;
        v = v / d; vr = vr || v2; j = k2;
      } else if (s[j] === "(" && at(PAREN, s, j)) {
        const [f, v2, k2] = readFactor(s, j);
        if (f === null) break;
        v = v * f; vr = vr || v2; j = k2;
      } else break;
    }
    return [v, vr, j];
  }
  function readSum(s, j) {
    let [v, vr, k] = readTerm(s, j);
    if (v === null) return [null, false, j];
    j = k;
    while (j < s.length && "+-".includes(s[j]) && startsAmount(s, j + 1)) {
      const [t, v2, k2] = readTerm(s, j + 1);
      if (t === null) break;
      v = s[j] === "+" ? v + t : v - t; vr = vr || v2; j = k2;
    }
    return [v, vr, j];
  }
  const readAmount = readSum;

  function readSymbol(s, i) {
    if (i < s.length && up(s[i])) {
      if (i + 1 < s.length && lo(s[i + 1]) && SYMBOLS.has(s.slice(i, i + 2))) return [s.slice(i, i + 2), i + 2];
      if (SYMBOLS.has(s[i])) return [s[i], i + 1];
    }
    return [null, i];
  }

  function parseItems(s, i, depth) {
    const items = [], flags = new Set();
    while (i < s.length) {
      const c = s[i];
      if (c === "(") {
        const [sub, j0, ok, f] = parseItems(s, i + 1, depth + 1);
        f.forEach(x => flags.add(x));
        if (!ok) return [items, j0, false, flags];
        const [amt, vr, j] = readAmount(s, j0);
        if (vr) flags.add("variable");
        items.push({ g: true, items: sub, amt });
        i = j;
      } else if (c === ")") {
        return depth > 0 ? [items, i + 1, true, flags] : [items, i, false, flags];
      } else if (up(c)) {
        const [sym, j0] = readSymbol(s, i);
        if (sym === null || (j0 < s.length && lo(s[j0]) && !varAt(s, j0))) return [items, i, false, flags];
        const [amt, vr, j] = readAmount(s, j0);
        if (vr) flags.add("variable");
        items.push({ g: false, sym, amt });
        i = j;
      } else if (depth > 0 && (",/ -".includes(c) || DASHES.includes(c))) {
        i += 1;
      } else if (depth > 0 && lo(c)) {
        let j = i;
        while (j < s.length && al(s[j])) j++;
        flags.add("prose_in_group");
        if (ADDITIVE.has(s.slice(i, j))) {
          let level = 0;
          while (j < s.length && !(s[j] === ")" && level === 0)) {
            level += (s[j] === "(") - (s[j] === ")"); j++;
          }
          flags.add("minor_component_dropped");
        }
        i = j;
      } else return [items, i, false, flags];
    }
    return [items, i, depth === 0, flags];
  }

  const isCation = it => it.g || !has(ANIONS, it.sym);
  function ambiguousGroups(items, acc) {
    const siblings = items.filter(isCation);
    for (const it of items) {
      if (!it.g) continue;
      ambiguousGroups(it.items, acc);
      const bare = it.items.some(sub => !sub.g && !has(ANIONS, sub.sym) && sub.amt === null);
      if (bare && siblings.length > 1) acc.push(it);
    }
    return acc;
  }
  function evaluate(items, mult, siteSet, cat, an) {
    for (const it of items) {
      if (!it.g) {
        const amt = (it.amt === null ? 1.0 : it.amt) * mult;
        const bucket = has(ANIONS, it.sym) ? an : cat;
        bucket[it.sym] = (has(bucket, it.sym) ? bucket[it.sym] : 0.0) + amt;
      } else {
        const gm = (it.amt === null ? 1.0 : it.amt) * mult;
        if (siteSet.has(it)) {
          const inner = {};
          evaluate(it.items, 1.0, siteSet, inner, {});
          const tot = nsumv(Object.values(inner)) || 1.0;
          for (const e of Object.keys(inner))
            cat[e] = (has(cat, e) ? cat[e] : 0.0) + inner[e] / tot * gm;
        } else evaluate(it.items, gm, siteSet, cat, an);
      }
    }
  }
  function chargeMisfit(cat, an) {
    if (!has(an, "O")) return null;
    const pos = nsumv(Object.keys(cat).filter(e => has(ELEM, e)).map(e => cat[e] * ELEM[e].ox));
    const neg = nsumv(Object.keys(an).map(a => -an[a] * ANIONS[a]));
    return Math.abs(pos - neg) / Math.max(neg, 1e-9);
  }
  function evaluateBest(items) {
    const amb = ambiguousGroups(items, []).slice(0, 4);
    let best = null;
    for (let code = 0; code < (1 << amb.length); code++) {
      /* itertools.product([True, False], repeat=n): True first, leftmost slowest */
      const choice = amb.map((_, k) => ((code >> (amb.length - 1 - k)) & 1) === 0);
      const siteSet = new Set(amb.filter((_, k) => choice[k]));
      const cat = {}, an = {};
      evaluate(items, 1.0, siteSet, cat, an);
      const mis = chargeMisfit(cat, an);
      const key = mis === null ? 0.0 : mis;
      if (best === null || key < best[0] - 1e-9) best = [key, cat, an, choice];
      if (mis === null) break;
    }
    const flags = new Set();
    if (amb.length) {
      flags.add("ambiguous_group");
      if (!best[3].every(Boolean)) flags.add("per_element_multiplier");
    }
    return [best[1], best[2], flags];
  }

  function fractionEndOk(s, i) {
    const m = at(NUM, s, i);
    const k = m ? i + m[0].length : i;
    return k >= s.length || "()-+ ".includes(s[k]) || readSymbol(s, k)[0] !== null;
  }
  function splitTop(s, seps) {
    const parts = []; let depth = 0, cur = "";
    for (let i = 0; i < s.length; i++) {
      const c = s[i];
      if (c === "(") depth++;
      else if (c === ")") depth = Math.max(depth - 1, 0);
      const fracBar = c === "/" && i > 0 && i < s.length - 1 &&
        (dg(s[i - 1]) || s[i - 1] === ")" || "xyz".includes(s[i - 1])) &&
        dg(s[i + 1]) && fractionEndOk(s, i + 1);
      if (depth === 0 && seps.includes(c) && !fracBar && !(c === "-" && startsAmount(s, i + 1))) {
        parts.push(cur); cur = "";
      } else cur += c;
    }
    parts.push(cur);
    return parts.map(p => p.trim()).filter(p => p);
  }
  function splitAfterOxygen(s) {
    const out = []; let start = 0, depth = 0, i = 0;
    while (i < s.length) {
      const c = s[i];
      depth += (c === "(") - (c === ")");
      if (depth === 0 && c === "O" && !(i + 1 < s.length && lo(s[i + 1]))) {
        const j = readAmount(s, i + 1)[2];
        const sym = readSymbol(s, j)[0];
        if (sym !== null && !has(ANIONS, sym)) { out.push(s.slice(start, j)); start = j; }
        i = Math.max(j, i + 1);
        continue;
      }
      i++;
    }
    out.push(s.slice(start));
    return out.filter(p => p.trim());
  }
  function stripDots(t) { return t.replace(/^\.+|\.+$/g, ""); }
  /* mirror of formula._a_site_type / formula._bare_abo3 */
  const aSiteType = e => (ELEM[e].Z >= 57 && ELEM[e].Z <= 71) || e === "Y" || e === "Bi" ||
    e === "Pb" || ELEM[e].group === 1 || ELEM[e].group === 2;
  function bareAbo3(items) {
    if (items.some(it => it.g)) return null;
    const cats = items.filter(it => !has(ANIONS, it.sym));
    const oxy = items.filter(it => it.sym === "O");
    if (cats.length < 4 || oxy.length !== 1 || oxy[0].amt === null ||
        Math.abs(oxy[0].amt - 3) > 0.5) return null;
    if (cats.some(it => it.amt !== null || !has(ELEM, it.sym))) return null;
    const syms = cats.map(it => it.sym);
    let nA = 0;
    while (nA < syms.length && aSiteType(syms[nA])) nA++;
    if (nA === 0 || syms.length - nA < 2 || syms.slice(nA).some(aSiteType)) return null;
    const out = {};
    syms.forEach((e, k) => {
      out[e] = (has(out, e) ? out[e] : 0.0) + 1.0 / (k < nA ? nA : syms.length - nA);
    });
    return out;
  }
  function parseChunk(chunk) {
    chunk = stripDots(chunk.trim());
    const m = /^[0-9]+(?:\.[0-9]+)?(?=[A-Z(])/.exec(chunk);
    if (m) chunk = chunk.slice(m[0].length);
    if (!chunk) return null;
    const [items, j, ok, flags] = parseItems(chunk, 0, 0);
    if (!ok || j !== chunk.length || !items.length) return null;
    const abo3 = bareAbo3(items);
    if (abo3 !== null) {
      const an = {};
      for (const it of items) if (has(ANIONS, it.sym)) an[it.sym] = it.amt;
      flags.add("abo3_bare_sequence");
      return [abo3, an, flags];
    }
    const [cat0, an, f2] = evaluateBest(items);
    const cat = {};
    for (const e of Object.keys(cat0)) if (cat0[e] > 1e-12) cat[e] = cat0[e];
    f2.forEach(x => flags.add(x));
    return Object.keys(cat).length ? [cat, an, flags] : null;
  }
  function oxideSum(s) {
    const pieces = s.split(new RegExp("\\s*[+" + DASHES + "]\\s*|\\s*-\\s*(?=[0-9]|[A-Z])"));
    if (pieces.length < 3) return null;
    const out = {};
    for (const p of pieces) {
      const m = BINOX.exec(p);
      if (!m || has(ANIONS, m[2]) || !SYMBOLS.has(m[2])) return null;
      const coef = m[1] ? parseFloat(m[1]) : 1.0;
      out[m[2]] = (has(out, m[2]) ? out[m[2]] : 0.0) + coef * (m[3] ? parseFloat(m[3]) : 1.0);
    }
    return out;
  }
  const single = p => Object.keys(p[0]).length === 1 && !Object.keys(p[1]).length &&
    Object.values(p[0])[0] === 1.0;
  function combine(pieces) {
    if (pieces.length === 1) return [pieces[0]];
    const singles = pieces.filter(single);
    if (singles.length === pieces.length) {
      const cat = {};
      for (const p of pieces) for (const e of Object.keys(p[0])) cat[e] = 1.0;
      return [[cat, {}, new Set(["element_list"])]];
    }
    if (singles.length === pieces.length - 1) {
      const main = pieces.filter(p => !single(p))[0];
      return [[main[0], main[1], new Set([...main[2], "minor_component_dropped"])]];
    }
    return pieces;
  }
  const DOPANT_PREFIX = new RegExp("(?<![A-Za-z])[A-Z][a-z]?-(?:doped|modified|decorated|" +
    "substituted|loaded|supported|coated|containing)(?![A-Za-z])\\s*", "g");
  function parseString(s) {
    const flags = new Set();
    if (s.includes(":")) { s = s.split(":")[0]; flags.add("dopant_dropped"); }
    const t = s.replace(DOPANT_PREFIX, "");      /* "Zn-doped Co-Fe-Cr-Mn-Ni spinel" */
    if (t !== s) { s = t.trim(); flags.add("dopant_dropped"); }
    const ox = oxideSum(s);
    if (ox && Object.keys(ox).length) return [ox, {}, new Set([...flags, "oxide_sum"])];
    const alts = [];
    for (const alt of splitTop(s, ALT_SEPS)) {
      const pieces = [];
      for (const ch of splitTop(alt, LIST_SEPS))
        for (const piece of (ch.includes("O") ? splitAfterOxygen(ch) : [ch])) {
          const r = parseChunk(piece);
          if (r !== null) pieces.push(r);
        }
      if (pieces.length) alts.push(...combine(pieces));
    }
    if (!alts.length) return [{}, {}, new Set([...flags, "unparsed"])];
    if (alts.length > 1) {
      if (alts.every(single) && alts.length >= 3) {
        const cat = {};
        for (const a of alts) for (const e of Object.keys(a[0])) cat[e] = 1.0;
        return [cat, {}, new Set([...flags, "element_list"])];
      }
      flags.add("composite_main_phase");
    }
    let best = alts[0];
    for (const a of alts) if (Object.keys(a[0]).length > Object.keys(best[0]).length) best = a;
    return [best[0], best[1], new Set([...flags, ...best[2]])];
  }
  function parse(text) {
    if (text === null || text === undefined) return { amounts: {}, flags: ["empty"] };
    let best = null;
    for (const [s, f0] of placeholderVariants(normalize(text))) {
      const [cat, , f] = parseString(s);
      const n = Object.values(cat).filter(v => v > 0).length;
      if (best === null || (best[0] < 5 && n > best[0]))
        best = [n, { amounts: cat, flags: [...new Set([...f, ...f0])].sort() }];
    }
    return best[1];
  }
  function fractions(amounts) {
    const keys = Object.keys(amounts).sort();
    const tot = nsumv(Object.values(amounts));
    const out = {};
    if (tot > 0) for (const e of keys) out[e] = amounts[e] / tot;
    return out;
  }
  function cationFractions(text) {
    const fr = fractions(parse(text).amounts);
    const keep = {};
    for (const e of Object.keys(fr)) if (fr[e] >= MIN_FRAC) keep[e] = fr[e];
    const tot = nsumv(Object.values(keep));
    const out = {};
    if (tot) for (const e of Object.keys(keep).sort()) out[e] = keep[e] / tot;
    return out;
  }
  return { parse, cationFractions, normalize };
})();

/* ---------- full prediction ---------- */
function predict(frac) {
  const x = featurize(frac);
  const raw = rawPredict(MODEL, x);
  const probs = softmax(raw, MODEL.temperature);
  const order = MODEL.classes.map((c, i) => ({ c, p: probs[i] })).sort((a, b) => b.p - a.p);
  const d = adDistance(x);
  const top = order[0].c;
  const a = predictA(top, frac);
  return {
    x, probs, order, top,
    adDistance: d, inDomain: d <= AD.threshold,
    a, aTol: a ? a * LATTICE.mae_pct / 100 : null,
    pattern: a ? simulatePattern(top, frac, a, 80) : [],
    desc: Object.fromEntries(REPORT_FEATS.map(n => [n, x[FEATNAMES.indexOf(n)]])),
  };
}
