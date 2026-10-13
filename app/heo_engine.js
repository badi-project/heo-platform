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
