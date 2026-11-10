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

/* ---------- ionic radii: mirror of src/radii.py ---------- */
/* Shannon (1976) effective radii (LATTICE.shannon), high spin except Co3+ and Ni3+; an
   untabulated coordination is interpolated between its neighbours, or extrapolated with
   the mean ratio of the ions of the same charge tabulated at both coordinations. */
const CN_ROMAN = { II: 2, III: 3, IV: 4, V: 5, VI: 6, VII: 7, VIII: 8, IX: 9, X: 10, XI: 11, XII: 12 };
const ROMAN_CN = {};
for (const k in CN_ROMAN) ROMAN_CN[CN_ROMAN[k]] = k;
const LOW_SPIN = new Set(["Co3", "Ni3"]);
const own = (o, k) => Object.prototype.hasOwnProperty.call(o, k);

function pickRadius(entry, el, ox) {
  if (LOW_SPIN.has(el + ox) && own(entry, "ls")) return entry.ls;
  return own(entry, "hs") ? entry.hs : entry.ls;
}

function sameChargeRatio(ox, cnFrom, cnTo) {
  const tab = LATTICE.shannon, a = ROMAN_CN[cnFrom], b = ROMAN_CN[cnTo];
  const ratios = [];
  for (const el of Object.keys(tab).sort()) {
    const byCn = tab[el][String(ox)];
    if (byCn && own(byCn, a) && own(byCn, b))
      ratios.push(pickRadius(byCn[b], el, ox) / pickRadius(byCn[a], el, ox));
  }
  if (!ratios.length) return null;
  let s = 0.0;
  for (const v of ratios) s += v;
  return s / ratios.length;
}

/* [radius in angstrom, how] for el^ox+ in coordination cn; [null, "missing"] when
   Shannon gives no radius for that oxidation state */
function ionicRadius(el, ox, cn) {
  const byCn = (LATTICE.shannon[el] || {})[String(ox)];
  if (!byCn) return [null, "missing"];
  const pts = Object.keys(byCn).filter(k => own(CN_ROMAN, k))
    .map(k => [CN_ROMAN[k], pickRadius(byCn[k], el, ox)])
    .sort((p, q) => p[0] - q[0] || p[1] - q[1]);
  if (!pts.length) return [null, "missing"];
  for (const [c, r] of pts) if (c === cn) return [r, "exact"];
  if (pts.length === 1) return [pts[0][1], "single"];
  const lower = pts.filter(p => p[0] < cn), upper = pts.filter(p => p[0] > cn);
  if (lower.length && upper.length) {
    const [c1, r1] = lower[lower.length - 1], [c2, r2] = upper[0];
    return [r1 + (r2 - r1) * (cn - c1) / (c2 - c1), "interpolated"];
  }
  const est = [];
  for (const [c, r] of pts) {
    const ratio = sameChargeRatio(ox, c, cn);
    if (ratio !== null) est.push(r * ratio);
  }
  if (!est.length) {
    const [[c1, r1], [c2, r2]] = upper.length ? [upper[0], upper[1]]
      : [lower[lower.length - 2], lower[lower.length - 1]];
    return [r1 + (r2 - r1) * (cn - c1) / (c2 - c1), "extrapolated"];
  }
  let s = 0.0;
  for (const v of est) s += v;
  return [s / est.length, "extrapolated"];
}

function oxygenRadius(cn) { return LATTICE.shannon.O["-2"][ROMAN_CN[cn]].hs; }

/* ---------- lattice parameter: mirror of src/lattice.py ---------- */
/* Shannon radii, oxidation states by charge balance in a fixed order, cation sites and
   one published relation per prototype (rules in LATTICE.rules, from
   data/reference/lattice_rules.json; protocol in data/reference/lattice_protocol.md).
   Ions are [site, element, state, amount]; every sum is a plain loop in the order of
   the Python module, so both give the same value. */
const LAT_EPS = 1e-12;
const LAT_HOW = { interpolated: "interpolé", extrapolated: "extrapolé",
  single: "pris dans la seule coordinence tabulée" };

class LatticeError extends Error {}

/* x with n decimals as Python's format() writes it: toFixed() rounds an exact tie (0.3125
   to three decimals) away from zero, Python to the even digit. toFixed(100) is the exact
   decimal expansion of a double of this magnitude, so a tie is recognised exactly. */
function pyFixed(x, n) {
  const ax = Math.abs(x), s = ax.toFixed(100), i = s.indexOf(".");
  let out = ax.toFixed(n);
  if (/^50*$/.test(s.slice(i + 1 + n))) {
    const t = n ? s.slice(0, i + 1 + n) : s.slice(0, i);
    if (Number(t[t.length - 1]) % 2 === 0) out = t;
  }
  return (x < 0 ? "-" : "") + out;
}

function latSum(values) { let s = 0.0; for (const v of values) s += v; return s; }
function latCharge(ions) { return latSum(ions.map(i => i[2] * i[3])); }
function latOx(table, e) { return own(table, e) ? table[e] : ELEM[e].ox; }

function latShift(ions, order, need, step) {
  for (const el of order) {
    const n = ions.length;
    for (let k = 0; k < n; k++) {
      const ion = ions[k];
      if (ion[1] !== el || need <= LAT_EPS) continue;
      const target = ion[2] + step;
      if (ionicRadius(el, target, 6)[0] === null) continue;
      const moved = Math.min(ion[3], need);
      need -= moved;
      if (moved >= ion[3] - LAT_EPS) ion[2] = target;
      else { ion[3] -= moved; ions.push([ion[0], el, target, moved]); }
    }
  }
  return need;
}

function latTwoSites(frac, aSet, notes) {
  const els = Object.keys(frac).sort();
  const A = els.filter(e => aSet.has(e)), B = els.filter(e => !aSet.has(e));
  if (A.length && B.length) {
    const ta = latSum(A.map(e => frac[e])), tb = latSum(B.map(e => frac[e]));
    if (Math.abs(ta - tb) > 0.02) notes.push(`rapport A/B = ${pyFixed((ta / tb), 3)}, différent de 1`);
    const An = {}, Bn = {};
    for (const e of A) An[e] = frac[e] / ta;
    for (const e of B) Bn[e] = frac[e] / tb;
    return [An, Bn];
  }
  notes.push("aucun cation de la classe attendue sur l'un des sites : sites attribués " +
             "par rayon décroissant");
  return abSplit(frac);
}

function latSpinelSites(ions, score) {
  const sc = i => own(score, i[1] + i[2]) ? score[i[1] + i[2]] : 0;
  const order = ions.slice().sort((p, q) => sc(q) - sc(p) || (p[1] < q[1] ? -1 : p[1] > q[1] ? 1 : 0)
    || p[2] - q[2]);
  let room = 1 / 3;
  const out = [];
  for (const [, el, ox, x] of order) {
    const t = room > LAT_EPS ? Math.min(x, room) : 0.0;
    room -= t;
    if (t > LAT_EPS) out.push(["T", el, ox, t * 3]);
    if (x - t > LAT_EPS) out.push(["O", el, ox, (x - t) * 3 / 2]);
  }
  return out;
}

function latAssign(proto, frac) {
  const R = LATTICE.rules, D = R.default_ox;
  const els = Object.keys(frac).sort();
  const notes = [];
  let ions;
  if (proto === "rocksalt") {
    /* 2+ unless listed; an element without a tabulated 2+ ion (Y, Sc, Zr...) takes the
       platform default (amendment A2, no validation composition concerned) */
    const d = D.rocksalt, ox0 = e => own(d, e) ? d[e]
      : own(LATTICE.shannon[e] || {}, String(d._)) ? d._ : ELEM[e].ox;
    ions = els.map(e => ["M", e, ox0(e), frac[e]]);
    const need = 2.0 - latCharge(ions);
    if (need > LAT_EPS) {
      const left = latShift(ions, R.oxidise.rocksalt, need, +1);
      if (left > 1e-9) notes.push(`charge inférieure de ${pyFixed(left, 3)} à +2 par cation : lacunes`);
    } else if (need < -1e-9) notes.push(`charge supérieure de ${pyFixed((-need), 3)} à +2 par cation : lacunes`);
    const big = els.filter(e => { const r = ionicRadius(e, ox0(e), 6)[0];
      return r !== null && r > R.rocksalt.large_cation_A; });
    if (big.length)
      notes.push("cation volumineux (" + big.join(", ") + ") : sur les rocksalt mesurés, " +
                 "une maille inchangée suggère une incorporation partielle ; la maille " +
                 "prédite peut être trop grande");
  } else if (proto === "spinel") {
    const d = D.spinel;
    ions = els.map(e => ["?", e, latOx(d, e), frac[e]]);
    const need = 8 / 3 - latCharge(ions);
    if (need > LAT_EPS) {
      const left = latShift(ions, R.oxidise.spinel, need, +1);
      if (left > 1e-9) notes.push(`charge inférieure de ${pyFixed((3 * left), 3)} à +8 par AB2O4`);
    } else if (need < -1e-9) notes.push(`charge supérieure de ${pyFixed((-3 * need), 3)} à +8 par AB2O4`);
    ions = latSpinelSites(ions, R.tet_score);
  } else if (proto === "fluorite") {
    const d = D.fluorite, mixed = R.fluorite_mixed;
    ions = [];
    for (const e of els) {
      if (own(mixed, e)) for (const [ox, share] of mixed[e]) ions.push(["M", e, ox, frac[e] * share]);
      else ions.push(["M", e, latOx(d, e), frac[e]]);
    }
  } else if (proto === "perovskite" || proto === "pyrochlore") {
    const [A, B] = latTwoSites(frac, new Set(R[proto + "_A"]), notes);
    const dA = D[proto + "_A"], dB = D[proto + "_B"];
    ions = Object.keys(A).sort().map(e => ["A", e, latOx(dA, e), A[e]])
      .concat(Object.keys(B).sort().map(e => ["B", e, latOx(dB, e), B[e]]));
    if (proto === "perovskite") {
      const need = 6.0 - latCharge(ions);
      if (need > LAT_EPS) {
        const left = latShift(ions, R.oxidise.perovskite, need, +1);
        if (left > 1e-9) notes.push(`charge inférieure de ${pyFixed(left, 3)} à +6 : lacunes d'oxygène`);
      } else if (need < -LAT_EPS) {
        const left = latShift(ions, R.reduce_perovskite, -need, -1);
        if (left > 1e-9) notes.push(`charge supérieure de ${pyFixed(left, 3)} à +6`);
      }
    }
  } else throw new LatticeError(`prototype sans modèle de maille : ${proto}`);
  const cmp = (p, q) => (p < q ? -1 : p > q ? 1 : 0);
  ions.sort((p, q) => cmp(p[0], q[0]) || cmp(p[1], q[1]) || p[2] - q[2]);
  return [ions, notes];
}

function latMeanRadius(ions, site, cn, used) {
  let num = 0.0, den = 0.0;
  for (const [s, el, ox, x] of ions) {
    if (s !== site) continue;
    const [r, how] = ionicRadius(el, ox, cn);
    if (r === null) throw new LatticeError(`pas de rayon de Shannon pour ${el}${ox}+`);
    used.push({ site, ion: `${el}${ox}+`, cn, r, how, amount: x });
    num += r * x;
    den += x;
  }
  return num / den;
}

function vacancyRadius(ions) {
  const hosts = LATTICE.rules.fluorite.vacancy_hosts, names = Object.keys(hosts).sort();
  const d = (4.0 - latCharge(ions)) / 2;
  if (d <= 1e-9) return [0.0, null];
  const rh = {};
  for (const h of names) rh[h] = ionicRadius(h, 4, 8)[0];
  let num = 0.0, den = 0.0;
  for (const [, el, ox, x] of ions) if (ox < 4) {
    const w = (4 - ox) * x;
    num += ionicRadius(el, ox, 8)[0] * w;
    den += w;
  }
  const rs = num / den;
  let rv = 0.0, tot = 0.0;
  for (const [, el, ox, x] of ions) {
    if (ox !== 4) continue;
    let h = el;
    if (!own(hosts, el)) {
      const r = ionicRadius(el, 4, 8)[0];
      h = names[0];
      for (const k of names) if (Math.abs(rh[k] - r) < Math.abs(rh[h] - r)) h = k;
    }
    const [c0, c1] = hosts[h];
    rv += (c0 - c1 * (rs - rh[h])) * x;
    tot += x;
  }
  if (tot <= LAT_EPS) throw new LatticeError("fluorine lacunaire sans cation tétravalent hôte");
  return [d, rv / tot];
}

/* {a, ions, radii, notes, estimated, vacancies}; throws LatticeError when there is no
   model for the prototype or no radius for an ion */
function latticeModel(proto, frac) {
  const R = LATTICE.rules;
  const [ions, notes] = latAssign(proto, frac);
  const used = [];
  let a, vac = null;
  if (proto === "rocksalt") {
    a = 2 * (latMeanRadius(ions, "M", 6, used) + oxygenRadius(6));
  } else if (proto === "fluorite") {
    const o4 = oxygenRadius(4);
    const rc = latMeanRadius(ions, "M", 8, used);
    const [d, rv] = vacancyRadius(ions);
    const rAn = rv === null ? o4 : ((2 - d) * o4 + d * rv) / 2;
    if (rv !== null) {
      vac = { per_cation: d, r_V: rv };
      notes.push(`lacunes d'oxygène : ${pyFixed(d, 3)} par cation, rayon effectif ${pyFixed(rv, 3)} Å ` +
                 "(sphères dures)");
    }
    a = 4 / Math.sqrt(3) * (rc + rAn);
  } else if (proto === "perovskite") {
    const c = R.perovskite.c, o6 = oxygenRadius(6);
    const rA = latMeanRadius(ions, "A", 6, used);
    const rB = latMeanRadius(ions, "B", 6, used);
    a = c[0] + c[1] * (rA + o6) + c[2] * (rB + o6);
    const q = latCharge(ions);
    if (q < 6 - 1e-9) vac = { per_cation: (6 - q) / 4, r_V: null };
  } else if (proto === "spinel") {
    const o4 = oxygenRadius(4);
    const RT = latMeanRadius(ions, "T", 4, used) + o4;
    const RO = latMeanRadius(ions, "O", 6, used) + o4;
    a = 8 / (11 * Math.sqrt(3)) * (5 * RT + Math.sqrt(33 * RO * RO - 8 * RT * RT));
  } else {
    const c = R.pyrochlore.c, o4 = oxygenRadius(4);
    const RA = latMeanRadius(ions, "A", 8, used) + o4;
    const RB = latMeanRadius(ions, "B", 6, used) + o4;
    a = 8 / Math.sqrt(3) * (c[0] * RA - c[1] * RA * RA / RB);
  }
  const estimated = used.filter(u => u.how !== "exact");
  for (const u of estimated)
    notes.push(`rayon ${u.ion} en coordinence ${u.cn} ${LAT_HOW[u.how]} : ${pyFixed(u.r, 3)} Å`);
  return { a, ions, radii: used, notes, estimated, vacancies: vac };
}

/* geometric lattice model, or null (mirror of Predictor.lattice_detail) */
function latticeDetail(proto, frac) {
  if (!LATTICE.rules.families.includes(proto)) return null;
  try { return latticeModel(proto, frac); }
  catch (err) { if (err instanceof LatticeError) return null; throw err; }
}

function predictA(proto, frac) {
  const r = latticeDetail(proto, frac);
  return r ? r.a : null;
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

function simulatePattern(proto, frac, a, ttMax, lambda) {
  const P = XRD.prototypes[proto];
  if (!P) return [];
  const occ = siteOccupancies(proto, frac);
  const lam = lambda || XRD.wavelength, sMax = Math.sin((ttMax / 2) * Math.PI / 180);
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
    if (!acc["hkl_" + tt]) { acc["hkl_" + tt] = e.hkl.join(""); acc["q2_" + tt] = e.q2; }
  }
  const peaks = Object.keys(acc).filter(k => !k.includes("_"))
    .map(k => ({ tt: parseFloat(k), I: acc[k], hkl: acc["hkl_" + k], q2: acc["q2_" + k] }));
  const mx = Math.max(...peaks.map(p => p.I), 1e-30);
  return peaks.map(p => ({ ...p, I: 100 * p.I / mx })).filter(p => p.I > 0.4)
    .sort((x, y) => x.tt - y.tt);
}

/* ---------- theoretical (X-ray) density ---------- */
/* Mass of the prototype cell over its volume, in g/cm3: every cation site carries the
   composition (split between A and B for ordered prototypes), every oxygen site is full.
   Exact once a is known. With a predicted a, the uncertainty is three times that of a, or
   the mean error observed for that lattice model on measured high-entropy oxides when it
   is larger (LATTICE.rho, written by tools/validate_density.py).
   Oxygen vacancies are not modelled: a fluorite rich in trivalent cations is lighter. */
const M_OXYGEN = 15.999, AVOGADRO_A3 = 0.602214076;   /* N_A x 1e-24 cm3 per cubic angstrom */
function xrayDensity(proto, frac, a) {
  const P = XRD.prototypes[proto];
  if (!P || !a) return null;
  const occ = siteOccupancies(proto, frac);
  let mass = 0;
  for (const t in P.nsites) {
    let m = 0;
    for (const el in occ[t]) m += occ[t][el] * (el === "O" ? M_OXYGEN : ELEM[el].mass);
    mass += P.nsites[t] * m;
  }
  return mass / (AVOGADRO_A3 * a * a * a);
}
function densityUncertaintyPct(aSrc, aPct) {
  const seen = (LATTICE.rho || {})[aSrc];
  return Math.max(3 * aPct, seen ? seen.mae_pct : 0);
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

/* ---------- computed thermodynamics (thermo/, CHGNet surrogates) ---------- */
/* These two functions used to live in heo_ui.js. They are part of the prediction, not
   of its display: src/predict.py mirrors them and tools/parity_check.py compares the
   two, so the page, the batch table and the command line give the same numbers. */
const hasThermo = () => typeof THERMO !== "undefined" && !!THERMO;

/* spinel made of cations covered by thermo/: additive per-cation model fitted on CHGNet
   cells and calibrated on binary spinels of known lattice parameter; null otherwise */
function thermoLattice(proto, frac) {
  if (proto !== "spinel" || !hasThermo() || !THERMO.lat) return null;
  const L = THERMO.lat, cov = L.cov || L.elements, els = Object.keys(frac).sort();
  if (!els.every(e => cov.includes(e))) return null;
  return L.k * wsum(els.map(e => L.a[L.elements.indexOf(e)]), els.map(e => frac[e]));
}

/* oxidation enthalpy rocksalt -> spinel (3 MO + O -> M3O4) and the rocksalt / spinel
   tendency in air it implies. The value computed directly with CHGNet is used when it
   was computed for this very composition (fractions within 0.02), the additive
   per-cation summary otherwise. */
function thermoOx(frac) {
  if (!hasThermo()) return null;
  const ox = THERMO.ox, cov = ox.cov || ox.elements, els = Object.keys(frac).sort();
  const missing = els.filter(e => !cov.includes(e));
  if (missing.length) return { covered: false, missing };
  const dhAdd = wsum(els.map(e => ox.h[ox.elements.indexOf(e)]), els.map(e => frac[e]));
  const exact = THERMO.systems[els.join("-")] || null;
  const direct = !!(exact && exact.dh !== undefined && exact.frac
    && els.every(e => Math.abs((exact.frac[e] || 0) - frac[e]) < 0.02));
  const dh = direct ? exact.dh : dhAdd;
  const p = 1 / (1 + Math.exp(-(ox.b0 + ox.b1 * dh)));
  return { covered: true, missing, dh, direct, pRocksalt: p,
    verdict: p > 0.7 ? "rocksalt" : p < 0.3 ? "spinel" : "boundary", system: exact };
}

/* ---------- full prediction ---------- */
/* frac: {element: cation fraction}, summing to 1. Keys are put in alphabetical order
   first so that the result does not depend on the order the cations were clicked. */
function predict(frac0) {
  const frac = {};
  for (const e of Object.keys(frac0).sort()) frac[e] = frac0[e];
  const x = featurize(frac);
  const raw = rawPredict(MODEL, x);
  const probs = softmax(raw, MODEL.temperature);
  const order = MODEL.classes.map((c, i) => ({ c, p: probs[i] })).sort((a, b) => b.p - a.p);
  const d = adDistance(x);
  const top = order[0].c;
  /* lattice parameter: geometric model (Shannon radii and the published relation of the
     prototype), replaced for spinels whose cations it covers by the CHGNet model, more
     accurate on the measured high-entropy spinels (QC/lattice_validation.txt) */
  const geo = latticeDetail(top, frac);
  let a = geo ? geo.a : null, aPct = geo ? LATTICE.mae_pct[top] : null,
    aSrc = geo ? "geo" : null, aNotes = geo ? geo.notes : [];
  const aTh = thermoLattice(top, frac);
  if (aTh !== null) { a = aTh; aPct = THERMO.lat.mae_pct; aSrc = "chgnet"; aNotes = []; }
  const rho = xrayDensity(top, frac, a);
  return {
    frac, x, probs, order, top,
    adDistance: d, inDomain: d <= AD.threshold,
    a, aPct, aSrc, aNotes, aTol: a ? a * aPct / 100 : null,
    rho, rhoPct: rho ? densityUncertaintyPct(aSrc, aPct) : null,
    rhoTol: rho ? rho * densityUncertaintyPct(aSrc, aPct) / 100 : null,
    pattern: a ? simulatePattern(top, frac, a, 80) : [],
    thermo: thermoOx(frac),
    desc: Object.fromEntries(REPORT_FEATS.map(n => [n, x[FEATNAMES.indexOf(n)]])),
  };
}

/* ---------- reading an input: mirror of Predictor.read() in src/predict.py ---------- */
const MIN_CATIONS = 4, TRAINED_MIN_CATIONS = 5;
/* formula text -> {text, frac, flags}, or {error} with the reason no prediction is
   possible. The fractions are the exact ones of the formula, renormalised with the same
   compensated sum as Python: rounding them, as the page did until now, moved a value
   across a tree threshold for a few dozen database formulas. */
function readComposition(text) {
  const t = String(text === null || text === undefined ? "" : text).trim();
  const raw = FX.cationFractions(t);
  const els = Object.keys(raw).sort();
  const unknown = els.filter(e => !Object.prototype.hasOwnProperty.call(ELEM, e));
  if (unknown.length) return { error: "cation sans propriétés tabulées : " + unknown.join(", ") };
  if (els.length < 2)
    return { error: "formule non lue : écrivez-la comme dans un article, par exemple La(CrMnFeCoNi)O3" };
  if (els.length < MIN_CATIONS)
    return { error: `${els.length} cations lus (${els.join(", ")}) : il en faut au moins ${MIN_CATIONS}` };
  return { text: t, frac: normalizeFractions(raw), flags: FX.parse(t).flags };
}
function normalizeFractions(amounts) {
  const els = Object.keys(amounts).filter(e => amounts[e] > 0).sort();
  const tot = nsum(els.map(e => amounts[e]));
  const frac = {};
  for (const e of els) frac[e] = amounts[e] / tot;
  return frac;
}
function predictionWarnings(r) {
  const w = (r.aNotes || []).filter(t => t.startsWith("cation volumineux")),
    n = Object.keys(r.frac).length;
  if (!r.inDomain)
    w.push("hors du domaine d'applicabilité : le modèle extrapole, la probabilité n'a pas de valeur");
  if (n < TRAINED_MIN_CATIONS)
    w.push(`${n} cations : le modèle est entraîné sur ${TRAINED_MIN_CATIONS} cations ou plus`);
  if (r.order[0].p - r.order[1].p < 0.25)
    w.push(`prédiction peu tranchée entre ${r.top} et ${r.order[1].c}`);
  return w;
}

/* ---------- flat record of a prediction: mirror of Predictor.record() ---------- */
/* One line of the batch table and of its CSV export. Same columns and same formatting as
   tools/heo_predict.py --csv; tools/parity_check.py compares the two character for
   character on every formula of the database. */
const ROW_COLUMNS = ["cations", "frac", "proto", "p", "proto_2", "p_2", "in_domain", "ad", "n_base",
  "a_predit", "a_tol_pct", "a_source", "rho_xray_g_cm3", "rho_tol_pct", "dH_ox_eV_per_O",
  "p_rocksalt_thermo", "flags", "warnings", "error"];
function recordOf(r, flags) {
  const row = {};
  for (const k of ROW_COLUMNS) row[k] = "";
  const els = Object.keys(r.frac), th = r.thermo, covered = !!(th && th.covered);
  row.cations = els.join("-");
  row.frac = "{" + els.map(e => `"${e}": ${Number(r.frac[e].toFixed(6))}`).join(", ") + "}";
  row.proto = r.top; row.p = r.order[0].p.toFixed(4);
  row.proto_2 = r.order[1].c; row.p_2 = r.order[1].p.toFixed(4);
  row.in_domain = r.inDomain ? "1" : "0"; row.ad = r.adDistance.toFixed(3);
  row.n_base = String((typeof DBSET !== "undefined" && DBSET[row.cations]) || 0);
  if (r.a) {
    row.a_predit = r.a.toFixed(4); row.a_tol_pct = r.aPct.toFixed(2);
    row.a_source = r.aSrc === "geo" ? "geometric" : r.aSrc;
    row.rho_xray_g_cm3 = r.rho.toFixed(3); row.rho_tol_pct = r.rhoPct.toFixed(1);
  }
  if (covered) { row.dH_ox_eV_per_O = th.dh.toFixed(3); row.p_rocksalt_thermo = th.pRocksalt.toFixed(3); }
  row.flags = (flags || []).join(";");
  row.warnings = predictionWarnings(r).join(" ; ");
  return row;
}
function predictRow(text) {
  const c = readComposition(text);
  if (c.error) {
    const row = {};
    for (const k of ROW_COLUMNS) row[k] = "";
    row.error = c.error;
    return row;
  }
  return recordOf(predict(c.frac), c.flags);
}

/* ---------- continuous profile of the simulated pattern ---------- */
/* Pseudo-Voigt peaks of constant width on a regular 2-theta grid, scaled to 100: what a
   diffractometer file looks like, for export (.xy) and for the comparison with a measured
   pattern. The width is instrumental, not predicted — it only makes the sticks comparable. */
const PROFILE = { fwhm: 0.15, step: 0.02, ttMin: 10, ttMax: 80, eta: 0.5 };
function patternProfile(peaks, opt) {
  const o = { ...PROFILE, ...(opt || {}) };
  const n = Math.round((o.ttMax - o.ttMin) / o.step), tt = [], I = [];
  let mx = 0;
  for (let i = 0; i <= n; i++) {
    const x = o.ttMin + i * o.step;
    let y = 0;
    for (const p of peaks) {
      const u = (x - p.tt) / o.fwhm, u2 = u * u;
      y += p.I * (o.eta / (1 + 4 * u2) + (1 - o.eta) * Math.exp(-4 * Math.LN2 * u2));
    }
    tt.push(x); I.push(y);
    if (y > mx) mx = y;
  }
  return { tt, I: I.map(y => (mx > 0 ? 100 * y / mx : 0)) };
}

/* ---------- measured powder pattern: peaks, indexing, lattice refinement ---------- */
/* Mirror of src/xrdfit.py (tools/parity_check.py compares the two on synthetic patterns).
   The prediction gives a prototype and a starting lattice parameter; a measured pattern
   gives peak positions. Indexing the peaks on the cubic prototype and refining a by least
   squares answers the experimental questions: is the predicted family the one observed,
   what is the real lattice parameter, and which peaks does the family not explain. */
const RADIATION = {            /* K-alpha1, K-alpha2, K-beta wavelengths in angstrom */
  Cu: [1.54056, 1.54439, 1.39222], Co: [1.78897, 1.79285, 1.62079],
  Mo: [0.70930, 0.71359, 0.63229], Cr: [2.28970, 2.29361, 2.08487],
  Fe: [1.93604, 1.93998, 1.75661],
};
const XFIT = { smooth: 0.04, background: 1.2, isolate: 0.08, nSigma: 6, minRel: 0.02,
  window: 0.06, gridStep: 0.0005, tolSearch: 0.25, tolAssign: 0.20, tolFinal: 0.10,
  minPred: 2, strong: 10, tolLine: 0.12, edge: 0.3, asymmetry: 0.12,
  valley: 0.5, prominence: 0.3,
  /* accuracy of the refined lattice parameter established on synthetic patterns
     (QC/xrd_fit_validation.txt), by signal-to-noise ratio of the strongest peak: the
     quoted uncertainty is never below it, whatever the statistical error of the fit */
  snrClean: 500, accClean: 5e-4, accNoisy: 2.5e-3, snrMin: 70 };
const DEG = 180 / Math.PI;
const median = arr => { const v = Array.from(arr).sort((a, b) => a - b), m = v.length >> 1;
  return v.length ? (v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2) : 0; };

/* value of a regularly sampled curve at 2-theta x, by linear interpolation */
function sampleAt(tt, y, x) {
  const n = tt.length, pos = (x - tt[0]) / ((tt[n - 1] - tt[0]) / (n - 1));
  if (pos <= 0) return y[0];
  if (pos >= n - 1) return y[n - 1];
  const i = Math.floor(pos);
  return y[i] + (pos - i) * (y[i + 1] - y[i]);
}
/* 2-theta where wavelength lam2 puts the line that lam1 puts at x */
const shiftLine = (x, lam1, lam2) => { const st = Math.sin(x / (2 * DEG)) * lam2 / lam1;
  return st < 1 ? 2 * Math.asin(st) * DEG : NaN; };

/* Is the K-alpha2 line in the data? A K-alpha1 peak alone is symmetric; with K-alpha2 it
   carries half its height again on the high-angle side. Median, over the strongest peaks
   above 40 degrees, of (signal at the K-alpha2 position - signal at the mirror position)
   over the peak height: about 0 without K-alpha2, 0.2 to 0.5 with it. */
function kalpha2Asymmetry(tt, net, peaks, rad) {
  const hmax = Math.max(...peaks.map(p => p.h), 0);
  const top = peaks.filter(p => p.tt >= 40 && p.h >= 0.1 * hmax)
    .sort((a, b) => b.h - a.h || a.tt - b.tt).slice(0, 8);
  const r = [];
  for (const p of top) {
    const d = shiftLine(p.tt, rad[0], rad[1]) - p.tt;
    if (Number.isNaN(d) || p.tt + d > tt[tt.length - 1]) continue;
    r.push((sampleAt(tt, net, p.tt + d) - sampleAt(tt, net, p.tt - d)) / p.h);
  }
  return r.length >= 2 ? median(r) : 0;
}

/* Rachinger correction: remove from each point half of the K-alpha1 signal found where
   K-alpha1 puts the line that K-alpha2 puts here. What remains is the K-alpha1 pattern,
   whose peak tops are no longer pulled towards high angles. */
function stripKalpha2(tt, y, rad) {
  const n = tt.length, step = (tt[n - 1] - tt[0]) / (n - 1), out = new Array(n);
  for (let i = 0; i < n; i++) {
    const pos = (shiftLine(tt[i], rad[1], rad[0]) - tt[0]) / step;
    let c = 0;
    if (pos > 0 && i > 0) {
      const k = Math.min(Math.floor(pos), i - 1), f = Math.min(pos - k, 1);
      c = k + 1 <= i - 1 ? out[k] + f * (out[k + 1] - out[k]) : out[k];
    }
    out[i] = y[i] - 0.5 * c;
  }
  return out;
}

/* local maxima of the net signal above the noise; position from the parabola through
   the three points at the top. level is the smoothed raw signal, background included. */
function pickPeaks(tt, net, noise, level) {
  const n = tt.length, step = (tt[n - 1] - tt[0]) / (n - 1);
  let mx = 0;
  for (let i = 0; i < n; i++) if (net[i] > mx) mx = net[i];
  /* the net signal is mostly background: its median and spread measure what the noise
     leaves after smoothing and background removal, better than the raw scatter does */
  const base = median(net), dev = new Array(n);
  for (let i = 0; i < n; i++) dev[i] = Math.abs(net[i] - base);
  const spread = 1.4826 * median(dev);
  const wp = Math.max(1, Math.round(XFIT.isolate / step));
  const wv = Math.max(1, Math.round(XFIT.valley / step));
  const floor = Math.max(base + XFIT.nSigma * Math.max(spread, noise), XFIT.minRel * mx);
  const levelMed = Math.max(median(level), 1e-300);
  const peaks = [];
  for (let i = 1; i < n - 1; i++) {
    if (net[i] < floor) continue;
    let top = true;
    for (let j = Math.max(0, i - wp); j <= Math.min(n - 1, i + wp) && top; j++)
      if (net[j] > net[i] || (net[j] === net[i] && j < i)) top = false;
    if (!top) continue;
    /* prominence: a bump of the noise on the tail of a strong peak is a local maximum
       too, but it does not rise above the valley that separates it from that peak */
    let valley = -Infinity;
    for (const dir of [-1, 1]) {
      let low = net[i];
      for (let j = i + dir; j >= 0 && j < n && Math.abs(j - i) <= wv && net[j] <= net[i]; j += dir)
        if (net[j] < low) low = net[j];
      if (low > valley) valley = low;
    }
    /* counting noise grows as the square root of the signal: on the flank of a strong
       peak it is larger than on the background where "noise" was measured */
    const local = noise * Math.sqrt(Math.max(level[i], 0) / levelMed);
    if (net[i] - valley < Math.max(XFIT.prominence * net[i], XFIT.nSigma * local)) continue;
    const y0 = net[i - 1], y1 = net[i], y2 = net[i + 1], den = y0 - 2 * y1 + y2;
    const dx = den < 0 ? 0.5 * (y0 - y2) / den : 0;
    peaks.push({ tt: tt[i] + dx * step, h: y1 - 0.25 * (y0 - y2) * dx });
  }
  return { peaks, max: mx };
}

/* peaks of a measured pattern: moving average, SNIP background, K-alpha2 removed when
   the data carry it, then the peak search. rad = [K-alpha1, K-alpha2, K-beta]. */
function findPeaks(tt, I, rad) {
  rad = rad || RADIATION.Cu;
  const n = tt.length, step = (tt[n - 1] - tt[0]) / (n - 1);
  const ws = Math.max(1, Math.round(XFIT.smooth / step));
  const sm = new Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0, k = 0;
    for (let j = Math.max(0, i - ws); j <= Math.min(n - 1, i + ws); j++) { acc += I[j]; k++; }
    sm[i] = acc / k;
  }
  const resid = new Array(n);
  for (let i = 0; i < n; i++) resid[i] = Math.abs(I[i] - sm[i]);
  const noise = 1.4826 * median(resid) / Math.sqrt(2 * ws + 1);
  let bg = sm.slice();
  for (let p = Math.max(1, Math.round(XFIT.background / step)); p >= 1; p--) {
    const nb = bg.slice();
    for (let i = p; i < n - p; i++) nb[i] = Math.min(bg[i], (bg[i - p] + bg[i + p]) / 2);
    bg = nb;
  }
  let net = new Array(n);
  for (let i = 0; i < n; i++) net[i] = sm[i] - bg[i];
  let found = pickPeaks(tt, net, noise, sm);
  const asym = kalpha2Asymmetry(tt, net, found.peaks, rad), stripped = asym > XFIT.asymmetry;
  if (stripped) { net = stripKalpha2(tt, net, rad); found = pickPeaks(tt, net, noise, sm); }
  return { peaks: found.peaks, net, noise, max: found.max, asym, stripped };
}

const ttOf = (a, q2, lam) => { const st = lam * Math.sqrt(q2) / (2 * a);
  return st < 1 ? 2 * Math.asin(st) * DEG : NaN; };

/* nearest measured peak to each reflection within tol, each measured peak used once */
function assignPeaks(peaks, refl, a, lam, zero, tol) {
  const pairs = [];
  refl.forEach((r, ri) => {
    const tc = ttOf(a, r.q2, lam) + zero;
    if (Number.isNaN(tc)) return;
    let best = -1, bd = tol;
    for (let j = 0; j < peaks.length; j++) {
      const d = Math.abs(peaks[j].tt - tc);
      if (d <= bd) { bd = d; best = j; }
    }
    if (best >= 0) pairs.push({ ri, j: best, d: bd });
  });
  pairs.sort((x, y) => x.d - y.d || x.ri - y.ri);
  const usedPeak = new Set(), out = [];
  for (const p of pairs) if (!usedPeak.has(p.j)) { usedPeak.add(p.j); out.push(p); }
  return out.sort((x, y) => x.ri - y.ri);
}

/* least squares on 2-theta: lattice parameter and, if asked, a zero shift */
function fitLattice(peaks, refl, pairs, a, lam, withZero) {
  let zero = 0, cov = 0;
  const fitZero = withZero && pairs.length >= 4;
  for (let it = 0; it < 6; it++) {
    let saa = 0, saz = 0, szz = 0, ra = 0, rz = 0;
    for (const p of pairs) {
      const tc = ttOf(a, refl[p.ri].q2, lam);
      const d = -2 * DEG * Math.tan(tc / (2 * DEG)) / a;       /* d(2theta)/da */
      const res = peaks[p.j].tt - zero - tc;
      saa += d * d; saz += d; szz += 1; ra += d * res; rz += res;
    }
    if (fitZero) {
      const det = saa * szz - saz * saz;
      a += (ra * szz - rz * saz) / det; zero += (saa * rz - saz * ra) / det;
      cov = szz / det;
    } else { a += ra / saa; cov = 1 / saa; }
  }
  let ss = 0;
  for (const p of pairs) ss += (peaks[p.j].tt - zero - ttOf(a, refl[p.ri].q2, lam)) ** 2;
  const dof = pairs.length - (fitZero ? 2 : 1);
  return { a, zero, rms: Math.sqrt(ss / pairs.length),
    sigmaA: dof > 0 ? Math.sqrt(ss / dof * cov) : null, fitZero };
}

/* index the measured peaks on one prototype and refine its lattice parameter.
   a0 is the starting value (the prediction); the search explores a0 x (1 +/- window). */
function refineLattice(found, proto, frac, a0, opt) {
  const o = opt || {}, rad = RADIATION[o.radiation || "Cu"], lam = rad[0];
  const peaks = found.peaks;
  if (!peaks.length || !a0 || !XRD.prototypes[proto]) return null;
  const lo = o.ttMin, hi = o.ttMax;
  /* reflections that can fall in the measured range for any lattice parameter explored */
  const every = simulatePattern(proto, frac, a0 * (1 - XFIT.window), 180, lam);
  const all = every.filter(r => r.I >= XFIT.minPred);
  const inRange = (a, zero) => all.filter(r => { const t = ttOf(a, r.q2, lam) + zero;
    return t >= lo && t <= hi; });
  const hmax = Math.max(...peaks.map(p => p.h));
  let best = null;
  const nGrid = Math.round(XFIT.window / XFIT.gridStep);
  for (let g = -nGrid; g <= nGrid; g++) {
    const a = a0 * (1 + g * XFIT.gridStep);
    let score = 0;
    for (const r of inRange(a, 0)) {
      const tc = ttOf(a, r.q2, lam);
      let bd = XFIT.tolSearch, bh = 0;
      for (const p of peaks) { const d = Math.abs(p.tt - tc); if (d < bd) { bd = d; bh = p.h; } }
      if (bh > 0) score += Math.sqrt(r.I) * (1 - (bd / XFIT.tolSearch) ** 2) * Math.sqrt(bh / hmax);
    }
    if (best === null || score > best.score + 1e-12) best = { a, score };
  }
  let refl = inRange(best.a, 0);
  let pairs = assignPeaks(peaks, refl, best.a, lam, 0, XFIT.tolAssign);
  if (pairs.length < 2) return { proto, a: null, n: pairs.length, nPred: refl.length };
  let fit = fitLattice(peaks, refl, pairs, best.a, lam, o.zero !== false);
  refl = inRange(fit.a, fit.zero);
  pairs = assignPeaks(peaks, refl, fit.a, lam, fit.zero, XFIT.tolFinal);
  if (pairs.length < 2) return { proto, a: null, n: pairs.length, nPred: refl.length };
  fit = fitLattice(peaks, refl, pairs, fit.a, lam, o.zero !== false);

  const matched = new Set(pairs.map(p => p.j));
  const matches = pairs.map(p => { const r = refl[p.ri], tc = ttOf(fit.a, r.q2, lam) + fit.zero;
    return { hkl: r.hkl, q2: r.q2, I: r.I, ttObs: peaks[p.j].tt, ttCalc: tc,
             delta: peaks[p.j].tt - tc, h: 100 * peaks[p.j].h / hmax }; });
  /* a measured peak sitting where K-alpha2 or K-beta puts an indexed reflection belongs
     to that reflection, not to another phase */
  const lines = [];
  for (const m of matches) for (const k of [1, 2]) {
    const t = ttOf(fit.a, m.q2, rad[k]);
    if (!Number.isNaN(t)) lines.push(t + fit.zero);
  }
  /* so does a peak sitting on a reflection too weak to take part in the fit */
  for (const r of every) if (r.I < XFIT.minPred) {
    const t = ttOf(fit.a, r.q2, lam);
    if (!Number.isNaN(t)) lines.push(t + fit.zero);
  }
  const unindexed = [];
  let explained = 0, total = 0;
  peaks.forEach((p, j) => {
    total += p.h;
    if (matched.has(j) || lines.some(t => Math.abs(t - p.tt) <= XFIT.tolLine)) explained += p.h;
    else unindexed.push({ tt: p.tt, h: 100 * p.h / hmax });
  });
  /* a strong reflection is expected only clear of the ends of the scan */
  const strong = refl.filter(r => { const t = ttOf(fit.a, r.q2, lam) + fit.zero;
    return r.I >= XFIT.strong && t >= lo + XFIT.edge && t <= hi - XFIT.edge; });
  const strongFound = strong.filter(r => matches.some(m => m.q2 === r.q2)).length;
  return { proto, a: fit.a, sigmaA: fit.sigmaA, zero: fit.zero, fitZero: fit.fitZero,
    rms: fit.rms, n: pairs.length, nPred: refl.length, nStrong: strong.length, strongFound,
    stripped: found.stripped,
    explained: total > 0 ? explained / total : 0, matches, unindexed,
    score: (strong.length ? strongFound / strong.length : 0) * (total > 0 ? explained / total : 0) };
}

/* the same measured pattern indexed on every cubic prototype, best explanation first:
   each one starts from the lattice parameter the platform predicts for it */
function compareMeasured(tt, I, frac, opt) {
  const found = findPeaks(tt, I, RADIATION[(opt || {}).radiation || "Cu"]);
  found.snr = found.noise > 0 ? found.max / found.noise : Infinity;
  found.quality = found.snr >= XFIT.snrClean ? "clean" : found.snr >= XFIT.snrMin ? "noisy" : "below";
  const acc = found.quality === "clean" ? XFIT.accClean : XFIT.accNoisy;
  const o = { ...(opt || {}), ttMin: tt[0], ttMax: tt[tt.length - 1] };
  const out = [];
  for (const proto of MODEL.classes) {
    if (!XRD.prototypes[proto]) continue;
    const aTh = thermoLattice(proto, frac);
    const a0 = aTh !== null ? aTh : predictA(proto, frac);
    const r = refineLattice(found, proto, frac, a0, o);
    if (r) out.push({ ...r, a0, rho: r.a ? xrayDensity(proto, frac, r.a) : null,
      aTol: r.a ? Math.max(r.sigmaA || 0, acc * r.a) : null });
  }
  out.sort((x, y) => (y.score || 0) - (x.score || 0) || x.proto.localeCompare(y.proto));
  return { found, fits: out };
}

/* text of a diffractometer export -> {tt, I}: two columns (2-theta, intensity) separated
   by spaces, tabs, commas or semicolons, decimal point or comma; or PANalytical .xrdml */
function parsePattern(text) {
  const tt = [], I = [];
  if (/<xrdMeasurement/.test(text)) {
    const pos = /<positions[^>]*axis="2Theta"[^>]*>\s*<startPosition>([^<]+)<\/startPosition>\s*<endPosition>([^<]+)<\/endPosition>/.exec(text);
    const cnt = /<(?:intensities|counts)[^>]*>([^<]+)<\/(?:intensities|counts)>/.exec(text);
    if (pos && cnt) {
      const y = cnt[1].trim().split(/\s+/).map(Number), a = parseFloat(pos[1]), b = parseFloat(pos[2]);
      y.forEach((v, i) => { tt.push(a + (b - a) * i / (y.length - 1)); I.push(v); });
    }
  } else for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /^[#;!%'"*a-zA-Z_]/.test(line)) continue;
    let tok;
    if (/[;\t]/.test(line)) tok = line.split(/[;\t]+/).map(t => t.trim().replace(",", "."));
    else if (/\s/.test(line)) tok = line.split(/\s+/).map(t => t.replace(/,$/, "").replace(",", "."));
    else tok = line.split(",");
    const x = parseFloat(tok[0]), y = parseFloat(tok[1]);
    if (tok.length >= 2 && Number.isFinite(x) && Number.isFinite(y)) { tt.push(x); I.push(y); }
  }
  const order = tt.map((_, i) => i).sort((i, j) => tt[i] - tt[j]);
  return { tt: order.map(i => tt[i]), I: order.map(i => I[i]) };
}
