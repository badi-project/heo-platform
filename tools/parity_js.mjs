/* Node driver for the JavaScript side of the Python/JS parity check.
   Loads app/heo_engine.js in a bare context with the globals the browser page
   supplies from its inlined payload, scores the 200 probe compositions shipped in
   app/heo_model.json, and prints {probs, feats} as JSON on stdout. With the argument
   "formulas" it instead parses the formula strings given as a JSON list on stdin with
   FX.cationFractions and prints the cation fractions. With the argument "predict" it
   reads {context, fracs} on stdin — the globals of the page, as assembled by
   src/predict.py, and a list of compositions — and prints what predict() returns for
   each: the complete prediction, to be compared with src/predict.py — plus the flat
   record of each formula in "texts" and the continuous profile of each composition in
   "profiles", and the indexing of each measured pattern in "measured".
   Called by tools/parity_check.py; not used by the page itself. */
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const read = (...p) => readFileSync(join(ROOT, ...p), "utf8");
const bundle = JSON.parse(read("app", "heo_model.json"));

const ctx = createContext({
  ELEM: JSON.parse(read("data", "reference", "element_properties.json")),
  PROPS: bundle.props,
  ONEHOT: bundle.onehot,
  MODEL: bundle.model,
});
runInContext(read("app", "heo_engine.js"), ctx, { filename: "heo_engine.js" });
runInContext(
  `globalThis.score = frac => { const x = featurize(frac);
     return { feats: x, probs: softmax(rawPredict(MODEL, x), MODEL.temperature) }; };`,
  ctx);

if (process.argv[2] === "predict") {
  /* full-prediction parity: both engines are fed the very same data */
  const job = JSON.parse(readFileSync(0, "utf8"));
  const full = createContext({ ...job.context });
  runInContext(read("app", "heo_engine.js"), full, { filename: "heo_engine.js" });
  runInContext(
    `globalThis.run = frac => { const r = predict(frac);
       return { probs: r.probs, top: r.top, ad: r.adDistance, inDomain: r.inDomain,
                a: r.a, aPct: r.aPct, aSrc: r.aSrc, rho: r.rho, pattern: r.pattern, thermo: r.thermo }; };`,
    full);
  runInContext(
    `globalThis.rows = texts => texts.map(predictRow);
     globalThis.prof = frac => { const r = predict(frac); return patternProfile(r.pattern).I; };
     globalThis.fitm = m => { const c = compareMeasured(m.tt, m.I, m.frac, { radiation: m.radiation, zero: m.zero });
       return { peaks: c.found.peaks, fits: c.fits.map(f => ({ proto: f.proto, a: f.a, a0: f.a0,
         sigma: f.sigmaA ?? null, zero: f.zero ?? null, rms: f.rms ?? null, n: f.n, nPred: f.nPred,
         explained: f.explained ?? null, score: f.score ?? null, rho: f.rho,
         unindexed: (f.unindexed || []).map(u => u.tt), hkl: (f.matches || []).map(x => x.hkl) })) }; };`,
    full);
  process.stdout.write(JSON.stringify({
    full: job.fracs.map(f => full.run(f)),
    rows: full.rows(job.texts || []),
    profiles: (job.profiles || []).map(f => full.prof(f)),
    measured: (job.measured || []).map(m => full.fitm(m)),
  }));
} else if (process.argv[2] === "formulas") {
  /* formula-parser parity: a JSON list of formula strings on stdin */
  const formulas = JSON.parse(readFileSync(0, "utf8"));
  runInContext("globalThis.fx = t => FX.cationFractions(t);", ctx);
  process.stdout.write(JSON.stringify(formulas.map(t => ctx.fx(t))));
} else {
  const out = bundle.probe.map(p => ctx.score(p.frac));
  process.stdout.write(JSON.stringify({
    probs: out.map(o => o.probs),
    feats: out.map(o => o.feats),
  }));
}
