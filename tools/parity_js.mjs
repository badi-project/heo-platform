/* Node driver for the JavaScript side of the Python/JS parity check.
   Loads app/heo_engine.js in a bare context with the globals the browser page
   supplies from its inlined payload, scores the 200 probe compositions shipped in
   app/heo_model.json, and prints {probs, feats} as JSON on stdout. With the argument
   "formulas" it instead parses the formula strings given as a JSON list on stdin with
   FX.cationFractions and prints the cation fractions. With the argument "predict" it
   reads {context, fracs} on stdin — the globals of the page, as assembled by
   src/predict.py, and a list of compositions — and prints what predict() returns for
   each: the complete prediction, to be compared with src/predict.py.
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
                a: r.a, aPct: r.aPct, aSrc: r.aSrc, pattern: r.pattern, thermo: r.thermo }; };`,
    full);
  process.stdout.write(JSON.stringify(job.fracs.map(f => full.run(f))));
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
