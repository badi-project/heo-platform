/* HEO designer — user interface */
const PROTO_FR = {spinel:"spinelle", perovskite:"pérovskite", rocksalt:"rocksalt (NaCl)",
  fluorite:"fluorine", pyrochlore:"pyrochlore", layered:"lamellaire"};
const PROTO_COL = {spinel:"#1b6ca8", perovskite:"#c75c1e", rocksalt:"#7b3fa0",
  fluorite:"#2e8b57", pyrochlore:"#0e7c7b", layered:"#8a6d3b"};
const FAMS = [
  ["Alcalins / alcalino-terreux", e => [1,2].includes(ELEM[e].group)],
  ["Terres rares", e => (ELEM[e].Z>=57&&ELEM[e].Z<=71)||e==="Y"||e==="Sc"],
  ["Métaux de transition 3d", e => ELEM[e].group>=3&&ELEM[e].group<=12&&ELEM[e].row===4],
  ["Métaux de transition 4d / 5d", e => ELEM[e].group>=3&&ELEM[e].group<=12&&[5,6].includes(ELEM[e].row)],
  ["Bloc p", e => ELEM[e].group>=13],
];
let sel = {};   // element -> fraction

const $ = id => document.getElementById(id);
const fmt = (v, n=3) => (v===null||v===undefined||Number.isNaN(v)) ? "—"
  : Number(v).toLocaleString("fr-FR",{minimumFractionDigits:n, maximumFractionDigits:n});

/* some database entries are arXiv preprints with no DOI; they reach the
   payload as NaN. Rendering them as a doi.org link produced dead https://doi.org/NaN
   references, so every DOI goes through hasDOI / refCell. */
const esc = s => String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;")
  .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
const hasDOI = d => d !== null && d !== undefined && d !== "" &&
  !(typeof d === "number" && Number.isNaN(d)) && String(d).trim().toLowerCase() !== "nan";
const refCell = d => hasDOI(d)
  ? `<a href="https://doi.org/${encodeURI(String(d).trim())}" target="_blank" rel="noopener noreferrer">${esc(d)}</a>`
  : `<span class="nodoi" title="notice sans DOI — préprint arXiv ou référence interne">sans DOI</span>`;

/* number of compositions enumerated in the screening (CANDS is the exported subset) and
   phase-label counts: both computed by tools/build_app.py, never typed by hand */
const SCREEN_TOTAL = SCREEN.total;
const PHASE_CAVEAT = "La colonne « phase » reproduit ce que rapporte la littérature, "
  + `qui compte ${PHASES.single} lots monophasés contre ${PHASES.multi} multiphasés : `
  + "l'absence de phase secondaire rapportée n'est pas une preuve de monophasage.";

/* ---------- picker ---------- */
function buildPicker(){
  const used = new Set();
  const host = $("picker");
  for (const [label, test] of FAMS){
    const els = CATIONS.filter(e => !used.has(e) && test(e));
    els.forEach(e => used.add(e));
    if (!els.length) continue;
    els.sort((a,b)=>(DBCOUNT[b]||0)-(DBCOUNT[a]||0));
    const d = document.createElement("div"); d.className="fam";
    d.innerHTML = `<div class="lab">${label}</div><div class="els">` +
      els.map(e=>`<button class="el" data-el="${e}">${e}<small>${DBCOUNT[e]||0}</small></button>`).join("") +
      `</div>`;
    host.appendChild(d);
  }
  host.querySelectorAll(".el").forEach(b=>b.onclick=()=>{
    const e=b.dataset.el;
    if (e in sel) delete sel[e]; else sel[e]=0;
    equimolar(); render();
  });
}
function equimolar(){ const k=Object.keys(sel); k.forEach(e=>sel[e]=1/k.length); }

/* ---------- formula input ---------- */
const FLAG_FR = {
  minor_component_dropped: "dopant ou métal déposé écarté",
  dopant_dropped: "dopant écarté", composite_main_phase: "composite : phase à haute entropie retenue",
  variable: "variable x évaluée à 0,1", placeholder_mixed_site: "site mixte développé",
  placeholder_variant: "première variante retenue", oxide_sum: "somme d'oxydes",
  per_element_multiplier: "coefficient appliqué à chaque élément (bilan de charge)",
  abo3_bare_sequence: "ABO₃ écrit sans parenthèses : gros cations en site A, les autres en site B",
};
function syncPicker(){
  document.querySelectorAll(".el").forEach(b=>b.classList.toggle("sel", b.dataset.el in sel));
}
function loadFormula(text){
  const fr = FX.cationFractions(text);
  const els = Object.keys(fr);
  const note = $("fxnote");
  const unknown = els.filter(e => !(e in ELEM));
  if (els.length < 2 || unknown.length){
    note.className = "note fxnote-warn";
    note.textContent = unknown.length
      ? `Cation sans propriétés tabulées : ${unknown.join(", ")}. Prédiction impossible.`
      : "Formule non lue : écrivez-la comme dans un article, par exemple La(CrMnFeCoNi)O3.";
    return;
  }
  sel = {}; for (const e of els) sel[e] = Math.round(fr[e]*1e6)/1e6;
  const flags = FX.parse(text).flags.map(f => FLAG_FR[f]).filter(Boolean);
  note.className = "note";
  note.textContent = "Lu : " + els.map(e => `${e} ${(100*fr[e]).toFixed(1).replace(".", ",")} %`).join(" · ")
    + (flags.length ? ` — ${flags.join(" ; ")}.` : ".");
  syncPicker(); render();
}

function renderChips(){
  const k=Object.keys(sel).sort();
  $("chips").innerHTML = k.map(e=>
    `<div class="chip"><b>${e}</b><input type="number" step="0.01" min="0" max="1" value="${sel[e].toFixed(3)}" data-el="${e}">
     <span style="color:#9aa0a6;font-size:11px">${(sel[e]*100).toFixed(1)} %</span>
     <button class="x" data-del="${e}">×</button></div>`).join("")
    || `<div class="note">Aucun cation sélectionné.</div>`;
  $("chips").querySelectorAll("input").forEach(i=>i.onchange=()=>{
    sel[i.dataset.el]=Math.max(0,parseFloat(i.value)||0); render();});
  $("chips").querySelectorAll("[data-del]").forEach(b=>b.onclick=()=>{
    delete sel[b.dataset.del]; equimolar(); render();});
  const s=Object.values(sel).reduce((a,b)=>a+b,0);
  $("sumnote").textContent = k.length
    ? `Somme = ${s.toFixed(3)} — les fractions sont renormalisées à 1 avant le calcul.`
    : "";
}

/* ---------- neighbours ---------- */
function neighbours(els){
  const S=new Set(els);
  const out=[];
  for (const r of DB){
    const c=r.c.split("-");
    let sh=0; for (const e of c) if (S.has(e)) sh++;
    if (sh>=Math.min(3, els.length-1)) out.push({...r, shared:sh, jac:sh/(new Set([...c,...els])).size});
  }
  out.sort((a,b)=> b.jac-a.jac || b.shared-a.shared);
  const seen=new Set(), keep=[];
  for (const r of out){ const k=r.c+"|"+r.p; if(seen.has(k))continue; seen.add(k); keep.push(r); if(keep.length>=8)break; }
  return keep;
}

/* ---------- XRD canvas ---------- */
function drawXRD(peaks, tol2th){
  const cv=$("xrd"), ctx=cv.getContext("2d");
  const W=cv.width, H=cv.height, L=46, R=12, T=12, B=34;
  ctx.clearRect(0,0,W,H); ctx.fillStyle="#fff"; ctx.fillRect(0,0,W,H);
  const x0=10, x1=80;
  const px = t => L + (t-x0)/(x1-x0)*(W-L-R);
  const py = i => H-B - i/100*(H-T-B);
  ctx.strokeStyle="#d9dde1"; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(L,T); ctx.lineTo(L,H-B); ctx.lineTo(W-R,H-B); ctx.stroke();
  ctx.fillStyle="#51585f"; ctx.font="11px -apple-system,Segoe UI,Arial";
  for(let t=10;t<=80;t+=10){
    ctx.strokeStyle="#eef0f2"; ctx.beginPath(); ctx.moveTo(px(t),T); ctx.lineTo(px(t),H-B); ctx.stroke();
    ctx.textAlign="center"; ctx.fillText(t+"°", px(t), H-B+15);
  }
  ctx.textAlign="right";
  for(const v of [0,50,100]) ctx.fillText(v, L-6, py(v)+4);
  ctx.save(); ctx.translate(13,(H-B+T)/2); ctx.rotate(-Math.PI/2); ctx.textAlign="center";
  ctx.fillText("intensité relative", 0,0); ctx.restore();
  ctx.textAlign="center"; ctx.fillText("2θ (degrés)", (L+W-R)/2, H-6);
  const col = PROTO_COL[CURRENT.top] || "#1b6ca8";
  for(const p of peaks){
    const dt = tol2th(p.tt);
    ctx.fillStyle = col+"26";
    ctx.fillRect(px(p.tt-dt), py(p.I), Math.max(px(p.tt+dt)-px(p.tt-dt),1.5), py(0)-py(p.I));
    ctx.strokeStyle=col; ctx.lineWidth=1.6;
    ctx.beginPath(); ctx.moveTo(px(p.tt),py(0)); ctx.lineTo(px(p.tt),py(p.I)); ctx.stroke();
  }
  const lab=[...peaks].sort((a,b)=>b.I-a.I).slice(0,6);
  ctx.fillStyle="#1d2228"; ctx.font="10px -apple-system,Segoe UI,Arial"; ctx.textAlign="center";
  for(const p of lab) ctx.fillText("("+p.hkl+")", px(p.tt), py(p.I)-4);
}

/* ---------- main render ---------- */
let CURRENT = {};
function render(){
  renderChips();
  const els=Object.keys(sel);
  const s=Object.values(sel).reduce((a,b)=>a+b,0);
  if (els.length<4 || s<=0){ $("out").className="hide"; $("empty").className="card"; return; }
  $("empty").className="hide"; $("out").className="";
  const frac={}; els.forEach(e=>frac[e]=sel[e]/s);
  const r=predict(frac); CURRENT=r;

  $("topclass").textContent = PROTO_FR[r.top];
  $("topprob").textContent = "probabilité " + (100*r.order[0].p).toFixed(0) + " %";
  $("dombadge").innerHTML = r.inDomain
    ? `<span class="badge in">dans le domaine d'applicabilité</span>`
    : `<span class="badge out">hors domaine — prédiction non fiable</span>`;
  $("dombox").innerHTML = r.inDomain ? "" :
    `<div class="warnbox">Cette composition est à ${fmt(r.adDistance,2)} unités du jeu d'entraînement
     (seuil ${fmt(AD.threshold,2)}). Aucun système comparable n'a été publié : le modèle extrapole et
     la probabilité affichée n'a pas de valeur. Traitez-la comme une absence de prédiction.</div>`;

  const key=els.slice().sort().join("-");
  const known=DBSET[key];
  $("novbadge").innerHTML = known
    ? `<span class="badge known">déjà publié (${known} entrée${known>1?"s":""})</span>`
    : `<span class="badge nov">combinaison absente de la base</span>`;

  $("bars").innerHTML = r.order.map(o=>
    `<div class="bar"><span>${PROTO_FR[o.c]}</span>
     <span><span class="t" style="display:block;width:${(100*o.p).toFixed(1)}%;background:${PROTO_COL[o.c]}"></span></span>
     <span class="v">${(100*o.p).toFixed(0)} %</span></div>`).join("");
  const second=r.order[1];
  $("protonote").textContent = (r.order[0].p-second.p<0.25)
    ? `Prédiction peu tranchée : ${PROTO_FR[r.top]} et ${PROTO_FR[second.c]} sont à moins de 25 points. Attendez-vous à un mélange de phases ou à une sensibilité forte à la température de calcination.`
    : `Marge de ${(100*(r.order[0].p-second.p)).toFixed(0)} points sur la deuxième hypothèse (${PROTO_FR[second.c]}).`;

  /* r.aSrc tells which lattice model predict() used (heo_engine.js) */
  const aPct = r.aPct, aSrc = r.aSrc;
  $("lattnote").textContent = aSrc === "chgnet"
    ? `Potentiel CHGNet, modèle additif par cation calibré sur 17 spinelles binaires de maille connue. `
      + `Écart absolu moyen sur les spinelles HEO mesurés : ${fmt(aPct, 2)} % (modèle géométrique : ${fmt(LATTICE.mae_pct, 1)} %).`
    : `Modèle géométrique à partir des rayons de Shannon, recalé sur un composé de référence par prototype. `
      + `Écart absolu moyen mesuré sur les valeurs expérimentales disponibles : ${fmt(LATTICE.mae_pct, 1)} %.`;
  /* density: what the stated uncertainty rests on, from tools/validate_density.py */
  const seen = (LATTICE.rho || {})[aSrc];
  $("rhonote").textContent = !r.rho ? "" :
    "Densité théorique : masse de la maille du prototype, sous-réseau d'oxygène complet, divisée par son volume. "
    + (seen ? `Sur les ${seen.n} oxydes à haute entropie de maille mesurée traités par ce modèle de maille, `
        + `l'écart moyen est de ${fmt(seen.mae_pct, 1)} % (maximum ${fmt(seen.max_pct, 1)} %)`
        + (seen.met ? ". " : " : valeur indicative. ")
      : "Aucune maille mesurée ne permet de la vérifier pour cette famille. ")
    + "Avec une maille mesurée, elle est exacte.";
  const A=r.a;
  $("latt").innerHTML = A
    ? `<dt>prototype</dt><dd>${PROTO_FR[r.top]} — ${XRD.prototypes[r.top].sg}</dd>
       <dt>paramètre <i>a</i></dt><dd>${fmt(A,3)} ± ${fmt(r.aTol,2)} Å</dd>
       <dt>volume de maille</dt><dd>${fmt(A**3,1)} Å³</dd>
       <dt>densité théorique</dt><dd>${fmt(r.rho,2)} ± ${fmt(r.rhoTol,2)} g/cm³</dd>
       <dt>cations / maille</dt><dd>${Object.entries(XRD.prototypes[r.top].nsites).filter(([t])=>t[0]!=="O").map(([t,n])=>t+" : "+n).join(", ")}</dd>`
    : `<dt>—</dt><dd>pas de prototype cubique pour cette famille</dd>`;

  $("desc").innerHTML = REPORT_FEATS.map(n=>
    `<dt>${DESC_FR[n]||n}</dt><dd>${fmt(r.desc[n], DESC_DEC[n]??3)}</dd>`).join("");
  renderThermo(r);

  if (r.pattern.length){
    const tol = tt => { const th=tt*Math.PI/360; return Math.abs(2*Math.tan(th)*(aPct/100))*180/Math.PI; };
    drawXRD(r.pattern, tol);
    $("xrdnote").textContent = `${r.pattern.length} réflexions sous 80°. La bande claire est l'incertitude de position induite par `
      + `l'incertitude de ${fmt(aPct, 2)} % sur a. Occupations cationiques supposées `
      + (XRD.prototypes[r.top].ordered ? "réparties entre sites A et B par rayon ionique décroissant." : "aléatoires sur tous les sites cationiques.");
  } else {
    const ctx=$("xrd").getContext("2d"); ctx.clearRect(0,0,1040,300);
    $("xrdnote").textContent = "Pas de diffractogramme simulé : cette famille n'a pas de prototype cubique dans le modèle.";
  }

  const nb=neighbours(els);
  $("nbrs").innerHTML = `<tr><th>cations</th><th>structure</th><th class="n">partagés</th><th>phase</th><th>référence</th></tr>` +
    (nb.map(r2=>`<tr><td>${r2.c.replace(/-/g,"·")}</td><td>${PROTO_FR[r2.p]||r2.p}</td>
      <td class="n">${r2.shared}/${els.length}</td><td>${r2.s==="single"?"monophasé":r2.s==="multi"?"multiphasé":"—"}</td>
      <td>${refCell(r2.d)}</td></tr>`).join("")
    || `<tr><td colspan="5">Aucun système partageant au moins 3 cations.</td></tr>`)
    + `<tr><td colspan="5" class="caveat">${PHASE_CAVEAT}</td></tr>`;
}

/* ---------- computed thermodynamics (thermo/, CHGNet) ---------- */
function renderThermo(r){
  const card = $("thermocard");
  if (!THERMO || !r.thermo){ card.className = "hide"; return; }
  card.className = "card";
  const ox = THERMO.ox, th = r.thermo, inDomain = r.inDomain;
  const cov = ox.cov || ox.elements, missing = th.missing;
  const pctFR = v => (100 * v).toFixed(0) + " %";
  if (!th.covered){
    $("thermo").innerHTML = `<dt>hors du calcul</dt><dd>cation${missing.length > 1 ? "s" : ""} non couvert${missing.length > 1 ? "s" : ""} : ${missing.join(", ")}</dd>`;
    $("thermonote").textContent = `Le calcul CHGNet couvre les cations ${cov.join(" ")} — `
      + "ceux d'au moins deux systèmes rocksalt ou spinelle calculés. Ailleurs, seule la prédiction statistique s'applique.";
    return;
  }
  const exact = th.system, p = th.pRocksalt;
  const verdict = {rocksalt: "plutôt rocksalt", spinel: "plutôt spinelle", boundary: "zone frontière"}[th.verdict];
  const rows = th.direct
    ? [`<dt>oxydation rocksalt → spinelle</dt><dd>${fmt(th.dh, 2)} eV/O, calcul direct de cette composition</dd>`]
    : [`<dt>oxydation rocksalt → spinelle</dt><dd>${fmt(th.dh, 2)} ± ${fmt(ox.mae, 2)} eV/O (modèle additif)</dd>`];
  rows.push(`<dt>dans l'air, la littérature rapporte</dt><dd>${verdict} (rocksalt ${pctFR(p)})</dd>`);
  if (r.aSrc === "chgnet")
    rows.push(`<dt>paramètre <i>a</i> si spinelle</dt><dd>${fmt(r.a, 3)} Å ± ${fmt(THERMO.lat.mae_pct, 1)} %</dd>`);
  if (exact && exact.rs)
    rows.push(`<dt>rocksalt : mélange, désordre</dt><dd>${fmt(exact.rs.dh, 1)} meV/atome, σ ${fmt(exact.rs.sigma, 3)} Å — `
      + (exact.rs.single ? "dans les seuils du monophasé" : "hors des seuils : multiphasé attendu") + `</dd>`);
  if (exact && exact.ev !== undefined)
    rows.push(`<dt>lacune d'oxygène (spinelle)</dt><dd>${fmt(exact.ev, 2)} eV en moyenne, ${fmt(exact.evmin, 2)} eV au plus facile</dd>`);
  if (!inDomain)
    rows.unshift(`<dt>hors domaine statistique</dt><dd>ce calcul ne dépend pas des données publiées. `
      + `Testé en retirant tour à tour chaque cation de l'entraînement, s'appuyer dessus hors domaine `
      + `fait passer l'exactitude rocksalt / spinelle de ${pctFR(ox.lco_comp)} à ${pctFR(ox.lco_switch)} `
      + `en moyenne — gain net mais inégal selon le cation</dd>`);
  $("thermo").innerHTML = rows.join("");
  $("thermonote").textContent = "Enthalpie d'oxydation 3 MO + O → M₃O₄ calculée avec le potentiel universel CHGNet "
    + `sur ${ox.n} systèmes de la base, corrigée par élément sur des couples redox mesurés, puis résumée par un modèle `
    + `additif par cation (erreur un-contre-tous ${fmt(ox.mae, 2)} eV/O). Elle sépare les systèmes rapportés rocksalt `
    + `et spinelle avec une AUC de ${fmt(ox.auc, 2)}. Plus elle est négative, plus les cations prennent l'oxygène : `
    + "spinelle à l'air, rocksalt seulement sous atmosphère réductrice ou à haute température. "
    + "C'est une tendance thermodynamique, pas une garantie de monophasage.";
}

/* ---------- candidates ---------- */
let sortKey="p", sortDir=-1;
function renderCands(){
  const pmin=parseFloat($("f-p").value)||0, pf=$("f-proto").value, nf=$("f-n").value;
  const nov=$("f-nov").checked, dom=$("f-dom").checked, noco=$("f-co").checked, nore=$("f-re").checked;
  let rows=CANDS.filter(c=> c.p>=pmin
    && (!pf||c.t===pf) && (!nf||c.n===+nf)
    && (!nov||c.new===1) && (!dom||c.in===1)
    && (!noco||!c.c.split("-").includes("Co"))
    && (!nore||!c.c.split("-").some(e=>(ELEM[e].Z>=57&&ELEM[e].Z<=71)||e==="Y"||e==="Sc")));
  rows.sort((a,b)=>(a[sortKey]>b[sortKey]?1:a[sortKey]<b[sortKey]?-1:0)*sortDir);
  const MAXROWS = 400, nf2 = n => n.toLocaleString("fr-FR");
  $("screencount").textContent =
    `${nf2(rows.length)} candidats retenus sur ${nf2(CANDS.length)} exportés `
    + `(sous-ensemble du criblage de ${nf2(SCREEN_TOTAL)} compositions, au plus ${SCREEN.cap} par prototype ; `
    + `le criblage intégral est dans screening_all.csv.gz)`
    + (rows.length > MAXROWS ? ` — seules les ${nf2(MAXROWS)} premières lignes du tri courant sont affichées.` : "");
  const head=[["c","composition"],["t","prototype"],["p","probabilité"],["a","a (Å)"],
              ["gt","facteur t"],["dr","δr"],["sc","S/R"],["ad","distance au connu"]];
  $("cands").innerHTML = `<tr>`+head.map(([k,l])=>
    `<th class="sortable ${["p","a","gt","dr","sc","ad"].includes(k)?"n":""}" data-k="${k}">${l}${sortKey===k?(sortDir>0?" ▲":" ▼"):""}</th>`).join("")+`</tr>`+
    rows.slice(0,MAXROWS).map(c=>`<tr data-c="${c.c}" style="cursor:pointer">
      <td>${c.c.replace(/-/g,"·")}</td><td style="color:${PROTO_COL[c.t]}">${PROTO_FR[c.t]}</td>
      <td class="n">${(100*c.p).toFixed(0)} %</td><td class="n">${c.a?fmt(c.a,3):"—"}</td>
      <td class="n">${fmt(c.gt,3)}</td><td class="n">${fmt(c.dr,3)}</td>
      <td class="n">${fmt(c.sc,2)}</td><td class="n">${fmt(c.ad,2)}</td></tr>`).join("");
  $("cands").querySelectorAll("th.sortable").forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(k===sortKey) sortDir*=-1; else {sortKey=k; sortDir=(k==="c"||k==="t")?1:-1;}
    renderCands();});
  $("cands").querySelectorAll("tr[data-c]").forEach(tr=>tr.onclick=()=>{
    sel={}; tr.dataset.c.split("-").forEach(e=>sel[e]=0); equimolar();
    document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("on", b.dataset.tab==="design"));
    ["design","screen","method"].forEach(t=>$(t).className = t==="design"?"":"hide");
    document.querySelectorAll(".el").forEach(b=>b.classList.toggle("sel", b.dataset.el in sel));
    render(); window.scrollTo(0,0);});
}

/* ---------- init ---------- */
document.querySelectorAll("nav button").forEach(b=>b.onclick=()=>{
  document.querySelectorAll("nav button").forEach(x=>x.classList.toggle("on", x===b));
  ["design","screen","method"].forEach(t=>$(t).className = t===b.dataset.tab?"":"hide");});
buildPicker();
$("f-proto").innerHTML = `<option value="">tous</option>`+
  MODEL.classes.map(c=>`<option value="${c}">${PROTO_FR[c]}</option>`).join("");
$("methodtext").innerHTML = METHOD_HTML;
$("btn-equi").onclick=()=>{equimolar(); render();};
$("btn-clear").onclick=()=>{sel={}; document.querySelectorAll(".el").forEach(b=>b.classList.remove("sel")); render();};
function loadEx(text){ $("fx").value = text; loadFormula(text); }
$("btn-fx").onclick=()=>loadFormula($("fx").value);
$("fx").addEventListener("keydown", e=>{ if (e.key==="Enter") loadFormula($("fx").value); });
$("ex1").onclick=()=>loadEx("(Mg0.2Co0.2Ni0.2Cu0.2Zn0.2)O");
$("ex2").onclick=()=>loadEx("(CrMnFeCoNi)3O4");
$("ex3").onclick=()=>loadEx("(La0.2Pr0.2Nd0.2Ba0.2Sr0.2)CoO3");
$("ex4").onclick=()=>loadEx("(Na0.2Bi0.2Ba0.2Sr0.2Ca0.2)TiO3");
renderCands();
loadEx("(Mg0.2Co0.2Ni0.2Cu0.2Zn0.2)O");
