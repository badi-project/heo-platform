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

  const A=r.a;
  $("latt").innerHTML = A
    ? `<dt>prototype</dt><dd>${PROTO_FR[r.top]} — ${XRD.prototypes[r.top].sg}</dd>
       <dt>paramètre <i>a</i></dt><dd>${fmt(A,3)} ± ${fmt(r.aTol,2)} Å</dd>
       <dt>volume de maille</dt><dd>${fmt(A**3,1)} Å³</dd>
       <dt>cations / maille</dt><dd>${Object.entries(XRD.prototypes[r.top].nsites).filter(([t])=>t[0]!=="O").map(([t,n])=>t+" : "+n).join(", ")}</dd>`
    : `<dt>—</dt><dd>pas de prototype cubique pour cette famille</dd>`;

  $("desc").innerHTML = REPORT_FEATS.map(n=>
    `<dt>${DESC_FR[n]||n}</dt><dd>${fmt(r.desc[n], DESC_DEC[n]??3)}</dd>`).join("");

  if (r.pattern.length){
    const tol = tt => { const th=tt*Math.PI/360; return Math.abs(2*Math.tan(th)*(LATTICE.mae_pct/100))*180/Math.PI; };
    drawXRD(r.pattern, tol);
    $("xrdnote").textContent = `${r.pattern.length} réflexions sous 80°. La bande claire est l'incertitude de position induite par `
      + `l'incertitude de ${String(LATTICE.mae_pct).replace(".", ",")} % sur a. Occupations cationiques supposées `
      + (XRD.prototypes[r.top].ordered ? "réparties entre sites A et B par rayon ionique décroissant." : "aléatoires sur tous les sites cationiques.");
  } else {
    const ctx=$("xrd").getContext("2d"); ctx.clearRect(0,0,1040,300);
    $("xrdnote").textContent = "Pas de diffractogramme simulé : cette famille n'a pas de prototype cubique dans le modèle.";
  }

  const nb=neighbours(els);
  $("nbrs").innerHTML = `<tr><th>cations</th><th>structure</th><th class="n">partagés</th><th>phase</th><th>référence</th></tr>` +
    nb.map(r2=>`<tr><td>${r2.c.replace(/-/g,"·")}</td><td>${PROTO_FR[r2.p]||r2.p}</td>
      <td class="n">${r2.shared}/${els.length}</td><td>${r2.s==="single"?"monophasé":r2.s==="multi"?"multiphasé":"—"}</td>
      <td><a href="https://doi.org/${r2.d}" target="_blank" rel="noopener">${r2.d}</a></td></tr>`).join("")
    || `<tr><td colspan="5">Aucun système partageant au moins 3 cations.</td></tr>`;
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
  $("screencount").textContent = rows.length.toLocaleString("fr-FR")+" candidats sur "+CANDS.length.toLocaleString("fr-FR")+" criblés";
  const head=[["c","composition"],["t","prototype"],["p","probabilité"],["a","a (Å)"],
              ["gt","facteur t"],["dr","δr"],["sc","S/R"],["ad","distance au connu"]];
  $("cands").innerHTML = `<tr>`+head.map(([k,l])=>
    `<th class="sortable ${["p","a","gt","dr","sc","ad"].includes(k)?"n":""}" data-k="${k}">${l}${sortKey===k?(sortDir>0?" ▲":" ▼"):""}</th>`).join("")+`</tr>`+
    rows.slice(0,400).map(c=>`<tr data-c="${c.c}" style="cursor:pointer">
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
function loadEx(list){ sel={}; list.forEach(e=>sel[e]=0); equimolar();
  document.querySelectorAll(".el").forEach(b=>b.classList.toggle("sel", b.dataset.el in sel)); render(); }
$("ex1").onclick=()=>loadEx(["Mg","Co","Ni","Cu","Zn"]);
$("ex2").onclick=()=>loadEx(["Cr","Mn","Fe","Co","Ni"]);
$("ex3").onclick=()=>loadEx(["La","Sr","Pr","Ba","Ce","Co"]);
renderCands();
loadEx(["Mg","Co","Ni","Cu","Zn"]);
