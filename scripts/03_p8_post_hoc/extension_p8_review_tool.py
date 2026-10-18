# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""
extension_p8_review_tool.py -- turns the blinded review sheet into a local
review page, so the 240 items can be labelled one at a time in a browser
instead of in a spreadsheet.

Reads   ~/extension_results/p8_adjudication/review_sheet.csv
        (written by extension_p8_vaso_diagnostic_v2.py; it holds only
        review_id and the excerpts -- no quartile, LLM answer or category)
Writes  ~/extension_results/p8_adjudication/review_tool.html

The page is a single self-contained file: no network access, nothing leaves
the machine. Labels are kept in the browser as you go (closing the page is
safe) and are exported with the "Export labels" button as review_labels.csv
(columns review_id, label, comment) into your Downloads folder.

USAGE
    uv run --offline extension_p8_review_tool.py
    open ~/extension_results/p8_adjudication/review_tool.html
    ... label, click "Export labels" ...
    uv run --offline extension_p8_vaso_diagnostic_v2.py --labels ~/Downloads/review_labels.csv

KEYS on the page
    A / O / N / U   set the label and move to the next item
    left / right    previous / next item
    J               jump to the next unlabelled item
    Enter           leave the comment box (keys work again)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

DEF_DIR = Path.home() / "extension_results" / "p8_adjudication"

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>P8 vasopressor review</title>
<style>
:root { --bg:#f7f7f5; --card:#fff; --ink:#1d1d1f; --mute:#6b6b70; --line:#dcdcdf;
        --A:#1f7a3a; --O:#8a5a00; --N:#a3262a; --U:#55565c; --hl:#ffe58a; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:15px/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; }
header { position:sticky; top:0; z-index:2; background:var(--card); border-bottom:1px solid var(--line);
         padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:16px; margin:0 12px 0 0; }
#progress { color:var(--mute); }
button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:6px;
         padding:6px 12px; cursor:pointer; }
button:hover { border-color:#999; }
button.primary { background:#1d1d1f; color:#fff; border-color:#1d1d1f; }
main { max-width:980px; margin:0 auto; padding:16px; }
details { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:8px 12px; margin-bottom:12px; }
details dl { display:grid; grid-template-columns:28px 1fr; gap:4px 10px; margin:8px 0 0; }
details dt { font-weight:700; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; }
.meta { display:flex; justify-content:space-between; color:var(--mute); margin-bottom:10px; }
.ex { border-left:3px solid var(--line); padding:6px 12px; margin:10px 0; white-space:pre-wrap;
      font-family:"SF Mono", Menlo, Consolas, monospace; font-size:13.5px; }
mark { background:var(--hl); font-weight:700; padding:0 1px; }
.labels { display:flex; gap:10px; flex-wrap:wrap; margin:14px 0 8px; }
.lab { min-width:150px; text-align:left; padding:8px 12px; }
.lab b { display:inline-block; width:22px; }
.lab.on.A { background:var(--A); color:#fff; border-color:var(--A); }
.lab.on.O { background:var(--O); color:#fff; border-color:var(--O); }
.lab.on.N { background:var(--N); color:#fff; border-color:var(--N); }
.lab.on.U { background:var(--U); color:#fff; border-color:var(--U); }
#comment { width:100%; font:inherit; padding:6px 8px; border:1px solid var(--line); border-radius:6px; }
.nav { display:flex; justify-content:space-between; margin-top:12px; }
#grid { display:flex; flex-wrap:wrap; gap:3px; margin-top:16px; }
#grid span { width:14px; height:14px; border-radius:3px; background:#e4e4e7; cursor:pointer; }
#grid span.A { background:var(--A); } #grid span.O { background:var(--O); }
#grid span.N { background:var(--N); } #grid span.U { background:var(--U); }
#grid span.cur { outline:2px solid #1d1d1f; outline-offset:1px; }
#saved { color:var(--mute); font-size:13px; }
</style></head>
<body>
<header>
  <h1>P8 vasopressor review</h1>
  <span id="progress"></span>
  <button id="next-un">Next unlabelled (J)</button>
  <button id="export" class="primary">Export labels</button>
  <label><button id="import-btn" type="button">Import labels</button>
    <input id="import" type="file" accept=".csv,text/csv" hidden></label>
  <span id="saved"></span>
</header>
<main>
<details open><summary><b>Question for each item</b>: do these excerpts document intravenous vasopressor administration during THIS admission?</summary>
<dl>
<dt>A</dt><dd><b>Administered this admission</b>: IV norepinephrine, epinephrine, phenylephrine, vasopressin or dopamine given during this admission, named generically, by brand, or as pressors/vasopressors. If given both before and during this admission: A.</dd>
<dt>O</dt><dd><b>Other admission only</b>: administration documented, but only before this admission (history, earlier admission, another hospital before transfer).</dd>
<dt>N</dt><dd><b>Not administration</b>: allergy, local anaesthetic, eye drops, dopamine agonist, a plan or consideration only, or negated.</dd>
<dt>U</dt><dd><b>Unclear</b> from the excerpts.</dd>
</dl>
<p style="margin:8px 0 0;color:var(--mute)">Highlighted words are the vasopressor terms the pattern matched. Separate passages of the same summary are shown as separate blocks. Keys: A / O / N / U label and advance; ← → move; J next unlabelled. In the comment box, Enter returns to the keys.</p>
</details>
<div class="card">
  <div class="meta"><span id="itemid"></span><span id="pos"></span></div>
  <div id="excerpts"></div>
  <div class="labels" id="labels"></div>
  <input id="comment" placeholder="Optional comment">
  <div class="nav"><button id="prev">← Previous</button><button id="next">Next →</button></div>
</div>
<div id="grid"></div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = "p8_review_" + __SIG__;
const TERM = /(norepinephrine|levophed|epinephrine|phenylephrine|neo-?synephrine|vasopressin|dopamine|vasopressors?|pressors?)/gi;
const LABELS = [["A","Administered this admission"],["O","Other admission only"],["N","Not administration"],["U","Unclear"]];
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { state = {}; }
let cur = 0;
const $ = id => document.getElementById(id);

function save() {
  try { localStorage.setItem(KEY, JSON.stringify(state)); $("saved").textContent = "saved in this browser"; }
  catch (e) { $("saved").textContent = "browser storage unavailable: export regularly"; }
}
function addText(el, text) {
  let last = 0; TERM.lastIndex = 0; let m;
  while ((m = TERM.exec(text)) !== null) {
    if (m.index > last) el.appendChild(document.createTextNode(text.slice(last, m.index)));
    const mk = document.createElement("mark"); mk.textContent = m[0]; el.appendChild(mk);
    last = m.index + m[0].length;
  }
  if (last < text.length) el.appendChild(document.createTextNode(text.slice(last)));
}
function render() {
  if (document.activeElement && document.activeElement.id === "comment") document.activeElement.blur();
  const it = ITEMS[cur], st = state[it.id] || {};
  $("itemid").textContent = "Item " + it.id;
  $("pos").textContent = (cur + 1) + " of " + ITEMS.length;
  const ex = $("excerpts"); ex.innerHTML = "";
  it.excerpts.split(" ||| ").forEach(p => { const d = document.createElement("div"); d.className = "ex"; addText(d, p); ex.appendChild(d); });
  const lb = $("labels"); lb.innerHTML = "";
  LABELS.forEach(([k, t]) => {
    const b = document.createElement("button"); b.className = "lab " + k + (st.label === k ? " on" : "");
    b.innerHTML = "<b>" + k + "</b>" + t; b.onclick = () => setLabel(k); lb.appendChild(b);
  });
  $("comment").value = st.comment || "";
  const n = ITEMS.filter(x => state[x.id] && state[x.id].label).length;
  $("progress").textContent = "Labelled " + n + " / " + ITEMS.length;
  const g = $("grid"); g.innerHTML = "";
  ITEMS.forEach((x, i) => {
    const s = document.createElement("span"); const l = state[x.id] && state[x.id].label;
    s.className = (l || "") + (i === cur ? " cur" : ""); s.title = x.id + (l ? ": " + l : "");
    s.onclick = () => { cur = i; render(); }; g.appendChild(s);
  });
  window.scrollTo(0, 0);
}
function setLabel(k) {
  const id = ITEMS[cur].id; state[id] = Object.assign(state[id] || {}, { label: k }); save();
  if (cur < ITEMS.length - 1) cur++; render();
}
function nextUnlabelled() {
  for (let j = 1; j <= ITEMS.length; j++) {
    const i = (cur + j) % ITEMS.length;
    if (!(state[ITEMS[i].id] && state[ITEMS[i].id].label)) { cur = i; render(); return; }
  }
  alert("All items are labelled. Click Export labels.");
}
function csvCell(v) { v = String(v == null ? "" : v); return /[",\n;]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }
function exportCsv() {
  const rows = [["review_id", "label", "comment"]];
  ITEMS.forEach(x => { const s = state[x.id] || {}; rows.push([x.id, s.label || "", s.comment || ""]); });
  const blob = new Blob([rows.map(r => r.map(csvCell).join(",")).join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "review_labels.csv";
  document.body.appendChild(a); a.click(); a.remove();
}
function parseCsv(t) {
  const out = []; let row = [], f = "", q = false;
  for (let i = 0; i < t.length; i++) {
    const c = t[i];
    if (q) { if (c === '"') { if (t[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === "," || c === ";") { row.push(f); f = ""; }
    else if (c === "\n" || c === "\r") { if (c === "\r" && t[i + 1] === "\n") i++; row.push(f); out.push(row); row = []; f = ""; }
    else f += c;
  }
  if (f || row.length) { row.push(f); out.push(row); }
  return out;
}
$("import").onchange = ev => {
  const file = ev.target.files[0]; if (!file) return;
  file.text().then(t => {
    const rows = parseCsv(t.replace(/^﻿/, "")); const h = rows.shift().map(x => x.trim().toLowerCase());
    const iid = h.indexOf("review_id"), il = h.indexOf("label"), ic = h.indexOf("comment");
    let n = 0;
    rows.forEach(r => { const id = (r[iid] || "").trim(); const l = (r[il] || "").trim().toUpperCase();
      if (id && "AONU".includes(l) && l.length === 1) { state[id] = { label: l, comment: ic >= 0 ? (r[ic] || "") : "" }; n++; } });
    save(); render(); alert("Imported " + n + " labels.");
  });
};
$("import-btn").onclick = () => $("import").click();
$("export").onclick = exportCsv;
$("next-un").onclick = nextUnlabelled;
$("prev").onclick = () => { if (cur > 0) { cur--; render(); } };
$("next").onclick = () => { if (cur < ITEMS.length - 1) { cur++; render(); } };
$("comment").oninput = ev => { const id = ITEMS[cur].id; state[id] = Object.assign(state[id] || {}, { comment: ev.target.value }); save(); };
$("comment").addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === "Escape") { ev.preventDefault(); ev.target.blur(); } });
document.addEventListener("keydown", ev => {
  if (ev.target.tagName === "INPUT" || ev.metaKey || ev.ctrlKey || ev.altKey) return;
  const k = ev.key.toUpperCase();
  if ("AONU".includes(k) && k.length === 1) { ev.preventDefault(); setLabel(k); }
  else if (ev.key === "ArrowRight") { $("next").click(); }
  else if (ev.key === "ArrowLeft") { $("prev").click(); }
  else if (k === "J") { nextUnlabelled(); }
});
render();
</script>
</body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=DEF_DIR,
                    help="folder holding review_sheet.csv")
    args = ap.parse_args()
    d = args.dir.expanduser()
    sheet = d / "review_sheet.csv"
    if not sheet.exists():
        sys.exit(f"[fatal] {sheet} not found; run extension_p8_vaso_diagnostic_v2.py first")
    items = []
    with sheet.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows or "review_id" not in rows[0] or "excerpts" not in rows[0]:
        sys.exit(f"[fatal] {sheet} lacks review_id/excerpts columns")
    for r in rows:
        items.append({"id": r["review_id"].strip(), "excerpts": r["excerpts"] or ""})
    sig = json.dumps(f"{len(items)}_{items[0]['id']}_{items[-1]['id']}")
    html = (PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False).replace("</", "<\\/"))
                .replace("__SIG__", sig))
    out = d / "review_tool.html"
    out.write_text(html, encoding="utf-8")
    print(f"{len(items)} items -> {out}")
    print(f"open it with:  open {out}")


if __name__ == "__main__":
    main()
