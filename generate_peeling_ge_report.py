"""Generate an HTML report for the PEELING->GE high-erasure experiment (0.30 -> 0.45).

This is the experiment that actually tests the proposal:

    peeling (M=0 dangling checks only)  ->  [optional DFS reorder]  ->  ONE Sparse GE
    on the whole residual graph (Hz restricted to the residual erased columns).

There is no M=1, no M=2, and no cluster-tree decomposition -- the decoder jumps straight
from peeling to a single Sparse Gaussian elimination. The only extra step between peeling
and GE is the DFS column reordering. Two conditions are compared on identical samples:

    sparse      : peeling -> plain Sparse GE          (natural column order)
    sparse_dfs  : peeling -> DFS reorder -> Sparse GE  (the proposal)

Reads only the data files tagged "PEELGEHIGH_" (written by peeling_cluster_decoder.py's
experiment_peeling_ge_high_erasure_main), so it never touches the older cascade campaign's
files or its report (sparse_high_erasure_report.html is left untouched).

Run with:  python generate_peeling_ge_report.py
Output:    peeling_ge_high_erasure_report.html
"""

import ast
import glob
import json
import os

# (condition_key, label, color, (ge_backend, ge_reorder)) -- the signature matches the data files.
CONDITIONS = [
    ("sparse",     "Peeling -> Sparse GE",            "#2da44e", ("sparse", None)),
    ("sparse_dfs", "Peeling -> DFS -> Sparse GE",     "#d1242f", ("sparse", "dfs")),
]

CODES = [("Toric3", "18 qubits"), ("C_625", "625 qubits")]

FILE_GLOB = "{0}_*PEELGEHIGH*trials_*.txt"


def load_results_for(code, signature):
    want_backend, want_reorder = signature
    best_rows = None
    best_mtime = -1.0
    for path in glob.glob(FILE_GLOB.format(code)):
        try:
            with open(path, "r") as f:
                rows = [d for d in ast.literal_eval(f.read()) if isinstance(d, dict)]
        except Exception:
            continue
        if not rows:
            continue
        first = rows[0]
        if (first.get("ge_backend") == want_backend) and (first.get("ge_reorder") == want_reorder):
            mtime = os.path.getmtime(path)
            if mtime > best_mtime:
                best_mtime = mtime
                best_rows = rows
    return best_rows


def build_data():
    data = {}
    x_by_code = {}
    trials = None
    for code, _ in CODES:
        data[code] = {}
        for cond, _, _, sig in CONDITIONS:
            rows = load_results_for(code, sig)
            if not rows:
                continue
            x_by_code[code] = [r["erasure_rate"] for r in rows]
            if trials is None:
                trials = rows[0].get("total_trials")
            data[code][cond] = {
                "fr": [r["failure_rate"] for r in rows],                  # overall failure rate
                "ge": [r["total_ge_time"] for r in rows],                 # elimination only (s)
                "re": [r.get("total_reorder_time", 0.0) for r in rows],   # DFS reorder (s)
                "tot": [r["total_ge_time"] + r.get("total_reorder_time", 0.0) for r in rows],  # GE+reorder (s)
                "dt": [r["mean_decode_time"] for r in rows],              # per-trial decode (s)
            }
    return data, x_by_code, trials


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PPVH: Peeling -> Sparse GE vs Peeling -> DFS -> Sparse GE (0.30-0.45)</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f6f8fa; color: #1f2328; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 24px 64px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  .sub { color: #57606a; margin: 0 0 18px; font-size: 14px; }
  .note { background: #fff8e6; border: 1px solid #f0d9a0; border-radius: 8px;
          padding: 12px 16px; font-size: 14px; margin: 0 0 18px; line-height: 1.5; }
  .pipe { background: #eef4ff; border: 1px solid #c5d7f5; border-radius: 8px;
          padding: 12px 16px; font-size: 14px; margin: 0 0 18px; line-height: 1.55; }
  .pipe code { background: #dde8fb; padding: 1px 5px; border-radius: 4px; }
  .verdict { background: #ffeef0; border: 1px solid #f3c2c8; border-radius: 8px;
             padding: 12px 16px; font-size: 14px; margin: 0 0 24px; line-height: 1.55; }
  section { background: #fff; border: 1px solid #d0d7de; border-radius: 10px;
            padding: 18px 20px; margin: 0 0 22px; }
  h2 { font-size: 19px; margin: 0 0 6px; }
  h3 { font-size: 15px; margin: 16px 0 8px; color: #424a53; }
  .muted { color: #8b949e; font-weight: 400; font-size: 14px; }
  .charts { display: flex; flex-wrap: wrap; gap: 18px; }
  .chartbox { flex: 1 1 320px; min-width: 300px; }
  .chartbox h4 { font-size: 13.5px; margin: 0 0 6px; text-align: center; color: #1f2328; }
  canvas { width: 100%; height: 280px; display: block; }
  .legend { display: flex; gap: 16px; justify-content: center; margin: 6px 0 0; font-size: 12.5px; flex-wrap: wrap; }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .swatch { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
  table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin-top: 6px; }
  th, td { border: 1px solid #d0d7de; padding: 7px 10px; text-align: left; }
  th { background: #f6f8fa; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Peeling &rarr; Sparse GE &nbsp;vs&nbsp; Peeling &rarr; DFS &rarr; Sparse GE</h1>
  <p class="sub">PPVH erasure decoder (arXiv:2208.01002) &mdash; high-erasure regime (0.30 &rarr; 0.45). __TRIALS__ trials per rate, fixed RNG seed so both conditions decode the <strong>identical</strong> samples.</p>

  <div class="pipe">
    <strong>Pipeline under test (the proposal).</strong> Both conditions run the <em>minimal</em> decoder: plain peeling
    (M=0 dangling checks only) to shrink the system for free, then the entire residual graph is handed to
    <strong>one</strong> GF(2) Gaussian elimination over <code>Hz</code>. We deliberately skip everything in between
    &mdash; no M=1 erased generators, no M=2 generator products, no cluster-tree decomposition. The <em>only</em> extra
    step before GE is the DFS column reordering:
    <br/>&nbsp;&nbsp;&bull; <strong>Peeling &rarr; Sparse GE</strong>: residual solved in natural column order.
    <br/>&nbsp;&nbsp;&bull; <strong>Peeling &rarr; DFS &rarr; Sparse GE</strong>: residual columns reordered by DFS, then the same Sparse GE.
  </div>

  <div class="note">
    The hypothesis is that DFS reordering makes the Sparse GE solve <strong>faster</strong>. To test this fairly,
    <strong>GE time is elimination only</strong> (reordering excluded) and the DFS reordering cost is measured and shown
    <strong>separately</strong>. The "GE + reorder (total)" chart is the practically relevant comparison, since the
    reordering must be paid for on every solve. Times are in milliseconds (GE / reorder / total are summed over all
    trials; decode time is the per-trial average). This is a separate campaign &mdash; the earlier report
    (<code>sparse_high_erasure_report.html</code>) and its data are left untouched.
  </div>

  <div class="verdict" id="verdict"></div>

  <div id="sections"></div>
</div>

<script>
const X_BY_CODE = __XBYCODE__;
const TRIALS = __TRIALS__;
const DATA = __DATA__;

const CONDS = [
  ["sparse", "Peeling -> Sparse GE", "#2da44e"],
  ["sparse_dfs", "Peeling -> DFS -> Sparse GE", "#d1242f"]
];
const CODES = __CODES__;

const METRICS = [
  ["fr",  "Failure rate vs erasure rate"],
  ["ge",  "GE time / elimination only (ms)"],
  ["re",  "DFS reorder time (ms)"],
  ["tot", "GE + reorder (total) (ms)"],
  ["dt",  "Mean decode time (ms)"]
];

function metricSeries(code, metric) {
  const X = X_BY_CODE[code];
  const scale = (metric === "fr") ? 1.0 : 1000.0; // times: seconds -> ms
  const out = [];
  for (const [cond, label, color] of CONDS) {
    if (!DATA[code] || !DATA[code][cond]) continue;
    out.push({ color, pts: X.map((x, i) => [x, DATA[code][cond][metric][i] * scale]) });
  }
  return out;
}

function drawChart(canvas, code, series) {
  const X = X_BY_CODE[code];
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth, cssH = canvas.clientHeight;
  canvas.width = cssW * dpr; canvas.height = cssH * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssW, cssH);
  const mL = 54, mR = 12, mT = 12, mB = 34;
  const pw = cssW - mL - mR, ph = cssH - mT - mB;
  let yMax = 0;
  series.forEach(s => s.pts.forEach(p => { if (p[1] > yMax) yMax = p[1]; }));
  if (yMax <= 0) yMax = 1;
  yMax *= 1.12;
  const xMin = X[0], xMax = X[X.length - 1];
  const sx = x => mL + (x - xMin) / (xMax - xMin) * pw;
  const sy = y => mT + ph - (y / yMax) * ph;
  ctx.strokeStyle = "#d0d7de"; ctx.lineWidth = 1; ctx.strokeRect(mL, mT, pw, ph);
  ctx.font = "10px -apple-system, Segoe UI, sans-serif";
  const yticks = 5;
  for (let i = 0; i <= yticks; i++) {
    const yv = yMax * i / yticks, py = sy(yv);
    ctx.strokeStyle = "#eef1f4"; ctx.beginPath(); ctx.moveTo(mL, py); ctx.lineTo(mL + pw, py); ctx.stroke();
    ctx.fillStyle = "#57606a"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText(yv < 1 ? yv.toPrecision(2) : yv.toPrecision(3), mL - 5, py);
  }
  ctx.textAlign = "center"; ctx.textBaseline = "top";
  X.forEach((xv, i) => { if (i % 2 === 0) { ctx.fillStyle = "#57606a"; ctx.fillText(xv.toFixed(2), sx(xv), mT + ph + 6); } });
  ctx.fillStyle = "#424a53"; ctx.fillText("Erasure rate", mL + pw / 2, mT + ph + 19);
  series.forEach(s => {
    ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.beginPath();
    s.pts.forEach((p, i) => { const px = sx(p[0]), py = sy(p[1]); if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py); });
    ctx.stroke();
    ctx.fillStyle = s.color;
    s.pts.forEach(p => { ctx.beginPath(); ctx.arc(sx(p[0]), sy(p[1]), 2.4, 0, Math.PI * 2); ctx.fill(); });
  });
}

function summaryTable(code) {
  const X = X_BY_CODE[code];
  const last = X.length - 1;
  const baseGe = DATA[code]["sparse"] ? DATA[code]["sparse"].ge[last] : null;
  const baseTot = DATA[code]["sparse"] ? DATA[code]["sparse"].tot[last] : null;
  let rows = "";
  for (const [cond, label, color] of CONDS) {
    if (!DATA[code] || !DATA[code][cond]) continue;
    const d = DATA[code][cond];
    const ge = d.ge[last], tot = d.tot[last];
    const geSpeed = (ge > 0 && baseGe) ? " (" + (baseGe / ge).toFixed(2) + "x)" : "";
    const totSpeed = (tot > 0 && baseTot) ? " (" + (baseTot / tot).toFixed(2) + "x)" : "";
    rows += "<tr>" +
      "<td><span class='swatch' style='background:" + color + ";vertical-align:middle;margin-right:6px'></span>" + label + "</td>" +
      "<td>" + X[last].toFixed(2) + "</td>" +
      "<td>" + d.fr[last].toFixed(4) + "</td>" +
      "<td>" + (ge * 1000).toFixed(3) + geSpeed + "</td>" +
      "<td>" + (d.re[last] * 1000).toFixed(3) + "</td>" +
      "<td>" + (tot * 1000).toFixed(3) + totSpeed + "</td>" +
      "<td>" + (d.dt[last] * 1000).toFixed(3) + "</td>" +
      "</tr>";
  }
  return "<table><thead><tr><th>Condition</th><th>Erasure rate</th><th>Failure rate</th>" +
    "<th>GE / elimination only (ms)</th><th>DFS reorder (ms)</th><th>GE + reorder total (ms)</th><th>Mean decode (ms)</th></tr></thead><tbody>" +
    rows + "</tbody></table>";
}

function legendHtml() {
  return "<div class='legend'>" + CONDS.map(([c, label, color]) =>
    "<span><i class='swatch' style='background:" + color + "'></i>" + label + "</span>").join("") + "</div>";
}

// Aggregate verdict: compare GE-only and GE+reorder totals summed across all rates and codes.
function buildVerdict() {
  let geBase = 0, geDfs = 0, totBase = 0, totDfs = 0, have = false;
  for (const [code] of CODES) {
    if (!DATA[code] || !DATA[code]["sparse"] || !DATA[code]["sparse_dfs"]) continue;
    have = true;
    const b = DATA[code]["sparse"], d = DATA[code]["sparse_dfs"];
    for (let i = 0; i < b.ge.length; i++) {
      geBase += b.ge[i]; geDfs += d.ge[i];
      totBase += b.tot[i]; totDfs += d.tot[i];
    }
  }
  if (!have) return "No PEELGEHIGH data files found. Run: <code>python peeling_cluster_decoder.py peelge</code>";
  const geRatio = geDfs > 0 ? (geBase / geDfs) : 0;     // >1 means DFS elimination faster
  const totRatio = totDfs > 0 ? (totBase / totDfs) : 0; // >1 means DFS total faster
  const elimVerb = geRatio >= 1.0 ? ("about " + geRatio.toFixed(2) + "x faster")
                                  : ("about " + (1 / geRatio).toFixed(2) + "x slower");
  const totVerb = totRatio >= 1.0 ? ("about " + totRatio.toFixed(2) + "x faster")
                                  : ("about " + (1 / totRatio).toFixed(2) + "x slower");
  const supported = totRatio > 1.0;
  return "<strong>Verdict (summed over all rates and codes).</strong> DFS reordering makes the <em>elimination</em> " +
    elimVerb + ", but once the reordering cost is included the DFS condition is " + totVerb +
    " end-to-end. The hypothesis that DFS reordering speeds up the sparse-GE residual solve is therefore <strong>" +
    (supported ? "supported" : "not supported") + "</strong> in this regime: any elimination saving is outweighed by " +
    "the cost of computing the ordering.";
}

function render() {
  document.getElementById("verdict").innerHTML = buildVerdict();
  const container = document.getElementById("sections");
  container.innerHTML = "";
  for (const [code, qubits] of CODES) {
    if (!DATA[code] || Object.keys(DATA[code]).length === 0) continue;
    const X = X_BY_CODE[code];
    const sec = document.createElement("section");
    let chartHtml = "";
    for (const [m, title] of METRICS) {
      chartHtml += "<div class='chartbox'><h4>" + title + "</h4><canvas data-code='" + code + "' data-metric='" + m + "'></canvas></div>";
    }
    sec.innerHTML =
      "<h2>" + code + " <span class='muted'>(" + qubits + ")</span></h2>" +
      legendHtml() +
      "<div class='charts'>" + chartHtml + "</div>" +
      "<h3>At the maximum erasure rate (" + X[X.length - 1].toFixed(2) + ")</h3>" +
      summaryTable(code);
    container.appendChild(sec);
  }
  document.querySelectorAll("canvas[data-code]").forEach(cv => drawChart(cv, cv.dataset.code, metricSeries(cv.dataset.code, cv.dataset.metric)));
}

window.addEventListener("load", render);
window.addEventListener("resize", render);
</script>
</body>
</html>
"""


def main():
    data, x_by_code, trials = build_data()
    if not x_by_code:
        print("No PEELGEHIGH result files found. Run the experiment first:")
        print("  python peeling_cluster_decoder.py peelge")
        return

    html = HTML_TEMPLATE
    html = html.replace("__XBYCODE__", json.dumps(x_by_code))
    html = html.replace("__DATA__", json.dumps(data))
    html = html.replace("__CODES__", json.dumps(CODES))
    html = html.replace("__TRIALS__", str(trials))

    out_path = "peeling_ge_high_erasure_report.html"
    with open(out_path, "w") as f:
        f.write(html)
    print("Wrote", out_path, "(" + str(os.path.getsize(out_path)) + " bytes)")


if __name__ == "__main__":
    main()
