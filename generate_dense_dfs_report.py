"""Generate an HTML report comparing Dense GE vs Dense GE + DFS reordering.

This is a SEPARATE experiment campaign from the three-way (dense/sparse/sparse+dfs)
comparison: it uses NO sparse GE at all, and isolates the effect of DFS reordering
on the original dense Gaussian-elimination solver.

Reads only the data files tagged "DENSEEXP_" (written by
peeling_cluster_decoder.py's experiment_dense_dfs_main), so it does not touch or
depend on the three-way campaign's files.

GE time shown here is ELIMINATION ONLY; the DFS reordering cost is tracked
separately (total_reorder_time) and excluded from the GE-time chart.

Run with:  python3 generate_dense_dfs_report.py
Output:    dense_dfs_comparison_report.html
"""

import ast
import glob
import json
import os

# (filename token, display label, color, content signature (ge_backend, ge_reorder)).
CONDITIONS = [
    ("dense",     "Dense GF(2) GE",          "#1f6feb", ("dense", None)),
    ("dense_dfs", "Dense GF(2) GE + DFS",    "#d1242f", ("dense", "dfs")),
]

CODES = [("Toric3", "18 qubits"), ("C_625", "625 qubits")]

# Only consider files from this experiment campaign.
FILE_GLOB = "{0}_*DENSEEXP*trials_*.txt"


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
    x_list = None
    trials = None
    for code, _ in CODES:
        data[code] = {}
        for cond, _, _, sig in CONDITIONS:
            rows = load_results_for(code, sig)
            if not rows:
                continue
            if x_list is None:
                x_list = [r["erasure_rate"] for r in rows]
            if trials is None:
                trials = rows[0].get("total_trials")
            data[code][cond] = {
                "fr": [r["failure_rate_peeling_M2_cluster"] for r in rows],
                "ge": [r["total_ge_time"] for r in rows],               # elimination only (seconds)
                "re": [r.get("total_reorder_time", 0.0) for r in rows],  # DFS reorder (seconds)
                "dt": [r["mean_decode_time"] for r in rows],            # seconds
            }
    return data, x_list, trials


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PPVH Decoder: Dense vs Dense+DFS comparison</title>
<style>
  :root { color-scheme: light; }
  body { font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; background: #f6f8fa; color: #1f2328; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 28px 24px 64px; }
  h1 { font-size: 24px; margin: 0 0 4px; }
  .sub { color: #57606a; margin: 0 0 18px; font-size: 14px; }
  .note { background: #fff8e6; border: 1px solid #f0d9a0; border-radius: 8px;
          padding: 12px 16px; font-size: 14px; margin: 0 0 24px; line-height: 1.5; }
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
  <h1>Dense GE vs Dense GE + DFS reorder</h1>
  <p class="sub">PPVH erasure decoder (arXiv:2208.01002). __TRIALS__ trials per erasure rate, fixed RNG seed so both conditions decode the <strong>identical</strong> erasure/error samples. No sparse GE is used in this experiment.</p>
  <div class="note">
    This isolates the effect of <strong>DFS reordering on the original dense solver</strong>. <strong>GE time is elimination only</strong> &mdash; the DFS reordering cost is measured separately, shown in its own chart and the summary table, and <strong>excluded</strong> from the GE-time chart.
    Times in milliseconds (GE/reorder are totals over all trials; decode time is the per-trial average).
  </div>
  <div id="sections"></div>
</div>

<script>
const X = __X__;
const TRIALS = __TRIALS__;
const DATA = __DATA__;

const CONDS = [
  ["dense", "Dense GF(2) GE", "#1f6feb"],
  ["dense_dfs", "Dense GF(2) GE + DFS", "#d1242f"]
];
const CODES = __CODES__;

function metricSeries(code, metric) {
  const scale = (metric === "fr") ? 1.0 : 1000.0; // times: seconds -> ms
  const out = [];
  for (const [cond, label, color] of CONDS) {
    if (!DATA[code] || !DATA[code][cond]) continue;
    out.push({ color, pts: X.map((x, i) => [x, DATA[code][cond][metric][i] * scale]) });
  }
  return out;
}

function drawChart(canvas, series) {
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
  const last = X.length - 1;
  const baseGe = DATA[code]["dense"] ? DATA[code]["dense"].ge[last] : null;
  let rows = "";
  for (const [cond, label, color] of CONDS) {
    if (!DATA[code] || !DATA[code][cond]) continue;
    const d = DATA[code][cond];
    const ge = d.ge[last];
    const speed = (ge > 0 && baseGe) ? " (" + (baseGe / ge).toFixed(2) + "x)" : "";
    rows += "<tr>" +
      "<td><span class='swatch' style='background:" + color + ";vertical-align:middle;margin-right:6px'></span>" + label + "</td>" +
      "<td>" + d.fr[last].toFixed(4) + "</td>" +
      "<td>" + (ge * 1000).toFixed(3) + speed + "</td>" +
      "<td>" + (d.re[last] * 1000).toFixed(3) + "</td>" +
      "<td>" + (d.dt[last] * 1000).toFixed(3) + "</td>" +
      "</tr>";
  }
  return "<table><thead><tr><th>Condition</th><th>Failure rate (full decoder)</th>" +
    "<th>GE time / elimination only (ms)</th><th>DFS reorder time (ms)</th><th>Mean decode time (ms)</th></tr></thead><tbody>" +
    rows + "</tbody></table>";
}

function legendHtml() {
  return "<div class='legend'>" + CONDS.map(([c, label, color]) =>
    "<span><i class='swatch' style='background:" + color + "'></i>" + label + "</span>").join("") + "</div>";
}

function render() {
  const container = document.getElementById("sections");
  container.innerHTML = "";
  for (const [code, qubits] of CODES) {
    if (!DATA[code] || Object.keys(DATA[code]).length === 0) continue;
    const sec = document.createElement("section");
    sec.innerHTML =
      "<h2>" + code + " <span class='muted'>(" + qubits + ")</span></h2>" +
      legendHtml() +
      "<div class='charts'>" +
        "<div class='chartbox'><h4>Failure rate vs erasure rate</h4><canvas data-code='" + code + "' data-metric='fr'></canvas></div>" +
        "<div class='chartbox'><h4>GE time / elimination only (ms)</h4><canvas data-code='" + code + "' data-metric='ge'></canvas></div>" +
        "<div class='chartbox'><h4>DFS reorder time (ms)</h4><canvas data-code='" + code + "' data-metric='re'></canvas></div>" +
        "<div class='chartbox'><h4>Mean decode time (ms)</h4><canvas data-code='" + code + "' data-metric='dt'></canvas></div>" +
      "</div>" +
      "<h3>At the maximum erasure rate (" + X[X.length - 1].toFixed(2) + ")</h3>" +
      summaryTable(code);
    container.appendChild(sec);
  }
  document.querySelectorAll("canvas[data-code]").forEach(cv => drawChart(cv, metricSeries(cv.dataset.code, cv.dataset.metric)));
}

window.addEventListener("load", render);
window.addEventListener("resize", render);
</script>
</body>
</html>
"""


def main():
    data, x_list, trials = build_data()
    if x_list is None:
        print("No DENSEEXP result files found. Run the experiment first:")
        print("  python3 peeling_cluster_decoder.py densedfs")
        return

    html = HTML_TEMPLATE
    html = html.replace("__X__", json.dumps(x_list))
    html = html.replace("__DATA__", json.dumps(data))
    html = html.replace("__CODES__", json.dumps(CODES))
    html = html.replace("__TRIALS__", str(trials))

    out_path = "dense_dfs_comparison_report.html"
    with open(out_path, "w") as f:
        f.write(html)
    print("Wrote", out_path, "(" + str(os.path.getsize(out_path)) + " bytes)")


if __name__ == "__main__":
    main()
