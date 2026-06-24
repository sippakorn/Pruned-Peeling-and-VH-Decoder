"""Generate an HTML report for the PEELING -> GE pipeline (high-erasure 0.30 -> 0.45).

This compares the minimal pipeline -- plain M=0 peeling, then a SINGLE global GF(2)
Gaussian elimination over the residual Hz system (no M=1/M=2, no cluster tree) -- in two
flavors against the full cascade decoder as a baseline:

  1. Peeling -> Sparse GE              (PEELGEHIGH_ files, backend sparse, reorder None)
  2. Peeling -> DFS -> Sparse GE       (PEELGEHIGH_ files, backend sparse, reorder dfs)
  3. Peeling -> VH -> Cluster (Sparse GE)  BASELINE
                                       (HIGHERASURE_ files, backend sparse, reorder None)

The baseline is the original paper decoder (full cascade: M=0 peeling -> M=1/M=2 VH pruned
peeling -> cluster decoder) with the ONLY change being dense GE replaced by sparse GF(2) GE.
It does NOT use DFS reordering -- DFS is a feature of the proposal being evaluated, not of
the baseline.

Apple-to-apple justification for the cascade baseline:
experiment_peeling_ge_high_erasure_main now produces BOTH the PEELGEHIGH_ files (the two
peeling->GE conditions) and the HIGHERASURE_ baseline file (cascade, sparse GE, no DFS) in a
single run, using seed=12345 with the same per-step seeding (step_seed = seed + step_index),
the same code (C_1600), the same erasure sweep (0.29 -> 0.45, 16 steps -> rates 0.30..0.45),
and the same trial count. Neither decoder consumes the random / np.random streams during
decoding (those are only used by sample generation and code construction), so all conditions
decode the IDENTICAL erasure/error samples per rate.

Failure-rate keys differ by pipeline: the peeling->GE result dicts store 'failure_rate';
the cascade result dicts store 'failure_rate_peeling_M2_cluster'. GE time is elimination only
(DFS reordering is tracked separately in 'total_reorder_time').

Run with:  python3 generate_peeling_ge_report.py
Output:    peeling_ge_high_erasure_report.html
"""

import ast
import glob
import json
import os

# Each condition: (key, label, color, file_tag, (ge_backend, ge_reorder), failure_rate_key)
CONDITIONS = [
    ("peel_sparse",     "Peeling \u2192 Sparse GE",
     "#2da44e", "PEELGEHIGH", ("sparse", None),  "failure_rate"),
    ("peel_sparse_dfs", "Peeling \u2192 DFS \u2192 Sparse GE",
     "#d1242f", "PEELGEHIGH", ("sparse", "dfs"), "failure_rate"),
    ("cascade_sparse", "Peeling \u2192 VH \u2192 Cluster (Sparse GE) [baseline]",
     "#0969da", "HIGHERASURE", ("sparse", None), "failure_rate_peeling_M2_cluster"),
]

CODES = [("C_1600", "1600 qubits")]


def load_results_for(code, file_tag, signature):
    """Return the rows from the most recent file for this code/tag whose stored
    (ge_backend, ge_reorder) matches the requested signature."""
    want_backend, want_reorder = signature
    file_glob = "{0}_*{1}*trials_*.txt".format(code, file_tag)
    best_rows = None
    best_mtime = -1.0
    for path in glob.glob(file_glob):
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
        for cond, _label, _color, file_tag, sig, fr_key in CONDITIONS:
            rows = load_results_for(code, file_tag, sig)
            if not rows:
                continue
            # Use the erasure rates from the first available condition for this code.
            if code not in x_by_code:
                x_by_code[code] = [r["erasure_rate"] for r in rows]
            if trials is None:
                trials = rows[0].get("total_trials")
            data[code][cond] = {
                "fr": [r.get(fr_key) for r in rows],
                "ge": [r["total_ge_time"] for r in rows],                # elimination only (seconds)
                "re": [r.get("total_reorder_time", 0.0) for r in rows],  # DFS reorder (seconds)
                "dt": [r["mean_decode_time"] for r in rows],            # seconds
            }
    return data, x_by_code, trials


# CONDS for the embedded JS: [key, label, color]
def conds_for_js():
    return [[c[0], c[1], c[2]] for c in CONDITIONS]


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>PPVH Decoder: Peeling -> GE pipeline (high-erasure 0.30-0.45)</title>
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
  .missing { color: #8b949e; font-style: italic; font-size: 13.5px; margin-top: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Peeling &rarr; GE pipeline &mdash; high-erasure regime (0.30 &rarr; 0.45)</h1>
  <p class="sub">PPVH erasure decoder (arXiv:2208.01002). __TRIALS__ trials per erasure rate, fixed RNG seed so all conditions decode the <strong>identical</strong> samples.</p>
  <div class="note">
    <strong>Pipeline under test:</strong> plain M=0 peeling, then a <strong>single global GF(2) Gaussian elimination over the residual Hz system</strong> (all remaining erased qubits at once) &mdash; no M=1, no M=2, no cluster-tree decomposition. The only optional step between peeling and GE is the DFS column reordering.<br/>
    <strong>Baseline:</strong> the original paper decoder (full cascade &mdash; Peeling &rarr; VH &rarr; Cluster) with the <strong>only</strong> change being dense GE replaced by sparse GF(2) GE. The baseline uses <strong>no DFS</strong>.<br/>
    <strong>GE time is elimination only</strong>; DFS reordering is measured separately and excluded from the GE-time chart. Times in milliseconds (GE/reorder are totals over all trials; decode time is the per-trial average).<br/>
    The baseline reuses the seed-matched <code>HIGHERASURE_</code> cascade data (identical samples; see the generator docstring).
  </div>
  <div id="sections"></div>
</div>

<script>
const X_BY_CODE = __XBYCODE__;
const TRIALS = __TRIALS__;
const DATA = __DATA__;

const CONDS = __CONDS__;
const CODES = __CODES__;

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
  // Speedup baseline is the plain Peeling -> Sparse GE GE time.
  const baseGe = (DATA[code]["peel_sparse"]) ? DATA[code]["peel_sparse"].ge[last] : null;
  let rows = "";
  for (const [cond, label, color] of CONDS) {
    if (!DATA[code] || !DATA[code][cond]) continue;
    const d = DATA[code][cond];
    const ge = d.ge[last];
    const speed = (ge > 0 && baseGe) ? " (" + (baseGe / ge).toFixed(2) + "x)" : "";
    rows += "<tr>" +
      "<td><span class='swatch' style='background:" + color + ";vertical-align:middle;margin-right:6px'></span>" + label + "</td>" +
      "<td>" + X[last].toFixed(2) + "</td>" +
      "<td>" + (d.fr[last] != null ? d.fr[last].toFixed(4) : "&mdash;") + "</td>" +
      "<td>" + (ge * 1000).toFixed(3) + speed + "</td>" +
      "<td>" + (d.re[last] * 1000).toFixed(3) + "</td>" +
      "<td>" + (d.dt[last] * 1000).toFixed(3) + "</td>" +
      "</tr>";
  }
  return "<table><thead><tr><th>Condition</th><th>Erasure rate</th><th>Failure rate</th>" +
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
    const X = X_BY_CODE[code];
    const present = CONDS.filter(([c]) => DATA[code][c]).map(([c]) => c);
    const missing = CONDS.filter(([c]) => !DATA[code][c]);
    let missingHtml = "";
    if (missing.length > 0) {
      missingHtml = "<p class='missing'>Missing data for: " + missing.map(([c, label]) => label).join(", ") + "</p>";
    }
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
      missingHtml +
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


def validate_html(html_text):
    """Lightweight well-formedness check (mirrors the AGENTS.md validation step)."""
    import html.parser as hp
    hp.HTMLParser().feed(html_text)


def main():
    data, x_by_code, trials = build_data()
    if not x_by_code:
        print("No result files found for the PEELING->GE report.")
        print("Generate the data first:")
        print("  python3 peeling_cluster_decoder.py peelge")
        print("(The cascade baseline reuses the HIGHERASURE_ sparse_dfs files; run")
        print("  python3 peeling_cluster_decoder.py higherasure   if those are missing too.)")
        return

    html = HTML_TEMPLATE
    html = html.replace("__XBYCODE__", json.dumps(x_by_code))
    html = html.replace("__DATA__", json.dumps(data))
    html = html.replace("__CONDS__", json.dumps(conds_for_js()))
    html = html.replace("__CODES__", json.dumps(CODES))
    html = html.replace("__TRIALS__", str(trials))

    out_path = "peeling_ge_high_erasure_report.html"
    with open(out_path, "w") as f:
        f.write(html)

    # Validate the emitted HTML so a malformed report is caught immediately.
    try:
        validate_html(html)
        print("HTML validation: OK")
    except Exception as exc:
        print("HTML validation FAILED:", exc)

    # Report which series actually made it into the file.
    for code, _ in CODES:
        present = sorted(data.get(code, {}).keys())
        print("  " + code + ": " + (", ".join(present) if present else "(no data)"))

    print("Wrote", out_path, "(" + str(os.path.getsize(out_path)) + " bytes)")


if __name__ == "__main__":
    main()
