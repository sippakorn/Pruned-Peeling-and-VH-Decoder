"""Benchmark comparing execution time of
perform_classical_syndrome_analysis_with_erasure (v1: dense standard_form)
vs perform_classical_syndrome_analysis_with_erasure_v2 (v2: sparse_form).

Runs several scenarios covering small / medium / larger random H matrices,
plus one real PEG HGP code if available. Prints a summary table.
"""

import os
import time
import numpy as np

from utilities import (
    perform_classical_syndrome_analysis_with_erasure,
    perform_classical_syndrome_analysis_with_erasure_v2,
    construct_HGP_code_from_classical_H_text_file,
)


SEED = 20260421


def _gen_random_H(m, n, density, rng):
    """Random binary matrix with given density of 1s."""
    H = (rng.random(size=(m, n)) < density).astype(int)
    return H


def _gen_decodable_case(H, erasure_fraction, rng):
    """Pick random erased qubits and a random error supported on them;
    return (E_index_set, e, s) where s = H @ e % 2."""
    m, n = H.shape
    k = max(1, int(erasure_fraction * n))
    E_idx = rng.choice(n, size=k, replace=False)
    E_index_set = set(int(x) for x in E_idx)
    e = np.zeros(n, dtype=int)
    for idx in E_index_set:
        e[idx] = int(rng.integers(0, 2))
    s = (H @ e) % 2
    return E_index_set, e, s


def _time_version(fn, H, trials_data):
    """Run fn on every (s, E) trial, return (total_seconds, all_results)."""
    results = []
    t0 = time.perf_counter()
    for s, E_index_set in trials_data:
        results.append(fn(H, s, E_index_set))
    t1 = time.perf_counter()
    return t1 - t0, results


def _run_scenario(label, H, n_trials, erasure_fraction, rng):
    """Generate n_trials decodable cases, time v1 and v2, return a result dict."""
    trials = []
    for _ in range(n_trials):
        E_index_set, _e, s = _gen_decodable_case(H, erasure_fraction, rng)
        trials.append((s, E_index_set))

    # Warm-up call (JIT-like caches, import overhead, first-call allocations)
    if trials:
        s0, E0 = trials[0]
        perform_classical_syndrome_analysis_with_erasure(H, s0, E0)
        perform_classical_syndrome_analysis_with_erasure_v2(H, s0, E0)

    t_v1, res_v1 = _time_version(
        perform_classical_syndrome_analysis_with_erasure, H, trials
    )
    t_v2, res_v2 = _time_version(
        perform_classical_syndrome_analysis_with_erasure_v2, H, trials
    )

    all_match = all(np.array_equal(a, b) for a, b in zip(res_v1, res_v2))

    return {
        "label": label,
        "shape": f"{H.shape[0]}x{H.shape[1]}",
        "density": f"{H.sum() / H.size:.3f}",
        "trials": n_trials,
        "erasure_frac": erasure_fraction,
        "v1_total_s": t_v1,
        "v2_total_s": t_v2,
        "v1_per_call_ms": 1000 * t_v1 / n_trials,
        "v2_per_call_ms": 1000 * t_v2 / n_trials,
        "speedup": (t_v1 / t_v2) if t_v2 > 0 else float("inf"),
        "match": all_match,
    }


def _print_summary(rows):
    headers = [
        "Scenario",
        "Shape",
        "Density",
        "Trials",
        "Eras%",
        "v1 total (s)",
        "v2 total (s)",
        "v1/call (ms)",
        "v2/call (ms)",
        "Speedup v1/v2",
        "Match",
    ]
    widths = [max(len(h), 12) for h in headers]

    def _fmt(row):
        return [
            row["label"],
            row["shape"],
            row["density"],
            str(row["trials"]),
            f"{row['erasure_frac']:.2f}",
            f"{row['v1_total_s']:.4f}",
            f"{row['v2_total_s']:.4f}",
            f"{row['v1_per_call_ms']:.3f}",
            f"{row['v2_per_call_ms']:.3f}",
            f"{row['speedup']:.2f}x",
            "OK" if row["match"] else "MISMATCH",
        ]

    formatted = [_fmt(r) for r in rows]
    for col_idx, h in enumerate(headers):
        widths[col_idx] = max(widths[col_idx], len(h))
        for row in formatted:
            widths[col_idx] = max(widths[col_idx], len(row[col_idx]))

    def _render(cells):
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    bar = "-+-".join("-" * w for w in widths)
    print()
    print("Execution time comparison: v1 (dense standard_form) vs v2 (sparse sparse_form)")
    print(bar)
    print(_render(headers))
    print(bar)
    for row in formatted:
        print(_render(row))
    print(bar)


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    # Scenario A: tiny handcrafted-size dense problem — expected v1 faster
    # due to numpy vectorization winning over Python set overhead.
    H_small = _gen_random_H(10, 16, density=0.35, rng=rng)
    rows.append(_run_scenario("small-dense", H_small, n_trials=200, erasure_fraction=0.5, rng=rng))

    # Scenario B: medium sparse random H.
    H_med_sparse = _gen_random_H(60, 120, density=0.05, rng=rng)
    rows.append(_run_scenario("medium-sparse-5%", H_med_sparse, n_trials=80, erasure_fraction=0.3, rng=rng))

    # Scenario C: medium dense.
    H_med_dense = _gen_random_H(60, 120, density=0.5, rng=rng)
    rows.append(_run_scenario("medium-dense-50%", H_med_dense, n_trials=80, erasure_fraction=0.3, rng=rng))

    # Scenario D: larger sparse.
    H_large_sparse = _gen_random_H(200, 400, density=0.02, rng=rng)
    rows.append(_run_scenario("large-sparse-2%", H_large_sparse, n_trials=30, erasure_fraction=0.3, rng=rng))

    # Scenario E: larger dense.
    H_large_dense = _gen_random_H(200, 400, density=0.5, rng=rng)
    rows.append(_run_scenario("large-dense-50%", H_large_dense, n_trials=30, erasure_fraction=0.3, rng=rng))

    # Scenario F: real PEG HGP code Hx (if file present). This is the kind of
    # sparse parity-check matrix the solvers are actually used on in practice.
    peg_file = "PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt"
    if os.path.exists(peg_file):
        try:
            code = construct_HGP_code_from_classical_H_text_file(peg_file)
            H_peg = code.Hx.astype(int)
            rows.append(_run_scenario(
                "PEG-HGP n=625 Hx", H_peg, n_trials=10, erasure_fraction=0.2, rng=rng
            ))
        except Exception as exc:
            print(f"[warn] Skipped PEG HGP scenario: {exc}")

    _print_summary(rows)


if __name__ == "__main__":
    main()
