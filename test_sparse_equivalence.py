"""Seeded equivalence check: dense GF(2) GE vs custom sparse GF(2) GE.

With the natural (non-reordered) column order, the sparse backend computes the
same reduced row echelon form as the dense backend, so for identical random
erasure/error samples it MUST produce identical failure / success counts. This
script asserts that equivalence on Toric3 and the n=625 PEG code, and then
reports (without asserting) how the sparse+DFS reordering differs, since that
condition is expected to change which minimum-weight solution is chosen.

Run with:  python3 test_sparse_equivalence.py
"""

import sys

from Hypergraph_Product_Code_Construction_v3 import Toric3
from utilities import set_solver_config, construct_HGP_code_from_classical_H_text_file
from peeling_cluster_decoder import run_combined_peeling_cluster_decoder_varying_erasure_rate


# The outcome counters that must match exactly between dense and sparse backends.
COUNT_KEYS = [
    'num_true_decoder_failures',
    'num_non_trivial_logical_errors',
    'num_total_failures',
    'num_total_successes_peeling_M0',
    'num_total_successes_peeling_M1',
    'num_total_successes_peeling_M2',
    'num_total_successes_peeling_M2_cluster',
]


def run_condition(code, backend, reorder, max_rate, steps, trials, seed):
    set_solver_config(backend=backend, reorder=reorder)
    return run_combined_peeling_cluster_decoder_varying_erasure_rate(
        code, max_rate, steps, trials, seed=seed)


def compare_counts(name, list_a, list_b):
    """Return True if every count key matches across every erasure-rate step."""
    ok = True
    for step_index, (da, db) in enumerate(zip(list_a, list_b)):
        for key in COUNT_KEYS:
            if (da[key] != db[key]):
                ok = False
                print("  [FAIL] {0}: step {1} (erasure_rate={2:.4f}) key '{3}': {4} != {5}".format(
                    name, step_index, da['erasure_rate'], key, da[key], db[key]))
    if ok:
        print("  [PASS] {0}: all outcome counts identical across all erasure rates".format(name))
    return ok


def report_dfs_differences(name, list_sparse, list_dfs):
    """Report (without asserting) failure-rate and timing deltas for sparse+DFS."""
    print("  {0} (sparse vs sparse+DFS):".format(name))
    for step_index, (ds, dd) in enumerate(zip(list_sparse, list_dfs)):
        fr_s = ds['failure_rate_peeling_M2_cluster']
        fr_d = dd['failure_rate_peeling_M2_cluster']
        t_s = ds['total_ge_time']
        t_d = dd['total_ge_time']
        if (fr_s != fr_d) or (step_index == len(list_sparse) - 1):
            print("    erasure_rate={0:.4f}  failure_rate: {1:.4f} -> {2:.4f}  total_ge_time: {3:.4e}s -> {4:.4e}s".format(
                ds['erasure_rate'], fr_s, fr_d, t_s, t_d))


def main():
    # Modest configuration so the equivalence check finishes quickly.
    max_rate = 0.32
    steps = 8
    trials = 300
    seed = 2208

    named_codes = [
        ("Toric3", Toric3),
        ("C_625", construct_HGP_code_from_classical_H_text_file(
            'PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt')),
    ]

    print("=" * 78)
    print("Dense vs Sparse equivalence (seed={0}, steps={1}, trials={2})".format(seed, steps, trials))
    print("=" * 78)

    all_ok = True
    for code_name, code in named_codes:
        print("Code:", code_name)
        dense = run_condition(code, "dense", None, max_rate, steps, trials, seed)
        sparse = run_condition(code, "sparse", None, max_rate, steps, trials, seed)
        all_ok &= compare_counts(code_name, dense, sparse)

        dfs = run_condition(code, "sparse", "dfs", max_rate, steps, trials, seed)
        report_dfs_differences(code_name, sparse, dfs)
        print()

    print("=" * 78)
    if all_ok:
        print("EQUIVALENCE PASSED: dense and sparse backends agree on all outcomes.")
        print("=" * 78)
        return 0
    else:
        print("EQUIVALENCE FAILED: dense and sparse backends disagree (see above).")
        print("=" * 78)
        return 1


if __name__ == "__main__":
    sys.exit(main())
