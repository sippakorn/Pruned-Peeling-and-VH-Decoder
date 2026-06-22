"""Setup verification / regression baseline for the PPVH erasure decoder.

Reproduces the concrete facts recorded in AGENTS.md by running the actual code.
Run with:  python3 verify_setup.py

This script is intentionally side-effect free (writes no data files) and exits
non-zero if any check fails, so it can be used as a regression gate before and
after the sparse-GE / DFS-reorder changes.
"""

import numpy as np

from Hypergraph_Product_Code_Construction_v3 import HGP_code, Toric3, Hrep3
from peeling_cluster_decoder import combined_peeling_and_cluster_decoder


PASS = "PASS"
FAIL = "FAIL"


def _report(name, ok):
    print(f"  [{PASS if ok else FAIL}] {name}")
    return ok


def check_classical_repetition_code():
    print("Classical 3-bit repetition code (Hrep3):")
    ok = True
    e = np.array([1, 0, 1], dtype=int)
    syndrome = Hrep3.dot(e) % 2
    ok &= _report("Hrep3 . (1,0,1) == (1,1,0)",
                  np.array_equal(syndrome, np.array([1, 1, 0], dtype=int)))
    return ok


def check_toric3_parameters():
    print("Toric3 = HGP_code(Hrep3, Hrep3):")
    ok = True
    ok &= _report("num_qubits == 18", Toric3.num_qubits == 18)
    ok &= _report("kx == 10", Toric3.kx == 10)
    ok &= _report("kz == 10", Toric3.kz == 10)
    ok &= _report("dim == 2", Toric3.dim == 2)
    ok &= _report("num_h_qubits == 9", Toric3.num_h_qubits == 9)
    ok &= _report("num_v_qubits == 9", Toric3.num_v_qubits == 9)
    ok &= _report("num_checks == 9", Toric3.num_checks == 9)
    ok &= _report("num_generators == 9", Toric3.num_generators == 9)

    # CSS commutation: Hx . Hz^T == 0 (mod 2)
    commutator = (Toric3.Hx.dot(Toric3.Hz.T)) % 2
    ok &= _report("Hx . Hz^T == 0 (CSS commutation)",
                  np.array_equal(commutator, np.zeros_like(commutator)))
    return ok


def check_toric3_check_adjacencies():
    print("Toric3 check adjacencies (list_of_qubits_per_check):")
    ok = True
    expected = {0: {0, 1, 9, 15}, 1: {1, 2, 10, 16}, 2: {0, 2, 11, 17}}
    for check_index, expected_set in expected.items():
        ok &= _report(f"check{check_index} -> {sorted(expected_set)}",
                      Toric3.list_of_qubits_per_check[check_index] == expected_set)
    return ok


def check_peeling_only_example():
    print("Peeling-only example: E={0,1}, e={0} -> s={0,2}, predicts {0}:")
    ok = True
    s_index_set = Toric3.Hz_syn_index_set_for_X_err({0})
    ok &= _report("syndrome of e={0} is {0,2}", s_index_set == {0, 2})

    predicted, results = combined_peeling_and_cluster_decoder(Toric3, {0, 1}, s_index_set)
    ok &= _report("predicted error == {0}", predicted == {0})
    ok &= _report("total_dangling_checks_corrected == 2",
                  results['total_dangling_checks_corrected'] == 2)
    return ok


def check_classical_stopping_set_example():
    print("Classical stopping-set example: E={0,1,2}, e={1} -> s={0,1}, predicts {1}:")
    ok = True
    s_index_set = Toric3.Hz_syn_index_set_for_X_err({1})
    ok &= _report("syndrome of e={1} is {0,1}", s_index_set == {0, 1})

    predicted, results = combined_peeling_and_cluster_decoder(Toric3, {0, 1, 2}, s_index_set)
    ok &= _report("predicted error == {1}", predicted == {1})
    ok &= _report("classical_stopping_sets_used == 1",
                  results['classical_stopping_sets_used'] == 1)
    return ok


def main():
    print("=" * 70)
    print("PPVH erasure decoder - setup verification")
    print("=" * 70)

    checks = [
        check_classical_repetition_code,
        check_toric3_parameters,
        check_toric3_check_adjacencies,
        check_peeling_only_example,
        check_classical_stopping_set_example,
    ]

    all_ok = True
    for check in checks:
        all_ok &= check()
        print()

    print("=" * 70)
    if all_ok:
        print("ALL CHECKS PASSED")
        print("=" * 70)
        return 0
    else:
        print("SOME CHECKS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
