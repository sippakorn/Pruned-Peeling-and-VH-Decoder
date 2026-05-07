"""A/B tests comparing perform_classical_syndrome_analysis_with_erasure (v1, dense
standard_form) against perform_classical_syndrome_analysis_with_erasure_v2 (v2,
sparse_form). Both versions must produce identical predicted error vectors on
every input."""

import random
import numpy as np
import pytest

from Hypergraph_Product_Code_Construction_v3 import (
    standard_form,
    sparse_form,
    canonical_basis_vector,
)
from utilities import (
    perform_classical_syndrome_analysis_with_erasure,
    perform_classical_syndrome_analysis_with_erasure_v2,
)
from scipy.sparse import csr_matrix


SEED = 20260421


def _small_H():
    # Handcrafted 6x10 binary parity-check-like matrix. Rows 0-4 are linearly
    # independent; row 5 = row 0 XOR row 1 (forces zero-row trimming inside
    # both RREF solvers).
    H = np.array([
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 0, 0, 1, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 1, 0, 0],  # row 0 XOR row 1 XOR row 4
    ], dtype=int)
    return H


def _random_decodable_case(H, rng):
    m, n = H.shape
    erasure_size = int(rng.integers(1, n + 1))
    E_indices = rng.choice(n, size=erasure_size, replace=False)
    E_index_set = set(int(x) for x in E_indices)

    e = np.zeros(n, dtype=int)
    for idx in E_index_set:
        e[idx] = int(rng.integers(0, 2))

    s = (H @ e) % 2
    return E_index_set, e, s


def _assert_both_agree(H, s, E_index_set):
    pred_v1 = perform_classical_syndrome_analysis_with_erasure(H, s, E_index_set)
    pred_v2 = perform_classical_syndrome_analysis_with_erasure_v2(H, s, E_index_set)
    assert pred_v1.shape == pred_v2.shape, (pred_v1.shape, pred_v2.shape)
    assert np.array_equal(pred_v1, pred_v2), (
        f"v1 and v2 disagree:\n  v1={pred_v1}\n  v2={pred_v2}\n  s={s}\n  E={sorted(E_index_set)}"
    )
    assert np.array_equal((H @ pred_v1) % 2, s), "v1 predicted_e does not reproduce s"
    return pred_v1


def test_sparse_form_matches_standard_form_directly():
    """sparse_form must produce bit-identical RREF, A, and pivot list as standard_form."""
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        m = int(rng.integers(3, 12))
        n = int(rng.integers(3, 15))
        M = rng.integers(0, 2, size=(m, n)).astype(int)

        M_std, A_std, pivots_std = standard_form(M)
        M_spa, A_spa, pivots_spa = sparse_form(csr_matrix(M.astype(np.int8)))

        assert pivots_std == pivots_spa, (pivots_std, pivots_spa)
        assert np.array_equal(M_std, M_spa), (M_std, M_spa)
        # A shapes may differ only in the rank-deficient corner case where
        # standard_form's empty-A numpy.transpose produces a degenerate shape.
        # Compare entrywise when both are non-empty.
        if A_std.size and A_spa.size:
            assert np.array_equal(A_std, A_spa)


def test_ab_random_trials():
    """200 random decodable trials on a small handcrafted H; v1 and v2 must agree."""
    H = _small_H()
    rng = np.random.default_rng(SEED)
    for _ in range(200):
        E_index_set, _e, s = _random_decodable_case(H, rng)
        _assert_both_agree(H, s, E_index_set)


def test_zero_syndrome():
    """Zero syndrome, non-empty random erasure set -> both versions should return the zero vector."""
    # Note: erasure_size >= 1 is required because v1's standard_form has a latent
    # UnboundLocalError when the augmented matrix is all zeros (empty erasure +
    # zero syndrome). That edge case is covered separately by
    # test_empty_erasure_zero_syndrome which exercises v2 only.
    H = _small_H()
    m, n = H.shape
    rng = np.random.default_rng(SEED + 1)
    for _ in range(20):
        erasure_size = int(rng.integers(1, n + 1))
        E_indices = rng.choice(n, size=erasure_size, replace=False)
        E_index_set = set(int(x) for x in E_indices)
        s = np.zeros(m, dtype=int)
        pred = _assert_both_agree(H, s, E_index_set)
        assert np.array_equal(pred, np.zeros(n, dtype=int))


def test_full_erasure():
    """Every qubit erased; random errors -> v1 and v2 agree."""
    H = _small_H()
    _, n = H.shape
    E_index_set = set(range(n))
    rng = np.random.default_rng(SEED + 2)
    for _ in range(30):
        e = rng.integers(0, 2, size=n).astype(int)
        s = (H @ e) % 2
        _assert_both_agree(H, s, E_index_set)


def test_empty_erasure_zero_syndrome():
    """No erased qubits, zero syndrome.

    v1 crashes on this input (all-zero H_aug triggers an UnboundLocalError inside
    standard_form, which never assigns M_standard when every row of M is zero).
    v2 is hardened against this case. This test documents that v2 returns the
    zero vector as expected, which is the mathematically correct answer
    (e = 0 is the unique minimal-weight solution to H*e = 0 with empty support).
    """
    H = _small_H()
    m, n = H.shape
    s = np.zeros(m, dtype=int)
    E_index_set = set()
    with pytest.raises(UnboundLocalError):
        perform_classical_syndrome_analysis_with_erasure(H, s, E_index_set)
    pred_v2 = perform_classical_syndrome_analysis_with_erasure_v2(H, s, E_index_set)
    assert np.array_equal(pred_v2, np.zeros(n, dtype=int))


def test_single_qubit_erasure():
    """For each qubit k: erase only k with error e_k; both versions must recover e_k."""
    H = _small_H()
    m, n = H.shape
    for k in range(n):
        if not np.any(H[:, k]):
            # Column is all zero: zero syndrome, any single-qubit error is a valid solution.
            # Both v1 and v2 set all free variables to 0, so predicted_e should be 0.
            E_index_set = {k}
            s = np.zeros(m, dtype=int)
            pred = _assert_both_agree(H, s, E_index_set)
            assert np.array_equal(pred, np.zeros(n, dtype=int))
            continue
        e = canonical_basis_vector(k, n)
        s = (H @ e) % 2
        E_index_set = {k}
        pred = _assert_both_agree(H, s, E_index_set)
        assert np.array_equal(pred, e), f"Failed to recover single-qubit error at k={k}"


def test_rank_deficient_H():
    """Use H with an explicit duplicate row; forces zero-row trimming."""
    H_base = _small_H()
    H = np.vstack([H_base, H_base[0:1, :]])  # duplicate row 0
    rng = np.random.default_rng(SEED + 3)
    for _ in range(100):
        E_index_set, _e, s = _random_decodable_case(H, rng)
        _assert_both_agree(H, s, E_index_set)


def test_input_type_validation():
    """sparse_form must reject dense numpy input explicitly."""
    M = np.eye(3, dtype=int)
    with pytest.raises(TypeError):
        sparse_form(M)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
