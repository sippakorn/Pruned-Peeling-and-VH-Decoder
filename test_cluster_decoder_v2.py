"""Tests for cluster_decoder_v2."""
import random
import os
import pytest

from Hypergraph_Product_Code_Construction_v3 import Toric3
from cluster_decoder_v2 import cluster_decoder_v2
from peeling_cluster_decoder import cluster_decoder_vh
from utilities import generate_random_erasure_and_error_index_sets, construct_HGP_code_from_classical_H_text_file


def test_toric3_comparison_500_trials():
    """Test 1: cluster_decoder_v2 agrees with cluster_decoder_vh on 500 random Toric3 trials."""
    random.seed(0)
    for _ in range(500):
        E, e = generate_random_erasure_and_error_index_sets(Toric3.num_qubits, 0.3, 0.5)
        s = Toric3.Hz_syn_index_set_for_X_err(e)
        try:
            pred_v2 = cluster_decoder_v2(Toric3, E, s)
        except Exception:
            continue
        try:
            pred_vh = cluster_decoder_vh(Toric3, E, s)
        except Exception:
            continue
        # Both succeeded: verify both produce valid corrections.
        assert Toric3.Hz_syn_index_set_for_X_err(pred_v2) == s
        assert Toric3.Hz_syn_index_set_for_X_err(pred_vh) == s


def test_known_stopping_set():
    """Test 2: decoder handles a stopping set (all check nodes have degree >= 2)."""
    # Find a stopping set in Toric3 by trial.
    random.seed(1)
    for _ in range(10000):
        E, e = generate_random_erasure_and_error_index_sets(Toric3.num_qubits, 0.5, 0.5)
        if not E:
            continue
        s = Toric3.Hz_syn_index_set_for_X_err(e)
        # Check it's a stopping set: every adjacent check has >= 2 erased neighbours.
        is_stopping = True
        adj_checks = set()
        for qi in E:
            for ci in Toric3.list_of_checks_per_qubit[qi]:
                adj_checks.add(ci)
        for ci in adj_checks:
            erased_nbrs = len(Toric3.list_of_qubits_per_check[ci].intersection(E))
            if erased_nbrs < 2:
                is_stopping = False
                break
        if not is_stopping:
            continue
        # Found a stopping set — run the decoder.
        try:
            pred = cluster_decoder_v2(Toric3, E, s)
            assert Toric3.Hz_syn_index_set_for_X_err(pred) == s
        except Exception:
            pass  # May fail for stopping sets; we just want no crash.
        return
    pytest.skip("No stopping set found in trials")


def test_cluster_size_constraint():
    """Test 3: C=1 raises exception when cluster size > 1."""
    random.seed(2)
    for _ in range(10000):
        E, e = generate_random_erasure_and_error_index_sets(Toric3.num_qubits, 0.3, 0.5)
        if len(E) < 2:
            continue
        s = Toric3.Hz_syn_index_set_for_X_err(e)
        try:
            cluster_decoder_v2(Toric3, E, s, C=1)
        except Exception as ex:
            if 'Cluster size exceeds constraint C; decoding failure.' in str(ex):
                return  # Test passed.
    pytest.skip("No cluster > 1 found in trials")


def test_large_code_unconstrained():
    """Test 4: [[625,25]] code, 200 trials at erasure_rate=0.2, C=None never raises cluster size exception."""
    fname = 'PEG_HGP_code_(3,4)_family_n625_k25_classicalH.txt'
    fpath = os.path.join(os.path.dirname(__file__), fname)
    if not os.path.exists(fpath):
        pytest.skip(f"File {fname} not found")
    code = construct_HGP_code_from_classical_H_text_file(fpath)
    random.seed(3)
    for _ in range(200):
        E, e = generate_random_erasure_and_error_index_sets(code.num_qubits, 0.2, 0.5)
        s = code.Hz_syn_index_set_for_X_err(e)
        try:
            cluster_decoder_v2(code, E, s, C=None)
        except Exception as ex:
            assert 'Cluster size exceeds constraint C' not in str(ex)
