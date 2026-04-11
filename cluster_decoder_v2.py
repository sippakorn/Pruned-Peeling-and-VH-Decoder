import numpy as np
import networkx as nx

from utilities import perform_classical_syndrome_analysis_with_erasure


def cluster_decoder_v2(HGP_code, E_index_set, s_index_set, C=None):
    """Cluster decoder based on biconnected component decomposition.

    Implements Algorithms 1-5 from:
    "Cluster Decomposition for Improved Erasure Decoding of Quantum LDPC Codes"
    by Yao, Gokduman, and Pfister (arXiv:2412.08817, December 2024).

    Parameters
    ----------
    HGP_code : HGP_code
        The HGP code class object.
    E_index_set : set
        Set of erased qubit indices remaining after pruned peeling.
    s_index_set : set
        Set of non-zero check indices (residual syndrome) after pruned peeling.
    C : int or None
        Optional cluster size constraint. If None, no constraint (full ML).
        If an integer, raises an exception if any cluster has more variable
        nodes than C.

    Returns
    -------
    predicted_e_index_set : set
        Set of qubit indices forming the predicted error vector.
    """

    # ------------------------------------------------------------------
    # Phase 1 – Build residual Tanner graph G'
    # ------------------------------------------------------------------
    G = nx.Graph()

    for i in E_index_set:
        G.add_node('v_{}'.format(i))

    for i in E_index_set:
        for j in HGP_code.list_of_checks_per_qubit[i]:
            G.add_edge('c_{}'.format(j), 'v_{}'.format(i))

    # Empty erasure case.
    if G.number_of_edges() == 0:
        if s_index_set != set():
            raise Exception('Predicted error does not match syndrome.')
        return set()

    # ------------------------------------------------------------------
    # Helper: syndrome value for a check index.
    # ------------------------------------------------------------------
    def _syn(ci):
        return 1 if ci in s_index_set else 0

    # ------------------------------------------------------------------
    # Helper: solve a cluster subproblem.
    #
    # Parameters
    # ----------
    # bcc : set of node labels in this biconnected component
    # fixed_vars : dict {var_idx (int): value (0/1)} — variables fixed
    # target_contribs : dict {check_idx (int): target (0/1)} — for each
    #     check in this dict, the cluster's variables must contribute
    #     exactly `target` to that check. For checks not in this dict,
    #     the target is taken from s_index_set.
    #
    # Returns a dict {var_idx: value} for all variable nodes in bcc,
    # or None if no solution exists.
    # ------------------------------------------------------------------
    def _solve_cluster(bcc, fixed_vars, target_contribs):
        c_labels = [n for n in bcc if n.startswith('c_')]
        v_labels = [n for n in bcc if n.startswith('v_')]

        check_indices = [int(n[2:]) for n in c_labels]
        var_indices   = [int(n[2:]) for n in v_labels]

        # No constraints: trivial solution.
        if not check_indices or not var_indices:
            return {vi: fixed_vars.get(vi, 0) for vi in var_indices}

        # Local index maps.
        check_to_local = {ci: k for k, ci in enumerate(check_indices)}

        # Build target syndrome vector (REPLACE semantics for cut-nodes).
        s_local = np.array(
            [target_contribs.get(ci, _syn(ci)) for ci in check_indices],
            dtype=int)

        # Fix variables and adjust syndrome.
        free_vars = [vi for vi in var_indices if vi not in fixed_vars]
        for vi, val in fixed_vars.items():
            if val == 1:
                for k, ci in enumerate(check_indices):
                    if HGP_code.Hz[ci, vi] == 1:
                        s_local[k] ^= 1

        # If no free variables remain, check whether syndrome is satisfied.
        if not free_vars:
            if np.all(s_local == 0):
                return {vi: fixed_vars.get(vi, 0) for vi in var_indices}
            return None

        # Build submatrix for free variables.
        var_to_local = {vi: k for k, vi in enumerate(var_indices)}
        free_local = [var_to_local[vi] for vi in free_vars]
        H_sub = HGP_code.Hz[np.ix_(check_indices, var_indices)]
        H_free = H_sub[:, free_local]

        try:
            sol_free = perform_classical_syndrome_analysis_with_erasure(
                H_free, s_local, set(range(len(free_vars))))
        except Exception:
            return None

        if not np.array_equal(np.dot(H_free, sol_free) % 2, s_local):
            return None

        result = dict(fixed_vars)
        for k, vi in enumerate(free_vars):
            result[vi] = int(sol_free[free_local[k]])
        return result

    # ------------------------------------------------------------------
    # Phase 2+3 – Process each connected component.
    # ------------------------------------------------------------------
    predicted_e_index_set = set()

    for comp_nodes in nx.connected_components(G):
        comp_G = G.subgraph(comp_nodes).copy()
        bcc_list = list(nx.biconnected_components(comp_G))
        art_pts  = set(nx.articulation_points(comp_G))

        # Check cluster size constraint.
        if C is not None:
            for bcc in bcc_list:
                var_count = sum(1 for n in bcc if n.startswith('v_'))
                if var_count > C:
                    raise Exception(
                        'Cluster size exceeds constraint C; decoding failure.')

        # ------------------------------------------------------------------
        # Build the cluster tree.
        # Node types:
        #   ('cluster', idx) – one per biconnected component
        #   ('cut', label)   – one per articulation point
        # Edges connect each cluster to its articulation points.
        # ------------------------------------------------------------------
        CT = nx.Graph()
        for idx, bcc in enumerate(bcc_list):
            CT.add_node(('cluster', idx), bcc=bcc)
            for node in bcc:
                if node in art_pts:
                    CT.add_node(('cut', node))
                    CT.add_edge(('cluster', idx), ('cut', node))

        root = ('cluster', 0)
        bfs_tree = nx.bfs_tree(CT, root)
        bfs_order = list(nx.bfs_layers(CT, root))
        bfs_order_flat = [n for layer in bfs_order for n in layer]

        # Parent map in the BFS tree.
        parent_of = {root: None}
        for u, v in bfs_tree.edges():
            parent_of[v] = u

        # Children in the BFS tree.
        def _children(node):
            return list(bfs_tree.successors(node))

        # ------------------------------------------------------------------
        # RecursiveCompute: leaf-to-root (reverse BFS order).
        #
        # cluster_info[idx]:
        #   'solutions'  : {0: sol0, 1: sol1} | {0: sol0} | {1: sol1}
        #   'is_free'    : bool
        #   'frozen_val' : int or None
        # ------------------------------------------------------------------
        cluster_info = {}

        for node in reversed(bfs_order_flat):
            if node[0] != 'cluster':
                continue

            idx = node[1]
            bcc = bcc_list[idx]
            parent_cut = parent_of[node]   # None or ('cut', label)

            # Gather constraints from child cut-nodes.
            fixed_vars     = {}  # {vi: val}  from frozen variable cut-nodes
            target_contribs = {}  # {ci: req}  required contrib from this cluster

            for cut_node in _children(node):
                # cut_node = ('cut', label)
                cut_label = cut_node[1]

                if cut_label.startswith('v_'):
                    vi = int(cut_label[2:])
                    # Each child cluster of this variable cut-node has been
                    # solved. If any is frozen, it constrains vi's value.
                    for child_cluster in _children(cut_node):
                        ci_idx = child_cluster[1]
                        ci_info = cluster_info[ci_idx]
                        if not ci_info['is_free']:
                            fixed_vars[vi] = ci_info['frozen_val']
                            break
                    # (If free, no constraint on vi from this direction.)

                else:  # check cut-node
                    ci = int(cut_label[2:])
                    s_k = _syn(ci)
                    # XOR-accumulate contributions from frozen child clusters.
                    accum = 0
                    all_frozen = True
                    for child_cluster in _children(cut_node):
                        ci_idx = child_cluster[1]
                        ci_info = cluster_info[ci_idx]
                        if ci_info['is_free']:
                            all_frozen = False
                        else:
                            accum ^= ci_info['frozen_val']
                    if all_frozen:
                        # B's required contribution to ci.
                        target_contribs[ci] = s_k ^ accum
                    # If some children are free, we leave ci unconstrained
                    # (any contribution from B will do; RecursiveSelect sorts it out).

            # Solve this cluster.
            if parent_cut is None:
                # Root cluster: one solution.
                sol = _solve_cluster(bcc, fixed_vars, target_contribs)
                if sol is None:
                    raise Exception('Predicted error does not match syndrome.')
                cluster_info[idx] = {
                    'solutions':   {0: sol},
                    'is_free':     False,
                    'frozen_val':  0,
                }

            else:
                cut_label = parent_cut[1]

                if cut_label.startswith('v_'):
                    vi = int(cut_label[2:])
                    fv0 = dict(fixed_vars); fv0[vi] = 0
                    fv1 = dict(fixed_vars); fv1[vi] = 1
                    sol0 = _solve_cluster(bcc, fv0, target_contribs)
                    sol1 = _solve_cluster(bcc, fv1, target_contribs)

                    solutions = {}
                    if sol0 is not None: solutions[0] = sol0
                    if sol1 is not None: solutions[1] = sol1
                    if not solutions:
                        raise Exception('Predicted error does not match syndrome.')

                    is_free   = len(solutions) == 2
                    frozen_val = None if is_free else next(iter(solutions))
                    cluster_info[idx] = {
                        'solutions':  solutions,
                        'is_free':    is_free,
                        'frozen_val': frozen_val,
                    }

                else:  # check cut-node
                    ci = int(cut_label[2:])
                    tc0 = dict(target_contribs); tc0[ci] = 0
                    tc1 = dict(target_contribs); tc1[ci] = 1
                    sol0 = _solve_cluster(bcc, fixed_vars, tc0)
                    sol1 = _solve_cluster(bcc, fixed_vars, tc1)

                    solutions = {}
                    if sol0 is not None: solutions[0] = sol0
                    if sol1 is not None: solutions[1] = sol1
                    if not solutions:
                        raise Exception('Predicted error does not match syndrome.')

                    is_free   = len(solutions) == 2
                    frozen_val = None if is_free else next(iter(solutions))
                    cluster_info[idx] = {
                        'solutions':  solutions,
                        'is_free':    is_free,
                        'frozen_val': frozen_val,
                    }

        # ------------------------------------------------------------------
        # RecursiveSelect: root-to-leaf (BFS order).
        #
        # selected_sol[idx]  : the chosen solution dict for cluster idx
        # cut_parent_contrib[cut_label] : the contribution/value selected
        #   by the PARENT cluster for this cut-node.
        #   • Variable cut-node: the variable's value (0/1).
        #   • Check cut-node:    the parent cluster's contribution (0/1).
        # ------------------------------------------------------------------
        selected_sol = {}
        cut_parent_contrib = {}  # cut_label -> value chosen by parent

        for node in bfs_order_flat:
            if node[0] != 'cluster':
                continue

            idx = node[1]
            info = cluster_info[idx]
            parent_cut = parent_of[node]

            # Determine which solution to pick.
            if parent_cut is None:
                # Root: only one solution.
                key = next(iter(info['solutions']))
                sel = info['solutions'][key]
            else:
                cut_label = parent_cut[1]
                parent_val = cut_parent_contrib.get(cut_label, None)

                if cut_label.startswith('v_'):
                    # parent_val = required value of vi.
                    if parent_val is not None and parent_val in info['solutions']:
                        sel = info['solutions'][parent_val]
                    else:
                        key = next(iter(info['solutions']))
                        sel = info['solutions'][key]
                else:
                    # Check cut-node: parent_val = parent's contribution.
                    # Child must contribute s_k XOR parent_val.
                    ci = int(cut_label[2:])
                    s_k = _syn(ci)
                    if parent_val is not None:
                        required = s_k ^ parent_val
                    else:
                        required = s_k
                    if required in info['solutions']:
                        sel = info['solutions'][required]
                    else:
                        key = next(iter(info['solutions']))
                        sel = info['solutions'][key]

            selected_sol[idx] = sel

            # Propagate to child cut-nodes.
            for cut_node in _children(node):
                cut_label = cut_node[1]

                if cut_label.startswith('v_'):
                    vi = int(cut_label[2:])
                    # Value of vi from this cluster's solution.
                    cut_parent_contrib[cut_label] = sel.get(vi, 0)

                else:
                    # Check cut-node: compute this cluster's contribution.
                    ci = int(cut_label[2:])
                    contrib = 0
                    for vi, val in sel.items():
                        if val == 1 and HGP_code.Hz[ci, vi] == 1:
                            contrib ^= 1
                    cut_parent_contrib[cut_label] = contrib

                    # Equation 17: for multiple free child clusters of a check
                    # cut-node, first free gets the required contribution,
                    # the rest get 0.
                    s_k = _syn(ci)
                    required_total = s_k ^ contrib  # sum of all child contributions
                    # Subtract frozen children's contributions to get what free children must provide.
                    frozen_sum = 0
                    free_children = []
                    for child_cluster in _children(cut_node):
                        cc_idx = child_cluster[1]
                        cc_info = cluster_info[cc_idx]
                        if cc_info['is_free']:
                            free_children.append(cc_idx)
                        else:
                            frozen_sum ^= cc_info['frozen_val']
                    # Free children together must contribute: required_total XOR frozen_sum
                    remaining = required_total ^ frozen_sum
                    # First free child gets `remaining`, rest get 0.
                    for k, cc_idx in enumerate(free_children):
                        if k == 0:
                            # Override: the first free child must pick the solution
                            # where its contribution to ci is `remaining`.
                            cc_info = cluster_info[cc_idx]
                            if remaining in cc_info['solutions']:
                                selected_sol[cc_idx] = cc_info['solutions'][remaining]
                            else:
                                key = next(iter(cc_info['solutions']))
                                selected_sol[cc_idx] = cc_info['solutions'][key]
                        else:
                            # Remaining free children contribute 0.
                            cc_info = cluster_info[cc_idx]
                            if 0 in cc_info['solutions']:
                                selected_sol[cc_idx] = cc_info['solutions'][0]
                            else:
                                key = next(iter(cc_info['solutions']))
                                selected_sol[cc_idx] = cc_info['solutions'][key]

        # ------------------------------------------------------------------
        # Merge: collect variable values from all selected solutions.
        # Use a dict to handle articulation points appearing in multiple
        # cluster solutions (values should be consistent).
        # ------------------------------------------------------------------
        merged = {}
        for idx, sol in selected_sol.items():
            for vi, val in sol.items():
                merged[vi] = val

        for vi, val in merged.items():
            if val == 1:
                predicted_e_index_set.add(vi)

    # ------------------------------------------------------------------
    # Verify.
    # ------------------------------------------------------------------
    if HGP_code.Hz_syn_index_set_for_X_err(predicted_e_index_set) != s_index_set:
        raise Exception('Predicted error does not match syndrome.')

    return predicted_e_index_set
