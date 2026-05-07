import numpy as np
import networkx as nx

from utilities import perform_classical_syndrome_analysis_with_erasure
from triconnected import triconnected_components as _triconn


def cluster_decoder_v3(HGP_code, E_index_set, s_index_set, C=None):
    """Cluster decoder using triconnected component decomposition.

    Extends cluster_decoder_v2 by replacing biconnected components with
    triconnected components.  Cut boundaries are either:

      cut1 - single articulation point (1-node): same as v2
      cut2 - separation pair (2-node): enumerate up to 4 (kp, kq) combinations

    This allows the decoder to handle cases where v2 would fail due to
    non-tree cluster structure at high erasure rates.

    Parameters
    ----------
    HGP_code : HGP_code
    E_index_set : set of erased qubit indices after pruned peeling
    s_index_set : set of non-zero check indices (residual syndrome)
    C : int or None – optional cluster size constraint (variable nodes per TCC)
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

    if G.number_of_edges() == 0:
        if s_index_set:
            raise Exception('Predicted error does not match syndrome.')
        return set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _syn(ci):
        return 1 if ci in s_index_set else 0

    def _solve_cluster(tcc_nodes, fixed_vars, target_contribs):
        """Solve one TCC subproblem; returns {var_idx: val} or None."""
        c_labels = [n for n in tcc_nodes if n.startswith('c_')]
        v_labels = [n for n in tcc_nodes if n.startswith('v_')]
        check_indices = [int(n[2:]) for n in c_labels]
        var_indices   = [int(n[2:]) for n in v_labels]
        var_set = set(var_indices)

        if not check_indices or not var_indices:
            return {vi: fixed_vars.get(vi, 0) for vi in var_indices}

        s_local = np.array(
            [target_contribs.get(ci, _syn(ci)) for ci in check_indices], dtype=int)

        free_vars = [vi for vi in var_indices if vi not in fixed_vars]
        for vi, val in fixed_vars.items():
            if vi in var_set and val == 1:
                for k, ci in enumerate(check_indices):
                    if HGP_code.Hz[ci, vi] == 1:
                        s_local[k] ^= 1

        if not free_vars:
            return ({vi: fixed_vars.get(vi, 0) for vi in var_indices}
                    if np.all(s_local == 0) else None)

        var_to_local = {vi: k for k, vi in enumerate(var_indices)}
        free_local   = [var_to_local[vi] for vi in free_vars]
        H_sub  = HGP_code.Hz[np.ix_(check_indices, var_indices)]
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

    def _contrib(label, sol):
        """Variable value or this cluster's XOR-contribution to a check, from sol."""
        if label.startswith('v_'):
            return sol.get(int(label[2:]), 0)
        ci = int(label[2:])
        return sum(v for k, v in sol.items() if v == 1 and HGP_code.Hz[ci, k] == 1) % 2

    def _key_space(cut_node):
        return [0, 1] if cut_node[0] == 'cut1' else [(0,0), (0,1), (1,0), (1,1)]

    def _apply_key(cut_node, key, fv, tc):
        """Write boundary key into fv/tc.  Returns True if a conflict is detected."""
        if cut_node[0] == 'cut1':
            label = cut_node[1]
            if label.startswith('v_'):
                vi = int(label[2:])
                if vi in fv and fv[vi] != key:
                    return True
                fv[vi] = key
            else:
                ci = int(label[2:])
                if ci in tc and tc[ci] != key:
                    return True
                tc[ci] = key
        else:
            for label, kval in zip(cut_node[1], key):
                if label.startswith('v_'):
                    vi = int(label[2:])
                    if vi in fv and fv[vi] != kval:
                        return True
                    fv[vi] = kval
                else:
                    ci = int(label[2:])
                    if ci in tc and tc[ci] != kval:
                        return True
                    tc[ci] = kval
        return False

    # ------------------------------------------------------------------
    # Phase 2+3 – Connected components
    # ------------------------------------------------------------------
    predicted_e_index_set = set()

    for comp_nodes in nx.connected_components(G):
        comp_G = G.subgraph(comp_nodes).copy()

        # Triconnected decomposition
        tcc_list_mg = _triconn(comp_G)
        tcc_list    = [set(tc.nodes()) for tc in tcc_list_mg]
        num_tcc     = len(tcc_list)

        if C is not None:
            for tnodes in tcc_list:
                if sum(1 for n in tnodes if n.startswith('v_')) > C:
                    raise Exception('Cluster size exceeds constraint C.')

        # ---- Build cluster tree CT ----
        # Nodes: ('cluster', idx)  +  ('cut1', label)  +  ('cut2', (p, q))
        CT = nx.Graph()
        for idx in range(num_tcc):
            CT.add_node(('cluster', idx))

        node_to_tccs = {}
        for idx, tnodes in enumerate(tcc_list):
            for nd in tnodes:
                node_to_tccs.setdefault(nd, []).append(idx)

        processed_pairs = set()
        for tidxs in node_to_tccs.values():
            for i in range(len(tidxs)):
                for j in range(i + 1, len(tidxs)):
                    ti, tj = tidxs[i], tidxs[j]
                    pair = (min(ti, tj), max(ti, tj))
                    if pair in processed_pairs:
                        continue
                    processed_pairs.add(pair)
                    shared = tcc_list[ti] & tcc_list[tj]
                    if len(shared) == 1:
                        cn = ('cut1', next(iter(shared)))
                    elif len(shared) == 2:
                        cn = ('cut2', tuple(sorted(shared)))
                    else:
                        continue
                    CT.add_node(cn)
                    CT.add_edge(('cluster', ti), cn)
                    CT.add_edge(('cluster', tj), cn)

        root     = ('cluster', 0)
        bfs_tree = nx.bfs_tree(CT, root)
        bfs_flat = [n for layer in nx.bfs_layers(CT, root) for n in layer]

        parent_of = {root: None}
        for u, v in bfs_tree.edges():
            parent_of[v] = u

        def _ch(node):
            return list(bfs_tree.successors(node))

        # ------------------------------------------------------------------
        # RecursiveCompute – leaf to root
        # cluster_info[idx] = {'solutions': {key: sol}, 'is_free': bool,
        #                       'frozen_key': key | None}
        # ------------------------------------------------------------------
        cluster_info = {}

        for node in reversed(bfs_flat):
            if node[0] != 'cluster':
                continue
            idx         = node[1]
            tnodes      = tcc_list[idx]
            parent_cut  = parent_of[node]

            fv0, tc0 = {}, {}

            for cut_node in _ch(node):
                child_cls = _ch(cut_node)

                if cut_node[0] == 'cut1':
                    label = cut_node[1]
                    if label.startswith('v_'):
                        vi = int(label[2:])
                        for cc in child_cls:
                            if not cluster_info[cc[1]]['is_free']:
                                fv0[vi] = cluster_info[cc[1]]['frozen_key']
                                break
                    else:
                        ci = int(label[2:])
                        accum, all_frz = 0, True
                        for cc in child_cls:
                            inf_cc = cluster_info[cc[1]]
                            if inf_cc['is_free']:
                                all_frz = False
                            else:
                                accum ^= inf_cc['frozen_key']
                        if all_frz:
                            tc0[ci] = _syn(ci) ^ accum

                else:  # cut2
                    p, q = cut_node[1]
                    pv_val = qv_val = None
                    pc_acc = qc_acc = 0
                    all_frz = True
                    for cc in child_cls:
                        inf_cc = cluster_info[cc[1]]
                        if inf_cc['is_free']:
                            all_frz = False
                        else:
                            kp, kq = inf_cc['frozen_key']
                            if p.startswith('v_'):
                                pv_val = kp
                            else:
                                pc_acc ^= kp
                            if q.startswith('v_'):
                                qv_val = kq
                            else:
                                qc_acc ^= kq
                    if all_frz:
                        if p.startswith('v_'):
                            if pv_val is not None:
                                fv0[int(p[2:])] = pv_val
                        else:
                            tc0[int(p[2:])] = _syn(int(p[2:])) ^ pc_acc
                        if q.startswith('v_'):
                            if qv_val is not None:
                                fv0[int(q[2:])] = qv_val
                        else:
                            tc0[int(q[2:])] = _syn(int(q[2:])) ^ qc_acc

            if parent_cut is None:
                sol = _solve_cluster(tnodes, fv0, tc0)
                if sol is None:
                    raise Exception('Predicted error does not match syndrome.')
                cluster_info[idx] = {
                    'solutions':  {0: sol},
                    'is_free':    False,
                    'frozen_key': 0,
                }
            else:
                solutions = {}
                for key in _key_space(parent_cut):
                    fv, tc = dict(fv0), dict(tc0)
                    if not _apply_key(parent_cut, key, fv, tc):
                        sol = _solve_cluster(tnodes, fv, tc)
                        if sol is not None:
                            solutions[key] = sol
                if not solutions:
                    raise Exception('Predicted error does not match syndrome.')
                is_free = len(solutions) > 1
                cluster_info[idx] = {
                    'solutions':  solutions,
                    'is_free':    is_free,
                    'frozen_key': None if is_free else next(iter(solutions)),
                }

        # ------------------------------------------------------------------
        # RecursiveSelect – root to leaf
        # req_key[cluster_idx] = key that the parent wants this cluster to use
        # ------------------------------------------------------------------
        selected_sol = {}
        req_key      = {}  # cluster_idx -> key

        for node in bfs_flat:
            if node[0] != 'cluster':
                continue
            idx  = node[1]
            info = cluster_info[idx]
            rk   = req_key.get(idx)
            sel  = (info['solutions'][rk]
                    if rk is not None and rk in info['solutions']
                    else next(iter(info['solutions'].values())))
            selected_sol[idx] = sel

            for cut_node in _ch(node):
                child_cls = _ch(cut_node)

                if cut_node[0] == 'cut1':
                    label = cut_node[1]
                    if label.startswith('v_'):
                        val = sel.get(int(label[2:]), 0)
                        for cc in child_cls:
                            req_key[cc[1]] = val
                    else:
                        ci           = int(label[2:])
                        s_k          = _syn(ci)
                        par_contrib  = _contrib(label, sel)
                        req_total    = s_k ^ par_contrib
                        frz_sum      = 0
                        free_list    = []
                        for cc in child_cls:
                            inf_cc = cluster_info[cc[1]]
                            if inf_cc['is_free']:
                                free_list.append(cc[1])
                            else:
                                frz_sum ^= inf_cc['frozen_key']
                        remaining = req_total ^ frz_sum
                        for k_idx, cc_idx in enumerate(free_list):
                            req_key[cc_idx] = remaining if k_idx == 0 else 0
                        for cc in child_cls:
                            cc_idx = cc[1]
                            if cc_idx not in req_key:
                                req_key[cc_idx] = cluster_info[cc_idx]['frozen_key']

                else:  # cut2
                    p, q    = cut_node[1]
                    kp      = _contrib(p, sel)
                    kq      = _contrib(q, sel)
                    # Required total from all children side
                    rtp = kp if p.startswith('v_') else _syn(int(p[2:])) ^ kp
                    rtq = kq if q.startswith('v_') else _syn(int(q[2:])) ^ kq

                    pf_sum = qf_sum = 0
                    p_free = []
                    q_free = []
                    for cc in child_cls:
                        inf_cc = cluster_info[cc[1]]
                        if inf_cc['is_free']:
                            if p.startswith('c_'):
                                p_free.append(cc[1])
                            if q.startswith('c_'):
                                q_free.append(cc[1])
                        else:
                            ckp, ckq = inf_cc['frozen_key']
                            if p.startswith('c_'):
                                pf_sum ^= ckp
                            if q.startswith('c_'):
                                qf_sum ^= ckq

                    p_rem = rtp ^ pf_sum
                    q_rem = rtq ^ qf_sum

                    for cc in child_cls:
                        cc_idx = cc[1]
                        inf_cc = cluster_info[cc_idx]
                        if inf_cc['is_free']:
                            tkp = rtp if p.startswith('v_') else (
                                p_rem if (p_free and cc_idx == p_free[0]) else 0)
                            tkq = rtq if q.startswith('v_') else (
                                q_rem if (q_free and cc_idx == q_free[0]) else 0)
                            req_key[cc_idx] = (tkp, tkq)
                        else:
                            req_key[cc_idx] = inf_cc['frozen_key']

        # ---- Merge ----
        merged = {}
        for sol in selected_sol.values():
            merged.update(sol)
        for vi, val in merged.items():
            if val == 1:
                predicted_e_index_set.add(vi)

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------
    if HGP_code.Hz_syn_index_set_for_X_err(predicted_e_index_set) != s_index_set:
        raise Exception('Predicted error does not match syndrome.')

    return predicted_e_index_set
