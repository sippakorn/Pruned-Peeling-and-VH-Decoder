"""Triconnected component decomposition using NetworkX.

Reads a simple undirected graph from stdin (adjacency-list format) and
decomposes it into triconnected components by:
  1. splitting into connected components,
  2. splitting each into biconnected components (blocks / bridges),
  3. recursively splitting each block at separation pairs (2-vertex cuts),
     inserting a virtual edge between the pair in each split piece.
"""

import sys

import networkx as nx


def read_graph_from_stdin():
    """Read an undirected graph from stdin.

    Format:
      First line: N M   (N = number of nodes labelled 1..N, M = number of edges)
      Next M lines: u v (an undirected edge between nodes u and v)
    """
    tokens = sys.stdin.read().split()
    it = iter(tokens)
    n = int(next(it))
    m = int(next(it))
    G = nx.Graph()
    G.add_nodes_from(range(1, n + 1))
    for _ in range(m):
        u = int(next(it))
        v = int(next(it))
        G.add_edge(u, v)
    return G


def to_multigraph(G):
    """Return a MultiGraph copy of G so virtual edges can coexist with real ones."""
    M = nx.MultiGraph()
    M.add_nodes_from(G.nodes())
    for u, v in G.edges():
        M.add_edge(u, v, virtual=False)
    return M


def connected_component_subgraphs(G):
    """Yield each connected component of G as a subgraph."""
    for nodes in nx.connected_components(G):
        yield G.subgraph(nodes).copy()


def biconnected_component_subgraphs(G):
    """Yield each biconnected component (block) of G as a subgraph.

    Bridges come out as 2-vertex blocks.
    """
    for nodes in nx.biconnected_components(G):
        yield G.subgraph(nodes).copy()


def find_separation_pair(G):
    """Return a separation pair (u, v) in biconnected graph G, or None if none exists.

    A separation pair is a pair of vertices whose removal disconnects G
    (with at least 2 remaining vertices). If G has at most 3 vertices it is
    triconnected by definition here.

    Uses minimum-node-cut instead of O(n^2) brute-force enumeration so that
    large biconnected blocks (which arise at high erasure rates) are handled
    in reasonable time.
    """
    nodes = list(G.nodes())
    if len(nodes) <= 3:
        return None

    # Collapse parallel edges (including virtual ones) into a simple graph for
    # the connectivity calculation.  Virtual edges count as real connections
    # here because each piece produced by split_at_separation_pair is required
    # to be biconnected (the virtual edge is what makes u–v still reachable).
    simple = nx.Graph(G)

    try:
        if nx.node_connectivity(simple) >= 3:
            return None
        cut = nx.minimum_node_cut(simple)
    except Exception:
        return None

    if len(cut) != 2:
        return None

    u, v = tuple(cut)
    H = simple.copy()
    H.remove_nodes_from([u, v])
    if H.number_of_nodes() >= 2 and not nx.is_connected(H):
        return (u, v)
    return None


def split_at_separation_pair(G, u, v):
    """Split biconnected multigraph G at separation pair (u, v).

    Each piece consists of one connected component of G - {u, v},
    together with u, v, all edges incident to that component from u or v,
    and a newly added virtual edge (u, v). The original edge (u, v), if any,
    is placed with the first piece.
    """
    H = G.copy()
    H.remove_nodes_from([u, v])
    components = list(nx.connected_components(H))

    pieces = []
    for comp in components:
        nodes = set(comp) | {u, v}
        piece = nx.MultiGraph()
        piece.add_nodes_from(nodes)
        for a, b, data in G.edges(data=True):
            if a in nodes and b in nodes and {a, b} != {u, v}:
                piece.add_edge(a, b, virtual=data.get("virtual", False))
        piece.add_edge(u, v, virtual=True)
        pieces.append(piece)

    has_real_uv = any(
        not data.get("virtual", False)
        for a, b, data in G.edges(data=True)
        if {a, b} == {u, v}
    )
    if has_real_uv and pieces:
        pieces[0].add_edge(u, v, virtual=False)

    return pieces


def decompose_block(G):
    """Recursively decompose a biconnected multigraph into triconnected pieces."""
    pair = find_separation_pair(G)
    if pair is None:
        return [G]
    u, v = pair
    result = []
    for piece in split_at_separation_pair(G, u, v):
        result.extend(decompose_block(piece))
    return result


def triconnected_components(G):
    """Return the list of triconnected components of G as MultiGraphs.

    Isolated vertices and bridges are returned as singleton / 2-vertex components.
    """
    components = []
    for cc in connected_component_subgraphs(G):
        if cc.number_of_nodes() == 1:
            components.append(to_multigraph(cc))
            continue
        for block in biconnected_component_subgraphs(cc):
            block_mg = to_multigraph(block)
            if block_mg.number_of_nodes() <= 2:
                components.append(block_mg)
            else:
                components.extend(decompose_block(block_mg))
    return components


def component_size(component):
    """Size of a component = number of vertices."""
    return component.number_of_nodes()


def format_component(component):
    """Format a component's vertices and edges for output, marking virtual edges."""
    verts = sorted(map(str, component.nodes()))
    edges = []
    for a, b, data in component.edges(data=True):
        mark = "*" if data.get("virtual") else ""
        s, t = sorted([str(a), str(b)])
        edges.append(f"({s},{t}){mark}")
    edges.sort()
    return verts, edges


def report(components):
    """Print number of components, each component's size, and the largest ones."""
    n = len(components)
    print(f"Number of triconnected components: {n}")

    print("\nComponent sizes (vertices, edges):")
    for i, c in enumerate(components):
        print(f"  [{i}] {c.number_of_nodes()} vertices, {c.number_of_edges()} edges")

    if not components:
        return

    max_size = max(component_size(c) for c in components)
    largest = [(i, c) for i, c in enumerate(components) if component_size(c) == max_size]
    print(f"\nLargest component(s): size = {max_size} vertices, count = {len(largest)}")
    for i, c in largest:
        verts, edges = format_component(c)
        print(f"  [{i}] vertices: {verts}")
        print(f"       edges:    {edges}   (*=virtual)")


# def main():
#     G = read_graph_from_stdin()
#     components = triconnected_components(G)
#     report(components)


# if __name__ == "__main__":
#     main()
