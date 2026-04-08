"""Graph construction, Leiden clustering, and cluster merging.

Pipeline
--------
1. build_base_graph      — load citation edges once into an undirected igraph.
2. filter_to_giant_component — optionally restrict to the largest connected network.
3. apply_leiden          — partition papers into communities using the CPM objective.
4. merge_to_target       — merge smallest communities by inter-cluster edge density
                           until the desired number of clusters is reached.
5. build_cluster_graph   — collapse the paper graph into a cluster-level graph
                           suitable for visualisation.
"""
from collections import defaultdict
import heapq

import igraph as ig


# ---------------------------------------------------------------------------
# Base graph
# ---------------------------------------------------------------------------

def build_base_graph(edges: list[tuple]) -> ig.Graph:
    """Build an undirected weighted igraph from an edge list.

    Each edge may be a (u, v) or (u, v, weight) tuple. Duplicate edges and
    reversed edges (u→v and v→u) are collapsed into a single undirected edge
    whose weight equals the sum of all occurrences.

    Vertex attributes set:
        name   (str)   — unique node ID
        degree (int)   — total number of links

    Edge attributes set:
        weight (float) — summed edge weight (1.0 per occurrence for unweighted input)

    Args:
        edges: List of (u, v) or (u, v, weight) tuples.

    Returns:
        Undirected igraph with one vertex per unique node ID.
    """
    agg: dict[tuple[str, str], float] = defaultdict(float)
    nodes_set: set[str] = set()
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        w = float(edge[2]) if len(edge) > 2 else 1.0
        nodes_set.add(u)
        nodes_set.add(v)
        key = (min(u, v), max(u, v))
        agg[key] += w

    nodes = list(nodes_set)
    node_index = {n: i for i, n in enumerate(nodes)}

    g = ig.Graph(n=len(nodes), directed=False)
    g.vs["name"] = nodes
    edge_list = [(node_index[u], node_index[v]) for u, v in agg]
    g.add_edges(edge_list)
    g.es["weight"] = list(agg.values())
    g.vs["degree"] = g.degree()
    return g


def filter_to_giant_component(g: ig.Graph) -> tuple[ig.Graph, int]:
    """Return only the largest connected component of *g*.

    Papers outside the giant component have no citation path to the main body
    of literature and cannot be meaningfully placed on a science map.

    Returns:
        (giant, removed) where giant is a new igraph containing only the
        largest component and removed is the count of excluded papers.
    """
    giant = g.clusters().giant()
    removed = g.vcount() - giant.vcount()
    return giant, removed


# ---------------------------------------------------------------------------
# Leiden clustering
# ---------------------------------------------------------------------------

def apply_leiden(g: ig.Graph, resolution: float) -> None:
    """Partition *g* into communities using the Leiden algorithm (CPM objective).

    The CPM (Constant Potts Model) objective defines community quality as
    internal_edges - resolution × possible_internal_pairs. At resolution = 1.0
    every node becomes its own singleton (maximum density required). Lower
    values produce larger, coarser communities.

    Results are deterministic: the random seed and iteration count are fixed
    in config. Community membership is stored in-place as the vertex attribute
    ``community_id`` (0-based integer).

    Args:
        g:          Undirected igraph (modified in-place).
        resolution: CPM resolution parameter. Higher values produce smaller,
                    more numerous communities.
    """
    import random
    import config
    ig.set_random_number_generator(random.Random(config.LEIDEN_SEED))
    weights = g.es["weight"] if "weight" in g.es.attributes() else None
    partition = g.community_leiden(
        objective_function="CPM",
        resolution=resolution,
        weights=weights,
        n_iterations=config.LEIDEN_ITERATIONS,
    )
    g.vs["community_id"] = partition.membership


# ---------------------------------------------------------------------------
# Cluster merging
# ---------------------------------------------------------------------------

def _build_adjacency(
    g: ig.Graph,
    cluster_nodes: dict[int, list[int]],
) -> dict[int, dict[int, int]]:
    """Build a cluster-level adjacency dict from a paper-level graph.

    Returns {cluster_id: {neighbour_cluster_id: weight_sum}} so that
    neighbours of a cluster can be looked up in O(degree) rather than by
    scanning all inter-cluster edges.

    Args:
        g:             Paper-level igraph with community_id vertex attribute set.
        cluster_nodes: Mapping of cluster_id → list of vertex indices.

    Returns:
        Symmetric adjacency dict (both directions stored). Values are summed
        edge weights (float), not raw edge counts.
    """
    v_to_cluster = {}
    for cid, nodes in cluster_nodes.items():
        for v in nodes:
            v_to_cluster[v] = cid

    adj: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    has_weights = "weight" in g.es.attributes()
    for e in g.es:
        ca = v_to_cluster.get(e.source)
        cb = v_to_cluster.get(e.target)
        if ca is not None and cb is not None and ca != cb:
            w = e["weight"] if has_weights else 1.0
            adj[ca][cb] += w
            adj[cb][ca] += w

    return {k: dict(v) for k, v in adj.items()}


def merge_to_target(g: ig.Graph, target_n: int) -> int:
    """Merge communities down to *target_n* using inter-cluster edge density.

    At each step the smallest community is merged into whichever neighbour
    maximises the relatedness score:

        score(A, B) = edges(A, B) / (|A| × |B|)

    This is the inter-cluster edge density — the same metric as the CPM
    resolution threshold, keeping the merging criterion consistent with
    the clustering objective.

    Communities with no edges to any other community are removed entirely
    rather than arbitrarily absorbed. These papers are citation islands that
    cannot be meaningfully placed on the map.

    Complexity: O(n_merges × degree) for the merge loop, where degree is the
    average number of inter-cluster neighbours. The adjacency dict is built
    once and updated incrementally — no re-scanning of graph edges after init.
    A min-heap (lazy deletion) provides O(log n) access to the smallest cluster.

    After completion, community_id is rewritten on all vertices, numbered
    0-based in descending size order (0 = largest community).

    Args:
        g:        Paper-level igraph with community_id vertex attribute set.
        target_n: Desired number of communities after merging.

    Returns:
        Number of papers removed because their community had no inter-cluster
        edges at the time of merging.
    """
    if target_n < 1:
        return 0

    cluster_nodes: dict[int, list[int]] = defaultdict(list)
    for v in g.vs:
        cluster_nodes[v["community_id"]].append(v.index)
    cluster_nodes = dict(cluster_nodes)

    if len(cluster_nodes) <= target_n:
        return 0

    adj = _build_adjacency(g, cluster_nodes)

    # Min-heap of (size, cluster_id).
    # Lazy deletion: stale entries (wrong size or deleted cluster) are skipped on pop.
    heap = [(len(nodes), cid) for cid, nodes in cluster_nodes.items()]
    heapq.heapify(heap)

    removed_papers = 0

    while len(cluster_nodes) > target_n:
        # Pop smallest valid cluster — O(log n), stale entries discarded
        while True:
            sz_a, a = heapq.heappop(heap)
            if a in cluster_nodes and len(cluster_nodes[a]) == sz_a:
                break

        # No neighbours: this community is a citation island — remove it
        if not adj.get(a):
            removed_papers += len(cluster_nodes.pop(a))
            adj.pop(a, None)
            continue

        # Find best merge target — O(degree(a))
        best_score = -1.0
        best_b = None
        for b, edges in adj[a].items():
            if b not in cluster_nodes:
                continue
            score = edges / (sz_a * len(cluster_nodes[b]))
            if score > best_score:
                best_score = score
                best_b = b

        if best_b is None:
            # All neighbours were already removed — treat as isolated
            removed_papers += len(cluster_nodes.pop(a))
            adj.pop(a, None)
            continue

        # Merge a into best_b — update adjacency dict incrementally
        # For each neighbour c of a (excluding best_b): add a's edges to best_b's entry
        for c, count in adj.pop(a, {}).items():
            if c == best_b or c not in cluster_nodes:
                continue
            adj[best_b][c] = adj[best_b].get(c, 0) + count
            adj[c][best_b] = adj[c].get(best_b, 0) + count
            adj[c].pop(a, None)

        adj[best_b].pop(a, None)
        cluster_nodes[best_b].extend(cluster_nodes.pop(a))

        # Push best_b with its new size — old heap entry becomes stale
        heapq.heappush(heap, (len(cluster_nodes[best_b]), best_b))

    # Renumber community IDs 0-based, largest community = 0
    rank = {cid: rank for rank, cid in
            enumerate(sorted(cluster_nodes, key=lambda c: len(cluster_nodes[c]), reverse=True))}
    for cid, nodes in cluster_nodes.items():
        for v_idx in nodes:
            g.vs[v_idx]["community_id"] = rank[cid]

    return removed_papers


# ---------------------------------------------------------------------------
# Cluster-level graph
# ---------------------------------------------------------------------------

def build_cluster_graph(
    g: ig.Graph,
    metadata: dict[str, dict],
) -> tuple["ig.Graph", dict[int, dict]]:
    """Collapse the paper-level graph into a cluster-level graph for visualisation.

    Each vertex in the output represents one community. Edge weight between
    two clusters equals the sum of citation edge weights between their member papers.

    Display metadata (titles, nouns, label) is returned separately in a labels dict
    keyed by community_id, keeping the graph object free of display concerns.

    Vertex attributes:
        community_id (int) — cluster index (matches g.vs["community_id"])
        size         (int) — number of member papers

    Edge attributes:
        weight (float) — summed cross-cluster citation weight

    Labels dict values:
        label      (str)  — "Cluster N" initially; overwritten by label_clusters()
        top_titles (list) — top-10 titles by intra-cluster degree

    Args:
        g:        Paper-level igraph with community_id and degree vertex attributes.
        metadata: Dict mapping paper_id → {"title": str, ...}.

    Returns:
        (cg, labels) tuple.
    """
    membership = g.vs["community_id"]
    n_clusters = max(membership) + 1

    cluster_sizes = [0] * n_clusters
    for c in membership:
        cluster_sizes[c] += 1

    intra_degree = [0.0] * g.vcount()
    edge_weights: dict[tuple[int, int], float] = {}
    has_weights = "weight" in g.es.attributes()
    for e in g.es:
        src_c = g.vs[e.source]["community_id"]
        dst_c = g.vs[e.target]["community_id"]
        w = e["weight"] if has_weights else 1.0
        if src_c == dst_c:
            intra_degree[e.source] += w
            intra_degree[e.target] += w
        else:
            key = (min(src_c, dst_c), max(src_c, dst_c))
            edge_weights[key] = edge_weights.get(key, 0.0) + w

    # Collect top 10 papers per cluster by intra-cluster degree
    cluster_papers: list[list[tuple[float, str]]] = [[] for _ in range(n_clusters)]
    for v in g.vs:
        c = v["community_id"]
        title = metadata.get(v["name"], {}).get("title", "") or ""
        cluster_papers[c].append((intra_degree[v.index], title))

    labels: dict[int, dict] = {}
    for cid in range(n_clusters):
        papers = sorted(cluster_papers[cid], key=lambda x: x[0], reverse=True)
        labels[cid] = {
            "label":      f"Cluster {cid}",
            "top_titles": [t for _, t in papers[:10]],
            "top_nouns":  [],
        }

    cg = ig.Graph(n=n_clusters, directed=False)
    cg.vs["community_id"] = list(range(n_clusters))
    cg.vs["size"] = cluster_sizes

    if edge_weights:
        cg.add_edges(list(edge_weights.keys()))
        cg.es["weight"] = list(edge_weights.values())

    return cg, labels


def top_clusters_by_size(
    cg: "ig.Graph",
    labels: dict[int, dict],
    n: int = 10,
    top_docs: int = 3,
    top_nouns: int = 5,
) -> list[dict]:
    """Return the *n* largest clusters as a list of dicts for table display.

    Each dict contains: c. (int), size (int), top documents (str), top nouns (str).

    Args:
        cg:       Cluster-level igraph with community_id and size vertex attributes.
        labels:   Labels dict as returned by build_cluster_graph / label_clusters.
        n:        Number of clusters to return (largest first).
        top_docs: Number of top document titles to include per cluster.
    """
    top = sorted(cg.vs, key=lambda v: v["size"], reverse=True)[:n]
    rows = []
    for v in top:
        cid  = v["community_id"]
        info = labels.get(cid, {})
        titles = info.get("top_titles", [])
        nouns  = info.get("top_nouns",  [])
        rows.append({
            "c.":           cid,
            "size":         v["size"],
            "top documents": "\n".join(f"{i+1}. {t}" for i, t in enumerate(titles[:top_docs])),
            "top nouns":    "\n".join(nouns[:top_nouns]),
        })
    return rows
