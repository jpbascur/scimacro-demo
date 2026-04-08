"""Bubble chart layout algorithm.

Places clusters as non-overlapping circles on a 2-D canvas by greedily
inserting nodes one at a time at circle-circle tangent positions, choosing
the position that minimises the local stress contribution of the inserted node.

Algorithm overview
------------------
1. build_node_list  — convert a cluster-level igraph into the flat node format.
2. build_graph      — try one or more starting nodes; return the lowest-stress layout.
3. _build_from_start — single layout attempt starting with a chosen first node.
4. _order_node_list — greedy insertion order: always place the node with the most
                      total edge weight to already-placed nodes next.
5. _get_coordinates — for the node being inserted, enumerate all candidate positions
                      (circle-circle intersections of already-placed nodes) and pick
                      the one with the lowest sum(edge_weight × distance²) to placed nodes.

Stress metric: Σ over all unordered node pairs of  edge_weight(a, b) × dist²(a, b).
No physics simulation is used — the result is deterministic.

Each node dict must contain:
    id     (any hashable) — unique node identifier
    radius (float)       — circle radius
    edges  (dict)        — {other_id: float weight} for every other node in the graph

After layout, each node dict also contains:
    coor   (tuple[float, float]) — (x, y) centre position
    stress (float)               — local stress at placement time (informational)
"""
import copy
import math

ERROR_MARGIN = 0.00001


def build_node_list(cg) -> list[dict]:
    """Convert a cluster-level igraph into the node list format the layout expects.

    Radius is sqrt(size) so that bubble area is proportional to cluster size.
    Edge weight between two clusters is the cross-cluster citation count; pairs
    with no direct citations get weight 0 (they are still in each node's edge dict).

    Args:
        cg: Cluster-level igraph produced by build_cluster_graph(). Expected vertex
            attributes: community_id (int), size (int). Expected edge attribute:
            weight (int).

    Returns:
        List of node dicts, one per cluster, each with:
            id     (int)   — community_id
            radius (float) — sqrt(size)
            edges  (dict)  — {other_community_id: weight} for every other cluster
    """
    import igraph as ig

    ids = [v["community_id"] for v in cg.vs]
    size_map = {v["community_id"]: v["size"] for v in cg.vs}

    # Edge weights keyed by (min_id, max_id)
    weight_map: dict[tuple[int, int], float] = {}
    for e in cg.es:
        src = cg.vs[e.source]["community_id"]
        dst = cg.vs[e.target]["community_id"]
        weight_map[(min(src, dst), max(src, dst))] = e["weight"]

    nodes = []
    for cid in ids:
        edges = {
            other: weight_map.get((min(cid, other), max(cid, other)), 0)
            for other in ids if other != cid
        }
        nodes.append({
            "id": cid,
            "radius": math.sqrt(size_map[cid]),
            "edges": edges,
        })
    return nodes


def build_graph(node_list: list[dict], max_starts: int | None = None) -> list[dict]:
    """Try one or more starting nodes and return the layout with the lowest total stress.

    The layout quality depends on which node is placed first, because the greedy
    insertion order and candidate positions are all anchored to the first node.
    Trying all possible starting nodes gives a quality guarantee at the cost of O(n³)
    time. Restricting to a subset trades quality for speed.

    Starting nodes are tried in descending order of total edge weight (most-connected
    first) because highly-connected nodes have the most influence on stress.

    Args:
        node_list:  List of node dicts as returned by build_node_list().
        max_starts: Maximum number of starting nodes to evaluate.
                    None  → try all n nodes; best quality, O(n³) time.
                    int k → try the k most-connected nodes; O(k × n²) time.

    Returns:
        Placed node list (each node has a ``coor`` key added) for the lowest-stress
        layout found, or None if every attempt failed to place all nodes.
    """
    # Sort by total edge weight descending — most connected node first
    ordered_starts = sorted(
        range(len(node_list)),
        key=lambda i: sum(node_list[i]["edges"].values()),
        reverse=True,
    )
    attempts = ordered_starts if max_starts is None else ordered_starts[:max_starts]

    best = None
    best_stress = float("inf")
    for i in attempts:
        candidate = _build_from_start(i, node_list)
        if candidate is None:
            continue
        s = _graph_stress(candidate)
        if s < best_stress:
            best_stress = s
            best = candidate
    return best


def _build_from_start(start_idx: int, node_list: list[dict]) -> list[dict] | None:
    """Run one complete layout attempt using *start_idx* as the first placed node.

    The first node is centred at the origin. The second node is placed to its right
    at distance r0 + r1 (tangent). Remaining nodes are placed by _get_coordinates().

    Args:
        start_idx: Index into node_list of the node to place first.
        node_list: List of node dicts (id, radius, edges).

    Returns:
        Placed node list with coor set for every node, or None if any node could
        not be placed (no valid non-overlapping tangent position was found).
    """
    nodes = copy.deepcopy(node_list)
    x_i = nodes[start_idx]
    ordered = _order_node_list(x_i, nodes)

    ordered[0]["coor"] = (0.0, 0.0)
    placed = [ordered[0]]

    r0, r1 = ordered[0]["radius"], ordered[1]["radius"]
    ordered[1]["coor"] = (r0 + r1, 0.0)
    placed.append(ordered[1])

    for node in ordered[2:]:
        result = _get_coordinates(node, placed)
        if result is None:
            return None
        placed.append(result)

    return placed


def _order_node_list(x_i: dict, node_list: list[dict]) -> list[dict]:
    """Return node_list reordered for greedy insertion starting from *x_i*.

    At each step, the unplaced node with the highest total edge weight to already-
    placed nodes is chosen next. This heuristic tends to place strongly-connected
    nodes early, reducing stress for the most important edges.

    Args:
        x_i:       The node to place first (must be in node_list).
        node_list: Full list of node dicts.

    Returns:
        Reordered list with x_i first and remaining nodes in greedy order.
    """
    ordered = [x_i]
    remaining = [n for n in node_list if n["id"] != x_i["id"]]
    while remaining:
        best = max(remaining, key=lambda n: _score(n, ordered))
        ordered.append(best)
        remaining.remove(best)
    return ordered


def _score(node: dict, placed: list[dict]) -> float:
    """Return the total edge weight from *node* to all already-placed nodes.

    Used by _order_node_list to rank unplaced nodes: higher score = stronger
    connection to the current placed set = more important to place next.
    """
    return sum(p["edges"][node["id"]] for p in placed if p["id"] != node["id"])


def _get_coordinates(node: dict, placed: list[dict]) -> dict | None:
    """Find the best non-overlapping position for *node* among already-placed nodes.

    Candidate positions are the intersection points of the two tangent circles
    for every pair of already-placed nodes (i.e., the points where *node* would
    simultaneously touch both circles in the pair). A small ERROR_MARGIN is added
    to the radii to avoid floating-point tangency detection failures.

    Among all valid (non-overlapping) candidates, the one with the lowest local
    stress — sum(edge_weight(node, p) × dist²(node, p)) over placed nodes — is chosen.

    Args:
        node:   Node dict to place (id, radius, edges). coor will be set in-place.
        placed: Already-placed node dicts (each has coor).

    Returns:
        The node dict with coor set to the best candidate position, or None if no
        valid non-overlapping position was found among all candidate pairs.
    """
    candidates = []
    for m, n in _pairs(placed):
        tm = {"coor": m["coor"], "radius": node["radius"] + m["radius"] + ERROR_MARGIN}
        tn = {"coor": n["coor"], "radius": node["radius"] + n["radius"] + ERROR_MARGIN}
        if not _overlaps(tm, tn):
            continue
        for coor in _circle_intersections(tm, tn):
            candidate = {**node, "coor": coor}
            if all(not _overlaps(candidate, p) for p in placed):
                candidate["stress"] = _node_stress(candidate, placed)
                candidates.append(candidate)

    if not candidates:
        return None

    best_coor = min(candidates, key=lambda c: c["stress"])["coor"]
    node["coor"] = best_coor
    return node


def _node_stress(node: dict, placed: list[dict]) -> float:
    """Return the local stress of *node* relative to already-placed nodes.

    Local stress = Σ edge_weight(node, p) × dist²(node.coor, p.coor) for all p in placed.
    Used by _get_coordinates to rank candidate positions.
    """
    return sum(
        node["edges"][p["id"]] * _dist2(node["coor"], p["coor"])
        for p in placed if p["id"] != node["id"]
    )


def _graph_stress(nodes: list[dict]) -> float:
    """Return the total layout stress across all pairs of nodes.

    Global stress = Σ over all unordered pairs (a, b) of  edge_weight(a, b) × dist²(a, b).
    Used by build_graph to compare layouts produced by different starting nodes.
    """
    total = 0.0
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            total += a["edges"][b["id"]] * _dist2(a["coor"], b["coor"])
    return total


def _pairs(nodes: list[dict]):
    """Yield all unordered pairs (a, b) from *nodes* without repetition."""
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            yield a, b


def _dist2(c1: tuple, c2: tuple) -> float:
    """Return the squared Euclidean distance between two (x, y) points."""
    return (c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2


def _overlaps(a: dict, b: dict) -> bool:
    """Return True if circle *a* and circle *b* overlap (centres closer than sum of radii)."""
    return (a["radius"] + b["radius"]) ** 2 > _dist2(a["coor"], b["coor"])


def _circle_intersections(a: dict, b: dict) -> list[tuple]:
    """Return the (up to two) intersection points of circles *a* and *b*.

    Each circle dict must have ``coor`` (x, y) and ``radius`` keys. When the
    circles do not intersect (discriminant < 0), an empty list is returned.
    When they are tangent or nearly so, a single repeated point is returned as
    two entries (the caller deduplicates naturally via stress comparison).

    This is the standard algebraic formula for circle-circle intersection.

    Args:
        a: Circle dict with coor and radius.
        b: Circle dict with coor and radius.

    Returns:
        List of 0 or 2 (x, y) tuples.
    """
    r1, (x1, y1) = a["radius"], a["coor"]
    r2, (x2, y2) = b["radius"], b["coor"]
    R2 = _dist2((x1, y1), (x2, y2))
    val = 2 * (r1 ** 2 + r2 ** 2) / R2 - (r1 ** 2 - r2 ** 2) ** 2 / R2 ** 2 - 1
    if val < 0:
        return []
    a_ = (r1 ** 2 - r2 ** 2) / (2 * R2)
    b_ = math.sqrt(val)
    fx = (x1 + x2) / 2 + a_ * (x2 - x1)
    fy = (y1 + y2) / 2 + a_ * (y2 - y1)
    gx = b_ * (y2 - y1) / 2
    gy = b_ * (x1 - x2) / 2
    return [(fx + gx, fy + gy), (fx - gx, fy - gy)]
