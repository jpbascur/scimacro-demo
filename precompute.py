"""Pre-compute and cache graph, abstracts, and nouns for each demo cluster.

Reads cluster definitions from data/demo_clusters.json and fetches data from
BigQuery. Produces three files per cluster in data/:

    cache.{id}.graph.pkl        — (base_graph, metadata) where metadata has titles only
    cache.{id}.abstracts.pkl.gz — compressed dict: paper_id → abstract string
    cache.{id}.nouns.pkl        — dict: paper_id → set of noun lemmas

The cluster list is loaded at runtime from demo_clusters.json, so adding new
clusters there is enough — no changes to this script are needed.

Run with:
    python precompute.py                    # all clusters, 5 parallel workers
    python precompute.py --clusters 102 9   # specific clusters only
    python precompute.py --workers 3        # override worker count
"""
import gzip
import json
import multiprocessing
import os
import pickle
import sys
import time

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLUSTERS_FILE = os.path.join(DATA_DIR, "demo_clusters.json")

BQ_PROJECT  = "scimacro-demo"
BQ_DATASET  = "cwts-leiden.openalex_2023nov_classification"
BQ_CITATIONS_TABLE = "cwts-leiden.openalex_2025aug.citation"
BQ_TITLES_TABLE    = "cwts-leiden.openalex_2025aug.work_title"
BQ_ABSTRACTS_TABLE = "cwts-leiden.openalex_2025aug.work_abstract"
BQ_CLUSTERING_TABLE = "cwts-leiden.openalex_2023nov_classification.clustering"

DEFAULT_WORKERS = 5


# ---------------------------------------------------------------------------
# BigQuery helpers
# ---------------------------------------------------------------------------

def _fetch_papers(client, cluster_id: int) -> dict[str, dict]:
    """Return metadata dict for all papers in cluster_id.

    Keys: paper_id (str)
    Values: {"title": str, "abstract": str}
    """
    sql = f"""
        SELECT a.work_id, IFNULL(b.title, '') AS title, IFNULL(c.abstract, '') AS abstract
        FROM `{BQ_CLUSTERING_TABLE}` AS a
        LEFT JOIN `{BQ_TITLES_TABLE}`    AS b ON a.work_id = b.work_id
        LEFT JOIN `{BQ_ABSTRACTS_TABLE}` AS c ON a.work_id = c.work_id
        WHERE a.micro_cluster_id = {cluster_id}
    """
    rows = client.query(sql).result()
    return {
        str(row.work_id): {"title": row.title or "", "abstract": row.abstract or ""}
        for row in rows
    }


def _fetch_edges(client, cluster_id: int) -> list[tuple]:
    """Return weighted edge list for all intra-cluster citations."""
    sql = f"""
        WITH cluster_papers AS (
            SELECT work_id
            FROM `{BQ_CLUSTERING_TABLE}`
            WHERE micro_cluster_id = {cluster_id}
        )
        SELECT DISTINCT
            a.citing_work_id,
            a.cited_work_id,
            1 AS weight
        FROM `{BQ_CITATIONS_TABLE}` AS a
        JOIN cluster_papers AS b ON a.citing_work_id = b.work_id
        JOIN cluster_papers AS c ON a.cited_work_id  = c.work_id
    """
    rows = client.query(sql).result()
    return [(str(row.citing_work_id), str(row.cited_work_id), float(row.weight))
            for row in rows]


# ---------------------------------------------------------------------------
# Per-cluster precomputation (runs in a worker process)
# ---------------------------------------------------------------------------

def precompute_cluster(cluster_id: int) -> None:
    """Fetch, build, and cache all artefacts for one cluster."""
    from google.cloud import bigquery
    from graph.builder import build_base_graph
    from graph.labeler import extract_nouns

    graph_path     = os.path.join(DATA_DIR, f"cache.{cluster_id}.graph.pkl")
    abstracts_path = os.path.join(DATA_DIR, f"cache.{cluster_id}.abstracts.pkl.gz")
    nouns_path     = os.path.join(DATA_DIR, f"cache.{cluster_id}.nouns.pkl")

    prefix = f"[cluster {cluster_id}]"
    print(f"{prefix} Starting")

    client = bigquery.Client(project=BQ_PROJECT)

    # --- Papers ---
    t0 = time.time()
    print(f"{prefix} Fetching papers from BigQuery...")
    metadata = _fetch_papers(client, cluster_id)
    print(f"{prefix}   {len(metadata):,} papers  ({time.time()-t0:.1f}s)")

    # --- Edges ---
    t0 = time.time()
    print(f"{prefix} Fetching edges from BigQuery...")
    edges = _fetch_edges(client, cluster_id)
    print(f"{prefix}   {len(edges):,} edges  ({time.time()-t0:.1f}s)")

    # --- Graph (titles only) ---
    t0 = time.time()
    print(f"{prefix} Building graph...")
    graph = build_base_graph(edges)
    titles_only = {pid: {"title": v["title"]} for pid, v in metadata.items()}
    with open(graph_path, "wb") as f:
        pickle.dump((graph, titles_only), f)
    print(f"{prefix}   {graph.vcount():,} nodes, {graph.ecount():,} edges  ({time.time()-t0:.1f}s)")

    # --- Abstracts (compressed) ---
    t0 = time.time()
    print(f"{prefix} Saving abstracts (compressed)...")
    abstracts = {pid: v["abstract"] for pid, v in metadata.items()}
    with gzip.open(abstracts_path, "wb") as f:
        pickle.dump(abstracts, f)
    print(f"{prefix}   Done  ({time.time()-t0:.1f}s)")

    # --- Nouns ---
    t0 = time.time()
    print(f"{prefix} Extracting nouns (spaCy)...")
    nouns = extract_nouns(metadata)
    with open(nouns_path, "wb") as f:
        pickle.dump(nouns, f)
    print(f"{prefix}   {len(nouns):,} papers processed  ({time.time()-t0:.1f}s)")

    print(f"{prefix} Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with open(CLUSTERS_FILE) as f:
        all_clusters = json.load(f)
    all_ids = [c["id"] for c in all_clusters]

    # Parse --clusters and --workers from argv
    cluster_ids = all_ids
    n_workers = DEFAULT_WORKERS

    args = sys.argv[1:]
    if "--clusters" in args:
        idx = args.index("--clusters")
        cluster_ids = []
        for token in args[idx + 1:]:
            if token.startswith("--"):
                break
            try:
                cluster_ids.append(int(token))
            except ValueError:
                print(f"Invalid cluster id: {token}")
                sys.exit(1)
    if "--workers" in args:
        idx = args.index("--workers")
        n_workers = int(args[idx + 1])

    print(f"Precomputing {len(cluster_ids)} clusters with {n_workers} workers")
    print(f"Clusters: {cluster_ids}\n")

    with multiprocessing.Pool(processes=n_workers) as pool:
        pool.map(precompute_cluster, cluster_ids)

    print("\nAll clusters done.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
