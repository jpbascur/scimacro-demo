"""BigQuery implementation of DataSource.

Requires the google-cloud-bigquery package and valid credentials. Credentials are
resolved in this order by the Google client library:
    1. GOOGLE_APPLICATION_CREDENTIALS environment variable (service account JSON key)
    2. gcloud application-default credentials (``gcloud auth application-default login``)
    3. Workload identity / metadata server (on GCP compute)

Set BQ_PROJECT, BQ_DATASET, and BQ_CITATIONS_TABLE in config.py before use.

Note: This backend is not used by the demo app, which reads from local CSV files via
LocalSource. It is provided as a reference implementation showing how to swap in a
live BigQuery backend by replacing LocalSource with BigQuerySource.
"""
from __future__ import annotations

from google.cloud import bigquery

import config
from data.base import DataSource


class BigQuerySource(DataSource):
    """DataSource backed by a Google BigQuery citations table.

    Implements BFS citation traversal up to a configurable depth and a simple
    title-based paper search. Both methods issue SQL queries against BigQuery.
    """

    def __init__(self):
        """Create a BigQuery client and cache table/column identifiers from config."""
        self.client = bigquery.Client(project=config.BQ_PROJECT)
        self.table = (
            f"`{config.BQ_PROJECT}.{config.BQ_DATASET}.{config.BQ_CITATIONS_TABLE}`"
        )
        self.col_citing = config.COL_CITING
        self.col_cited = config.COL_CITED

    # ------------------------------------------------------------------
    # DataSource interface
    # ------------------------------------------------------------------

    def get_citations(self, paper_ids: list[str], depth: int) -> list[tuple[str, str]]:
        """Expand citations from *paper_ids* up to *depth* hops via BFS.

        At each hop, all edges incident on the current frontier (both citing and cited
        directions) are fetched in a single query and the newly discovered papers become
        the next frontier. Stops early if the total node count exceeds config.MAX_NODES.

        Args:
            paper_ids: Seed paper IDs to start expansion from.
            depth:     Maximum number of BFS hops. depth=1 returns direct neighbours only.

        Returns:
            Deduplicated list of (citing_id, cited_id) tuples reachable within *depth*
            hops. May contain more edges than nodes if the same edge appears in multiple
            hops (BigQuery DISTINCT handles this per-query but not across hops).
        """
        if not paper_ids:
            return []

        visited: set[str] = set(paper_ids)
        frontier: set[str] = set(paper_ids)
        all_edges: list[tuple[str, str]] = []

        for _ in range(depth):
            if not frontier or len(visited) >= config.MAX_NODES:
                break

            ids_sql = ", ".join(f"'{pid}'" for pid in frontier)
            query = f"""
                SELECT DISTINCT {self.col_citing} AS citing, {self.col_cited} AS cited
                FROM {self.table}
                WHERE {self.col_citing} IN ({ids_sql})
                   OR {self.col_cited} IN ({ids_sql})
                LIMIT {config.MAX_NODES}
            """
            rows = self.client.query(query).result()

            new_frontier: set[str] = set()
            for row in rows:
                citing, cited = str(row.citing), str(row.cited)
                all_edges.append((citing, cited))
                for node in (citing, cited):
                    if node not in visited:
                        visited.add(node)
                        new_frontier.add(node)

            frontier = new_frontier

        return all_edges

    def search_papers(self, query: str, limit: int = 20) -> list[dict]:
        """Search for papers by title substring in the BigQuery papers table.

        Assumes a table named ``papers`` in the same project/dataset as the citations
        table, with at least ``id``, ``title``, and ``year`` columns. If that table
        does not exist or the query fails, falls back to returning the raw query string
        as a single stub result (useful for direct ID lookups).

        Args:
            query: Substring to search for in the title column (case-insensitive).
            limit: Maximum number of results to return.

        Returns:
            List of dicts with keys "id" (str), "title" (str), "year" (int or None).
            Falls back to [{"id": query, "title": query, "year": None}] on error.
        """
        papers_table = (
            f"`{config.BQ_PROJECT}.{config.BQ_DATASET}.papers`"
        )
        sql = f"""
            SELECT id, title, IFNULL(year, 0) AS year
            FROM {papers_table}
            WHERE LOWER(title) LIKE LOWER('%{query}%')
            LIMIT {limit}
        """
        try:
            rows = self.client.query(sql).result()
            return [{"id": r.id, "title": r.title, "year": r.year} for r in rows]
        except Exception:
            # If there is no papers table, treat the query as a direct ID lookup
            return [{"id": query.strip(), "title": query.strip(), "year": None}]
