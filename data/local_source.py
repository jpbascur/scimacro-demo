"""Local CSV data source using DuckDB.

Reads two CSV files — one for citation edges, one for paper metadata — as in-memory
DuckDB views. DuckDB's read_csv_auto() infers schema automatically, so no manual
column-type declarations are needed. The connection is kept open for the lifetime of
the LocalSource instance (typically cached by Streamlit).

This is the recommended backend for the demo. To switch to a different backend (e.g.,
BigQuery), replace LocalSource with an implementation of DataSource.
"""
import duckdb
import config
from data.base import DataSource


class LocalSource(DataSource):
    """DataSource backed by local CSV files queried via an in-memory DuckDB connection.

    On construction, two DuckDB views are created:
        citations — all citation edges (citing_work_id, cited_work_id columns)
        papers    — paper metadata (work_id, title, … columns)

    Column names are read from config so the CSV schema can be changed without
    touching this class.
    """

    def __init__(self):
        """Open an in-memory DuckDB connection and register the two CSV files as views."""
        self.db = duckdb.connect()
        self.db.execute(f"""
            CREATE VIEW citations AS SELECT * FROM read_csv_auto('{config.CITATIONS_FILE}');
            CREATE VIEW papers    AS SELECT * FROM read_csv_auto('{config.PAPERS_FILE}');
        """)

    def get_all_edges(self) -> list[tuple]:
        """Return every edge as (u, v) or (u, v, weight) tuples.

        Returns weighted tuples when COL_WEIGHT is set and the column exists.
        Both IDs are coerced to str; weight is returned as float.
        """
        if config.COL_WEIGHT:
            rows = self.db.execute(
                f"SELECT {config.COL_CITING}, {config.COL_CITED}, {config.COL_WEIGHT} FROM citations"
            ).fetchall()
            return [(str(r[0]), str(r[1]), float(r[2])) for r in rows]
        rows = self.db.execute(
            f"SELECT {config.COL_CITING}, {config.COL_CITED} FROM citations"
        ).fetchall()
        return [(str(r[0]), str(r[1])) for r in rows]

    def get_paper_metadata(self) -> dict[str, dict]:
        """Return paper metadata keyed by paper ID.

        Currently only the title is loaded. Additional columns (year, abstract, …)
        can be added here and will be available as metadata[paper_id]["key"].

        Returns:
            Dict mapping paper_id (str) → {"title": str}.
        """
        rows = self.db.execute(
            f"SELECT {config.COL_PAPER_ID}, {config.COL_TITLE}, {config.COL_ABSTRACT} FROM papers"
        ).fetchall()
        return {str(r[0]): {"title": r[1], "abstract": r[2]} for r in rows}

    # --- DataSource interface (unused locally but keeps the contract) ---

    def get_citations(self, paper_ids, depth):
        """DataSource contract method — returns all edges regardless of seed or depth.

        The local dataset is small enough to load in full. A real implementation
        would do BFS from paper_ids up to depth hops.
        """
        return self.get_all_edges()

    def search_papers(self, query, limit=20):
        """Return papers whose title contains *query* (case-insensitive substring match).

        Args:
            query: Substring to search for in the title column.
            limit: Maximum number of results to return.

        Returns:
            List of dicts with keys "id" (str) and "title" (str).
        """
        rows = self.db.execute(f"""
            SELECT {config.COL_PAPER_ID}, {config.COL_TITLE}
            FROM papers
            WHERE lower({config.COL_TITLE}) LIKE lower('%{query}%')
            LIMIT {limit}
        """).fetchall()
        return [{"id": str(r[0]), "title": r[1]} for r in rows]
