"""Abstract data source interface.

Any backend (BigQuery, OpenAlex, CSV, etc.) must implement DataSource.
This keeps the graph logic completely decoupled from where data comes from.
"""
from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def get_citations(self, paper_ids: list[str], depth: int) -> list[tuple[str, str]]:
        """Return a list of (citing_id, cited_id) edges reachable from
        *paper_ids* within *depth* hops.

        Args:
            paper_ids: Seed paper IDs to start traversal from.
            depth: Number of citation hops to expand.

        Returns:
            List of directed edges as (citing_id, cited_id) tuples.
        """

    @abstractmethod
    def search_papers(self, query: str, limit: int = 20) -> list[dict]:
        """Search for papers matching *query*.

        Returns a list of dicts, each with at least:
            - id:    unique paper identifier
            - title: paper title (str)
        Additional fields (year, authors, doi, …) are optional but pass through.
        """
