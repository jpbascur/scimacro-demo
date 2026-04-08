"""Central configuration for the Science Map demo.

All file paths, column names, and algorithm parameters live here so that the rest
of the codebase never contains magic strings or numbers. To adapt this demo to a
different dataset, only this file should need to change.

BigQuery parameters (BQ_*) are used only by BigQuerySource and are not needed when
running against local CSV files via LocalSource.
"""
import os

# --- Local data files ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CITATIONS_FILE = os.path.join(DATA_DIR, "scimacro-demo.cache.citing_cited_MICROCLUSTERID102.csv")
PAPERS_FILE    = os.path.join(DATA_DIR, "scimacro-demo.cache.workid_title_abstract_MICROCLUSTERID102.csv")

# Column names
COL_CITING  = "citing_work_id"
COL_CITED   = "cited_work_id"
COL_WEIGHT  = "weight"          # optional — set to None if not present in the CSV
COL_PAPER_ID = "work_id"
COL_TITLE    = "title"
COL_ABSTRACT = "abstract"

# --- Graph / clustering ---
DEFAULT_RESOLUTION = 1e-5
MAX_RENDERABLE_CLUSTERS = 300   # warn above this
LEIDEN_SEED = 42
LEIDEN_ITERATIONS = 5
