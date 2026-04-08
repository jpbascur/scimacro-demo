"""Run once to export citations and papers from BigQuery to local Parquet files."""
from google.cloud import bigquery

# ── Configure these ──────────────────────────────────────────────────────────
PROJECT   = "your-gcp-project"
DATASET   = "your_dataset"

CITATIONS_QUERY = f"""
    SELECT citing_id, cited_id
    FROM `{PROJECT}.{DATASET}.citations`
"""

PAPERS_QUERY = f"""
    SELECT paper_id, title, year
    FROM `{PROJECT}.{DATASET}.papers`
"""

OUTPUT_CITATIONS = "citations.parquet"
OUTPUT_PAPERS    = "papers.parquet"
# ─────────────────────────────────────────────────────────────────────────────

client = bigquery.Client(project=PROJECT)

print("Exporting citations…")
df = client.query(CITATIONS_QUERY).to_dataframe()
df.to_parquet(OUTPUT_CITATIONS, index=False)
print(f"  {len(df):,} rows → {OUTPUT_CITATIONS}")

print("Exporting papers…")
df = client.query(PAPERS_QUERY).to_dataframe()
df.to_parquet(OUTPUT_PAPERS, index=False)
print(f"  {len(df):,} rows → {OUTPUT_PAPERS}")

print("Done.")
