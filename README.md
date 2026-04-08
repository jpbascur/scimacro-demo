# SciMacro — Science Macroscope

An interactive tool for exploring the structure of scientific literature. SciMacro takes a large collection of research papers connected by citation links and groups them into thematic clusters, then renders the result as a navigable science map.

**Live demo:** https://scimacro-demo-35001728336.us-central1.run.app

---

## What it does

Science does not advance in isolation — papers cite each other, ideas cross disciplinary boundaries, and subfields cluster around shared methods and problems. SciMacro makes this structure visible.

Given a citation network, the tool:

1. **Clusters papers** using the [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z), a community detection method that finds groups of papers more densely connected to each other than to the rest of the network.
2. **Labels each cluster** automatically by scoring noun phrases extracted from titles and abstracts. Terms that are common within a cluster but rare across the full map score highest, producing labels like *"CRISPR / gene editing / genome"* rather than generic words like *"study"* or *"method"*.
3. **Renders a science map** in two layouts:
   - **Bubble chart** — clusters as sized circles arranged to minimise overlap while preserving citation proximity. Cluster size reflects paper count.
   - **Force-directed graph** — a physics simulation that pulls strongly-connected clusters together and pushes weakly-connected ones apart.

Both views are interactive: hover over a cluster for its top documents and keywords, search for a noun to highlight which clusters use it most, or colour clusters by citation links to a specific cluster.

---

## Demo datasets

The demo includes 24 pre-loaded datasets drawn from the [Leiden Ranking Open Edition](https://open.leidenranking.com/) citation network (OpenAlex 2023). Each dataset is a micro-cluster — a coherent subfield of world science — spanning five broad fields:

| Field | Examples |
|---|---|
| Social sciences and humanities | Bibliometrics, Consciousness, Cultural Theory |
| Biomedical and health sciences | Alzheimer's, CRISPR, Cancer Immunotherapy |
| Physical sciences and engineering | Graphene, Superconductivity, Battery Technology |
| Life and earth sciences | Climate Change, Coral Reefs, Mars Exploration |
| Mathematics and computer science | NLP, Information Retrieval, Complex Networks |

Users can also **upload their own citation network** (CSV edges + document metadata) to cluster and explore any collection of documents.

---

## How cluster labels work

Each cluster is labelled using a term-scoring formula:

```
score(term, cluster) = n_cluster / (n_map + m)
```

where `n_cluster` is the number of papers in the cluster containing the term, `n_map` is the number of papers across the entire map containing the term, and `m` is a smoothing parameter (default 25). This rewards terms that are locally concentrated — appearing frequently in one cluster while being rare elsewhere — and suppresses generic academic vocabulary.

Noun phrases are extracted from titles and abstracts using [spaCy](https://spacy.io/) (`en_core_web_sm`). Users can adjust the smoothing parameter and regenerate labels interactively.

---

## Architecture

```
app.py                  — Streamlit UI, all session state and widget logic
config.py               — File paths, column names, algorithm parameters
precompute.py           — One-time cache generation from BigQuery
graph/
  builder.py            — igraph construction, Leiden clustering, cluster merging
  labeler.py            — spaCy noun extraction, cluster labeling, SPECTER2 embeddings
  visualizer.py         — Plotly bubble chart and PyVis force-directed renderer
  bubble_layout.py      — Stress-minimising layout algorithm for the bubble chart
data/
  demo_clusters.json    — Registry of demo datasets (id, label, field)
  bigquery_source.py    — BigQuery data backend (production)
  local_source.py       — CSV backend via DuckDB (development)
models/
  specter2_adapter/     — SPECTER2 adapter weights for paper embeddings
```

### SPECTER2 embeddings

[SPECTER2](https://huggingface.co/allenai/specter2) is a transformer model trained by the Allen Institute for AI specifically to embed scientific papers. It encodes a paper's title and abstract into a 768-dimensional vector that captures its semantic content in the context of scientific literature — papers on similar topics end up close in embedding space regardless of the specific words used.

In SciMacro, SPECTER2 embeddings are computed at precompute time and cached. They are available for downstream use (e.g. semantic similarity ranking) and are part of the precomputed artefacts stored per cluster.

---

## Running locally

### Requirements

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

For SPECTER2 embeddings (optional, only needed for precompute):
```
pip install torch transformers adapters
```

### With pre-built cache files

If you have the cache files (`.pkl`, `.pkl.gz`) in `data/`, just run:

```
streamlit run app.py
```

### Generating cache files from BigQuery

Edit the BigQuery project and table references in `precompute.py`, then:

```bash
python precompute.py                    # all clusters
python precompute.py --clusters 102 9   # specific clusters only
python precompute.py --workers 3        # control parallelism
```

This produces three files per cluster in `data/`:
- `cache.{id}.graph.pkl` — citation graph and paper titles
- `cache.{id}.abstracts.pkl.gz` — compressed abstracts
- `cache.{id}.nouns.pkl` — extracted noun index for labeling and search

---

## Deployment

The app is deployed on [Google Cloud Run](https://cloud.google.com/run) with cache files stored in Google Cloud Storage. Cache files are downloaded lazily on first use — each dataset is fetched from GCS the first time a user selects it, then kept in memory for subsequent requests.

Set the `GCS_CACHE_BUCKET` environment variable to point the app at your bucket:

```
GCS_CACHE_BUCKET=your-bucket-name
```

To rebuild and redeploy after code changes:

```bash
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT/REPO/app:latest
gcloud run deploy SERVICE --image ... --region us-central1
```

---

## Data source

Demo datasets are drawn from the [OpenAlex](https://openalex.org/) open catalogue of scientific works, clustered using the [Leiden Ranking Open Edition 2023](https://open.leidenranking.com/) micro-cluster assignments produced by the [Centre for Science and Technology Studies (CWTS)](https://www.cwts.nl/) at Leiden University.
