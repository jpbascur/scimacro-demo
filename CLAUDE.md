# Science Macroscope — Claude Instructions

## Project overview
Streamlit app that clusters scientific papers by citation network using the Leiden algorithm and visualises the result as an interactive science map. Source data is local CSV files; a BigQuery backend also exists for deployment.

## File map
- `app.py` — Streamlit entry point, all UI logic
- `config.py` — all magic strings/numbers (file paths, column names, algorithm params)
- `precompute.py` — one-time precomputation of graph and nouns
- `data/local_source.py` — CSV backend via DuckDB
- `data/bigquery_source.py` — BigQuery backend (deploy only)
- `graph/builder.py` — igraph construction, Leiden clustering, merging, cluster graph
- `graph/labeler.py` — spaCy noun extraction, cluster labeling
- `graph/visualizer.py` — PyVis (force-directed) and Plotly (bubble chart) renderers
- `graph/bubble_layout.py` — stress-minimising layout algorithm

## Cache files — NEVER delete without explicit user permission
`data/cache.graph.pkl`, `data/cache.nouns.pkl`, `data/cache.embeddings.pkl` are expensive to regenerate (minutes to hours). If a cache appears stale, warn the user and explain why — never delete or overwrite it unilaterally. Let the user decide.

## Key conventions
- All session state lives in `st.session_state` — never use module-level globals for UI state
- Clustering results stored in `st.session_state["cluster_result"]` so reruns from widgets don't clear the map
- Selected clusters stored in `st.session_state["selected_clusters"]` (a `set` of ints)
- Sidebar `on_click` callbacks are used for selection so state updates before the sidebar re-renders
- Color mode (`"search"` or `"connectivity"`) stored in `st.session_state["color_mode"]`
- igraph Vertex has no `.get()` method — use `v.attributes().get("key", default)`

## Cluster labeling formula
`score(term, cluster) = n_ut / (n_map_t + m)` where `n_ut` = papers in cluster containing term, `n_map_t` = papers in map containing term, `m` = smoothing parameter (default 25, user-controllable)

## Visualizer color scheme
- Plasma colorscale with log transform focused on 0.1%–5% range
- `_COLORBAR_MAX_PCT = 25` — colorbar maximum for search term mode
- Edges mode uses raw integer counts, not percentages
- Selected clusters get thick white border (14px bubble chart, borderWidth=4 PyVis)

## DO NOT
- Delete cache files without user permission
- Add docstrings, comments, or type annotations to code you didn't change
- Add error handling for scenarios that can't happen
- Refactor or clean up code beyond what was asked
- Use `st.experimental_rerun()` — use callbacks or session state instead

## Deploy notes (not yet done)
- Usage counter via Firestore before clustering (see DEPLOY comments in app.py)
- LLM cluster labeling via Vertex AI / Gemini (see DEPLOY comments in app.py)
- Switch `LocalSource` to `BigQuerySource` and set `GCP_PROJECT` env var
