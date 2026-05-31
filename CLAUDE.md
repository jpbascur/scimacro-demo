# Science Macroscope - Claude Instructions

## Project Overview
Streamlit app that clusters scientific papers by citation network using the Leiden algorithm and visualises the result as an interactive science map. Demo data is loaded from precomputed cache files; custom data can be uploaded through the app.

## File Map
- `app.py` - Streamlit entry point, all UI logic
- `config.py` - file paths, column names, and algorithm parameters
- `precompute.py` - one-time precomputation of graph, abstracts, and nouns
- `data/demo_clusters.json` - registry of demo datasets
- `graph/builder.py` - igraph construction, Leiden clustering, merging, cluster graph
- `graph/labeler.py` - spaCy noun extraction, cluster labeling
- `graph/visualizer.py` - PyVis force-directed and Plotly bubble chart renderers
- `graph/bubble_layout.py` - stress-minimising layout algorithm

## Cache Files - Never Delete Without Explicit User Permission
`data/cache.*.graph.pkl`, `data/cache.*.nouns.pkl`, and `data/cache.*.abstracts.pkl.gz` are expensive to regenerate. If a cache appears stale, warn the user and explain why; never delete or overwrite it unilaterally.

## Key Conventions
- All session state lives in `st.session_state`; never use module-level globals for UI state.
- Loaded datasets are immutable source data.
- Saved collections store selected documents, an induced subgraph, connection counts, and term counts.
- Clustering results are stored in `st.session_state["cluster_result"]` so widget reruns do not clear the map.
- `document_to_cluster` is a partial mapping; selected documents missing from it are unassigned in the current run.
- Selected clusters are stored in `st.session_state["selected_clusters"]`.
- Sidebar `on_click` callbacks are used for selection so state updates before the sidebar re-renders.
- Color mode (`"search"` or `"connectivity"`) is stored in `st.session_state["color_mode"]`.
- igraph Vertex has no `.get()` method; use `v.attributes().get("key", default)`.

## Cluster Labeling Formula
`score(term, cluster) = n_ut / (n_map_t + m)` where `n_ut` = papers in cluster containing term, `n_map_t` = papers in map containing term, `m` = smoothing parameter.

## Visualizer Color Scheme
- Plasma colorscale with log transform focused on the low-frequency range.
- `_COLORBAR_MAX_PCT = 25` is the colorbar maximum for search term mode.
- Edges mode uses raw integer counts, not percentages.
- Selected clusters get thick white borders.

## Do Not
- Delete cache files without user permission.
- Refactor or clean up code beyond what was asked.
- Use `st.experimental_rerun()`; use callbacks or session state instead.
