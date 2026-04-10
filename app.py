"""Science Macroscope — Streamlit entry point.

Run with:
    streamlit run app.py
"""
import gzip
import json
import pickle
import os
from collections import defaultdict
import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import config

from graph.builder import (
    build_base_graph, filter_to_giant_component,
    apply_leiden, merge_to_target, build_cluster_graph, top_clusters_by_size,
)
from graph.visualizer import build_pyvis_html, build_bubble_figure
from graph.labeler import extract_nouns, label_clusters, cluster_term_frequency

CONTACT = "juanpablobascurcifuentes@gmail.com"
CLUSTERS_FILE = os.path.join(config.DATA_DIR, "demo_clusters.json")

st.set_page_config(page_title="Science Macroscope", layout="wide")
st.markdown(
    "<style>[data-testid='stFileUploaderDropzoneInstructions'] div small { display: none !important; }</style>",
    unsafe_allow_html=True,
)
st.markdown("<h1 style='pointer-events:none'>SciMacro <span style='color:#888;font-weight:400;font-size:0.6em'>Science Macroscope</span></h1>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Demo cluster definitions
# ---------------------------------------------------------------------------
with open(CLUSTERS_FILE) as _f:
    _DEMO_CLUSTERS = json.load(_f)

# ---------------------------------------------------------------------------
# Dataset selector — top of sidebar
# ---------------------------------------------------------------------------
# Defaults
_cluster_id = _DEMO_CLUSTERS[0]["id"]   # Bibliometrics

with st.sidebar:
    st.markdown("### Options")
    with st.expander("Selected dataset", expanded=True):

        # ---- Demo datasets ----
        st.caption(
            f"These are {len(_DEMO_CLUSTERS)} demo datasets drawn from the "
            "Leiden Ranking Open Edition 2023. Select one and load it to start exploring."
        )
        _fields_ordered = sorted({c["field"] for c in _DEMO_CLUSTERS})
        _default_field  = next(c["field"] for c in _DEMO_CLUSTERS if c["id"] == _DEMO_CLUSTERS[0]["id"])
        _sel_field = st.selectbox(
            "Field",
            options=_fields_ordered,
            index=_fields_ordered.index(_default_field),
            label_visibility="collapsed",
        )
        _clusters_in_field = sorted(
            [c for c in _DEMO_CLUSTERS if c["field"] == _sel_field],
            key=lambda c: c["label"],
        )
        _label_options = {c["label"]: c["id"] for c in _clusters_in_field}
        _default_label = _DEMO_CLUSTERS[0]["label"] if _DEMO_CLUSTERS[0]["field"] == _sel_field else _clusters_in_field[0]["label"]
        _sel_label = st.selectbox(
            "Topic",
            options=list(_label_options.keys()),
            index=list(_label_options.keys()).index(_default_label) if _default_label in _label_options else 0,
            label_visibility="collapsed",
        )
        _demo_cluster_id = _label_options[_sel_label]
        if st.button("Load demo data", use_container_width=True):
            st.session_state["dataset_source"] = "demo"
            st.session_state["dataset_cluster_id"] = _demo_cluster_id
            st.session_state.pop("cluster_result", None)


        st.markdown("<div style='text-align:center;color:#888;margin:8px 0'>— or —</div>", unsafe_allow_html=True)

        with st.expander("Upload your own data"):
            # Step 1 — Documents
            st.caption("1. Documents — CSV with columns: id, title, abstract")
            _docs_file = st.file_uploader(
                "Documents",
                type=["csv"],
                label_visibility="collapsed",
                help="CSV with columns: id, title, abstract. Titles appear on the map. Titles and abstracts are used to extract noun phrases.",
            )

            # Step 2 — Network
            st.caption("2. Network — CSV with columns: source, target, weight")
            _network_file = st.file_uploader(
                "Network",
                type=["csv"],
                label_visibility="collapsed",
                help="CSV with columns: source, target, weight. Each row is an edge. Weight is optional and defaults to 1.",
            )

            # Step 3 — Generate nouns
            st.caption("3. Generate nouns file — extracted from the documents file above")
            if st.button(
                "Generate nouns file",
                use_container_width=True,
                disabled=_docs_file is None,
                help="Run noun extraction (spaCy) on the uploaded documents. This may take several minutes. Download the result and upload it in step 4." if _docs_file else "Upload a documents file first.",
            ):
                _docs_df = pd.read_csv(_docs_file, dtype={"id": str})
                _docs_meta = {
                    row["id"].strip(): {"title": str(row.get("title", "") or ""), "abstract": str(row.get("abstract", "") or "")}
                    for _, row in _docs_df.iterrows()
                }
                with st.spinner("Extracting nouns — this may take several minutes…"):
                    _generated_nouns = extract_nouns(_docs_meta)
                _nouns_json = json.dumps(
                    {pid: sorted(nouns) for pid, nouns in _generated_nouns.items()},
                    indent=2,
                ).encode()
                st.session_state["_generated_nouns_json"] = _nouns_json

            if "_generated_nouns_json" in st.session_state:
                st.download_button(
                    "Download nouns file",
                    data=st.session_state["_generated_nouns_json"],
                    file_name="nouns.json",
                    mime="application/json",
                    use_container_width=True,
                )

            # Step 4 — Nouns
            st.caption("4. Nouns — JSON dictionary: document id → list of noun lemmas")
            _nouns_file = st.file_uploader(
                "Nouns",
                type=["json"],
                label_visibility="collapsed",
                help="JSON file mapping document id to a list of noun lemmas. Generate this file using the button above.",
            )

            # Load
            _custom_ready = _docs_file is not None and _network_file is not None and _nouns_file is not None
            if st.button(
                "Load custom data",
                use_container_width=True,
                disabled=not _custom_ready,
                help="Load the uploaded files." if _custom_ready else "Upload documents, network, and nouns files first.",
            ):
                st.session_state["dataset_source"] = "custom"
                st.session_state["custom_docs_file"]    = _docs_file
                st.session_state["custom_network_file"] = _network_file
                st.session_state["custom_nouns_file"]   = _nouns_file
                st.session_state.pop("cluster_result", None)

# ---------------------------------------------------------------------------
# Resolve active dataset
# ---------------------------------------------------------------------------
_dataset_source = st.session_state.get("dataset_source")

if _dataset_source is None:
    with st.sidebar:
        with st.expander("Selected documents", expanded=False):
            st.caption("No selected clusters.")
    st.info("Choose a demo dataset from the sidebar and click **Load demo data**, or upload your own data.")
    st.stop()

if _dataset_source == "custom":
    _network_file = st.session_state.get("custom_network_file")
    _docs_file    = st.session_state.get("custom_docs_file")
    _nouns_file   = st.session_state.get("custom_nouns_file")

    _network_file.seek(0)
    _net_df  = pd.read_csv(_network_file, dtype={"source": str, "target": str})
    _has_w   = "weight" in _net_df.columns
    _edges   = [
        (r["source"].strip(), r["target"].strip(), float(r["weight"]) if _has_w else 1.0)
        for _, r in _net_df.iterrows()
    ]
    _custom_graph = build_base_graph(_edges)

    if _docs_file is not None:
        _docs_file.seek(0)
        _docs_df  = pd.read_csv(_docs_file, dtype={"id": str})
        _custom_metadata = {
            row["id"].strip(): {
                "title":    str(row.get("title",    "") or ""),
                "abstract": str(row.get("abstract", "") or ""),
            }
            for _, row in _docs_df.iterrows()
        }
    else:
        _custom_metadata = {v["name"]: {"title": v["name"], "abstract": ""} for v in _custom_graph.vs}

    if _nouns_file is not None:
        _nouns_file.seek(0)
        _custom_nouns = {
            str(pid): set(lemmas)
            for pid, lemmas in json.load(_nouns_file).items()
        }
    else:
        _custom_nouns = {}

    _cluster_id = None
else:
    _cluster_id = st.session_state.get("dataset_cluster_id")

# ---------------------------------------------------------------------------
# GCS lazy download — only used when running on Cloud Run
# ---------------------------------------------------------------------------
_GCS_BUCKET = os.environ.get("GCS_CACHE_BUCKET")  # e.g. "scimacro-demo-cache"

def _ensure_cache_file(filename: str) -> bool:
    """Download *filename* from GCS into DATA_DIR if it isn't already there.

    Returns True if the file is available locally after the call, False if it
    could not be obtained (GCS not configured, or file not in bucket).
    """
    path = os.path.join(config.DATA_DIR, filename)
    if os.path.exists(path):
        return True
    if not _GCS_BUCKET:
        return False
    try:
        from google.cloud import storage as gcs
        client = gcs.Client()
        bucket = client.bucket(_GCS_BUCKET)
        blob   = bucket.blob(f"data/{filename}")
        if not blob.exists():
            return False
        os.makedirs(config.DATA_DIR, exist_ok=True)
        blob.download_to_filename(path)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Load data once per cluster
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading graph…")
def load_base_graph(cluster_id: int):
    filename = f"cache.{cluster_id}.graph.pkl"
    _ensure_cache_file(filename)
    path = os.path.join(config.DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    st.error(f"Cache not found for cluster {cluster_id}. If running locally, run `python precompute.py` first.")
    st.stop()

@st.cache_resource(show_spinner="Loading noun index…")
def load_paper_nouns(cluster_id: int):
    filename = f"cache.{cluster_id}.nouns.pkl"
    _ensure_cache_file(filename)
    path = os.path.join(config.DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return {}

@st.cache_resource(show_spinner="Loading abstracts…")
def load_abstracts(cluster_id: int) -> dict[str, str]:
    filename = f"cache.{cluster_id}.abstracts.pkl.gz"
    _ensure_cache_file(filename)
    path = os.path.join(config.DATA_DIR, filename)
    if os.path.exists(path):
        with gzip.open(path, "rb") as f:
            return pickle.load(f)
    return {}

if _dataset_source == "custom":
    base_graph  = _custom_graph
    metadata    = _custom_metadata
    paper_nouns = _custom_nouns
else:
    base_graph, metadata = load_base_graph(_cluster_id)
    paper_nouns = load_paper_nouns(_cluster_id)

# ---------------------------------------------------------------------------
# Document selections sidebar panel — rendered after metadata is available
# so it shows on every rerun, even before clustering.
# ---------------------------------------------------------------------------
def _cb_set_active():
    chosen = st.session_state["_active_sel_radio"]
    if chosen == "All documents":
        st.session_state["active_selection"] = None
    else:
        st.session_state["active_selection"] = next(
            (s for s in st.session_state["saved_selections"] if s["name"] == chosen), None
        )

with st.sidebar:
    with st.expander("Document selections", expanded=False):
        st.caption(
            "Did a document grab your attention? All demo datasets use OpenAlex document IDs. "
            "To see its entry, go to https://openalex.org/W followed by the doc_id — "
            "e.g. https://openalex.org/W2741809807"
        )

        _saved = st.session_state["saved_selections"]
        _active = st.session_state.get("active_selection")

        _options = [{"name": "All documents", "doc_ids": list(metadata.keys())}] + _saved
        _option_names = [o["name"] for o in _options]
        _active_name = _active["name"] if _active else "All documents"
        _active_idx = _option_names.index(_active_name) if _active_name in _option_names else 0

        st.radio(
            "Active selection for clustering",
            options=_option_names,
            index=_active_idx,
            key="_active_sel_radio",
            label_visibility="collapsed",
            on_change=_cb_set_active,
        )

        _view = _options[_active_idx]
        _view_names = _view["doc_ids"]
        _total_view = len(_view_names)

        _intra_deg = {name: 0 for name in _view_names}
        for _e in base_graph.es:
            _sn = base_graph.vs[_e.source]["name"]
            _tn = base_graph.vs[_e.target]["name"]
            if _sn in _intra_deg and _tn in _intra_deg:
                _intra_deg[_sn] += 1
                _intra_deg[_tn] += 1

        _rows_top = sorted(_view_names, key=lambda n: _intra_deg[n], reverse=True)[:1000]
        _sel_df = pd.DataFrame([
            {"doc_id": n, "edges": _intra_deg[n], "title": metadata.get(n, {}).get("title", "")}
            for n in _rows_top
        ])

        _caption = f"{_total_view:,} documents"
        if _total_view > 1000:
            _caption += " — showing top 1,000"
        st.caption(_caption)

        _abstracts = load_abstracts(_cluster_id) if _cluster_id is not None else {
            pid: metadata.get(pid, {}).get("abstract", "") for pid in _view_names
        }
        st.download_button(
            "Export CSV",
            data=pd.DataFrame([
                {"doc_id": n, "edges": _intra_deg[n],
                 "title": metadata.get(n, {}).get("title", ""),
                 "abstract": _abstracts.get(n, "")}
                for n in _view_names
            ]).to_csv(index=False).encode(),
            file_name=f"{_view['name'].lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(_sel_df, use_container_width=True, hide_index=True)

        if _active_idx > 0:
            if st.button("Delete this selection", key="del_sel", use_container_width=True):
                st.session_state["saved_selections"] = [
                    s for s in _saved if s["name"] != _active_name
                ]
                st.session_state["active_selection"] = None
                st.rerun()

def _to_superscript(n: int) -> str:
    return str(n).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))

# ---------------------------------------------------------------------------
# Selection callbacks — run before the script body so the sidebar sees
# up-to-date selected_clusters on the same rerun as the button click.
# ---------------------------------------------------------------------------
if "selected_clusters" not in st.session_state:
    st.session_state["selected_clusters"] = set()
if "saved_selections" not in st.session_state:
    st.session_state["saved_selections"] = []   # list of {"name": str, "doc_ids": list[str]}
if "active_selection" not in st.session_state:
    st.session_state["active_selection"] = None  # None means "All documents"

def _cb_save_selection():
    prev = st.session_state.get("cluster_result")
    if not prev:
        return
    selected_ids = st.session_state.get("selected_clusters", set())
    if not selected_ids:
        return
    membership = prev["membership"]
    doc_ids = [name for name, cid in membership.items() if cid in selected_ids]
    doc_ids += list(st.session_state.get("outside_papers", set()))
    n = len(st.session_state["saved_selections"]) + 1
    name = st.session_state.get("_save_sel_name", "").strip() or f"Selection {n}"
    st.session_state["saved_selections"].append({"name": name, "doc_ids": doc_ids})
    st.session_state["selected_clusters"] = set()
    st.session_state["outside_papers"] = set()
    st.session_state["expand_outside_enabled"] = False
    st.session_state["expand_outside_cb"] = False
    st.session_state.pop("_save_sel_name", None)

def _cb_add_selection():
    tokens = st.session_state.get("sel_input", "").strip().split()
    prev = st.session_state.get("cluster_result")
    valid_ids = {v["community_id"] for v in prev["cg"].vs} if prev else set()
    not_found = []
    for token in tokens:
        try:
            cid = int(token)
            if cid in valid_ids:
                st.session_state["selected_clusters"].add(cid)
            else:
                not_found.append(token)
        except ValueError:
            not_found.append(token)
    if not_found:
        st.session_state["_sel_warning"] = f"Not found: {', '.join(not_found)}"
    else:
        st.session_state.pop("_sel_warning", None)

def _cb_clear_selection():
    st.session_state["selected_clusters"] = set()
    st.session_state.pop("_sel_warning", None)


def _cb_select_all():
    prev = st.session_state.get("cluster_result")
    if prev:
        st.session_state["selected_clusters"] = {v["community_id"] for v in prev["cg"].vs}
    st.session_state.pop("_sel_warning", None)

def _cb_remove_selection():
    tokens = st.session_state.get("sel_input", "").strip().split()
    not_found = []
    for token in tokens:
        try:
            cid = int(token)
            st.session_state["selected_clusters"].discard(cid)
        except ValueError:
            not_found.append(token)
    if not_found:
        st.session_state["_sel_warning"] = f"Invalid: {', '.join(not_found)}"
    else:
        st.session_state.pop("_sel_warning", None)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    with st.expander("Run clustering", expanded=False):
        _active_sel = st.session_state.get("active_selection")
        if _active_sel is None:
            _run_label = "Cluster all documents"
            _run_help  = "Run Leiden on the full dataset."
        else:
            _n_docs = len(_active_sel["doc_ids"])
            _run_label = f"Cluster — {_active_sel['name']} ({_n_docs:,} docs)"
            _run_help  = f"Run Leiden on the {_n_docs:,} documents in '{_active_sel['name']}'."
        run = st.button(
            _run_label,
            type="secondary",
            use_container_width=True,
            help=_run_help,
        )
        run_selection = False  # kept for compatibility with clustering block below
        main_component_only = st.checkbox(
            "Main component only",
            value=True,
            help=(
                "Keeps only documents belonging to the largest connected network. "
                "A connected network is a group of documents where every document can "
                "be reached from every other through citation links. "
                "Documents outside this main network are removed as they cannot be "
                "meaningfully placed on the map."
            ),
        )
        st.caption(
            "Higher resolution tends to produce more clusters. "
            "More merging (larger gap between pre-merge and target) decreases clustering quality "
            "but makes clusters more evenly sized."
        )
        e_col, m_col = st.columns(2)
        with e_col:
            exponent = st.number_input(
                "Exponent",
                value=-3,
                step=1,
                help="Main lever — each step multiplies or divides the resolution by 10. Use integers only.",
            )
        with m_col:
            mantissa = st.number_input(
                "Coefficient",
                value=1.00,
                step=0.01,
                format="%.2f",
                help="Fine-tuning — adjusts the resolution within the order of magnitude set by the exponent.",
            )
        resolution = mantissa * (10 ** exponent)
        st.info(f"**{mantissa:.2f} × 10{_to_superscript(exponent)}**")
        target_n = st.number_input(
            "Target clusters",
            min_value=1, max_value=2000, value=10, step=1,
            help=(
                "Merge smallest clusters by edge density until this many remain. "
                "For more than 30 clusters, the bubble layout can take a long time to calculate — "
                "consider enabling the Fast layout option."
            ),
        )

    with st.expander("Science map display", expanded=False):
        layout = st.radio(
            "Layout",
            options=["Bubble chart", "Force-directed"],
            help=(
                "Bubble chart: deterministic, no overlap, clusters placed to minimise stress. "
                "Slower for >30 clusters.\n\n"
                "Force-directed: physics simulation in the browser, faster to compute."
            ),
        )
        is_bubble = layout == "Bubble chart"

        st.caption("Bubble chart settings")
        fast_layout = st.checkbox(
            "Fast layout",
            value=False,
            disabled=not is_bubble,
            help="Try only a subset of starting nodes instead of all — faster but slightly lower quality.",
        )
        max_starts = None
        if fast_layout and is_bubble:
            max_starts = st.number_input(
                "Starting nodes to try",
                min_value=1, max_value=500, value=10, step=1,
                help="More starting nodes = better layout quality, slower computation.",
            )

        st.caption("Force-directed settings")
        physics_on = st.checkbox(
            "Physics on",
            value=True,
            disabled=is_bubble,
            help="Run Barnes-Hut physics simulation in the browser.",
        )

        st.caption("Both layouts")
        show_edges = st.checkbox("Show edges", value=False)

    _has_results = "cluster_result" in st.session_state
    st.markdown("### Data")
    with st.expander("Cluster labels", expanded=False):
        if not _has_results:
            st.caption("No generated clusters.")
        _nd_col, _nn_col, _lm_col, _lb_col = st.columns([1, 1, 1, 2])
        with _nd_col:
            top_docs_n = st.number_input(
                "Top docs",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                help="Number of top documents to show per cluster in the table.",
            )
        with _nn_col:
            top_nouns_n = st.number_input(
                "Top nouns",
                min_value=1,
                max_value=10,
                value=5,
                step=1,
                help="Number of top nouns to show per cluster in the table and tooltip.",
            )
        with _lm_col:
            label_m = st.number_input(
                "Noun smoothing",
                min_value=0,
                max_value=10000,
                value=25,
                step=1,
                help=(
                    "Label smoothing parameter. "
                    "Higher values reduce the influence of terms that appear only in a few documents, "
                    "favouring terms that are consistently frequent across the cluster."
                ),
            )
        with _lb_col:
            st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
            if st.button("Regenerate labels", use_container_width=True, disabled=not _has_results,
                         help="Recompute cluster labels with the current smoothing value."):
                _prev = st.session_state["cluster_result"]
                label_clusters(_prev["cg"], _prev["labels"], paper_nouns, _prev["membership"], m=label_m)
        if _has_results:
            st.divider()
            _cg_sidebar     = st.session_state["cluster_result"]["cg"]
            _labels_sidebar = st.session_state["cluster_result"]["labels"]
            _sel_sidebar    = st.session_state.get("selected_clusters", set())
            _rows_sidebar = top_clusters_by_size(_cg_sidebar, _labels_sidebar, n=_cg_sidebar.vcount(), top_docs=top_docs_n, top_nouns=top_nouns_n)
            _df_sidebar = pd.DataFrame(_rows_sidebar)
            _all_cols = ["c.", "size", "top documents", "top nouns"]
            _display_df = _df_sidebar[[c for c in _all_cols if c in _df_sidebar.columns]].copy()

            def _cell_html(val):
                text = str(val) if val is not None else ""
                return text.replace("\n", "<br>")

            _header = "".join(
                f"<th style='padding:6px 8px;border:1px solid #444;text-align:left;"
                f"white-space:nowrap;background:#1e1e2e;font-size:12px'>{col}</th>"
                for col in _display_df.columns
            )
            _rows_html = ""
            for _idx, _row in _display_df.iterrows():
                _is_sel = int(_df_sidebar.at[_idx, "c."]) in _sel_sidebar
                _bg = "background:#2a1f4e;" if _is_sel else ""
                _cells = "".join(
                    f"<td style='padding:6px 8px;border:1px solid #444;"
                    f"vertical-align:top;font-size:12px;{_bg}"
                    f"{'white-space:nowrap;' if col in ('c.', 'size') else ('min-width:120px;overflow-wrap:break-word;word-break:normal;' if col == 'top nouns' else 'word-break:break-word;')}'>"
                    f"{_cell_html(_row[col])}</td>"
                    for col in _display_df.columns
                )
                _rows_html += f"<tr>{_cells}</tr>"

            st.markdown(
                f"<table style='width:100%;border-collapse:collapse'>"
                f"<thead><tr>{_header}</tr></thead>"
                f"<tbody>{_rows_html}</tbody>"
                f"</table>",
                unsafe_allow_html=True,
            )



# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
if run:
    try:
        with st.spinner("Running Leiden…"):
            _active_sel = st.session_state.get("active_selection")
            if _active_sel is not None:
                keep_names = set(_active_sel["doc_ids"])
            else:
                keep_names = None  # all documents

            g = base_graph.copy()
            if keep_names is not None:
                drop = [v.index for v in g.vs if v["name"] not in keep_names]
                g.delete_vertices(drop)
            removed_disconnected = 0
            if main_component_only:
                g, removed_disconnected = filter_to_giant_component(g)
            apply_leiden(g, resolution=resolution)
            n_leiden = len(set(g.vs["community_id"]))

        if n_leiden < target_n:
            st.session_state["_ran_selection"] = False
            st.warning(
                f"Leiden produced **{n_leiden} clusters** at this resolution — "
                f"lower than your target of {target_n}. Increase the resolution to get more clusters."
            )
            st.stop()

        with st.spinner(f"Merging {n_leiden} → {target_n} clusters…"):
            removed_isolated = merge_to_target(g, target_n) if n_leiden > target_n else 0
            cg, labels = build_cluster_graph(g, metadata)
            membership = {v["name"]: v["community_id"] for v in g.vs}
            label_clusters(cg, labels, paper_nouns, membership, m=label_m)

        st.session_state["cluster_result"] = {
            "cg": cg,
            "labels": labels,
            "membership": membership,
            "n_papers": g.vcount(),
            "n_citations": g.ecount(),
            "n_leiden": n_leiden,
            "removed_disconnected": removed_disconnected,
            "removed_isolated": removed_isolated,
        }
        st.session_state["selected_clusters"] = set()
        st.session_state["outside_papers"] = set()
        st.session_state["expand_outside_enabled"] = False
        st.session_state["expand_outside_cb"] = False

    except st.runtime.scriptrunner.StopException:
        raise
    except Exception as e:
        st.error(
            f"Something went wrong: `{e}`\n\n"
            f"Please contact the demo administrator at "
            f"[{CONTACT}](mailto:{CONTACT})."
        )
        st.stop()
    st.rerun()

if "cluster_result" not in st.session_state:
    st.info("Choose a demo dataset from the sidebar and click **Load demo data**, then set the resolution and target clusters and click **Cluster**.")
    st.stop()

try:
    result     = st.session_state["cluster_result"]
    cg         = result["cg"]
    labels     = result["labels"]
    membership = result["membership"]
    n_clusters = cg.vcount()

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("Documents",       f"{result['n_papers']:,}")
    col2.metric("Citations",       f"{result['n_citations']:,}")
    col3.metric("Leiden clusters", result["n_leiden"])
    col4.metric("After merge",     n_clusters)
    col5.metric("Resolution",      f"{resolution:.2e}")
    col6.metric(
        "Removed (disconnected)",
        f"{result['removed_disconnected']:,}",
        help="Documents not part of the largest connected network — there is no citation path connecting them to the main body of literature.",
    )
    col7.metric(
        "Removed (isolated)",
        f"{result['removed_isolated']:,}",
        help="Documents in clusters with no citation links to neighbouring clusters at merge time.",
    )

    if n_clusters > config.MAX_RENDERABLE_CLUSTERS:
        st.warning(
            f"**{n_clusters} clusters** is too many to render clearly "
            f"(limit: {config.MAX_RENDERABLE_CLUSTERS}). "
            f"Lower the resolution or enable **Main component only**."
        )
        st.stop()

    # ---------------------------------------------------------------------------
    # Controls: search / edges / selection — uniform text | apply | clear rows
    # ---------------------------------------------------------------------------
    if "selected_clusters" not in st.session_state:
        st.session_state["selected_clusters"] = set()

    st.markdown("**Coloring options**")
    c_label, c_input, c_apply, c_clear = st.columns([2, 5, 2, 2])
    c_label.markdown("<div style='padding-top:32px;font-size:13px;color:#aaa'>Search term</div>", unsafe_allow_html=True)
    with c_input:
        search_term = st.text_input("search_term", placeholder="e.g. network", label_visibility="collapsed",
            help="Highlight clusters by how many documents contain this noun.")
    with c_apply:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        if st.button("Apply", key="apply_search", use_container_width=True):
            st.session_state["color_mode"] = "search"
    with c_clear:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        if st.button("Clear", key="clear_search", use_container_width=True):
            if st.session_state.get("color_mode") == "search":
                st.session_state.pop("color_mode", None)
                st.session_state.pop("color_values", None)
                st.session_state.pop("color_label", None)

    c_label, c_input, c_apply, c_clear = st.columns([2, 5, 2, 2])
    c_label.markdown("<div style='padding-top:32px;font-size:13px;color:#aaa'>Edges from cluster</div>", unsafe_allow_html=True)
    with c_input:
        conn_input = st.text_input("conn_input", placeholder="e.g. 3", label_visibility="collapsed",
            help="Colour clusters by number of citation edges shared with this cluster.")
    with c_apply:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        if st.button("Apply", key="apply_conn", use_container_width=True):
            st.session_state["color_mode"] = "connectivity"
    with c_clear:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        if st.button("Clear", key="clear_conn", use_container_width=True):
            if st.session_state.get("color_mode") == "connectivity":
                st.session_state.pop("color_mode", None)
                st.session_state.pop("color_values", None)
                st.session_state.pop("color_label", None)

    st.markdown("**Select documents**")
    c_label, c_input, c_add, c_rem, c_all, c_clear = st.columns([2, 5, 2, 2, 2, 2])
    c_label.markdown("<div style='padding-top:32px;font-size:13px;color:#aaa'>Select clusters</div>", unsafe_allow_html=True)
    with c_input:
        st.text_input("sel_input", key="sel_input", placeholder="e.g. 0 3 7", label_visibility="collapsed",
            help="Space-separated cluster IDs. Use Add to select, Remove to deselect.")
    with c_add:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        st.button("Add", key="add_sel", use_container_width=True, on_click=_cb_add_selection)
    with c_rem:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        st.button("Remove", key="rem_sel", use_container_width=True, on_click=_cb_remove_selection)
    with c_all:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        st.button("All", key="sel_all", use_container_width=True, on_click=_cb_select_all,
            help="Select all clusters currently in the map.")
    with c_clear:
        st.markdown("<div style='padding-top:24px'></div>", unsafe_allow_html=True)
        st.button("Clear", key="clear_sel", use_container_width=True, on_click=_cb_clear_selection)

    if st.session_state.get("_sel_warning"):
        st.warning(st.session_state["_sel_warning"])

    selected_clusters = st.session_state["selected_clusters"]
    if selected_clusters:
        ids_sorted = sorted(selected_clusters)
        total_papers = sum(
            v["size"] for v in cg.vs if v["community_id"] in selected_clusters
        )
        st.caption(f"Selected: {', '.join(str(i) for i in ids_sorted)} — {total_papers:,} documents")

        c_name, c_save = st.columns([4, 2])
        with c_name:
            st.text_input(
                "Save name",
                key="_save_sel_name",
                placeholder=f"Selection {len(st.session_state['saved_selections']) + 1}",
                label_visibility="collapsed",
            )
        with c_save:
            st.button("Save as selection", key="save_sel", use_container_width=True,
                      on_click=_cb_save_selection,
                      help="Save the selected clusters as a named document set for clustering.")

    # ---------------------------------------------------------------------------
    # Outside papers
    # ---------------------------------------------------------------------------
    expand_outside = st.checkbox(
        "Add documents connected to the selected documents",
        value=st.session_state.get("expand_outside_enabled", False),
        key="expand_outside_cb",
        help="Find documents linked to the selected clusters, regardless of whether they are in the map.",
        disabled=not selected_clusters,
        on_change=lambda: st.session_state.update(
            expand_outside_enabled=st.session_state.get("expand_outside_cb", False)
        ),
    )
    st.session_state["expand_outside_enabled"] = expand_outside
    outside_threshold = 1
    outside_papers: set[str] = set()
    if expand_outside and selected_clusters:
        outside_threshold = st.number_input(
            "Minimum links to selected documents",
            min_value=1, value=20, step=1,
            help="A document is included if it has at least this many citation links to documents in the selected clusters.",
        )
        # Compute outside papers — any paper not in selected clusters with enough links
        selected_names = {name for name, cid in membership.items() if cid in selected_clusters}
        selected_idx = {v.index for v in base_graph.vs if v["name"] in selected_names}

        link_count: dict[str, int] = defaultdict(int)
        for idx in selected_idx:
            for nb in base_graph.neighbors(idx):
                nb_name = base_graph.vs[nb]["name"]
                if nb_name not in selected_names:
                    link_count[nb_name] += 1

        outside_papers = {name for name, cnt in link_count.items() if cnt >= outside_threshold}
        st.caption(f"{len(outside_papers):,} documents outside selection with ≥ {outside_threshold} links to selection")

    st.session_state["outside_papers"] = outside_papers

    # Compute color values from active mode
    color_mode = st.session_state.get("color_mode")

    if color_mode == "search" and search_term.strip() and paper_nouns:
        st.session_state["color_values"] = cluster_term_frequency(cg, search_term, paper_nouns, membership)
        st.session_state["color_label"]  = "% docs<br>with term"

    elif color_mode == "connectivity" and conn_input.strip():
        try:
            source_id = int(conn_input.strip())
            src_v = next((v for v in cg.vs if v["community_id"] == source_id), None)
            if src_v is None:
                st.warning(f"Cluster {source_id} not found.")
                st.session_state.pop("color_mode", None)
            else:
                weights = {}
                for e in cg.es:
                    s = cg.vs[e.source]["community_id"]
                    t = cg.vs[e.target]["community_id"]
                    if s == source_id:
                        weights[t] = e["weight"]
                    elif t == source_id:
                        weights[s] = e["weight"]
                max_w = max(weights.values()) if weights else 1
                st.session_state["color_values"] = [
                    weights.get(v["community_id"], 0.0) / max_w
                    for v in cg.vs
                ]
                st.session_state["color_raw_values"] = [
                    weights.get(v["community_id"], 0)
                    for v in cg.vs
                ]
                st.session_state["color_label"] = f"Edges from<br>cluster {source_id}"
                st.session_state["color_max_val"] = max_w
        except ValueError:
            st.warning("Cluster ID must be an integer.")
            st.session_state.pop("color_mode", None)

    color_values     = st.session_state.get("color_values")
    color_label      = st.session_state.get("color_label", "% docs<br>with term")
    color_raw_values = st.session_state.get("color_raw_values") if color_mode == "connectivity" else None
    hover_label      = "edges" if color_mode == "connectivity" else "% of documents contain this term"

    # ---------------------------------------------------------------------------
    # Cluster map (full width)
    # ---------------------------------------------------------------------------
    if layout == "Bubble chart":
        with st.spinner("Computing bubble layout…"):
            fig = build_bubble_figure(cg, labels, show_edges=show_edges, max_starts=max_starts, term_freq=color_values, color_label=color_label, hover_label=hover_label, raw_values=color_raw_values, selected=selected_clusters, top_docs=top_docs_n, top_nouns=top_nouns_n)
        st.plotly_chart(fig, use_container_width=True)
    else:
        html = build_pyvis_html(cg, labels, physics=physics_on, show_edges=show_edges, term_freq=color_values, color_label=color_label, hover_label=hover_label, raw_values=color_raw_values, selected=selected_clusters, top_docs=top_docs_n, top_nouns=top_nouns_n)
        components.html(html, height=720, scrolling=False)

    components.html("""
        <style>
            #back-to-top {
                position: fixed;
                bottom: 24px;
                right: 24px;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                background: rgba(255,255,255,0.15);
                backdrop-filter: blur(6px);
                border: 1px solid rgba(255,255,255,0.3);
                color: white;
                font-size: 20px;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 9999;
                text-decoration: none;
            }
        </style>
        <a id="back-to-top" onclick="window.parent.scrollTo({top:0,behavior:'smooth'})" title="Back to top">↑</a>
    """, height=0)


except st.runtime.scriptrunner.StopException:
    raise
except Exception as e:
    st.error(
        f"Something went wrong: `{e}`\n\n"
        f"Please contact the demo administrator at "
        f"[{CONTACT}](mailto:{CONTACT})."
    )
    st.stop()
