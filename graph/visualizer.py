"""Renders a cluster-level igraph as an interactive PyVis or Plotly figure.

Two rendering modes are provided:

build_pyvis_html   — Force-directed layout using vis.js / PyVis. Physics simulation
                     (Barnes-Hut) runs in the browser. Good for exploratory use where
                     the user wants to drag and rearrange nodes. Returns an HTML string
                     suitable for Streamlit's components.html().

build_bubble_figure — Deterministic bubble chart using the stress-minimising layout
                      from bubble_layout.py. Circles are drawn as Plotly layout.shapes
                      in data coordinates so that bubble sizes exactly match the layout
                      algorithm. An invisible scatter trace carries hover tooltips and
                      cluster labels.

Color assignment is cyclic over a fixed 20-color palette; cluster 0 gets the first
color, cluster 1 the second, etc. Node/bubble size in both renderers is proportional
to cluster size (number of member documents).
"""
import math

import igraph as ig
import plotly.graph_objects as go
from pyvis.network import Network


EDGE_COLOR = "#4a4e69"
MIN_NODE_SIZE = 10
MAX_NODE_SIZE = 60
MAX_LABEL_LEN = 50

_COMMUNITY_PALETTE = [
    "#457b9d", "#2a9d8f", "#c9a227", "#f4a261", "#264653",
    "#6a4c93", "#1982c4", "#8ac926", "#ff595e", "#6a994e",
    "#5a9ea0", "#52b788", "#f77f00", "#7209b7", "#3a86ff",
    "#fb8500", "#023e8a", "#80b918", "#d62828", "#606c38",
]


def _cluster_color(community_id: int) -> str:
    """Return a hex color string for *community_id* by cycling over the palette."""
    return _COMMUNITY_PALETTE[community_id % len(_COMMUNITY_PALETTE)]


# Plasma colorscale stops (t=0 → absent, t=1 → frequent)
_PLASMA_STOPS = [
    (0.050, (13,   8,  135)),   # deep purple
    (0.125, (84,   2,  163)),   # purple
    (0.250, (139,  10, 165)),   # magenta-purple
    (0.375, (185,  50, 137)),   # pink-purple
    (0.500, (219,  92,  104)),  # salmon
    (0.625, (244, 136,   73)),  # orange
    (0.750, (254, 188,   43)),  # yellow-orange
    (1.000, (240, 249,   33)),  # bright yellow
]


_LOG_MIN_PCT = 0.05   # frequencies below this percentage are treated as absent


def _log_transform(freq: float) -> float:
    """Map freq ∈ [0,1] to t ∈ [0,1] using a log scale anchored at 0.1%–5%.

    Frequencies below _LOG_MIN_PCT% → 0 (absent).
    The log scale spreads the 0.1%–5% range across ~50% of the colour axis.
    """
    pct = freq * 100
    if pct < _LOG_MIN_PCT:
        return 0.0
    log_min = math.log10(_LOG_MIN_PCT)
    log_max = math.log10(100)
    t = (math.log10(pct) - log_min) / (log_max - log_min)
    return max(0.0, min(1.0, t))


def _gradient_color(freq: float) -> str:
    """Map freq ∈ [0,1] to a plasma-like colour with log scaling."""
    t = _log_transform(freq)
    if t <= _PLASMA_STOPS[0][0]:
        r, g, b = _PLASMA_STOPS[0][1]
        return f"#{r:02x}{g:02x}{b:02x}"
    for i in range(len(_PLASMA_STOPS) - 1):
        t0, c0 = _PLASMA_STOPS[i]
        t1, c1 = _PLASMA_STOPS[i + 1]
        if t <= t1:
            alpha = (t - t0) / (t1 - t0)
            r = int(c0[0] + alpha * (c1[0] - c0[0]))
            g = int(c0[1] + alpha * (c1[1] - c0[1]))
            b = int(c0[2] + alpha * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    r, g, b = _PLASMA_STOPS[-1][1]
    return f"#{r:02x}{g:02x}{b:02x}"


def _term_low_hex()  -> str: return "#{:02x}{:02x}{:02x}".format(*_PLASMA_STOPS[0][1])
def _term_high_hex() -> str: return "#{:02x}{:02x}{:02x}".format(*_PLASMA_STOPS[-1][1])


_COLORBAR_MAX_PCT = 25   # values above this are clamped to the top colour


def _sqrt_colorscale(n: int = 512) -> list:
    """Plotly colorscale where position 1.0 corresponds to _COLORBAR_MAX_PCT%."""
    return [[i / n, _gradient_color(i / n * _COLORBAR_MAX_PCT / 100)] for i in range(n + 1)]


def _inject_gradient_legend(
    html: str,
    color_label: str = "% docs with term",
    min_label: str = "0%",
    max_label: str = "100%",
) -> str:
    """Inject a fixed-position plasma gradient legend into a PyVis HTML string."""
    stops = ", ".join(
        f"#{r:02x}{g:02x}{b:02x}" for _, (r, g, b) in _PLASMA_STOPS
    )
    legend = f"""
    <div style="position:fixed;bottom:20px;right:20px;background:rgba(0,0,0,0.75);
                padding:10px 14px;border-radius:8px;color:white;
                font-family:sans-serif;font-size:12px;z-index:999;">
      <div style="margin-bottom:5px;text-align:center;font-weight:bold;">{color_label.replace("<br>", " ")}</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span>{min_label}</span>
        <div style="width:120px;height:14px;
                    background:linear-gradient(to right,{stops});
                    border-radius:3px;border:1px solid rgba(255,255,255,0.2);"></div>
        <span>{max_label}</span>
      </div>
    </div>
    """
    return html.replace("</body>", legend + "\n</body>")


def _scale(value: int, min_val: int, max_val: int) -> float:
    """Linearly interpolate *value* from [min_val, max_val] to [MIN_NODE_SIZE, MAX_NODE_SIZE].

    When min_val == max_val (all clusters the same size), returns the midpoint size.
    Used only for PyVis pixel-space node sizes — bubble layout uses radii in data space.
    """
    if max_val == min_val:
        return (MIN_NODE_SIZE + MAX_NODE_SIZE) / 2
    t = (value - min_val) / (max_val - min_val)
    return MIN_NODE_SIZE + t * (MAX_NODE_SIZE - MIN_NODE_SIZE)


def build_pyvis_html(
    cg: ig.Graph,
    labels: dict,
    height: str = "700px",
    width: str = "100%",
    physics: bool = True,
    show_edges: bool = False,
    term_freq: list[float] | None = None,
    color_label: str = "% docs<br>with term",
    hover_label: str = "% of documents contain this term",
    raw_values: list | None = None,
    selected: set[int] | None = None,
    top_docs: int = 3,
    top_nouns: int = 5,
) -> str:
    """Render a cluster-level igraph as a PyVis HTML string.

    The HTML runs a vis.js force simulation in the browser. When *physics* is
    True, the Barnes-Hut algorithm is used (spring_length=150, spring_strength=0.04,
    damping=0.09) with 300 stabilization iterations, which gives good results for
    citation cluster graphs in the 10–300 node range.

    Node size is linearly scaled from MIN_NODE_SIZE to MAX_NODE_SIZE based on the
    number of papers in the cluster. Edge width is linearly scaled 1–5 based on the
    cross-cluster citation count relative to the heaviest edge.

    Args:
        cg:         Cluster-level igraph as returned by build_cluster_graph().
                    Required vertex attrs: community_id, size, top_title.
                    Required edge attr (if show_edges): weight.
        height:     Canvas height CSS string (default "700px").
        width:      Canvas width CSS string (default "100%").
        physics:    Run Barnes-Hut physics simulation in the browser. When False the
                    graph is static but nodes can still be dragged manually.
        show_edges: Draw inter-cluster edges. Hidden by default to reduce visual clutter
                    when there are many clusters.

    Returns:
        Self-contained HTML string suitable for Streamlit components.html().
    """
    net = Network(
        height=height,
        width=width,
        directed=False,
        notebook=False,
        bgcolor="#1a1a2e",
        font_color="white",
    )

    if physics:
        net.barnes_hut(spring_length=150, spring_strength=0.04, damping=0.09)
        net.set_options("""
        {
          "physics": {
            "minVelocity": 0.75,
            "stabilization": { "iterations": 300, "fit": true }
          }
        }
        """)
    else:
        net.toggle_physics(False)

    sizes = cg.vs["size"]
    min_s, max_s = min(sizes), max(sizes)

    for v in cg.vs:
        cid = v["community_id"]
        node_size = _scale(v["size"], min_s, max_s)
        info      = labels.get(cid, {})
        top_titles = info.get("top_titles", [])
        docs_html = "".join(
            f"<br>{i+1}. {t[:80] + '…' if len(t) > 80 else t}"
            for i, t in enumerate(top_titles[:top_docs]) if t
        )
        nouns_html = "".join(
            f"<br><span style='color:#aaa'>{i+1}. {n}</span>"
            for i, n in enumerate(info.get("top_nouns", [])[:top_nouns])
        )
        if term_freq is not None:
            if raw_values is not None:
                display = f"{int(raw_values[cid]):,}"
            else:
                display = f"{term_freq[cid]*100:.2f}%"
            metric_html = f"<br><b>{display} {hover_label}</b>"
            color = _gradient_color(term_freq[cid])
        else:
            metric_html = ""
            color = _cluster_color(cid)
        tooltip = (
            f"<div style='font-size:13px;max-width:400px;line-height:1.5'>"
            f"<b>Cluster {cid}</b> &nbsp;·&nbsp; {v['size']:,} documents"
            f"{metric_html}"
            f"{docs_html}"
            f"{nouns_html}"
            f"</div>"
        )
        is_selected = selected and cid in selected
        node_color = {"background": color, "border": "#ffffff"} if is_selected else color
        net.add_node(
            cid,
            label=str(cid),
            title=tooltip,
            color=node_color,
            size=node_size,
            borderWidth=4 if is_selected else 1,
            font={"size": 20, "color": "#333333", "bold": True},
        )

    if show_edges:
        weights = cg.es["weight"] if cg.ecount() > 0 else []
        max_w = max(weights) if weights else 1
        for e, w in zip(cg.es, weights):
            width = 1 + 4 * (w / max_w)
            net.add_edge(
                cg.vs[e.source]["community_id"],
                cg.vs[e.target]["community_id"],
                color=EDGE_COLOR,
                width=width,
                title=f"{w:,} citations",
            )

    html = net.generate_html()
    if term_freq is not None:
        if raw_values is not None:
            max_raw = max(raw_values) if raw_values else 1
            html = _inject_gradient_legend(
                html,
                color_label.replace("<br>", " "),
                min_label="0",
                max_label=f"{int(max_raw):,}",
            )
        else:
            html = _inject_gradient_legend(html, color_label.replace("<br>", " "))
    return html


def build_bubble_figure(
    cg: ig.Graph,
    labels: dict,
    show_edges: bool = False,
    max_starts: int | None = None,
    term_freq: list[float] | None = None,
    color_label: str = "% docs<br>with term",
    hover_label: str = "% of documents contain this term",
    raw_values: list | None = None,
    selected: set[int] | None = None,
    top_docs: int = 3,
    top_nouns: int = 5,
) -> go.Figure:
    """Render a cluster-level igraph as a Plotly bubble chart.

    Positions are computed by the stress-minimising layout in bubble_layout.py.
    Clusters are drawn as Plotly layout.shapes (type="circle") in data coordinates so
    that bubble radii exactly match the layout algorithm — using marker.size in pixels
    would produce correctly-sized markers but they would not match the non-overlap
    guarantees of the layout.

    An invisible scatter trace is drawn on top to supply hover tooltips (cluster ID,
    paper count, top paper title) and centred text labels.

    Args:
        cg:         Cluster-level igraph as returned by build_cluster_graph().
                    Required vertex attrs: community_id, size, top_title.
                    Required edge attr (if show_edges): weight.
        show_edges: Draw inter-cluster citation edges as grey lines. Hidden by default.
        max_starts: Maximum starting nodes for the bubble layout.
                    None  → try all n starting nodes (best quality, O(n³)).
                    int k → try the k most-connected starting nodes (O(k × n²)).
                    See bubble_layout.build_graph() for details.

    Returns:
        Plotly Figure with dark background (#1a1a2e), invisible axes, fixed 700 px height.

    Raises:
        RuntimeError: If the bubble layout fails to place all nodes (should not happen
                      in practice for well-formed cluster graphs).
    """
    from .bubble_layout import build_node_list, build_graph

    node_list = build_node_list(cg)
    placed = build_graph(node_list, max_starts=max_starts)

    if placed is None:
        raise RuntimeError("Bubble layout failed to place all nodes.")

    # Scale coordinates to a reasonable canvas
    coords = [n["coor"] for n in placed]
    radii  = [n["radius"] for n in placed]

    id_to_placed = {n["id"]: n for n in placed}

    shapes = []
    scatter_x, scatter_y, scatter_text, scatter_hover, scatter_colors = [], [], [], [], []

    fig = go.Figure()

    if show_edges:
        weights = cg.es["weight"] if cg.ecount() > 0 else []
        edge_x, edge_y = [], []
        for e in cg.es:
            src_id = cg.vs[e.source]["community_id"]
            dst_id = cg.vs[e.target]["community_id"]
            x0, y0 = id_to_placed[src_id]["coor"]
            x1, y1 = id_to_placed[dst_id]["coor"]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode="lines",
            line=dict(color="#4a4e69", width=1),
            hoverinfo="skip",
            showlegend=False,
        ))

    # Nodes: draw as shapes (data coordinates) so sizes match layout exactly
    meta = {v["community_id"]: v for v in cg.vs}
    for n in placed:
        cid = n["id"]
        v   = meta[cid]
        cx, cy = n["coor"]
        r = n["radius"]
        color = _gradient_color(term_freq[cid]) if term_freq is not None else _cluster_color(cid)
        is_selected = bool(selected and cid in selected)
        shapes.append(dict(
            type="circle",
            xref="x", yref="y",
            x0=cx - r, y0=cy - r,
            x1=cx + r, y1=cy + r,
            fillcolor=color,
            line=dict(color="#ffffff", width=14) if is_selected else dict(color="white", width=1),
            opacity=0.9,
            layer="below",
        ))

        info_v      = labels.get(cid, {})
        top_titles_v = info_v.get("top_titles", [])
        docs_html_v = "".join(
            f"<br>{i+1}. {t[:80] + '…' if len(t) > 80 else t}"
            for i, t in enumerate(top_titles_v[:top_docs]) if t
        )
        nouns_html_v = "".join(
            f"<br><span style='color:#aaa'>{i+1}. {n}</span>"
            for i, n in enumerate(info_v.get("top_nouns", [])[:top_nouns])
        )
        if term_freq is not None:
            if raw_values is not None:
                display = f"{int(raw_values[cid]):,}"
            else:
                display = f"{term_freq[cid]*100:.2f}%"
            metric_html_v = f"<br><b>{display} {hover_label}</b>"
        else:
            metric_html_v = ""
        hover = (
            f"<b>Cluster {cid}</b> &nbsp;·&nbsp; {v['size']:,} documents"
            f"{metric_html_v}"
            f"{docs_html_v}"
            f"{nouns_html_v}"
        )

        scatter_x.append(cx)
        scatter_y.append(cy)
        scatter_text.append(str(cid))
        scatter_colors.append(color)
        scatter_hover.append(hover)

    fig.add_trace(go.Scatter(
        x=scatter_x, y=scatter_y,
        mode="text",
        text=scatter_text,
        textfont=dict(color="#ffffff", size=24, family="Arial"),
        hovertext=scatter_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    if term_freq is not None:
        if raw_values is not None:
            max_raw = max(raw_values) if raw_values else 1
            # Colorbar in raw counts: 5 evenly spaced ticks
            step = max_raw / 4
            tick_vals_norm = [i * step / max_raw * _COLORBAR_MAX_PCT for i in range(5)]
            tick_texts = [f"{int(round(i * step)):,}" for i in range(4)] + [f"{int(max_raw):,}"]
            cb_cmin, cb_cmax = 0, _COLORBAR_MAX_PCT
        else:
            tick_vals_norm = [0.1, 0.5, 1, 2, 5, 10, 25]
            tick_texts = ["0.1%", "0.5%", "1%", "2%", "5%", "10%", "≥25%"]
            cb_cmin, cb_cmax = 0, _COLORBAR_MAX_PCT
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(
                colorscale=_sqrt_colorscale(),
                showscale=True,
                cmin=cb_cmin, cmax=cb_cmax,
                colorbar=dict(
                    tickvals=tick_vals_norm,
                    ticktext=tick_texts,
                    thickness=15,
                    len=0.9,
                    bgcolor="rgba(0,0,0,0.5)",
                    bordercolor="rgba(255,255,255,0.2)",
                    tickfont=dict(color="white"),
                    title=dict(text=color_label, font=dict(color="white")),
                ),
                size=1,
                opacity=0,
            ),
            hoverinfo="skip",
            showlegend=False,
        ))

    annotations = []
    if term_freq is None:
        annotations.append(dict(
            text="Colors are arbitrary<br>and only serve to<br>distinguish clusters",
            xref="paper", yref="paper",
            x=0.98, y=0.5,
            xanchor="right", yanchor="middle",
            showarrow=False,
            font=dict(color="rgba(255,255,255,0.4)", size=11, family="Arial"),
            align="right",
        ))

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        xaxis=dict(visible=False, scaleanchor="y"),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=700,
    )
    return fig
