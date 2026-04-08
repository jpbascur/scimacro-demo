"""Noun-based cluster labeling, SPECTER2 embeddings, and per-cluster term frequency.

Requires spaCy and the small English model:
    pip install spacy
    python -m spacy download en_core_web_sm

For SPECTER2 embeddings:
    pip install torch transformers adapters
    python precompute.py  (downloads model on first run)
"""
import os
from collections import defaultdict

try:
    import spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

_nlp = None

# Generic academic nouns that carry no cluster-specific meaning
_GENERIC = {
    "study", "analysis", "approach", "method", "paper", "result",
    "use", "application", "work", "model", "system", "framework",
    "research", "datum", "data", "review", "effect", "impact", "role",
    "type", "form", "way", "part", "number", "year", "time",
    "level", "rate", "factor", "case", "group", "set", "issue",
    "problem", "question", "process", "structure", "function",
    "example", "point", "value", "term", "field", "area", "topic",
    "sample", "article", "author", "journal", "publication", "finding",
}


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    return _nlp


def extract_nouns(metadata: dict) -> dict[str, set[str]]:
    """Extract noun lemmas from title + abstract for each paper.

    Returns dict mapping paper_id → set of lowercase noun lemmas.
    Returns an empty dict if spaCy is not available.
    """
    if not _SPACY_AVAILABLE:
        return {}

    nlp = _get_nlp()
    paper_ids = list(metadata.keys())
    texts = [
        (metadata[pid].get("title") or "") + " " + (metadata[pid].get("abstract") or "")
        for pid in paper_ids
    ]

    result: dict[str, set[str]] = {}
    for pid, doc in zip(paper_ids, nlp.pipe(texts, batch_size=256)):
        result[pid] = {
            token.lemma_.lower()
            for token in doc
            if token.pos_ == "NOUN"
            and token.is_alpha
            and len(token.text) > 2
            and token.lemma_.lower() not in _GENERIC
        }
    return result


def label_clusters(
    cg,
    labels: dict[int, dict],
    paper_nouns: dict[str, set[str]],
    membership: dict[str, int],
    m: int = 25,
) -> None:
    """Update the labels dict with noun-based cluster labels and top nouns.

    Scoring formula: score(term, cluster) = n_ut / (n_map_t + m), where
    n_ut is the number of papers in the cluster containing the term,
    n_map_t is the number of papers across the whole map containing the term,
    and m is a smoothing parameter. Higher m reduces the influence of terms
    that appear in only a few papers.

    Updates labels[cid]["label"] and labels[cid]["top_nouns"] (list of str)
    in-place. Falls back to the existing label if no nouns are available.
    Does nothing if paper_nouns is empty.

    Args:
        cg:          Cluster-level igraph (used only for vcount).
        labels:      Labels dict as returned by build_cluster_graph().
        paper_nouns: Dict mapping paper_id → set of noun lemmas.
        membership:  Dict mapping paper_id → cluster_id.
        m:           Smoothing parameter.
    """
    if not paper_nouns:
        return

    n = cg.vcount()
    counts: list[dict[str, int]] = [defaultdict(int) for _ in range(n)]

    for pid, nouns in paper_nouns.items():
        c = membership.get(pid)
        if c is None or c >= n:
            continue
        for noun in nouns:
            counts[c][noun] += 1

    # Total papers containing each noun across the entire map
    map_freq: dict[str, int] = defaultdict(int)
    for c_counts in counts:
        for noun, cnt in c_counts.items():
            map_freq[noun] += cnt

    for c in range(n):
        if not counts[c]:
            labels[c]["top_nouns"] = []
            continue
        scores = {
            noun: cnt / (map_freq[noun] + m)
            for noun, cnt in counts[c].items()
        }
        top = sorted(scores, key=scores.__getitem__, reverse=True)[:10]
        labels[c]["label"]     = " / ".join(top[:3]) if top else labels[c]["label"]
        labels[c]["top_nouns"] = top


def cluster_term_frequency(
    cg,
    term: str,
    paper_nouns: dict[str, set[str]],
    membership: dict[str, int],
) -> list[float]:
    """Return fraction of papers per cluster containing *term* (binary per paper).

    The search term is lemmatized before lookup so e.g. 'networks' matches 'network'.
    Returns a list of zeros if paper_nouns is empty or term is blank.
    """
    n = cg.vcount()
    if not paper_nouns or not term.strip():
        return [0.0] * n

    nlp = _get_nlp()
    doc = nlp(term.strip().lower())
    lemma = doc[0].lemma_.lower() if doc else term.strip().lower()

    hits = [0] * n
    totals = [0] * n
    for pid, nouns in paper_nouns.items():
        c = membership.get(pid)
        if c is None or c >= n:
            continue
        totals[c] += 1
        if lemma in nouns:
            hits[c] += 1

    return [hits[c] / totals[c] if totals[c] > 0 else 0.0 for c in range(n)]


# ---------------------------------------------------------------------------
# SPECTER2 embeddings
# ---------------------------------------------------------------------------

SPECTER2_ADAPTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "specter2_adapter"
)
SPECTER2_BATCH_SIZE = 128
SPECTER2_MAX_LENGTH = 256


def extract_embeddings(metadata: dict) -> dict[str, list[float]]:
    """Embed each paper using SPECTER2 (title + SEP + abstract).

    Returns dict mapping paper_id → 768-dim embedding as a Python list.
    Returns empty dict if torch/transformers/adapters are not installed.

    Uses GPU if available. Processes papers in batches for efficiency.
    """
    try:
        import torch
        from transformers import AutoTokenizer
        from adapters import AutoAdapterModel
    except ImportError:
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    model.load_adapter(SPECTER2_ADAPTER_PATH, load_as="specter2", set_active=True)
    model.eval().to(device)

    paper_ids = list(metadata.keys())
    texts = [
        (metadata[pid].get("title") or "") +
        tokenizer.sep_token +
        (metadata[pid].get("abstract") or "")
        for pid in paper_ids
    ]

    # Sort by token length so each batch has similar-length papers, minimising padding waste.
    # This does not affect results — each paper still gets its own full embedding.
    order = sorted(range(len(paper_ids)), key=lambda i: len(tokenizer.encode(texts[i])))
    paper_ids = [paper_ids[i] for i in order]
    texts     = [texts[i]     for i in order]

    result: dict[str, list[float]] = {}
    for i in range(0, len(paper_ids), SPECTER2_BATCH_SIZE):
        batch_ids   = paper_ids[i: i + SPECTER2_BATCH_SIZE]
        batch_texts = texts[i: i + SPECTER2_BATCH_SIZE]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=False,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs)
        embeddings = out.last_hidden_state[:, 0, :].cpu().tolist()
        for pid, emb in zip(batch_ids, embeddings):
            result[pid] = emb

    return result
