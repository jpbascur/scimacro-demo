"""Generate small dummy data files for testing the 'Upload your own data' workflow.

Produces three files in data/test/:
    test_docs.csv     — id, title, abstract  (60 papers, 3 topics)
    test_network.csv  — source, target, weight  (dense within topics, sparse between)
    test_nouns.json   — paper_id → [noun lemmas]

Run with:
    python generate_test_data.py
"""
import json
import os
import random
import csv

random.seed(42)
OUT_DIR = os.path.join(os.path.dirname(__file__), "data", "test")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Topic definitions
# ---------------------------------------------------------------------------
TOPICS = {
    "ml": {
        "titles": [
            "Deep Learning for Image Classification",
            "Convolutional Neural Networks in Computer Vision",
            "Transformer Models for Natural Language Processing",
            "Reinforcement Learning in Robotics",
            "Graph Neural Networks for Node Classification",
            "Attention Mechanisms in Sequence Modelling",
            "Transfer Learning Across Domains",
            "Generative Adversarial Networks for Image Synthesis",
            "Self-Supervised Representation Learning",
            "Neural Architecture Search Methods",
            "Federated Learning for Privacy Preservation",
            "Recurrent Networks for Time Series Forecasting",
            "Contrastive Learning of Visual Representations",
            "Object Detection with Anchor-Free Methods",
            "Diffusion Models for Image Generation",
            "Sparse Autoencoders for Feature Extraction",
            "Multi-Task Learning in Neural Networks",
            "Knowledge Distillation for Model Compression",
            "Explainability in Deep Neural Networks",
            "Bayesian Optimisation for Hyperparameter Tuning",
        ],
        "abstract_words": [
            "neural network training accuracy loss gradient descent optimizer",
            "model performance benchmark dataset evaluation metric precision recall",
            "layer activation function weight initialisation batch normalisation dropout",
        ],
    },
    "networks": {
        "titles": [
            "Community Detection in Large-Scale Networks",
            "Spectral Clustering of Graph Structures",
            "Random Walk Models on Complex Networks",
            "Centrality Measures in Social Networks",
            "Epidemic Spreading on Contact Networks",
            "Network Resilience Under Random Failures",
            "Link Prediction in Knowledge Graphs",
            "Temporal Networks and Dynamic Graph Analysis",
            "Multiplex Networks and Layer Correlation",
            "Influence Maximisation in Social Media",
            "Motif Detection in Biological Networks",
            "Network Embedding Methods",
            "Power-Law Degree Distributions in Scale-Free Networks",
            "Synchronisation in Coupled Oscillator Networks",
            "Flow Networks and Bottleneck Analysis",
            "Hypergraph Modelling of Group Interactions",
            "Percolation Theory in Random Graphs",
            "Bipartite Network Projections",
            "Network Reconstruction from Partial Observations",
            "Cascade Failures in Interdependent Networks",
        ],
        "abstract_words": [
            "graph node edge degree adjacency matrix clustering coefficient",
            "network topology connectivity path length shortest path betweenness",
            "community structure modularity partition algorithm convergence",
        ],
    },
    "climate": {
        "titles": [
            "Global Temperature Trends Under Greenhouse Gas Scenarios",
            "Ocean Heat Uptake and Sea Level Rise Projections",
            "Arctic Sea Ice Decline and Albedo Feedback",
            "Carbon Cycle Responses to Land Use Change",
            "Extreme Precipitation Events in a Warming Climate",
            "Tropical Cyclone Intensification Under Climate Change",
            "Permafrost Thaw and Methane Release",
            "Drought Frequency in Semi-Arid Regions",
            "Monsoon Variability and Climate Teleconnections",
            "Coral Bleaching and Ocean Acidification",
            "Glacier Mass Balance in Mountain Regions",
            "Urban Heat Islands and Mitigation Strategies",
            "Wildfire Risk in Mediterranean Ecosystems",
            "Biodiversity Loss Under Climate Scenarios",
            "Crop Yield Impacts of Rising Temperatures",
            "Solar Radiation Management as Climate Intervention",
            "Climate Model Uncertainty and Ensemble Methods",
            "Attribution of Extreme Events to Climate Change",
            "Marine Ecosystem Shifts Under Warming Oceans",
            "Freshwater Availability Under Future Climate",
        ],
        "abstract_words": [
            "temperature warming emission scenario projection model simulation",
            "ocean atmosphere feedback forcing response sensitivity equilibrium",
            "species ecosystem habitat carbon dioxide concentration threshold",
        ],
    },
}

# ---------------------------------------------------------------------------
# Build paper list
# ---------------------------------------------------------------------------
papers = []
pid = 1
topic_ids = {}
for topic, data in TOPICS.items():
    topic_ids[topic] = []
    for title in data["titles"]:
        abstract = " ".join(random.choices(data["abstract_words"], k=3))
        papers.append({"id": str(pid), "title": title, "abstract": abstract, "topic": topic})
        topic_ids[topic].append(str(pid))
        pid += 1

# ---------------------------------------------------------------------------
# docs CSV
# ---------------------------------------------------------------------------
docs_path = os.path.join(OUT_DIR, "test_docs.csv")
with open(docs_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "title", "abstract"])
    w.writeheader()
    for p in papers:
        w.writerow({"id": p["id"], "title": p["title"], "abstract": p["abstract"]})
print(f"Wrote {docs_path}  ({len(papers)} papers)")

# ---------------------------------------------------------------------------
# network CSV — dense within topics, sparse between
# ---------------------------------------------------------------------------
edges = {}

def add_edge(u, v, w=1.0):
    key = (min(u, v), max(u, v))
    edges[key] = edges.get(key, 0.0) + w

# Within-topic: each paper connected to ~8 others in same topic
for topic, ids in topic_ids.items():
    for i, u in enumerate(ids):
        neighbours = random.sample([v for v in ids if v != u], min(8, len(ids) - 1))
        for v in neighbours:
            add_edge(u, v, random.uniform(1.0, 3.0))

# Between topics: ~5 cross-edges per topic pair (sparse)
topic_list = list(topic_ids.keys())
for i in range(len(topic_list)):
    for j in range(i + 1, len(topic_list)):
        for _ in range(5):
            u = random.choice(topic_ids[topic_list[i]])
            v = random.choice(topic_ids[topic_list[j]])
            add_edge(u, v, 0.5)

network_path = os.path.join(OUT_DIR, "test_network.csv")
with open(network_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["source", "target", "weight"])
    for (u, v), wt in edges.items():
        w.writerow([u, v, f"{wt:.2f}"])
print(f"Wrote {network_path}  ({len(edges)} edges)")

# ---------------------------------------------------------------------------
# nouns JSON — simple word extraction (no spaCy needed for dummy data)
# ---------------------------------------------------------------------------
import re

STOPWORDS = {
    "in", "on", "of", "and", "for", "the", "a", "an", "to", "with",
    "under", "from", "by", "at", "as", "is", "are", "its", "via",
}

nouns = {}
for p in papers:
    text = p["title"] + " " + p["abstract"]
    words = re.findall(r"[a-z]+", text.lower())
    lemmas = sorted({w for w in words if len(w) > 3 and w not in STOPWORDS})
    nouns[p["id"]] = lemmas

nouns_path = os.path.join(OUT_DIR, "test_nouns.json")
with open(nouns_path, "w", encoding="utf-8") as f:
    json.dump(nouns, f, indent=2)
print(f"Wrote {nouns_path}  ({len(nouns)} papers)")

print("\nTest files ready in data/test/")
print("Upload them via the 'Upload your own data' panel in the app.")
