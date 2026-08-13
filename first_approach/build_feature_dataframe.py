from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

from clean_corpus_make_graph import clean_corpus, read_corpus, text_to_graph
from graph_utils import *

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from main import DATA_DIR, EXPERT_DIR, NOVICE_DIR

# config stuff

DATA_DIR = Path('data')
EXPERT_DIR = DATA_DIR / 'expert'
NOVICE_DIR = DATA_DIR / 'novice'
OUTPUT_DIR = Path('outputs')
OUTPUT_DIR.mkdir(exist_ok = True)

EMB_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

#emb features

def embedding_stats(sentences):
    if len(sentences)<2:
        return {}
    E = EMB_MODEL.encode(sentences)

    sims = cosine_similarity(E)
    upper = sims[np.triu_indices_from(sims, 1)]

    pca = PCA(n_components = 2)
    pca.fit(E)

    return{
        'emb_mean_norm' : np.linalg.norm(E.mean(0)),
        'emb_variance': E.var(),
        'avg_pairwise_cosine': upper.mean(),
        'pca_var_1': pca.explained_variance_ratio_[1],
        "pca_var_2": pca.explained_variance_ratio_[1] if pca.n_components_ > 1 else 0.0
    }


# hybrid features
def hybrid_graph_embedding_stats(G: nx.DiGraph, word_vecs: dict):
    distances = []

    for u, v in G.edges():
        if u in word_vecs and v in word_vecs:
            d = 1 - cosine_similarity(
                word_vecs[u].reshape(1, -1),
                word_vecs[v].reshape(1, -1)
            )[0, 0]
            distances.append(d)

    if not distances:
        return {}

    distances = np.array(distances)

    bet = nx.betweenness_centrality(G)
    semantic_bet = np.mean([
        bet[n] * np.std([
            word_vecs[m]
            for m in G.neighbors(n)
            if m in word_vecs
        ])
        for n in G.nodes if n in word_vecs
    ])

    return {
        "mean_edge_semantic_distance": distances.mean(),
        "std_edge_semantic_distance": distances.std(),
        "long_edge_ratio": np.mean(distances > np.percentile(distances, 90)),
        "semantic_betweenness": semantic_bet,
    }


# process one folder
def process_folder(folder: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(folder.glob("*.txt")):
        text = read_corpus(str(p))
        cleaned = clean_corpus(text, stop_words=None)
        G = text_to_graph(cleaned)
        df = graph_stats(G, graph_id=p.stem)
        rows.append(df)
    return pd.concat(rows) if rows else pd.DataFrame()

