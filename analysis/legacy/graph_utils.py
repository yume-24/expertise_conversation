# graph_utils.py
import networkx as nx
import pandas as pd

def graph_stats(G: nx.DiGraph, graph_id: str = "") -> pd.DataFrame:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    U = G.to_undirected()

    def _safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    # --- FIX: guard n==0 and divide by n for both in/out ---
    avg_in  = (sum(dict(G.in_degree()).values())  / n) if n else 0.0
    avg_out = (sum(dict(G.out_degree()).values()) / n) if n else 0.0
    density = nx.density(G)

    # clustering, connectivity
    avg_clust = _safe(lambda: nx.average_clustering(U))
    weak_cc   = nx.number_weakly_connected_components(G) if n else 0
    strong_cc = nx.number_strongly_connected_components(G) if n else 0

    # --- FIX: compute path stats on UNDIRECTED LCC (not directed) ---
    if n > 1 and m > 0:
        lcc_nodes = max(nx.weakly_connected_components(G), key=len)
        LCCu = G.subgraph(lcc_nodes).to_undirected()
        avg_path = _safe(lambda: nx.average_shortest_path_length(LCCu))
        diameter = _safe(lambda: nx.diameter(LCCu))
    else:
        avg_path = diameter = None

    # reciprocity & assortativity
    reciprocity  = _safe(lambda: nx.reciprocity(G))
    assortativity = _safe(lambda: nx.degree_pearson_correlation_coefficient(U))

    # edge-weight summaries
    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    avg_weight = (sum(weights) / len(weights))
    max_weight = max(weights) if weights else None

    stats = {
        "n_nodes": n,
        "n_edges": m,
        "density": density,
        "avg_in_degree": avg_in,
        "avg_out_degree": avg_out,
        "avg_clustering": avg_clust,
        "weak_components": weak_cc,
        "strong_components": strong_cc,
        "reciprocity": reciprocity,
        "assortativity": assortativity,
        "avg_shortest_path_length": avg_path,   # <-- will now populate
        "diameter": diameter,
        "avg_edge_weight": avg_weight,
        "max_edge_weight": max_weight,
    }

    df = pd.DataFrame([stats])
    if graph_id:
        df.index = [graph_id]
    return df
