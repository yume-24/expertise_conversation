#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
conversation_embedding_metrics.py

Compute embedding-based interaction metrics for conversation CSVs.
Outputs one row per conversation file.

Input CSV columns required:
  - Sequence (numeric or sortable)
  - Speaker
  - Utterance

Usage:
  # single file
  python conversation_embedding_metrics.py --input data/wired/wired_Talia_12.csv --output convo_metrics.csv

  # directory (process all *.csv)
  python conversation_embedding_metrics.py --input data/wired/ --output convo_metrics.csv

Optional:
  --speaker-a "Speaker 1" --speaker-b "Speaker 5"  (to force speaker pair selection)
  --model all-MiniLM-L6-v2
  --k 8  (clusters for entropy/MI; default auto)

Dependencies:
  pip install pandas numpy scikit-learn sentence-transformers
Optional (recommended for hull metrics; script falls back gracefully if missing):
  pip install scipy
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics import mutual_info_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Optional SciPy acceleration (hull, Hausdorff, fast NN)
try:
    from scipy.spatial import ConvexHull, Delaunay, cKDTree
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ---------------------------
# Utilities
# ---------------------------

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    s = counts.sum()
    if s <= 0:
        return np.nan
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Slope of y ~ a + b*x. Returns b."""
    if len(x) < 2:
        return np.nan
    x = x.astype(float)
    y = y.astype(float)
    x0 = x - x.mean()
    denom = np.dot(x0, x0)
    if denom == 0:
        return np.nan
    return float(np.dot(x0, y - y.mean()) / denom)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # 1 - cosine similarity
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan
    return float(1.0 - (np.dot(a, b) / (na * nb)))

def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def pick_top2_speakers(df: pd.DataFrame) -> Tuple[str, str]:
    counts = df["Speaker"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need at least two speakers in the CSV.")
    top2 = counts.index[:2].tolist()
    return top2[0], top2[1]

def ensure_two_speakers(df: pd.DataFrame, speaker_a: Optional[str], speaker_b: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    if speaker_a and speaker_b:
        df2 = df[df["Speaker"].isin([speaker_a, speaker_b])].copy()
        if df2["Speaker"].nunique() < 2:
            raise ValueError(f"Could not find both speakers: {speaker_a}, {speaker_b}")
        return df2, speaker_a, speaker_b

    a, b = pick_top2_speakers(df)
    df2 = df[df["Speaker"].isin([a, b])].copy()
    return df2, a, b


# ---------------------------
# NN distances / Hausdorff
# ---------------------------

def mean_cross_nn_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Mean over points in A of min distance to B (euclidean)."""
    if len(A) == 0 or len(B) == 0:
        return np.nan
    if SCIPY_OK:
        tree = cKDTree(B)
        d, _ = tree.query(A, k=1)
        return float(np.mean(d))
    # fallback: O(n^2) (fine for smallish)
    dmins = []
    for a in A:
        dmins.append(np.min(np.linalg.norm(B - a, axis=1)))
    return float(np.mean(dmins))

def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Symmetric Hausdorff distance between point sets."""
    if len(A) == 0 or len(B) == 0:
        return np.nan
    dAB = mean_cross_nn_distance(A, B)
    dBA = mean_cross_nn_distance(B, A)
    return float(max(dAB, dBA))


# ---------------------------
# Convex hull overlap via Monte Carlo (2D)
# ---------------------------

def hull_area(points_2d: np.ndarray) -> float:
    if len(points_2d) < 3:
        return 0.0
    if not SCIPY_OK:
        return np.nan
    return float(ConvexHull(points_2d).volume)  # in 2D, .volume = area

def points_in_hull(points_2d: np.ndarray, hull_points_2d: np.ndarray) -> np.ndarray:
    """Return boolean mask of points inside convex hull."""
    if len(hull_points_2d) < 3 or not SCIPY_OK:
        return np.zeros(len(points_2d), dtype=bool)
    delaunay = Delaunay(hull_points_2d)
    return delaunay.find_simplex(points_2d) >= 0

def hull_overlap_stats(A2: np.ndarray, B2: np.ndarray, n_samples: int = 4000) -> Dict[str, float]:
    """
    Approximate overlap area between hulls(A) and hulls(B) using Monte Carlo.
    Returns:
      hull_area_A, hull_area_B, overlap_area, overlap_ratio_min, containment_A_in_B, containment_B_in_A
    """
    out = {
        "hull_area_A": np.nan,
        "hull_area_B": np.nan,
        "hull_overlap_area": np.nan,
        "hull_overlap_ratio_min": np.nan,
        "containment_A_in_B": np.nan,
        "containment_B_in_A": np.nan,
    }

    if not SCIPY_OK or len(A2) < 3 or len(B2) < 3:
        return out

    areaA = hull_area(A2)
    areaB = hull_area(B2)
    out["hull_area_A"] = areaA
    out["hull_area_B"] = areaB

    # bounding box over union
    U = np.vstack([A2, B2])
    xmin, ymin = U.min(axis=0)
    xmax, ymax = U.max(axis=0)
    if xmax == xmin or ymax == ymin:
        return out

    rng = np.random.default_rng(42)
    samples = np.column_stack([
        rng.uniform(xmin, xmax, size=n_samples),
        rng.uniform(ymin, ymax, size=n_samples),
    ])

    inA = points_in_hull(samples, A2)
    inB = points_in_hull(samples, B2)
    inBoth = inA & inB

    box_area = (xmax - xmin) * (ymax - ymin)
    overlap_area = float(inBoth.mean() * box_area)
    out["hull_overlap_area"] = overlap_area

    denom = min(areaA, areaB)
    if denom > 0:
        out["hull_overlap_ratio_min"] = float(overlap_area / denom)

    # containment: proportion of B points inside A hull, and vice versa
    out["containment_B_in_A"] = float(points_in_hull(B2, A2).mean())
    out["containment_A_in_B"] = float(points_in_hull(A2, B2).mean())

    return out


# ---------------------------
# Core metric computation
# ---------------------------

def embed_utterances(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    E = model.encode(texts, show_progress_bar=False)
    return np.asarray(E, dtype=float)

def compute_conversation_metrics(
    df: pd.DataFrame,
    convo_id: str,
    model: SentenceTransformer,
    speaker_a: Optional[str] = None,
    speaker_b: Optional[str] = None,
    k_clusters: Optional[int] = None,
) -> Dict[str, float]:
    required = {"Sequence", "Speaker", "Utterance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{convo_id}: missing columns {missing}")

    df = df.copy()
    df["Utterance"] = df["Utterance"].fillna("").astype(str)

    # sort by sequence
    df = df.sort_values("Sequence").reset_index(drop=True)

    # keep 2 speakers
    df, A_name, B_name = ensure_two_speakers(df, speaker_a, speaker_b)

    df = df.sort_values("Sequence").reset_index(drop=True)

    # embed each utterance (one vector per utterance)
    texts = df["Utterance"].tolist()
    E = embed_utterances(model, texts)

    # indices by speaker
    spk = df["Speaker"].to_numpy()
    idxA = np.where(spk == A_name)[0]
    idxB = np.where(spk == B_name)[0]

    EA = E[idxA]
    EB = E[idxB]

    out: Dict[str, float] = {}
    out["convo_id"] = convo_id
    out["speaker_A"] = A_name
    out["speaker_B"] = B_name
    out["n_utterances"] = float(len(df))
    out["n_A"] = float(len(idxA))
    out["n_B"] = float(len(idxB))

    # -------------------------
    # 1) Centroid distance over time (cumulative)
    # -------------------------
    # Build cumulative centroids at each time step t for each speaker using utterances up to t
    muA = np.zeros((len(df), E.shape[1]))
    muB = np.zeros((len(df), E.shape[1]))
    countA = 0
    countB = 0
    sumA = np.zeros(E.shape[1])
    sumB = np.zeros(E.shape[1])

    for t in range(len(df)):
        if spk[t] == A_name:
            sumA += E[t]; countA += 1
        else:
            sumB += E[t]; countB += 1

        muA[t] = sumA / max(countA, 1)
        muB[t] = sumB / max(countB, 1)

    d_cent = np.array([euclid(muA[t], muB[t]) for t in range(len(df))], dtype=float)

    out["centroid_dist_mean"] = float(np.nanmean(d_cent))
    out["centroid_dist_std"] = float(np.nanstd(d_cent))
    out["centroid_dist_slope"] = linear_slope(np.arange(len(df)), d_cent)

    # -------------------------
    # 2) Cross-speaker nearest neighbor distances (global)
    # -------------------------
    out["cross_nn_A_to_B"] = mean_cross_nn_distance(EA, EB)
    out["cross_nn_B_to_A"] = mean_cross_nn_distance(EB, EA)
    out["hausdorff_AB"] = hausdorff_distance(EA, EB)

    # -------------------------
    # 3) Directional semantic influence (toward other’s centroid)
    #    Compute separately: A influenced by B, and B influenced by A
    # -------------------------
    # For each utterance by A at time t, compute delta in A centroid from previous A utterance,
    # and compare to vector from A centroid (prev) toward B centroid (at t-1).
    def directional_influence(target_name: str, other_name: str) -> float:
        t_idx = np.where(spk == target_name)[0]
        if len(t_idx) < 2:
            return np.nan
        vals = []
        prev_t = t_idx[0]
        # Need centroid at times; use muA/muB arrays we computed cumulatively
        for curr_t in t_idx[1:]:
            # delta in target centroid between prev_t and curr_t
            mu_target_prev = muA[prev_t] if target_name == A_name else muB[prev_t]
            mu_target_curr = muA[curr_t] if target_name == A_name else muB[curr_t]
            delta = mu_target_curr - mu_target_prev

            # vector toward other centroid at time prev_t
            mu_other_prev = muB[prev_t] if other_name == B_name else muA[prev_t]
            toward = mu_other_prev - mu_target_prev

            # cosine similarity between delta and toward
            nd = np.linalg.norm(delta)
            nt = np.linalg.norm(toward)
            if nd == 0 or nt == 0:
                continue
            vals.append(float(np.dot(delta, toward) / (nd * nt)))
            prev_t = curr_t
        return float(np.mean(vals)) if vals else np.nan

    out["influence_A_toward_B"] = directional_influence(A_name, B_name)
    out["influence_B_toward_A"] = directional_influence(B_name, A_name)
    out["influence_asymmetry_AminusB"] = safe_float(out["influence_A_toward_B"] - out["influence_B_toward_A"])

    # -------------------------
    # 4) Turn-based semantic jump distance
    # -------------------------
    # For each turn transition where speaker changes, distance between consecutive utterance embeddings
    jumps = []
    jumps_A_to_B = []
    jumps_B_to_A = []
    for t in range(1, len(df)):
        if spk[t] != spk[t-1]:
            d = euclid(E[t], E[t-1])
            jumps.append(d)
            if spk[t-1] == A_name and spk[t] == B_name:
                jumps_A_to_B.append(d)
            elif spk[t-1] == B_name and spk[t] == A_name:
                jumps_B_to_A.append(d)

    out["turn_jump_mean"] = float(np.mean(jumps)) if jumps else np.nan
    out["turn_jump_std"] = float(np.std(jumps)) if jumps else np.nan
    out["turn_jump_A_to_B_mean"] = float(np.mean(jumps_A_to_B)) if jumps_A_to_B else np.nan
    out["turn_jump_B_to_A_mean"] = float(np.mean(jumps_B_to_A)) if jumps_B_to_A else np.nan

    # -------------------------
    # 5) Geometry metrics (2D PCA hull overlap/containment)
    # -------------------------
    # Project utterance embeddings to 2D
    pca = PCA(n_components=2, random_state=0)
    E2 = pca.fit_transform(E)
    A2 = E2[idxA]
    B2 = E2[idxB]

    hull_stats = hull_overlap_stats(A2, B2)  # returns NaNs if SciPy missing
    out.update(hull_stats)

    # -------------------------
    # 6) Semantic entropy and Mutual Information via clustering
    # -------------------------
    n = len(df)
    if k_clusters is None:
        # heuristic: between 4 and 12, but not more than n/3
        k = int(round(math.sqrt(max(n, 1))))
        k = max(4, min(12, k))
        k = min(k, max(2, n // 3)) if n >= 6 else 2
    else:
        k = int(k_clusters)

    # If too few points, skip
    if n < 4:
        out["k_clusters"] = float(k)
        out["entropy_A"] = np.nan
        out["entropy_B"] = np.nan
        out["entropy_total"] = np.nan
        out["mi_speaker_cluster"] = np.nan
        out["nmi_speaker_cluster"] = np.nan
        out["mi_turn_clusters"] = np.nan
        out["nmi_turn_clusters"] = np.nan
    else:
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        labels = km.fit_predict(E)  # cluster label per utterance

        out["k_clusters"] = float(k)

        # entropy per speaker and overall
        counts_total = np.bincount(labels, minlength=k)
        out["entropy_total"] = entropy_from_counts(counts_total)

        counts_A = np.bincount(labels[idxA], minlength=k) if len(idxA) else np.zeros(k)
        counts_B = np.bincount(labels[idxB], minlength=k) if len(idxB) else np.zeros(k)
        out["entropy_A"] = entropy_from_counts(counts_A)
        out["entropy_B"] = entropy_from_counts(counts_B)

        # MI between speaker ID and cluster label (how strongly clusters separate speakers)
        speaker_binary = (spk == A_name).astype(int)
        out["mi_speaker_cluster"] = float(mutual_info_score(speaker_binary, labels))
        out["nmi_speaker_cluster"] = float(normalized_mutual_info_score(speaker_binary, labels))

        # Interaction MI: cluster(label at t-1) vs cluster(label at t) for *speaker-change turns only*
        prev_clusters = []
        next_clusters = []
        for t in range(1, n):
            if spk[t] != spk[t-1]:
                prev_clusters.append(labels[t-1])
                next_clusters.append(labels[t])
        if len(prev_clusters) >= 2:
            out["mi_turn_clusters"] = float(mutual_info_score(prev_clusters, next_clusters))
            out["nmi_turn_clusters"] = float(normalized_mutual_info_score(prev_clusters, next_clusters))
        else:
            out["mi_turn_clusters"] = np.nan
            out["nmi_turn_clusters"] = np.nan

        # Entropy growth slopes (cumulative over time) per speaker
        def entropy_slope_for_speaker(target_name: str) -> float:
            # compute entropy of target's cluster distribution cumulatively
            idx = np.where(spk == target_name)[0]
            if len(idx) < 4:
                return np.nan
            cum_counts = np.zeros(k)
            ent = []
            tvals = []
            for j, t in enumerate(idx):
                cum_counts[labels[t]] += 1
                ent.append(entropy_from_counts(cum_counts))
                tvals.append(j)
            return linear_slope(np.array(tvals), np.array(ent))

        out["entropy_slope_A"] = entropy_slope_for_speaker(A_name)
        out["entropy_slope_B"] = entropy_slope_for_speaker(B_name)

    # -------------------------
    # 7) Alignment lag (both directions)
    # -------------------------
    # Define epsilon as a percentile of cross distances between utterances
    # (Smaller epsilon -> stricter “entered same region” criterion)
    def alignment_lag(source_idx: np.ndarray, target_idx: np.ndarray, eps: float) -> float:
        if len(source_idx) == 0 or len(target_idx) == 0:
            return np.nan
        # for each source utterance time i, find earliest target time j>i within eps
        lags = []
        for i in source_idx:
            # only look forward in time
            future_targets = target_idx[target_idx > i]
            if len(future_targets) == 0:
                continue
            # compute distances to future targets
            d = np.linalg.norm(E[future_targets] - E[i], axis=1)
            ok = np.where(d < eps)[0]
            if len(ok) == 0:
                continue
            j = future_targets[ok[0]]
            lags.append(j - i)
        return float(np.mean(lags)) if lags else np.nan

    # compute epsilon from cross distances (sampled if large)
    if len(idxA) > 0 and len(idxB) > 0:
        # sample up to 2000 pairs for stability
        rng = np.random.default_rng(0)
        sampleA = rng.choice(idxA, size=min(len(idxA), 60), replace=False)
        sampleB = rng.choice(idxB, size=min(len(idxB), 60), replace=False)
        pairs = []
        for i in sampleA:
            for j in sampleB:
                pairs.append(np.linalg.norm(E[i] - E[j]))
        pairs = np.array(pairs)
        eps = float(np.percentile(pairs, 25))  # adjustable; 25th percentile is “moderately strict”
    else:
        eps = np.nan

    out["align_eps"] = eps
    out["align_lag_A_to_B"] = alignment_lag(idxA, idxB, eps) if not np.isnan(eps) else np.nan
    out["align_lag_B_to_A"] = alignment_lag(idxB, idxA, eps) if not np.isnan(eps) else np.nan
    out["align_lag_asymmetry_AminusB"] = safe_float(out["align_lag_A_to_B"] - out["align_lag_B_to_A"])

    return out


# ---------------------------
# Batch processing
# ---------------------------

def iter_csv_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".csv":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.glob("*.csv"))
    raise ValueError(f"Input must be a CSV file or a directory of CSVs: {input_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="CSV file or directory containing CSVs")
    ap.add_argument("--output", type=str, required=True, help="Output CSV path for metrics")
    ap.add_argument("--speaker-a", type=str, default=None, help="Optional: force Speaker A name")
    ap.add_argument("--speaker-b", type=str, default=None, help="Optional: force Speaker B name")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    ap.add_argument("--k", type=int, default=None, help="Optional: number of clusters for entropy/MI")
    args = ap.parse_args()

    input_path = Path(args.input)
    files = iter_csv_files(input_path)
    if len(files) == 0:
        raise SystemExit("No CSV files found.")

    print(f"Found {len(files)} file(s). Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    rows = []
    for f in files:
        try:
            df = pd.read_csv(f)
            convo_id = f.stem
            row = compute_conversation_metrics(
                df=df,
                convo_id=convo_id,
                model=model,
                speaker_a=args.speaker_a,
                speaker_b=args.speaker_b,
                k_clusters=args.k,
            )
            rows.append(row)
            print(f"✓ {convo_id}")
        except Exception as e:
            print(f"✗ {f.name}: {e}")

    out_df = pd.DataFrame(rows)

    # make sure stable column order: identifiers first
    front = ["convo_id", "speaker_A", "speaker_B", "n_utterances", "n_A", "n_B"]
    cols = front + [c for c in out_df.columns if c not in front]
    out_df = out_df[cols]

    out_path = Path(args.output)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to: {out_path.resolve()}")
    print(out_df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
