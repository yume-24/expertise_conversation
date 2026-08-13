#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
visualize_conversation_metrics.py

Visual, step-by-step illustrations of how each embedding-based conversation metric
is computed.

Design goals
------------
1) Coherent with the *actual* computations in `conversation_embedding_metrics.py`.
2) Every utterance is plotted as a point in a shared 2D projection, labeled
   as `<A_or_B>_<t>` (e.g., `A_2`, `B_7`) where `t` is the time index after sorting
   by the `Sequence` column.
3) Either produce:
   - a static multi-panel figure (default), or
   - per-metric animations (optional).

Inputs
------
CSV columns required:
  - Sequence
  - Speaker
  - Utterance

Usage examples
--------------
  # Static multi-panel figure
  python visualize_conversation_metrics.py \
    --input wired_moravec_16.csv \
    --outdir viz_out

  # Animate centroid distance over time
  python visualize_conversation_metrics.py \
    --input wired_moravec_16.csv \
    --outdir viz_out \
    --animate centroid

Dependencies
------------
  pip install numpy pandas matplotlib scikit-learn sentence-transformers
Optional (enables hull visuals and fast NN):
  pip install scipy

Notes
-----
The 2D plot is for *illustration*. Metrics such as NN distances, epsilon threshold,
and alignment lags are computed in the *original embedding dimension*, matching
`conversation_embedding_metrics.py`. We only *draw* the matched relationships in 2D.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score

# Sentence embeddings (same as metrics script)
from sentence_transformers import SentenceTransformer

# Optional SciPy for hull + KDTree (same spirit as metrics script)
try:
    from scipy.spatial import ConvexHull, cKDTree
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# -----------------------------
# Core math (matches metrics script)
# -----------------------------

def euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def linear_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return np.nan
    x = x.astype(float)
    y = y.astype(float)
    x0 = x - x.mean()
    denom = float(np.dot(x0, x0))
    if denom == 0:
        return np.nan
    return float(np.dot(x0, y - y.mean()) / denom)

def entropy_from_counts(counts: np.ndarray) -> float:
    counts = counts.astype(float)
    s = counts.sum()
    if s <= 0:
        return np.nan
    p = counts / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

def mean_cross_nn_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Mean over points in A of min distance to B (Euclidean)."""
    if len(A) == 0 or len(B) == 0:
        return np.nan
    if SCIPY_OK:
        tree = cKDTree(B)
        d, _ = tree.query(A, k=1)
        return float(np.mean(d))
    # fallback: O(n^2)
    dmins = []
    for a in A:
        dmins.append(float(np.min(np.linalg.norm(B - a, axis=1))))
    return float(np.mean(dmins))

def hausdorff_soft(A: np.ndarray, B: np.ndarray) -> float:
    """Matches your script: max(mean A->B NN, mean B->A NN)."""
    if len(A) == 0 or len(B) == 0:
        return np.nan
    dAB = mean_cross_nn_distance(A, B)
    dBA = mean_cross_nn_distance(B, A)
    return float(max(dAB, dBA))

def compute_cumulative_centroids(E: np.ndarray, spk_is_A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return muA[t], muB[t] cumulative over time (exactly like your metrics script)."""
    n, d = E.shape
    muA = np.zeros((n, d), dtype=float)
    muB = np.zeros((n, d), dtype=float)
    sumA = np.zeros(d, dtype=float)
    sumB = np.zeros(d, dtype=float)
    cA = 0
    cB = 0
    for t in range(n):
        if spk_is_A[t]:
            sumA += E[t]
            cA += 1
        else:
            sumB += E[t]
            cB += 1
        muA[t] = sumA / max(cA, 1)
        muB[t] = sumB / max(cB, 1)
    return muA, muB

def directional_influence(
    E: np.ndarray,
    muA: np.ndarray,
    muB: np.ndarray,
    spk_is_A: np.ndarray,
    target_is_A: bool
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Matches your directional_influence():
      delta = mu_target(curr_t) - mu_target(prev_t)
      toward = mu_other(prev_t) - mu_target(prev_t)
      influence_step = cosine(delta, toward)
    Returns mean influence and per-step details (prev_t, curr_t, cosine).
    """
    t_idx = np.where(spk_is_A if target_is_A else ~spk_is_A)[0]
    if len(t_idx) < 2:
        return np.nan, []

    vals: List[Tuple[int, int, float]] = []
    prev_t = int(t_idx[0])
    for curr_t in t_idx[1:]:
        curr_t = int(curr_t)

        mu_target_prev = muA[prev_t] if target_is_A else muB[prev_t]
        mu_target_curr = muA[curr_t] if target_is_A else muB[curr_t]
        delta = mu_target_curr - mu_target_prev

        mu_other_prev = muB[prev_t] if target_is_A else muA[prev_t]
        toward = mu_other_prev - mu_target_prev

        nd = float(np.linalg.norm(delta))
        nt = float(np.linalg.norm(toward))
        if nd == 0 or nt == 0:
            prev_t = curr_t
            continue

        cs = float(np.dot(delta, toward) / (nd * nt))
        vals.append((prev_t, curr_t, cs))
        prev_t = curr_t

    mean_val = float(np.mean([v[2] for v in vals])) if vals else np.nan
    return mean_val, vals

def alignment_lag(
    E: np.ndarray,
    source_idx: np.ndarray,
    target_idx: np.ndarray,
    eps: float
) -> Tuple[float, List[Tuple[int, int, int]]]:
    """Matches your alignment_lag(): earliest future target j>i within eps, lag=j-i."""
    if len(source_idx) == 0 or len(target_idx) == 0 or np.isnan(eps):
        return np.nan, []
    matches: List[Tuple[int, int, int]] = []
    lags: List[int] = []
    for i in source_idx:
        future = target_idx[target_idx > i]
        if len(future) == 0:
            continue
        d = np.linalg.norm(E[future] - E[i], axis=1)
        ok = np.where(d < eps)[0]
        if len(ok) == 0:
            continue
        j = int(future[int(ok[0])])
        lag = int(j - i)
        lags.append(lag)
        matches.append((int(i), j, lag))
    return (float(np.mean(lags)) if lags else np.nan), matches

def compute_align_eps_like_script(E: np.ndarray, idxA: np.ndarray, idxB: np.ndarray) -> float:
    """Matches your epsilon: 25th percentile of sampled cross distances."""
    if len(idxA) == 0 or len(idxB) == 0:
        return np.nan
    rng = np.random.default_rng(0)
    sampleA = rng.choice(idxA, size=min(len(idxA), 60), replace=False)
    sampleB = rng.choice(idxB, size=min(len(idxB), 60), replace=False)
    pairs = []
    for i in sampleA:
        for j in sampleB:
            pairs.append(float(np.linalg.norm(E[i] - E[j])))
    return float(np.percentile(np.array(pairs, dtype=float), 25))


# -----------------------------
# Data prep
# -----------------------------

def pick_top2_speakers(df: pd.DataFrame) -> Tuple[str, str]:
    counts = df["Speaker"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need at least two speakers.")
    a, b = counts.index[:2].tolist()
    return str(a), str(b)

def load_and_prepare(csv_path: Path, speaker_a: Optional[str], speaker_b: Optional[str]) -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(csv_path)
    need = {"Sequence", "Speaker", "Utterance"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df = df.copy()
    df["Utterance"] = df["Utterance"].fillna("").astype(str)
    df = df.sort_values("Sequence").reset_index(drop=True)

    if speaker_a and speaker_b:
        df = df[df["Speaker"].isin([speaker_a, speaker_b])].copy()
        if df["Speaker"].nunique() < 2:
            raise ValueError("Could not find both specified speakers in the CSV.")
        A_name, B_name = speaker_a, speaker_b
    else:
        A_name, B_name = pick_top2_speakers(df)
        df = df[df["Speaker"].isin([A_name, B_name])].copy()

    df = df.sort_values("Sequence").reset_index(drop=True)
    return df, A_name, B_name

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    E = model.encode(texts, show_progress_bar=True)
    return np.asarray(E, dtype=float)

def pca_2d(E: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=0).fit_transform(E)

def make_labels(df: pd.DataFrame, spk_is_A: np.ndarray) -> List[str]:
    """Label every point as A_t or B_t where t is time index in sorted df."""
    labels = []
    for t in range(len(df)):
        labels.append(("A" if spk_is_A[t] else "B") + f"_{t}")
    return labels


# -----------------------------
# Plot primitives
# -----------------------------

def _scatter_with_labels(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray, title: str = ""):
    idxA = np.where(spk_is_A)[0]
    idxB = np.where(~spk_is_A)[0]

    ax.scatter(E2[idxA, 0], E2[idxA, 1], s=40, alpha=0.85, label="A")
    ax.scatter(E2[idxB, 0], E2[idxB, 1], s=40, alpha=0.85, label="B")

    for i, lab in enumerate(labels):
        ax.text(E2[i, 0], E2[i, 1], lab, fontsize=8, alpha=0.9)

    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([])
    ax.set_yticks([])

def _set_equalish(ax):
    ax.set_aspect("equal", adjustable="datalim")

def _draw_arrow(ax, p: np.ndarray, q: np.ndarray, *, lw: float = 2.0, alpha: float = 0.8, label: Optional[str] = None):
    ax.annotate(
        "",
        xy=(q[0], q[1]),
        xytext=(p[0], p[1]),
        arrowprops=dict(arrowstyle="->", lw=lw, alpha=alpha),
    )
    if label:
        mid = (p + q) / 2.0
        ax.text(mid[0], mid[1], label, fontsize=9, alpha=0.9)


# -----------------------------
# Metric panels (each matches your definitions)
# -----------------------------

def panel_centroid(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray, muA2: np.ndarray, muB2: np.ndarray, t: int):
    _scatter_with_labels(ax, E2[: t + 1], labels[: t + 1], spk_is_A[: t + 1], title=f"Centroids @ t={t} (cumulative)")
    a = muA2[t]
    b = muB2[t]
    ax.scatter([a[0]], [a[1]], s=160, marker="X", alpha=0.95)
    ax.scatter([b[0]], [b[1]], s=160, marker="X", alpha=0.95)
    ax.text(a[0], a[1], "  μA(t)", fontsize=10, weight="bold")
    ax.text(b[0], b[1], "  μB(t)", fontsize=10, weight="bold")
    ax.plot([a[0], b[0]], [a[1], b[1]], lw=2.5, alpha=0.85)
    ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, f"d_cent(t)={euclid(a, b):.3f} (in 2D)", fontsize=9)
    _set_equalish(ax)

def panel_cross_nn(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray, E: np.ndarray):
    _scatter_with_labels(ax, E2, labels, spk_is_A, title="Cross-speaker nearest neighbors")
    idxA = np.where(spk_is_A)[0]
    idxB = np.where(~spk_is_A)[0]
    EA = E[idxA]
    EB = E[idxB]

    # NN matches computed in D dims (correct), drawn in 2D (illustrative)
    if len(idxA) and len(idxB):
        if SCIPY_OK:
            treeB = cKDTree(EB)
            _, jA = treeB.query(EA, k=1)
            jA = jA.astype(int)
            treeA = cKDTree(EA)
            _, iB = treeA.query(EB, k=1)
            iB = iB.astype(int)
        else:
            jA, iB = [], []
            for a in EA:
                jA.append(int(np.argmin(np.linalg.norm(EB - a, axis=1))))
            for b in EB:
                iB.append(int(np.argmin(np.linalg.norm(EA - b, axis=1))))
            jA, iB = np.array(jA), np.array(iB)

        # draw a few links (avoid clutter)
        for local_i, local_j in enumerate(jA[: min(10, len(jA))]):
            i = int(idxA[local_i]); j = int(idxB[local_j])
            ax.plot([E2[i, 0], E2[j, 0]], [E2[i, 1], E2[j, 1]], lw=1.2, alpha=0.25)

        for local_j, local_i in enumerate(iB[: min(10, len(iB))]):
            j = int(idxB[local_j]); i = int(idxA[local_i])
            ax.plot([E2[j, 0], E2[i, 0]], [E2[j, 1], E2[i, 1]], lw=1.2, alpha=0.25)

        nnA = mean_cross_nn_distance(EA, EB)
        nnB = mean_cross_nn_distance(EB, EA)
        haus = hausdorff_soft(EA, EB)
        ax.text(0.01, 0.01, f"NN(A→B)={nnA:.3f}\nNN(B→A)={nnB:.3f}\nHausdorff_soft=max(...)={haus:.3f}",
                transform=ax.transAxes, fontsize=9, va="bottom")
    _set_equalish(ax)

def panel_influence(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray,
                    muA2: np.ndarray, muB2: np.ndarray, influence_steps: List[Tuple[int, int, float]], title: str):
    _scatter_with_labels(ax, E2, labels, spk_is_A, title=title)
    if not influence_steps:
        ax.text(0.01, 0.01, "Not enough turns for influence.", transform=ax.transAxes, fontsize=9, va="bottom")
        _set_equalish(ax); return

    prev_t, curr_t, cs = influence_steps[0]
    target_is_A = bool(spk_is_A[prev_t])

    mu_target_prev = muA2[prev_t] if target_is_A else muB2[prev_t]
    mu_target_curr = muA2[curr_t] if target_is_A else muB2[curr_t]
    mu_other_prev  = muB2[prev_t] if target_is_A else muA2[prev_t]

    delta  = mu_target_curr - mu_target_prev
    toward = mu_other_prev - mu_target_prev

    ax.scatter([mu_target_prev[0]], [mu_target_prev[1]], s=160, marker="X", alpha=0.95)
    ax.scatter([mu_other_prev[0]],  [mu_other_prev[1]],  s=160, marker="X", alpha=0.95)
    ax.text(mu_target_prev[0], mu_target_prev[1], "  μ_target(prev)", fontsize=10, weight="bold")
    ax.text(mu_other_prev[0],  mu_other_prev[1],  "  μ_other(prev)",  fontsize=10, weight="bold")

    _draw_arrow(ax, mu_target_prev, mu_other_prev, lw=2.0, alpha=0.7, label="v_toward")
    _draw_arrow(ax, mu_target_prev, mu_target_prev + delta, lw=2.0, alpha=0.9, label="Δμ_target")

    ax.text(0.01, 0.01, f"Influence step example: cos(Δμ, v_toward) = {cs:.3f}\n(mean over steps computed in code)",
            transform=ax.transAxes, fontsize=9, va="bottom")
    _set_equalish(ax)

def panel_turn_jumps(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray, E: np.ndarray):
    _scatter_with_labels(ax, E2, labels, spk_is_A, title="Turn jump distances (speaker-change transitions)")
    n = len(labels)
    jumps = []
    for t in range(1, n):
        if spk_is_A[t] != spk_is_A[t - 1]:
            d = euclid(E[t], E[t - 1])   # computed in D dims (correct)
            jumps.append(d)
            ax.plot([E2[t - 1, 0], E2[t, 0]], [E2[t - 1, 1], E2[t, 1]], lw=2.0, alpha=0.35, linestyle="--")
            mid = (E2[t - 1] + E2[t]) / 2
            ax.text(mid[0], mid[1], f"{d:.2f}", fontsize=8, alpha=0.9)
    if jumps:
        ax.text(0.01, 0.01, f"TurnJump_mean={float(np.mean(jumps)):.3f}\nTurnJump_std={float(np.std(jumps)):.3f}",
                transform=ax.transAxes, fontsize=9, va="bottom")
    _set_equalish(ax)

def panel_entropy_mi(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray, E: np.ndarray, k: int):
    title = f"Entropy & MI via k-means (k={k})"
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.set_xticks([]); ax.set_yticks([])

    n = len(labels)
    if n < 4:
        _scatter_with_labels(ax, E2, labels, spk_is_A, title=title)
        ax.text(0.01, 0.01, "Too few points for clustering metrics.", transform=ax.transAxes, fontsize=9, va="bottom")
        _set_equalish(ax); return

    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    cl = km.fit_predict(E)

    for i in range(n):
        ax.scatter([E2[i, 0]], [E2[i, 1]], s=40, alpha=0.85)
        ax.text(E2[i, 0], E2[i, 1], f"{labels[i]}(c{cl[i]})", fontsize=8, alpha=0.9)

    counts_total = np.bincount(cl, minlength=k)
    H_total = entropy_from_counts(counts_total)
    idxA = np.where(spk_is_A)[0]
    idxB = np.where(~spk_is_A)[0]
    H_A = entropy_from_counts(np.bincount(cl[idxA], minlength=k)) if len(idxA) else np.nan
    H_B = entropy_from_counts(np.bincount(cl[idxB], minlength=k)) if len(idxB) else np.nan

    speaker_binary = spk_is_A.astype(int)
    mi  = float(mutual_info_score(speaker_binary, cl))
    nmi = float(normalized_mutual_info_score(speaker_binary, cl))

    ax.text(0.01, 0.01, f"H_total={H_total:.3f}\nH_A={H_A:.3f}  H_B={H_B:.3f}\nMI(speaker,cluster)={mi:.3f}\nNMI={nmi:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom")
    _set_equalish(ax)

def panel_hull(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray):
    _scatter_with_labels(ax, E2, labels, spk_is_A, title="2D PCA hulls (illustrative; metrics use PCA hull overlap)")
    if not SCIPY_OK:
        ax.text(0.01, 0.01, "SciPy not installed: hull visual disabled (metrics script also skips hull stats).",
                transform=ax.transAxes, fontsize=9, va="bottom")
        _set_equalish(ax); return

    idxA = np.where(spk_is_A)[0]
    idxB = np.where(~spk_is_A)[0]
    A2 = E2[idxA]; B2 = E2[idxB]

    if len(A2) >= 3:
        hullA = ConvexHull(A2)
        polyA = A2[hullA.vertices]
        ax.fill(polyA[:, 0], polyA[:, 1], alpha=0.08)
        ax.plot(np.r_[polyA[:, 0], polyA[0, 0]], np.r_[polyA[:, 1], polyA[0, 1]], lw=2, alpha=0.5)
    if len(B2) >= 3:
        hullB = ConvexHull(B2)
        polyB = B2[hullB.vertices]
        ax.fill(polyB[:, 0], polyB[:, 1], alpha=0.08)
        ax.plot(np.r_[polyB[:, 0], polyB[0, 0]], np.r_[polyB[:, 1], polyB[0, 1]], lw=2, alpha=0.5)

    ax.text(0.01, 0.01, "Hull overlap/containment in your metrics code is Monte Carlo over a bounding box.",
            transform=ax.transAxes, fontsize=9, va="bottom")
    _set_equalish(ax)

def panel_alignment(ax, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray, E: np.ndarray):
    _scatter_with_labels(ax, E2, labels, spk_is_A, title="Alignment lag (A→B and B→A)")

    idxA = np.where(spk_is_A)[0]
    idxB = np.where(~spk_is_A)[0]
    eps = compute_align_eps_like_script(E, idxA, idxB)

    lagA, matchesA = alignment_lag(E, idxA, idxB, eps)
    lagB, matchesB = alignment_lag(E, idxB, idxA, eps)

    # draw match arrows in 2D (matches computed in D dims)
    for (i, j, lag) in matchesA[: min(8, len(matchesA))]:
        _draw_arrow(ax, E2[i], E2[j], lw=1.8, alpha=0.55, label=f"{lag}")
    for (i, j, lag) in matchesB[: min(8, len(matchesB))]:
        _draw_arrow(ax, E2[i], E2[j], lw=1.8, alpha=0.35, label=f"{lag}")

    ax.text(0.01, 0.01,
            f"epsilon (25th pct of sampled cross distances, in D dims) = {eps:.3f}\n"
            f"AlignLag A→B = {lagA:.3f}    AlignLag B→A = {lagB:.3f}\n"
            f"Asym (A−B) = {(lagA - lagB) if (not np.isnan(lagA) and not np.isnan(lagB)) else np.nan:.3f}",
            transform=ax.transAxes, fontsize=9, va="bottom")
    _set_equalish(ax)


# -----------------------------
# Rendering (static + animation)
# -----------------------------

def render_static(out_png: Path, df: pd.DataFrame, E: np.ndarray, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray, k_clusters: int):
    # centroids in D, then projected using the same PCA fit (so it's consistent with E2)
    muA, muB = compute_cumulative_centroids(E, spk_is_A)
    pca = PCA(n_components=2, random_state=0).fit(E)
    muA2 = pca.transform(muA)
    muB2 = pca.transform(muB)

    infA_mean, infA_steps = directional_influence(E, muA, muB, spk_is_A, target_is_A=True)
    infB_mean, infB_steps = directional_influence(E, muA, muB, spk_is_A, target_is_A=False)

    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    axes = axes.ravel()

    panel_centroid(axes[0], E2, labels, spk_is_A, muA2, muB2, t=len(labels) - 1)
    axes[0].text(0.01, 0.99,
                 "Centroid metric uses d_cent(t)=||μA(t)-μB(t)|| in embedding space,\nthen summarizes mean/std/slope over t.",
                 transform=axes[0].transAxes, fontsize=9, va="top")

    panel_cross_nn(axes[1], E2, labels, spk_is_A, E)
    panel_influence(axes[2], E2, labels, spk_is_A, muA2, muB2, infA_steps, title=f"Influence: A toward B (mean={infA_mean:.3f})")
    panel_influence(axes[3], E2, labels, spk_is_A, muA2, muB2, infB_steps, title=f"Influence: B toward A (mean={infB_mean:.3f})")
    panel_turn_jumps(axes[4], E2, labels, spk_is_A, E)
    panel_entropy_mi(axes[5], E2, labels, spk_is_A, E, k_clusters)
    panel_hull(axes[6], E2, labels, spk_is_A)
    panel_alignment(axes[7], E2, labels, spk_is_A, E)

    handles, leg_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center", ncol=2)

    fig.suptitle("Metric computation visualizations (A/B are the top-2 speakers in file)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

def render_animation_centroid(out_gif: Path, E: np.ndarray, E2: np.ndarray, labels: List[str], spk_is_A: np.ndarray):
    """Animate cumulative centroid distance over time."""
    import matplotlib.animation as animation

    muA, muB = compute_cumulative_centroids(E, spk_is_A)
    pca = PCA(n_components=2, random_state=0).fit(E)
    muA2 = pca.transform(muA)
    muB2 = pca.transform(muB)

    fig, ax = plt.subplots(figsize=(10, 7))

    def draw_frame(t: int):
        ax.clear()
        panel_centroid(ax, E2, labels, spk_is_A, muA2, muB2, t=t)

    ani = animation.FuncAnimation(fig, draw_frame, frames=len(labels), interval=600)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    ani.save(out_gif, writer="pillow")
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Input CSV")
    ap.add_argument("--outdir", required=True, type=str, help="Output directory")
    ap.add_argument("--speaker-a", type=str, default=None, help="Optional: force speaker A name")
    ap.add_argument("--speaker-b", type=str, default=None, help="Optional: force speaker B name")
    ap.add_argument("--model", type=str, default="all-MiniLM-L6-v2", help="SentenceTransformer model")
    ap.add_argument("--k", type=int, default=None, help="k for k-means (entropy/MI). Default matches your heuristic.")
    ap.add_argument("--animate", type=str, default=None, choices=[None, "centroid"], help="Optional: output a metric animation")
    args = ap.parse_args()

    in_path = Path(args.input)
    outdir = Path(args.outdir)

    df, A_name, B_name = load_and_prepare(in_path, args.speaker_a, args.speaker_b)

    texts = df["Utterance"].tolist()
    E = embed_texts(texts, args.model)
    E2 = pca_2d(E)

    spk = df["Speaker"].to_numpy()
    spk_is_A = (spk == A_name)
    labels = make_labels(df, spk_is_A)

    # k heuristic: identical to your metrics script
    n = len(df)
    if args.k is None:
        k = int(round(math.sqrt(max(n, 1))))
        k = max(4, min(12, k))
        k = min(k, max(2, n // 3)) if n >= 6 else 2
    else:
        k = int(args.k)

    if args.animate == "centroid":
        out_gif = outdir / f"{in_path.stem}__centroid.gif"
        render_animation_centroid(out_gif, E, E2, labels, spk_is_A)
        print(f"Saved: {out_gif}")
        return

    out_png = outdir / f"{in_path.stem}__metric_panels.png"
    render_static(out_png, df, E, E2, labels, spk_is_A, k)
    print(f"Saved: {out_png}")

if __name__ == "__main__":
    main()
