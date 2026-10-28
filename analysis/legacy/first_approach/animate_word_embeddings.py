#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
animate_word_embeddings_pairs.py

Reads a CSV with columns: Sequence, Speaker, Utterance
Builds an interactive Plotly animation (with slider + play/pause) that shows
cumulative word-embedding points over time for a chosen speaker pair.

- Speaker A points: red
- Speaker B points: blue
- Points accumulate as Sequence increases
- Outputs an HTML you can open in a browser

Dependencies:
  pip install pandas numpy scikit-learn plotly sentence-transformers

Usage (CLI examples):
  python animate_word_embeddings_pairs.py --csv wired_Talia_12.csv --speaker-a "Speaker 1" --speaker-b "Speaker 5" --out out_1_5.html
  python animate_word_embeddings_pairs.py --csv wired_Talia_12.csv --speaker-a "Speaker 1" --speaker-b "Speaker 6" --out out_1_6.html
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity  # not strictly needed, but handy later

import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer


# -----------------------------
# Tokenization / filtering
# -----------------------------
WORD_RE = re.compile(r"[A-Za-z']+")

DEFAULT_STOPWORDS = {
    "the","a","an","and","or","but","if","then","so","to","of","in","on","for","with","at","by",
    "is","are","was","were","be","been","being","it","this","that","these","those","i","you","we","they",
    "my","your","our","their","me","him","her","them","as","from","not","do","does","did","can","could",
    "will","would","should","about","into","over","under","up","down","out","just","kind","like",
    "yeah","um","uh","okay","ok","really"
}

def tokenize(
    text: str,
    min_word_len: int = 2,
    remove_stopwords: bool = True,
    stopwords: Optional[Set[str]] = None,
) -> List[str]:
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS
    toks = [t.lower() for t in WORD_RE.findall(text or "")]
    toks = [t for t in toks if len(t) >= min_word_len]
    if remove_stopwords:
        toks = [t for t in toks if t not in stopwords]
    return toks


# -----------------------------
# Core reusable function
# -----------------------------
def animate_speaker_pair(
    df: pd.DataFrame,
    speaker_a: str,
    speaker_b: str,
    output_html: str | Path,
    *,
    model_name: str = "all-MiniLM-L6-v2",
    model: Optional[SentenceTransformer] = None,
    max_unique_words_per_speaker: Optional[int] = 300,
    min_word_len: int = 2,
    remove_stopwords: bool = True,
    stopwords: Optional[Set[str]] = None,
    show_text_labels: bool = False,   # if False: hover labels only (cleaner)
    frame_duration_ms: int = 200,
    point_size: int = 7,
) -> Path:
    """
    Build and save an interactive HTML animation for (speaker_a vs speaker_b).

    Returns: Path to saved HTML.
    """

    # Basic validation
    required = {"Sequence", "Speaker", "Utterance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV/DataFrame missing columns: {missing}")

    df_pair = df[df["Speaker"].isin([speaker_a, speaker_b])].copy()
    df_pair = df_pair.sort_values("Sequence").reset_index(drop=True)
    df_pair["Utterance"] = df_pair["Utterance"].fillna("").astype(str)

    if len(df_pair) == 0:
        raise ValueError(f"No rows found for speakers: {speaker_a} and {speaker_b}")

    print(f"\nAnimating pair:\n  A: {speaker_a}\n  B: {speaker_b}\n  Rows: {len(df_pair)}")

    # Build cumulative unique word sets over time
    cum_a: List[List[str]] = []
    cum_b: List[List[str]] = []

    seen_a: Set[str] = set()
    seen_b: Set[str] = set()
    all_words: Set[str] = set()

    for _, row in df_pair.iterrows():
        spk = row["Speaker"]
        toks = tokenize(
            row["Utterance"],
            min_word_len=min_word_len,
            remove_stopwords=remove_stopwords,
            stopwords=stopwords,
        )

        if spk == speaker_a:
            for w in toks:
                if max_unique_words_per_speaker is None or len(seen_a) < max_unique_words_per_speaker or w in seen_a:
                    seen_a.add(w)
                    all_words.add(w)
        else:  # speaker_b
            for w in toks:
                if max_unique_words_per_speaker is None or len(seen_b) < max_unique_words_per_speaker or w in seen_b:
                    seen_b.add(w)
                    all_words.add(w)

        cum_a.append(sorted(seen_a))
        cum_b.append(sorted(seen_b))

    print(f"  Unique words A: {len(seen_a)}")
    print(f"  Unique words B: {len(seen_b)}")
    print(f"  Unique total:   {len(all_words)}")

    # Prepare embedding model (reuse if passed in)
    if model is None:
        model = SentenceTransformer(model_name)

    # Embed all unique words once (per pair)
    all_words_sorted = sorted(all_words)
    if len(all_words_sorted) == 0:
        raise ValueError("No tokens found after filtering. Try loosening stopwords/min length.")

    emb = model.encode(all_words_sorted, show_progress_bar=True)
    emb = np.asarray(emb)

    # Project to 2D (PCA is stable & fast)
    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(emb)
    word_to_xy: Dict[str, Tuple[float, float]] = {
        w: (float(x), float(y)) for w, (x, y) in zip(all_words_sorted, xy)
    }

    def words_to_points(words: List[str]):
        xs, ys, labs = [], [], []
        for w in words:
            if w in word_to_xy:
                x, y = word_to_xy[w]
                xs.append(x); ys.append(y); labs.append(w)
        return xs, ys, labs

    # Plot settings: text on hover (clean) vs always visible (can get cluttered fast)
    def make_trace(xs, ys, labs, color, name):
        if show_text_labels:
            return go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                text=labs,
                textposition="top center",
                marker=dict(size=point_size, color=color, opacity=0.7),
                name=name
            )
        else:
            return go.Scatter(
                x=xs, y=ys,
                mode="markers",
                text=labs,
                hoverinfo="text",
                marker=dict(size=point_size, color=color, opacity=0.7),
                name=name
            )

    # Build frames (one frame per row / sequence step)
    frames = []
    for i in range(len(df_pair)):
        words_a = cum_a[i]
        words_b = cum_b[i]
        xA, yA, tA = words_to_points(words_a)
        xB, yB, tB = words_to_points(words_b)

        frames.append(go.Frame(
            name=str(df_pair.iloc[i]["Sequence"]),
            data=[
                make_trace(xA, yA, tA, "red", f"A: {speaker_a}"),
                make_trace(xB, yB, tB, "blue", f"B: {speaker_b}")
            ],
            layout=go.Layout(
                title=f"Word Embeddings Progression — {speaker_a} vs {speaker_b} — Seq {df_pair.iloc[i]['Sequence']}"
            )
        ))

    # Initial plot = first frame
    xA0, yA0, tA0 = words_to_points(cum_a[0])
    xB0, yB0, tB0 = words_to_points(cum_b[0])

    fig = go.Figure(
        data=[
            make_trace(xA0, yA0, tA0, "red", f"A: {speaker_a}"),
            make_trace(xB0, yB0, tB0, "blue", f"B: {speaker_b}")
        ],
        frames=frames
    )

    # Slider steps
    steps = [{
        "method": "animate",
        "args": [[fr.name], {"mode": "immediate", "frame": {"duration": frame_duration_ms, "redraw": True}, "transition": {"duration": 0}}],
        "label": fr.name
    } for fr in frames]

    # Layout controls
    fig.update_layout(
        title=f"Word Embeddings Progression — {speaker_a} vs {speaker_b}",
        xaxis_title="PCA-1",
        yaxis_title="PCA-2",
        showlegend=True,
        updatemenus=[{
            "type": "buttons",
            "direction": "left",
            "x": 0.1,
            "y": -0.15,
            "buttons": [
                {"label": "Play", "method": "animate",
                 "args": [None, {"frame": {"duration": frame_duration_ms, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                   "mode": "immediate", "transition": {"duration": 0}}]}
            ]
        }],
        sliders=[{
            "active": 0,
            "y": -0.2,
            "x": 0.1,
            "len": 0.85,
            "steps": steps
        }],
        margin=dict(l=30, r=30, t=70, b=130),
    )

    output_html = Path(output_html)
    fig.write_html(str(output_html))
    print(f"Saved HTML: {output_html.resolve()}")
    return output_html


# -----------------------------
# Convenience helpers
# -----------------------------
def list_speakers(df: pd.DataFrame) -> List[str]:
    return list(df["Speaker"].dropna().unique())


def load_csv(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.sort_values("Sequence").reset_index(drop=True)
    return df


# -----------------------------
# CLI entry point (optional)
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="wired_Talia_12.csv", help="Path to CSV with Sequence,Speaker,Utterance")
    parser.add_argument("--speaker-a", type=str, required=False, help="Exact Speaker name for A (red)")
    parser.add_argument("--speaker-b", type=str, required=False, help="Exact Speaker name for B (blue)")
    parser.add_argument("--out", type=str, default="word_embeddings_animation.html", help="Output HTML file path")
    parser.add_argument("--show-text", action="store_true", help="Show word labels always (can get cluttered)")
    args = parser.parse_args()

    df = load_csv(args.csv)

    if args.speaker_a is None or args.speaker_b is None:
        spks = list_speakers(df)
        print("Speakers found in CSV:")
        for s in spks:
            print(" -", s)
        raise SystemExit("\nPass --speaker-a and --speaker-b (exactly as shown above).")

    # Reuse one model instance across runs if you call animate_speaker_pair multiple times in code
    shared_model = SentenceTransformer("all-MiniLM-L6-v2")

    animate_speaker_pair(
        df=df,
        speaker_a=args.speaker_a,
        speaker_b=args.speaker_b,
        output_html=args.out,
        model=shared_model,
        show_text_labels=args.show_text,
    )
