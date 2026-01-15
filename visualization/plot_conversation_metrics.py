#!/usr/bin/env python3
"""
plot_conversation_metrics.py

One-stop script to generate basic bar plots, line plots, and distribution plots
from your per-conversation metrics CSV (one row per conversation).

Usage:
  python plot_conversation_metrics.py --csv /path/to/all_conversations_metrics_speaker1A.csv --out plots

Notes:
- Expects columns like: speaker_A, speaker_B, centroid_dist_mean, etc.
- Tries to infer expertise_B from speaker_B labels like "... (Child)" or keywords.
"""

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn is optional but nice for violin/box aesthetics
try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False


EXPERTISE_ORDER = ["Child", "Teen", "College Student", "Graduate Student", "Expert"]


# ---------- helpers ----------

def infer_expertise_from_speaker_label(s: str) -> str:
    """
    Infer expertise label from a speaker string.
    Examples:
      "Kayla Martini (Child)" -> "Child"
      "Maria Guseva (Teen)" -> "Teen"
      "Steve - Expert/Physicist" -> "Expert"
    """
    if not isinstance(s, str) or not s.strip():
        return "Expert"

    # 1) try "(...)" parenthetical
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        inside = m.group(1).strip()
        # normalize common variants
        inside_low = inside.lower()
        if "child" in inside_low:
            return "Child"
        if "teen" in inside_low:
            return "Teen"
        if "college" in inside_low or "undergrad" in inside_low:
            return "College Student"
        if "graduate" in inside_low or "grad" in inside_low:
            return "Graduate Student"
        if "expert" in inside_low:
            return "Expert"

    # 2) keyword fallback in full string
    low = s.lower()
    if "child" in low:
        return "Child"
    if "teen" in low:
        return "Teen"
    if "college" in low or "undergrad" in low:
        return "College Student"
    if "graduate student" in low or "grad student" in low or "grad" in low:
        return "Graduate Student"
    if "expert" in low or "phd" in low or "prof" in low:
        return "Expert"

    return "Expert"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def barplot_groupmean(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str = None, hline: float = None, ylim=None):
    # group mean + std as error bars
    g = df.groupby(x)[y].agg(["mean", "std", "count"]).reset_index()
    # keep order
    if pd.api.types.is_categorical_dtype(df[x]):
        g[x] = pd.Categorical(g[x], categories=df[x].cat.categories, ordered=True)
        g = g.sort_values(x)

    xs = np.arange(len(g))
    plt.figure(figsize=(8, 4.2))
    plt.bar(xs, g["mean"].values, yerr=g["std"].values, capsize=4)
    plt.xticks(xs, g[x].astype(str).values, rotation=20, ha="right")
    plt.ylabel(y)
    plt.xlabel(x)
    plt.title(title or f"{y} by {x} (mean ± SD)")
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1)
    if ylim is not None:
        plt.ylim(*ylim)

    # annotate n
    for i, (_, row) in enumerate(g.iterrows()):
        plt.text(i, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

    safe_savefig(outpath)


def lineplot_groupmean(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str = None, hline: float = None):
    g = df.groupby(x)[y].mean().reset_index()
    if pd.api.types.is_categorical_dtype(df[x]):
        g[x] = pd.Categorical(g[x], categories=df[x].cat.categories, ordered=True)
        g = g.sort_values(x)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(g[x].astype(str).values, g[y].values, marker="o")
    plt.ylabel(y)
    plt.xlabel(x)
    plt.title(title or f"{y} vs {x} (group mean)")
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1)
    safe_savefig(outpath)


def violin_or_box(df: pd.DataFrame, x: str, y: str, outpath: Path, title: str = None, ylim=None):
    plt.figure(figsize=(8, 4.2))
    if _HAVE_SNS:
        sns.violinplot(data=df, x=x, y=y, inner="box")
    else:
        # fallback: boxplot
        groups = []
        labels = []
        # preserve order if categorical
        if pd.api.types.is_categorical_dtype(df[x]):
            cats = list(df[x].cat.categories)
        else:
            cats = sorted(df[x].dropna().unique().tolist())
        for c in cats:
            vals = df.loc[df[x] == c, y].dropna().values
            if len(vals) == 0:
                continue
            groups.append(vals)
            labels.append(str(c))
        plt.boxplot(groups, labels=labels)

    plt.ylabel(y)
    plt.xlabel(x)
    plt.title(title or f"{y} distribution by {x}")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xticks(rotation=20, ha="right")
    safe_savefig(outpath)


def overlay_bars(df: pd.DataFrame, x: str, y1: str, y2: str, outpath: Path, label1: str = None, label2: str = None):
    g1 = df.groupby(x)[y1].agg(["mean", "std"]).reset_index()
    g2 = df.groupby(x)[y2].agg(["mean", "std"]).reset_index()
    if pd.api.types.is_categorical_dtype(df[x]):
        cats = list(df[x].cat.categories)
    else:
        cats = sorted(df[x].dropna().unique().tolist())

    # align to same order
    g1 = g1.set_index(x).reindex(cats).reset_index()
    g2 = g2.set_index(x).reindex(cats).reset_index()

    xs = np.arange(len(cats))
    width = 0.38
    plt.figure(figsize=(8, 4.2))
    plt.bar(xs - width/2, g1["mean"].values, width=width, yerr=g1["std"].values, capsize=4, label=label1 or y1)
    plt.bar(xs + width/2, g2["mean"].values, width=width, yerr=g2["std"].values, capsize=4, alpha=0.7, label=label2 or y2)
    plt.xticks(xs, [str(c) for c in cats], rotation=20, ha="right")
    plt.ylabel("value")
    plt.xlabel(x)
    plt.title(f"{y1} vs {y2} by {x} (mean ± SD)")
    plt.legend()
    safe_savefig(outpath)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to metrics CSV (one row per conversation).")
    ap.add_argument("--out", default="plots", help="Output folder for plots.")
    ap.add_argument("--topic_col", default=None, help="Optional: name of a topic column if you have one.")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    df = pd.read_csv(csv_path)

    # infer expertise_B if not present
    if "expertise_B" not in df.columns:
        if "speaker_B" not in df.columns:
            raise ValueError("CSV must contain 'speaker_B' or an existing 'expertise_B' column.")
        df["expertise_B"] = df["speaker_B"].apply(infer_expertise_from_speaker_label)

    df["expertise_B"] = pd.Categorical(df["expertise_B"], categories=EXPERTISE_ORDER, ordered=True)

    # ---- choose core metrics (edit freely) ----
    anchor_metrics = [
        ("centroid_dist_mean", dict(title="Centroid distance (mean)", hline=None, ylim=None)),
        ("centroid_dist_slope", dict(title="Centroid distance slope (negative = convergence)", hline=0.0, ylim=None)),
        ("influence_asymmetry_AminusB", dict(title="Influence asymmetry (A→B minus B→A)", hline=0.0, ylim=None)),
        ("containment_B_in_A", dict(title="Containment of B inside A", hline=None, ylim=(0, 1))),
        ("hull_overlap_ratio_min", dict(title="Hull overlap ratio (min-normalized)", hline=None, ylim=(0, 1))),
        ("entropy_B", dict(title="Entropy (Speaker B)", hline=None, ylim=None)),
        ("entropy_total", dict(title="Entropy (Total)", hline=None, ylim=None)),
        ("align_lag_asymmetry_AminusB", dict(title="Alignment lag asymmetry (A→B minus B→A)", hline=0.0, ylim=None)),
    ]

    # filter only those present
    present = []
    for m, cfg in anchor_metrics:
        if m in df.columns:
            present.append((m, cfg))

    if not present:
        raise ValueError("None of the expected anchor metrics were found in the CSV. Check column names.")

    # ---- quick summary table ----
    summary_cols = ["expertise_B"] + [m for m, _ in present]
    summary = df[summary_cols].groupby("expertise_B").agg(["mean", "std", "count"])
    summary.to_csv(out_dir / "group_summary_by_expertise.csv")

    print(f"[OK] Loaded: {csv_path}")
    print(f"[OK] Rows: {len(df)}")
    print(f"[OK] Wrote: {out_dir / 'group_summary_by_expertise.csv'}")
    print(f"[OK] Seaborn available: {_HAVE_SNS}")

    # ---- generate plots ----
    for metric, cfg in present:
        # bar plot
        barplot_groupmean(
            df=df,
            x="expertise_B",
            y=metric,
            outpath=out_dir / f"bar_{metric}.png",
            title=cfg.get("title"),
            hline=cfg.get("hline"),
            ylim=cfg.get("ylim"),
        )

        # line plot (means)
        lineplot_groupmean(
            df=df,
            x="expertise_B",
            y=metric,
            outpath=out_dir / f"line_{metric}.png",
            title=cfg.get("title"),
            hline=cfg.get("hline"),
        )

        # distribution plot
        violin_or_box(
            df=df,
            x="expertise_B",
            y=metric,
            outpath=out_dir / f"dist_{metric}.png",
            title=cfg.get("title"),
            ylim=cfg.get("ylim"),
        )

    # ---- special overlay: turn jump A->B vs B->A if present ----
    if "turn_jump_A_to_B_mean" in df.columns and "turn_jump_B_to_A_mean" in df.columns:
        overlay_bars(
            df=df,
            x="expertise_B",
            y1="turn_jump_A_to_B_mean",
            y2="turn_jump_B_to_A_mean",
            outpath=out_dir / "bar_turn_jump_AtoB_vs_BtoA.png",
            label1="A→B",
            label2="B→A",
        )

    # ---- special overlay: cross NN A->B vs B->A if present ----
    if "cross_nn_A_to_B" in df.columns and "cross_nn_B_to_A" in df.columns:
        overlay_bars(
            df=df,
            x="expertise_B",
            y1="cross_nn_A_to_B",
            y2="cross_nn_B_to_A",
            outpath=out_dir / "bar_cross_nn_AtoB_vs_BtoA.png",
            label1="A→B",
            label2="B→A",
        )

    print(f"[DONE] Plots written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
