#!/usr/bin/env python3
"""
plot_conversation_metrics_guest_group.py

Goal
----
Plot conversation-level metrics grouped by the *guest’s expertise level*.

Your dataset assumptions (implemented here)
-------------------------------------------
- There are 5 rows per WIRED video (one row per level / speaker pair).
- convo_id ends with a level suffix number: _12, _13, _14, _15, _16
- Speaker A is always the host.
- Speaker B is always the non-host (guest), including expert in expert–expert.
- Therefore: guest expertise is determined ONLY by convo_id suffix (not by speaker strings).

What this script does
---------------------
1) Reads the metrics CSV (one row per conversation pair).
2) Parses:
   - level_num = trailing number in convo_id
   - video_id  = convo_id with trailing _NN removed
3) Maps level_num -> expertise_guest:
   12 Child, 13 Teen, 14 College Student, 15 Graduate Student, 16 Expert
4) (Recommended) Aggregates to exactly one row per (video_id, expertise_guest)
   in case duplicates exist.
5) Produces bar/line/distribution plots by expertise_guest, and a numeric summary CSV.

Usage
-----
python plot_conversation_metrics_guest_group.py --csv all_conversations_metrics_speaker1A.csv --out plots_guest

Notes
-----
- n shown on bars = number of (video_id, expertise_guest) rows that have non-NaN values for that metric.
  Alignment metrics can have smaller n if align_lag_* is NaN for some videos/levels (no match within epsilon).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# seaborn optional
try:
    import seaborn as sns
    _HAVE_SNS = True
except Exception:
    _HAVE_SNS = False


EXPERTISE_ORDER = ["Child", "Teen", "College Student", "Graduate Student", "Expert"]

LEVEL_TO_EXPERTISE = {
    12: "Child",
    13: "Teen",
    14: "College Student",
    15: "Graduate Student",
    16: "Expert",
}


# -------------------------
# Plot helpers
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def safe_savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def barplot_groupmean(df: pd.DataFrame, x: str, y: str, outpath: Path,
                      title: str = None, hline: float = None, ylim=None) -> None:
    # Use only non-null values for y when computing mean/std/count
    d = df[[x, y]].dropna(subset=[y]).copy()
    g = d.groupby(x, observed=True)[y].agg(["mean", "std", "count"]).reset_index()

    if pd.api.types.is_categorical_dtype(df[x]):
        g[x] = pd.Categorical(g[x], categories=df[x].cat.categories, ordered=True)
        g = g.sort_values(x)

    xs = np.arange(len(g))
    plt.figure(figsize=(8, 4.2))
    plt.bar(xs, g["mean"].values, yerr=g["std"].values, capsize=4)
    plt.xticks(xs, g[x].astype(str).values, rotation=20, ha="right")
    plt.ylabel(y)
    plt.xlabel("Guest expertise (from convo_id suffix)")
    plt.title(title or f"{y} by guest expertise (mean ± SD)")
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1)
    if ylim is not None:
        plt.ylim(*ylim)

    for i, row in g.iterrows():
        plt.text(i, row["mean"], f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

    safe_savefig(outpath)

def lineplot_groupmean(df: pd.DataFrame, x: str, y: str, outpath: Path,
                       title: str = None, hline: float = None) -> None:
    d = df[[x, y]].dropna(subset=[y]).copy()
    g = d.groupby(x, observed=True)[y].mean().reset_index()

    if pd.api.types.is_categorical_dtype(df[x]):
        g[x] = pd.Categorical(g[x], categories=df[x].cat.categories, ordered=True)
        g = g.sort_values(x)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(g[x].astype(str).values, g[y].values, marker="o")
    plt.ylabel(y)
    plt.xlabel("Guest expertise (from convo_id suffix)")
    plt.title(title or f"{y} vs guest expertise (group mean)")
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1)

    safe_savefig(outpath)

def violin_or_box(df: pd.DataFrame, x: str, y: str, outpath: Path,
                  title: str = None, ylim=None) -> None:
    d = df[[x, y]].dropna(subset=[y]).copy()

    plt.figure(figsize=(8, 4.2))
    if _HAVE_SNS:
        sns.violinplot(data=d, x=x, y=y, inner="box")
    else:
        cats = list(df[x].cat.categories) if pd.api.types.is_categorical_dtype(df[x]) else sorted(d[x].unique())
        groups = []
        labels = []
        for c in cats:
            vals = d.loc[d[x] == c, y].dropna().values
            if len(vals) == 0:
                continue
            groups.append(vals)
            labels.append(str(c))
        plt.boxplot(groups, labels=labels)

    plt.ylabel(y)
    plt.xlabel("Guest expertise (from convo_id suffix)")
    plt.title(title or f"{y} distribution by guest expertise")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xticks(rotation=20, ha="right")

    safe_savefig(outpath)

def overlay_bars(df: pd.DataFrame, x: str, y1: str, y2: str, outpath: Path,
                 label1: str = None, label2: str = None, title: str = None) -> None:
    d1 = df[[x, y1]].dropna(subset=[y1]).copy()
    d2 = df[[x, y2]].dropna(subset=[y2]).copy()

    g1 = d1.groupby(x, observed=True)[y1].agg(["mean", "std", "count"]).reset_index()
    g2 = d2.groupby(x, observed=True)[y2].agg(["mean", "std", "count"]).reset_index()

    cats = list(df[x].cat.categories) if pd.api.types.is_categorical_dtype(df[x]) else sorted(df[x].dropna().unique().tolist())
    g1 = g1.set_index(x).reindex(cats).reset_index()
    g2 = g2.set_index(x).reindex(cats).reset_index()

    xs = np.arange(len(cats))
    width = 0.38

    plt.figure(figsize=(8, 4.2))
    plt.bar(xs - width/2, g1["mean"].values, width=width, yerr=g1["std"].values, capsize=4,
            label=f"{label1 or y1} (n varies)")
    plt.bar(xs + width/2, g2["mean"].values, width=width, yerr=g2["std"].values, capsize=4, alpha=0.7,
            label=f"{label2 or y2} (n varies)")
    plt.xticks(xs, [str(c) for c in cats], rotation=20, ha="right")
    plt.ylabel("value")
    plt.xlabel("Guest expertise (from convo_id suffix)")
    plt.title(title or f"{y1} vs {y2} by guest expertise (mean ± SD)")
    plt.legend()

    safe_savefig(outpath)


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to metrics CSV (one row per speaker pair / level).")
    ap.add_argument("--out", default="plots_guest", help="Output folder for plots.")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="If set, do NOT collapse duplicates per (video_id, expertise_guest).")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    ensure_dir(out_dir)

    df = pd.read_csv(csv_path)

    if "convo_id" not in df.columns:
        raise ValueError("CSV must contain 'convo_id' with trailing _12/_13/_14/_15/_16.")

    # Parse level suffix and base video id
    df["level_num"] = df["convo_id"].astype(str).str.extract(r"_(\d+)$")[0]
    df = df.dropna(subset=["level_num"]).copy()
    df["level_num"] = df["level_num"].astype(int)

    df["video_id"] = df["convo_id"].astype(str).str.replace(r"_\d+$", "", regex=True)

    # Map level -> guest expertise
    df["expertise_guest"] = df["level_num"].map(LEVEL_TO_EXPERTISE)
    df = df.dropna(subset=["expertise_guest"]).copy()
    df["expertise_guest"] = pd.Categorical(df["expertise_guest"], categories=EXPERTISE_ORDER, ordered=True)

    # Optional: collapse to exactly 1 row per (video_id, expertise_guest)
    # This is what you want if you truly have 13 videos x 5 levels and you want n≈13 per category.
    if not args.no_aggregate:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        df = (
            df.groupby(["video_id", "expertise_guest"], observed=True)[numeric_cols]
              .mean()
              .reset_index()
        )

    print(f"[OK] Loaded: {csv_path}")
    print("[OK] Unique videos:", df["video_id"].nunique())
    print("[OK] Rows per expertise (unique videos):\n",
          df.groupby("expertise_guest", observed=True)["video_id"].nunique())
    print("[OK] Total rows:", len(df))

    # Summary table: numeric only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summary = df.groupby("expertise_guest", observed=True)[numeric_cols].agg(["mean", "std", "count"])
    summary.to_csv(out_dir / "group_summary_by_guest_expertise.csv")
    print(f"[OK] Wrote: {out_dir / 'group_summary_by_guest_expertise.csv'}")

    # Choose metrics to plot (edit as desired)
    # Choose metrics to plot (edit as desired)
    anchor_metrics = [
        ("centroid_dist_mean", dict(title="Centroid distance (mean)", hline=None, ylim=None)),
        ("centroid_dist_slope", dict(title="Centroid distance slope (negative = convergence)", hline=0.0, ylim=None)),
        ("cross_nn_A_to_B", dict(title="Cross NN (A→B)", hline=None, ylim=None)),
        ("cross_nn_B_to_A", dict(title="Cross NN (B→A)", hline=None, ylim=None)),
        ("hausdorff_AB", dict(title="Hausdorff_soft (max of directional mean NN)", hline=None, ylim=None)),
        ("influence_A_toward_B", dict(title="Influence A→B (cosine)", hline=0.0, ylim=(-1, 1))),
        ("influence_B_toward_A", dict(title="Influence B→A (cosine)", hline=0.0, ylim=(-1, 1))),
        ("influence_asymmetry_AminusB", dict(title="Influence asymmetry (A→B − B→A)", hline=0.0, ylim=(-2, 2))),
        ("turn_jump_mean", dict(title="Turn jump mean", hline=None, ylim=None)),
        ("containment_B_in_A", dict(title="Containment of B in A", hline=None, ylim=(0, 1))),
        ("containment_A_in_B", dict(title="Containment of A in B", hline=None, ylim=(0, 1))),
        ("hull_overlap_ratio_min", dict(title="Hull overlap ratio (min-normalized)", hline=None, ylim=(0, 1))),

        # Entropy
        ("entropy_total", dict(title="Entropy (Total)", hline=None, ylim=None)),
        ("entropy_A", dict(title="Entropy (A)", hline=None, ylim=None)),
        ("entropy_B", dict(title="Entropy (B)", hline=None, ylim=None)),

        # Entropy slopes
        ("entropy_slope_A", dict(title="Entropy slope (A)", hline=0.0, ylim=None)),
        ("entropy_slope_B", dict(title="Entropy slope (B)", hline=0.0, ylim=None)),

        # NMI "clusters" (two different ones you compute)
        ("nmi_speaker_cluster", dict(title="NMI(speaker, cluster)", hline=None, ylim=(0, 1))),
        ("nmi_turn_clusters",
         dict(title="NMI(cluster(t-1), cluster(t)) at speaker-change turns", hline=None, ylim=(0, 1))),

        # Alignment
        ("align_eps", dict(title="Alignment epsilon", hline=None, ylim=None)),
        ("align_lag_A_to_B", dict(title="Alignment lag A→B", hline=None, ylim=None)),
        ("align_lag_B_to_A", dict(title="Alignment lag B→A", hline=None, ylim=None)),
        ("align_lag_asymmetry_AminusB", dict(title="Alignment lag asymmetry (A→B − B→A)", hline=0.0, ylim=None)),
    ]

    present = [(m, cfg) for (m, cfg) in anchor_metrics if m in df.columns]
    if not present:
        raise ValueError("None of the expected metrics were found in the CSV. Check column names.")

    # Generate plots
    for metric, cfg in present:
        barplot_groupmean(
            df=df,
            x="expertise_guest",
            y=metric,
            outpath=out_dir / f"bar_{metric}.png",
            title=cfg.get("title"),
            hline=cfg.get("hline"),
            ylim=cfg.get("ylim"),
        )

        lineplot_groupmean(
            df=df,
            x="expertise_guest",
            y=metric,
            outpath=out_dir / f"line_{metric}.png",
            title=cfg.get("title"),
            hline=cfg.get("hline"),
        )

        violin_or_box(
            df=df,
            x="expertise_guest",
            y=metric,
            outpath=out_dir / f"dist_{metric}.png",
            title=cfg.get("title"),
            ylim=cfg.get("ylim"),
        )

    # Directional overlays (optional)
    if "cross_nn_A_to_B" in df.columns and "cross_nn_B_to_A" in df.columns:
        overlay_bars(
            df=df,
            x="expertise_guest",
            y1="cross_nn_A_to_B",
            y2="cross_nn_B_to_A",
            outpath=out_dir / "bar_cross_nn_AtoB_vs_BtoA.png",
            label1="A→B",
            label2="B→A",
            title="Cross NN (A→B vs B→A) by guest expertise",
        )
    # Entropy slope overlay (optional)
    if "entropy_slope_A" in df.columns and "entropy_slope_B" in df.columns:
        overlay_bars(
            df=df,
            x="expertise_guest",
            y1="entropy_slope_A",
            y2="entropy_slope_B",
            outpath=out_dir / "bar_entropy_slope_A_vs_B.png",
            label1="slope A",
            label2="slope B",
            title="Entropy slope (A vs B) by guest expertise",
        )

    # NMI overlay (optional)
    if "nmi_speaker_cluster" in df.columns and "nmi_turn_clusters" in df.columns:
        overlay_bars(
            df=df,
            x="expertise_guest",
            y1="nmi_speaker_cluster",
            y2="nmi_turn_clusters",
            outpath=out_dir / "bar_nmi_speaker_vs_turn.png",
            label1="NMI speaker↔cluster",
            label2="NMI turn cluster(t-1)↔cluster(t)",
            title="NMI metrics by guest expertise",
        )

    if "align_lag_A_to_B" in df.columns and "align_lag_B_to_A" in df.columns:
        overlay_bars(
            df=df,
            x="expertise_guest",
            y1="align_lag_A_to_B",
            y2="align_lag_B_to_A",
            outpath=out_dir / "bar_align_lag_AtoB_vs_BtoA.png",
            label1="A→B",
            label2="B→A",
            title="Alignment lag (A→B vs B→A) by guest expertise",
        )

    print(f"[DONE] Plots written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
