#!/usr/bin/env python3
"""Walk through one conversation's paper metrics using cached utterance embeddings.

Run: python -m analysis.visualize_metric_walkthrough --conversation wired_gravity_12

Dependencies: numpy, pandas, matplotlib (no model, downloads, or project imports).
Outputs: self-contained HTML with playback/slider, final PNG/PDF, and numeric CSV/JSON.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"A": "#167d9a", "B": "#d56b35"}
QUARTILES = ["#4c9eaa", "#79ae62", "#e0a63e", "#b45b83"]
TOL = 100 * np.finfo(float).eps


def load_conversation(input_dir, conversation, dataset):
    """Select in transcript order, indexing embeddings by explicit row IDs."""
    rows = pd.read_csv(input_dir / "utterances_with_embedding_index.csv")
    required = {"dataset", "conversation_id", "utterance_number", "embedding_idx", "speaker", "text"}
    if required - set(rows):
        raise ValueError(f"Missing columns: {sorted(required - set(rows))}")
    rows = rows.loc[rows.dataset.eq(dataset) & rows.conversation_id.eq(conversation)]
    rows = rows.sort_values("utterance_number", kind="stable").reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No conversation {dataset}/{conversation}; use --list.")
    if rows.utterance_number.isna().any() or rows.utterance_number.duplicated().any():
        raise ValueError("Utterance numbers must be present and unique within the conversation.")
    if rows.text.isna().any() or rows.text.str.strip().eq("").any():
        raise ValueError("Expected nonempty utterance text.")
    if not rows.speaker.isin(COLORS).all():
        raise ValueError("Expected canonical speakers A (expert) and B (partner).")
    embeddings = np.load(input_dir / "utterance_embeddings.npy", allow_pickle=False, mmap_mode="r")
    idx = pd.to_numeric(rows.embedding_idx, errors="raise").to_numpy(float)
    if (embeddings.ndim != 2 or embeddings.shape[1] < 1 or not np.isfinite(idx).all()
            or np.any(idx != np.floor(idx)) or np.any(idx < 0)
            or np.any(idx >= len(embeddings)) or len(np.unique(idx)) != len(idx)):
        raise ValueError("Invalid embedding matrix or embedding_idx mapping.")
    x = np.asarray(embeddings[idx.astype(int)], dtype=float)
    if not np.isfinite(x).all() or np.any(np.linalg.norm(x, axis=1) <= TOL):
        raise ValueError("Embeddings must be finite and nonzero.")
    # These exports are normalized by notebook 01. Reject a different input policy.
    if not np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5):
        raise ValueError("Expected unit-normalized embeddings from paper notebook 01.")
    rows["n_words"] = rows.text.map(lambda t: len(re.findall(r"\b[\w'-]+\b", t)))
    rows["centroid_eligible"] = rows.text.str.strip().ne("")
    return rows, x


def calculate_metrics(rows, x):
    """Match notebook 01; all arithmetic precedes the illustrative PCA projection."""
    unit = x / np.linalg.norm(x, axis=1, keepdims=True)
    pooled = unit.mean(axis=0)
    centered = unit - pooled
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    # One basis fitted to the entire conversation, including future utterances.
    basis = np.zeros((2, x.shape[1]))
    basis[:min(2, len(vt))] = vt[:2]
    for component in basis:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    xy = centered @ basis.T
    radial = (np.clip(1 - unit @ (pooled / np.linalg.norm(pooled)), 0, 2)
              if np.linalg.norm(pooled) > TOL else np.full(len(x), np.nan))
    quantiles = (np.quantile(radial, [.25, .5, .75, .9])
                 if np.isfinite(radial).all() else np.full(4, np.nan))
    centroids = {}
    for speaker in COLORS:
        keep = rows.speaker.eq(speaker) & rows.centroid_eligible
        if keep.any():
            # Notebook speaker means use the cached (already normalized) vectors.
            centroids[speaker] = x[keep].mean(axis=0)
    distance = np.nan
    if len(centroids) == 2:
        a, b = centroids["A"], centroids["B"]
        if min(np.linalg.norm(a), np.linalg.norm(b)) > TOL:
            distance = float(1 - np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))
    eigenvalues = singular**2 / max(len(x) - 1, 1)
    shares = np.zeros_like(eigenvalues)
    pr = np.nan
    if not np.isclose(eigenvalues.sum(), 0, atol=1e-15, rtol=0):
        active = singular > max(x.shape) * np.finfo(float).eps * singular[0]
        shares = eigenvalues / eigenvalues.sum()
        active_shares = eigenvalues[active] / eigenvalues[active].sum()
        pr = float(1 / np.sum(active_shares**2))
    return dict(xy=xy, pooled=pooled, basis=basis, radial=radial, quantiles=quantiles,
                centroid_xy={k: (v - pooled) @ basis.T for k, v in centroids.items()},
                distance=distance, shares=shares, pr=pr)


def number(value):
    return f"{value:.4f}" if np.isfinite(value) else "undefined"


def setup_plane(ax, xy, title):
    ax.set_title(title, loc="left", fontweight="bold", pad=12)
    low, high = xy.min(axis=0), xy.max(axis=0)
    pad = np.maximum(high - low, .1) * .14
    ax.set(xlim=(low[0] - pad[0], high[0] + pad[0]),
           ylim=(low[1] - pad[1], high[1] + pad[1]), xlabel="PC1", ylabel="PC2")
    ax.set_aspect("equal", adjustable="box")


def speaker_points(ax, rows, xy, n=None):
    n = len(rows) if n is None else n
    for speaker, color in COLORS.items():
        mask = rows.speaker.iloc[:n].eq(speaker).to_numpy()
        ax.scatter(*xy[:n][mask].T, s=30, color=color, alpha=.8,
                   label=f"{speaker} · {'expert' if speaker == 'A' else 'partner'}")


def make_first_steps(rows, m, conversation, output):
    """A cumulative t=0/1/2 storyboard, with each complete source utterance."""
    count = min(3, len(rows))
    wrapped = [textwrap.fill(str(row.text), width=26) for row in rows.iloc[:count].itertuples()]
    fig, axes = plt.subplots(1, count, figsize=(6 * count, 6.3), layout="constrained", squeeze=False)
    offsets = [(-16, 32), (-12, 34), (12, -38)]
    for t in range(count):
        ax = axes[0, t]
        setup_plane(ax, m["xy"], f"t={t} · {t + 1} embedding{'s' if t else ''}")
        speaker_points(ax, rows, m["xy"], t + 1)
        ax.plot(*m["xy"][:t + 1].T, color="#c7cdd1", linewidth=1, zorder=0)
        ax.scatter(*m["xy"][t], s=180, facecolors="none", edgecolors="#202c35", linewidth=1.8)
        for j in range(t + 1):
            row = rows.iloc[j]
            dx, dy = offsets[j]
            ax.annotate(f"t={j} · Speaker {row.speaker}\n{wrapped[j]}", m["xy"][j],
                        xytext=(dx, dy), textcoords="offset points", fontsize=10,
                        ha="right" if dx < 0 else "left", va="bottom" if dy > 0 else "top",
                        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
                              "edgecolor": COLORS[row.speaker], "alpha": .96},
                        arrowprops={"arrowstyle": "-", "color": COLORS[row.speaker], "shrinkB": 8},
                        zorder=6)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("First utterances · fixed PCA projection", fontsize=13)
    save_figure(fig, output, "01_time_series_t0_t2")
    plt.close(fig)


def make_timeline(rows, metrics, conversation, interval):
    xy = metrics["xy"]
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.subplots_adjust(bottom=.27, top=.88)
    transcript = fig.text(.08, .12, "", va="top", fontsize=10)
    fig.text(.08, .025, "One point = one utterance. Fixed full-conversation PCA; axes never refit during playback.",
             fontsize=9, color="#555")

    def draw(t):
        ax.clear()
        setup_plane(ax, xy, f"{conversation}  |  t={t}  |  {t + 1}/{len(rows)} utterances")
        ax.plot(*xy[:t + 1].T, color="#c7cdd1", linewidth=1, zorder=0)
        speaker_points(ax, rows, xy, t + 1)
        ax.scatter(*xy[t], s=170, facecolors="none", edgecolors="#202c35", linewidth=1.8)
        ax.annotate(f"t={t}", xy[t], xytext=(8, 8), textcoords="offset points", fontsize=10)
        ax.legend(loc="upper left", fontsize=9)
        row = rows.iloc[t]
        # Full transcript remains available in the HTML table and CSV.
        snippet = textwrap.shorten(row.text, width=360, placeholder=" … [full text below]")
        transcript.set_text(f"Speaker {row.speaker} · original utterance {row.utterance_number}\n"
                            + textwrap.fill(snippet, 112))
        return []

    animation = FuncAnimation(fig, draw, frames=len(rows), interval=interval, blit=False, repeat=False)
    # Matplotlib embeds every frame; no CDN, server or network access is needed.
    with matplotlib.rc_context({"animation.embed_limit": 1024}):
        playback = animation.to_jshtml(default_mode="once")
    # Matplotlib's controls otherwise load Font Awesome from a CDN.
    playback = re.sub(r'<link rel="stylesheet"\s+href="[^"]+">', '', playback)
    icons = {"fast-backward": "|◀", "step-backward": "◀|", "play": "▶", "pause": "Ⅱ",
             "step-forward": "|▶", "fast-forward": "▶|", "minus": "−", "plus": "+"}
    playback = playback.replace('<i class="fa fa-play fa-flip-horizontal"></i>', '◀')
    playback = re.sub(r'<i class="fa fa-([a-z-]+)"></i>',
                      lambda match: icons.get(match[1], match[1]), playback)
    plt.close(fig)
    return playback


def draw_radial_rings(ax, m):
    """Illustrate spread in the unchanged PCA plane; rings use 2D Euclidean radii."""
    points = m["xy"]
    radial_2d = np.linalg.norm(points, axis=1)
    quantiles_2d = np.quantile(radial_2d, [.25, .5, .75, .9])
    setup_plane(ax, points, "Radial spread · PCA")
    # Keep the same coordinates and axis directions; allow space for full circles.
    extent = max(float(np.abs(points).max()), float(quantiles_2d[-1]), .1) * 1.15
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent))
    for q, radius, angle in zip([25, 50, 75, 90], quantiles_2d,
                                      np.deg2rad([35, 125, 210, 300])):
        ax.add_patch(Circle((0, 0), radius, fill=False, edgecolor="#a4acb2",
                            linewidth=1, linestyle="--" if q == 90 else "-"))
        position = radius * np.array([np.cos(angle), np.sin(angle)])
        ax.annotate(f"Q{q}", position, ha="center", va="center", fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 2}, zorder=6)
    ax.scatter(*points.T, color="#537b8a", s=27, edgecolors="white", linewidths=.4, zorder=4)
    ax.scatter(0, 0, marker="*", s=150, color="#222", zorder=5)
    ax.annotate("Centroid", (0, 0), xytext=(0, -15), textcoords="offset points",
                ha="center", va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    ax.set_xlabel("PC1\nIllustrative 2D Euclidean quantiles; analysis uses full-dimensional cosine distances.",
                  fontsize=8)


def draw_centroids(ax, rows, m):
    xy = m["xy"]
    setup_plane(ax, xy, "Speaker centroids")
    speaker_points(ax, rows, xy)
    for speaker, center in m["centroid_xy"].items():
        ax.scatter(*center, marker="X", s=190, color=COLORS[speaker], edgecolors="white", zorder=5)
        ax.annotate(f"mean {speaker}", center, xytext=(20, 25) if speaker == "A" else (-55, -25),
                    textcoords="offset points", arrowprops={"arrowstyle": "-", "color": COLORS[speaker]},
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
    if len(m["centroid_xy"]) == 2:
        ax.plot(*np.array(list(m["centroid_xy"].values())).T, "--", color="#374151")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel(f"PC1\nCosine distance (full dimensions): {number(m['distance'])}. Connector is illustrative.", fontsize=9)



def draw_sorted_radial(ax, rows, m):
    ax.set_title("Radial distances", loc="left", fontweight="bold")
    if np.isfinite(m["radial"]).all():
        bands = np.searchsorted(m["quantiles"][:3], m["radial"], side="left")
        order = np.argsort(m["radial"], kind="stable")
        ranks = np.arange(1, len(rows) + 1)
        ax.scatter(ranks, m["radial"][order], c=bands[order], cmap=ListedColormap(QUARTILES),
                   vmin=0, vmax=3, s=24)
        for q, val, color in zip([25, 50, 75, 90], m["quantiles"], QUARTILES):
            ax.axhline(val, color=color, linestyle="--", label=f"Q{q} = {val:.4f}")
        ax.legend(fontsize=9)
    else:
        ax.text(.5, .5, "Pooled centroid has no direction.\nRadial metrics are undefined.",
                transform=ax.transAxes, ha="center")
    ax.set(xlabel="Utterances sorted by distance", ylabel="Cosine distance to centroid")



def draw_participation_ratio(ax, m):
    ax.set_title("Participation ratio", loc="left", fontweight="bold")
    ax.bar(np.arange(1, len(m["shares"]) + 1), m["shares"], color="#6678a5")
    ax.set(xlabel="Principal component", ylabel="Variance share")
    ax.text(.97, .93, f"PR = {m['pr']:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=11)



def save_figure(fig, output, name):
    for extension in ("png", "pdf"):
        fig.savefig(output / f"{name}.{extension}", dpi=160)


def make_final_figure(rows, m, conversation, output):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), layout="constrained")
    draw_centroids(axes[0, 0], rows, m)
    draw_radial_rings(axes[0, 1], m)
    draw_sorted_radial(axes[1, 0], rows, m)
    draw_participation_ratio(axes[1, 1], m)
    fig.suptitle(f"{conversation} · {len(rows)} utterances", fontsize=13)
    save_figure(fig, output, "final_metrics")
    plt.close(fig)
    # Each teaching step also stands alone, in presentation order.
    for name, draw in [
        ("03_speaker_centroids", lambda ax: draw_centroids(ax, rows, m)),
        ("04_radial_quartiles", lambda ax: draw_radial_rings(ax, m)),
        ("05_sorted_radial_distances", lambda ax: draw_sorted_radial(ax, rows, m)),
        ("06_participation_ratio", lambda ax: draw_participation_ratio(ax, m)),
    ]:
        fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
        draw(ax)
        ax.set_title("Radial spread · PCA" if name == "04_radial_quartiles"
                     else name.split("_", 1)[1].replace("_", " ").capitalize(), loc="left", fontweight="bold")
        save_figure(fig, output, name)
        if name == "04_radial_quartiles":
            fig.savefig(output / "radial_quartiles.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 8), layout="constrained")
    setup_plane(ax, m["xy"], "Final conversation")
    speaker_points(ax, rows, m["xy"])
    ax.legend()
    save_figure(fig, output, "02_final_embeddings")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conversation", default="wired_gravity_14")
    parser.add_argument("--dataset", default="wired")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "outputs/paper/embeddings")
    parser.add_argument("--output-dir", type=Path, help="Default: outputs/metric_walkthrough/<conversation>")
    parser.add_argument("--interval", type=int, default=600, help="Milliseconds per utterance (default: 600)")
    parser.add_argument("--static-only", action="store_true", help="Skip animation; still export HTML, figures and data")
    parser.add_argument("--list", action="store_true", help="List available conversation IDs and exit")
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error("--interval must be positive")
    try:
        if args.list:
            table = pd.read_csv(args.input_dir / "utterances_with_embedding_index.csv")
            print("\n".join(sorted(table.loc[table.dataset.eq(args.dataset), "conversation_id"].unique())))
            return
        rows, x = load_conversation(args.input_dir, args.conversation, args.dataset)
        m = calculate_metrics(rows, x)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    safe_id = re.sub(r"[^\w.-]+", "_", args.conversation).strip(".") or "conversation"
    output = args.output_dir or ROOT / "outputs/metric_walkthrough" / safe_id
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 11, "font.size": 10})
    make_first_steps(rows, m, args.conversation, output)
    make_final_figure(rows, m, args.conversation, output)
    values = dict(speaker_centroid_cosine_distance=m["distance"], radial_distance_q25=m["quantiles"][0],
                  radial_distance_median=m["quantiles"][1], radial_distance_q75=m["quantiles"][2],
                  radial_distance_q90=m["quantiles"][3], participation_ratio=m["pr"])
    provenance = dict(conversation=args.conversation, dataset=args.dataset, n_utterances=len(rows),
                      embedding_dimension=x.shape[1], pca_variance_share=float(m["shares"][:2].sum()),
                      metrics={k: float(v) if np.isfinite(v) else None for k, v in values.items()},
                      definitions={"centroids": "Means of cached normalized embeddings; all nonempty utterances per speaker",
                                   "radial": "1 - dot(unit utterance, normalized pooled mean); all utterances",
                                   "quantiles": "NumPy linear interpolation, full pooled centroid (not leave-one-out)",
                                   "pr": "Ordinary centered covariance, all utterances, active singular values",
                                   "projection": "Fixed full-conversation centered PCA of unit embeddings; retrospective",
                                   "radial_diagram": "Unchanged PCA coordinates; rings are illustrative 2D Euclidean-distance quantiles, not full-dimensional cosine quantiles"},
                      input_sha256={name: hashlib.sha256((args.input_dir / name).read_bytes()).hexdigest()
                                    for name in ["utterances_with_embedding_index.csv", "utterance_embeddings.npy"]})
    (output / "metrics.json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
    rows["t"] = np.arange(len(rows))
    rows["pc1"], rows["pc2"] = m["xy"].T
    rows["radial_cosine_distance"] = m["radial"]
    rows.to_csv(output / "utterance_walkthrough.csv", index=False)
    print(f"Rendering {len(rows)} time steps…" if not args.static_only else "Rendering static walkthrough…", flush=True)
    playback = "<p>Animation omitted (--static-only).</p>" if args.static_only else make_timeline(
        rows, m, args.conversation, args.interval)
    encoded = base64.b64encode((output / "final_metrics.png").read_bytes()).decode("ascii")
    title = html.escape(args.conversation)
    result_table = pd.DataFrame({"Metric": list(values), "Value": [number(v) for v in values.values()]}).to_html(index=False)
    transcript = rows[["t", "speaker", "text", "centroid_eligible", "radial_cosine_distance"]].to_html(index=False, escape=True)
    page = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} · Metric walkthrough</title>
<style>body{{max-width:1150px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#24313c;background:#fafbfc}}
h1,h2{{line-height:1.2}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:9px;text-align:left;border:1px solid #ddd}}.animation{{overflow-x:auto}}details{{margin-top:25px}}
code{{background:#eef1f4;padding:2px 5px}}section{{margin:35px 0}}</style>
<h1>{title}: how the conversation metrics work</h1>
<p>{len(rows)} utterances · {x.shape[1]} embedding dimensions · speakers A (expert) and B (partner).</p>
<section><h2>1. Follow the conversation, t=0 onward</h2>
<p>Press play or drag the slider. Each new dot is the next utterance, with a ring around the current one.
The connecting path follows transcript order; it is not a metric. The fixed PCA plane uses the complete
conversation and retains {m['shares'][:2].sum():.1%} of its variance. This is a retrospective view.</p>
{playback}</section><section><h2>2. Inspect the final conversation</h2>
<img src="data:image/png;base64,{encoded}" alt="Speaker centroids, radial quartile map, sorted radial distances and participation-ratio spectrum">
<p><b>Speaker centroids:</b> average all nonempty utterance embeddings for each speaker, then compute
<code>1 − (cA · cB) / (‖cA‖ ‖cB‖)</code>. The dashed line connects their projected means;
its length does not measure cosine distance. Short acknowledgments and questions contribute equally with other utterances.</p>
<p><b>Radial distances:</b> normalize all utterance vectors to uᵢ, take the pooled mean μ,
then compute <code>rᵢ = 1 − uᵢ · (μ / ‖μ‖)</code>. Sort these distances to read Q25, Q50, Q75, and Q90.
The paper uses Q50 (median) and Q90; Q90 is a percentile, not a quartile.
The ring diagram keeps the same PCA coordinates as the other embedding plots.
Its rings enclose approximately 25%, 50%, 75% and 90% of points by <b>2D Euclidean distance</b>
from the projected pooled centroid (interpolation and ties affect exact counts).
<b>For interpretation only:</b> these rings are not the paper's full-dimensional cosine-distance
quantiles. Projection can change both distances and their rankings. The separate sorted-distance
plot and numeric table report the actual full-dimensional radial metrics.</p>
<p><b>Participation ratio:</b> center all unit embeddings, calculate covariance eigenvalues λⱼ,
and compute <code>PR = (Σλⱼ)² / Σλⱼ² = 1 / Σpⱼ²</code>. Equal variance in k directions gives PR=k;
concentration in one direction gives PR=1. The spectrum uses all components, including those outside the picture.</p>
{result_table}<p>Undefined values occur when a required speaker mean or pooled direction is unavailable,
or when there is no embedding variance.</p></section>
<details><summary>Full transcript and per-utterance calculations</summary>{transcript}</details></html>"""
    (output / "walkthrough.html").write_text(page, encoding="utf-8")
    print(f"Open {output / 'walkthrough.html'}")
    print("\n".join(f"{k}: {number(v)}" for k, v in values.items()))


if __name__ == "__main__":
    main()
