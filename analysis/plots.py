"""Shared plotting conventions for paper and exploratory notebooks."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

PRIMARY = '#156e8a'
SECONDARY = '#ca6b32'
NEUTRAL = '#8a9299'
TEXT = '#28323b'
LEVEL_ORDER = ['child', 'teenager', 'undergraduate', 'graduate', 'expert']
LEVEL_LABELS = ['Child', 'Teenager', 'Undergraduate', 'Graduate', 'Expert']
EXPERTISE_COLORS = dict(zip(LEVEL_ORDER, sns.color_palette('cividis', 5)))
MODEL_COLORS = {'Unadjusted': PRIMARY, 'Count-adjusted': SECONDARY}
MODEL_MARKERS = {'Unadjusted': 'o', 'Count-adjusted': 's'}
METRIC_LABELS = {
    'speaker_centroid_cosine_distance': 'Speaker-centroid distance',
    'radial_distance_median': 'Median radial distance',
    'radial_distance_q90': 'Q90 radial distance',
    'participation_ratio': 'Participation ratio (ordinary)',
}
METRIC_DETAILS = {
    'speaker_centroid_cosine_distance': 'Cosine distance · ≥5-word utterances',
    'radial_distance_median': 'Cosine distance · all utterances',
    'radial_distance_q90': 'Cosine distance · all utterances',
    'participation_ratio': 'Effective dimensions · all utterances',
}

# Paper figures: plain axis labels, short level names, one common size.
AXIS_LABELS = {
    'speaker_centroid_cosine_distance': 'Centroid distance',
    'radial_distance_median': 'Median radial distance',
    'radial_distance_q90': '90th-percentile radial distance',
    'participation_ratio': 'Participation ratio',
}
LEVEL_SHORT = ['Child', 'Teen', 'Undergrad', 'Graduate', 'Expert']
PAPER_FIGSIZE = (4.2, 3.0)
PAPER_RC = {
    'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9, 'legend.frameon': False,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False, 'axes.linewidth': 0.8,
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'text.color': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'lines.linewidth': 1.5, 'lines.markersize': 5,
    'figure.dpi': 110, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
    'pdf.fonttype': 42, 'ps.fonttype': 42,
}


def apply_style():
    """Plain black-and-white style shared with the walkthrough figures (no grid, no color)."""
    sns.set_theme(style='ticks', rc=PAPER_RC)


def expertise_palette(levels):
    """Keep expertise colors stable when levels are reordered or missing."""
    aliases = {'teen': 'teenager', 'college': 'undergraduate', 'college student': 'undergraduate',
               'graduate student': 'graduate'}
    return {level: EXPERTISE_COLORS[aliases.get(str(level).lower().strip(), str(level).lower().strip())]
            for level in levels}


def save_figure(fig, path):
    """Save consistent raster and vector versions of a paper figure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')


def _expertise_axis(ax, metric):
    ax.set(xticks=range(5), xticklabels=LEVEL_SHORT, xlim=(-.4, 4.4),
           xlabel='Partner expertise', ylabel=AXIS_LABELS[metric])


def plot_metric_trajectories(frame, centroid_min_words=5):
    """Return {metric: (figure, axis)}: one gray line per video, black across-video mean.

    centroid_min_words is kept for compatibility; the cutoff belongs in the caption."""
    figures = {}
    for metric in METRIC_LABELS:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, constrained_layout=True)
        for _, group in frame.groupby(['dataset', 'video_id'], observed=True):
            group = group.sort_values('level')
            ax.plot(group['level'], group[metric], color='0.75', linewidth=0.7)
        means = frame.groupby('level')[metric].mean()
        ax.plot(means.index, means.values, 'o-', color='black', linewidth=2)
        _expertise_axis(ax, metric)
        figures[metric] = (fig, ax)
    return figures


def plot_metric_distributions(frame, seed=0):
    """Return {metric: (figure, axis)}: box plot per level with one gray dot per conversation."""
    rng = np.random.default_rng(seed)
    black = {'color': 'black'}
    figures = {}
    for metric in METRIC_LABELS:
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, constrained_layout=True)
        data = [frame.loc[frame['level'].eq(level), metric].to_numpy() for level in range(5)]
        ax.boxplot(data, positions=range(5), widths=0.5, showfliers=False,
                   boxprops=black, whiskerprops=black, capprops=black,
                   medianprops={'color': 'black', 'linewidth': 1.5})
        for level, values in enumerate(data):
            ax.scatter(level + rng.uniform(-0.12, 0.12, len(values)), values,
                       s=10, color='0.5', linewidths=0, zorder=3)
        _expertise_axis(ax, metric)
        figures[metric] = (fig, ax)
    return figures


def plot_expertise_effects(results):
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for adjustment, offset in [("Unadjusted", -0.12), ("Count-adjusted", 0.12)]:
        color = MODEL_COLORS[adjustment]
        subset = results[results.adjustment.eq(adjustment)].set_index("metric").loc[list(METRIC_LABELS)]
        ax.errorbar(subset.beta_sd, np.arange(4)+offset,
                    xerr=np.vstack([subset.beta_sd-subset.ci_low_sd, subset.ci_high_sd-subset.beta_sd]),
                    fmt=MODEL_MARKERS[adjustment], capsize=3, color=color, label=adjustment)
    ax.axvline(0, color=NEUTRAL, linestyle="--", linewidth=1)
    ax.set(yticks=np.arange(4), yticklabels=list(METRIC_LABELS.values()),
           xlabel="Metric SD change per expertise level\n(pointwise 95% Wald CI)",
           title="Expertise association with and without utterance-count adjustment")
    ax.invert_yaxis()
    ax.legend()
    ax.grid(axis="y", visible=False)
    return fig


def plot_diagnostics(results, diagnostics):
    from scipy import stats
    fig, axes = plt.subplots(len(results), 2, figsize=(12, 3*len(results)), constrained_layout=True)
    for i, row in results.iterrows():
        points = diagnostics.loc[diagnostics.metric.eq(row.metric) & diagnostics.adjustment.eq(row.adjustment)]
        axes[i, 0].scatter(points.fitted, points.residual, alpha=0.65, s=22, color=MODEL_COLORS[row.adjustment], marker=MODEL_MARKERS[row.adjustment])
        axes[i, 0].axhline(0, color=NEUTRAL, linestyle='--')
        axes[i, 0].set(xlabel='Conditional fitted value (SD)', ylabel='Residual (SD)',
                       title=f'{METRIC_LABELS[row.metric]} — {row.adjustment}')
        stats.probplot(points.residual, dist='norm', plot=axes[i, 1])
        samples, reference = axes[i, 1].get_lines()
        samples.set(color=MODEL_COLORS[row.adjustment], marker=MODEL_MARKERS[row.adjustment], markersize=4)
        reference.set(color=NEUTRAL, linewidth=1.2)
        axes[i, 1].set(title='Residual Q–Q plot', xlabel='Theoretical normal quantiles', ylabel='Ordered residual (SD)')
    return fig
