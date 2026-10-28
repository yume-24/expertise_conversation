"""Shared plotting conventions for paper and exploratory notebooks."""
from pathlib import Path
import matplotlib.pyplot as plt
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


def apply_style():
    """Set defaults without changing data, scales, or uncertainty calculations."""
    sns.set_theme(style='whitegrid', context='notebook', palette=list(EXPERTISE_COLORS.values()), rc={
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 11, 'axes.titleweight': 'normal', 'axes.labelsize': 10,
        'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
        'legend.title_fontsize': 9, 'legend.frameon': False,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': NEUTRAL, 'axes.labelcolor': TEXT, 'text.color': TEXT,
        'xtick.color': TEXT, 'ytick.color': TEXT,
        'grid.color': '#dce1e5', 'grid.linewidth': 0.6, 'grid.alpha': 0.6,
        'axes.axisbelow': True, 'lines.linewidth': 1.8, 'lines.markersize': 5,
        'figure.figsize': (8, 5), 'figure.dpi': 110, 'savefig.dpi': 300,
        'savefig.bbox': 'tight', 'savefig.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


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


def plot_metric_trajectories(frame, centroid_min_words=5):
    """Return {metric: (figure, axis)} with one descriptive figure per metric.

    Both paper notebooks save these separately. Lines and means use the same
    data as the former four-panel figure; no uncertainty bands are added.
    """
    from matplotlib.lines import Line2D

    figures = {}
    for metric, label in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        for _, group in frame.groupby(['dataset', 'video_id'], observed=True):
            group = group.sort_values('level')
            ax.plot(group['level'], group[metric], color=NEUTRAL, alpha=0.35, linewidth=0.8)
        means = frame.groupby('level')[metric].mean()
        ax.plot(means.index, means.values, 'o-', color=PRIMARY, linewidth=2.3, label='Across-video mean')
        ax.set(title=label, ylabel=("Cosine distance · all utterances" if centroid_min_words == 0 else f"Cosine distance · ≥{centroid_min_words}-word utterances") if metric == "speaker_centroid_cosine_distance" else METRIC_DETAILS[metric], xlabel='Partner expertise',
               xticks=range(5), xticklabels=LEVEL_LABELS)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='x', visible=False)
        ax.legend(handles=[Line2D([], [], color=NEUTRAL, linewidth=1, label='Within-video trajectory'),
                           Line2D([], [], color=PRIMARY, marker='o', label='Across-video mean')])
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
