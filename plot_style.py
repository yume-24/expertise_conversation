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


def plot_metric_trajectories(frame):
    """Same descriptive figure in both paper notebooks; no uncertainty bands."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (metric, label) in zip(axes.flat, METRIC_LABELS.items()):
        for _, group in frame.groupby(['dataset', 'video_id'], observed=True):
            group = group.sort_values('level')
            ax.plot(group['level'], group[metric], color=NEUTRAL, alpha=0.35, linewidth=0.8)
        means = frame.groupby('level')[metric].mean()
        ax.plot(means.index, means.values, 'o-', color=PRIMARY, linewidth=2.3, label='Across-video mean')
        ax.set(title=label, ylabel=METRIC_DETAILS[metric], xlabel='Partner expertise',
               xticks=range(5), xticklabels=LEVEL_LABELS)
        ax.tick_params(axis='x', rotation=20)
        ax.grid(axis='x', visible=False)
    from matplotlib.lines import Line2D
    axes[0, 0].legend(handles=[Line2D([], [], color=NEUTRAL, linewidth=1, label='Within-video trajectory'),
                              Line2D([], [], color=PRIMARY, marker='o', label='Across-video mean')])
    return fig, axes
