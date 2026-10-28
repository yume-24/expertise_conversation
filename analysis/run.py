"""Run the configured paper pipeline: python -m analysis.run --help."""
import argparse
import json
import random
import numpy as np
import pandas as pd
from .config import load_config
from .provenance import (check_metric_inputs, code_hashes, packages, save_run_config,
                         sha256, transcript_hashes, write_json)


def embeddings_stage(config, rebuild=False):
    from .transcripts import load_transcripts
    from .embeddings import embed_utterances
    random.seed(config.seed)
    np.random.seed(config.seed)
    rows, embeddings = embed_utterances(load_transcripts(config), config, rebuild)
    rows.to_csv(config.output / 'embeddings/utterances_with_embedding_index.csv', index=False)
    return rows, embeddings


def metrics_stage(config, rebuild=False):
    from .metrics import calculate_all_metrics
    rows, embeddings = embeddings_stage(config, rebuild)
    metrics = calculate_all_metrics(rows, embeddings, config.centroid_min_words)
    tables = config.output / 'tables'
    tables.mkdir(parents=True, exist_ok=True)
    target = tables / 'conversation_metrics.csv'
    metrics.to_csv(target, index=False)
    manifest = {
        'model': config.model, 'model_revision': config.revision, 'device': 'cpu', 'seed': config.seed,
        'n_utterances': len(rows), 'n_conversations': len(metrics),
        'n_videos': metrics[['dataset', 'video_id']].drop_duplicates().shape[0],
        'centroid_selection': config.selection, 'radial_and_pr_selection': 'all nonempty utterances',
        'radial_variant': 'full pooled centroid, not leave-one-out',
        'pr_variant': 'ordinary centered covariance, not row-bias-corrected',
        'count_covariate': 'total nonempty utterances', 'transcript_sha256': transcript_hashes(config),
        'metrics_sha256': sha256(target), 'packages': packages(), 'config': config.to_dict(),
        'code_sha256': code_hashes(),
    }
    write_json(tables / 'metrics_manifest.json', manifest)
    print(f'Exported {len(metrics)} conversations, {len(rows)} utterances.', flush=True)
    return metrics


def statistics_stage(config):
    from .statistics import prepare_data, fit_models, bootstrap_tests, diagnostic_data, REPORT_COLUMNS
    check_metric_inputs(config)
    source = config.output / 'tables/conversation_metrics.csv'
    directory = config.output / 'tables/statistics'
    directory.mkdir(parents=True, exist_ok=True)
    data, excluded, coverage = prepare_data(pd.read_csv(source))
    data.to_csv(directory / 'analysis_data.csv', index=False)
    excluded.to_csv(directory / 'excluded_conversations.csv', index=False)
    coverage.to_csv(directory / 'video_level_coverage.csv')
    results, fits, frames = fit_models(data)
    results, draws = bootstrap_tests(data, results, frames, config.seed, config.bootstrap_replicates, config.alpha)
    diagnostics = diagnostic_data(data, results, fits, frames)
    results.to_csv(directory / 'mixed_model_results.csv', index=False)
    results[REPORT_COLUMNS].to_csv(directory / 'paper_results_table.csv', index=False)
    draws.to_csv(directory / 'bootstrap_null_statistics.csv', index=False)
    diagnostics.to_csv(directory / 'diagnostics.csv', index=False)
    summaries = '\n\n'.join(f'{metric} | {adjustment}\n{fit.summary()}' for (metric, adjustment), fit in fits.items())
    (directory / 'model_summaries.txt').write_text(summaries)
    write_json(directory / 'analysis_manifest.json', {
        'source_sha256': sha256(source), 'config': config.to_dict(), 'packages': packages(),
        'centroid_selection': config.selection, 'n_conversations': len(data), 'n_videos': data.video_group.nunique(),
        'seed': config.seed, 'bootstrap_replicates_per_test': config.bootstrap_replicates,
        'estimation': 'REML coefficients and pointwise Wald CIs; ML parametric-bootstrap LRT p-values',
        'multiplicity_family': 'all eight expertise tests; BH and Holm', 'code_sha256': code_hashes(),
        'outputs_sha256': {p.name: sha256(p) for p in directory.glob('*.csv')},
    })
    return results


def figures_stage(config):
    import matplotlib.pyplot as plt
    from .plots import apply_style, plot_metric_trajectories, save_figure, plot_expertise_effects, plot_diagnostics
    check_metric_inputs(config)
    apply_style()
    metrics = pd.read_csv(config.output / 'tables/conversation_metrics.csv')
    for name, (fig, _) in plot_metric_trajectories(metrics, config.centroid_min_words).items():
        save_figure(fig, config.output / 'figures' / f'main_metric_trajectories_{name}')
        plt.close(fig)
    directory = config.output / 'tables/statistics'
    manifest_path = directory / 'analysis_manifest.json'
    if not manifest_path.exists():
        print('Saved metric figures. Run statistics for effects and diagnostics.', flush=True)
        return
    manifest = json.loads(manifest_path.read_text())
    expected = manifest['config']
    if (manifest['source_sha256'] != sha256(config.output / 'tables/conversation_metrics.csv') or
            any(expected[k] != config.to_dict()[k] for k in ['seed', 'bootstrap_replicates', 'alpha', 'centroid_min_words'])):
        raise ValueError('Statistical results are stale for this config; rerun statistics.')
    for name, digest in manifest['outputs_sha256'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Statistical output changed: {name}; rerun statistics.')
    results = pd.read_csv(directory / 'mixed_model_results.csv')
    diagnostics = pd.read_csv(directory / 'diagnostics.csv')
    for name, figure in [('expertise_effects', plot_expertise_effects(results)),
                         ('residual_diagnostics', plot_diagnostics(results, diagnostics))]:
        save_figure(figure, config.output / 'figures' / name)
        plt.close(figure)
    print('Saved separate PNG/PDF figures.', flush=True)


def run(config, stage='all', rebuild_embeddings=False):
    if stage not in ['all', 'embeddings', 'metrics', 'statistics', 'figures']:
        raise ValueError(f'Unknown stage: {stage}')
    save_run_config(config)
    if stage == 'embeddings':
        embeddings_stage(config, rebuild_embeddings)
    if stage in ['all', 'metrics']:
        metrics_stage(config, rebuild_embeddings)
    if stage in ['all', 'statistics']:
        statistics_stage(config)
    if stage in ['all', 'figures']:
        figures_stage(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Default: analysis/configs/paper.toml')
    parser.add_argument('--stage', choices=['all', 'embeddings', 'metrics', 'statistics', 'figures'], default='all')
    parser.add_argument('--output-dir', help='Override output directory (relative to repository or absolute)')
    parser.add_argument('--rebuild-embeddings', action='store_true', help='Re-encode even if a validated cache exists')
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    config = load_config(args.config, args.output_dir)
    print(f'{config.name}: {config.selection}; output: {config.output}', flush=True)
    run(config, args.stage, args.rebuild_embeddings)


if __name__ == '__main__':
    main()
