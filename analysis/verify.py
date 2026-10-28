"""Execute frozen notebooks and Python modules independently, then compare results.

python -m analysis.verify --config analysis/configs/paper.toml
Uses separate output directories; does not overwrite the main analysis outputs.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from .config import ROOT, load_config
from .provenance import sha256, write_json


def notebook_reference(config, destination):
    import nbformat
    from nbclient import NotebookClient
    reference = ROOT / 'notebooks/reference'
    snapshot = json.loads((reference / 'snapshot.json').read_text())
    for name, expected in snapshot.items():
        if sha256(reference / name) != expected:
            raise ValueError(f'Frozen notebook reference was modified: {name}')
    destination.mkdir(parents=True, exist_ok=True)
    for name in ['01_static_geometric_metrics.ipynb', '02_statistical_significance.ipynb']:
        notebook = nbformat.read(reference / name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            cell.outputs = []
            cell.execution_count = None
            source = cell.source
            source = source.replace('from plot_style import (', 'from notebooks.reference.plot_style import (')
            source = source.replace('PROJECT_ROOT / "outputs/paper/statistical_significance"', f'Path({str(destination / "statistical_significance")!r})')
            source = source.replace('PROJECT_ROOT / "outputs/paper/conversation_metrics.csv"', f'Path({str(destination / "conversation_metrics.csv")!r})')
            source = source.replace('PROJECT_ROOT / "outputs/paper"', f'Path({str(destination)!r})')
            source = source.replace('PROJECT_ROOT / "data/wired"', f'Path({str(config.input)!r})')
            source = source.replace('SEED = 20260913', f'SEED = {config.seed}')
            source = source.replace('N_BOOTSTRAP = 999', f'N_BOOTSTRAP = {config.bootstrap_replicates}')
            source = source.replace('ALPHA = 0.05', f'ALPHA = {config.alpha}')
            source = source.replace('LOCAL_FILES_ONLY = True', f'LOCAL_FILES_ONLY = {config.local_files_only}')
            source = source.replace('batch_size=32', f'batch_size={config.batch_size}')
            source = source.replace('"sentence-transformers/all-MiniLM-L6-v2"', repr(config.model))
            source = source.replace('"c9745ed1d9f207416be6d2e6f8de32d1f16199bf"', repr(config.revision))
            source = source.replace('utterances["n_words"] >= 5', f'utterances["n_words"] >= {config.centroid_min_words}')
            source = source.replace('n_words >= 5', config.selection)
            cell.source = source
        client = NotebookClient(notebook, kernel_name='python3', timeout=1800,
                                resources={'metadata': {'path': str(ROOT)}})
        manager = client.create_kernel_manager()
        manager.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
        print(f'Reference: executing {name}', flush=True)
        try:
            client.execute()
        finally:
            nbformat.write(notebook, destination / name)
        print(f'Reference: completed {name}', flush=True)


def compare(reference, candidate, report_path, config):
    comparisons = []
    left = np.load(reference / 'utterance_embeddings.npy', allow_pickle=False)
    right = np.load(candidate / 'embeddings/utterance_embeddings.npy', allow_pickle=False)
    np.testing.assert_array_equal(left, right)
    comparisons.append({'artifact': 'utterance_embeddings.npy', 'exact': True, 'shape': list(left.shape)})
    pairs = [
        ('utterances_with_embedding_index.csv', 'embeddings/utterances_with_embedding_index.csv'),
        ('conversation_metrics.csv', 'tables/conversation_metrics.csv'),
    ]
    for name in ['analysis_data.csv', 'excluded_conversations.csv', 'video_level_coverage.csv',
                 'mixed_model_results.csv', 'paper_results_table.csv', 'bootstrap_null_statistics.csv']:
        pairs.append((f'statistical_significance/{name}', f'tables/statistics/{name}'))
    for before, after in pairs:
        a, b = pd.read_csv(reference / before), pd.read_csv(candidate / after)
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        comparisons.append({'artifact': after, 'exact': True, 'rows': len(a), 'columns': len(a.columns)})
    write_json(report_path, {'status': 'passed', 'comparison': 'Exact arrays and parsed CSV values (no tolerance)',
                             'config': config.to_dict(), 'artifacts': comparisons,
                             'notes': 'PDF metadata and model-summary timestamps are not compared. Numerical tables include all bootstrap draws.'})
    print(f'PASS: exact equality for embeddings and {len(pairs)} tables. Report: {report_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--compare-only', action='store_true', help='Compare existing validation runs without executing them')
    args = parser.parse_args()
    config = load_config(args.config)
    # Name validated by fixed directory policy: do not accept traversal here.
    if Path(config.name).name != config.name or config.name in ['.', '..']:
        parser.error('config.name must be a simple directory name')
    directory = ROOT / 'outputs/validation' / config.name
    reference, candidate = directory / 'notebook', directory / 'python'
    if not args.compare_only:
        notebook_reference(config, reference)
        from .run import run
        import matplotlib
        matplotlib.use('Agg')
        print('Python: executing configured pipeline', flush=True)
        run(replace(config, output_dir=str(candidate)), rebuild_embeddings=True)
    try:
        compare(reference, candidate, directory / 'comparison.json', config)
    except Exception as exc:
        write_json(directory / 'comparison.json', {'status': 'failed', 'error': str(exc), 'config': config.to_dict()})
        raise


if __name__ == '__main__':
    main()
