"""Content hashes and stage manifests prevent using unrelated cached results."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
from .config import ROOT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def packages():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'pandas', 'scipy', 'statsmodels', 'matplotlib', 'sentence-transformers', 'torch']}


def transcript_hashes(config):
    paths = sorted(config.input.glob('wired_*/*.csv'))
    if not paths:
        raise FileNotFoundError(f'No transcripts under {config.input}')
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha256(path) for path in paths}


def code_hashes():
    return {p.name: sha256(p) for p in sorted((ROOT / 'analysis').glob('*.py'))}


def save_run_config(config):
    write_json(config.output / 'run_config.json', config.to_dict())


def check_metric_inputs(config):
    path = config.output / 'tables/metrics_manifest.json'
    if not path.exists():
        raise FileNotFoundError('Run the metrics stage first.')
    manifest = json.loads(path.read_text())
    if manifest['centroid_selection'] != config.selection:
        raise ValueError('Cached metrics use a different centroid selection; rerun metrics.')
    if manifest['transcript_sha256'] != transcript_hashes(config):
        raise ValueError('Transcripts changed; rerun metrics.')
    if manifest['model'] != config.model or manifest['model_revision'] != config.revision or manifest['seed'] != config.seed:
        raise ValueError('Embedding settings changed; rerun metrics.')
    if manifest['metrics_sha256'] != sha256(config.output / 'tables/conversation_metrics.csv'):
        raise ValueError('Metric export does not match its manifest; rerun metrics.')
    return manifest
