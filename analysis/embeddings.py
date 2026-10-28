"""One normalized MiniLM embedding per utterance, with a validated local cache."""
import hashlib
import json
import os
import numpy as np
from .provenance import packages, sha256, write_json


def embed_utterances(rows, config, rebuild=False):
    directory = config.output / 'embeddings'
    directory.mkdir(parents=True, exist_ok=True)
    matrix_path = directory / 'utterance_embeddings.npy'
    manifest_path = directory / 'embedding_manifest.json'
    texts = rows['text'].astype(str).tolist()
    signature = {
        'texts_sha256': hashlib.sha256(json.dumps(texts, ensure_ascii=False).encode()).hexdigest(),
        'ids_sha256': hashlib.sha256('\n'.join(rows.utterance_id).encode()).hexdigest(),
        'model': config.model, 'revision': config.revision, 'seed': config.seed,
        'batch_size': config.batch_size, 'device': 'cpu', 'packages': packages(),
    }
    if not rebuild and matrix_path.exists() and manifest_path.exists():
        cached = json.loads(manifest_path.read_text())
        if cached.get('signature') == signature and cached.get('matrix_sha256') == sha256(matrix_path):
            matrix = np.load(matrix_path, allow_pickle=False)
            validate_embeddings(matrix, len(rows))
            if len(cached['n_model_tokens']) != len(rows):
                raise ValueError('Cached token counts do not align with utterances')
            rows = rows.copy()
            rows['n_model_tokens'] = cached['n_model_tokens']
            rows['exceeds_model_limit'] = False
            print('Using validated embedding cache.', flush=True)
            return rows, matrix
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    from sentence_transformers import SentenceTransformer
    torch.manual_seed(config.seed)
    model = SentenceTransformer(config.model, revision=config.revision,
                                local_files_only=config.local_files_only, device='cpu')
    rows = rows.copy()
    rows['n_model_tokens'] = [len(model.tokenizer(text, add_special_tokens=True, truncation=False)['input_ids']) for text in texts]
    rows['exceeds_model_limit'] = rows.n_model_tokens > model.max_seq_length
    if rows.exceeds_model_limit.any():
        raise ValueError('At least one utterance exceeds the model token limit; no silent truncation is allowed.')
    matrix = np.asarray(model.encode(texts, batch_size=config.batch_size, show_progress_bar=True,
                                     convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
    validate_embeddings(matrix, len(rows))
    np.save(matrix_path, matrix)
    write_json(manifest_path, {'signature': signature, 'matrix_sha256': sha256(matrix_path),
                              'n_model_tokens': rows.n_model_tokens.tolist()})
    return rows, matrix


def validate_embeddings(matrix, n):
    if matrix.ndim != 2 or matrix.shape[0] != n or not np.isfinite(matrix).all():
        raise ValueError('Invalid embedding dimensions or values')
    if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5):
        raise ValueError('Expected unit-normalized embeddings')
