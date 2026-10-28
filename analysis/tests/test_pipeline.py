"""Fast behavioral tests; full notebook parity is checked by analysis.verify."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from analysis.config import Config, load_config
from analysis.metrics import calculate_conversation_geometry, calculate_speaker_centroid_distance
from analysis.transcripts import create_utterances_table
from analysis.provenance import sha256, write_json, check_metric_inputs


class PipelineTests(unittest.TestCase):
    def test_config_is_strict(self):
        for kwargs in [dict(centroid_min_words=-1), dict(bootstrap_replicates=0),
                       dict(seed=True), dict(local_files_only='false'), dict(alpha=2),
                       dict(output_dir='data/wired')]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Config(**kwargs)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bad.toml';path.write_text('bootstrap_replicate = 10')
            with self.assertRaises(TypeError):load_config(path)

    def test_selection_config_changes_centroids_not_transcript_order(self):
        raw = pd.DataFrame(dict(dataset=['wired']*4, conversation_id=['test']*4,
                                sequence=[3, 1, 4, 2], speaker=['A','A','B','B'],
                                speaker_role=['expert','expert','partner','partner'],
                                text=['These are five or more words.', 'Yes.',
                                      'These also have five or more words.', 'Why?']))
        all_rows = create_utterances_table(raw, 0)
        filtered_rows = create_utterances_table(raw, 5)
        self.assertEqual(all_rows.text.iloc[0], 'Yes.')
        self.assertTrue(all_rows.include_geometry_primary.all())
        self.assertEqual(filtered_rows.include_geometry_primary.sum(), 2)
        pd.testing.assert_frame_equal(all_rows.drop(columns='include_geometry_primary'), filtered_rows.drop(columns='include_geometry_primary'))
        all_rows['_centroid_role'] = all_rows.speaker_role
        x = np.array([[1.,0.],[0.,1.],[0.,1.],[0.,1.]])
        result = calculate_speaker_centroid_distance(all_rows, x)
        self.assertAlmostEqual(result['speaker_centroid_cosine_distance'], 1 - 1/np.sqrt(2))
        result = calculate_speaker_centroid_distance(all_rows.loc[filtered_rows.include_geometry_primary], x)
        self.assertAlmostEqual(result['speaker_centroid_cosine_distance'], 0)

    def test_known_geometry_and_undefined_direction(self):
        result, _ = calculate_conversation_geometry(np.eye(2))
        self.assertAlmostEqual(result['radial_distance_median'], 1 - 1/np.sqrt(2))
        self.assertAlmostEqual(result['participation_ratio'], 1)
        result, _ = calculate_conversation_geometry(np.array([[1.,0.],[-1.,0.]]))
        self.assertTrue(np.isnan(result['radial_distance_median']))

    def test_stale_metric_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(output_dir=d)
            write_json(Path(d)/'tables/metrics_manifest.json', {'centroid_selection': 'all nonempty utterances'})
            with self.assertRaisesRegex(ValueError, 'different centroid selection'):
                check_metric_inputs(cfg)

    def test_changed_text_rejects_embedding_cache(self):
        # Matching shape is insufficient: text/ID fingerprint and matrix hash are required.
        from analysis.embeddings import embed_utterances
        # Run a valid cache hit using the actual fingerprint construction, without a model.
        from analysis.provenance import packages
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(output_dir=d)
            rows = pd.DataFrame({'text':['Yes.'], 'utterance_id':['test::1']})
            folder=Path(d)/'embeddings';folder.mkdir()
            np.save(folder/'utterance_embeddings.npy',np.array([[1.,0.]],dtype=np.float32))
            signature={'texts_sha256':hashlib.sha256(json.dumps(['Yes.'],ensure_ascii=False).encode()).hexdigest(),
                       'ids_sha256':hashlib.sha256(b'test::1').hexdigest(), 'model':cfg.model,'revision':cfg.revision,
                       'seed':cfg.seed,'batch_size':cfg.batch_size,'device':'cpu','packages':packages()}
            write_json(folder/'embedding_manifest.json',{'signature':signature,'matrix_sha256':sha256(folder/'utterance_embeddings.npy'),'n_model_tokens':[4]})
            actual,matrix=embed_utterances(rows,cfg)
            self.assertEqual(actual.n_model_tokens.iloc[0],4)
            self.assertEqual(matrix.shape,(1,2))
            from unittest.mock import patch
            with patch('sentence_transformers.SentenceTransformer', side_effect=RuntimeError('cache invalidated')):
                with self.assertRaisesRegex(RuntimeError,'cache invalidated'):
                    embed_utterances(rows.assign(text='No.'),cfg)


if __name__ == '__main__':unittest.main()
