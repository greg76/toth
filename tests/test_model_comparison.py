import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from clustering_experiment import read_json, write_csv, write_json
from compare_embeddings import prepare_model


class FakeModel:
    device = 'cpu'

    def encode(self, texts, **kwargs):
        return np.array([[1, 0, 0], [.9, .1, 0], [0, 1, 0]], dtype=float)


class ModelComparisonTests(unittest.TestCase):
    def test_fixed_inputs_user_authority_holdout_exclusion_and_cache_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [{"id": name, "name_description": name + ': a tool'} for name in ['a', 'b', 'c']]
            write_json(root/'comparison_inputs.json', inputs)
            write_json(root/'assignments.json', [{"id": r['id'], "description": 'a tool'} for r in inputs])
            write_json(root/'comparison_manifest.json', {"model_revision": 'mini-revision'})
            write_json(root/'baseline.json', {"min_cluster_size": 5, "min_samples": 2})
            np.savez_compressed(root/'comparison_vectors.npz', name_description=np.array([[1, 0], [.9, .1], [0, 1]]))
            fields = ['split', 'anchor_id', 'neighbor_id', 'related_functionality', 'notes']
            write_csv(root/'neighbor_judgments.csv', [
                dict(zip(fields, ['development', 'a', 'b', 'yes', 'user reason'])),
                dict(zip(fields, ['holdout', 'c', 'b', 'no', 'do not use'])),
            ], fields)
            labels = ['anchor_id', 'neighbor_id', 'related_functionality', 'reviewer', 'notes']
            write_csv(root/'comparison_judgments.csv', [
                dict(zip(labels, ['a', 'b', 'no', 'assistant', 'superseded'])),
                dict(zip(labels, ['a', 'c', 'uncertain', 'assistant', 'keep uncertainty'])),
            ], labels)
            before = (root/'neighbor_judgments.csv').read_bytes()
            with patch('compare_embeddings.load_model', return_value=(FakeModel(), 'mpnet-revision')) as loader, \
                    patch('compare_embeddings.inspect_model_clusters'):
                prepare_model(root)
                prepare_model(root)
            self.assertEqual(loader.call_count, 1)
            self.assertEqual((root/'neighbor_judgments.csv').read_bytes(), before)
            neighbors = read_json(root/'model_neighbors.json')
            self.assertEqual({r['anchor_id'] for r in neighbors}, {'a'})
            with (root/'model_judgments.csv').open() as stream:
                judgments = {(r['anchor_id'],r['neighbor_id']):r for r in csv.DictReader(stream)}
            self.assertEqual(judgments[('a','b')]['related_functionality'], 'yes')
            self.assertEqual(judgments[('a','b')]['reviewer'], 'user')
            self.assertEqual(judgments[('a','b')]['notes'], 'user reason')
            self.assertEqual(judgments[('a','c')]['related_functionality'], 'uncertain')
            self.assertEqual(read_json(root/'model_manifest.json')['model_revision'], 'mpnet-revision')


if __name__ == '__main__':
    unittest.main()
