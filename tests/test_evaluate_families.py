import tempfile
import unittest
from pathlib import Path

import numpy as np

from clustering_experiment import write_csv
from evaluate_families import (
    choose_threshold, family_id, family_map, family_neighbors,
    inherited_judgments, pair_quality, reference_checks, select_model,
)


class FamilyTests(unittest.TestCase):
    def test_versions_collapse_without_merging_roles_or_unresolved_taps(self):
        formula = {'id': 'formula:postgresql@14', 'name': 'postgresql@14', 'package_type': 'formula'}
        cask = {'id': 'cask:postgresql@14', 'name': 'postgresql@14', 'package_type': 'cask'}
        unknown = {'id': 'formula:some/tap/postgresql@14', 'name': 'some/tap/postgresql@14', 'package_type': 'formula'}
        self.assertEqual(family_id(formula, {}), 'formula:postgresql')
        self.assertNotEqual(family_id(formula, {}), family_id(cask, {}))
        self.assertEqual(family_id(unknown, {}), 'formula:some/tap/postgresql')
        alias = {formula['id']: {'canonical_name': 'postgresql@14'}}
        self.assertEqual(family_id(formula, alias), 'formula:postgresql')

    def test_neighbors_use_distinct_families_and_exclude_anchor_versions(self):
        packages = [{'id': f'formula:{name}', 'name': name, 'package_type': 'formula'}
                    for name in ['a@1', 'a@2', 'b@1', 'b@2', 'c']]
        vectors = np.array([[1,0], [.99,.01], [.9,.1], [.95,.05], [0,1]])
        rows = family_neighbors(vectors, ['formula:a@1'], packages, family_map(packages, {}))
        self.assertEqual([r['neighbor_family'] for r in rows], ['formula:b', 'formula:c'])
        self.assertEqual(rows[0]['neighbor_id'], 'formula:b@2')

    def test_uncertainty_and_user_priority_survive_family_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            packages = [{'id': f'formula:{name}', 'name': name, 'package_type': 'formula'} for name in ['a', 'b@1', 'b@2', 'c']]
            fields = ['anchor_id', 'neighbor_id', 'related_functionality', 'reviewer', 'notes']
            write_csv(output/'comparison_judgments.csv', [dict(zip(fields,['formula:a','formula:b@1','yes','assistant','']))], fields)
            write_csv(output/'model_judgments.csv', [], fields)
            user_fields = fields + ['split']
            write_csv(output/'neighbor_judgments.csv', [
                dict(zip(user_fields,['formula:a','formula:b@1','yes','user','','development'])),
                dict(zip(user_fields,['formula:a','formula:b@2','uncertain','user','','development'])),
                dict(zip(user_fields,['formula:c','formula:a','no','user','','holdout'])),
            ], user_fields)
            anchors, labels = inherited_judgments(output, packages, {})
            self.assertEqual(anchors, ['formula:a'])
            self.assertEqual(labels[('formula:a','formula:b')]['related_functionality'], 'uncertain')
            self.assertEqual(labels[('formula:a','formula:b')]['reviewer'], 'user')

    def test_unassigned_reference_pairs_are_not_false_successes(self):
        refs = [{'left':'a','right':'b','related':'no'}]
        self.assertEqual(reference_checks(np.array([-1,-1]), ['a','b'], refs)[0]['status'], 'unassigned')
        self.assertEqual(reference_checks(np.array([0,1]), ['a','b'], refs)[0]['status'], 'pass')
        self.assertEqual(reference_checks(np.array([0,0]), ['a','b'], refs)[0]['status'], 'fail')

    def test_unassigned_pairs_do_not_count_as_coassigned(self):
        reviews = [{'anchor_family':'a','neighbor_family':'b','related_functionality':'yes'},
                   {'anchor_family':'a','neighbor_family':'c','related_functionality':'no'}]
        quality = pair_quality(np.array([-1,-1,-1]), ['a','b','c'], reviews)
        self.assertEqual(quality['coassigned_yes'],0)
        self.assertEqual(quality['coassigned_no'],0)
        self.assertIsNone(quality['decisive_pair_precision'])
        self.assertEqual(quality['known_positive_recall'],0)
        quality = pair_quality(np.array([0,0,0]), ['a','b','c'], reviews)
        self.assertEqual(quality['decisive_pair_precision'],.5)
        self.assertEqual(quality['known_positive_recall'],1)


class SelectiveTests(unittest.TestCase):
    def test_threshold_requires_support_and_can_abstain_entirely(self):
        rows = [{'anchor_id':'a','neighbor_family':str(i),'similarity':1-i/10} for i in range(6)]
        labels = {('a',str(i)):{'related_functionality':'no'} for i in range(6)}
        self.assertIsNone(choose_threshold(rows,labels))
        labels[('a','0')]['related_functionality'] = 'yes'
        self.assertIsNone(choose_threshold(rows,labels))
        for i in range(5):labels[('a',str(i))]['related_functionality']='yes'
        self.assertEqual(choose_threshold(rows,labels),.5)

    def test_model_selection_prioritizes_supported_selective_matches(self):
        def scores(yes,no,top):
            return {'selective':{'yes':yes,'no':no,'uncertain':0,'decisive_relevance':yes/(yes+no)},'top_five':{'yes':top}}
        metrics={'minilm':scores(5,0,20),'mpnet':scores(9,1,21),'qwen':scores(8,2,30)}
        self.assertEqual(select_model(metrics)[0],'mpnet')


if __name__ == '__main__':
    unittest.main()
