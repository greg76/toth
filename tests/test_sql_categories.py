import sqlite3
import unittest

import pandas as pd

import charts
from fetch import init_db


class SqlCategoryTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        init_db(self.db)
        charts.register_sql_functions(self.db)

    def package(self, name, description, kind=0):
        cursor = self.db.execute(
            'INSERT INTO names (kind_id, name, description) VALUES (?, ?, ?)',
            (kind, name, description),
        )
        return cursor.lastrowid

    def members(self, title):
        return {row[0] for row in self.db.execute(
            'SELECT name_id FROM names WHERE ' + charts.CATEGORY_FILTERS[title]
        )}

    def test_regex_lookarounds_boundaries_case_and_nulls(self):
        agent = self.package('agent', 'AI Coding Assistant')
        self.package('pet', 'Menu bar coding assistant')
        self.package('substring', 'Decoding assistant')
        self.package('missing', None)
        self.package('empty', '')
        self.assertEqual(self.members('Coding Agents & Harnesses'), {agent})

    def test_curated_names_match_without_metadata_and_exclusions_apply_to_both_branches(self):
        shells = {self.package(name, None) for name in
                  ('nushell', 'ksh93', 'elvish', 'powershell')}
        shells.add(self.package('bash', 'Bourne-Again SHell, a UNIX shell'))
        self.assertEqual(self.members('Command-line shells'), shells)

        runtime = self.package('runtime', 'Agent runtime')
        assistant = self.package('assistant', 'AI assistant')
        self.package('google-gemini', 'Agent runtime')
        self.package('coding-runtime', 'Agent runtime')
        self.package('code-assistant', 'AI assistant')
        self.package('scanner', 'AI assistant scanner')
        self.assertEqual(self.members('General-purpose Agents & Harnesses'), {runtime, assistant})

    def test_category_join_preserves_kind_tap_and_version_identity(self):
        selected = {
            self.package('same', 'Code editor'),
            self.package('tool@1', 'Code editor'),
            self.package('owner/tap/tool', 'Code editor'),
        }
        self.package('same', 'Image editor', kind=1)
        self.package('tool@2', 'Image editor')
        for name_id, in self.db.execute('SELECT name_id FROM names').fetchall():
            self.db.execute(
                'INSERT INTO counts (date, kind_id, name_id, count) '
                'SELECT ?, kind_id, name_id, ? FROM names WHERE name_id = ?',
                ('20261025', 1000, name_id),
            )
        frame = pd.read_sql_query(charts.category_query('Code editors'), self.db)
        ids = {int(value) for cell in frame.package_ids for value in cell.split(',')}
        self.assertEqual(ids, selected)
        self.assertEqual(set(frame.name), {'same', 'tool', 'owner/tap/tool'})

    def test_top_five_use_latest_available_counts_and_preserve_gaps(self):
        def record(name_id, date, count):
            self.db.execute(
                'INSERT INTO counts (date, kind_id, name_id, count) VALUES (?, 0, ?, ?)',
                (date, name_id, count),
            )

        for index in range(4):
            leader = self.package(f'leader-{index}', 'Code editor')
            record(leader, '20261025', 500 + index)

        # Versions merge before ranking, even when absent in the latest snapshot.
        for name in ('intermittent@1', 'intermittent@2'):
            package = self.package(name, 'Code editor')
            record(package, '20261018', 110)

        low = self.package('low', 'Code editor')
        record(low, '20261025', 200)
        faded = self.package('faded', 'Code editor')
        record(faded, '20261018', 9000)
        record(faded, '20261025', 1)

        frame = pd.read_sql_query(charts.category_query('Code editors'), self.db)
        self.assertEqual(set(frame.name), {
            'leader-0', 'leader-1', 'leader-2', 'leader-3', 'intermittent',
        })
        intermittent = frame.loc[frame.name == 'intermittent']
        self.assertEqual(intermittent['count'].tolist(), [220])
        chart = charts.chartjs_record(charts.ChartData(
            title='Code editors', chart_type=charts.ChartType.LINE, df=frame,
        ))
        dataset = next(item for item in chart['config']['data']['datasets']
                       if item['label'] == 'intermittent')
        self.assertEqual(dataset['data'], [220, None])

    def test_linter_exclusions_apply_to_every_alternative(self):
        analyzer = self.package('analyzer', 'Static code analyzer')
        linter = self.package('linter', 'Python linter')
        self.package('roapi', 'Static code analyzer')
        self.package('memoryanalyzer', 'Python linter')
        self.package('prose', 'Prose linter')
        self.assertEqual(self.members('Code analyzers and linters'), {analyzer, linter})

if __name__ == '__main__':
    unittest.main()
