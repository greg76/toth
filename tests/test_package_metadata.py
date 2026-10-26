import json
import sqlite3
import unittest
from unittest.mock import patch

import pandas as pd

import charts
from fetch import init_db, load_name_map, refresh_metadata, store_records, update_metadata

STAMP = '2026-10-20T12:00:00Z'


def formula(name, **kwargs):
    return {'name': name, 'desc': 'Code editor', 'homepage': f'https://example.com/{name}', **kwargs}


def cask(name, **kwargs):
    return {'token': name, 'name': ['Example App'], 'desc': 'An application',
            'homepage': f'https://example.org/{name}', **kwargs}


class PackageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        init_db(self.db)

    def tearDown(self):
        self.db.close()

    def seed(self, records):
        return store_records(self.db, '20261018', records, load_name_map(self.db))

    def test_migration_preserves_existing_ids_counts_and_is_repeatable(self):
        legacy = sqlite3.connect(':memory:')
        self.addCleanup(legacy.close)
        legacy.executescript('''
            CREATE TABLE names (name_id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind_id INTEGER NOT NULL, name TEXT NOT NULL, UNIQUE(kind_id,name));
            INSERT INTO names VALUES (42,0,'git');
            CREATE TABLE counts (date TEXT, kind_id INTEGER, name_id INTEGER,
                count INTEGER, UNIQUE(date,kind_id,name_id));
            INSERT INTO counts VALUES ('20261018',0,42,1234);
        ''')
        init_db(legacy)
        init_db(legacy)
        self.assertEqual(legacy.execute('SELECT name_id,name,description FROM names').fetchall(), [(42, 'git', None)])
        self.assertEqual(legacy.execute('SELECT * FROM counts').fetchall(), [('20261018', 0, 42, 1234)])
        self.assertEqual(len(legacy.execute('PRAGMA table_info(names)').fetchall()), 7)

    def test_kind_alias_tap_matching_and_ambiguous_or_absent_metadata(self):
        self.seed([(0, 'same', 500), (1, 'same', 500), (0, 'alias', 500),
                   (0, 'owner/tap/tool', 500), (0, 'other/tap/tool', 500),
                   (0, 'shared', 500), (0, 'missing', 500)])
        sources = {0: [formula('same', aliases=['alias']), formula('tool', tap='owner/tap'),
                       formula('first', aliases=['shared']), formula('second', aliases=['shared'])],
                   1: [cask('same')]}
        self.assertEqual(update_metadata(self.db, sources, STAMP), 4)
        rows = {(kind, name): (display, homepage, stamp) for kind, name, display, homepage, stamp in self.db.execute(
            'SELECT kind_id,name,display_name,homepage,metadata_refreshed_at FROM names')}
        self.assertEqual(rows[(0, 'same')], ('same', 'https://example.com/same', STAMP))
        self.assertEqual(rows[(1, 'same')], ('Example App', 'https://example.org/same', STAMP))
        self.assertEqual(rows[(0, 'alias')][1], 'https://example.com/same')
        self.assertEqual(rows[(0, 'owner/tap/tool')][1], 'https://example.com/tool')
        for name in ('other/tap/tool', 'shared', 'missing'):
            self.assertEqual(rows[(0, name)], (None, None, None))

    def test_failed_second_source_and_invalid_response_leave_metadata_unchanged(self):
        self.seed([(0, 'same', 500), (1, 'same', 500)])
        update_metadata(self.db, {0: [formula('same')], 1: [cask('same')]}, STAMP)
        before = self.db.execute('SELECT * FROM names').fetchall()
        with patch('fetch.fetch_json', side_effect=[json.dumps([formula('same', desc='Changed')]), OSError('offline')]):
            with self.assertRaises(OSError):
                refresh_metadata(self.db)
        self.assertEqual(self.db.execute('SELECT * FROM names').fetchall(), before)
        for bad in ([], {}, [{'token': 'same', 'name': [123]}]):
            with self.assertRaises(ValueError):
                update_metadata(self.db, {0: [formula('same', desc='Changed')], 1: bad}, 'new')
            self.assertEqual(self.db.execute('SELECT * FROM names').fetchall(), before)

    def test_repeat_updates_replace_metadata_without_duplicate_names_or_counts(self):
        self.seed([(0, 'same', 500)])
        self.seed([(0, 'same', 600)])
        update_metadata(self.db, {0: [formula('same')], 1: [cask('app')]}, STAMP)
        update_metadata(self.db, {0: [formula('same', desc='New description', homepage=None)],
                                  1: [cask('app')]}, '2026-10-26T12:00:00Z')
        self.assertEqual(self.db.execute('SELECT count FROM counts').fetchall(), [(600,)])
        self.assertEqual(self.db.execute('SELECT name,description,homepage,metadata_refreshed_at FROM names').fetchall(),
                         [('same', 'New description', None, '2026-10-26T12:00:00Z')])

    def test_export_keeps_kinds_taps_and_versions_attached_to_correct_series(self):
        self.seed([(0, 'same', 700), (1, 'same', 600), (0, 'tool@1', 550),
                   (0, 'tool@2', 650), (0, 'owner/tap/tool', 900)])
        update_metadata(self.db, {0: [formula('same'), formula('tool@1'), formula('tool@2'),
                                     formula('tool', tap='owner/tap')], 1: [cask('same')]}, STAMP)
        with patch.object(charts, 'conn', self.db):
            df = pd.read_sql_query(charts.QUERY_TEMPLATE.format('1'), self.db)
            self.assertEqual(dict(zip(df.name, df['count'])), {
                'same (formula)': 700, 'same (cask)': 600, 'tool': 1200, 'owner/tap/tool': 900})
            chart = charts.ChartData('Example', charts.ChartType.LINE, df)
            record = charts.chartjs_record(chart)
            packages = record['packages']
            self.assertEqual(packages['same (formula)'][0]['kind'], 'formula')
            self.assertEqual(packages['same (cask)'][0]['homepage'], 'https://example.org/same')
            self.assertEqual({row['name'] for row in packages['tool']}, {'tool@1', 'tool@2'})
            self.assertEqual(set(charts.search_package_ids('Code editor')), {row[0] for row in self.db.execute('SELECT name_id FROM names WHERE kind_id=0')})
            for query in (charts.QUERY_GAINERS, charts.QUERY_NEW_ENTRIES):
                frame = pd.read_sql_query(query, self.db)
                labels = list(frame.name)
                self.assertEqual(len(labels), len(set(labels)))

    def test_empty_description_category_can_be_exported(self):
        df = pd.DataFrame(columns=['date', 'name', 'count', 'package_ids'])
        with patch.object(charts, 'conn', self.db):
            record = charts.chartjs_record(charts.ChartData('Empty', charts.ChartType.LINE, df))
        self.assertEqual(record['config']['data'], {'labels': [], 'datasets': []})
        self.assertEqual(record['packages'], {})


if __name__ == '__main__':
    unittest.main()
