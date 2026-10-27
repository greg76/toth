import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fetch


class TotalsTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(':memory:')
        self.addCleanup(self.db.close)
        fetch.init_db(self.db)

    def existing_tables(self):
        return {
            table: (
                self.db.execute('SELECT sql FROM sqlite_master WHERE name=?', (table,)).fetchone(),
                self.db.execute(f'SELECT * FROM {table} ORDER BY rowid').fetchall(),
            )
            for table in ('names', 'counts', 'sqlite_sequence')
        }

    def test_totals_only_backfill_preserves_existing_tables_and_is_repeatable(self):
        fetch.store_records(self.db, '20261018', [(0, 'git', 900)], {})
        before = self.existing_tables()
        with tempfile.TemporaryDirectory() as directory:
            dumps = Path(directory)
            for kind, total in zip(fetch.KIND_SPECS, (1234, 567)):
                fetch.write_dump(dumps / f'20261018{kind.name}.zst',
                                 '{"total_count": ' + str(total) + '}')
            fetch.init_totals_db(self.db)
            fetch.run_totals_backfill(self.db, dumps)
            rows = self.db.execute('SELECT * FROM totals').fetchall()
            fetch.run_totals_backfill(self.db, dumps)
        self.assertEqual(rows, [('20261018', 1234, 567)])
        self.assertEqual(rows, self.db.execute('SELECT * FROM totals').fetchall())
        self.assertEqual(before, self.existing_tables())

    def test_fetch_keeps_unfiltered_totals_and_independent_dates(self):
        formula = {'total_count': 1050, 'items': [
            {'formula': 'git', 'count': '1,000'}, {'formula': 'tiny', 'count': '50'}]}
        cask = {'total_count': 40, 'items': [{'cask': 'app', 'count': '40'}]}
        with patch('fetch.get_today_data', side_effect=[('20261018', formula), ('20261019', cask)]):
            fetch.run_fetch(self.db, Path('unused'), 500)
        self.assertEqual(self.db.execute('SELECT * FROM totals ORDER BY date').fetchall(),
                         [('20261018', 1050, None), ('20261019', None, 40)])
        self.assertEqual(self.db.execute('SELECT count FROM counts').fetchall(), [(1000,)])
        with self.db:
            fetch.upsert_total(self.db, '20261018', fetch.KIND_SPECS[1], 60)
            fetch.upsert_total(self.db, '20261018', fetch.KIND_SPECS[0], 1100)
        self.assertEqual(self.db.execute("SELECT * FROM totals WHERE date='20261018'").fetchone(),
                         ('20261018', 1100, 60))

    def test_general_backfill_populates_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            dumps = Path(directory)
            fetch.write_dump(dumps / '20261018formula.zst',
                             '{"total_count": 20, "items": [{"formula": "tiny", "count": "20"}]}')
            fetch.run_backfill(self.db, dumps, 500)
        self.assertEqual(self.db.execute('SELECT * FROM totals').fetchall(), [('20261018', 20, None)])

    def test_invalid_totals_fail_before_package_writes(self):
        before = self.existing_tables()
        for total in (None, -1, True, '100', 1.5):
            with self.subTest(total=total), self.assertRaises(ValueError):
                fetch.store_snapshot(self.db, '20261018', {
                    'total_count': total, 'items': [{'formula': 'git', 'count': '900'}]},
                    fetch.KIND_SPECS[0], 500, {})
        self.assertEqual(before, self.existing_tables())
        self.assertEqual(self.db.execute('SELECT * FROM totals').fetchall(), [])

    def test_snapshot_rolls_back_counts_when_total_write_fails(self):
        self.db.execute("CREATE TRIGGER reject_total BEFORE INSERT ON totals BEGIN SELECT RAISE(ABORT, 'test'); END")
        name_map = {}
        with self.assertRaises(sqlite3.IntegrityError):
            fetch.store_snapshot(self.db, '20261018', {
                'total_count': 900, 'items': [{'formula': 'git', 'count': '900'}]},
                fetch.KIND_SPECS[0], 500, name_map)
        self.assertEqual(self.db.execute('SELECT * FROM names').fetchall(), [])
        self.assertEqual(self.db.execute('SELECT * FROM counts').fetchall(), [])
        self.assertEqual(name_map, {})

if __name__ == '__main__':
    unittest.main()
