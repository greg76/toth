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

    def test_gainers_merge_versions_per_snapshot_and_keep_kinds_and_taps_separate(self):
        for name, kind, previous, current in (
            ('tool', 0, 1000, 1100), ('tool@latest', 0, 500, 1000),
            ('tool@new', 0, None, 300), ('tool@old', 0, 200, None),
            ('tool', 1, 500, 1000), ('owner/tap/tool@2', 0, 500, 750),
            ('new@1', 0, None, 900),
        ):
            name_id = self.package(name, None, kind)
            for date, count in (('20261115', previous), ('20261122', current)):
                if count is not None:
                    self.db.execute('INSERT INTO counts VALUES (?, ?, ?, ?)',
                                    (date, kind, name_id, count))
        rows = self.db.execute(charts.QUERY_GAINERS).fetchall()
        growth = {name: pct for name, _, pct in rows}
        self.assertEqual(growth, {'tool (formula)': 41.18, 'tool (cask)': 100.0,
                                  'owner/tap/tool': 50.0})
        ids = next(ids for name, ids, _ in rows if name == 'tool (formula)')
        names = {self.db.execute('SELECT name FROM names WHERE name_id = ?', (int(i),)).fetchone()[0]
                 for i in ids.split(',')}
        self.assertEqual(names, {'tool', 'tool@latest', 'tool@new'})

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

    def test_code_editors_include_text_editors_and_curated_variants(self):
        selected = {
            self.package('fresh-editor', 'Text editor for your terminal'),
            self.package('micro', 'Modern terminal-based text editor'),
            self.package('emacs', 'GNU Emacs text editor'),
            self.package('cursor', 'Write, edit, and chat about your code'),
            *(self.package(name, None) for name in (
                'vim', 'neovim', 'macvim', 'neovide',
                'd12frosted/emacs-plus/emacs-plus@31',
                'd12frosted/emacs-plus/emacs-plus-app',
            )),
        }
        self.package('wispr-flow', 'Voice-to-text dictation with AI-powered auto-editing')
        self.package('image-editor', 'Image editor')
        self.assertEqual(self.members('Code editors'), selected)

    def test_local_container_platforms_include_orbstack_and_exclude_add_ons(self):
        selected = {
            self.package('orbstack', 'Replacement for Docker Desktop', kind=1),
            self.package('colima', 'Container runtimes on macOS and Linux'),
            self.package('podman-desktop', 'Browse, manage, inspect containers', kind=1),
        }
        self.package('docker-compose', 'Isolated development environments using Docker')
        self.package('docker-buildx', 'Docker CLI plugin for extended build capabilities')
        self.package('trivy', 'Vulnerability scanner for container images')
        self.assertEqual(self.members('Local container platforms'), selected)

    def test_terminal_emulators_include_terminal_apps_and_exclude_widgets(self):
        selected = {
            self.package('warp', 'Rust-based terminal', kind=1),
            self.package('cmux', 'Ghostty-based terminal with vertical tabs', kind=1),
            self.package('ghostty', 'Terminal emulator', kind=1),
        }
        self.package('vte3', 'Terminal emulator widget used by GNOME terminal')
        self.package('tmux', 'Terminal multiplexer')
        self.package('fresh-editor', 'Text editor for your terminal')
        self.assertEqual(self.members('Terminal emulators'), selected)

    def test_top_ten_use_latest_available_counts_and_preserve_gaps(self):
        def record(name_id, date, count):
            self.db.execute(
                'INSERT INTO counts (date, kind_id, name_id, count) VALUES (?, 0, ?, ?)',
                (date, name_id, count),
            )

        for index in range(9):
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
            *(f'leader-{index}' for index in range(9)), 'intermittent',
        })
        intermittent = frame.loc[frame.name == 'intermittent']
        self.assertEqual(intermittent['count'].tolist(), [220])
        chart = charts.chartjs_record(charts.ChartData(
            title='Code editors', chart_type=charts.ChartType.LINE, df=frame,
        ))
        dataset = next(item for item in chart['config']['data']['datasets']
                       if item['label'] == 'intermittent')
        self.assertEqual(dataset['data'], [220, None])

    def test_programming_error_checkers_exclude_other_kinds_of_tools(self):
        selected = {
            self.package('future-tool', 'Fast linter for Python'),
            self.package('future-types', 'Type checker for Rust'),
            self.package('shellcheck', 'Static analysis and lint tool for shell scripts'),
            self.package('golangci-lint', 'Fast linters runner for Go'),
            *(self.package(name, None) for name in ('biome', 'eslint', 'flake8')),
        }
        self.package('swiftlint', 'Tool to enforce Swift style and conventions')
        self.package('yamllint', 'Linter for YAML files')
        self.package('trivy', 'Vulnerability scanner')
        self.package('ast-grep', 'Code searching, linting, rewriting')
        self.package('ktlint', 'Kotlin linter with built-in formatter')
        self.assertEqual(self.members('Programming error checkers'), selected)

if __name__ == '__main__':
    unittest.main()
