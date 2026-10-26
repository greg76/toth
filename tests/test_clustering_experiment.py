import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from clustering_experiment import (
    metadata_index, readme_prose, readme_result, repository_candidate, save_readme_review,
)


class MetadataTests(unittest.TestCase):
    def test_package_types_and_qualified_names_remain_distinct(self):
        formula = {"name": "same", "tap": "homebrew/core", "aliases": ["alias"]}
        cask = {"token": "same", "name": ["Same App"], "tap": "homebrew/cask"}
        index = metadata_index({"formula": [formula], "cask": [cask]})
        self.assertEqual(index[("formula", "same")], [formula])
        self.assertEqual(index[("cask", "same")], [cask])
        self.assertEqual(index[("formula", "homebrew/core/same")], [formula])
        self.assertEqual(index[("formula", "alias")], [formula])
        self.assertEqual(index.get(("formula", "other/tap/same"), []), [])

    def test_alias_collisions_are_retained_for_audit(self):
        first = {"name": "one", "aliases": ["shared"]}
        second = {"name": "two", "aliases": ["shared"]}
        self.assertEqual(metadata_index({"formula": [first, second]})[("formula", "shared")], [first, second])

    def test_consistent_repository_signals(self):
        candidate = repository_candidate({
            "homepage": "https://github.com/Owner/Project",
            "urls": {"stable": {"url": "https://github.com/owner/project/archive/v1.tar.gz"},
                     "head": {"url": "https://github.com/Owner/Project.git"}},
        })
        self.assertEqual(candidate["repository_status"], "candidate")
        self.assertEqual(candidate["repository"], "Owner/Project")

    def test_conflicting_repository_signals_require_review(self):
        candidate = repository_candidate({
            "homepage": "https://github.com/org/one",
            "url": "https://github.com/org/two/releases/download/v1/app.zip",
        })
        self.assertEqual(candidate["repository_status"], "conflicting_urls")
        self.assertIsNone(candidate["repository"])

    def test_subdirectories_and_non_github_sources(self):
        self.assertEqual(repository_candidate({"homepage": "https://github.com/org/repo/tree/main/tool"})["repository_status"], "subdirectory_review")
        self.assertEqual(repository_candidate({"homepage": "https://www.gnu.org/software/wget/"})["repository_status"], "unresolved")
        self.assertEqual(repository_candidate({"url": "https://github.com/org/homebrew-tap/archive/v1.tar.gz"})["repository_status"], "unresolved")


class ReadmeTests(unittest.TestCase):
    def test_boilerplate_and_nested_sections_are_removed(self):
        text = """# Project
A [database](https://example.com) for analytics.
![build](https://shields.io/build)
## Installation
brew install project
### Linux
apt install project
## Features
- Fast queries
```sh
bad command
```
## License
MIT
"""
        prose = readme_prose(text)
        self.assertIn("database for analytics", prose)
        self.assertIn("Fast queries", prose)
        for excluded in ["brew install", "apt install", "bad command", "shields", "MIT"]:
            self.assertNotIn(excluded, prose)

    def test_html_comments_and_code_are_removed(self):
        prose = readme_prose("<!-- hidden -->\n<pre>build commands</pre>\n<p>A search tool &amp; indexer.</p>")
        self.assertEqual(prose, "A search tool & indexer.")

    def test_script_closing_tag_variants_are_removed(self):
        for closing in ["</script>", "</script >", "</ScRiPt\n>", '</script ignored="value">']:
            with self.subTest(closing=closing):
                prose = readme_prose(f"<SCRIPT>alert(1){closing}\nA search tool.")
                self.assertEqual(prose, "A search tool.")

    def test_pre_contents_and_nested_markup_are_removed(self):
        prose = readme_prose('<pre><code>build commands</code><pre>nested</pre></pre >\nA search tool.')
        self.assertEqual(prose, "A search tool.")

    def test_html_attributes_and_alternative_comment_endings(self):
        prose = readme_prose('<!-- hidden --!><p title="a > b">A search tool.</p>')
        self.assertEqual(prose, "A search tool.")

    def test_unclosed_script_contents_are_excluded(self):
        self.assertEqual(readme_prose("A search tool.\n<script>alert(1)"), "A search tool.")

    def test_html_entities_are_decoded_once(self):
        self.assertEqual(readme_prose("A tool &amp;amp; &#38; &#x26; indexer."),
                         "A tool &amp; & & indexer.")

    def test_alternative_markdown_headings(self):
        prose = readme_prose("Project\n=======\nA search tool.\nInstallation\n------------\nbrew install project\nFeatures\n--------\nFast queries.")
        self.assertIn("Fast queries", prose)
        self.assertNotIn("brew install", prose)

    def test_project_titles_and_introductory_links_do_not_skip_overview(self):
        prose = readme_prose("# Please refer to documentation for installation\nA Kubernetes tool.\n## Features\nLocal clusters.")
        self.assertIn("A Kubernetes tool", prose)
        prose = readme_prose("sccache - Shared Compilation Cache\n======\nCaches compiler output.")
        self.assertIn("Caches compiler output", prose)
        prose = readme_prose("libass [![Build status](https://badge)](https://status)\n======\nA subtitle renderer.")
        self.assertIn("A subtitle renderer", prose)
        self.assertNotIn("](", prose)

    def test_multiline_html_and_table_of_contents(self):
        prose = readme_prose('<img\nsrc="image.png"\n/>\n# Project\n## Table of Contents\n- Install\n- Usage\n## Overview\nA package manager.')
        self.assertNotIn("image.png", prose)
        self.assertNotIn("Install", prose)
        self.assertIn("A package manager", prose)

    def test_readme_is_cached_by_repository_with_source_revision(self):
        data = {"encoding": "base64", "size": 10, "content": base64.b64encode(b"# A tool").decode(),
                "sha": "blob-revision", "path": "README.md", "html_url": "https://github.com/org/repo/blob/main/README.md"}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            with patch("clustering_experiment.request_json", return_value=data) as fetch:
                first = readme_result("Org/Repo", cache)
                second = readme_result("org/repo", cache)
            self.assertEqual(first, second)
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(first["blob_sha"], "blob-revision")
            self.assertEqual(first["markdown"], "# A tool")

    def test_regenerated_reviews_preserve_decisions_only_for_unchanged_text(self):
        import csv

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.csv"
            result = {"id": "formula:one", "repository": "org/one", "blob_sha": "sha",
                      "excerpt": "A search tool", "review": "reviewed", "usable": "yes", "notes": "Good"}
            save_readme_review(path, [result])
            pending = {**result, "review": "pending", "usable": "", "notes": ""}
            save_readme_review(path, [pending])
            with path.open() as stream:
                self.assertEqual(next(csv.DictReader(stream))["usable"], "yes")
            save_readme_review(path, [{**pending, "excerpt": "Different text"}])
            with path.open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["usable"], "")
            self.assertEqual(row["review"], "pending")
            self.assertIn("Previous decision: yes", row["notes"])


if __name__ == "__main__":
    unittest.main()
