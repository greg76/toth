import unittest

from compare_embeddings import review_excerpts, summarize


class ComparisonTests(unittest.TestCase):
    def test_uncertain_and_unknown_are_not_negative_labels(self):
        rows = [{"anchor_id": "a", "neighbor_id": neighbor} for neighbor in ["b", "c", "d", "e"]]
        judgments = {("a", "b"): {"related_functionality": "yes"},
                     ("a", "c"): {"related_functionality": "no"},
                     ("a", "d"): {"related_functionality": "uncertain"}}
        score = summarize(rows, judgments)
        self.assertEqual(score["decisive_relevance"], 0.5)
        self.assertEqual(score["uncertain"], 1)
        self.assertEqual(score["unjudged"], 1)
        self.assertEqual(score["precision_lower"], 0.25)
        self.assertEqual(score["precision_upper"], 0.75)

    def test_invalid_judgments_fail_instead_of_being_silently_ignored(self):
        with self.assertRaises(ValueError):
            summarize([{"anchor_id": "a", "neighbor_id": "b"}],
                      {("a", "b"): {"related_functionality": "maybe"}})

    def test_pending_and_rejected_readmes_are_not_embedded(self):
        rows = [{"id": "one", "usable": "yes", "review": "pending"},
                {"id": "two", "usable": "no", "review": "assistant"},
                {"id": "three", "usable": "yes", "review": ""}]
        self.assertEqual(review_excerpts(rows), {})

    def test_reviewed_text_must_come_from_the_downloaded_excerpt(self):
        row = {"id": "one", "usable": "yes", "review": "assistant", "excerpt": "A search tool.",
               "reviewed_excerpt": "A database."}
        with self.assertRaises(ValueError):
            review_excerpts([row])
        row["reviewed_excerpt"] = "search tool"
        self.assertEqual(review_excerpts([row]), {"one": "search tool"})


if __name__ == "__main__":
    unittest.main()
