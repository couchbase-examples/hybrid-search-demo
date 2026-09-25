import unittest
from pathlib import Path
import re

import pandas as pd

from hybrid_search import cleanup_poster_url, create_filter


class HybridSearchHelperTests(unittest.TestCase):
    def test_cleanup_poster_url_restores_full_size_image_url(self):
        thumbnail_url = (
            "https://m.media-amazon.com/images/M/example@._V1_"
            "UX67_CR0,0,67,98_AL_.jpg"
        )

        self.assertEqual(
            cleanup_poster_url(thumbnail_url),
            "https://m.media-amazon.com/images/M/example@..jpg",
        )

    def test_create_filter_describes_year_and_rating_prefilter(self):
        _, description = create_filter((2000, 2024), 7.5)

        self.assertIn("ConjunctionQuery", description)
        self.assertIn("Released_Year", description)
        self.assertIn("min=2000", description)
        self.assertIn("max=2024", description)
        self.assertIn("IMDB_Rating", description)
        self.assertIn("min=7.5", description)

    def test_create_filter_allows_year_only_prefilter(self):
        _, description = create_filter((1990, 1999), 0.0)

        self.assertNotIn("ConjunctionQuery", description)
        self.assertIn("Released_Year", description)
        self.assertIn("min=1990", description)
        self.assertIn("max=1999", description)

    def test_included_imdb_dataset_has_required_columns(self):
        data = pd.read_csv(Path(__file__).resolve().parents[1] / "imdb_top_1000.csv")

        self.assertEqual(len(data), 1000)
        self.assertTrue(
            {
                "Series_Title",
                "Overview",
                "Poster_Link",
                "Released_Year",
                "IMDB_Rating",
                "Runtime",
            }.issubset(data.columns)
        )

    def test_sample_configuration_files_document_required_runtime_keys(self):
        repo_root = Path(__file__).resolve().parents[1]
        required_keys = {
            "OPENAI_API_KEY",
            "DB_CONN_STR",
            "DB_USERNAME",
            "DB_PASSWORD",
            "DB_BUCKET",
            "DB_SCOPE",
            "DB_COLLECTION",
            "INDEX_NAME",
            "EMBEDDING_MODEL",
        }

        env_example_keys = _keys_from_assignment_file(repo_root / ".env.example")
        streamlit_example_keys = _keys_from_assignment_file(
            repo_root / ".streamlit" / "secrets.example.toml"
        )

        self.assertTrue(required_keys.issubset(env_example_keys))
        self.assertTrue(required_keys.issubset(streamlit_example_keys))


def _keys_from_assignment_file(path):
    key_pattern = re.compile(r"^([A-Z0-9_]+)\s*=")
    keys = set()
    for line in path.read_text().splitlines():
        match = key_pattern.match(line.strip())
        if match:
            keys.add(match.group(1))
    return keys


if __name__ == "__main__":
    unittest.main()
