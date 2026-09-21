import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
