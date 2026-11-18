"""Tests for the Couchbase Python SDK search path, using fake adapters for
the OpenAI embedding client and the Couchbase scope."""

import unittest
from types import SimpleNamespace

from hybrid_search import (
    DEFAULT_EMBEDDING_MODEL,
    create_filter,
    generate_embeddings,
    search_couchbase,
)


class FakeEmbeddingClient:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []
        self.embeddings = self

    def create(self, input, model):
        self.calls.append({"input": input, "model": model})
        return SimpleNamespace(data=[SimpleNamespace(embedding=self.vector)])


class FakeScope:
    def __init__(self, rows):
        self.rows_to_return = rows
        self.calls = []

    def search(self, index_name, search_request, options):
        self.calls.append(
            {"index_name": index_name, "request": search_request, "options": options}
        )
        return SimpleNamespace(rows=lambda: iter(self.rows_to_return))


class SearchCouchbaseTests(unittest.TestCase):
    def test_generate_embeddings_defaults_to_default_model(self):
        client = FakeEmbeddingClient([0.1, 0.2])

        self.assertEqual(generate_embeddings(client, "space"), [0.1, 0.2])
        self.assertEqual(client.calls[0]["model"], DEFAULT_EMBEDDING_MODEL)

    def test_search_couchbase_builds_vector_query_and_returns_fields_with_score(self):
        client = FakeEmbeddingClient([0.1, 0.2, 0.3])
        rows = [
            SimpleNamespace(fields={"Series_Title": "Interstellar"}, score=0.91),
            SimpleNamespace(fields={"Series_Title": "Gravity"}, score=0.82),
        ]
        scope = FakeScope(rows)
        prefilter, _ = create_filter((2000, 2024), 7.5)

        results = search_couchbase(
            scope,
            "movies-search-demo",
            client,
            "Overview_embedding",
            "astronauts lost in space",
            k=2,
            search_filters=prefilter,
            embedding_model="text-embedding-3-large",
        )

        self.assertEqual(
            client.calls,
            [{"input": "astronauts lost in space", "model": "text-embedding-3-large"}],
        )
        self.assertEqual(len(scope.calls), 1)
        call = scope.calls[0]
        self.assertEqual(call["index_name"], "movies-search-demo")
        self.assertEqual(dict(call["options"]), {"limit": 2, "fields": ["*"]})

        (vector_query,) = call["request"].vector_search.queries
        self.assertEqual(vector_query.field_name, "Overview_embedding")
        self.assertEqual(vector_query.vector, [0.1, 0.2, 0.3])
        self.assertEqual(vector_query.num_candidates, 2)
        self.assertIs(vector_query.prefilter, prefilter)

        self.assertEqual(
            results,
            [({"Series_Title": "Interstellar"}, 0.91), ({"Series_Title": "Gravity"}, 0.82)],
        )

    def test_search_couchbase_without_filters_has_no_prefilter(self):
        scope = FakeScope([])

        results = search_couchbase(
            scope, "idx", FakeEmbeddingClient([0.5]), "Overview_embedding", "heist"
        )

        self.assertEqual(results, [])
        (vector_query,) = scope.calls[0]["request"].vector_search.queries
        self.assertIsNone(vector_query.prefilter)
        self.assertEqual(vector_query.num_candidates, 5)


if __name__ == "__main__":
    unittest.main()
