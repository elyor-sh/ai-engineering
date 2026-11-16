"""Проверки lookup, косинуса и связи ответа embeddings API с исходными текстами.

Сервисы не вызываются: математические тесты работают без Ollama и API-ключей.
"""

import runpy
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from openai.types import CreateEmbeddingResponse

HERE = Path(__file__).parent


class LookupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "01_embedding_lookup.py"
        assert source.is_file(), "Эксперимент с таблицей ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_ids_select_rows_and_preserve_order(self):
        table = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = self.experiment["lookup_embeddings"](table, [2, 0, 2])
        np.testing.assert_array_equal(result, [[5, 6], [1, 2], [5, 6]])
        self.assertEqual(result.shape, (3, 2))

    def test_empty_sequence_preserves_embedding_dimension(self):
        result = self.experiment["lookup_embeddings"](np.ones((3, 4)), [])
        self.assertEqual(result.shape, (0, 4))

    def test_invalid_ids_are_not_numpy_negative_indices(self):
        table = np.ones((3, 4))
        for ids in [[-1], [3]]:
            with self.subTest(ids=ids), self.assertRaises(IndexError):
                self.experiment["lookup_embeddings"](table, ids)


class SimilarityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = HERE / "02_local_embeddings.py"
        assert source.is_file(), "Эксперимент с эмбеддингами ещё не реализован"
        cls.experiment = runpy.run_path(str(source))

    def test_normalization_returns_unit_vectors_without_changing_input(self):
        matrix = np.array([[3.0, 4.0], [0.0, 2.0]])
        original = matrix.copy()
        normalized = self.experiment["normalize_rows"](matrix)
        np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), [1, 1])
        np.testing.assert_array_equal(matrix, original)

    def test_zero_vector_is_rejected(self):
        with self.assertRaises(ValueError):
            self.experiment["normalize_rows"](np.zeros((1, 3)))

    def test_nonfinite_coordinates_are_rejected(self):
        for value in [np.nan, np.inf]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.experiment["normalize_rows"](np.array([[value, 1.0]]))

    def test_normalization_requires_rows_of_vectors(self):
        with self.assertRaises(ValueError):
            self.experiment["normalize_rows"](np.array([1.0, 2.0]))

    def test_cosine_for_same_orthogonal_and_opposite_directions(self):
        scores = self.experiment["cosine_scores"](np.array([[1, 0], [0, 1], [-1, 0]]), np.array([1, 0]))
        np.testing.assert_allclose(scores, [1, 0, -1])

    def test_positive_scaling_does_not_change_cosine(self):
        vectors = np.array([[1.0, 2.0], [3.0, -1.0]])
        query = np.array([2.0, 1.0])
        scores = self.experiment["cosine_scores"]
        np.testing.assert_allclose(scores(vectors, query), scores(vectors * 7, query * 3))

    def test_query_and_documents_need_matching_dimension(self):
        with self.assertRaises(ValueError):
            self.experiment["cosine_scores"](np.ones((2, 3)), np.ones(4))

    def test_api_vectors_are_sorted_by_input_index(self):
        client = Mock()
        client.embeddings.create.return_value = CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": "test-embedding-model",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
            }
        )
        result = self.experiment["fetch_embeddings"](client, ["первый", "второй"], "test-embedding-model")
        np.testing.assert_allclose(result, [[1, 0], [0, 1]])

    def test_incomplete_api_response_is_not_matched_to_wrong_text(self):
        client = Mock()
        client.embeddings.create.return_value = CreateEmbeddingResponse.model_validate(
            {
                "object": "list",
                "model": "test-embedding-model",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
                "data": [{"object": "embedding", "index": 0, "embedding": [1.0, 0.0]}],
            }
        )
        with self.assertRaises(ValueError):
            self.experiment["fetch_embeddings"](client, ["первый", "второй"], "test-embedding-model")


if __name__ == "__main__":
    unittest.main()
