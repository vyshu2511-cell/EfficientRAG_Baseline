"""Unit tests for CrossEncoderReranker using mocked Hugging Face models."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from reranker import CrossEncoderReranker


class FakeTokenizer:
    score_by_text = {"low": 0.1, "medium": 0.5, "high": 0.9}

    def __call__(self, queries, documents, **kwargs):
        values = [
            self.score_by_text[document.split("\n", 1)[-1]]
            for document in documents
        ]
        return {
            "input_ids": torch.tensor(
                values, dtype=torch.float32
            ).unsqueeze(1)
        }


class FakeModel(torch.nn.Module):
    def forward(self, input_ids):
        return SimpleNamespace(logits=input_ids)


class CrossEncoderRerankerTests(unittest.TestCase):
    def test_reranks_and_preserves_dense_scores(self):
        with patch(
            "reranker.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ), patch(
            "reranker.AutoModelForSequenceClassification.from_pretrained",
            return_value=FakeModel(),
        ):
            reranker = CrossEncoderReranker(
                batch_size=2,
                device=torch.device("cpu"),
            )
            candidates = [
                {"title": "T", "text": "low", "score": 0.99},
                {"title": "T", "text": "high", "score": 0.20},
                {"title": "T", "text": "medium", "score": 0.50},
            ]

            ranked = reranker.rerank("query", candidates)

        self.assertEqual(
            [item["text"] for item in ranked],
            ["high", "medium", "low"],
        )
        self.assertEqual(
            [item["score"] for item in ranked],
            [0.20, 0.50, 0.99],
        )
        self.assertEqual(
            [round(item["reranker_score"], 1) for item in ranked],
            [0.9, 0.5, 0.1],
        )

    def test_empty_candidates(self):
        with patch(
            "reranker.AutoTokenizer.from_pretrained",
            return_value=FakeTokenizer(),
        ), patch(
            "reranker.AutoModelForSequenceClassification.from_pretrained",
            return_value=FakeModel(),
        ):
            reranker = CrossEncoderReranker(device=torch.device("cpu"))
            self.assertEqual(reranker.rerank("query", []), [])


if __name__ == "__main__":
    unittest.main()
