"""Cross-encoder reranking for dense-retrieval candidates.

The Contriever retriever supplies a broad candidate set. This module scores
(query, document) pairs and returns candidates in descending relevance order.
The original dense score is preserved as ``score`` and the cross-encoder score
is added as ``reranker_score``.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class CrossEncoderReranker:
    """Rerank retrieved chunks with a Hugging Face cross-encoder.

    The lock protects model inference because the main experiment evaluates
    questions concurrently using ThreadPoolExecutor. Tokenization and model
    inference are kept inside the lock for predictable behavior on MPS/CPU.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 8,
        max_length: int = 512,
        device: Optional[torch.device] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or self._default_device()
        self._inference_lock = threading.Lock()

        print(f"Loading cross-encoder reranker: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Reranker device: {self.device}")

    @staticmethod
    def _default_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def rerank(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Return copied candidates sorted by cross-encoder score."""
        if not candidates:
            return []

        documents = [
            f"{candidate.get('title', '')}\n{candidate.get('text', '')}".strip()
            for candidate in candidates
        ]
        scores: List[float] = []

        with self._inference_lock:
            for start in range(0, len(documents), self.batch_size):
                document_batch = documents[start:start + self.batch_size]
                encoded = self.tokenizer(
                    [query] * len(document_batch),
                    document_batch,
                    padding=True,
                    truncation="only_second",
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {
                    name: tensor.to(self.device)
                    for name, tensor in encoded.items()
                }

                with torch.inference_mode():
                    logits = self.model(**encoded).logits

                # ms-marco-MiniLM-L-6-v2 returns one relevance logit per pair.
                batch_scores = logits.reshape(-1).float().cpu().tolist()
                scores.extend(float(score) for score in batch_scores)

        # Python's sort is stable, preserving dense order for tied logits.
        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            {**candidate, "reranker_score": float(score)}
            for candidate, score in ranked
        ]
