# Checkpoint 2 Implementation Status

This status compares the submitted progress memo with the current repository.

## Implemented

- MultiHopRAG data is present: 2,556 questions and the news corpus.
- Whitespace cleaning and 250-word chunks with 50-word overlap are implemented in `preprocessing.py`.
- Contriever-MSMARCO embeddings are generated and saved as NumPy vectors.
- Single-step dense retrieval and Ollama answer generation are implemented.
- EM and token-level Answer F1 are implemented.
- Batched question embeddings, bounded Ollama concurrency, configurable model settings, and per-question checkpointing are implemented in `run_direct_r10_prompt_v3.py`.

## Newly implemented

- `split_data.py` creates seed-42 development, held-out test, and remaining splits.
- `eval.py` computes EM, Answer F1, retrieval precision, retrieval recall, and MRR from saved results.

## Partial or not implemented

- FAISS `IndexFlatIP`: not implemented; retrieval currently scans the NumPy matrix.
- Iterative query reformulation and second-hop retrieval: not implemented.
- `ms-marco-MiniLM-L-6-v2` cross-encoder reranking: not implemented.
- Faithfulness or entailment filtering: not implemented.
- Shared configuration file for all scripts: not implemented.
- HotpotQA pilot: not present in the repository.
- Memo's reported 200-question held-out metrics cannot be reproduced from current artifacts because the fixed test predictions and multi-hop system are absent.

## Implementation queue

- [x] Add reproducible seed-42 dataset splits.
- [x] Add evaluation metrics for saved baseline results.
- [ ] Add FAISS index with a NumPy fallback.
- [ ] Implement iterative reformulation and second-hop retrieval.
- [ ] Add cross-encoder reranking.
- [ ] Add faithfulness checking.
- [ ] Add shared configuration and reproducible held-out evaluation.
- [ ] Add the 100-question HotpotQA pilot.