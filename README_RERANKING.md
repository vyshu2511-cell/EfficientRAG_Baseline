# Cross-encoder reranking integration

Target: `vyshu2511-cell/EfficientRAG_Baseline`, branch `llama_3.2`

Inspected branch head: `05f410e52ca690b01a791566f43fc7b17c736939`

## Files

- `reranker.py`: reusable batched Hugging Face cross-encoder scoring module.
- `run_direct_r10_prompt_v3.py`: updated active runner with opt-in candidate retrieval and reranking.
- `tests/test_reranker.py`: unit tests using mocked tokenizer/model objects (no model download required).

## Retrieval flow

1. Contriever dense retrieval retrieves `candidate-k` chunks (default 50 when reranking is enabled).
2. `cross-encoder/ms-marco-MiniLM-L-6-v2` scores each `(question, title + chunk text)` pair.
3. The highest scoring `top-k` chunks are passed to the existing Llama/Ollama answer generation.

Both scores are retained in each saved result: `score` is the original Contriever dot-product score; `reranker_score` is the cross-encoder logit. The reranker is optional, so a run without `--rerank` retains the existing baseline behavior and output filename. Reranked runs use a separate output filename to avoid accidentally resuming/overwriting baseline results.

## Run

The existing requirements already include `torch` and `transformers`; no new Python package is required.

```bash
ollama pull llama3.2:latest

OLLAMA_NUM_PARALLEL=2 \
python run_direct_r10_prompt_v3.py \
  --num-questions 2556 \
  --top-k 10 \
  --rerank \
  --candidate-k 50 \
  --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank-batch-size 8 \
  --embed-batch-size 32 \
  --workers 2 \
  --llm-model llama3.2:latest \
  --num-context 4096 \
  --num-predict 64
```

On a memory-constrained Mac, start with `--candidate-k 30 --rerank-batch-size 4`. Model weights are downloaded from Hugging Face on the first run and cached locally.

## Evaluation

Use the repository's existing evaluator on the reranked output JSON:

```bash
python eval.py processed/<reranked-results-file>.json
```

Compare against a baseline run over the same fixed question split, with the same `top-k`, LLM model, prompt, and generation settings. Record retrieval Precision@K, Recall@K, and MRR in addition to EM and Answer F1. Reranking is not guaranteed to improve every metric; validate empirically.

## Scope note

The inspected branch's `IMPLEMENTATION_STATUS.md` explicitly lists iterative query reformulation and second-hop retrieval as not implemented. This change adds a reranking stage to the current single-query dense retrieval runner. It does not itself implement the missing iterative multi-hop controller. A true multi-hop flow should rerank each hop's candidate set (or rerank the merged candidate pool) after query reformulation is added.

Run the lightweight unit tests (after installing the repository requirements):

```bash
python -m unittest discover -s tests -v
```
