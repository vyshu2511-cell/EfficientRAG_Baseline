# EfficientRAG_Baseline
NLP Multi-Hop RAG Question Answering Project

llama3-8b, top-K 10, 98 questions, EM and F1 score
98 60.2 62.98

To run command llama3:8b
OLLAMA_NUM_PARALLEL=2 \
python run_direct_r10_prompt_v3.py \
  --num-questions 2556 \
  --top-k 10 \
  --embed-batch-size 32 \
  --workers 2 \
  --llm-model llama3:8b

Run llama3.2 
OLLAMA_NUM_PARALLEL=2 \
python run_direct_r10_prompt_v3.py \
  --num-questions 2556 \
  --top-k 10 \
  --embed-batch-size 32 \
  --workers 2 \
  --llm-model llama3.2:latest \
  --num-context 4096 \
  --num-predict 64