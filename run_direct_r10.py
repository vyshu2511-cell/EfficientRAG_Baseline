
import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import requests
import torch
from transformers import AutoTokenizer, AutoModel

# -----------------------------------
# 1. Command-line arguments
# -----------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--num-questions",
    type=int,
    default=1,
    help="Number of questions to evaluate"
)

parser.add_argument(
    "--top-k",
    type=int,
    default=10,
    help="Number of chunks to retrieve"
)

args = parser.parse_args()

NUM_QUESTIONS = args.num_questions
TOP_K = args.top_k

if NUM_QUESTIONS < 1 or TOP_K < 1:
    parser.error("Both arguments must be positive integers.")

MODEL_NAME = "facebook/contriever-msmarco"
LLM_NAME = "llama3:8b"

PROCESSED = Path(__file__).resolve().parent / "processed"

# -----------------------------------
# 2. Load data
# -----------------------------------

with open(PROCESSED / "questions.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f]

with open(PROCESSED / "chunks.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

embeddings = np.load(
    PROCESSED / "contriever_embeddings.npy"
)

assert len(embeddings) == len(chunks)

NUM_QUESTIONS = min(NUM_QUESTIONS, len(questions))
TOP_K = min(TOP_K, len(chunks))

print("Total questions available:", len(questions))
print("Questions to evaluate:", NUM_QUESTIONS)
print("Top-K retrieval:", TOP_K)
print("Embedding shape:", embeddings.shape)

# -----------------------------------
# 3. Load Contriever
# -----------------------------------

print("\nLoading Contriever-MSMARCO...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = model.to(device)
model.eval()

# -----------------------------------
# 4. Question embedding
# -----------------------------------

def embed_question(question):

    inputs = tokenizer(
        [question],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1)

        embedding = (
            (outputs.last_hidden_state * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1)
        )

    return embedding.cpu().numpy()[0]

# -----------------------------------
# 5. Retrieve top-K chunks
# -----------------------------------

def retrieve(question):

    query_vector = embed_question(question)

    scores = embeddings @ query_vector

    indices = np.argsort(scores)[::-1][:TOP_K]

    return [
        {
            "chunk_index": int(index),
            "title": chunks[int(index)]["title"],
            "text": chunks[int(index)]["text"],
            "score": float(scores[index])
        }
        for index in indices
    ]

# -----------------------------------
# 6. Generate answer using Llama 3
# -----------------------------------

def generate_answer(question, retrieved):

    context = "\n\n".join(
        f"[Document {i}]\n"
        f"Title: {chunk['title']}\n"
        f"Content: {chunk['text']}"
        for i, chunk in enumerate(retrieved, 1)
    )

    prompt = f"""Answer the question using only the
provided documents.

Give a short, direct answer.
If the documents do not contain enough
information, answer: Insufficient evidence.

DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_ctx": 8192,
                "num_predict": 120
            }
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()["response"].strip()

# -----------------------------------
# 7. Evaluation metrics
# -----------------------------------

def normalize_answer(text):

    text = str(text or "").lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    text = re.sub(r"\b(a|an|the)\b", " ", text)

    return " ".join(text.split())


def exact_match(prediction, reference):

    return int(
        normalize_answer(prediction)
        == normalize_answer(reference)
    )


def calculate_f1(prediction, reference):

    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    common = Counter(pred_tokens) & Counter(ref_tokens)
    overlap = sum(common.values())

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)

    return 2 * precision * recall / (precision + recall)

# -----------------------------------
# 8. Run the experiment
# -----------------------------------

results = []

output_file = (
    PROCESSED
    / f"direct_r{TOP_K}_results_{NUM_QUESTIONS}q.json"
)

for i, item in enumerate(questions[:NUM_QUESTIONS], 1):

    question = item["query"]
    reference = item["answer"]

    print(f"\n========== QUESTION {i}/{NUM_QUESTIONS} ==========")
    print("Question:", question)

    retrieved = retrieve(question)

    prediction = generate_answer(question, retrieved)

    em = exact_match(prediction, reference)
    f1 = calculate_f1(prediction, reference)

    print("Generated answer:", prediction)
    print("Correct answer:", reference)
    print("Exact Match:", em)
    print("Answer F1:", round(f1, 4))

    results.append({
        "question_number": i,
        "question": question,
        "reference_answer": reference,
        "generated_answer": prediction,
        "exact_match": em,
        "answer_f1": f1,
        "retrieved_chunks": [
            {
                "chunk_index": c["chunk_index"],
                "title": c["title"],
                "score": c["score"]
            }
            for c in retrieved
        ]
    })

    # Save after each question so progress is retained
    current_em = sum(r["exact_match"] for r in results)
    current_f1 = sum(r["answer_f1"] for r in results)

    data = {
        "model": LLM_NAME,
        "retriever": MODEL_NAME,
        "top_k": TOP_K,
        "questions_evaluated": len(results),
        "exact_match_accuracy": current_em / len(results),
        "average_answer_f1": current_f1 / len(results),
        "results": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# -----------------------------------
# 9. Final evaluation results
# -----------------------------------

correct = sum(r["exact_match"] for r in results)

average_em = correct / len(results)

average_f1 = sum(
    r["answer_f1"] for r in results
) / len(results)

print("\n====================================")
print("DIRECT-R@10 BASELINE EVALUATION")
print("====================================")
print("Questions evaluated:", len(results))
print("Exact matches:", correct)
print("Exact Match Accuracy:", round(average_em * 100, 2), "%")
print("Average Answer F1:", round(average_f1, 4))
print("Average Answer F1 (%):", round(average_f1 * 100, 2), "%")
print("Results saved to:", output_file)
