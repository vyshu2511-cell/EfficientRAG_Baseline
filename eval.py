import argparse
import json
import string
from collections import Counter
from pathlib import Path


def normalize(text):
    text = str(text or "").lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return " ".join(text.split())


def f1_score(prediction, reference):
    predicted = normalize(prediction).split()
    expected = normalize(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def evidence_keys(question):
    return {
        (item.get("title"), item.get("url"))
        for item in question.get("evidence_list", [])
    }


def retrieval_metrics(result, question):
    gold = evidence_keys(question)
    retrieved = result.get("retrieved_chunks", [])
    hits = [
        item for item in retrieved
        if (item.get("title"), item.get("url")) in gold
        or any(item.get("title") == title for title, _ in gold)
    ]
    if not gold:
        return 0.0, 0.0, 0.0
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    recall = len({item.get("title") for item in hits}) / len(gold)
    rank = next(
        (index for index, item in enumerate(retrieved, 1) if item in hits),
        None,
    )
    return precision, recall, 1 / rank if rank else 0.0


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved RAG results.")
    parser.add_argument("results")
    parser.add_argument("--questions", default="processed/questions.jsonl")
    args = parser.parse_args()

    result_data = json.loads(Path(args.results).read_text(encoding="utf-8"))
    questions = [
        json.loads(line)
        for line in Path(args.questions).read_text(encoding="utf-8").splitlines()
    ]
    by_number = {index + 1: question for index, question in enumerate(questions)}

    em = []
    f1 = []
    precision = []
    recall = []
    reciprocal_rank = []
    for result in result_data["results"]:
        question = by_number[result["question_number"]]
        em.append(int(normalize(result["generated_answer"]) == normalize(result["reference_answer"])))
        f1.append(f1_score(result["generated_answer"], result["reference_answer"]))
        p, r, mrr = retrieval_metrics(result, question)
        precision.append(p)
        recall.append(r)
        reciprocal_rank.append(mrr)

    summary = {
        "questions": len(em),
        "exact_match": sum(em) / len(em),
        "answer_f1": sum(f1) / len(f1),
        "precision_at_k": sum(precision) / len(precision),
        "recall_at_k": sum(recall) / len(recall),
        "mrr": sum(reciprocal_rank) / len(reciprocal_rank),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()