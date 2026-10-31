
import json
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--question", type=int, default=10)
args = parser.parse_args()

processed = Path("processed")

# Load baseline results
with open(
    processed / "direct_r10_results_10q.json",
    encoding="utf-8"
) as f:
    data = json.load(f)

results = data["results"]

# Load document chunks
with open(
    processed / "chunks.jsonl",
    encoding="utf-8"
) as f:
    chunks = [json.loads(line) for line in f]

if not 1 <= args.question <= len(results):
    parser.error("Question number is outside saved results.")

result = results[args.question - 1]

print("\nQUESTION:", result["question"])
print("CORRECT ANSWER:", result["reference_answer"])
print("GENERATED ANSWER:", result["generated_answer"])

output_file = (
    processed / f"retrieval_q{args.question}.txt"
)

with open(output_file, "w", encoding="utf-8") as f:

    f.write(f"QUESTION: {result['question']}\n")
    f.write(f"CORRECT ANSWER: {result['reference_answer']}\n")
    f.write(f"GENERATED: {result['generated_answer']}\n\n")

    for rank, item in enumerate(
        result["retrieved_chunks"], start=1
    ):

        chunk = chunks[item["chunk_index"]]

        print(f"{rank}. {chunk['title']}")

        f.write(f"\n{'=' * 55}\n")
        f.write(f"RANK: {rank}\n")
        f.write(f"TITLE: {chunk['title']}\n")
        f.write(f"SCORE: {item['score']:.4f}\n\n")
        f.write(chunk["text"] + "\n")

print("\nFull retrieved text saved to:", output_file)
