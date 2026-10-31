
import json
import re
from pathlib import Path
from collections import Counter

# File paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load datasets
with open(BASE_DIR / "MultiHopRAG.json", encoding="utf-8") as f:
    questions = json.load(f)

with open(BASE_DIR / "corpus.json", encoding="utf-8") as f:
    documents = json.load(f)

# Clean unnecessary whitespace
def clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()

# Split documents into overlapping word chunks
def create_chunks(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []

    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

        if start + chunk_size >= len(words):
            break

    return chunks

# Save questions, preserving supporting evidence
with open(OUTPUT_DIR / "questions.jsonl", "w", encoding="utf-8") as f:
    for i, question in enumerate(questions):
        record = {
            "question_id": f"q_{i:04d}",
            **question
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

# Process documents and generate chunks
all_chunks = []
empty_documents = 0

with open(OUTPUT_DIR / "documents.jsonl", "w", encoding="utf-8") as f:
    for i, document in enumerate(documents):
        document_id = f"doc_{i:04d}"
        body = clean_text(document.get("body", ""))
        title = clean_text(document.get("title", ""))

        record = {
            **document,
            "document_id": document_id,
            "clean_body": body
        }

        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if not body:
            empty_documents += 1
            continue

        chunks = create_chunks(body)

        for j, chunk in enumerate(chunks):
            all_chunks.append({
                "chunk_id": f"{document_id}_chunk_{j:04d}",
                "document_id": document_id,
                "title": title,
                "source": document.get("source"),
                "url": document.get("url"),
                "text": chunk
            })

# Save chunks
with open(OUTPUT_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
    for chunk in all_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

# Dataset statistics
print("PREPROCESSING COMPLETED")
print("-----------------------------")
print("Total Questions:", len(questions))
print("Total Documents:", len(documents))
print("Empty Documents:", empty_documents)
print("Total Chunks:", len(all_chunks))

print("\nQuestion Types:")
for question_type, count in Counter(
    q.get("question_type", "Unknown") for q in questions
).items():
    print(f"{question_type}: {count}")

print("\nSample Chunk:")
if all_chunks:
    print(all_chunks[0]["text"][:500])

print("\nFiles saved in:", OUTPUT_DIR)
