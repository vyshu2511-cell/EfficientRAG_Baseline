
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# -----------------------------------
# STEP 1: Set up file paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "processed"

MODEL_NAME = "facebook/contriever-msmarco"
MAX_TOKENS = 512

# -----------------------------------
# STEP 2: Load preprocessed chunks
# -----------------------------------

with open(PROCESSED_DIR / "chunks.jsonl", encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

print("Total chunks loaded:", len(chunks))

texts = [
    f"{chunk['title']}\n{chunk['text']}"
    for chunk in chunks
]

# -----------------------------------
# STEP 3: Load Contriever model
# -----------------------------------

print("\nLoading Contriever-MSMARCO...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = model.to(device)
model.eval()

print("Using device:", device)

# -----------------------------------
# STEP 4: Convert text into embeddings
# -----------------------------------

def generate_embeddings(texts, batch_size=8):

    all_embeddings = []
    truncated_count = 0

    for start in tqdm(
        range(0, len(texts), batch_size),
        desc="Generating embeddings"
    ):

        batch = texts[start:start + batch_size]

        # Check whether any text exceeds 512 tokens
        full_tokens = tokenizer(
            batch,
            add_special_tokens=True,
            truncation=False
        )

        truncated_count += sum(
            len(ids) > MAX_TOKENS
            for ids in full_tokens["input_ids"]
        )

        # Tokenize the text
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # Generate token representations
        with torch.no_grad():
            outputs = model(**inputs)

            # Mean pooling, excluding padding tokens
            mask = inputs["attention_mask"].unsqueeze(-1)

            masked = outputs.last_hidden_state * mask

            embeddings = masked.sum(dim=1) / mask.sum(
                dim=1
            ).clamp(min=1)

        all_embeddings.append(
            embeddings.cpu().numpy()
        )

    return np.vstack(all_embeddings), truncated_count

print("\nGenerating document embeddings...")

embeddings, truncated_count = generate_embeddings(texts)

# -----------------------------------
# STEP 5: Save embeddings
# -----------------------------------

output_file = (
    PROCESSED_DIR / "contriever_embeddings.npy"
)

np.save(output_file, embeddings)

print("\nEMBEDDING GENERATION COMPLETED")
print("--------------------------------")
print("Total embeddings:", len(embeddings))
print("Embedding shape:", embeddings.shape)
print("Chunks exceeding token limit:", truncated_count)
print("Saved to:", output_file)

# -----------------------------------
# STEP 6: Test retrieval on one question
# -----------------------------------

with open(PROCESSED_DIR / "questions.jsonl", encoding="utf-8") as f:
    first_question = json.loads(next(f))

question = first_question["query"]

query_embeddings, _ = generate_embeddings([question])

# Contriever dot-product similarity
scores = embeddings @ query_embeddings[0]

top_indices = np.argsort(scores)[::-1][:10]

print("\nTEST QUESTION:")
print(question)

print("\nTOP 10 RETRIEVED CHUNKS:")

for rank, index in enumerate(top_indices, start=1):
    chunk = chunks[int(index)]

    print(f"\nRank {rank}")
    print("Title:", chunk["title"])
    print("Score:", round(float(scores[index]), 4))
