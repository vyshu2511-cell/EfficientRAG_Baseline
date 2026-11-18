import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def stratified_split(records, dev_size, test_size, seed=42):
    if dev_size + test_size > len(records):
        raise ValueError("Development and test sizes exceed the dataset size.")

    groups = defaultdict(list)
    for record in records:
        groups[record.get("question_type", "Unknown")].append(record)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    ordered = [record for group in groups.values() for record in group]
    dev = ordered[:dev_size]
    test = ordered[dev_size:dev_size + test_size]
    remaining = ordered[dev_size + test_size:]
    return dev, test, remaining


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Create reproducible RAG splits.")
    parser.add_argument("--input", default="processed/questions.jsonl")
    parser.add_argument("--output-dir", default="processed/splits")
    parser.add_argument("--dev-size", type=int, default=300)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source]

    dev, test, remaining = stratified_split(
        records,
        args.dev_size,
        args.test_size,
        args.seed,
    )

    write_jsonl(output_dir / "development.jsonl", dev)
    write_jsonl(output_dir / "test.jsonl", test)
    write_jsonl(output_dir / "remaining.jsonl", remaining)

    metadata = {
        "seed": args.seed,
        "source": str(input_path),
        "development_size": len(dev),
        "test_size": len(test),
        "remaining_size": len(remaining),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()