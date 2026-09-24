"""Train a BPE tokenizer and keep its artifacts for later experiments."""

import argparse
import json
import time
from pathlib import Path

from cs336_basics.train_bpe import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_path", type=Path, help="UTF-8 training corpus")
    parser.add_argument("output_dir", type=Path, help="New directory for results")
    parser.add_argument("--vocab-size", type=int, required=True)
    args = parser.parse_args()

    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    special_tokens = ["<|endoftext|>"]

    if not input_path.is_file():
        parser.error(f"input file does not exist: {input_path}")
    if args.vocab_size < 256 + len(special_tokens):
        parser.error("vocab size must leave room for bytes and special tokens")
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")

    started = time.perf_counter()
    vocab, merges = train_bpe(str(input_path), args.vocab_size, special_tokens)
    elapsed_seconds = time.perf_counter() - started

    output_dir.mkdir(parents=True)
    with (output_dir / "vocab.json").open("w", encoding="utf-8") as file:
        json.dump({str(token_id): value.hex() for token_id, value in vocab.items()}, file)
    with (output_dir / "merges.json").open("w", encoding="utf-8") as file:
        json.dump([[left.hex(), right.hex()] for left, right in merges], file)

    longest_id, longest_bytes = max(vocab.items(), key=lambda item: len(item[1]))
    metadata = {
        "input_path": str(input_path),
        "input_bytes": input_path.stat().st_size,
        "requested_vocab_size": args.vocab_size,
        "actual_vocab_size": len(vocab),
        "num_merges": len(merges),
        "special_tokens": special_tokens,
        "elapsed_seconds": elapsed_seconds,
        "longest_token_id": longest_id,
        "longest_token_bytes": len(longest_bytes),
        "longest_token_hex": longest_bytes.hex(),
        "longest_token_text": longest_bytes.decode("utf-8", errors="replace"),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
