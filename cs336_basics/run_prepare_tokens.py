import argparse
from pathlib import Path
from cs336_basics.run_training import positive_int
import errno
import stat
import json
from time import perf_counter
from cs336_basics.prepare_tokens import prepare_token_dataset
from cs336_basics.prepare_tokens import _check_output_path
from cs336_basics.run_generation import load_tokenizer

def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a token dataset from raw text.")
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--vocab-file", type=Path, required=True)
    parser.add_argument("--merges-file", type=Path, required=True)
    parser.add_argument("--vocab-size", type=positive_int, required=True)
    parser.add_argument("--chunk-chars", type=positive_int, default=65536)
    parser.add_argument("--batch-tokens", type=positive_int, default=65536)
    parser.add_argument("--copy-tokens", type=positive_int, default=65536)
    parser.add_argument("--eos-token", type=str, default="<|endoftext|>")
    return parser.parse_args(argv)

def validate_args(args: argparse.Namespace) -> None:
    if not 256 <= args.vocab_size <= 65536:
        raise ValueError(f"vocab_size must be in [256, 65536]; got {args.vocab_size}")
    if args.eos_token == "":
        raise ValueError("eos_token must not be an empty string")

    for name, path in (
        ("input_path", args.input_path),
        ("vocab_file", args.vocab_file),
        ("merges_file", args.merges_file),
    ):
        try:
            info = path.stat()
        except FileNotFoundError as exc:
            raise FileNotFoundError(errno.ENOENT,
                f"{name} does not exist",
                str(path),
            ) from exc
        if stat.S_ISDIR(info.st_mode):
            raise IsADirectoryError(
                errno.EISDIR,
                f"{name} must be a regular file, not a directory",
                str(path),
            )
        if not stat.S_ISREG(info.st_mode):
            raise ValueError(
                f"{name} must be a regular file; got {path}"
            )

    _check_output_path(args.output_path)

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    tokenizer = load_tokenizer(args)
    input_bytes = args.input_path.stat().st_size
    started_at = perf_counter()
    num_tokens = prepare_token_dataset(
        input_path=args.input_path,
        output_path=args.output_path,
        tokenizer=tokenizer,
        chunk_chars=args.chunk_chars,
        batch_tokens=args.batch_tokens,
        copy_tokens=args.copy_tokens,
    )
    elapsed_seconds = perf_counter() - started_at
    processing_bytes_per_second = (
        input_bytes / elapsed_seconds if elapsed_seconds > 0 else None
    )

    metadata = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "vocab_file": str(args.vocab_file),
        "merges_file": str(args.merges_file),
        "vocab_size": args.vocab_size,
        "eos_token": args.eos_token,
        "dtype": "<u2",
        "input_bytes": input_bytes,
        "num_tokens": num_tokens,
        "output_bytes": args.output_path.stat().st_size,
        "elapsed_seconds": elapsed_seconds,
        "processing_bytes_per_second": processing_bytes_per_second,
        "chunk_chars": args.chunk_chars,
        "batch_tokens": args.batch_tokens,
        "copy_tokens": args.copy_tokens,
    }

    print(json.dumps(metadata, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
      