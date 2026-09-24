"""Small, self-contained checks for BPE tokenizer serialization."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from cs336_basics.tokenizer import Tokenizer
from cs336_basics.train_bpe import train_bpe


def write_tokenizer_files(
    directory: Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
) -> tuple[Path, Path]:
    vocab_path = directory / "vocab.json"
    merges_path = directory / "merges.json"

    with vocab_path.open("w", encoding="utf-8") as file:
        json.dump({str(token_id): value.hex() for token_id, value in vocab.items()}, file)

    with merges_path.open("w", encoding="utf-8") as file:
        json.dump([[left.hex(), right.hex()] for left, right in merges], file)

    return vocab_path, merges_path


def test_from_files_roundtrip() -> None:
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    vocab[256] = b"ab"
    merges = [(b"a", b"b")]
    special_token = "<|endoftext|>"
    original = Tokenizer(vocab.copy(), merges, [special_token])

    with TemporaryDirectory() as temporary_directory:
        vocab_path, merges_path = write_tokenizer_files(
            Path(temporary_directory), original.vocab, original.merges
        )
        loaded = Tokenizer.from_files(
            str(vocab_path), str(merges_path), [special_token]
        )

    assert loaded.vocab == original.vocab
    assert loaded.merges == original.merges
    assert loaded.special_tokens == original.special_tokens

    for text in (
        "",
        "ab",
        "hello ab",
        "你好 🙂",
        "ab<|endoftext|>ab",
        "<|endoftext|><|endoftext|>",
    ):
        ids = original.encode(text)
        assert loaded.encode(text) == ids
        assert loaded.decode(ids) == text


def test_from_files_adds_missing_special_token() -> None:
    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    vocab[256] = b"ab"
    merges = [(b"a", b"b")]
    special_token = "<|endoftext|>"

    with TemporaryDirectory() as temporary_directory:
        vocab_path, merges_path = write_tokenizer_files(
            Path(temporary_directory), vocab, merges
        )
        loaded = Tokenizer.from_files(
            str(vocab_path), str(merges_path), [special_token]
        )

    assert loaded.vocab[257] == special_token.encode("utf-8")
    assert loaded.encode("ab<|endoftext|>ab") == [256, 257, 256]


def test_trained_tokenizer_roundtrip() -> None:
    fixtures = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
    training_path = fixtures / "tinystories_sample_5M.txt"
    sample_path = fixtures / "tinystories_sample.txt"
    special_token = "<|endoftext|>"

    vocab, merges = train_bpe(str(training_path), 1000, [special_token])
    tokenizer = Tokenizer(vocab, merges, [special_token])
    sample = sample_path.read_text(encoding="utf-8")
    ids = tokenizer.encode(sample)

    assert tokenizer.decode(ids) == sample
    with sample_path.open(encoding="utf-8") as file:
        assert list(tokenizer.encode_iterable(file)) == ids

    with TemporaryDirectory() as temporary_directory:
        vocab_path, merges_path = write_tokenizer_files(
            Path(temporary_directory), vocab, merges
        )
        loaded = Tokenizer.from_files(
            str(vocab_path), str(merges_path), [special_token]
        )

    assert loaded.vocab == vocab
    assert loaded.merges == merges
    assert loaded.encode(sample) == ids
    assert loaded.decode(ids) == sample


if __name__ == "__main__":
    test_from_files_roundtrip()
    test_from_files_adds_missing_special_token()
    test_trained_tokenizer_roundtrip()
    print("Tokenizer serialization and end-to-end checks passed")
