from collections.abc import Iterable, Iterator
import errno
from pathlib import Path
import stat
import numpy as np
from tempfile import TemporaryDirectory
from contextlib import closing
from cs336_basics.tokenizer import Tokenizer


def _checked_integer(
    value: object,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Validate an integer and normalize NumPy scalars to Python int."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{name} must be an integer, not a boolean; "
            f"got {value!r} ({type(value).__name__})"
        )

    result = int(value)
    if result < minimum or (maximum is not None and result > maximum):
        allowed = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ValueError(f"{name} must be {allowed}; got {result}")
    return result


def _check_output_path(output_path: Path) -> None:
    """Reject an existing output and require an existing parent directory."""
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(
            errno.EEXIST,
            "output_path already exists; refusing to overwrite",
            str(output_path),
        )

    parent = output_path.parent
    try:
        parent_stat = parent.stat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            errno.ENOENT,
            "output_path parent directory does not exist",
            str(parent),
        ) from exc
    if not stat.S_ISDIR(parent_stat.st_mode):
        raise NotADirectoryError(
            errno.ENOTDIR,
            "output_path parent must be a directory",
            str(parent),
        )


def iter_text_chunks(
    input_path: Path,
    chunk_chars: int = 65536,
) -> Iterator[str]:
    chunk_chars = _checked_integer(chunk_chars, "chunk_chars", minimum=1)

    with input_path.open("r", encoding="utf-8",newline="") as f:
        while True:
            chunk = f.read(chunk_chars)
            if chunk == "":
                break
            yield chunk

def write_token_ids_raw(
    token_ids: Iterable[int],
    output_path: Path,
    batch_tokens: int = 65536,
) -> int:
    batch_tokens = _checked_integer(batch_tokens, "batch_tokens", minimum=1)
    _check_output_path(output_path)

    with output_path.open("xb") as f:
        buffer = []
        count = 0

        for index, value in enumerate(token_ids):
            value = _checked_integer(
                value, f"token_ids[{index}]", minimum=0, maximum=65535
            )

            buffer.append(value)
            if len(buffer) >= batch_tokens:
                np.asarray(buffer, dtype="<u2").tofile(f)
                buffer = []
                count += batch_tokens

        if buffer:
            np.asarray(buffer, dtype="<u2").tofile(f)
            count += len(buffer)

        return count

def raw_to_npy(
    raw_path: Path,
    output_path: Path,
    num_tokens: int,
    copy_tokens: int = 65536,
) -> Path:
    num_tokens = _checked_integer(num_tokens, "num_tokens", minimum=0)
    copy_tokens = _checked_integer(copy_tokens, "copy_tokens", minimum=1)

    actual_bytes = raw_path.stat().st_size
    expected_bytes = num_tokens * 2
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"raw_path {str(raw_path)!r} has {actual_bytes} bytes; "
            f"num_tokens={num_tokens} requires {expected_bytes} bytes"
        )
    _check_output_path(output_path)

    with raw_path.open("rb") as f:
        with output_path.open("xb") as target:
            header = {
                "descr": "<u2",
                "fortran_order": False,
                "shape": (num_tokens,),
            }

            np.lib.format.write_array_header_1_0(target, header)

            for start in range(0,num_tokens,copy_tokens):
                cur_count = min(copy_tokens, num_tokens - start)
                expected_chunk_bytes = cur_count * 2
                chunk = f.read(expected_chunk_bytes)
                if len(chunk) != expected_chunk_bytes:
                    raise ValueError(
                        f"raw_path {str(raw_path)!r}: unexpected end at token {start}; "
                        f"expected {expected_chunk_bytes} bytes, got {len(chunk)}"
                    )
                target.write(chunk)

    return output_path

def prepare_token_dataset(
    input_path: Path,
    output_path: Path,
    tokenizer: Tokenizer,
    chunk_chars: int = 65536,
    batch_tokens: int = 65536,
    copy_tokens: int = 65536,
) -> int:
    chunk_chars = _checked_integer(chunk_chars, "chunk_chars", minimum=1)
    batch_tokens = _checked_integer(batch_tokens, "batch_tokens", minimum=1)
    copy_tokens = _checked_integer(copy_tokens, "copy_tokens", minimum=1)
    _check_output_path(output_path)

    with TemporaryDirectory(
        prefix="cs336-prepare-",
        dir=output_path.parent,
    ) as directory:
        directory = Path(directory)
        raw_path = directory / "tokens.raw"
        with closing(iter_text_chunks(input_path, chunk_chars)) as iterator:
            with closing(tokenizer.encode_iterable(iterator)) as token_ids:
                count = write_token_ids_raw(token_ids, raw_path, batch_tokens)
                raw_to_npy(raw_path, output_path, count, copy_tokens)

    return count
