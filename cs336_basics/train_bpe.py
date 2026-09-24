from multiprocessing import Pool
import regex as re
import os
from typing import BinaryIO
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pre_and_count_process(
    file_path: str,
    start: int,
    end: int,
    special_tokens:list[str]
):
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8",errors="ignore")
        if special_tokens:
            pattern = "|".join(re.escape(token) for token in special_tokens)
            segments = re.split(f"({pattern})", chunk)
        else:
            segments = [chunk]

        pre_tokens = []
        for seg in segments:
            if seg in special_tokens:
                continue;
            pre_tokens.extend(m.group().encode("utf-8") for m in re.finditer(PAT, seg))

        # --- OLD: count pairs over every occurrence (kept for reference) ---
        # freq: dict[tuple[bytes, bytes], int] = {}
        # for token in pre_tokens:
        #     for i in range(len(token) - 1):
        #         pair = (token[i:i+1], token[i+1:i+2])
        #         freq[pair] = freq.get(pair, 0) + 1
        #
        # return pre_tokens, freq

        # --- NEW: aggregate identical pre-tokens into (symbol sequence -> count) ---
        # e.g. "low" appearing 5000 times is stored once with count 5000.
        word_counts: dict[tuple[bytes, ...], int] = {}
        for token in pre_tokens:
            symbols = tuple(token[i:i+1] for i in range(len(token)))
            word_counts[symbols] = word_counts.get(symbols, 0) + 1

        # Pair counts are the weighted sum over unique sequences.
        freq: dict[tuple[bytes, bytes], int] = {}
        for symbols, count in word_counts.items():
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])
                freq[pair] = freq.get(pair, 0) + count

        return word_counts, freq

def _pre_and_count_task(task):
    return pre_and_count_process(*task)

def merge_most_freq(tokens: list[list[bytes]], max_pair: tuple[bytes, bytes],freq:dict[tuple[bytes,bytes],int]):
    new_tokens : list[list[bytes]] = []
    for token in tokens:
        new_token = []
        i = 0
        changed = False
        while i < len(token):
            if i < len(token) - 1 and (token[i], token[i+1]) == max_pair:
                new_token.append(token[i] + token[i+1])
                changed = True
                i += 2
            else:
                new_token.append(token[i])
                i += 1

        # Only sequences that actually changed need their pair counts adjusted.
        # Do it per whole sequence: subtract all old pairs, then add all new pairs.
        # This avoids double-counting shared edges and never creates transient pairs.
        if changed:
            for j in range(len(token) - 1):
                old_pair = (token[j], token[j+1])
                freq[old_pair] = freq.get(old_pair, 0) - 1
                if freq[old_pair] == 0:
                    del freq[old_pair]
            for j in range(len(new_token) - 1):
                new_pair = (new_token[j], new_token[j+1])
                freq[new_pair] = freq.get(new_pair, 0) + 1

        new_tokens.append(new_token)

    return new_tokens, freq


def merge_most_freq_weighted(
    word_counts: dict[tuple[bytes, ...], int],
    max_pair: tuple[bytes, bytes],
    freq: dict[tuple[bytes, bytes], int],
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
):
    """
    Merge `max_pair` in every unique symbol sequence, weighted by occurrence count.

    `pair_to_words` maps a pair to the set of unique sequences containing it, so we
    only touch sequences that actually contain `max_pair` instead of scanning all
    pre-tokens every round.
    """
    # Snapshot the affected sequences: we mutate pair_to_words while iterating.
    affected = list(pair_to_words.get(max_pair, ()))

    for symbols in affected:
        count = word_counts[symbols]

        # Build the merged sequence.
        new_symbols: list[bytes] = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and (symbols[i], symbols[i+1]) == max_pair:
                new_symbols.append(symbols[i] + symbols[i+1])
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        new_symbols_t = tuple(new_symbols)

        # Remove the old sequence from the index and its weighted pair counts.
        for j in range(len(symbols) - 1):
            old_pair = (symbols[j], symbols[j+1])
            freq[old_pair] -= count
            if freq[old_pair] <= 0:
                del freq[old_pair]
            bucket = pair_to_words.get(old_pair)
            if bucket is not None:
                bucket.discard(symbols)
                if not bucket:
                    del pair_to_words[old_pair]

        # Drop the old sequence, then register the new one with its weighted pairs.
        del word_counts[symbols]
        word_counts[new_symbols_t] = word_counts.get(new_symbols_t, 0) + count

        for j in range(len(new_symbols_t) - 1):
            new_pair = (new_symbols_t[j], new_symbols_t[j+1])
            freq[new_pair] = freq.get(new_pair, 0) + count
            pair_to_words.setdefault(new_pair, set()).add(new_symbols_t)

    return word_counts, freq, pair_to_words
        
def train_bpe(input_path:str,vocab_size:int,special_tokens:list[str]):
    desired_num_chunks =10000

    chunk_boundaries = []
    encoded_special_tokens = [special_token.encode("utf-8") for special_token in special_tokens]
    with open(input_path,"rb") as f:
        for special_token in encoded_special_tokens:
            chunk_boundaries.extend(find_chunk_boundaries(f, desired_num_chunks, special_token))
    chunk_boundaries = sorted(set(chunk_boundaries))
    tasks = (
        (input_path, start, end, special_tokens)
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    )

    word_counts: dict[tuple[bytes, ...], int] = {}
    freq: dict[tuple[bytes, bytes], int] = {}

    with Pool(processes=4) as pool:
        for chunk_word_counts, chunk_freq in pool.imap_unordered(
            _pre_and_count_task, tasks, chunksize=16
        ):
            for symbols, count in chunk_word_counts.items():
                word_counts[symbols] = word_counts.get(symbols, 0) + count
            for pair, count in chunk_freq.items():
                freq[pair] = freq.get(pair, 0) + count

    # pair -> set of unique sequences containing that pair, so each merge round
    # only visits sequences that actually contain the chosen pair.
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}
    for symbols in word_counts:
        for i in range(len(symbols) - 1):
            pair_to_words.setdefault((symbols[i], symbols[i+1]), set()).add(symbols)

    vocab : dict[int, bytes] = {}
    merge : list[tuple[bytes, bytes]] = []
    for i in range(0,256):
        vocab[i] = bytes([i])
    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    while len(vocab) < vocab_size:
        if not freq:
            break
        max_pair = max(freq.items(), key=lambda x: (x[1],x[0]), default=(None, 0))[0]

        # --- OLD: scan every pre-token each round (kept for reference) ---
        # tokens,freq = merge_most_freq(tokens, max_pair, freq)

        # --- NEW: only touch sequences containing max_pair ---
        word_counts, freq, pair_to_words = merge_most_freq_weighted(
            word_counts, max_pair, freq, pair_to_words
        )

        vocab[len(vocab)] = max_pair[0] + max_pair[1]
        merge.append(max_pair)

    return vocab,merge