from collections.abc import Iterable, Iterator
import regex as re
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    token_to_id: dict[bytes, int]
    merge_ranks: dict[tuple[bytes, bytes], int]


    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merge_ranks = {merges[i] : i for i in range(len(merges))}
        self.special_tokens = list(special_tokens or [])
        next_id = max(self.vocab, default=-1) + 1

        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                self.vocab[next_id] = token_bytes
                self.token_to_id[token_bytes] = next_id
                next_id += 1

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        chunk = [bytes([token]) for token in pretoken.encode("utf-8")]

        while len(chunk) >= 2:

            min_rank = float('inf')
            best_merge = None
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])

                if pair in self.merge_ranks:
                    if best_merge is None or self.merge_ranks[pair] < min_rank:
                        min_rank = self.merge_ranks[pair]
                        best_merge = pair
            if best_merge is None:
                break
            new_chunk = []
            i = 0

            while i < len(chunk):
                if i + 1 < len(chunk) and (chunk[i], chunk[i + 1]) == best_merge:
                    new_chunk.append(chunk[i] + chunk[i + 1])
                    i += 2
                else:
                    new_chunk.append(chunk[i])
                    i += 1

            chunk = new_chunk

        return [self.token_to_id[token] for token in chunk]

    def _encode_ordinary_text(self,text:str) -> list[int]:
        res = []
        for m in re.finditer(PAT,text):
            res.extend(self._encode_pretoken(m.group()))
        return res

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_ordinary_text(text)

        special_tokens_set = sorted(self.special_tokens, key=len, reverse=True)
        patstr = "|".join(re.escape(token) for token in special_tokens_set)
        pattern = re.compile(f"({patstr})")

        res = []
        lst = 0
        for m in pattern.finditer(text):
            res.extend(self._encode_ordinary_text(text[lst:m.start()]))
            res.append(self.token_to_id[m.group().encode("utf-8")])
            lst = m.end()
        res.extend(self._encode_ordinary_text(text[lst:]))
        return res


    def _encode_ordinary_stream(self, text: str, carry: str, final: bool) -> tuple[list[int], str]:
        buf = carry + text
        if not buf:
            return [], ""

        matches = list(re.finditer(PAT, buf))
        if not matches:
            return [], buf

        # 最后一个匹配若结束于 buf 末尾，它可能被下一块延长，必须扣下。
        if not final and matches[-1].end() == len(buf):
            emit = matches[:-1]
            new_carry = matches[-1].group()
        else:
            emit = matches
            new_carry = ""

        res: list[int] = []
        for m in emit:
            res.extend(self._encode_pretoken(m.group()))
        return res, new_carry

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = (
            re.compile("|".join(re.escape(token) for token in ordered))
            if ordered
            else None
        )

        # pending 是 str，所以这里按特殊 token 的字符数计算。
        # 至少保留 1 个字符，使末尾的普通 pre-token 有机会继续延长。
        reserve = max((len(token) for token in ordered), default=1)
        pending = ""

        for fragment in iterable:
            pending += fragment
            safe_limit = len(pending) - reserve
            if safe_limit <= 0:
                continue

            consumed = 0
            cursor = 0
            stop = False
            special_matches = (
                special_pattern.finditer(pending)
                if special_pattern is not None
                else ()
            )

            for special in special_matches:
                # 先处理该特殊 token 前面的普通文本。
                ordinary = pending[cursor:special.start()]
                for match in re.finditer(PAT, ordinary):
                    end = cursor + match.end()
                    if end > safe_limit:
                        stop = True
                        break

                    yield from self._encode_pretoken(match.group())
                    consumed = end

                if stop:
                    break

                # 靠近缓冲区末尾的特殊 token 可能是更长 token 的前缀。
                if special.end() > safe_limit:
                    stop = True
                    break

                yield self.token_to_id[special.group().encode("utf-8")]
                consumed = special.end()
                cursor = special.end()

            if not stop:
                # 处理最后一个特殊 token 后面的普通文本；
                # 没有特殊 token 时，这里处理整个 pending。
                ordinary = pending[cursor:]
                for match in re.finditer(PAT, ordinary):
                    end = cursor + match.end()
                    if end > safe_limit:
                        break

                    yield from self._encode_pretoken(match.group())
                    consumed = end

            pending = pending[consumed:]

        # 输入结束后，不会再有新字符改变剩余文本的匹配结果。
        yield from self.encode(pending)
        
    def decode(self, ids: list[int]) -> str:
        return (b"".join(self.vocab[i] for i in ids)).decode("utf-8", errors="replace")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        with open(vocab_filepath, encoding="utf-8") as f:
            saved_vocab = json.load(f)

        with open(merges_filepath, encoding="utf-8") as f:
            saved_merges = json.load(f)

        vocab = {
            int(token_id): bytes.fromhex(token_hex)
            for token_id, token_hex in saved_vocab.items()
        }
        merges = [
            (bytes.fromhex(left_hex), bytes.fromhex(right_hex))
            for left_hex, right_hex in saved_merges
        ]

        return cls(vocab, merges, special_tokens)

if __name__ == "__main__":
    pass