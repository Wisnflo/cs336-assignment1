from collections.abc import Iterable, Iterator
from functools import lru_cache
import regex as re
import json

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    """BPE 分词器 (Byte-Pair Encoding)。

    持有词表（id -> 字节串）和合并规则（按优先级排序的字节对列表）。
    编码时先把文本切成 pre-token，再对每个 pre-token 按合并规则反复合并，
    最后查表得到 token id；解码时把 id 对应的字节串拼起来再按 UTF-8 解码。
    """
    token_to_id: dict[bytes, int]
    merge_ranks: dict[tuple[bytes, bytes], int]


    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Args:
            vocab (dict[int, bytes]): 词表，id -> 字节串。
            merges (list[tuple[bytes, bytes]]): 合并规则，列表顺序即优先级（越靠前越先合并）。
            special_tokens (list[str] | None): 特殊 token（如 <|endoftext|>），
                若不在词表中则追加到末尾。
        """
        self.vocab = vocab
        self.merges = merges
        # 反向索引：字节串 -> id，便于编码时查表。
        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        # 合并规则 -> 优先级（下标越小优先级越高）。
        self.merge_ranks = {merges[i] : i for i in range(len(merges))}
        self.special_tokens = list(special_tokens or [])
        # 下一个可用的新 id（从当前最大 id + 1 开始）。
        next_id = max(self.vocab, default=-1) + 1

        # 把不在词表中的特殊 token 追加进去。
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                self.vocab[next_id] = token_bytes
                self.token_to_id[token_bytes] = next_id
                next_id += 1

        # 为每个实例单独建立 pre-token 编码缓存（最多 512 条，LRU 淘汰）。
        # 注意：这里传的是函数对象本身（不加括号），lru_cache 返回一个可调用对象。
        # 缓存成立的前提：实例创建后不再修改词表和合并规则。
        cache_decorator = lru_cache(maxsize=512)
        self._encode_pretoken_cached = cache_decorator(
            self._encode_pretoken_uncached
        )

    def _encode_pretoken_uncached(
        self,
        pretoken: str,
    ) -> tuple[int, ...]:
        """对单个 pre-token 做 BPE 合并（无缓存版本），返回 token id 元组。

        先把 pre-token 的每个字节当作一个初始符号，然后反复找出当前
        优先级最高（rank 最小）的可合并字节对并合并，直到没有可合并的对。
        返回元组而非列表，是为了让 lru_cache 缓存不可变结果，避免调用者
        修改列表后污染后续缓存命中。
        """
        # 初始：每个字节一个符号。
        chunk = [bytes([token]) for token in pretoken.encode("utf-8")]

        while len(chunk) >= 2:

            # 找出当前所有相邻对中 rank 最小（优先级最高）的那个。
            min_rank = float('inf')
            best_merge = None
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i + 1])

                if pair in self.merge_ranks:
                    if best_merge is None or self.merge_ranks[pair] < min_rank:
                        min_rank = self.merge_ranks[pair]
                        best_merge = pair
            # 没有可合并的对，结束。
            if best_merge is None:
                break
            # 从左到右把 best_merge 的相邻对合并成一个新符号。
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

        # 把最终符号查表转成 id，返回元组（不可变，适合缓存）。
        return tuple(self.token_to_id[token] for token in chunk)

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        """对单个 pre-token 做 BPE 合并，返回 token id 列表（带缓存）。

        实际计算委托给带 lru_cache 的 _encode_pretoken_cached；命中缓存时
        直接返回已算好的元组。这里把元组转成新列表返回，避免调用者修改
        缓存中的共享对象。

        注意：不要对 pretoken 做 strip()，前导空格有意义（"the" 与 " the" 不同）。
        """
        return list(self._encode_pretoken_cached(pretoken))

    def _encode_ordinary_text(self,text:str) -> list[int]:
        """对不含特殊 token 的普通文本编码：先按 PAT 切分，再逐个 BPE 合并。"""
        res = []
        for m in re.finditer(PAT,text):
            res.extend(self._encode_pretoken(m.group()))
        return res

    def encode(self, text: str) -> list[int]:
        """把文本编码成 token id 列表。

        若配置了特殊 token，则先按特殊 token 切分，特殊 token 直接映射为
        对应 id，其余普通文本再走普通编码流程。

        Args:
            text (str): 待编码文本。

        Returns:
            list[int]: token id 列表。
        """
        # 没有特殊 token 时直接走普通编码。
        if not self.special_tokens:
            return self._encode_ordinary_text(text)

        # 按长度降序排列，保证更长的特殊 token 优先匹配（避免前缀误匹配）。
        special_tokens_set = sorted(self.special_tokens, key=len, reverse=True)
        patstr = "|".join(re.escape(token) for token in special_tokens_set)
        pattern = re.compile(f"({patstr})")

        res = []
        lst = 0
        for m in pattern.finditer(text):
            # 特殊 token 之前的普通文本。
            res.extend(self._encode_ordinary_text(text[lst:m.start()]))
            # 特殊 token 本身直接查表。
            res.append(self.token_to_id[m.group().encode("utf-8")])
            lst = m.end()
        # 最后一个特殊 token 之后的普通文本。
        res.extend(self._encode_ordinary_text(text[lst:]))
        return res


    def _encode_ordinary_stream(self, text: str, carry: str, final: bool) -> tuple[list[int], str]:
        """流式编码的辅助函数：处理一块普通文本，返回 (已编码 id, 需保留的尾部)。

        难点在于块边界可能把一个 pre-token 切断，因此不能立即编码末尾那个
        可能被下一块延长的匹配，要把它作为 carry 留给下一块。

        Args:
            text (str): 本块新增文本。
            carry (str): 上一块遗留的、可能被延长的文本。
            final (bool): 是否为最后一块（最后一块无需保留尾部）。

        Returns:
            tuple[list[int], str]: (本块可安全编码的 id 列表, 需保留到下一块的文本)。
        """
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
        """流式编码：逐块读取文本并产出 token id，内存占用与文本总长无关。

        维护一个 pending 缓冲区，只编码“安全”的部分（不会被后续字符改变
        匹配结果的部分），其余留到下一轮。

        Args:
            iterable (Iterable[str]): 文本块迭代器（如逐行读取的文件对象）。

        Yields:
            int: 逐个 token id。
        """
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
            # 只处理到 safe_limit 之前的内容，之后的部分可能被后续字符影响。
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

            # 丢弃已消费部分，剩余留到下一轮。
            pending = pending[consumed:]

        # 输入结束后，不会再有新字符改变剩余文本的匹配结果。
        yield from self.encode(pending)
        
    def decode(self, ids: list[int]) -> str:
        """把 token id 列表解码回文本。

        把每个 id 对应的字节串拼接后按 UTF-8 解码；遇到非法字节用
        errors="replace" 替换，避免抛异常。
        """
        return (b"".join(self.vocab[i] for i in ids)).decode("utf-8", errors="replace")

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """从 JSON 文件加载词表和合并规则，构造 Tokenizer。

        文件格式：vocab 为 {id字符串: 十六进制字节串}，merges 为
        [[左十六进制, 右十六进制], ...]。

        Args:
            vocab_filepath (str): 词表 JSON 路径。
            merges_filepath (str): 合并规则 JSON 路径。
            special_tokens (list[str] | None): 特殊 token 列表。

        Returns:
            Tokenizer: 构造好的分词器实例。
        """
        with open(vocab_filepath, encoding="utf-8") as f:
            saved_vocab = json.load(f)

        with open(merges_filepath, encoding="utf-8") as f:
            saved_merges = json.load(f)

        # 把十六进制字符串还原成字节串。
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