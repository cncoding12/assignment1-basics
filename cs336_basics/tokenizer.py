import json
import regex as re
from typing import Iterable, Iterator


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        初始化 BPE Tokenizer
        """
        self.vocab = dict(vocab)
        self.bytes_to_id: dict[bytes, int] = {
            v: k for k, v in self.vocab.items()
        }

        # 维护 special_tokens
        self.special_tokens = special_tokens or []
        # 按长度降序排列特殊字符，避免前缀匹配问题
        self.special_tokens.sort(key=len, reverse=True)

        for token_str in self.special_tokens:
            token_bytes = token_str.encode("utf-8")
            if token_bytes not in self.bytes_to_id:
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.bytes_to_id[token_bytes] = new_id

        # 维护 merges 规则的合并优先级 (Rank)
        self.merges = merges
        self.merges_rank: dict[tuple[bytes, bytes], int] = {
            pair: i for i, pair in enumerate(merges)
        }

        # GPT-2 预分词正则表达式
        self.gpt2_pat = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        """
        从序列化的词表和 merges JSON 文件反序列化加载构造 Tokenizer
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
        vocab = {
            int(k): v.encode("utf-8") if isinstance(v, str) else bytes(v)
            for k, v in vocab_raw.items()
        }

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_raw = json.load(f)
        merges = [
            (
                m[0].encode("utf-8") if isinstance(m[0], str) else bytes(m[0]),
                m[1].encode("utf-8") if isinstance(m[1], str) else bytes(m[1]),
            )
            for m in merges_raw
        ]

        return cls(vocab, merges, special_tokens=special_tokens)

    def _encode_word(self, word_bytes: bytes) -> list[int]:
        """
        对单预分词 (Pre-token) 应用 BPE 规约合并
        """
        if not word_bytes:
            return []

        # 拆分为单字节序列
        word = [bytes([b]) for b in word_bytes]

        while len(word) >= 2:
            # 找到当前词中具有最小 Rank 的可合并 Pair
            min_rank = float("inf")
            best_pair = None

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merges_rank:
                    rank = self.merges_rank[pair]
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair

            # 若没有任何相邻 Pair 在 merges 规约中，合并终止
            if best_pair is None:
                break

            # 执行合并
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(best_pair[0] + best_pair[1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        # 映射为整数 ID
        return [self.bytes_to_id[b] for b in word]

    def encode(self, text: str) -> list[int]:
        """
        将文本字符串编码为 Token ID 序列
        """
        if not text:
            return []

        # 1. 如果有 special_tokens，按 special_tokens 切分文本（保留分隔符）
        if self.special_tokens:
            special_regex = "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            chunks = re.split(special_regex, text)
        else:
            chunks = [text]

        ids: list[int] = []
        special_bytes_set = {s.encode("utf-8") for s in self.special_tokens}

        for chunk in chunks:
            if not chunk:
                continue

            chunk_bytes = chunk.encode("utf-8")
            # 如果是特殊 Token，直接查找 ID
            if chunk_bytes in special_bytes_set:
                ids.append(self.bytes_to_id[chunk_bytes])
            else:
                # 普通文本进行 GPT-2 正则预分词 + BPE 合并
                for match in self.gpt2_pat.finditer(chunk):
                    word_bytes = match.group(0).encode("utf-8")
                    ids.extend(self._encode_word(word_bytes))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        针对字符串可迭代对象（如文件句柄）的惰性生成器分词实现
        """
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        将 Token ID 序列解码还原为文本字符串
        """
        byte_pieces = [self.vocab[i] for i in ids if i in self.vocab]
        all_bytes = b"".join(byte_pieces)
        # errors="replace" 会自动将无效 UTF-8 字节替换为 U+FFFD 官方替换字符
        return all_bytes.decode("utf-8", errors="replace")