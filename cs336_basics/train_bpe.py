import regex as re
from collections import Counter, defaultdict


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    100% 准确匹配官方 Snapshot 且高性能的 Byte-Level BPE 训练实现。
    """
    # 1. 初始化基础词表 (0-255 基础字节)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # 2. 将 special_tokens 顺延加入词表
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, merges

    # 3. 读取完整文件（不能按 \n 逐行读取，必须保持 \n\n 等连续空白字符完整）
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 4. 根据 special_tokens 进行文档级切分
    if special_tokens:
        special_regex = re.compile(
            "|".join(
                re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)
            )
        )
        chunks = [c for c in special_regex.split(text) if c]
    else:
        chunks = [text]

    # 5. GPT-2 正则表达式预分词
    gpt2_pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

    word_counts: dict[tuple[bytes, ...], int] = Counter()
    for chunk in chunks:
        for word_str in gpt2_pat.findall(chunk):
            word_bytes = word_str.encode("utf-8")
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            word_counts[word_tuple] += 1

    # 6. 构建初始 pair 频次表 (pair_counts) 及 倒排索引 (pair_to_words)
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set] = defaultdict(set)

    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    # 7. 增量 BPE 迭代合并
    for _ in range(num_merges):
        if not pair_counts:
            break

        # 选取最高频 pair；频次相同时按字典序大的优先 (Tie-breaking)
        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        # 仅获取包含 best_pair 的词（避免全量扫描）
        words_to_update = [w for w in pair_to_words[best_pair] if w in word_counts]

        # 清理已合并的 best_pair 记录
        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word in words_to_update:
            count = word_counts[word]

            # (A) 扣除旧词中所有 pair 的频次
            old_word_pairs = Counter()
            for i in range(len(word) - 1):
                old_word_pairs[(word[i], word[i + 1])] += 1

            for p, freq in old_word_pairs.items():
                if p in pair_counts:
                    pair_counts[p] -= freq * count
                    if pair_counts[p] <= 0:
                        del pair_counts[p]

            # (B) 替换 best_pair 生成新词
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)

            # 更新词频表
            del word_counts[word]
            word_counts[new_word_tuple] = (
                word_counts.get(new_word_tuple, 0) + count
            )

            # (C) 增加新词中所有 pair 的频次并更新倒排索引
            new_word_pairs = Counter()
            for i in range(len(new_word_tuple) - 1):
                new_word_pairs[(new_word_tuple[i], new_word_tuple[i + 1])] += 1

            for p, freq in new_word_pairs.items():
                pair_counts[p] += freq * count
                pair_to_words[p].add(new_word_tuple)

    return vocab, merges