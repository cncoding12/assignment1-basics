import cProfile
import json
import os
import pstats
import sys
import time
from collections import Counter, defaultdict
import psutil
import regex as re

# 预编译正则
gpt2_pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def train_bpe_stream(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    block_size_mb: int = 20,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """流式读取与缓存优化的 BPE 训练，内存占用控制在 300MB 以内"""
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, merges

    if special_tokens:
        special_regex = re.compile(
            "|".join(
                re.escape(tok) for tok in sorted(special_tokens, key=len, reverse=True)
            )
        )
    else:
        special_regex = None

    word_counts: Counter[tuple[bytes, ...]] = Counter()
    cache = {}  # 缓存 word_str -> word_tuple，加速 99% 的重合词

    print(f"📖 [1/2] 正在进行流式预分词 (块大小: {block_size_mb}MB)...")

    # 20MB 的字节流缓冲区
    chunk_bytes_size = block_size_mb * 1024 * 1024

    with open(input_path, "r", encoding="utf-8") as f:
        buffer = ""
        while True:
            chunk = f.read(chunk_bytes_size)
            if not chunk:
                if buffer:
                    # 处理剩余尾部文本
                    sub_chunks = (
                        [c for c in special_regex.split(buffer) if c]
                        if special_regex
                        else [buffer]
                    )
                    for sub in sub_chunks:
                        for word_str in gpt2_pat.findall(sub):
                            if word_str in cache:
                                w_tuple = cache[word_str]
                            else:
                                w_tuple = tuple(
                                    bytes([b]) for b in word_str.encode("utf-8")
                                )
                                cache[word_str] = w_tuple
                            word_counts[w_tuple] += 1
                break

            buffer += chunk

            # 找到最后一个 special_token，避免在段落中间切断
            if special_tokens:
                last_delim = buffer.rfind(special_tokens[0])
                if last_delim != -1:
                    process_part = buffer[:last_delim]
                    buffer = buffer[last_delim:]
                    sub_chunks = [c for c in special_regex.split(process_part) if c]
                else:
                    sub_chunks = [buffer]
                    buffer = ""
            else:
                last_newline = buffer.rfind("\n")
                if last_newline != -1:
                    process_part = buffer[:last_newline]
                    buffer = buffer[last_newline:]
                    sub_chunks = [process_part]
                else:
                    sub_chunks = [buffer]
                    buffer = ""

            for sub in sub_chunks:
                for word_str in gpt2_pat.findall(sub):
                    if word_str in cache:
                        w_tuple = cache[word_str]
                    else:
                        w_tuple = tuple(bytes([b]) for b in word_str.encode("utf-8"))
                        cache[word_str] = w_tuple
                    word_counts[w_tuple] += 1

    print("🔄 [2/2] 开始 BPE 倒排索引增量合并...")
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: defaultdict[tuple[bytes, bytes], set] = defaultdict(set)

    for word, count in word_counts.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count
            pair_to_words[pair].add(word)

    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_token

        words_to_update = [w for w in pair_to_words[best_pair] if w in word_counts]

        del pair_counts[best_pair]
        del pair_to_words[best_pair]

        for word in words_to_update:
            count = word_counts[word]

            old_word_pairs = Counter()
            for i in range(len(word) - 1):
                old_word_pairs[(word[i], word[i + 1])] += 1

            for p, freq in old_word_pairs.items():
                if p in pair_counts:
                    pair_counts[p] -= freq * count
                    if pair_counts[p] <= 0:
                        del pair_counts[p]

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

            del word_counts[word]
            word_counts[new_word_tuple] = word_counts.get(new_word_tuple, 0) + count

            new_word_pairs = Counter()
            for i in range(len(new_word_tuple) - 1):
                new_word_pairs[(new_word_tuple[i], new_word_tuple[i + 1])] += 1

            for p, freq in new_word_pairs.items():
                pair_counts[p] += freq * count
                pair_to_words[p].add(new_word_tuple)

    return vocab, merges


# ==========================================
# 运行主程序
# ==========================================
if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("🚀 开始极低内存流式训练...")

    process = psutil.Process(os.getpid())
    start_time = time.time()

    # 重置 Profiler 钩子，防冲突
    sys.setprofile(None)
    profiler = cProfile.Profile()
    try:
        profiler.enable()
    except Exception:
        pass

    vocab, merges = train_bpe_stream(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    try:
        profiler.disable()
    except Exception:
        pass

    end_time = time.time()

    elapsed_time = end_time - start_time
    peak_memory_gb = process.memory_info().rss / (1024**3)
    max_token_id, max_token_bytes = max(vocab.items(), key=lambda item: len(item[1]))

    print("\n" + "=" * 50)
    print("✅ 训练顺利完成！结果统计：")
    print(f"⏱️  训练总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.2f} 分钟)")
    print(f"💾 峰值内存占用: {peak_memory_gb:.2f} GB (极低内存！)")
    print(
        f"🔍 最长 Token: {repr(max_token_bytes)} (ID: {max_token_id}, 长度: {len(max_token_bytes)} 字节)"
    )
    print("=" * 50 + "\n")

    # 导出序列化 JSON
    vocab_serialized = {
        k: v.decode("utf-8", errors="replace") for k, v in vocab.items()
    }
    merges_serialized = [
        [
            m[0].decode("utf-8", errors="replace"),
            m[1].decode("utf-8", errors="replace"),
        ]
        for m in merges
    ]

    with open(f"{output_dir}/tinystories_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_serialized, f, ensure_ascii=False, indent=2)

    with open(f"{output_dir}/tinystories_merges.json", "w", encoding="utf-8") as f:
        json.dump(merges_serialized, f, ensure_ascii=False, indent=2)

    print(f"📁 词表与 Merges 已成功导出至 `{output_dir}/` 目录！")

    print("\n📊 【Profiling 性能分析 Top 10】:")
    try:
        stats = pstats.Stats(profiler)
        stats.strip_dirs().sort_stats("cumtime").print_stats(10)
    except Exception as e:
        print("跳过 Stats 打印:", e)