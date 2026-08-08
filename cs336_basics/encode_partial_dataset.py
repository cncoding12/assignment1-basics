import os
import time
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def encode_partial_dataset(
    tokenizer, input_path, output_npy_path, max_bytes_mb=50
):
    """
    仅截取文本的前 max_bytes_mb (兆字节) 进行编码保存
    """
    print(
        f"📦 正在截取 `{input_path}` 的前 {max_bytes_mb} MB 数据并进行编码..."
    )
    start_t = time.time()

    max_bytes = max_bytes_mb * 1024 * 1024

    # 仅读取前 max_bytes 字节
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read(max_bytes)

    # 编码为 Token ID
    all_token_ids = tokenizer.encode(text)

    # 转换为 uint16 NumPy 数组并保存
    token_array = np.array(all_token_ids, dtype=np.uint16)
    np.save(output_npy_path, token_array)

    end_t = time.time()
    file_size_mb = os.path.getsize(output_npy_path) / (1024**2)

    print(
        f"✅ 截取编码完成！"
        f"\n  - 截取文本大小: {len(text.encode('utf-8')) / (1024**2):.2f} MB"
        f"\n  - 生成 Token 数量: {len(token_array):,} 个"
        f"\n  - 耗时: {end_t - start_t:.2f} 秒"
        f"\n  - 保存至: `{output_npy_path}` (文件大小: {file_size_mb:.2f} MB)\n"
    )


if __name__ == "__main__":
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 TinyStories (10K) Tokenizer
    print("📖 加载 TinyStories Tokenizer...")
    ts_tokenizer = Tokenizer.from_files(
        f"{output_dir}/tinystories_vocab.json",
        f"{output_dir}/tinystories_merges.json",
        special_tokens=["<|endoftext|>"],
    )

    # 2. 仅截取前 50 MB 训练集数据导出（可以自行调整 max_bytes_mb，比如设为 20 或 50）
    ts_train_path = "data/TinyStoriesV2-GPT4-train.txt"

    if os.path.exists(ts_train_path):
        encode_partial_dataset(
            tokenizer=ts_tokenizer,
            input_path=ts_train_path,
            output_npy_path=f"{output_dir}/tinystories_train_50mb_uint16.npy",
            max_bytes_mb=50,  # 👈 这里可以自由修改截取的大小 (单位: MB)
        )