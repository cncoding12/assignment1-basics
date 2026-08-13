import numpy as np
import os

def split_npy_dataset(
    input_npy_path: str,
    train_output_path: str,
    val_output_path: str,
    val_ratio: float = 0.05,  # 5% 作为验证集，95% 作为训练集
):
    print(f"📖 正在加载已有数据集 `{input_npy_path}` ...")
    data = np.load(input_npy_path)
    total_tokens = len(data)

    val_size = int(total_tokens * val_ratio)
    train_size = total_tokens - val_size

    train_data = data[:train_size]
    val_data = data[train_size:]

    np.save(train_output_path, train_data)
    np.save(val_output_path, val_data)

    print("✅ 数据集划分完成！")
    print(f"  - 总 Token 数量: {total_tokens:,}")
    print(f"  - 训练集 Token 数: {len(train_data):,} ({len(train_data)/total_tokens*100:.1f}%) -> Saved to `{train_output_path}`")
    print(f"  - 验证集 Token 数: {len(val_data):,} ({len(val_data)/total_tokens*100:.1f}%) -> Saved to `{val_output_path}`")

if __name__ == "__main__":
    split_npy_dataset(
        input_npy_path="results/tinystories_train_50mb_uint16.npy",
        train_output_path="results/tinystories_train_split.npy",
        val_output_path="results/tinystories_val_split.npy",
        val_ratio=0.05, # 划分 5%（约 2.5MB 数据）给验证集
    )