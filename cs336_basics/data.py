import numpy as np
import torch

def get_batch(
    x: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 1D Token 数组中采样一个 Batch 的训练输入和目标序列
    x: 一维 Token ID 数组 (可以是普通 np.ndarray 或 np.memmap)
    batch_size: 批大小 B
    context_length: 序列长度 m
    device: PyTorch 设备 ('cpu', 'cuda:0', 'mps' 等)
    """
    # 1. 确定起始索引 i 的合法最大边界
    high = len(x) - context_length
    
    # 2. 随机采样 B 个起始位置索引
    starting_indices = np.random.randint(0, high, size=batch_size)

    # 3. 构造 Input 和 Target 序列 (利用列表推导式与 np.stack)
    inputs = np.stack([x[i : i + context_length] for i in starting_indices])
    targets = np.stack([x[i + 1 : i + context_length + 1] for i in starting_indices])

    # 4. 转换成 PyTorch LongTensor (int64)，并移动到指定设备
    inputs_tensor = torch.tensor(inputs, dtype=torch.long, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.long, device=device)

    return inputs_tensor, targets_tensor