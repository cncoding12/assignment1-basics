import os
import typing
import torch

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    """
    保存模型检查点到 out (文件路径或文件流)
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    从 src (文件路径或文件流) 加载检查点，恢复 model 与 optimizer 状态，并返回保存时的 iteration
    """
    # 使用 weights_only=False 以允许恢复包含普通 python 字典和整数步数的 checkpoint
    checkpoint = torch.load(src, weights_only=False)

    # 1. 恢复模型权重
    model.load_state_dict(checkpoint["model"])

    # 2. 恢复优化器状态 (如 AdamW 的 m 和 v 动量矩阵)
    optimizer.load_state_dict(checkpoint["optimizer"])

    # 3. 返回训练步数
    return checkpoint["iteration"]