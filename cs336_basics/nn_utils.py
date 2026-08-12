import math

def get_lr_cosine_schedule(
    t: int,
    alpha_max: float,
    alpha_min: float,
    T_w: int,
    T_c: int,
) -> float:
    """
    带预热的余弦学习率调度器
    t: 当前迭代步数
    alpha_max: 最大学习率
    alpha_min: 最小学习率
    T_w: Warmup 阶段步数
    T_c: Cosine 衰减结束的总步数
    """
    # 1. Warm-up 预热阶段
    if t < T_w:
        return (t / T_w) * alpha_max

    # 2. Cosine annealing 余弦衰减阶段
    if t <= T_c:
        # progress 范围在 [0, 1]
        progress = (t - T_w) / (T_c - T_w)
        return alpha_min + 0.5 * (1.0 + math.cos(math.pi * progress)) * (alpha_max - alpha_min)

    # 3. Post-annealing 衰减结束阶段，保持最小学习率
    return alpha_min


from typing import Iterable
import torch

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> float:
    """
    梯度裁剪 (In-place)
    parameters: 包含参数的迭代器
    max_norm: 最大允许的 L2 范数 M
    eps: 数值稳定小常数，默认 1e-6
    """
    # 筛选出含有梯度的参数
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    # 1. 计算所有参数梯度的全局 L2 范数
    total_norm_sq = 0.0
    for p in params:
        param_norm = torch.norm(p.grad.detach(), 2)
        total_norm_sq += param_norm.item() ** 2
    
    total_norm = total_norm_sq ** 0.5

    # 2. 若超过 max_norm，计算缩放因子并原地修改梯度 (In-place)
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in params:
            p.grad.detach().mul_(clip_coef)

    return total_norm