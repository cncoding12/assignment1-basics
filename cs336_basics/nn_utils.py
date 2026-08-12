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