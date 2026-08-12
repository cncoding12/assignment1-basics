import math
import torch
from torch.optim import Optimizer

class AdamW(Optimizer):
    """
    遵循 Algorithm 1 实现的 AdamW 优化器
    """
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW 不支持稀疏梯度")

                state = self.state[p]

                # 1. 状态初始化 (m 和 v 向量)
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)     # m
                    state["exp_avg_sq"] = torch.zeros_like(p)  # v

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 步数 t 从 1 开始计算
                state["step"] += 1
                t = state["step"]

                # 2. 计算偏差修正调整后的 α_t (Line 7)
                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)

                # 3. 施加解耦的 Weight Decay (Line 8): θ = θ - α * λ * θ
                if weight_decay != 0:
                    p.mul_(1.0 - lr * weight_decay)

                # 4. 更新一阶矩 m = β1 * m + (1 - β1) * g (Line 9)
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 5. 更新二阶矩 v = β2 * v + (1 - β2) * g^2 (Line 10)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 6. 参数更新 θ = θ - α_t * m / (sqrt(v) + ε) (Line 11)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-alpha_t)

        return loss