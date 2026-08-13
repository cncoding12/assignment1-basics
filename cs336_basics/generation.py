import torch
import torch.nn as nn
from cs336_basics.model import softmax

def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """
    单步下一个 Token 采样函数
    logits: (batch_size, vocab_size)
    temperature: 采样温度 (<= 0 即为 Argmax 贪婪搜索)
    top_p: Nucleus 采样概率阈值
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)  # (1, vocab_size)

    # 1. 贪婪搜索 (Greedy Argmax)
    if temperature <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    # 2. Temperature Scaling
    logits = logits / temperature
    probs = softmax(logits, dim=-1)

    # 3. Top-p (Nucleus) Sampling
    if top_p < 1.0:
        # 按概率降序排列
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)

        # 标记需移除的 Token：累积概率超过 top_p 的位置（保持突破 top_p 阈值的第一个 Token 被保留）
        sorted_indices_to_remove = cum_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # 概率清零并重新归一化
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)

        # 从截断后的分布中采样
        next_token_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_indices, dim=-1, index=next_token_sorted_idx)
    else:
        # 标准多项式采样
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def generate(
    model: nn.Module,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    完整自回归文本生成主函数
    prompt_tokens: (batch_size, seq_len) 或 (seq_len,)
    """
    model.eval()
    if prompt_tokens.ndim == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)  # (1, seq_len)

    generated = prompt_tokens.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 取最近 context_length 长度的输入，防止滑动窗口超限
            context = generated
            if hasattr(model, "context_length") and context.size(1) > model.context_length:
                context = context[:, -model.context_length :]

            # 模型前向传播拿到 Logits
            logits = model(context)  # (B, S, vocab_size)

            # 提取最后一个预测位置的 Logits
            next_logits = logits[:, -1, :]  # (B, vocab_size)

            # 采样下一个 Token
            next_token = sample_next_token(next_logits, temperature=temperature, top_p=top_p)

            # 拼接生成的 Token 到序列末尾
            generated = torch.cat([generated, next_token], dim=1)

            # 若采样到了 <|endoftext|> 并且批大小为 1，提前停止
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

    return generated