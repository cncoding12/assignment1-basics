import argparse
import math
import os
import time
import numpy as np
import torch

from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import (
    get_lr_cosine_schedule,
    gradient_clipping,
)
from cs336_basics.data import get_batch
from cs336_basics.checkpointing import save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM with CS336 Basics")

    # 1. 数据集与路径配置
    parser.add_argument("--train_data_path", type=str, required=True, help="训练集 .npy/.bin 文件路径")
    parser.add_argument("--val_data_path", type=str, default=None, help="验证集 .npy/.bin 文件路径")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="检查点保存目录")

    # 2. 模型超参数
    parser.add_argument("--vocab_size", type=int, default=10000, help="词表大小")
    parser.add_argument("--context_length", type=int, default=256, help="上下文序列长度")
    parser.add_argument("--d_model", type=int, default=512, help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=6, help="Transformer Block 层数")
    parser.add_argument("--num_heads", type=int, default=8, help="多头注意力头数")
    parser.add_argument("--d_ff", type=int, default=2048, help="SwiGLU FFN 隐藏层维度")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE 位置编码 theta 参数")

    # 3. 训练与优化器超参数
    parser.add_argument("--batch_size", type=int, default=32, help="批大小")
    parser.add_argument("--max_iters", type=int, default=10000, help="总迭代步数")
    parser.add_argument("--max_lr", type=float, default=5e-4, help="最大学习率")
    parser.add_argument("--min_lr", type=float, default=5e-5, help="最小学习率")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="Warmup 预热步数")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="AdamW 权重衰减率")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.95, help="AdamW beta2")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")

    # 4. 日志与评估间隔
    parser.add_argument("--log_interval", type=int, default=100, help="训练日志打印间隔")
    parser.add_argument("--eval_interval", type=int, default=500, help="验证集评估间隔")
    parser.add_argument("--eval_iters", type=int, default=20, help="验证集采样评估批次数")
    parser.add_argument("--save_interval", type=int, default=1000, help="检查点保存间隔")

    # 5. 设备选择
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """在验证集上采样计算平均损失与 Perplexity"""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    mean_loss = sum(losses) / len(losses)
    perplexity = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return mean_loss, perplexity


def train():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"🚀 [CS336] Starting Training on Device: {device}")
    print(f"📦 Model Config: d_model={args.d_model}, layers={args.num_layers}, heads={args.num_heads}, d_ff={args.d_ff}")

    # 1. 内存映射 (np.memmap) 加载训练与验证数据
    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r") if args.val_data_path else None
    print(f"💾 Loaded Train Tokens: {len(train_data):,}")

    # 2. 实例化 TransformerLM 模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.rope_theta,
        device=device,
    )
    model.to(device)

    # 3. 实例化 AdamW 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # 4. 训练主循环
    start_time = time.time()
    for iter_num in range(1, args.max_iters + 1):
        # A. 更新学习率 (Cosine Schedule with Warmup)
        lr = get_lr_cosine_schedule(
            t=iter_num,
            alpha_max=args.max_lr,
            alpha_min=args.min_lr,
            T_w=args.warmup_iters,
            T_c=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # B. 采样数据 Batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # C. 前向传播 + 计算 Cross-Entropy 损失
        logits = model(x)
        loss = cross_entropy(logits, y)

        # D. 反向传播与梯度更新
        optimizer.zero_grad()
        loss.backward()

        # E. 梯度裁剪 (Gradient Clipping)
        if args.grad_clip > 0:
            gradient_clipping(model.parameters(), max_norm=args.grad_clip)

        # F. 优化器 Step
        optimizer.step()

        # G. 打印训练日志 (Train Log)
        if iter_num % args.log_interval == 0:
            elapsed = time.time() - start_time
            train_ppl = math.exp(loss.item()) if loss.item() < 20 else float("inf")
            print(
                f"Step {iter_num:6d}/{args.max_iters} | "
                f"Loss: {loss.item():.4f} | "
                f"PPL: {train_ppl:7.2f} | "
                f"LR: {lr:.2e} | "
                f"Time: {elapsed:.2f}s"
            )
            start_time = time.time()

        # H. 验证集评估 (Validation Eval)
        if val_data is not None and (iter_num % args.eval_interval == 0 or iter_num == args.max_iters):
            val_loss, val_ppl = estimate_loss(
                model, val_data, args.batch_size, args.context_length, device, args.eval_iters
            )
            print(f"📊 [Validation @ Step {iter_num}] Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}")

        # I. 检查点保存 (Save Checkpoint)
        if iter_num % args.save_interval == 0 or iter_num == args.max_iters:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_step_{iter_num}.pt")
            save_checkpoint(model, optimizer, iter_num, ckpt_path)
            print(f"💾 Checkpoint saved to {ckpt_path}")

    print("🎉 Training Completed Successfully!")


if __name__ == "__main__":
    train()