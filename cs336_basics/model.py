import math
import torch
import torch.nn as nn
from einops import einsum


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 参数 W 形状: (out_features, in_features)
        self.W = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        # 截断正态分布初始化
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(
            self.W,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        使用 einops.einsum，支持使用完整的自解释维度名称
        """
        return einsum(
            x,
            self.W,
            "... in_features, out_features in_features -> ... out_features",
        )
    
class Embedding(nn.Module):
    """
    自定义 Embedding 模块
    根据 Token ID 查找对应的词向量，形状为 (vocab_size, d_model)。
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        
        # 1. 初始化权重矩阵为 nn.Parameter，形状为 (vocab_size, d_model)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
        )
        
        # 2. 权重初始化：使用 trunc_normal_，mean=0.0, std=1.0, a=-3.0, b=3.0
        self.reset_parameters()

    def reset_parameters(self):
        # 按照 CS336 规范，使用截断正态分布初始化
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        输入: token_ids (batch_size, sequence_length) 的 LongTensor
        输出: (batch_size, sequence_length, embedding_dim) 的 FloatTensor
        """
        # 直接通过张量索引（Tensor Indexing）实现 lookup 功能
        return self.weight[token_ids]
    
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    """
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        # 可学习参数 g_i (gain)，形状为 (d_model,)，初始化为全 1
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x 形状: (batch_size, sequence_length, d_model) 或 (..., d_model)
        """
        in_dtype = x.dtype
        # 1. 向上转型为 float32，防止 x^2 溢出
        x_fp32 = x.to(torch.float32)

        # 2. 计算最后一个维度的均方值 (Mean Square): (1/d) * sum(a_i^2)
        variance = x_fp32.pow(2).mean(dim=-1, keepdim=True)

        # 3. 计算 1 / RMS(a) = 1 / sqrt(variance + eps)
        # 使用 torch.rsqrt 性能更好
        rms_inv = torch.rsqrt(variance + self.eps)

        # 4. 归一化并乘以 gain 参数 self.weight
        normed_x = x_fp32 * rms_inv
        result = normed_x * self.weight

        # 5. 转回输入张量原始的 dtype 并返回
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. 计算 d_ff：若未显式传入，取 8/3 * d_model 并向上对齐到 64 的倍数 (LLaMA 官方对齐逻辑)
        if d_ff is None:
            raw_d_ff = int(2 * 4 * d_model / 3)
            self.d_ff = int((raw_d_ff + 63) // 64 * 64)
        else:
            self.d_ff = d_ff

        factory_kwargs = {"device": device, "dtype": dtype}

        # 2. 声明 3 个线性层（必须透传 device 和 dtype！）
        self.w1 = Linear(in_features=self.d_model, out_features=self.d_ff, **factory_kwargs)  # Gate
        self.w3 = Linear(in_features=self.d_model, out_features=self.d_ff, **factory_kwargs)  # Up
        self.w2 = Linear(in_features=self.d_ff, out_features=self.d_model, **factory_kwargs)  # Down

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        # 讲义要求：使用 torch.sigmoid 以保证数值稳定性
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Gate 分支: SiLU(W1 * x)
        gate = self.silu(self.w1(x))

        # 2. Up 分支: W3 * x
        up = self.w3(x)

        # 3. 门控点乘: gate ⊙ up
        hidden = gate * up

        # 4. Down 降维输出: W2 * hidden
        return self.w2(hidden)

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将相邻的两两维度 (x0, x1, x2, x3, ...) 变换为 (-x1, x0, -x3, x2, ...)
    支持任意 Batch 维度 (..., seq_len, d_k)
    """
    x_even = x[..., 0::2]  # x0, x2, ...
    x_odd  = x[..., 1::2]  # x1, x3, ...
    
    # 组合为 [-x1, x0], [-x3, x2], ...
    stacked = torch.stack((-x_odd, x_even), dim=-1)
    
    # 展开最后两个维度 (d_k // 2, 2) -> d_k
    return stacked.flatten(-2)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    """
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # 1. 计算频率 freqs: theta ** (-2 * (k-1) / d_k) for k in 1..d_k/2
        # arange(0, d_k, 2) -> 0, 2, 4, ..., d_k-2
        freqs = 1.0 / (
            theta ** (torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        )  # 形状: (d_k // 2,)

        # 2. 生成所有位置 pos: 0, 1, ..., max_seq_len - 1
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)  # (max_seq_len,)

        # 3. 外积得到每个位置每个通道的角度 theta_{i, k}
        # angles 形状: (max_seq_len, d_k // 2)
        angles = torch.outer(positions, freqs)

        # 4. 计算 cos 和 sin，并交错复制扩展到 d_k 维
        # cos_cached / sin_cached 形状: (max_seq_len, d_k)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        
        # 将每个角度重复两次，对应 (x0, x1) 使用同一个角度 theta_{i, k}
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)

        # 5. 注册为不需要梯度且非持久化 (persistent=False) 的 Buffer
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len) - 每个 token 的绝对位置编号
        """
        # 利用 PyTorch 高级索引直接切片: self.cos[token_positions]
        # 如果 token_positions 形状为 (..., seq_len)，切片后形状自动变为 (..., seq_len, d_k)
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        # RoPE 旋转计算公式: x * cos + rotate_half(x) * sin
        return (x * cos) + (_rotate_half(x) * sin)