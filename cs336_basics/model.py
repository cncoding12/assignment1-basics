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