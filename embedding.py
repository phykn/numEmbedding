import torch
import torch.nn as nn
from einops import rearrange, repeat

class numEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nan = nn.Parameter(torch.randn(embedding_dim))
        self.linear = nn.Linear(1, embedding_dim)

    def forward(self, x):
        b, n = x.shape

        mask = torch.isnan(x).float()
        mask = repeat(mask, "b n -> b n d", d = self.embedding_dim)

        x = torch.nan_to_num(x)
        x = rearrange(x, "b n -> b n 1")
        x = self.linear(x)

        nan = repeat(self.nan, "d -> b n d", b = b, n = n)
        return x * (1 - mask) + nan * mask