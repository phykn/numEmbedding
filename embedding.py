import torch
import torch.nn as nn
from einops import rearrange, repeat

class numEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nan = nn.Parameter(torch.randn(embedding_dim))
        self.linear = nn.Conv1d(1, embedding_dim, 1)

    def forward(self, x):
        b, n = x.shape

        mask = torch.isnan(x).float()
        mask = repeat(mask, "b n -> b c n", c = self.embedding_dim)

        x = torch.nan_to_num(x)
        x = rearrange(x, "b n -> b 1 n")
        x = self.linear(x)

        nan = repeat(self.nan, "c -> b c n", b = b, n = n)
        return x * (1 - mask) + nan * mask