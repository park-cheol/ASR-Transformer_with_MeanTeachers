import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import init_weight


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dim, dropout_p=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.sqrt_dim = np.sqrt(dim)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, query, key, value, mask=None):
        ## print("    [scaled_Dot_Product]: ")
        score = torch.bmm(query, key.transpose(1, 2)) / self.sqrt_dim
        ## print("      Score(q*k): ", score.size())

        if mask is not None:
            # ## print("score: ", score.size())
            # ## print("mask: ", mask.size())
            score.masked_fill_(mask, -np.inf)

        attn = F.softmax(score, -1)
        ## print("      Sofrtmax[score]: ", attn.size())
        attn = self.dropout(attn)
        context = torch.bmm(attn, value)
        ## print("      Score(q*k)*V: ", context.size())

        return context, attn


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "attention dim = d_model / heads 해야 하기 때문에"

        self.attn_dim = int(d_model / n_heads) # default:64
        self.n_heads = n_heads

        # todo 뒤 사이즈가 조금 다름 원래 attn_dim만 들어갔는데
        # Projection
        self.Linear_Q = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_K = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        self.Linear_V = nn.Linear(d_model, self.attn_dim * n_heads, bias=True)
        init_weight(self.Linear_Q)
        init_weight(self.Linear_K)
        init_weight(self.Linear_V)

        self.scaled_dot_attn = ScaledDotProductAttention(self.attn_dim) # sqrt(d_k)

    def forward(self, q, k, v, mask=None):
        ## print("    [MultiHeadAttention]")
        ## print("      Q: ", q.size())
        ## print("      K: ", k.size())
        ## print("      V: ", v.size())
        batch_size = v.size(0)

        # [Batch, Length, N, D] = [Batch, Length, 8, 64]
        query = self.Linear_Q(q).view(batch_size, -1, self.n_heads, self.attn_dim)
        ## print("    Linear_Q: ", self.Linear_Q, "/Reshaped=>", query.size())
        key = self.Linear_K(k).view(batch_size, -1, self.n_heads, self.attn_dim)
        ## print("    Linear_K: ", self.Linear_K, "/Reshaped=>", key.size())
        value = self.Linear_V(v).view(batch_size, -1, self.n_heads, self.attn_dim)
        ## print("    Linear_V: ", self.Linear_V, "/Reshaped=>", value.size())

        # [Batch * N, Length, Dim]
        query = query.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        ## print("    reshaped_Q: ", query.size())
        key = key.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        ## print("    reshaped_K: ", key.size())
        value = value.permute(2, 0, 1, 3).contiguous().view(batch_size * self.n_heads, -1, self.attn_dim)
        ## print("    reshaped_V: ", value.size())

        # mask
        if mask is not None:
            mask = mask.repeat(self.n_heads, 1, 1)

        context, attn = self.scaled_dot_attn(query, key, value, mask)
        context = context.view(self.n_heads, batch_size, -1, self.attn_dim)
        context = context.permute(1, 2, 0, 3).contiguous().view(batch_size, -1, self.n_heads * self.attn_dim)
        ## print("    outputs: : ", context.size())

        return context, attn



if __name__ == '__main__':
    test = MultiHeadAttention()
    input = torch.randn(16, 512)
    a,b = test(input,input,input)
    ## print(a.size()) # [16, 1, 512]






























