import torch
import torch.nn as nn


# 원래 LayerNorm 식 구현
class LayerNorm(nn.Module):

    def __init__(self, dim, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, inputs):
        mean = inputs.mean(dim=-1, keepdim=True)
        # keepdim = dim 유지하지만 1로 바뀜
        # e.g) Tensor[3, 3, 4] --> Tensor[3, 3, 1]
        std = inputs.std(dim=-1, keepdim=True)

        output = (inputs - mean) / (std + self.eps)
        output = self.gamma * output + self.beta

        return output
























