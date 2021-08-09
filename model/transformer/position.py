import math

import numpy as np
import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# Transformer: 'Attention is all you need' 참고

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=512, max_len=128):
        super(PositionalEncoding, self).__init__()

        PE = torch.zeros(max_len, d_model, requires_grad=False)
        """
        PE_(pos, 2i)    =  sin(pos / power(10000, 2i / d_model))
        PE_(pos, 2i+1)  =  cos(pos / power(10000, 2i / d_model))
        """
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 밑에 div_term은 원본 논문과 다름
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        # print(pos.size()) # default:[5000, 1]
        # print(div_term.size()) # default: [256]
        # print((pos * div_term).size()) # default: [5000, 256]
        # plt.figure()
        # plt.plot(div_term * pos)
        # plt.show()

        PE[:, 0::2] = torch.sin(pos * div_term)
        PE[:, 1::2] = torch.cos(pos * div_term)

        # fig, ax = plt.subplots(figsize=(100, 100))
        # cax = ax.matshow(PE)
        # plt.colorbar(cax)
        # plt.xlabel("d_model")
        # plt.ylabel("length")
        # plt.show()

        PE = PE.unsqueeze(0)
        self.register_buffer('PE', PE)

    def forward(self, length):
        # todo input.size(1) 아니면 length
        return self.PE[:, :length]

class Embedding(nn.Module):
    """
    Decoder Input 은 음성이 아닌 Transcript 이니까 그 때 사용
    """
    def __init__(self, num_embeddings, pad_id, d_model=512):
        super(Embedding, self).__init__()
        self.sqrt_dim = math.sqrt(d_model)
        self.embedding = nn.Embedding(num_embeddings, d_model, padding_idx=pad_id)

    def forward(self, inputs):
        # Transformer Paper) we multiply those weights by sqrt(d_model)
        return self.embedding(inputs) * self.sqrt_dim














