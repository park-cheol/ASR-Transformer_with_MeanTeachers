import numpy as np

import torch
from torch.utils.data import DataLoader, Sampler

def _collate_fn(batch, pad_token=0):
    # print("batch[list 형식]: ", np.array(batch).shape) # (16, 2) 16=batch, 2=Tensor + transcript
    # print("batch[list 형식]: ", batch[1][0].size()) # (161, Freame)

    def seq_length_(p): # todo 용도
        return len(p[0])
    def target_length_(p): # todo 용도
        return len(p[1])
    def noisy_lenght_(p):
        return len(p[2])

    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    seq_lengths = [len(s[0]) for s in batch]
    target_lengths = [len(s[1]) for s in batch]
    # print("target_lengths: ", target_lengths)
    noisy_seq_lengths = [len(s[2]) for s in batch]

    max_seq_size = max(seq_lengths)
    # print("max_seq_size: ", max_seq_size)
    max_target_size = max(target_lengths)
    # print("max_target_size: ", max_target_size)
    max_noisy_seq_size = max(noisy_seq_lengths)

    feat_size = batch[0][0].size(0)
    noisy_size = batch[0][0].size(0)
    # print("feat_size: ", feat_size) # 161 : 1+ n_fft/2
    batch_size = len(batch)
    # print("batch_size: ", batch_size) # 16

    seqs = torch.zeros(batch_size, 1, feat_size, max_seq_size)
    noisy_seqs = torch.zeros(batch_size, 1, noisy_size, feat_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)

    for x in range(batch_size):
        sample = batch[x]
        # print("sample: ", sample)
        tensor = sample[0]
        # print("tensor: ", x, tensor.size()) # [161, Frame]
        target = sample[1]
        noisy_tensor = sample[2]
        # print("target: ", target) : transcript (index 번호들)
        seq_length = tensor.size(1)
        noisy_seq_length = noisy_tensor.size(1)
        # print(tensor.size(1))
        seqs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        noisy_seqs[x][0].narrow(1, 0, noisy_seq_length).copy_(noisy_tensor)
        # print("seq: ", x,seqs[x][0].size()) # [161, length]
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))
        # print("target: ", targets[x].size())

    seq_lengths = torch.IntTensor(seq_lengths)  # [16]
    noisy_seq_lengths = torch.IntTensor(noisy_seq_lengths)  # [16]
    target_lengths = torch.IntTensor(target_lengths)


    return seqs, targets, seq_lengths, target_lengths, noisy_seqs, noisy_seq_lengths

class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
