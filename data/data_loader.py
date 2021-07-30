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

    max_seq_sample = max(batch, key=seq_length_)[0] # 값
    max_noisy_seq_sample = max(batch, key=noisy_lenght_)[0]

    max_target_sample = max(batch, key=target_length_)[1] # idx

    max_seq_size = max_seq_sample.size(0)
    noisy_max_seq_size = max_noisy_seq_sample.size(0)
    max_target_size = len(max_target_sample)

    feat_size = max_seq_sample.size(1)
    noisy_feat_size = max_noisy_seq_sample.size(1)

    batch_size = len(batch)

    seqs = torch.zeros(batch_size, max_seq_size, feat_size)
    noisy_seqs = torch.zeros(batch_size, noisy_max_seq_size, noisy_feat_size)
    targets = torch.zeros(batch_size, max_target_size).to(torch.long)
    targets.fill_(pad_token)

    for x in range(batch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        noisy_tensor = sample[2]

        seq_length = tensor.size(0)
        noisy_seq_length = noisy_tensor.size(0)

        seqs[x].narrow(0, 0, seq_length).copy_(tensor)
        noisy_seqs[x].narrow(0, 0, noisy_seq_length).copy_(noisy_tensor)
        targets[x].narrow(0, 0, len(target)).copy_(torch.LongTensor(target))

    seq_lengths = torch.IntTensor(seq_lengths)
    noisy_seq_lengths = torch.IntTensor(noisy_seq_lengths)
    target_lengths = torch.IntTensor(target_lengths)
    # print("targets1: ", targets.size())

    return seqs, targets, seq_lengths, target_lengths, noisy_seqs, noisy_seq_lengths

class AudioDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn
