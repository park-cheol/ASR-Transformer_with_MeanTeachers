import math

import torch
import torch.nn as nn

class MaskConv(nn.Module):

    def __init__(self, seq_module, args):
        super(MaskConv, self).__init__()
        self.args = args
        self.seq_module = seq_module
        # nn.Sequential(
        #     nn.Conv2d(1, outputs_channel, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
        #     nn.BatchNorm2d(outputs_channel),
        #     nn.Hardtanh(0, 20, inplace=True), # paper) min(max(x, 0), 20)
        #     nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
        #     nn.BatchNorm2d(outputs_channel),
        #     nn.Hardtanh(0, 20, inplace=True)

    def forward(self, input, lengths):
        """
        Adds padding to the output of the module based on the given lengths..
        This is to ensure that
        the results of the model do not change when batch sizes change during inference.
        :param input: shape [B, C, D, T]
        :param lengths: batch에서 각각의 sequence 실제 길이 # 각 input_lengths에서 conv 계산값입힌 후의 lengths
        :return: module 로부터 masked output
        """
        # print("   [MaskConv]  ")
        # print("     {")
        for module in self.seq_module:
            input = module(input)
            # print("      ", module, " => ", input.size())
            mask = torch.BoolTensor(input.size()).fill_(0) # 같은 사이즈로 모두 False로 선언
            # print("      ", "Mask 생성", mask.size())

            if input.is_cuda: # cuda 인지 확인
                mask = mask.cuda(self.args.gpu) # mask 도 gpu로 올려줌

            for i, length in enumerate(lengths):
                # print("      ", lengths.size())
                length = length.item()
                # print("      ", "length: ", length)
                # 682 681 677 669 665 660 658 .... 100 총 batch_size 만큼 반환
                if (mask[i].size(2) - length) > 0:
                    # print(mask[i].size(2)) = Frame = T
                    # print("A", mask[i].size(2))
                    # print("B", length)
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
                    # torch.narrow(input, dim, start, length) → Tensor
                    # input 의 좁아진 verson을 반환
                    # dim: narrow 할 dim , start 부터 start + length(포함x) 까지 출력

            # print("      ", "Mask: ", mask)
            """
            batch 안에서 최대 Length 를 기준으로 mask 생성함으로 짧은 Length 를 가진 것들은
            불필요한 값들을 가짐 그걸 삭제해버림
            """
            input = input.masked_fill(mask, 0)
            # print("      ", "MasK 적용", input.size())
            # print("      ", "결과 ", input)
            # masked_fill(mask, value): bool 이 True 곳에 mask, value: 채울 값
        # print("     }")
        return input, lengths
