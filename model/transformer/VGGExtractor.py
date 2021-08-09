import numpy as np

import torch
import torch.nn as nn

###############
# MaskCNN(Deep Speech)
###############

class MaskCNN(nn.Module):

    def __init__(self, sequential, args):
        super(MaskCNN, self).__init__()
        self.args = args
        self.sequential = sequential

    def forward(self, inputs, seq_lengths): # todo ## print
        output = None

        for module in self.sequential:
            output = module(inputs)
            ## print("    module: ", module, "=>", output.size())
            mask = torch.BoolTensor(output.size()).fill_(0)
            mask = mask.cuda(self.args.gpu)

            seq_lengths = self._get_sequence_lengths(module, seq_lengths)

            for i, length in enumerate(seq_lengths):
                length = length.item()

                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
                    # 즉 max_length에서 불필요한 뒷 부분을 다 mask

            output = output.masked_fill(mask, 0)
            inputs = output

        return output, seq_lengths

    def _get_sequence_lengths(self, module, seq_lengths):
        # 아래 식은 그저 원래 CNN 걸친 결과 식
        if isinstance(module, nn.Conv2d):
            numerator = seq_lengths + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1
            seq_lengths = numerator.float() / float(module.stride[1])
            seq_lengths = seq_lengths.int() + 1

        elif isinstance(module, nn.MaxPool2d):
            seq_lengths >>= 1 # todo >> 아닌지 확인

        return seq_lengths.int()

############################
# VGG Extractor
############################

class VGGExtractor(nn.Module):

    def __init__(self, args, input_dim=80, in_channels=1, out_channels=(64, 128)):
        self.args = args
        super(VGGExtractor, self).__init__()
        self.input_dim = input_dim # Frame 수가 됨
        self.in_channels = in_channels # 1
        self.out_channels = out_channels # 64, 128

        # todo 얼마나 작아는지 확인
        self.conv = MaskCNN(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                nn.ReLU(),
                nn.Conv2d(out_channels[0], out_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[0]),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                nn.ReLU(),
                nn.Conv2d(out_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(num_features=out_channels[1]),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
            ), args=args
        )

    # todo 의미
    def get_output_dim(self):
        return (self.input_dim - 1) << 5 if self.input_dim % 2 else self.input_dim << 5

    def forward(self, inputs, input_lengths):
        ## print("    [VGG Extractor]")

        outputs, output_lengths = self.conv(inputs.unsqueeze(1).transpose(2, 3), input_lengths)

        batch_size, channels, dimension, seq_lengths = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2) # [batch, seq_lengths, channels, dimension]
        outputs = outputs.view(batch_size, seq_lengths, channels * dimension)
        ## print("    reshape idx[0, 3, 1*2]: ", outputs.size())


        return outputs, output_lengths
