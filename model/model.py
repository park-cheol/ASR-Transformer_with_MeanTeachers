import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import MultiHeadAttention
from model.VGGExtractor import VGGExtractor
from model.position import *
from model.mask import *
from model.sublayers import *
from utils import init_weight

################################
# Encoder
################################

class EncoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.self_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff), d_model)

    def forward(self, inputs, mask=None):
        attn_output, attn_map = self.self_attention(inputs, inputs, inputs, mask)
        output = self.feed_forward(attn_output)

        return output, attn_map

class Encoder(nn.Module):

    def __init__(self, args, d_model=512, input_dim=80, d_ff=2048, n_layers=6, n_heads=8, pad_id=0, dropout_p=0.3, max_length=128):
        super(Encoder, self).__init__()

        self.args = args
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_id = pad_id

        self.conv = VGGExtractor(args=args, input_dim=input_dim)
        self.linear = nn.Linear(self.conv.get_output_dim(), d_model) # README 참고 Linear 한번 해주고 나서 Encoder layer 실행
        init_weight(self.linear)

        self.dropout = nn.Dropout(dropout_p)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_length)

        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])

    def forward(self, inputs, inputs_lengths):
        ## print('--[Processing]--')
        ## print("   inputs: ", inputs.size())
        ## print("   input_lengths: ", inputs_lengths.size())

        conv_outputs, output_lengths = self.conv(inputs, inputs_lengths)
        # output_lengths: maskCNN 을 걸친 후에 줄어드는 length 값

        encoder_pad_mask = get_attn_pad_mask(conv_outputs, output_lengths, conv_outputs.size(1))
        # ## print(encoder_pad_mask)
        outputs = self.linear(conv_outputs)
        ## print("    module: ", self.linear, "=>", outputs.size())
        outputs += self.positional_encoding(outputs.size(1))
        ## print("    Sum pos_encoding: ", self.positional_encoding, "=>", outputs.size())
        outputs = self.dropout(outputs)

        ## print('--[Encoder]--')
        for layer in self.layers:
            outputs, attn = layer(outputs, encoder_pad_mask)

        return outputs, output_lengths


################################
# Decoder
################################
class DecoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super(DecoderLayer, self).__init__()

        self.self_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.cross_attention = AddNorm(MultiHeadAttention(d_model, n_heads), d_model)
        self.feed_forward = AddNorm(PositionWiseFeedForwardNet(d_model, d_ff), d_model)

    def forward(self, inputs, encoder_outputs, mask=None, cross_mask=None):
        output, self_attn = self.self_attention(inputs, inputs, inputs, mask)
        output, cross_attn = self.cross_attention(output, encoder_outputs, encoder_outputs, cross_mask)
        output = self.feed_forward(output)
        # ## print("output: ", output.size())
        # ## print("cross_attn: ", cross_attn.size())

        return output, self_attn, cross_attn

class Decoder(nn.Module):

    def __init__(
            self,
            args,
            num_classes,
            d_model=512,
            d_ff=2048,
            n_layers=6,
            n_heads=8,
            dropout_p=0.3,
            pad_id=0,
            sos_id=2001,
            eos_id=2002,
            max_length=128
    ):
        super(Decoder, self).__init__()

        self.args = args
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.embedding = Embedding(num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_length)
        self.dropout = nn.Dropout(dropout_p)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        self.layerNorm = LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes, bias=False)
        init_weight(self.fc)

    def forward_step(
            self,
            decoder_inputs,
            decoder_inputs_lengths,
            encoder_outputs,
            encoder_outputs_lengths,
            positional_encoding_length
    ):
        decoder_pad_mask = get_attn_pad_mask(
            decoder_inputs, decoder_inputs_lengths, decoder_inputs.size(1)
        )
        decoder_regression_mask = get_attn_subsequent_mask(decoder_inputs, self.args)
        # ## print("decoder_pad_mask: ", decoder_pad_mask)
        # ## print("decoder_regression_mask: ", decoder_regression_mask)
        decoder_mask = torch.gt((decoder_regression_mask + decoder_pad_mask), 0)
        # ## print("decoder_mask: ", decoder_mask)
        # gt 는 lt와 반대 lt: input < other / gt: input > output
        # 즉 0 이랑 같거나 작으면 False
        ## print("     decoder_mask: ", decoder_mask)

        encoder_pad_mask = get_attn_pad_mask(
            encoder_outputs, encoder_outputs_lengths, decoder_inputs.size(1)
        )

        embedding = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        ## print("     embedding: ", embedding.size())
        outputs = self.dropout(embedding)

        for layer in self.layers:
            outputs, self_attn, cross_attn = layer(
                outputs, encoder_outputs, decoder_mask, encoder_pad_mask
            )

        # ## print("outputs: ", outputs.size())
        return outputs

    def forward(self, encoder_outputs, encoder_outputs_lengths=None, targets=None, target_lengths=None, teacher_forcing_p=1.0):
        ## print("--[Decoder]--")
        ## print("    encoder_outputs: ", encoder_outputs.size())
        ## print("    encoder_output_lengths: ", encoder_outputs_lengths.size())
        ## print("    targets: ", targets.size())
        ## print("    targe_lengths: ", target_lengths)
        batch_size = encoder_outputs.size(0)
        use_teacher_forcing = True if random.random() < teacher_forcing_p else False

        # teacher forcing
        if targets is not None and use_teacher_forcing:
            targets = targets[targets != self.eos_id].view(batch_size, -1) # eos_id 제외
            # ## print("targets: ", targets.size())
            target_length = targets.size(1) # eos 제외한 real length

            outputs = self.forward_step(
                decoder_inputs=targets,
                decoder_inputs_lengths=target_lengths,
                encoder_outputs=encoder_outputs,
                encoder_outputs_lengths=encoder_outputs_lengths,
                positional_encoding_length=target_length # 딱 여기까지만 pos encoding 함
            )
            ## print("    outputs: ", self.fc, "=>", self.fc(outputs).size())
            return self.fc(outputs).log_softmax(dim=-1)

        # inference 할 때도 사용
        else:
            logits = list()

            input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
            # todo max_length??s
            input_var = input_var.fill_(self.pad_id)
            input_var[:, 0] = self.sos_id

            for di in range(1, self.max_length):
                input_lengths = torch.IntTensor(batch_size).fill_(di)

                outputs = self.forward_step(
                    decoder_inputs=input_var[:, :di],
                    decoder_inputs_lengths=input_lengths,
                    encoder_outputs=encoder_outputs,
                    encoder_outputs_lengths=encoder_outputs_lengths,
                    positional_encoding_length=di
                )

                step_output = self.fc(outputs).log_softmax(dim=-1)

                logits.append(step_output[:, -1, :])
                input_var = logits[-1].topk(1)[1]

            return torch.stack(logits, dim=1)

###################################
# Transformer
###################################
class SpeechTransformer(nn.Module):

    def __init__(self,
                 args,
                 num_classes,
                 d_model=512,
                 input_dim=80,
                 pad_id=0,
                 sos_id=2001,
                 eos_id=2002,
                 d_ff=2048,
                 n_heads=8,
                 n_encoder_layers=6,
                 n_decoder_layers=6,
                 dropout_p=0.3,
                 max_len=128,
                 teacher_forcing_p=1.0):

        super(SpeechTransformer, self).__init__()
        assert d_model % n_heads ==0

        self.num_classes = num_classes
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.max_len = max_len
        self.teacher_forcing_p = teacher_forcing_p

        self.encoder = Encoder(
            args=args,
            d_model=d_model,
            input_dim=input_dim,
            d_ff=d_ff,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            dropout_p=dropout_p,
            pad_id=pad_id,
            max_length=max_len
        )

        self.decoder = Decoder(
            args=args,
            num_classes=num_classes,
            d_model=d_model,
            d_ff=d_ff,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            dropout_p=dropout_p,
            pad_id=pad_id,
            sos_id=sos_id,
            eos_id=eos_id,
            max_length=max_len
        )

    def forward(self, inputs, input_lengths, targets, target_lengths):
        # ## print("target: ",targets.size())
        # ## print("target_lengths: ",target_lengths.size())
        logits = None
        encoder_outputs, encoder_output_lengths = self.encoder(inputs, input_lengths)
        # ## print("encoder_outputs 2: ", encoder_outputs.size())
        # ## print("encoder_outputs 3: ", encoder_output_lengths.size())
        logits = self.decoder(
            encoder_outputs, encoder_output_lengths, targets, target_lengths, self.teacher_forcing_p
        )
        # ## print("logits: ", logits.size())

        return logits

