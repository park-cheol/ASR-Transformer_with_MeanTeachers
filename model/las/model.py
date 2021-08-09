import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from model.las.maskcnn import *
###################
# Encoder
###################
class EncoderRNN(nn.Module):

    def __init__(self, args, input_size, hidden_size, n_layers=1, dropout_p=0,
                bidirectional=False, rnn_cell='gru', variable_lengths=False):
        """
        Default: input_size = 161 / hidden_size = 512 / n_layers = 3 / dropout_p = 0.3
                 bidirectional = True / rnn_cell = 'lstm' / variable_lengths=False
        """
        super(EncoderRNN, self).__init__()
        self.args = args
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.variable_lengths = variable_lengths

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        outputs_channel = 32

        # deep speech 2 채택
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, outputs_channel, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True), # paper) min(max(x, 0), 20)
            nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True)
        ), self.args)

        # size : (input_size + 2 * pad - filter) / stride + 1
        rnn_input_dims = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        # print("input_dims: ", rnn_input_dims) # 81
        rnn_input_dims = int(math.floor(rnn_input_dims + 2 * 10 - 21) / 2 + 1)
        # print("input_dims2: ", rnn_input_dims) # 41
        rnn_input_dims *= outputs_channel
        # print("input_dims3: ", rnn_input_dims) # 1312


        self.rnn = self.rnn_cell(rnn_input_dims, self.hidden_size, self.n_layers, dropout=self.dropout_p,
                                 bidirectional=self.bidirectional)

    def forward(self, input, input_lengths=None):
        """
        :param input: Spectrogram shape(B, 1, D, T) = (batch, 1, n_fft/2 +1= frequency bin, Frame)
        :param input_lengths: zero-pad 적용 되지 않은 inputs sequence length
        """
        # print("--[Encoder]-- ")
        # print("\tEncoder Argument")
        # print("\t{ input(spectrogram): ", input.size(), ",") # [16, 1, 161, Frame]
        # print("\t  input_lengths: ", input_lengths.size(), "}") # [16]

        output_lengths = self.get_seq_lens(input_lengths)
        # #("  output_lengths: ", output_lengths.size()) # [16]: 각 다
        # tensor([1409, 1409, 1397, 1396, 1391, 1387, 1386, 1386, 1385, 1383, 1369, 1364,
        #         1356, 1353, 1341, 1333]

        x = input # (B, 1, D, T) = (16, 1, 161, Frame)

        conv_output, _ = self.conv(x, output_lengths) # (B, C, D, T)
        # print("  conv_output: ", conv_output.size()) # [16, 32, 41, Frame / conv]

        conv_output_size = conv_output.size()
        conv_output_reshape = conv_output.view(conv_output_size[0], conv_output_size[1] * conv_output_size[2],
                                               conv_output_size[3]) # (B, C * D, T)
        # print("  reshape(B, C*D, Frame): ", conv_output_reshape.size())
        # rnn 에 넣기 위한 reshape
        output_permute = conv_output_reshape.permute(0, 2, 1).contiguous() # (B, T, D)
        # print("  permute(B, Frame, C*D): ", output_permute.size())

        total_length = conv_output_size[3] # T

        # 패딩된 문장을 패킹(패딩은 연산 안들어가게)
        # packed: B * T, E
        pack_padded_sequence = nn.utils.rnn.pack_padded_sequence(output_permute,
                                              output_lengths.cpu(), # 각각 batch 요소들의 sequence length 의 list
                                              batch_first=True,
                                              enforce_sorted=False) # True이면 감소하는 방향으로 정렬
        # print("  pack_padded_sequence: ", pack_padded_sequence[0].size())
        # 패딩이 제거된 [batch 안에 있는 모든 length, C*D]

        rnn_output, h_state = self.rnn(pack_padded_sequence)
        # print("  RNN_output: ", rnn_output[0].size())
        # [batch 안에 있는 모든 length, 1024 = bidirectional (2) x hidden_dim (512) ]

        # print("  h_state : ", h_state)

        # 다시 패킹된 문장을 pad
        # unpacked: B, T, H
        pad_packed_sequence, _ = nn.utils.rnn.pad_packed_sequence(rnn_output,
                                                batch_first=True,
                                                total_length=total_length)
        # print("  pad_packed_sequence: ", pad_packed_sequence.size())
        # [16, 849, 1024]
        # print("\n")

        return pad_packed_sequence, h_state

    def get_seq_lens(self, input_length):
        seq_len = input_length

        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)

        return seq_len.int()


#####################
# Attention
#####################
class Attention(nn.Module):
    """
    Location-based
    자세한것:
    https://arxiv.org/pdf/1506.07503.pdf
    https://arxiv.org/pdf/1508.01211.pdf
    """
    def __init__(self, dec_dim, enc_dim, conv_dim, attn_dim, smoothing=False):
        super(Attention, self).__init__()
        self.dec_dim = dec_dim # 512
        self.enc_dim = enc_dim # 1024
        self.conv_dim = conv_dim # 1
        self.attn_dim = attn_dim # 512
        self.smoothing = smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.attn_dim, kernel_size=3, padding=1)

        self.W = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.V = nn.Linear(self.enc_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.randn(attn_dim))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.mask = None

    def set_mask(self, mask):
        """
        mask 지정
        """
        self.mask = mask

    def forward(self, queries, values, last_attn):
        """
        :param queries: Decoder hidden state (s_i), shape=(B, 1, dec_D)
        :param values: Encoder output (h_i), shape=(B, enc_T, enc_D)
        :param last_attn: 이전 step 의 weight Attention, shape=(batch, enc_T)
        """
        # Q: RNN output
        # V: encoder outputs

        # print("   [Attention]  ")
        # print("      {")
        # print("        last_attention: ", last_attn)
        conv_attn = torch.transpose(self.conv(last_attn.unsqueeze(dim=1)), 1, 2) # [B, enc_T, conv_D]
        # print("        conv_attn: ", conv_attn.size())
        # paper 내용참고
        score = self.fc(self.tanh(
            self.W(queries) + self.V(values) + conv_attn + self.b
        )).squeeze(dim=-1) # [B, enc_T]
        #("        attn_weight: ", score.size())

        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float('inf'))

        # attn_weight : (B, enc_T)
        if self.smoothing:
            score = torch.sigmoid(score)
            attn_weight = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
            # div(x,y) = x/y
        else:
            attn_weight = self.softmax(score)
            # print("        softmax(attn_weight): ", attn_weight.size())
        # (B, 1, enc_T) * (B, enc_T, enc_D) -> (B, 1, enc_D)
        context = torch.bmm(attn_weight.unsqueeze(dim=1), values)
        # print("        softmax(attn_weight)*V: ", context.size())
        # print("        context = softmax(attn_weight)*V")
        # print("      }")

        # C_i = context

        return context, attn_weight

#####################
# Decoder
#####################
class DecoderRNN(nn.Module):

    def __init__(self, args, vocab_size, max_len, hidden_size, encoder_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru',
                 bidirectional_encoder=False, bidirectional_decoder=False,
                 dropout_p=0, use_attention=True):
        super(DecoderRNN, self).__init__()

        self.args = args
        self.output_size = vocab_size
        self.vocab_size = vocab_size # len(char2index) = 2003
        self.hidden_size = hidden_size # 512
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_output_size = encoder_size * 2 if self.bidirectional_encoder else encoder_size
        # encoder_size 512
        self.n_layers = n_layers # 2
        self.dropout_p = dropout_p # 0.3
        self.max_length = max_len # 80
        self.use_attention = use_attention
        self.sos_id = sos_id # 2001
        self.eos_id = eos_id # 2002

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.init_input = None
        self.rnn = self.rnn_cell(self.hidden_size + self.encoder_output_size, self.hidden_size,
                                 self.n_layers, batch_first=True, dropout=dropout_p,
                                 bidirectional=self.bidirectional_decoder)
        # hidden_size = 512 / encoder_output_size = 1024 / n_layers = 2 / dropout_p = 0.3
        # S_i = RNN(s_i-1, y_i-1, c_i-1)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        # vocab_size = 512
        self.input_dropout = nn.Dropout(self.dropout_p)

        self.attention = Attention(dec_dim=self.hidden_size, enc_dim=self.encoder_output_size,
                                   conv_dim=1, attn_dim=self.hidden_size)

        self.fc = nn.Linear(self.hidden_size + self.encoder_output_size, self.output_size)

    def forward_step(self, input, hidden, encoder_outputs, context, attn_w, function):
        # hidden = None default
        batch_size = input.size(0)
        # print("  batch_size: ", batch_size) # 16
        dec_len = input.size(1)
        # print("  den_length: ", dec_len) # Max length - 1

        embedded = self.embedding(input) # (B, dec_T, voc_D) -> (B, dec_T, dec_D)
        # print("  embedding: ", embedded.size()) # [16, dec_length, vocab_shape]
        embedded = self.input_dropout(embedded)

        y_all = []
        attn_w_all = []

        for i in range(embedded.size(1)): # dec_length 만큼 반복
            # print("  -- Max Length 반복", embedded.size(1), " --")
            embedded_inputs = embedded[:, i, :] # (B, dec_D)

            rnn_input = torch.cat([embedded_inputs, context], dim=1) # (B, dec_D + enc_D)
            # print("  concat[embed, context]: ", rnn_input.size())
            # print("  context: ", context.size()) # [16, encoder_output: 1024]
            rnn_input = rnn_input.unsqueeze(1)
            # print("  rnn_input: ", rnn_input.size())
            output, hidden = self.rnn(rnn_input, hidden) # (B, 1, dec_D)
            # print("  rnn_output: ", output.size())

            # queries, values, last_attn
            # attn_w = [16, length]
            context, attn_w = self.attention(output, encoder_outputs, attn_w)
            # C_i=[B, 1, enc_D], a_i=[B, enc_T]
            attn_w_all.append(attn_w)

            context = context.squeeze(1)
            output = output.squeeze(1) # [B, 1, dec_D] -> [B, dec_D]
            context = self.input_dropout(context)
            output = self.input_dropout(output)
            output = torch.cat((output, context), dim=1) # [B, dec_D + enc_D]
            # print("  cat(rnn_output, context): ", output.size())

            pred = function(self.fc(output), dim=-1)
            # print("  F.log_softmax(Linear): ", pred.size())
            y_all.append(pred)
            # print("  y_all: ", y_all) # tensor pred 을 차곡 쌓음
        if embedded.size(1) != 1:
            y_all = torch.stack(y_all, dim=1) # (B, dec_T, out_D)
            # print("  decoder_output: ", y_all.size()) # [16, length, 2003= index 총 output]
            # torch.stack: cat가 다르게 차원을 확장하여 Tensor 쌓음
            # e.g) [M, N, K] satack [N, N, K] -> [M, 2, N, K]
            attn_w_all = torch.stack(attn_w_all, dim=1)  # (B, dec_T, enc_T)
            # print("  attn_w_all: ", attn_w_all.size()) # [16, dec_length, encoder length ]
        else:
            y_all = y_all[0].unsqueeze(1) # (B, 1, out_D)
            attn_w_all = attn_w_all[0] # [B, 1, enc_T]

        return y_all, hidden, context, attn_w_all

    def forward(self, targets=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0):
        """
        :param inputs: [B, dec_T]
        :param encoder_hidden: Encoder last hidden states
        :param encoder_outputs: Encdoer output, [B, enc_T, enc_D]
        """
        # 지정한 확률로 teacher forcing 사용
        # print("--[Decoder]-- ")
        # print("\tDecoder Argument")
        # print("\t{ targets(transcript)[batch, Length]: ", targets.size(), ",")
        # print("\t  encoder_outputs[batch, encoder length, dim]: ", encoder_outputs.size(), "}")
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if teacher_forcing_ratio != 0:
            targets, batch_size, max_length = self._validate_args(targets, encoder_hidden, encoder_outputs,
                                                         function, teacher_forcing_ratio)

        else:
            batch_size = encoder_outputs.size(0) # 16

            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            targets = targets.cuda(self.args.gpu)
            max_length = self.max_length

        decoder_hidden = None
        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2))  # (B, D)
        # print("  context 생성: ", context.size())
        # print("  context: ", context.size()) # [16, 1024]
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1))  # (B, T)
        # print("  attn_w: ", attn_w.size()) # [16, 682=length]

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size) # max_length 계속 변함


        def decode(step, step_output):
            decoder_outputs.append(step_output) # decoder output length 1개씩 넣음 [batch, output: 2003]
            # [tensor[....], tensor[....]] 형식
            symbols = decoder_outputs[-1].topk(1)[1] # 그중 값이 제일 큰 값 Index 가져옴
            # print("  symboles: ", symbols)
            sequence_symbols.append(symbols) # List에 Index 담아넣음

            # print(  "sequence_symbols: ", np.array(sequence_symbols).shape)
            eos_batches = symbols.data.eq(self.eos_id)
            # print(  "eos_batches: ", eos_batches)
            if eos_batches.dim() > 0: # 2
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols


        if use_teacher_forcing:
            # print("  < Use Teacher Forcing >")
            decoder_input = targets[:, :-1]
            # print("  decoder_input: ", decoder_input.size()) # [batch, lenght -1] eos 제거
            decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input,
                                                                                decoder_hidden,
                                                                                encoder_outputs,
                                                                                context,
                                                                                attn_w,
                                                                                function=function)

            for di in range(decoder_output.size(1)): # decoder length 만큼
                step_output = decoder_output[:, di, :] # 한 값씩 투입
                decode(di, step_output)

        else:
            decoder_input = targets[:, 0].unsqueeze(1)
            # print("  decoder_input: ", decoder_input.size())
            for di in range(max_length):
                decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input,
                                                                                    decoder_hidden,
                                                                                    encoder_outputs,
                                                                                    context,
                                                                                    attn_w,
                                                                                    function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output)
                decoder_input = symbols

        # print("  seqeuence_symbols:  ", sequence_symbols)
        return decoder_outputs



    def _validate_args(self, targets, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)

        if targets is None:
            if teacher_forcing_ratio > 0: # teacher focing 쓸려면 target이 있어야함
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            # 모든 batch 앞에 sos_id 추가
            targets = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = targets.cuda(self.args.gpu)
            max_length = self.max_length
        else:
            max_length = targets.size(1) - 1  # sos index 삭제

        return targets, batch_size, max_length



#################
# Seq2Seq
#################
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        pass

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        """
        :param input_variable: feats = seqs
        :param input_lengths: feat_lengths = seq_lengths
        :param target_variable: scripts = targets
        """
        # print("input_variable", input_variable.size())
        # print("input_lengths", input_lengths.size())
        # print("target_variable", target_variable.size())


        self.encoder.rnn.flatten_parameters()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        # print("encoder_outputs: ", encoder_outputs.size())
        # print("target_variable: ", target_variable) = [batch, length]
        self.decoder.rnn.flatten_parameters()
        decoder_output = self.decoder(targets=target_variable,
                                      encoder_hidden=None,
                                      encoder_outputs=encoder_outputs,
                                      function=self.decode_function,
                                      teacher_forcing_ratio=teacher_forcing_ratio)

        return decoder_output

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params