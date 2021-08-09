import os
import json
import math
import random
import argparse
import numpy as np
import warnings
import os
import time
import datetime

from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn


import Levenshtein as Lev
# Levenshtein distance: 두 시퀀스 간의 차이를 측정하기 위한 문자열 메트릭

from data.vocab import *
from data.data_loader import AudioDataLoader
from data.sampler import BucketingSampler
from data.dataset import MelFilterBankDataset

from model.transformer.model import SpeechTransformer
from model.las.model import *
from utils import *

parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('--train-file', type=str,
                    help='data list about train dataset', default='dataset/train.json')
parser.add_argument('--test-file-list',
                    help='data list about test dataset', default=['dataset/test.json'])
parser.add_argument('--labels-path', default='dataset/kor_syllable.json', help='Contains large characters over korean')
parser.add_argument('--dataset-path', default='dataset/clovacall', help='Target dataset path')
# Hyperparameters
parser.add_argument('--rnn-type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--encoder-layers', type=int, default=3, help='number of layers of model (default: 3)')
parser.add_argument('--encoder-size', type=int, default=512, help='hidden size of model (default: 512)')
parser.add_argument('--decoder-layers', type=int, default=2, help='number of pyramidal layers (default: 2)')
parser.add_argument('--decoder-size', type=int, default=512, help='hidden size of model (default: 512)')
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate in training (default: 0.3)')
parser.add_argument('--bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')

parser.add_argument('--batch-size', type=int, default=32, help='Batch size in training (default: 32)')
parser.add_argument('--num-workers', type=int, default=4, help='Number of workers in dataset loader (default: 4)')
parser.add_argument('--num-gpu', type=int, default=1, help='Number of gpus (default: 1)')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--epochs', type=int, default=100, help='Number of max epochs in training (default: 100)')
parser.add_argument('--start-epoch', type=int, default=0)
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing learning rate every epoch')
parser.add_argument('--teacher-forcing', type=float, default=1.0,
                    help='Teacher forcing ratio in decoder (default: 1.0)')
parser.add_argument('--max-len', type=int, default=80, help='Maximum characters of sentence (default: 80)')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--warm-steps', default=4000, type=int)

# Audio Config
parser.add_argument('--sample-rate', default=16000, type=int, help='Sampling Rate')
parser.add_argument('--num-mels', default=80, type=int)
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram')

# Meanteacher
parser.add_argument('--consistency', default=None, type=float,
                    help='use consistency loss with given weight')
parser.add_argument('--consistency-type', default='mse', type=str,
                    help='mse or kl type')
parser.add_argument('--consistency-rampup', default=30, type=int,
                    help="consistency loss ramp-up epoch")
parser.add_argument('--ema-decay', default=0.999, type=float,
                    help='ema variable decay rate')


# System
parser.add_argument('--print-freq', default=1, type=int)
parser.add_argument('--resume', default=None, type=int, metavar='PATH', help="model_args.resume")
parser.add_argument('--save-folder', default='saved_models', help='Location to save epoch models')
parser.add_argument('--log-path', default='log/', help='path to predict log about valid and test dataset')
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--mode', type=str, default='train', help='Train or Test')
parser.add_argument('--finetune', dest='finetune', action='store_true', default=False,
                    help='Finetune the model after load model')

char2index = dict()
index2char = dict()
SOS_token = 0
EOS_token = 0
PAD_token = 0

global_step = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count() # node: server(기계)라고 생각

    main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    global char2index
    global index2char
    global SOS_token
    global EOS_token
    global PAD_token

    args.gpu = gpu

    char2index, index2char = load_label_json(args.labels_path) # data/kor_syllable.json
    # char2index = {'_': 0, 'ⓤ': 1, '☞': 2, '☜': 3, ' ': 4, '이': 5, '다': 6, '는': 7, '에': 8 ...}
    # index2char = {0: '_', 1: 'ⓤ', 2: '☞', 3: '☜', 4: ' ', 5: '이', 6: '다', 7: '는', ...}
    SOS_token = char2index['<s>'] # 2001
    EOS_token = char2index['</s>'] # 2002
    PAD_token = char2index['_'] # 0
    num_classes = len(char2index)

    # audio 설정
    audio_conf = dict(sample_rate=args.sample_rate, # 16,000
                      num_mel=args.num_mels, # 80
                      window_size=args.window_size, # 25ms
                      window_stride=args.window_stride) # 10ms

    batch_size = args.batch_size * args.num_gpu # 32 * 1

    # Train dataset/ loader
    trainData_list = []
    with open(args.train_file, 'r', encoding='utf-8') as f: # train_file: data/train.json
        trainData_list = json.load(f) # data/train.json
        # print(trainData_list)
        # [{'wav': '42_0604_654_0_03223_03.wav', 'text': '자가용 끌고 가도 되나요?', 'speaker_id': '03223'}
        # ,{'wav': '41_0521_958_0_08827_06.wav', 'text': '아 네 감사합니다! 혹시 그때 4인으로 예약했는데 2명이 더 갈 거같은데 6인으로 가능한가요?', 'speaker_id': '08827'}]


    # print(len(trainData_list)) : 59662
    if args.num_gpu != 1: # Multi GPU라면
        last_batch = len(trainData_list) % batch_size # 마지막 찌꺼기 batch(batch_size 보다 작음)
        if last_batch != 0 and last_batch < args.num_gpu: # 찌꺼기가 있고 사용할 gpu 수보다 작을시
            trainData_list = trainData_list[:-last_batch] # 삭제

    train_dataset_path = os.path.join(args.dataset_path, "wavs_train")
    noisy_train_dataset_path = os.path.join(args.dataset_path, "noisy_wavs_train")
    # print(train_dataset_path) = data/wavs_train
    train_dataset = MelFilterBankDataset(audio_conf=audio_conf,
                                         dataset_path=train_dataset_path,
                                         noisy_dataset_path=noisy_train_dataset_path,
                                         data_list=trainData_list, # 파일명, text, speaker id
                                         char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                         normalize=True,
                                         mode='train')

    train_sampler = BucketingSampler(data_source=train_dataset, batch_size=batch_size)
    train_loader = AudioDataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=train_sampler, pin_memory=True)

    # Test dataset/ loader
    testLoader_dict = {}
    test_dataset_path = os.path.join(args.dataset_path, "wavs_test")
    noisy_test_dataset_path = os.path.join(args.dataset_path, "noisy_wavs_test")
    for test_file in args.test_file_list: # ['data/test.json']
        testData_list = []
        with open(test_file, 'r', encoding='utf-8') as f:
            testData_list = json.load(f)
            # print(testData_list)
            # [{"wav": "....", "text":, ...., speaker_id: ....}]

        test_dataset = MelFilterBankDataset(audio_conf=audio_conf,
                                            dataset_path=test_dataset_path,
                                            noisy_dataset_path=noisy_test_dataset_path,
                                            data_list=testData_list,
                                            char2index=char2index, sos_id=SOS_token, eos_id=EOS_token,
                                            normalize=True,
                                            mode='test')
        testLoader_dict[test_file] = AudioDataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
        # print("testLoader_dict: ", testLoader_dict) #{'data/test.json': <Dataset.AudioDataLoader object at 0x7fe37f7311d0>}
        # test file 들을 dictionary 형태로 저장 각 파일명해가지고

        # Model
        input_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1)
        # print(input_size) # 161 : n_fft = sample_rate * window_size /2 + 1 따라 맞춘것 같다.
        enc = EncoderRNN(args, input_size, args.encoder_size, n_layers=args.encoder_layers,
                         dropout_p=args.dropout, bidirectional=args.bidirectional,
                         rnn_cell=args.rnn_type, variable_lengths=False).cuda(args.gpu)

        dec = DecoderRNN(args, len(char2index), args.max_len, args.decoder_size, args.encoder_size,
                         SOS_token, EOS_token,
                         n_layers=args.decoder_layers, rnn_cell=args.rnn_type,
                         dropout_p=args.dropout, bidirectional_encoder=args.bidirectional).cuda(args.gpu)

        model = Seq2Seq(enc, dec)

        ema_enc = EncoderRNN(args, input_size, args.encoder_size, n_layers=args.encoder_layers,
                         dropout_p=args.dropout, bidirectional=args.bidirectional,
                         rnn_cell=args.rnn_type, variable_lengths=False).cuda(args.gpu)

        ema_dec = DecoderRNN(args, len(char2index), args.max_len, args.decoder_size, args.encoder_size,
                         SOS_token, EOS_token,
                         n_layers=args.decoder_layers, rnn_cell=args.rnn_type,
                         dropout_p=args.dropout, bidirectional_encoder=args.bidirectional).cuda(args.gpu)

        ema_model = Seq2Seq(enc, dec)

    if args.num_gpu != 1:
        print("DataParallel 사용")
        model = nn.DataParallel(model).cuda(args.gpu)
        ema_model = nn.DataParallel(ema_model).cuda(args.gpu)
    else:
        print("GPU 1개만 사용")
        model = model.cuda(args.gpu)
        ema_model = ema_model.cuda(args.gpu)
    # print("[Model]")

    # Optimizer / Criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(reduction='mean').cuda(args.gpu)



    if args.resume is not None:
        torch.load(args.resume, map_location='cuda:0')
        model.load_state_dict(torch.load('saved_models/model_%d.pth' % (args.resume), map_location="cuda:0"))
        ema_model.load_state_dict(torch.load('saved_models/ema_model_%d.pth' % (args.resume), map_location="cuda:0"))
    else:
        print("No saved model")


    # Test
    if args.mode != "train":
        for test_file in args.test_file_list:
            test_loader = testLoader_dict[test_file]
            test_loss, test_cer, transcripts_list = evaluate(model, test_loader, criterion, args, save_output=True)
            ema_test_loss, ema_test_cer, ema_transcripts_list = evaluate(ema_model, test_loader, criterion, args, save_output=True)

            for line in ema_transcripts_list:
                print(line)

            print("Student: Test {} CER : {}".format(test_file, test_cer))
            print("Teacher: Test {} CER : {}".format(test_file, ema_test_cer))

    # Train
    else:
        best_cer = 1e10

        for epoch in range(args.start_epoch, args.epochs):
            train_loss, train_cer = train(model, ema_model, train_loader, criterion, optimizer, args, epoch, train_sampler,
                                          args.max_norm, args.teacher_forcing)
            # args.max_norm = 400

            cer_list = []
            ema_cer_list = []

            for test_file in args.test_file_list:
                print(test_file)
                test_loader = testLoader_dict[test_file] # test.json
                test_loss, test_cer, _ = evaluate(model, test_loader, criterion, args, save_output=False)
                test_log = 'Student: Test({name}) Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\tAverage CER {cer:.3f}\t'.format(
                    epoch + 1, name=test_file, loss=test_loss, cer=test_cer)
                print(test_log)

                ema_test_loss, ema_test_cer, _ = evaluate(ema_model, test_loader, criterion, args, save_output=False)
                ema_test_log = 'Teacher: Test({name}) Summary Epoch: [{0}]\tAverage Loss {loss:.3f}\tAverage CER {cer:.3f}\t'.format(
                    epoch + 1, name=test_file, loss=ema_test_loss, cer=ema_test_cer)
                print(ema_test_log)

                cer_list.append(test_cer)
                ema_cer_list.append(ema_test_cer)

            if best_cer > ema_cer_list[0]:
                print("Found better validated model")
                torch.save(model.state_dict(), "saved_models/model_%d.pth" % (epoch + 1))
                torch.save(ema_model.state_dict(), "saved_models/ema_model_%d.pth" % (epoch + 1))
                best_cer = ema_cer_list[0]

            print("Shuffling batches...")
            train_sampler.shuffle(epoch)


def train(model, ema_model, data_loader, criterion, optimizer, args, epoch, train_sampler, max_norm=400,
              teacher_forcing_ratio=1.0):

    global global_step
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0

    # consistency_type 고르기
    if args.consistency_type == 'mse':  # default 되어있음
        consistency_criterion = softmax_mes_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = softmax_kl_loss
    else:
        assert False, args.consistency_type

    model.train()
    ema_model.train()
    print(" [ TRAIN ] ")

    end = time.time()

    for i, (data) in enumerate(data_loader):
        feats, scripts, feat_lengths, script_lengths, noisy, noisy_lengths = data
        # print("1", feats)
        # print("2", noisy)
        # seqs, targets, seq_lengths, target_lengths
        # print("seqs: ", feats.size())
        # print("targets: ", scripts.size())
        # print("seq_lengths: ", feat_lengths.size())
        # print("scripts_lengths: ", len(script_lengths))


        optimizer.zero_grad()

        feats = feats.cuda(args.gpu, non_blocking=True)
        scripts = scripts.cuda(args.gpu, non_blocking=True)
        feat_lengths = feat_lengths.cuda(args.gpu, non_blocking=True)

        src_len = scripts.size(1)
        target = scripts[:, 1:]
        # print("target: ", target.size())

        logit = model(feats, feat_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        # print("logit: ", logit.size())
        # print("logit2: ", logit.contiguous().view(-1, logit.size(-1)).size())
        logit = torch.stack(logit, dim=1).cuda(args.gpu)
        y_hat = logit.max(-1)[1]

        class_loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
        total_loss += class_loss.item()
        total_num += sum(feat_lengths).item()

        with torch.no_grad():
            noisy = noisy.cuda(args.gpu, non_blocking=True)
            noisy_lengths = noisy_lengths.cuda(args.gpu, non_blocking=True)

            ema_logit = ema_model(noisy, noisy_lengths, scripts, teacher_forcing_ratio=teacher_forcing_ratio)

        ema_logit = torch.stack(ema_logit, dim=1).cuda(args.gpu)
        ema_y_hat = ema_logit.max(-1)[1]
        # print("A", y_hat)
        # print("B", ema_y_hat)

        if args.consistency:
            consistency_weight = get_current_consistency_weight(epoch, args)
            consistency_loss = consistency_weight * consistency_criterion(logit, ema_logit) / len(script_lengths)
        else:
            consistency_loss = 0

        loss = class_loss + consistency_loss

        total_loss += consistency_loss.item()
        total_num += sum(feat_lengths).item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        global_step += 1

        dist, length, _ = get_distance(target, y_hat)
        ema_dist, ema_length, _ = get_distance(target, ema_y_hat)

        total_dist += dist
        total_dist += ema_dist

        total_length += length
        total_length += ema_length

        cer = float(dist / length) * 100 # 백분율: (ref 와 hyp 의 차이) / ref 문자열
        ema_cer = float(ema_dist / ema_length) * 100

        update_ema_variables(model, ema_model, args.ema_decay, global_step)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss:.4f}\t'
                  'Cer {cer:.4f} \t ema_CER {ema_cer:.4f}'.format(
                (epoch + 1), (i + 1), len(train_sampler), loss=loss, cer=cer, ema_cer=ema_cer))

    print("Time per Epoch: ", datetime.timedelta(seconds=time.time() - end))
    return total_loss / total_num, (total_dist / total_length) * 100


def evaluate(model, data_loader, criterion, args, save_output=False):
    total_loss = 0.
    total_num = 0
    total_dist = 0
    total_length = 0
    total_sent_num = 0
    transcripts_list = []

    model.eval()
    with torch.no_grad():
        for i, (data) in tqdm(enumerate(data_loader), total=len(data_loader)):
            # tqdm 작업줄 표시
            feats, scripts, feat_lengths, script_lengths, noisy, noisy_lengths = data

            feats = feats.cuda(args.gpu)
            scripts = scripts.cuda(args.gpu)
            feat_lengths = feat_lengths.cuda(args.gpu)
            noisy = noisy.cuda(args.gpu, non_blocking=True)
            noisy_lengths = noisy_lengths.cuda(args.gpu, non_blocking=True)

            target = scripts[:, 1:]
            src_len = scripts.size(1)

            # teacher forcing 안쓰므로 scripts 필요 X
            logit = model(noisy, noisy_lengths, None, eacher_forcing_ratio=0.0)
            logit = torch.stack(logit, dim=1).cuda(args.gpu)
            y_hat = logit.max(-1)[1]

            logit = logit[:,:target.size(1),:] # target length까지만
            loss = criterion(logit.contiguous().view(-1, logit.size(-1)), target.contiguous().view(-1))
            total_loss += loss.item()
            total_num += sum(feat_lengths).item()

            dist, length, transcripts = get_distance(target, y_hat)
            cer = float(dist / length) * 100

            total_dist += dist
            total_length += length

            if save_output == True:
                transcripts_list += transcripts
            total_sent_num += target.size(0)

    aver_loss = total_loss / total_num
    aver_cer = float(total_dist / total_length) * 100

    return aver_loss, aver_cer, transcripts_list



def get_distance(ref_labels, hyp_labels):
    # 각 ref: target indices hyp: predict indices
    total_dist = 0
    total_length = 0
    transcripts = []
    for i in range(len(ref_labels)):
        ref = label_to_string(ref_labels[i])
        hyp = label_to_string(hyp_labels[i])
        # ref, hyp: 4명이요 이런식으로

        transcripts.append('output: {hyp} \n target: {ref} \n'.format(hyp=hyp, ref=ref))

        dist, length = char_distance(ref, hyp)
        total_dist += dist
        total_length += length

    return total_dist, total_length, transcripts


def label_to_string(labels):
    # print(" labels.shape: ", labels.shape) # [8] batchsize만큼
    if len(labels.shape) == 1:
        sent = str()
        for i in labels:
            if i.item() == EOS_token: # EOS 이면 정지
                break
            sent += index2char[i.item()]

        return sent

    elif len(labels.shape) == 2:
        sents = list()
        for i in labels:
            sent = str()
            for j in i:
                if j.item() == EOS_token:
                    break
                sent += index2char[j.item()]
            sents.append(sent)

        return sents


def char_distance(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')

    dist = Lev.distance(hyp, ref)
    length = len(ref.replace(' ', '')) # 총 문자열 수

    return dist, length


if __name__ == "__main__":
    main()
