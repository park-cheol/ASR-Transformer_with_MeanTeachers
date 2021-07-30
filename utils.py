import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weight(m):
    nn.init.xavier_normal_(m.weight)
    if m.bias is not None:
        nn.init.constant_(m.bias, 0)

class ScheduleAdam():

    def __init__(self, optimizer, hidden_dim, warm_steps):
        self.init_lr = np.power(hidden_dim, -0.5) # 1 / sqrt(hidden_dim)
        self.optimizer = optimizer
        self.current_steps = 0
        self.warm_steps = warm_steps

    def step(self):
        # current_step 정보를 이용해서 lr Update
        self.current_steps += 1
        lr = self.init_lr * self.get_scale()

        for p in self.optimizer.param_groups:
            p['lr'] = lr # lr Update

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_scale(self):
        return np.min([np.power(self.current_steps, -0.5),
                       self.current_steps * np.power(self.warm_steps, -1.5)
                       ])

#################### mean teachers############################
def sigmoid_rampup(current, rampup_length): # current: epoch
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        # np.clip(array, min, max): array를 min max안으로 좁힘
        phase = 1.0 - current / rampup_length

        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, args):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    # paper: alpha는 0에서 시간이 지날수록 점점 증가하는 방법(ramp up)
    # 이러한 방법이 student를 더욱빠르게 초기에 학습
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        # EMA undata 하는 과정
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

# target에는 send gradient X
def softmax_mes_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1] # channel

    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes
# size_average(default:None) = 마지막에 1/N을 관한 것 False로 안해주면 taget과 input
# 의 값들이 다 더해져서 나눠짐 즉 2 * num_classes로 나눠짐

# mse_loss가 더 좋다고 함
def softmax_kl_loss(input_logits, target_logits):
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    return F.kl_div(input_log_softmax, target_softmax, size_average=False)



if __name__ == "__main__":
    a = []
    for i in range(10000):
        a.append(np.min([np.power(i, -0.5),
                         i * np.power(4000, -1.5)
                         ]))
    a = np.array(a)
    plt.figure()
    plt.plot(a)
    plt.show()