import random
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import os
import time

import numpy as np
from numpy.random import multivariate_normal as MultiVariateNormal
import pprint as pprint

_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)

# random seed
def set_seed(seed):
    if seed == 0:
        print(' random seed')
        torch.backends.cudnn.benchmark = True
    else:
        print('manual seed:', seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_gpu(args):
    gpu_list = [int(x) for x in args.gpu.split(',')]
    print('use gpu:', gpu_list)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return gpu_list.__len__()

# class Sample(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, mean, cov, length):
#         return MultiVariateNormal(mean, cov, length)

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output.clamp_(-1, 1)
    
# def sample(mean, cov, length):
#     return Sample.apply(mean, cov, length)
    
def sample_v2(mean, cov, length):
    std_mean = np.zeros((mean.shape[0]))
    std_cov = np.ones((mean.shape[0], mean.shape[0]))
    std_samples = MultiVariateNormal(std_mean, std_cov, length)
    g_samples = mean+torch.mm(torch.tensor(std_samples).float().cuda(),cov)  # reparameterization
    return g_samples
    
    
def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

# calculate classification accuracy
def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(str(item) + '\n')
    f.close()
