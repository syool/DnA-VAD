import torch

import numpy as np
from glob import glob
import math
import random

import os


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def seeds(val):
    ''' generate fixed random values '''
    random.seed(val)
    np.random.seed(val)
    torch.manual_seed(val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(val)
        torch.cuda.manual_seed_all(val)


def label_encapsule(labels, path, clip_length):
    ''' encapsule labels w.r.t the window size '''
    videos = sorted(glob(os.path.join(path, '*')))
    
    capsules = []
    count = 0
    for vid in videos:
        frames = sorted(glob(os.path.join(vid, '*')))
            
        n_frames = len(frames)
        l = labels[count:count+n_frames]
        
        cap = l[clip_length:]
        capsules = np.append(capsules, cap)
        
        count += n_frames
    
    return capsules


def psnr(mse):
    return 10 * math.log10(1/mse)


def score_norm(arr):
    ''' input must be a numpy array '''
    return (arr - arr.min()) / (arr.max() - arr.min())

