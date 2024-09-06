# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.

import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def image_postprocess(x):
    # [-1, 1] -> [0, 1]
    return torch.clamp((x + 1) / 2, 0, 1)  # x / 2 + 0.5

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

