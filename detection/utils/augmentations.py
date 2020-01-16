import torch
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def vertical_flip(images, targets):
    images = torch.flip(images, [-2])
    targets[:, 1] = 1 - targets[:, 1]
    return images, targets



