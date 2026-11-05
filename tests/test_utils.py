"""
Unit tests for utils.py: the random seed.
utils.py 的单元测试：随机种子。
"""

import random

import numpy as np
import torch

from dra.utils import set_seed


def draw():
    return random.random(), np.random.rand(), torch.rand(1).item()


def test_same_seed_gives_same_random_numbers():
    set_seed(42)
    first = draw()
    set_seed(42)
    assert draw() == first


def test_different_seed_gives_different_random_numbers():
    set_seed(42)
    first = draw()
    set_seed(43)
    assert draw() != first
