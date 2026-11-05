"""
General helper functions.
通用工具函数。
"""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    """
    Fix every source of randomness so that LLM sampling (temperature > 0) gives the same results every run.
    Same as set_seed in the official utils/general_utils.py.
    固定所有随机数来源，使 LLM 采样（temperature > 0）每次运行结果相同。
    与官方 utils/general_utils.py 的 set_seed 一致。
    """
    # PyTorch, on CPU and on every GPU / PyTorch：CPU 和所有 GPU
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # NumPy and Python's own random module / NumPy 和 Python 自带的 random
    np.random.seed(seed)
    random.seed(seed)
    # Python's hash randomisation / Python 的哈希随机化
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Make cuDNN (GPU kernels) deterministic instead of picking the fastest, possibly non-deterministic, algorithm.
    # 让 cuDNN（GPU 计算库）使用确定性的算法，而不是挑最快但结果可能不固定的算法。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
