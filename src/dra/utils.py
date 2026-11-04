"""通用工具函数。"""

import os
import random

import numpy as np
import torch


def set_seed(seed):
    """
    固定所有随机数来源，使 LLM 采样（temperature > 0）的结果可复现。
    与官方 utils/general_utils.py 的 set_seed 一致。
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
