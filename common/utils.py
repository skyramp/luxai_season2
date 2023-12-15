import errno
import os
import random
from dataclasses import field, dataclass
from typing import List
import numpy as np
import torch


@dataclass
class AvgCounter:
    _cumulative: float = 0
    _n: int = 0
    _values: List[float] = field(default_factory=lambda :[])

    def append(self, mc: 'AvgCounter'):
        self._cumulative += mc._cumulative
        self._n += mc._n
        self._values += mc._values

    def update(self, v):
        self._cumulative += v
        self._n += 1
        self._values += [v]

    def mean(self):
        return self._cumulative / (self._n or 1)

    def percentile(self, v):
        return np.percentile(self._values, v)

    def values(self):
        return self._values

    def reset(self):
        self._values = []
        self._n = 0
        self._cumulative = 0



def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    if seed_value is not None:
        os.environ['PYTHONHASHSEED'] = str(seed_value)

    if seed_value is not None:
        torch.manual_seed(seed_value)
    else:
        torch.seed()



def mkdir_p(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(dir_path):
            pass
        else:
            raise

