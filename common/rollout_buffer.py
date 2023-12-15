from dataclasses import dataclass
from typing import List, Dict, Any

import torch

from common.buffers import Buffer, buffer_fill, buffer_create, buffer_apply


@dataclass
class RolloutBuffer:
    max_length: int
    used: int
    data: Buffer
    info: List[Dict[str, float]]
    meta: Dict[str, Any]

    @torch.no_grad()
    def update(self, source: Buffer, info):
        assert not self.full()
        buffer_fill( self.data, self.used, source)
        self.used += 1
        self.info.append(info)

    @staticmethod
    @torch.no_grad()
    def create_buffer(buffer_spec, rollout_len: int, batch_len=None):
        data = buffer_create(buffer_spec, rollout_len, batch_len)
        return RolloutBuffer(
            used=0,
            max_length=rollout_len,
            data=data,
            info=[],
            meta={}
        )

    def share(self):
        self.data = buffer_apply(self.data, lambda x: x.share_memory_())

    def full(self):
        return self.used == self.max_length

    def reset(self):
        self.used = 0
        self.info = []
        self.meta = {}


