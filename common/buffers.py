from copy import copy
from dataclasses import dataclass
from typing import List, Dict, Union, Callable, Any, Tuple, Optional

import torch
from gym import spaces


@dataclass
class BufferSpec:
    shape: Tuple
    dtype: Any = None


Buffer = Union[Dict, torch.Tensor]

def stack_buffers(buffers: List[Buffer], dim: int) -> Buffer:
    stacked_buffers = {}
    for key, val in copy(buffers[0]).items():
        if isinstance(val, dict):
            stacked_buffers[key] = stack_buffers([b[key] for b in buffers], dim)
        else:
            stacked_buffers[key] = torch.cat([b[key] for b in buffers], dim=dim)
    return stacked_buffers


def split_buffers(
        buffers: Buffer,
        split_size_or_sections: Union[int, List[int]],
        dim: int,
        contiguous: bool,
) -> List[Buffer]:
    buffers_split = None
    for key, val in copy(buffers).items():
        if isinstance(val, dict):
            bufs = split_buffers(val, split_size_or_sections, dim, contiguous)
        else:
            bufs = torch.split(val, split_size_or_sections, dim=dim)
            if contiguous:
                bufs = [b.contiguous() for b in bufs]

        if buffers_split is None:
            buffers_split = [{} for _ in range(len(bufs))]
        assert len(bufs) == len(buffers_split)
        buffers_split = [dict(**{key: buf}, **d) for buf, d in zip(bufs, buffers_split)]
    return buffers_split



def buffer_apply(buffer: Buffer, func: Callable[[torch.Tensor], Any]) -> Buffer:
    if isinstance(buffer, dict):
        return {
            key: buffer_apply(val, func) for key, val in copy(buffer).items()
        }
    else:
        return func(buffer)

def buffer_fill(buffer: Buffer, step: int, fill_vals: Optional[Buffer]):
    if isinstance(buffer, dict):
        for key in buffer:
            buffer_fill(buffer[key], step, fill_vals.get(key) if fill_vals else None)
    else:
        buffer[step, ...] = torch.as_tensor(fill_vals) if fill_vals is not None else 0


def buffer_fill_range(buffer: Buffer, s1, s2, source: Optional[Buffer]):
    if isinstance(buffer, dict):
        for key in buffer:
            if key not in source:
                continue
            buffer_fill_range(buffer[key], s1, s2, source.get(key) if source else None)
    else:
        buffer[s1:s2, ...] = source[:] if source is not None else 0


def buffer_copy(buffer: Buffer, source: Optional[Buffer]):
    if isinstance(buffer, dict):
        for key in buffer:
            if key not in source:
                continue
            buffer_copy(buffer[key], source.get(key) if source else None)
    else:
        buffer[:, ...] = source[:] if source is not None else 0



def space_to_spec(spec: spaces.Space) -> Union[Dict,BufferSpec]:
    if isinstance(spec, spaces.Dict) or isinstance(spec, dict):
        return {k: space_to_spec(spec[k]) for k in spec}
    elif isinstance(spec, tuple):
        return BufferSpec(spec, dtype=torch.float32)
    elif isinstance(spec, BufferSpec):
        return spec
    else:
        assert spec.shape is not None
        return BufferSpec(spec.shape, dtype=spec.dtype)


def buffer_create(space, rlen:int, blen:Optional[int], **kwargs):
    spec = space_to_spec(space)

    if isinstance(spec, dict):
        d = {}
        for k in spec:
            d[k] = buffer_create(spec[k], rlen, blen, **kwargs)
        return d
    else:
        if blen is not None:
            shape = (rlen, blen) + spec.shape
        else:
            shape = (rlen, ) + spec.shape
        v = torch.zeros(shape, dtype=spec.dtype, **kwargs)
        return v

def buffer_shape(buffer: Buffer):
    if isinstance(buffer, dict):
        return {k: buffer_shape(v) for k, v in buffer.items()}
    else:
        return buffer.shape

def buffer_size(buffer: Buffer):
    if isinstance(buffer, dict):
        s = 0
        for k in buffer:
            s += buffer_size(buffer[k])
        return s
    else:
        numel = buffer.storage().size()
        element_size = buffer.storage().element_size()
        mem = numel * element_size
        return mem
