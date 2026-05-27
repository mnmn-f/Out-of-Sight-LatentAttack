from typing import Optional, Tuple

import torch

from models import _past_length

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def slice_tensor(tensor: torch.Tensor, tokens_to_keep: int) -> torch.Tensor:
    if tokens_to_keep <= 0:
        return tensor[..., 0:0, :].contiguous()
    keep = min(tokens_to_keep, tensor.shape[-2])
    start = tensor.shape[-2] - keep
    return tensor[..., start:, :].contiguous()


def truncate_past(past_kv: Optional[Tuple], tokens_to_keep: int) -> Optional[Tuple]:
    if past_kv is None or tokens_to_keep <= 0:
        return None
    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        trimmed_legacy = tuple(
            tuple(slice_tensor(t, tokens_to_keep) for t in layer)
            for layer in legacy
        )
        return past_kv.__class__.from_legacy_cache(trimmed_legacy)

    trimmed_layers = []
    for layer in past_kv:
        if isinstance(layer, tuple):
            trimmed_layers.append(tuple(slice_tensor(t, tokens_to_keep) for t in layer))
        elif torch.is_tensor(layer):
            trimmed_layers.append(slice_tensor(layer, tokens_to_keep))
        else:
            trimmed_layers.append(layer)
    return tuple(trimmed_layers)


def slice_past_for_trace(
    past_kv: Optional[Tuple],
    *,
    prev_past_len: int,
    latent_steps: int,
    mode: str,
) -> Optional[Tuple]:
    if past_kv is None:
        return None
    total_len = _past_length(past_kv)
    if total_len <= 0:
        return None

    if mode == "full":
        start = 0
    elif mode == "delta":
        start = prev_past_len
    elif mode == "latent_only":
        keep = min(max(latent_steps, 0), total_len)
        start = total_len - keep
    else:
        raise ValueError(f"Unsupported trace_kv_mode: {mode}")

    start = max(0, min(start, total_len))
    if start >= total_len:
        return None

    sliced_layers = []
    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
    else:
        legacy = past_kv

    for layer in legacy:
        if not isinstance(layer, tuple):
            continue
        sliced_layers.append(tuple(t[..., start:, :].detach().cpu().contiguous() for t in layer))
    return tuple(sliced_layers)
