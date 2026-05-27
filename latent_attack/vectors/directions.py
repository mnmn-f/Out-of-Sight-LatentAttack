from typing import Dict, List, Tuple

import torch


def direction_from_stacked(stacked: torch.Tensor, *, direction_method: str) -> torch.Tensor:
    if direction_method in {"mean", "diffmean"}:
        return stacked.mean(dim=0)
    if direction_method != "pca":
        raise ValueError(f"Unsupported direction_method: {direction_method}")

    original_shape = stacked.shape[1:]
    flat = stacked.to(torch.float32).reshape(stacked.shape[0], -1)
    mean_direction = flat.mean(dim=0)
    if flat.shape[0] == 1:
        return mean_direction.reshape(original_shape)

    _, _, vh = torch.linalg.svd(flat, full_matrices=False)
    direction = vh[0]
    if torch.dot(direction, mean_direction) < 0:
        direction = -direction

    mean_norm = mean_direction.norm()
    if mean_norm > 0:
        direction = direction * mean_norm
    return direction.reshape(original_shape)


def pool_hidden(hidden: torch.Tensor, *, field: str, step_pooling: str, k: int, hidden_layer: str = "last") -> torch.Tensor:
    if field == "prompt_last_hidden":
        if hidden.dim() != 1:
            raise ValueError(f"Expected 1D tensor for prompt_last_hidden, got shape {tuple(hidden.shape)}")
        return hidden.to(torch.float32)

    if hidden.dim() not in (2, 3):
        raise ValueError(f"Expected 2D or 3D tensor for latent hidden field, got shape {tuple(hidden.shape)}")
    if hidden.shape[0] == 0:
        raise ValueError("Latent hidden tensor has zero steps. Increase --latent_steps when collecting traces.")

    if step_pooling == "mean_all":
        selected = hidden
    elif step_pooling == "first_k":
        selected = hidden[: max(1, min(k, hidden.shape[0]))]
    elif step_pooling == "last_k":
        selected = hidden[-max(1, min(k, hidden.shape[0])) :]
    elif step_pooling == "step":
        step_idx = k if k >= 0 else hidden.shape[0] + k
        if step_idx < 0 or step_idx >= hidden.shape[0]:
            raise ValueError(f"Requested step index {k}, but hidden has {hidden.shape[0]} steps.")
        selected = hidden[step_idx : step_idx + 1]
    else:
        raise ValueError(f"Unsupported step_pooling: {step_pooling}")

    pooled = selected.to(torch.float32).mean(dim=0)
    if hidden.dim() == 2:
        return pooled
    if hidden_layer == "all":
        return pooled
    if hidden_layer == "last":
        return pooled[-1]

    try:
        layer_idx = int(hidden_layer)
    except ValueError as exc:
        raise ValueError("--hidden_layer must be all, last, or a 0-based transformer layer index") from exc
    if layer_idx < 0:
        layer_idx = pooled.shape[0] + layer_idx
    if layer_idx < 0 or layer_idx >= pooled.shape[0]:
        raise ValueError(f"Requested hidden layer {hidden_layer}, but trace has {pooled.shape[0]} layers.")
    return pooled[layer_idx]


def extract_hidden_vectors(
    role_pairs: List[Tuple[str, str]],
    *,
    hidden_field: str,
    step_pooling: str,
    k: int,
    hidden_layer: str,
    direction_method: str,
) -> Dict:
    diffs: List[torch.Tensor] = []
    per_pair_meta = []

    for clean_path, attacked_path in role_pairs:
        clean_payload = torch.load(clean_path, map_location="cpu")
        attacked_payload = torch.load(attacked_path, map_location="cpu")
        clean_vec = pool_hidden(
            clean_payload["hidden_trace"][hidden_field],
            field=hidden_field,
            step_pooling=step_pooling,
            k=k,
            hidden_layer=hidden_layer,
        )
        attacked_vec = pool_hidden(
            attacked_payload["hidden_trace"][hidden_field],
            field=hidden_field,
            step_pooling=step_pooling,
            k=k,
            hidden_layer=hidden_layer,
        )
        diffs.append(attacked_vec - clean_vec)
        per_pair_meta.append(
            {
                "sample_idx": attacked_payload.get("sample_idx"),
                "clean_file": clean_path,
                "attacked_file": attacked_path,
            }
        )

    if not diffs:
        raise ValueError("No matched hidden-trace pairs found.")

    stacked = torch.stack(diffs, dim=0)
    vector = direction_from_stacked(stacked, direction_method=direction_method)
    payload = {
        "vector": vector,
        "hidden_layer": hidden_layer,
        "direction_method": direction_method,
        "num_pairs": len(diffs),
        "per_pair_meta": per_pair_meta,
        "stacked_diffs": stacked,
    }
    if vector.dim() == 2:
        payload["layer_vectors"] = [vector[idx].contiguous() for idx in range(vector.shape[0])]
    return payload


def mask_kv_position(diff: torch.Tensor, kv_position: str) -> torch.Tensor:
    if kv_position == "all":
        return diff
    pos = int(kv_position)
    if pos < 0:
        pos = diff.shape[-2] + pos
    if pos < 0 or pos >= diff.shape[-2]:
        raise ValueError(f"Requested KV position {kv_position}, but KV diff has {diff.shape[-2]} positions.")
    masked = torch.zeros_like(diff)
    masked[..., pos : pos + 1, :] = diff[..., pos : pos + 1, :]
    return masked


def mean_kv_diff(role_pairs: List[Tuple[str, str]], *, kv_position: str = "all", direction_method: str = "mean") -> Dict:
    k_diffs_by_layer: List[List[torch.Tensor]] = []
    v_diffs_by_layer: List[List[torch.Tensor]] = []
    per_pair_meta = []

    for clean_path, attacked_path in role_pairs:
        clean_payload = torch.load(clean_path, map_location="cpu")
        attacked_payload = torch.load(attacked_path, map_location="cpu")
        clean_kv = clean_payload.get("kv_trace")
        attacked_kv = attacked_payload.get("kv_trace")
        if clean_kv is None or attacked_kv is None:
            raise ValueError("Missing kv_trace in saved payload. Re-run with --trace_save_kv.")
        if len(clean_kv) != len(attacked_kv):
            raise ValueError("Layer mismatch between clean and attacked kv traces.")

        while len(k_diffs_by_layer) < len(clean_kv):
            k_diffs_by_layer.append([])
            v_diffs_by_layer.append([])

        for layer_idx, (clean_layer, attacked_layer) in enumerate(zip(clean_kv, attacked_kv)):
            clean_k, clean_v = clean_layer
            attacked_k, attacked_v = attacked_layer
            if clean_k.shape != attacked_k.shape or clean_v.shape != attacked_v.shape:
                raise ValueError(
                    f"KV shape mismatch at layer {layer_idx}: "
                    f"clean K {tuple(clean_k.shape)} vs attacked K {tuple(attacked_k.shape)}"
                )
            k_diffs_by_layer[layer_idx].append(mask_kv_position((attacked_k - clean_k).to(torch.float32), kv_position))
            v_diffs_by_layer[layer_idx].append(mask_kv_position((attacked_v - clean_v).to(torch.float32), kv_position))

        per_pair_meta.append(
            {
                "sample_idx": attacked_payload.get("sample_idx"),
                "clean_file": clean_path,
                "attacked_file": attacked_path,
            }
        )

    if not per_pair_meta:
        raise ValueError("No matched kv-trace pairs found.")

    stacked_k_diffs_by_layer = [torch.stack(layer_diffs, dim=0) for layer_diffs in k_diffs_by_layer]
    stacked_v_diffs_by_layer = [torch.stack(layer_diffs, dim=0) for layer_diffs in v_diffs_by_layer]
    return {
        "k_vectors": [direction_from_stacked(stacked, direction_method=direction_method) for stacked in stacked_k_diffs_by_layer],
        "v_vectors": [direction_from_stacked(stacked, direction_method=direction_method) for stacked in stacked_v_diffs_by_layer],
        "direction_method": direction_method,
        "stacked_k_diffs_by_layer": stacked_k_diffs_by_layer,
        "stacked_v_diffs_by_layer": stacked_v_diffs_by_layer,
        "num_pairs": len(per_pair_meta),
        "per_pair_meta": per_pair_meta,
        "kv_position": kv_position,
    }
