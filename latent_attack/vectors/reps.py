from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from attacks import attack_metadata
from .pairs import OUTPUT_SOURCES, PAIR_FILTERS, load_preference_pairs, namespace_from_results

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None


def _load_runtime_dependencies() -> None:
    global F, LatentMASMethod, ModelWrapper, auto_device, build_agent_message_hierarchical_latent_mas
    global build_agent_message_sequential_latent_mas, set_seed, torch, tqdm
    global truncate_past

    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    from methods.latent_mas import LatentMASMethod
    from methods.cache_ops import truncate_past
    from models import ModelWrapper
    from prompts import build_agent_message_hierarchical_latent_mas, build_agent_message_sequential_latent_mas
    from utils import auto_device, set_seed


def freeze_model(model: ModelWrapper) -> None:
    model.model.eval()
    for param in model.model.parameters():
        param.requires_grad_(False)


def _render_agent_prompts(method: LatentMASMethod, items: List[Dict], role: str) -> List[str]:
    if method.args.prompt == "sequential":
        messages = [
            build_agent_message_sequential_latent_mas(
                role=role,
                question=item["question"],
                context="",
                method=method.method_name,
                args=method.args,
                gold=item.get("gold", ""),
            )
            for item in items
        ]
    elif method.args.prompt == "hierarchical":
        messages = [
            build_agent_message_hierarchical_latent_mas(
                role=role,
                question=item["question"],
                context="",
                method=method.method_name,
                args=method.args,
                gold=item.get("gold", ""),
            )
            for item in items
        ]
    else:
        raise ValueError(f"Unsupported prompt type: {method.args.prompt}")
    prompts = [method.model.render_chat(msg, add_generation_prompt=True) for msg in messages]
    if method.args.think:
        prompts = [f"{prompt}<think>" for prompt in prompts]
    return prompts


def build_judger_context(
    method: LatentMASMethod,
    items: List[Dict],
    *,
    target_role: str,
    target_layer: int,
    vector: torch.Tensor,
    alpha: float,
    vector_type: str = "hidden",
    kv_mode: str = "kv_both",
    k_vector: Optional[torch.Tensor] = None,
    v_vector: Optional[torch.Tensor] = None,
) -> Tuple[List[str], Optional[Tuple]]:
    past_kv: Optional[Tuple] = None

    for agent in method.agents:
        prompts = _render_agent_prompts(method, items, agent.role)
        if agent.role == "judger":
            return prompts, past_kv if method.latent_steps > 0 else None

        prev_past_len = 0 if past_kv is None else past_kv[0][0].shape[-2]
        encoded = method.model.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(method.model.device)
        attention_mask = encoded["attention_mask"].to(method.model.device)
        layer_vectors = {target_layer: vector} if vector_type == "hidden" and agent.role == target_role else None
        past_kv = method.model.generate_latent_batch_trainable(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=method.latent_steps,
            past_key_values=past_kv,
            latent_layer_injection_vectors=layer_vectors,
            latent_injection_alpha=alpha if layer_vectors else 0.0,
        )
        if method.sequential_info_only or method.latent_only:
            new_past_len = past_kv[0][0].shape[-2]
            tokens_added = new_past_len - prev_past_len
            tokens_to_keep = method.latent_steps if method.latent_only else tokens_added
            past_kv = truncate_past(past_kv, tokens_to_keep)
        if vector_type == "kv" and agent.role == target_role:
            past_kv = apply_trainable_kv_injection(
                past_kv,
                target_layer=target_layer,
                k_vector=k_vector,
                v_vector=v_vector,
                kv_mode=kv_mode,
                alpha=alpha,
            )

    raise RuntimeError("LatentMAS agent list did not include a judger.")


def apply_trainable_kv_injection(
    past_kv: Optional[Tuple],
    *,
    target_layer: int,
    k_vector: Optional[torch.Tensor],
    v_vector: Optional[torch.Tensor],
    kv_mode: str,
    alpha: float,
) -> Optional[Tuple]:
    if past_kv is None:
        return past_kv

    if Cache is not None and isinstance(past_kv, Cache):
        legacy = past_kv.to_legacy_cache()
        return_cache_cls = past_kv.__class__
    else:
        legacy = past_kv
        return_cache_cls = None

    if target_layer < 0 or target_layer >= len(legacy):
        raise ValueError(f"--layer {target_layer} is out of range for KV cache with {len(legacy)} layers")

    updated_layers = []
    for layer_idx, layer in enumerate(legacy):
        if layer_idx != target_layer or not isinstance(layer, tuple):
            updated_layers.append(layer)
            continue

        k_cache, v_cache = layer
        inject_len = k_cache.shape[-2]
        k_updated = k_cache.clone()
        v_updated = v_cache.clone()

        if kv_mode in ("k_only", "kv_both"):
            if k_vector is None:
                raise ValueError("k_vector is required for k_only/kv_both RePS training")
            k_vec = k_vector.to(device=k_cache.device, dtype=k_cache.dtype)
            if k_vec.shape[-2] > inject_len:
                k_vec = k_vec[..., -inject_len:, :]
            suffix = slice(k_cache.shape[-2] - k_vec.shape[-2], k_cache.shape[-2])
            if k_vec.shape[0] == 1 and k_cache.shape[0] != 1:
                k_vec = k_vec.expand(k_cache.shape[0], -1, -1, -1)
            k_updated[..., suffix, :] = k_updated[..., suffix, :] + alpha * k_vec

        if kv_mode in ("v_only", "kv_both"):
            if v_vector is None:
                raise ValueError("v_vector is required for v_only/kv_both RePS training")
            v_vec = v_vector.to(device=v_cache.device, dtype=v_cache.dtype)
            if v_vec.shape[-2] > inject_len:
                v_vec = v_vec[..., -inject_len:, :]
            suffix = slice(v_cache.shape[-2] - v_vec.shape[-2], v_cache.shape[-2])
            if v_vec.shape[0] == 1 and v_cache.shape[0] != 1:
                v_vec = v_vec.expand(v_cache.shape[0], -1, -1, -1)
            v_updated[..., suffix, :] = v_updated[..., suffix, :] + alpha * v_vec

        updated_layers.append((k_updated, v_updated))

    updated_layers = tuple(updated_layers)
    if return_cache_cls is not None:
        return return_cache_cls.from_legacy_cache(updated_layers)
    return updated_layers


def scaled_simpo_loss(
    *,
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    chosen_lens: torch.Tensor,
    rejected_lens: torch.Tensor,
    simpo_scaler: float,
) -> torch.Tensor:
    ref_reverse = (ref_rejected_logps - ref_chosen_logps).detach()
    scale = torch.maximum(ref_reverse * simpo_scaler, torch.ones_like(ref_reverse))
    logits = (scale / chosen_lens.to(chosen_logps.dtype)) * chosen_logps
    logits = logits - (1.0 / rejected_lens.to(rejected_logps.dtype)) * rejected_logps
    return -F.logsigmoid(logits).mean()


def compute_reference_logps(
    method: LatentMASMethod,
    items: List[Dict],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        prompts, past_kv = build_judger_context(
            method,
            items,
            target_role="",
            target_layer=0,
            vector=torch.zeros(method.model.model.config.hidden_size, device=method.model.device),
            alpha=0.0,
        )
        chosen = [item["chosen"] for item in items]
        rejected = [item["rejected"] for item in items]
        chosen_logps, chosen_lens = method.model.sequence_logprob_batch(prompts, chosen, past_key_values=past_kv)
        rejected_logps, rejected_lens = method.model.sequence_logprob_batch(prompts, rejected, past_key_values=past_kv)
        return chosen_logps.detach(), rejected_logps.detach(), chosen_lens.detach(), rejected_lens.detach()


def load_init_vector(path: str, role: str, layer: int, hidden_size: int) -> Optional[torch.Tensor]:
    if not path:
        return None
    payload = torch.load(path, map_location="cpu")
    role_payload = payload.get("roles", {}).get(role)
    if not role_payload or "hidden" not in role_payload:
        raise ValueError(f"No hidden vector for role={role} in init vector payload: {path}")
    hidden = role_payload["hidden"]
    if "layer_vectors" in hidden:
        vector = hidden["layer_vectors"][layer]
    else:
        vector = hidden["vector"]
        if vector.dim() == 2:
            vector = vector[layer]
    if vector.numel() != hidden_size:
        raise ValueError(f"Init vector has {vector.numel()} elements, expected hidden_size={hidden_size}")
    return vector.to(torch.float32)


def load_init_kv_vectors(path: str, role: str, layer: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not path:
        return None, None
    payload = torch.load(path, map_location="cpu")
    role_payload = payload.get("roles", {}).get(role)
    if not role_payload or "kv" not in role_payload:
        raise ValueError(f"No kv vector for role={role} in init vector payload: {path}")
    kv = role_payload["kv"]
    k_vectors = kv.get("k_vectors")
    v_vectors = kv.get("v_vectors")
    if not isinstance(k_vectors, list) or not isinstance(v_vectors, list):
        raise ValueError(f"Invalid kv vector payload: {path}")
    return k_vectors[layer].to(torch.float32), v_vectors[layer].to(torch.float32)


def save_reps_payload(
    *,
    output_path: str,
    vector: torch.Tensor,
    num_layers: int,
    hidden_size: int,
    target_role: str,
    target_layer: int,
    args: argparse.Namespace,
    num_pairs: int,
    train_sample_indices: List[int],
) -> None:
    layer_vectors = [torch.zeros(hidden_size, dtype=torch.float32) for _ in range(num_layers)]
    layer_vectors[target_layer] = vector.detach().to(torch.float32).cpu().contiguous()
    stacked = torch.stack(layer_vectors, dim=0)
    payload = {
        "meta": {
            "method": "reps",
            "target_role": target_role,
            "target_layer": target_layer,
            "direction_method": "reps",
            "loss_type": "scaled_simpo",
            "pair_filter": args.pair_filter,
            "context_results_json": os.path.abspath(args.context_results_json or args.attacked_results_json),
            "clean_results_json": os.path.abspath(args.clean_results_json),
            "attacked_results_json": os.path.abspath(args.attacked_results_json),
            "chosen_source": args.chosen_source,
            "rejected_source": args.rejected_source,
            "alpha_values": args.alpha_values,
            "sub_loss_mode": args.sub_loss_mode,
            "num_pairs": num_pairs,
            "train_sample_indices": train_sample_indices,
        },
        "roles": {
            target_role: {
                "num_pairs": num_pairs,
                "hidden": {
                    "vector": stacked,
                    "layer_vectors": layer_vectors,
                    "hidden_layer": str(target_layer),
                    "direction_method": "reps",
                    "num_pairs": num_pairs,
                    "per_pair_meta": [{"sample_idx": idx} for idx in train_sample_indices],
                },
            }
        },
    }
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, output_path)


def save_reps_kv_payload(
    *,
    output_path: str,
    k_vector: Optional[torch.Tensor],
    v_vector: Optional[torch.Tensor],
    num_layers: int,
    kv_shape: Tuple[int, int, int, int],
    target_role: str,
    target_layer: int,
    args: argparse.Namespace,
    num_pairs: int,
    train_sample_indices: List[int],
) -> None:
    k_vectors = [torch.zeros(kv_shape, dtype=torch.float32) for _ in range(num_layers)]
    v_vectors = [torch.zeros(kv_shape, dtype=torch.float32) for _ in range(num_layers)]
    if k_vector is not None:
        k_vectors[target_layer] = k_vector.detach().to(torch.float32).cpu().contiguous()
    if v_vector is not None:
        v_vectors[target_layer] = v_vector.detach().to(torch.float32).cpu().contiguous()
    payload = {
        "meta": {
            "method": "reps",
            "vector_type": "kv",
            "target_role": target_role,
            "target_layer": target_layer,
            "kv_mode": args.kv_mode,
            "direction_method": "reps",
            "loss_type": "scaled_simpo",
            "pair_filter": args.pair_filter,
            "context_results_json": os.path.abspath(args.context_results_json or args.attacked_results_json),
            "clean_results_json": os.path.abspath(args.clean_results_json),
            "attacked_results_json": os.path.abspath(args.attacked_results_json),
            "chosen_source": args.chosen_source,
            "rejected_source": args.rejected_source,
            "alpha_values": args.alpha_values,
            "sub_loss_mode": args.sub_loss_mode,
            "num_pairs": num_pairs,
            "train_sample_indices": train_sample_indices,
        },
        "roles": {
            target_role: {
                "num_pairs": num_pairs,
                "kv": {
                    "k_vectors": k_vectors,
                    "v_vectors": v_vectors,
                    "direction_method": "reps",
                    "num_pairs": num_pairs,
                    "kv_position": "all",
                    "per_pair_meta": [{"sample_idx": idx} for idx in train_sample_indices],
                },
            }
        },
    }
    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a single-layer RePS steering vector for LatentMAS latent-state injection.")
    parser.add_argument("--clean_results_json", required=True)
    parser.add_argument("--attacked_results_json", required=True)
    parser.add_argument(
        "--context_results_json",
        default="",
        help=(
            "Result JSON whose saved args define the LatentMAS context used during training. "
            "Use the clean JSON when training an attack/degradation vector to inject into clean runs. "
            "Defaults to --attacked_results_json for backward compatibility."
        ),
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_name", default="", help="Override model path; defaults to context_results_json args.model_name.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target_role", default="planner")
    parser.add_argument("--layer", type=int, required=True, help="Single transformer layer to train.")
    parser.add_argument("--vector_type", choices=["hidden", "kv"], default="hidden", help="Train hidden-state or KV-cache RePS vectors.")
    parser.add_argument("--kv_mode", choices=["k_only", "v_only", "kv_both"], default="kv_both", help="For --vector_type kv, train K only, V only, or both.")
    parser.add_argument("--pair_filter", default="clean_correct_attack_wrong", choices=PAIR_FILTERS)
    parser.add_argument("--chosen_source", default="clean_raw", choices=OUTPUT_SOURCES)
    parser.add_argument("--rejected_source", default="attacked_raw", choices=OUTPUT_SOURCES)
    parser.add_argument("--max_pairs", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--simpo_scaler", type=float, default=1.0)
    parser.add_argument("--alpha_values", default="1,2,4,6", help="Comma-separated alpha values sampled during training.")
    parser.add_argument("--sub_loss_mode", default="neg", choices=["none", "neg"], help="Use -alpha vector with swapped preference as the bidirectional RePS sub loss.")
    parser.add_argument("--latent_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init_vector_path", default="")
    parser.add_argument("--init_scale", type=float, default=1.0)
    args = parser.parse_args()
    _load_runtime_dependencies()

    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    alpha_values = [float(x.strip()) for x in args.alpha_values.split(",") if x.strip()]
    if not alpha_values:
        raise ValueError("--alpha_values must contain at least one number")
    args.alpha_values = alpha_values

    set_seed(args.seed)
    random.seed(args.seed)
    context_results_json = args.context_results_json or args.attacked_results_json

    pairs = load_preference_pairs(
        clean_results_json=args.clean_results_json,
        attacked_results_json=args.attacked_results_json,
        pair_filter=args.pair_filter,
        chosen_source=args.chosen_source,
        rejected_source=args.rejected_source,
        max_pairs=args.max_pairs,
    )
    train_args = namespace_from_results(context_results_json, args)
    device = auto_device(args.device)
    model = ModelWrapper(train_args.model_name, device, use_vllm=False, args=train_args)
    freeze_model(model)
    method = LatentMASMethod(
        model,
        latent_steps=train_args.latent_steps,
        judger_max_new_tokens=train_args.max_new_tokens,
        temperature=train_args.temperature,
        top_p=train_args.top_p,
        generate_bs=args.batch_size,
        args=train_args,
    )

    layers = model._transformer_layers()
    num_layers = len(layers)
    if args.layer < 0 or args.layer >= num_layers:
        raise ValueError(f"--layer {args.layer} is out of range for model with {num_layers} layers")
    hidden_size = int(model.model.config.hidden_size)

    vector: Optional[torch.nn.Parameter] = None
    k_vector: Optional[torch.nn.Parameter] = None
    v_vector: Optional[torch.nn.Parameter] = None
    kv_shape: Optional[Tuple[int, int, int, int]] = None

    if args.vector_type == "hidden":
        init_vector = load_init_vector(args.init_vector_path, args.target_role, args.layer, hidden_size)
        if init_vector is None:
            init_vector = torch.empty(hidden_size, dtype=torch.float32).normal_(mean=0.0, std=1.0 / math.sqrt(hidden_size))
        init_vector = init_vector * args.init_scale
        vector = torch.nn.Parameter(init_vector.to(device=model.device, dtype=torch.float32))
        train_params = [vector]
    else:
        num_attention_heads = int(getattr(model.model.config, "num_attention_heads"))
        num_kv_heads = int(getattr(model.model.config, "num_key_value_heads", num_attention_heads))
        head_dim = int(getattr(model.model.config, "head_dim", hidden_size // num_attention_heads))
        kv_shape = (1, num_kv_heads, int(train_args.latent_steps), head_dim)
        init_k, init_v = load_init_kv_vectors(args.init_vector_path, args.target_role, args.layer)
        if init_k is None:
            init_k = torch.empty(kv_shape, dtype=torch.float32).normal_(mean=0.0, std=1.0 / math.sqrt(head_dim))
        if init_v is None:
            init_v = torch.empty(kv_shape, dtype=torch.float32).normal_(mean=0.0, std=1.0 / math.sqrt(head_dim))
        init_k = init_k * args.init_scale
        init_v = init_v * args.init_scale
        train_params = []
        if args.kv_mode in ("k_only", "kv_both"):
            k_vector = torch.nn.Parameter(init_k.to(device=model.device, dtype=torch.float32))
            train_params.append(k_vector)
        if args.kv_mode in ("v_only", "kv_both"):
            v_vector = torch.nn.Parameter(init_v.to(device=model.device, dtype=torch.float32))
            train_params.append(v_vector)
        if not train_params:
            raise ValueError("--kv_mode selected no trainable parameters")

    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.weight_decay)

    print(
        json.dumps(
            {
                "method": "reps",
                "target_role": args.target_role,
                "layer": args.layer,
                "vector_type": args.vector_type,
                "kv_mode": args.kv_mode if args.vector_type == "kv" else "",
                "num_pairs": len(pairs),
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "alpha_values": args.alpha_values,
                "sub_loss_mode": args.sub_loss_mode,
                "context_results_json": context_results_json,
                "chosen_source": args.chosen_source,
                "rejected_source": args.rejected_source,
                "attack": attack_metadata(train_args),
            },
            ensure_ascii=False,
        )
    )

    best_vector = vector.detach().clone() if vector is not None else None
    best_k_vector = k_vector.detach().clone() if k_vector is not None else None
    best_v_vector = v_vector.detach().clone() if v_vector is not None else None
    best_acc = -1.0
    for epoch in range(args.epochs):
        random.shuffle(pairs)
        progress = tqdm(range(0, len(pairs), args.batch_size), desc=f"epoch {epoch + 1}/{args.epochs}")
        epoch_losses: List[float] = []
        epoch_accs: List[float] = []
        for start in progress:
            batch = pairs[start:start + args.batch_size]
            alpha = random.choice(args.alpha_values)

            ref_chosen, ref_rejected, chosen_lens, rejected_lens = compute_reference_logps(method, batch)
            prompts, past_kv = build_judger_context(
                method,
                batch,
                target_role=args.target_role,
                target_layer=args.layer,
                vector=vector if vector is not None else torch.zeros(hidden_size, device=model.device),
                alpha=alpha,
                vector_type=args.vector_type,
                kv_mode=args.kv_mode,
                k_vector=k_vector,
                v_vector=v_vector,
            )
            chosen = [item["chosen"] for item in batch]
            rejected = [item["rejected"] for item in batch]
            chosen_logps, _ = model.sequence_logprob_batch(prompts, chosen, past_key_values=past_kv)
            rejected_logps, _ = model.sequence_logprob_batch(prompts, rejected, past_key_values=past_kv)
            loss = scaled_simpo_loss(
                chosen_logps=chosen_logps,
                rejected_logps=rejected_logps,
                ref_chosen_logps=ref_chosen,
                ref_rejected_logps=ref_rejected,
                chosen_lens=chosen_lens,
                rejected_lens=rejected_lens,
                simpo_scaler=args.simpo_scaler,
            )

            if args.sub_loss_mode == "neg":
                sub_prompts, sub_past_kv = build_judger_context(
                    method,
                    batch,
                    target_role=args.target_role,
                    target_layer=args.layer,
                    vector=vector if vector is not None else torch.zeros(hidden_size, device=model.device),
                    alpha=-alpha,
                    vector_type=args.vector_type,
                    kv_mode=args.kv_mode,
                    k_vector=k_vector,
                    v_vector=v_vector,
                )
                sub_chosen_logps, _ = model.sequence_logprob_batch(sub_prompts, rejected, past_key_values=sub_past_kv)
                sub_rejected_logps, _ = model.sequence_logprob_batch(sub_prompts, chosen, past_key_values=sub_past_kv)
                sub_loss = scaled_simpo_loss(
                    chosen_logps=sub_chosen_logps,
                    rejected_logps=sub_rejected_logps,
                    ref_chosen_logps=ref_rejected,
                    ref_rejected_logps=ref_chosen,
                    chosen_lens=rejected_lens,
                    rejected_lens=chosen_lens,
                    simpo_scaler=args.simpo_scaler,
                )
                loss = 0.5 * (loss + sub_loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(train_params, 1.0)
            optimizer.step()

            with torch.no_grad():
                acc = (chosen_logps / chosen_lens.to(chosen_logps.dtype) > rejected_logps / rejected_lens.to(rejected_logps.dtype)).float().mean().item()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_accs.append(acc)
            progress.set_postfix(loss=sum(epoch_losses) / len(epoch_losses), pref_acc=sum(epoch_accs) / len(epoch_accs))

        mean_acc = sum(epoch_accs) / max(1, len(epoch_accs))
        if mean_acc > best_acc:
            best_acc = mean_acc
            best_vector = vector.detach().clone() if vector is not None else None
            best_k_vector = k_vector.detach().clone() if k_vector is not None else None
            best_v_vector = v_vector.detach().clone() if v_vector is not None else None

    if args.vector_type == "hidden":
        save_reps_payload(
            output_path=args.output_path,
            vector=best_vector,
            num_layers=num_layers,
            hidden_size=hidden_size,
            target_role=args.target_role,
            target_layer=args.layer,
            args=args,
            num_pairs=len(pairs),
            train_sample_indices=[int(item["sample_idx"]) for item in pairs],
        )
    else:
        save_reps_kv_payload(
            output_path=args.output_path,
            k_vector=best_k_vector,
            v_vector=best_v_vector,
            num_layers=num_layers,
            kv_shape=kv_shape,
            target_role=args.target_role,
            target_layer=args.layer,
            args=args,
            num_pairs=len(pairs),
            train_sample_indices=[int(item["sample_idx"]) for item in pairs],
        )
    print(f"Saved RePS vector payload to {args.output_path} (best_train_pref_acc={best_acc:.4f})")


if __name__ == "__main__":
    main()
