import os
from typing import Dict, List, Optional, Tuple

from . import default_agents
from .cache_ops import slice_past_for_trace, truncate_past
from models import ModelWrapper, _past_length
from prompts import build_agent_message_sequential_latent_mas, build_agent_message_hierarchical_latent_mas
from utils import extract_binary_choice_answer, extract_gsm8k_answer, extract_markdown_python_block, extract_multiple_choice_answer, normalize_answer, run_with_timeout
import torch
import argparse

try:
    from vllm import SamplingParams
except ImportError:
    SamplingParams = None

try:
    from transformers.cache_utils import Cache
except ImportError:
    Cache = None

class LatentMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        latent_steps: int = 10,
        judger_max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:
        self.args = args
        self.model = model
        self.latent_steps = latent_steps
        self.judger_max_new_tokens = judger_max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.method_name = 'latent_mas'
        self.vllm_device = args.device 
        self.HF_device = args.device2
        self.latent_only = bool(getattr(args, "latent_only", False)) if args else False
        self.sequential_info_only = bool(getattr(args, "sequential_info_only", False)) if args else False

        if self.latent_only:
            self.sequential_info_only = True

        self.sampling_params = None
        if bool(getattr(args, "use_vllm", False)):
            if SamplingParams is None:
                raise ImportError("vLLM is required when --use_vllm is enabled. Install vllm or remove --use_vllm.")
            self.sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=args.max_new_tokens,
            )
        self.task = args.task
        self.trace_enabled = bool(getattr(args, "trace_export", False)) if args else False
        self.trace_save_hidden = bool(getattr(args, "trace_save_hidden", False)) if args else False
        self.trace_save_kv = bool(getattr(args, "trace_save_kv", False)) if args else False
        self.trace_roles = self._parse_trace_roles(getattr(args, "trace_roles", "planner,critic,refiner") if args else "planner,critic,refiner")
        self.trace_kv_mode = getattr(args, "trace_kv_mode", "latent_only") if args else "latent_only"
        self.trace_root = self._resolve_trace_root(args) if args else None
        self.state_injection_enabled = bool(getattr(args, "state_injection", False)) if args else False
        self.state_injection_role = (getattr(args, "state_injection_role", "planner") or "planner").strip().lower() if args else "planner"
        self.state_injection_layers = self._normalize_state_layers(getattr(args, "state_injection_layers", "last") if args else "last")
        self.state_injection_alpha = float(getattr(args, "state_injection_alpha", 0.0)) if args else 0.0
        self.state_injection_vector = self._load_state_injection_vector(args) if args else None
        self.kv_injection_enabled = bool(getattr(args, "kv_injection", False)) if args else False
        self.kv_injection_edge = self._normalize_kv_edge(getattr(args, "kv_injection_edge", "") if args else "")
        self.kv_injection_role = self._kv_source_role_from_args(args) if args else "planner"
        self.kv_injection_mode = self._normalize_kv_mode(getattr(args, "kv_injection_mode", "kv_both") if args else "kv_both")
        self.kv_injection_layers = self._normalize_kv_layers(getattr(args, "kv_injection_layers", "all") if args else "all")
        self.kv_injection_position = self._normalize_kv_position(getattr(args, "kv_injection_position", "all") if args else "all")
        self.kv_injection_alpha_k = float(getattr(args, "kv_injection_alpha_k", 1.0)) if args else 1.0
        self.kv_injection_alpha_v = float(getattr(args, "kv_injection_alpha_v", 1.0)) if args else 1.0
        self.kv_injection_payload = self._load_kv_injection_payload(args) if args else None
        if self.state_injection_enabled and bool(getattr(args, "use_vllm", False)):
            raise NotImplementedError("state_injection is currently supported only on the standard HF backend. Remove --use_vllm for hidden-state injection.")
        if self.kv_injection_enabled and bool(getattr(args, "use_vllm", False)):
            raise NotImplementedError("kv_injection is currently supported only on the standard HF backend. Remove --use_vllm for KV injection.")

    @staticmethod
    def _parse_trace_roles(raw_roles: str) -> Optional[set]:
        raw = (raw_roles or "").strip().lower()
        if not raw or raw == "all":
            return None
        return {part.strip() for part in raw.split(",") if part.strip()}

    def _resolve_trace_root(self, args: argparse.Namespace) -> Optional[str]:
        if not self.trace_enabled:
            return None
        trace_root = (getattr(args, "trace_output_dir", "") or "").strip()
        if trace_root:
            return os.path.abspath(trace_root)

        output_path = (getattr(args, "output_path", "") or "").strip()
        if output_path:
            output_abs = os.path.abspath(output_path)
            output_dir = os.path.dirname(output_abs)
            stem = os.path.splitext(os.path.basename(output_abs))[0]
            return os.path.join(output_dir, f"{stem}_traces")

        return os.path.abspath(os.path.join("trace_outputs", f"{self.task}_seed{self.args.seed}"))

    @staticmethod
    def _normalize_state_layers(layer_spec: str) -> str:
        layer_spec = (layer_spec or "last").strip().lower()
        if layer_spec in {"last", "all"}:
            return layer_spec
        for chunk in layer_spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                raise ValueError("--state_injection_layers contains an empty layer entry")
            if "-" in chunk:
                start, end = chunk.split("-", 1)
                if not start.strip().isdigit() or not end.strip().isdigit():
                    raise ValueError(
                        "--state_injection_layers ranges must use non-negative integer bounds, e.g. 12-23"
                    )
                if int(start) > int(end):
                    raise ValueError("--state_injection_layers range start must be <= range end")
            elif not chunk.isdigit():
                raise ValueError(
                    "Unsupported --state_injection_layers. Use last/all, a 0-based layer index like 18, "
                    "a comma list like 16,18,20, or a range like 12-23."
                )
        return layer_spec

    @staticmethod
    def _state_layer_indices(num_layers: int, layer_spec: str) -> List[int]:
        if layer_spec == "last":
            return [num_layers - 1]
        if layer_spec == "all":
            return list(range(num_layers))

        selected = []
        for chunk in layer_spec.split(","):
            chunk = chunk.strip()
            if "-" in chunk:
                start_str, end_str = chunk.split("-", 1)
                selected.extend(range(int(start_str), int(end_str) + 1))
            else:
                selected.append(int(chunk))

        unique_selected = sorted(set(selected))
        invalid = [idx for idx in unique_selected if idx < 0 or idx >= num_layers]
        if invalid:
            raise ValueError(
                f"--state_injection_layers contains out-of-range layer(s) {invalid}; "
                f"valid range is 0..{num_layers - 1}"
            )
        return unique_selected

    def _load_state_injection_vector(self, args: argparse.Namespace) -> Optional[object]:
        if not self.state_injection_enabled:
            return None
        vector_path = (getattr(args, "state_injection_vector_path", "") or "").strip()
        if not vector_path:
            raise ValueError("--state_injection requires --state_injection_vector_path")
        payload = torch.load(vector_path, map_location="cpu")
        role = self.state_injection_role
        if "roles" not in payload or role not in payload["roles"]:
            raise ValueError(f"Role '{role}' not found in state injection payload: {vector_path}")
        role_payload = payload["roles"][role]
        if "hidden" not in role_payload or "vector" not in role_payload["hidden"]:
            raise ValueError(f"No hidden vector found for role '{role}' in: {vector_path}")
        hidden_payload = role_payload["hidden"]
        if "layer_vectors" in hidden_payload:
            layer_vectors = hidden_payload["layer_vectors"]
            if not isinstance(layer_vectors, list) or not layer_vectors:
                raise ValueError(f"Invalid layer_vectors in hidden payload: {vector_path}")
            selected_layers = self._state_layer_indices(len(layer_vectors), self.state_injection_layers)
            return {
                "type": "layer",
                "path": vector_path,
                "layer_vectors": {
                    idx: layer_vectors[idx].to(torch.float32).cpu().contiguous()
                    for idx in selected_layers
                },
            }

        vector = hidden_payload["vector"]
        if vector.dim() == 2:
            layer_vectors = [vector[idx] for idx in range(vector.shape[0])]
            selected_layers = self._state_layer_indices(len(layer_vectors), self.state_injection_layers)
            return {
                "type": "layer",
                "path": vector_path,
                "layer_vectors": {
                    idx: layer_vectors[idx].to(torch.float32).cpu().contiguous()
                    for idx in selected_layers
                },
            }
        if vector.dim() != 1:
            raise ValueError(f"Expected 1D or layer-wise 2D hidden vector, got shape {tuple(vector.shape)} from {vector_path}")
        return {
            "type": "last_hidden",
            "path": vector_path,
            "vector": vector.to(torch.float32).cpu().contiguous(),
        }

    @staticmethod
    def _normalize_kv_mode(mode: str) -> str:
        mode = (mode or "kv_both").strip().lower()
        valid = {"k_only", "v_only", "kv_both"}
        if mode not in valid:
            raise ValueError(f"Unsupported kv injection mode: {mode}. Expected one of {sorted(valid)}")
        return mode

    @staticmethod
    def _normalize_kv_layers(layer_spec: str) -> str:
        layer_spec = (layer_spec or "all").strip().lower()
        if layer_spec == "injection":
            return layer_spec
        if layer_spec == "mid":
            layer_spec = "middle"
        valid = {"all", "shallow", "middle", "deep"}
        if layer_spec in valid:
            return layer_spec

        for chunk in layer_spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                raise ValueError("--kv_injection_layers contains an empty layer entry")
            if "-" in chunk:
                start, end = chunk.split("-", 1)
                if not start.strip().isdigit() or not end.strip().isdigit():
                    raise ValueError(
                        "--kv_injection_layers ranges must use non-negative integer bounds, e.g. 12-23"
                    )
                if int(start) > int(end):
                    raise ValueError("--kv_injection_layers range start must be <= range end")
            elif not chunk.isdigit():
                raise ValueError(
                    "Unsupported --kv_injection_layers. Use all/shallow/middle/deep, "
                    "a 0-based layer index like 18, a comma list like 16,18,20, or a range like 12-23."
                )
        return layer_spec

    @staticmethod
    def _normalize_kv_edge(edge_spec: str) -> str:
        edge_spec = (edge_spec or "").strip().lower().replace(" ", "")
        if not edge_spec:
            return ""
        edge_spec = edge_spec.replace("_to_", "->").replace("=>", "->")
        valid_edges = {"planner->critic", "critic->refiner", "refiner->judger"}
        if edge_spec not in valid_edges:
            raise ValueError(f"Unsupported kv injection edge: {edge_spec}. Expected one of {sorted(valid_edges)}")
        return edge_spec

    def _kv_source_role_from_args(self, args: argparse.Namespace) -> str:
        if self.kv_injection_edge:
            return self.kv_injection_edge.split("->", 1)[0]
        return (getattr(args, "kv_injection_role", "planner") or "planner").strip().lower()

    @staticmethod
    def _normalize_kv_position(position: str) -> str:
        position = (position or "all").strip().lower()
        if position in {"all", "first", "last"}:
            return position
        try:
            idx = int(position)
        except ValueError as exc:
            raise ValueError("--kv_injection_position must be all, first, last, or a 0-based integer") from exc
        if idx < 0:
            raise ValueError("--kv_injection_position index must be non-negative")
        return str(idx)

    def _select_kv_injection_position(self, tensor: torch.Tensor) -> torch.Tensor:
        position = self.kv_injection_position
        if position == "all":
            return tensor

        seq_len = tensor.shape[-2]
        if seq_len <= 0:
            return tensor
        if position == "first":
            idx = 0
        elif position == "last":
            idx = seq_len - 1
        else:
            idx = int(position)
            if idx >= seq_len:
                raise ValueError(
                    f"--kv_injection_position {idx} is out of range for vector length {seq_len}"
                )

        masked = torch.zeros_like(tensor)
        masked[..., idx:idx + 1, :] = tensor[..., idx:idx + 1, :]
        return masked

    def _load_kv_injection_payload(self, args: argparse.Namespace) -> Optional[Dict[str, List[torch.Tensor]]]:
        if not self.kv_injection_enabled:
            return None
        vector_path = (getattr(args, "kv_injection_vector_path", "") or "").strip()
        if not vector_path:
            raise ValueError("--kv_injection requires --kv_injection_vector_path")
        payload = torch.load(vector_path, map_location="cpu")
        role = self.kv_injection_role
        if "roles" not in payload or role not in payload["roles"]:
            raise ValueError(f"Role '{role}' not found in kv injection payload: {vector_path}")
        role_payload = payload["roles"][role]
        if "kv" not in role_payload:
            raise ValueError(f"No kv payload found for role '{role}' in: {vector_path}")
        kv_payload = role_payload["kv"]
        k_vectors = kv_payload.get("k_vectors")
        v_vectors = kv_payload.get("v_vectors")
        if not isinstance(k_vectors, list) or not isinstance(v_vectors, list) or not k_vectors or not v_vectors:
            raise ValueError(f"Invalid kv vectors in payload: {vector_path}")
        if len(k_vectors) != len(v_vectors):
            raise ValueError(f"k/v layer count mismatch in payload: {vector_path}")
        return {
            "path": vector_path,
            "k_vectors": [tensor.to(torch.float32).cpu().contiguous() for tensor in k_vectors],
            "v_vectors": [tensor.to(torch.float32).cpu().contiguous() for tensor in v_vectors],
        }

    def _should_export_role(self, role: str) -> bool:
        if not self.trace_enabled:
            return False
        return self.trace_roles is None or role in self.trace_roles

    def _state_injection_vector_for_role(self, role: str) -> Optional[torch.Tensor]:
        if not self.state_injection_enabled:
            return None
        if role.strip().lower() != self.state_injection_role:
            return None
        if isinstance(self.state_injection_vector, dict) and self.state_injection_vector.get("type") == "last_hidden":
            return self.state_injection_vector["vector"]
        return None

    def _state_layer_injection_vectors_for_role(self, role: str) -> Optional[Dict[int, torch.Tensor]]:
        if not self.state_injection_enabled:
            return None
        if role.strip().lower() != self.state_injection_role:
            return None
        if isinstance(self.state_injection_vector, dict) and self.state_injection_vector.get("type") == "layer":
            return self.state_injection_vector["layer_vectors"]
        return None

    def _kv_injection_payload_for_role(self, role: str) -> Optional[Dict[str, List[torch.Tensor]]]:
        if not self.kv_injection_enabled:
            return None
        if role.strip().lower() != self.kv_injection_role:
            return None
        return self.kv_injection_payload

    @staticmethod
    def _layer_indices_for_band(num_layers: int, band: str) -> List[int]:
        if band == "injection":
            raise ValueError("'injection' must be resolved before calling _layer_indices_for_band")
        if band == "all":
            return list(range(num_layers))

        third = max(1, num_layers // 3)
        if band == "shallow":
            return list(range(0, third))
        if band == "middle":
            start = third
            end = min(num_layers, 2 * third)
            return list(range(start, end))
        if band == "deep":
            start = min(num_layers, 2 * third)
            return list(range(start, num_layers))

        selected = []
        for chunk in band.split(","):
            chunk = chunk.strip()
            if "-" in chunk:
                start_str, end_str = chunk.split("-", 1)
                start = int(start_str)
                end = int(end_str)
                selected.extend(range(start, end + 1))
            else:
                selected.append(int(chunk))

        unique_selected = sorted(set(selected))
        invalid = [idx for idx in unique_selected if idx < 0 or idx >= num_layers]
        if invalid:
            raise ValueError(
                f"--kv_injection_layers contains out-of-range layer(s) {invalid}; "
                f"valid range is 0..{num_layers - 1}"
            )
        return unique_selected

    def _apply_kv_injection(
        self,
        past_kv: Optional[Tuple],
        *,
        role: str,
    ) -> Optional[Tuple]:
        payload = self._kv_injection_payload_for_role(role)
        if payload is None or past_kv is None:
            return past_kv

        if Cache is not None and isinstance(past_kv, Cache):
            legacy = past_kv.to_legacy_cache()
            return_cache_cls = past_kv.__class__
        else:
            legacy = past_kv
            return_cache_cls = None

        num_layers = len(legacy)
        if num_layers != len(payload["k_vectors"]):
            raise ValueError(
                f"KV layer count mismatch: runtime cache has {num_layers} layers but vector file has {len(payload['k_vectors'])}"
            )

        target_layers = set(self._layer_indices_for_band(num_layers, self.kv_injection_layers))
        updated_layers = []

        for layer_idx, layer in enumerate(legacy):
            if not isinstance(layer, tuple):
                updated_layers.append(layer)
                continue
            k_cache, v_cache = layer
            if layer_idx not in target_layers:
                updated_layers.append((k_cache, v_cache))
                continue

            k_vec = payload["k_vectors"][layer_idx].to(device=k_cache.device, dtype=k_cache.dtype)
            v_vec = payload["v_vectors"][layer_idx].to(device=v_cache.device, dtype=v_cache.dtype)
            k_vec = self._select_kv_injection_position(k_vec)
            v_vec = self._select_kv_injection_position(v_vec)

            inject_len = k_vec.shape[-2]
            if k_cache.shape[-2] < inject_len or v_cache.shape[-2] < inject_len:
                raise ValueError(
                    f"Cache too short for KV injection at layer {layer_idx}: "
                    f"cache_len={k_cache.shape[-2]}, inject_len={inject_len}"
                )

            if k_vec.shape[0] == 1 and k_cache.shape[0] != 1:
                k_vec = k_vec.expand(k_cache.shape[0], -1, -1, -1)
            if v_vec.shape[0] == 1 and v_cache.shape[0] != 1:
                v_vec = v_vec.expand(v_cache.shape[0], -1, -1, -1)

            k_updated = k_cache.clone()
            v_updated = v_cache.clone()
            suffix = slice(k_cache.shape[-2] - inject_len, k_cache.shape[-2])

            if self.kv_injection_mode in ("k_only", "kv_both"):
                k_updated[..., suffix, :] = k_updated[..., suffix, :] + self.kv_injection_alpha_k * k_vec
            if self.kv_injection_mode in ("v_only", "kv_both"):
                v_updated[..., suffix, :] = v_updated[..., suffix, :] + self.kv_injection_alpha_v * v_vec

            updated_layers.append((k_updated, v_updated))

        updated_layers = tuple(updated_layers)
        if return_cache_cls is not None:
            return return_cache_cls.from_legacy_cache(updated_layers)
        return updated_layers

    @staticmethod
    def _cpu_hidden_trace(hidden_trace: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            key: value.detach().to(torch.float32).cpu().contiguous()
            for key, value in hidden_trace.items()
        }

    def _export_agent_trace(
        self,
        *,
        sample_idx: int,
        item: Dict,
        agent,
        attack_tag: str,
        hidden_trace: Optional[Dict[str, torch.Tensor]],
        kv_trace: Optional[Tuple],
        prev_past_len: int,
        total_past_len: int,
    ) -> Optional[str]:
        if not self._should_export_role(agent.role):
            return None
        if not self.trace_root:
            return None

        sample_dir = os.path.join(self.trace_root, f"sample_{sample_idx:05d}")
        os.makedirs(sample_dir, exist_ok=True)
        file_path = os.path.join(sample_dir, f"{agent.role}_{attack_tag}.pt")

        payload = {
            "sample_idx": sample_idx,
            "question": item.get("question", ""),
            "gold": item.get("gold", ""),
            "role": agent.role,
            "agent_name": agent.name,
            "attack": {
                "surface": getattr(self.args, "attack_surface", "none"),
                "type": getattr(self.args, "attack_type", "none"),
                "target_role": getattr(self.args, "attack_target_role", "none"),
                "attack_id": getattr(self.args, "attack_id", ""),
            },
            "trace_meta": {
                "latent_steps": self.latent_steps,
                "prev_past_len": prev_past_len,
                "total_past_len": total_past_len,
                "saved_kv_mode": self.trace_kv_mode if kv_trace is not None else "none",
            },
        }

        if hidden_trace is not None and self.trace_save_hidden:
            payload["hidden_trace"] = self._cpu_hidden_trace(hidden_trace)
        if kv_trace is not None and self.trace_save_kv:
            payload["kv_trace"] = kv_trace

        torch.save(payload, file_path)
        return file_path

    def _attack_tag(self) -> str:
        surface = getattr(self.args, "attack_surface", "none")
        attack_type = getattr(self.args, "attack_type", "none")
        attack_id = getattr(self.args, "attack_id", "") or "none"
        target_role = getattr(self.args, "attack_target_role", "none")
        return f"{surface}__{attack_type}__{target_role}__{attack_id}"

    @torch.no_grad()
    def run_batch(self, items: List[Dict], sample_offset: int = 0) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]
        attack_tag = self._attack_tag()

        for agent in self.agents:

            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args, gold=item.get("gold", ""))
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args, gold=item.get("gold", ""))
                    for item in items
                ]


            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                need_hidden_trace = self.trace_enabled and self.trace_save_hidden
                latent_generate_kwargs = {
                    "latent_steps": self.latent_steps,
                    "past_key_values": past_kv,
                    "return_trace": need_hidden_trace,
                    "latent_injection_vector": self._state_injection_vector_for_role(agent.role),
                    "latent_layer_injection_vectors": self._state_layer_injection_vectors_for_role(agent.role),
                    "latent_injection_alpha": self.state_injection_alpha,
                }
                past_kv = self.model.generate_latent_batch(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    **latent_generate_kwargs,
                )
                hidden_trace_batch = None
                if need_hidden_trace:
                    past_kv, hidden_trace_batch = past_kv
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = truncate_past(past_kv, tokens_to_keep)
                past_kv = self._apply_kv_injection(
                    past_kv,
                    role=agent.role,
                )
                total_past_len = _past_length(past_kv)
                kv_trace_batch = None
                if self.trace_enabled and self.trace_save_kv:
                    kv_trace_batch = slice_past_for_trace(
                        past_kv,
                        prev_past_len=prev_past_len,
                        latent_steps=self.latent_steps,
                        mode=self.trace_kv_mode,
                    )

                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    hidden_trace = None
                    if hidden_trace_batch is not None:
                        hidden_trace = {
                            key: value[idx]
                            for key, value in hidden_trace_batch.items()
                        }
                    kv_trace = None
                    if kv_trace_batch is not None:
                        kv_trace = tuple(
                            tuple(t[idx:idx + 1] for t in layer)
                            for layer in kv_trace_batch
                        )
                    trace_path = self._export_agent_trace(
                        sample_idx=sample_offset + idx,
                        item=items[idx],
                        agent=agent,
                        attack_tag=attack_tag,
                        hidden_trace=hidden_trace,
                        kv_trace=kv_trace,
                        prev_past_len=prev_past_len,
                        total_past_len=total_past_len,
                    )
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                            "trace_path": trace_path,
                        }
                    )
            else:

                past_for_decoding = past_kv if self.latent_steps > 0 else None

                if self.args.think:
                        judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                judger_ids = judger_encoded["input_ids"].to(self.model.device)
                judger_mask = judger_encoded["attention_mask"].to(self.model.device)
                judger_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(judger_ids, judger_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    judger_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))
                hidden_trace_batch = None
                if self._should_export_role(agent.role) and self.trace_save_hidden:
                    hidden_trace_batch = {
                        "prompt_last_hidden": self.model.prompt_last_hidden_batch(
                            judger_ids,
                            judger_mask,
                            past_key_values=past_for_decoding,
                        )
                    }
                generated_batch, _ = self.model.generate_text_batch(
                    judger_ids,
                    judger_mask,
                    max_new_tokens=self.judger_max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    past_key_values=past_for_decoding,
                )
                for idx in range(batch_size):
                    final_text = generated_batch[idx].strip()
                    final_texts[idx] = final_text
                    mask = judger_mask[idx].bool()
                    trimmed_ids = judger_ids[idx][mask].to("cpu").tolist()
                    trace_path = None
                    if self._should_export_role(agent.role):
                        hidden_trace = None
                        if hidden_trace_batch is not None:
                            hidden_trace = {
                                key: value[idx]
                                for key, value in hidden_trace_batch.items()
                            }
                        trace_path = self._export_agent_trace(
                            sample_idx=sample_offset + idx,
                            item=items[idx],
                            agent=agent,
                            attack_tag=attack_tag,
                            hidden_trace=hidden_trace,
                            kv_trace=None,
                            prev_past_len=_past_length(past_for_decoding),
                            total_past_len=_past_length(past_for_decoding),
                        )
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": judger_tokens_batch[idx],
                            "output": final_text,
                            "trace_path": trace_path,
                        }
                    )

        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ['mbppplus', 'humanevalplus']:
                pred = extract_markdown_python_block(final_text)
                gold = item.get("gold", "")

                if pred is None:
                    ok = False
                    error_msg = "python error: No python code block found"
                else:
                    python_code_to_exe = pred + "\n" + gold
                    ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
                
                print(f'=========================================')
                print(f'Question {idx}')
                print(f'error_msg: {error_msg}')
                # print(f'=========================================')

            elif self.task in ["aime2024", "aime2025"]:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = str(item.get("gold", "")).strip()
                try:
                    pred_int = int(pred)
                    gold_int = int(gold)
                    ok = (pred_int == gold_int)
                    error_msg = None
                except ValueError:
                    ok = False
                    error_msg = f'Value error in parsing answer. Pred: {pred}, Gold: {gold}'

            elif self.task in ["arc_easy", "arc_challenge", "openbookqa", "gpqa", "medqa"]:
                pred = normalize_answer(extract_multiple_choice_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            elif self.task in ["winogrande"]:
                pred = normalize_answer(extract_binary_choice_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None

            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
                gold = item.get("gold", "")
                ok = (pred == gold) if (pred and gold) else False
                error_msg = None
            
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results
    
    def run_batch_vllm(self, items: List[Dict], sample_offset: int = 0) -> List[Dict]:
        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")
        if self.trace_enabled:
            raise NotImplementedError("trace_export is currently supported only on the standard HF backend. Remove --use_vllm for trace collection.")

        batch_size = len(items)
        past_kv: Optional[Tuple] = None
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        embedding_record = []
        for agent in self.agents:
            
            if self.args.prompt == "sequential":
                batch_messages = [
                    build_agent_message_sequential_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args, gold=item.get("gold", ""))
                    for item in items
                ]
            elif self.args.prompt == "hierarchical":
                batch_messages = [
                    build_agent_message_hierarchical_latent_mas(role=agent.role, question=item["question"], context="", method=self.method_name, args=self.args, gold=item.get("gold", ""))
                    for item in items
                ]
                
            prompts, input_ids, attention_mask, tokens_batch = self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

            if agent.role != "judger":
                prev_past_len = _past_length(past_kv)

                # to wrap all latent thoughts from previous agents
                if self.args.think:
                        wrapped_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    wrapped_prompts = prompts

                wrapped_encoded = self.model.tokenizer(
                    wrapped_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
                wrapped_ids = wrapped_encoded["input_ids"].to(self.model.HF_device)
                wrapped_mask = wrapped_encoded["attention_mask"].to(self.model.HF_device)
                wrapped_tokens_batch: List[List[str]] = []
                for ids_row, mask_row in zip(wrapped_ids, wrapped_mask):
                    active_ids = ids_row[mask_row.bool()].tolist()
                    wrapped_tokens_batch.append(self.model.tokenizer.convert_ids_to_tokens(active_ids))

                past_kv, previous_hidden_embedding = self.model.generate_latent_batch_hidden_state(
                    wrapped_ids,
                    attention_mask=wrapped_mask,
                    latent_steps=self.latent_steps,
                    past_key_values=past_kv,
                )
                if self.sequential_info_only or self.latent_only:
                    new_past_len = _past_length(past_kv)
                    tokens_added = new_past_len - prev_past_len
                    tokens_to_keep = self.latent_steps if self.latent_only else tokens_added
                    past_kv = truncate_past(past_kv, tokens_to_keep)

                if self.latent_only:
                    if self.latent_steps > 0:
                        previous_hidden_embedding = previous_hidden_embedding[:, -self.latent_steps:, :]
                    else:
                        previous_hidden_embedding = previous_hidden_embedding[:, 0:0, :]

                embedding_record.append(previous_hidden_embedding)

                if self.sequential_info_only or self.latent_only:
                    embedding_record = embedding_record[-1:]
                
                for idx in range(batch_size):
                    mask = wrapped_mask[idx].bool()
                    trimmed_ids = wrapped_ids[idx][mask].to("cpu").tolist()
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": wrapped_prompts[idx],
                            "input_ids": trimmed_ids,
                            "input_tokens": wrapped_tokens_batch[idx],
                            "latent_steps": self.latent_steps,
                            "output": "",
                        }
                    )
            else:
                
                # A stack of [B, L_i, H]
                past_embedding = torch.cat(embedding_record, dim=1).to(self.vllm_device)
                
                if self.args.think:
                    judger_prompts = [f"{prompt}<think>" for prompt in prompts]
                else: 
                    judger_prompts = prompts
                
                judger_encoded = self.model.tokenizer(
                    judger_prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                ) 
                judger_encoded = judger_encoded["input_ids"].to(self.model.HF_device)
                # Get current prompt embedding
                curr_prompt_emb = self.model.embedding_layer(judger_encoded).squeeze(0).to(self.vllm_device)
                
                # assert Qwen model
                assert "Qwen" in self.args.model_name or "qwen" in self.args.model_name, "latent_embedding_position is only supported for Qwen models currently."

                # handle latent embedding insertion position    
                len_of_left = []
                for p in judger_prompts:
                    idx = p.find("<|im_start|>user\n")
                    # Get the text up to and including "<|im_start|>user\n"
                    left = p[: idx + len("<|im_start|>user\n")]
                    len_of_left.append(len(self.model.tokenizer(left)['input_ids']))
                    
                B, L, H = curr_prompt_emb.shape
                _, Lp, H = past_embedding.shape  # assume shape consistency
                    
                whole_prompt_emb_list = []
                for i in range(B):
                    insert_idx = len_of_left[i]
                    left_emb = curr_prompt_emb[i, :insert_idx, :]
                    right_emb = curr_prompt_emb[i, insert_idx:, :]
                    combined = torch.cat([left_emb, past_embedding[i], right_emb], dim=0)
                    whole_prompt_emb_list.append(combined)

                # Pad back to max length if needed
                max_len = max(x.shape[0] for x in whole_prompt_emb_list)
                whole_prompt_emb = torch.stack([
                    torch.cat([x, torch.zeros(max_len - x.shape[0], H, device=x.device)], dim=0)
                    for x in whole_prompt_emb_list
                ])

                # else:
                    # Get full prompt embedding from cat with previous ones 
                    # B L H B L H
                    # whole_prompt_emb = torch.cat([past_embedding, curr_prompt_emb], dim=1)
                
                # pdb.set_trace()              
                
                # Use vLLM 
                prompt_embeds_list = [
                    {
                        "prompt_embeds": embeds
                    } for embeds in whole_prompt_emb 
                ]
                
                
                outputs = self.model.vllm_engine.generate(
                    prompt_embeds_list,
                    self.sampling_params,
                )

                generated_texts = [out.outputs[0].text.strip() for out in outputs]
                    
                for idx in range(batch_size):
                    text_out = generated_texts[idx].strip()
                    final_texts[idx] = text_out
                    agent_traces[idx].append(
                        {
                            "name": agent.name,
                            "role": agent.role,
                            "input": judger_prompts[idx],
                            "output": text_out,
                        }
                    )


        results: List[Dict] = []
        for idx, item in enumerate(items):
            final_text = final_texts[idx]
            if self.task in ["arc_easy", "arc_challenge", "openbookqa", "gpqa", "medqa"]:
                pred = normalize_answer(extract_multiple_choice_answer(final_text))
            elif self.task in ["winogrande"]:
                pred = normalize_answer(extract_binary_choice_answer(final_text))
            else:
                pred = normalize_answer(extract_gsm8k_answer(final_text))
            gold = item["gold"]
            ok = (pred == gold) if (pred and gold) else False
            results.append(
                {
                    "question": item["question"],
                    "gold": gold,
                    "solution": item["solution"],
                    "prediction": pred,
                    "raw_prediction": final_text,
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )
        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
