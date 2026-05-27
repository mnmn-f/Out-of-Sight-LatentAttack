import os
import csv
import torch
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from vllm import LLM, SamplingParams
    _HAS_VLLM = True
except ImportError:
    _HAS_VLLM = False


def _ensure_pad_token(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.padding_side = "left"


def _past_length(past_key_values: Optional[Tuple]) -> int:
    if not past_key_values:
        return 0
    k = past_key_values[0][0]
    return k.shape[-2]


def _past_requires_grad(past_key_values: Optional[Tuple]) -> bool:
    if not past_key_values:
        return False
    for layer in past_key_values:
        if isinstance(layer, tuple):
            for tensor in layer:
                if torch.is_tensor(tensor) and tensor.requires_grad:
                    return True
        elif torch.is_tensor(layer) and layer.requires_grad:
            return True
    return False


class ModelWrapper:
    def __init__(self, model_name: str, device: torch.device, use_vllm: bool = False, args = None):
        self.model_name = model_name
        self.device = device
        if use_vllm and not _HAS_VLLM:
            raise ImportError("vLLM is required when --use_vllm is enabled. Install vllm or remove --use_vllm.")
        self.use_vllm = use_vllm
        self.vllm_engine = None
        self.latent_space_realign = bool(getattr(args, "latent_space_realign", False)) if args else False
        self._latent_realign_matrices: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.args = args

        # for ablation
        self.pre_aligned = None

        if self.use_vllm:
            
            tp_size = max(1, int(getattr(args, "tensor_parallel_size", 1)))
            gpu_util = float(getattr(args, "gpu_memory_utilization", 0.9))
            
            print(f"[vLLM] Using vLLM backend for model {model_name}")
            if args.enable_prefix_caching and args.method == "latent_mas": 
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util, enable_prefix_caching=True, enable_prompt_embeds=True)
            else:
                self.vllm_engine = LLM(model=model_name, tensor_parallel_size=tp_size, gpu_memory_utilization=gpu_util)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            use_second_hf = bool(getattr(args, "use_second_HF_model", False)) if args else False
            if use_second_hf:
                self.HF_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
                ).to(args.device2).eval() 
                self.embedding_layer = self.HF_model.get_input_embeddings()
                self.HF_device = args.device2
                # if self.latent_space_realign:
                self._ensure_latent_realign_matrix(self.HF_model, torch.device(self.HF_device), args)
            elif self.latent_space_realign:
                raise ValueError("latent_space_realign requires --use_second_HF_model when using vLLM backend.")
            _ensure_pad_token(self.tokenizer)
            return  # skip loading transformers model

        # fallback: normal transformers path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        _ensure_pad_token(self.tokenizer)
        with torch.no_grad():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            )
        if len(self.tokenizer) != self.model.get_input_embeddings().weight.shape[0]:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(device)
        self.model.eval()
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = True
        if self.latent_space_realign:
            self._ensure_latent_realign_matrix(self.model, self.device, args)

    def render_chat(self, messages: List[Dict], add_generation_prompt: bool = True) -> str:
        tpl = getattr(self.tokenizer, "chat_template", None)
        if tpl:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        segments = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            segments.append(f"<|{role}|>\n{content}\n</|{role}|>")
        if add_generation_prompt:
            segments.append("<|assistant|>")
        return "\n".join(segments)

    def prepare_chat_input(
        self, messages: List[Dict], add_generation_prompt: bool = True
    ) -> Tuple[str, torch.Tensor, torch.Tensor, List[str]]:
        prompt_text = self.render_chat(messages, add_generation_prompt=add_generation_prompt)
        encoded = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        active_ids = input_ids[0][attention_mask[0].bool()].tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(active_ids)
        return prompt_text, input_ids, attention_mask, tokens

    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        prompts: List[str] = []
        for messages in batch_messages:
            prompts.append(self.render_chat(messages, add_generation_prompt=add_generation_prompt))
        encoded = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        tokens_batch: List[List[str]] = []
        for ids_row, mask_row in zip(input_ids, attention_mask):
            active_ids = ids_row[mask_row.bool()].tolist()
            tokens_batch.append(self.tokenizer.convert_ids_to_tokens(active_ids))
        return prompts, input_ids, attention_mask, tokens_batch

    def vllm_generate_text_batch(
        self,
        prompts: List[str],
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> List[str]:
        if not self.vllm_engine:
            raise RuntimeError("vLLM engine not initialized. Pass use_vllm=True to ModelWrapper.")
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
        )
        outputs = self.vllm_engine.generate(prompts, sampling_params)
        generations = [out.outputs[0].text.strip() for out in outputs]
        return generations
    
    def _build_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        input_embeds = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        output_embeds = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if output_embeds is None:
            output_embeds = getattr(model, "lm_head", None)
        if (
            input_embeds is None
            or output_embeds is None
            or not hasattr(input_embeds, "weight")
            or not hasattr(output_embeds, "weight")
        ):
            raise RuntimeError("Cannot build latent realignment matrix: embedding weights not accessible.")
        input_weight = input_embeds.weight.detach().to(device=device, dtype=torch.float32)
        output_weight = output_embeds.weight.detach().to(device=device, dtype=torch.float32)
        gram = torch.matmul(output_weight.T, output_weight)
        reg = 1e-5 * torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        gram = gram + reg
        rhs = torch.matmul(output_weight.T, input_weight)
        realign_matrix = torch.linalg.solve(gram, rhs)
        target_norm = input_weight.norm(dim=1).mean().detach()

        if self.args.latent_space_realign:
            pass
        else:
            # keep the matrix, for further normalization
            realign_matrix = torch.eye(realign_matrix.shape[0], device=realign_matrix.device, dtype=realign_matrix.dtype)

        return realign_matrix, target_norm

    def _ensure_latent_realign_matrix(self, model, device, args) -> Tuple[torch.Tensor, torch.Tensor]:
        key = id(model)
        info = self._latent_realign_matrices.get(key)
        target_device = torch.device(device)

        if info is None:
            matrix, target_norm = self._build_latent_realign_matrix(model, target_device, args)
        else:
            matrix, target_norm = info
            if matrix.device != target_device:
                matrix = matrix.to(target_device)

        target_norm = target_norm.to(device=target_device, dtype=matrix.dtype) if isinstance(target_norm, torch.Tensor) else torch.as_tensor(target_norm, device=target_device, dtype=matrix.dtype)
        self._latent_realign_matrices[key] = (matrix, target_norm)

        return matrix, target_norm

    def _apply_latent_realignment(self, hidden: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        matrix, target_norm = self._ensure_latent_realign_matrix(model, hidden.device, self.args)
        hidden_fp32 = hidden.to(torch.float32)
        aligned = torch.matmul(hidden_fp32, matrix)

        aligned_norm = aligned.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pre_aligned = aligned.detach().clone()
        self.pre_aligned = pre_aligned
        aligned = aligned * (target_norm / aligned_norm)
        return aligned.to(hidden.dtype)

    @torch.no_grad()
    def generate_text_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[List[str], Optional[Tuple]]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        do_sample = float(temperature) > 0.0
        generate_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "return_dict_in_generate": True,
            "output_scores": False,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p
        outputs = self.model.generate(**generate_kwargs)
        sequences = outputs.sequences
        generations: List[str] = []
        for idx, length in enumerate(prompt_lengths):
            length = int(length)
            generated_ids = sequences[idx, length:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            generations.append(text)
        return generations, outputs.past_key_values

    @torch.no_grad()
    def prompt_last_hidden_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        past_key_values: Optional[Tuple] = None,
    ) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)

        prompt_lengths = attention_mask.sum(dim=1).to(torch.long).clamp_min(1)
        model_attention_mask = attention_mask
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                model_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        final_layer_hidden = outputs.hidden_states[-1]
        batch_indices = torch.arange(input_ids.shape[0], device=final_layer_hidden.device)
        token_indices = (prompt_lengths - 1).to(final_layer_hidden.device)
        return final_layer_hidden[batch_indices, token_indices, :].detach().clone()

    def tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(self.device)

    def _transformer_layers(self):
        base = getattr(self.model, "model", None)
        layers = getattr(base, "layers", None)
        if layers is None:
            layers = getattr(self.model, "layers", None)
        if layers is None:
            raise ValueError("Could not locate transformer layers for layer-wise state injection.")
        return layers

    def _register_layer_injection_hooks(
        self,
        layer_vectors: Optional[Dict[int, torch.Tensor]],
        alpha: float,
    ) -> List:
        if not layer_vectors or alpha == 0.0:
            return []

        layers = self._transformer_layers()
        hooks = []

        def make_hook(vector: torch.Tensor):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                delta = vector.to(device=hidden.device, dtype=hidden.dtype).view(1, 1, -1)
                updated_hidden = hidden + alpha * delta
                if isinstance(output, tuple):
                    return (updated_hidden,) + output[1:]
                return updated_hidden

            return hook

        for layer_idx in sorted(layer_vectors):
            if layer_idx < 0 or layer_idx >= len(layers):
                raise ValueError(f"Layer {layer_idx} is out of range for model with {len(layers)} layers.")
            hooks.append(layers[layer_idx].register_forward_hook(make_hook(layer_vectors[layer_idx])))
        return hooks

    def sequence_logprob_batch(
        self,
        prompts: List[str],
        continuations: List[str],
        *,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(prompts) != len(continuations):
            raise ValueError("prompts and continuations must have the same length")
        if not prompts:
            raise ValueError("sequence_logprob_batch requires at least one prompt")

        full_texts = [prompt + continuation for prompt, continuation in zip(prompts, continuations)]
        prompt_lengths = [
            int(
                self.tokenizer(
                    prompt,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"].shape[-1]
            )
            for prompt in prompts
        ]
        encoded = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        labels = input_ids.clone()

        for row_idx, prompt_len in enumerate(prompt_lengths):
            non_pad_len = int(attention_mask[row_idx].sum().item())
            pad_len = input_ids.shape[-1] - non_pad_len
            response_start = min(input_ids.shape[-1], pad_len + prompt_len)
            labels[row_idx, :response_start] = -100
            labels[row_idx, attention_mask[row_idx] == 0] = -100

        model_attention_mask = attention_mask
        cache_position = None
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            cache_position = torch.arange(
                past_len,
                past_len + input_ids.shape[-1],
                dtype=torch.long,
                device=self.device,
            )
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                model_attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=False,
            return_dict=True,
        )
        shifted_logits = outputs.logits[:, :-1, :]
        shifted_labels = labels[:, 1:].clone()
        loss_mask = shifted_labels != -100
        shifted_labels[shifted_labels == -100] = 0
        token_logps = torch.gather(
            shifted_logits.log_softmax(dim=-1),
            dim=2,
            index=shifted_labels.unsqueeze(2),
        ).squeeze(2)
        lengths = loss_mask.sum(dim=-1).clamp_min(1)
        logps = (token_logps * loss_mask).sum(dim=-1)
        return logps, lengths

    def generate_latent_batch_trainable(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        latent_layer_injection_vectors: Optional[Dict[int, torch.Tensor]] = None,
        latent_injection_alpha: float = 0.0,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        context = nullcontext() if _past_requires_grad(past_key_values) else torch.no_grad()
        with context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :].detach()

        for _ in range(latent_steps):
            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            hooks = self._register_layer_injection_hooks(
                latent_layer_injection_vectors,
                latent_injection_alpha,
            )
            try:
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                for hook in hooks:
                    hook.remove()
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

        return past

    @torch.no_grad()
    def generate_latent_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        return_trace: bool = False,
        latent_injection_vector: Optional[torch.Tensor] = None,
        latent_layer_injection_vectors: Optional[Dict[int, torch.Tensor]] = None,
        latent_injection_alpha: float = 0.0,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values

        e_t = outputs.hidden_states[0][:, -1, :]          # [B, D]
        last_hidden = outputs.hidden_states[-1][:, -1, :] # [B, D]
        h_t = last_hidden.detach().clone()

        e_t_plus_1 = None
        latent_vecs_all: List[torch.Tensor] = []
        latent_vecs_all.append(e_t.detach().clone())
        latent_input_vectors: List[torch.Tensor] = []
        latent_output_hidden_states: List[torch.Tensor] = []
        latent_output_layer_hidden_states: List[torch.Tensor] = []

        for step in range(latent_steps):
            if latent_injection_vector is not None and latent_injection_alpha != 0.0:
                last_hidden = last_hidden + latent_injection_alpha * latent_injection_vector.to(
                    device=last_hidden.device,
                    dtype=last_hidden.dtype,
                )

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)

            latent_vecs_all.append(latent_vec.detach().clone())
            if return_trace:
                latent_input_vectors.append(latent_vec.detach().clone())

            if step == 0:
                e_t_plus_1 = latent_vec.detach().clone()
            
            latent_embed = latent_vec.unsqueeze(1)

            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=self.device,
            )
            hooks = self._register_layer_injection_hooks(
                latent_layer_injection_vectors,
                latent_injection_alpha,
            )
            try:
                outputs = self.model(
                    inputs_embeds=latent_embed,
                    attention_mask=latent_mask,
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            finally:
                for hook in hooks:
                    hook.remove()
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]
            if return_trace:
                latent_output_hidden_states.append(last_hidden.detach().clone())
                layer_hidden = torch.stack(
                    [hidden_state[:, -1, :] for hidden_state in outputs.hidden_states[1:]],
                    dim=1,
                )
                latent_output_layer_hidden_states.append(layer_hidden.detach().clone())

        if not return_trace:
            return past

        batch_size, hidden_dim = h_t.shape
        empty_steps = h_t.new_empty((batch_size, 0, hidden_dim))
        trace = {
            "prompt_last_hidden": h_t,
            "latent_input_vectors": torch.stack(latent_input_vectors, dim=1) if latent_input_vectors else empty_steps,
            "latent_output_hidden_states": torch.stack(latent_output_hidden_states, dim=1) if latent_output_hidden_states else empty_steps,
            "latent_output_layer_hidden_states": torch.stack(latent_output_layer_hidden_states, dim=1) if latent_output_layer_hidden_states else h_t.new_empty((batch_size, 0, 0, hidden_dim)),
        }
        return past, trace
    
    @torch.no_grad()
    def generate_latent_batch_hidden_state(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
    ) -> Tuple:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be 2D with shape [batch, seq_len]")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.HF_device)
        else:
            attention_mask = attention_mask.to(self.HF_device)
        if past_key_values is not None:
            past_len = _past_length(past_key_values)
            if past_len > 0:
                past_mask = torch.ones(
                    (attention_mask.shape[0], past_len),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([past_mask, attention_mask], dim=-1)
        outputs = self.HF_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        last_hidden = outputs.hidden_states[-1][:, -1, :]
        
        curr_output_embedding = [] 
        curr_output_embedding.append(outputs.hidden_states[0])  # input embedding
        
        
        for _ in range(latent_steps):

            source_model = self.HF_model if hasattr(self, "HF_model") else self.model
            latent_vec = self._apply_latent_realignment(last_hidden, source_model)
            latent_embed = latent_vec.unsqueeze(1)
            past_len = _past_length(past)
            latent_mask = torch.ones(
                (latent_embed.shape[0], past_len + 1),
                dtype=torch.long,
                device=latent_embed.device,
            )
            outputs = self.HF_model(
                inputs_embeds=latent_embed,
                attention_mask=latent_mask,
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            curr_output_embedding.append(latent_embed.detach())

        return past, torch.cat(curr_output_embedding, dim=1) # Output input embeddings

