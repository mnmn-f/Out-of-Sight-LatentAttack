import argparse

from attacks import ATTACK_ROLES, ATTACK_SURFACES

from .tasks import TASK_LOADERS


MAINLINE_ATTACK_TYPES = ["none", "mi"]


def add_core_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--task", choices=sorted(TASK_LOADERS), default="gsm8k")
    parser.add_argument("--prompt", choices=["sequential", "hierarchical"], default="sequential")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device2", default="cuda:1")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--latent_only", action="store_true")
    parser.add_argument("--sequential_info_only", action="store_true")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--resume_partial", action="store_true")


def add_attack_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--attack_surface", choices=ATTACK_SURFACES, default="none")
    parser.add_argument("--attack_type", choices=MAINLINE_ATTACK_TYPES, default="none")
    parser.add_argument("--attack_target_role", choices=ATTACK_ROLES, default="none")
    parser.add_argument("--attack_wrong_answer_strategy", default="random")
    parser.add_argument("--mi_roles", default="planner")
    parser.add_argument("--mi_reference_answer", default="0")


def add_trace_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trace_export", action="store_true")
    parser.add_argument("--trace_output_dir", default="")
    parser.add_argument("--trace_roles", default="planner,critic,refiner")
    parser.add_argument("--trace_save_hidden", action="store_true")
    parser.add_argument("--trace_save_kv", action="store_true")
    parser.add_argument("--trace_kv_mode", choices=["latent_only", "delta", "full"], default="latent_only")


def add_injection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state_injection", action="store_true")
    parser.add_argument("--state_injection_vector_path", default="")
    parser.add_argument("--state_injection_role", default="planner")
    parser.add_argument("--state_injection_layers", default="last")
    parser.add_argument("--state_injection_alpha", type=float, default=0.0)
    parser.add_argument("--kv_injection", action="store_true")
    parser.add_argument("--kv_injection_vector_path", default="")
    parser.add_argument("--kv_injection_role", default="planner")
    parser.add_argument("--kv_injection_edge", default="")
    parser.add_argument("--kv_injection_mode", choices=["k_only", "v_only", "kv_both"], default="kv_both")
    parser.add_argument("--kv_injection_layers", default="all")
    parser.add_argument("--kv_injection_position", default="all")
    parser.add_argument("--kv_injection_alpha_k", type=float, default=1.0)
    parser.add_argument("--kv_injection_alpha_v", type=float, default=1.0)


def add_backend_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument("--enable_prefix_caching", action="store_true")
    parser.add_argument("--use_second_HF_model", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LatentMAS attack pipeline.")
    add_core_args(parser)
    add_attack_args(parser)
    add_trace_args(parser)
    add_injection_args(parser)
    add_backend_args(parser)
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.start_index < 0:
        raise ValueError("--start_index must be non-negative")
    if args.attack_type == "mi" and args.attack_surface != "role_prompt":
        raise NotImplementedError("mi supports role-prompt attacker-node runs only.")
    if args.trace_export and not args.trace_save_hidden and not args.trace_save_kv:
        args.trace_save_hidden = True
    if args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True
