from typing import Iterable


TASK_LOADERS = {
    "gsm8k": "load_gsm8k",
    "aime2024": "load_aime2024",
    "aime2025": "load_aime2025",
    "gpqa": "load_gpqa_diamond",
    "arc_easy": "load_arc_easy",
    "arc_challenge": "load_arc_challenge",
    "openbookqa": "load_openbookqa",
    "mbppplus": "load_mbppplus",
    "humanevalplus": "load_humanevalplus",
    "medqa": "load_medqa",
}


def iter_dataset(args) -> Iterable[dict]:
    if args.task not in TASK_LOADERS:
        raise ValueError(f"Unsupported task: {args.task}")

    from .loaders import (
        load_aime2024,
        load_aime2025,
        load_arc_challenge,
        load_arc_easy,
        load_gpqa_diamond,
        load_gsm8k,
        load_humanevalplus,
        load_mbppplus,
        load_medqa,
        load_openbookqa,
    )

    loaders = {
        "load_gsm8k": lambda split: load_gsm8k(split=split),
        "load_aime2024": lambda split: load_aime2024(split="train"),
        "load_aime2025": lambda split: load_aime2025(split="train"),
        "load_gpqa_diamond": lambda split: load_gpqa_diamond(split="test"),
        "load_arc_easy": lambda split: load_arc_easy(split="test"),
        "load_arc_challenge": lambda split: load_arc_challenge(split="test"),
        "load_openbookqa": lambda split: load_openbookqa(split="test"),
        "load_mbppplus": lambda split: load_mbppplus(split="test"),
        "load_humanevalplus": lambda split: load_humanevalplus(split="test"),
        "load_medqa": lambda split: load_medqa(split="test"),
    }
    return loaders[TASK_LOADERS[args.task]](args.split)
