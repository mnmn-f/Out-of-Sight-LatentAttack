import argparse
import json
import os
from typing import Dict, List, Tuple

from attacks import attack_metadata


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for pred in preds if pred.get("correct", False))
    return (correct / total if total else 0.0), correct


def partial_jsonl_path(output_path: str) -> str:
    return output_path + ".jsonl"


def append_partial_result(output_path: str, result: Dict) -> None:
    partial_path = partial_jsonl_path(output_path)
    os.makedirs(os.path.dirname(os.path.abspath(partial_path)), exist_ok=True)
    with open(partial_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def load_partial_results(output_path: str) -> List[Dict]:
    partial_path = partial_jsonl_path(output_path)
    if not os.path.exists(partial_path):
        return []

    preds: List[Dict] = []
    with open(partial_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                preds.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return preds


def write_final_results(output_path: str, summary: Dict, args: argparse.Namespace, preds: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "args": vars(args), "preds": preds}, f, ensure_ascii=False, indent=2)


def build_summary(args: argparse.Namespace, method, preds: List[Dict], total_time: float) -> Dict:
    acc, correct = evaluate(preds)
    total = max(1, args.max_samples)
    return {
        "method": "latent_mas",
        "model": args.model_name,
        "split": args.split,
        "seed": args.seed,
        "start_index": args.start_index,
        "max_samples": args.max_samples,
        "accuracy": acc,
        "correct": correct,
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": round(total_time / total, 4),
        "attack": attack_metadata(args),
        "trace_output_dir": getattr(method, "trace_root", None),
        "state_injection": {
            "enabled": args.state_injection,
            "vector_path": args.state_injection_vector_path,
            "role": args.state_injection_role,
            "layers": args.state_injection_layers,
            "alpha": args.state_injection_alpha,
        },
        "kv_injection": {
            "enabled": args.kv_injection,
            "vector_path": args.kv_injection_vector_path,
            "role": args.kv_injection_role,
            "edge": args.kv_injection_edge,
            "mode": args.kv_injection_mode,
            "layers": args.kv_injection_layers,
            "position": args.kv_injection_position,
            "alpha_k": args.kv_injection_alpha_k,
            "alpha_v": args.kv_injection_alpha_v,
        },
    }
