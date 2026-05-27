import argparse
import copy
import json
from typing import Dict, List, Optional, Set


PAIR_FILTERS = ["all", "clean_correct_attack_wrong", "clean_correct", "attack_wrong", "clean_wrong_attack_correct"]
OUTPUT_SOURCES = ["clean_raw", "attacked_raw", "gold_box", "clean_prediction_box", "attacked_prediction_box"]


def load_result_payload(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _boxed_answer(answer: str) -> str:
    answer = (answer or "").strip()
    return f"\\boxed{{{answer.upper()}}}"


def _prediction_box(pred: Dict) -> str:
    return _boxed_answer(str(pred.get("prediction", "") or pred.get("gold", "")))


def _raw_or_box(pred: Dict) -> str:
    raw = (pred.get("raw_prediction", "") or "").strip()
    return raw if raw else _prediction_box(pred)


def output_from_source(source: str, clean_pred: Dict, attacked_pred: Dict) -> str:
    if source == "clean_raw":
        return _raw_or_box(clean_pred)
    if source == "attacked_raw":
        return _raw_or_box(attacked_pred)
    if source == "gold_box":
        return _boxed_answer(str(clean_pred.get("gold", "") or attacked_pred.get("gold", "")))
    if source == "clean_prediction_box":
        return _prediction_box(clean_pred)
    if source == "attacked_prediction_box":
        return _prediction_box(attacked_pred)
    raise ValueError(f"Unsupported output source: {source}")


def pair_matches_filter(pair_filter: str, clean_ok: bool, attacked_ok: bool) -> bool:
    if pair_filter == "clean_correct_attack_wrong":
        return clean_ok and not attacked_ok
    if pair_filter == "clean_correct":
        return clean_ok
    if pair_filter == "attack_wrong":
        return not attacked_ok
    if pair_filter == "clean_wrong_attack_correct":
        return (not clean_ok) and attacked_ok
    if pair_filter == "all":
        return True
    raise ValueError(f"Unsupported pair_filter: {pair_filter}")


def load_matched_predictions(clean_results_json: str, attacked_results_json: str):
    clean_payload = load_result_payload(clean_results_json)
    attacked_payload = load_result_payload(attacked_results_json)
    clean_preds = clean_payload.get("preds", [])
    attacked_preds = attacked_payload.get("preds", [])
    n = min(len(clean_preds), len(attacked_preds))
    if n == 0:
        raise ValueError("No predictions found in the provided result JSON files.")

    start_index = int(clean_payload.get("summary", {}).get("start_index", 0))
    attacked_start_index = int(attacked_payload.get("summary", {}).get("start_index", start_index))
    if attacked_start_index != start_index:
        raise ValueError(
            f"Result start_index mismatch: clean={start_index}, attacked={attacked_start_index}. "
            "Use matched result files."
        )
    return clean_preds, attacked_preds, start_index, n


def load_result_filter(
    clean_results_json: str,
    attacked_results_json: str,
    pair_filter: str,
) -> Optional[Set[int]]:
    if pair_filter == "all":
        return None
    if not clean_results_json or not attacked_results_json:
        raise ValueError("--pair_filter requires both --clean_results_json and --attacked_results_json unless pair_filter=all")

    clean_preds, attacked_preds, start_index, n = load_matched_predictions(clean_results_json, attacked_results_json)
    allowed: Set[int] = set()
    for idx in range(n):
        clean_ok = bool(clean_preds[idx].get("correct", False))
        attacked_ok = bool(attacked_preds[idx].get("correct", False))
        if pair_matches_filter(pair_filter, clean_ok, attacked_ok):
            allowed.add(start_index + idx)

    if not allowed:
        raise ValueError(f"pair_filter={pair_filter} matched zero samples.")
    return allowed


def load_preference_pairs(
    *,
    clean_results_json: str,
    attacked_results_json: str,
    pair_filter: str,
    chosen_source: str,
    rejected_source: str,
    max_pairs: int,
) -> List[Dict]:
    clean_preds, attacked_preds, start_index, n = load_matched_predictions(clean_results_json, attacked_results_json)
    pairs: List[Dict] = []
    for idx in range(n):
        clean_pred = clean_preds[idx]
        attacked_pred = attacked_preds[idx]
        clean_ok = bool(clean_pred.get("correct", False))
        attacked_ok = bool(attacked_pred.get("correct", False))
        if not pair_matches_filter(pair_filter, clean_ok, attacked_ok):
            continue

        chosen = output_from_source(chosen_source, clean_pred, attacked_pred)
        rejected = output_from_source(rejected_source, clean_pred, attacked_pred)
        if not chosen.strip() or not rejected.strip():
            continue
        pairs.append(
            {
                "sample_idx": start_index + idx,
                "question": attacked_pred.get("question", clean_pred.get("question", "")),
                "gold": attacked_pred.get("gold", clean_pred.get("gold", "")),
                "solution": attacked_pred.get("solution", clean_pred.get("solution", "")),
                "chosen": chosen,
                "rejected": rejected,
                "clean_correct": clean_ok,
                "attacked_correct": attacked_ok,
            }
        )

    if max_pairs > 0:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError(f"pair_filter={pair_filter} produced zero usable preference pairs.")
    return pairs


def namespace_from_results(results_json: str, cli_args: argparse.Namespace) -> argparse.Namespace:
    payload = load_result_payload(results_json)
    values = copy.deepcopy(payload.get("args", {}))
    values.update(
        {
            "method": "latent_mas",
            "model_name": cli_args.model_name or values.get("model_name", ""),
            "device": cli_args.device,
            "use_vllm": False,
            "enable_prefix_caching": False,
            "use_second_HF_model": False,
            "generate_bs": cli_args.batch_size,
            "latent_steps": cli_args.latent_steps if cli_args.latent_steps is not None else values.get("latent_steps", 5),
            "max_new_tokens": values.get("max_new_tokens", 1024),
            "temperature": values.get("temperature", 0.0),
            "top_p": values.get("top_p", 1.0),
            "state_injection": False,
            "kv_injection": False,
            "trace_export": False,
            "trace_save_hidden": False,
            "trace_save_kv": False,
        }
    )
    if not values["model_name"]:
        raise ValueError("--model_name is required when it cannot be inferred from --context_results_json.")
    return argparse.Namespace(**values)
