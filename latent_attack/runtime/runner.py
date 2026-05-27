import time
from typing import Dict, List, Tuple

from tqdm import tqdm

from ..tasks import iter_dataset
from .results import (
    append_partial_result,
    build_summary,
    load_partial_results,
    partial_jsonl_path,
    write_final_results,
)


def print_result(problem_idx: int, result: Dict) -> None:
    print(f"\n==================== Problem #{problem_idx} ====================")
    print("Question:")
    print(result.get("question", "").strip())
    for agent in result.get("agents", []):
        name = agent.get("name", "Agent")
        role = agent.get("role", "")
        print(f"----- Agent: {name} ({role}) -----")
        print("[To Tokenize]")
        print(agent.get("input", "").rstrip())
        if agent.get("latent_steps") is not None:
            print("[Latent Steps]")
            print(agent.get("latent_steps"))
        print("[Output]")
        print(agent.get("output", "").rstrip())
        print("----------------------------------------------")
    print(f"Result: Pred={result.get('prediction')} | Gold={result.get('gold')} | OK={result.get('correct')}")


def process_batch(
    method,
    batch: List[Dict],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds

    current_batch = batch[:remaining]
    sample_offset = args.start_index + processed
    if args.use_vllm:
        results = method.run_batch_vllm(current_batch, sample_offset=sample_offset)
    else:
        results = method.run_batch(current_batch, sample_offset=sample_offset)
    results = results[:remaining]

    for offset, result in enumerate(results):
        preds.append(result)
        if args.output_path:
            append_partial_result(args.output_path, result)
        print_result(processed + offset + 1, result)

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def build_method(args):
    from methods.latent_mas import LatentMASMethod
    from models import ModelWrapper
    from utils import auto_device

    device = auto_device(args.device)
    model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)
    return LatentMASMethod(
        model,
        latent_steps=args.latent_steps,
        judger_max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        generate_bs=args.generate_bs,
        args=args,
    )


def run_experiment(args) -> Dict:
    from utils import set_seed

    set_seed(args.seed)
    dataset = iter_dataset(args)
    if args.max_samples == -1:
        dataset = list(dataset)
        args.max_samples = max(0, len(dataset) - args.start_index)

    method = build_method(args)
    preds: List[Dict] = []
    processed = 0
    if args.resume_partial and args.output_path:
        preds = load_partial_results(args.output_path)[: args.max_samples]
        processed = len(preds)
        if processed:
            print(f"[resume_partial] loaded {processed} rows from {partial_jsonl_path(args.output_path)}")

    start_time = time.time()
    batch: List[Dict] = []
    progress = tqdm(total=args.max_samples, initial=processed)
    for dataset_idx, item in enumerate(dataset):
        if dataset_idx < args.start_index + processed:
            continue
        if processed >= args.max_samples:
            break
        batch.append(item)
        if len(batch) == args.generate_bs or processed + len(batch) == args.max_samples:
            processed, preds = process_batch(method, batch, processed, preds, progress, args.max_samples, args)
            batch = []

    if batch and processed < args.max_samples:
        processed, preds = process_batch(method, batch, processed, preds, progress, args.max_samples, args)
    progress.close()

    summary = build_summary(args, method, preds, time.time() - start_time)
    if args.output_path:
        write_final_results(args.output_path, summary, args, preds)
    return summary
