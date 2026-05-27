import argparse
import os

from .pairs import PAIR_FILTERS, load_result_filter
from .trace_store import collect_trace_pairs, edge_label_for_role, parse_roles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract clean-vs-attacked steering vectors from LatentMAS trace exports.")
    parser.add_argument("--clean_trace_dir", required=True, help="Trace directory from the clean run.")
    parser.add_argument("--attacked_trace_dir", required=True, help="Trace directory from the attacked run.")
    parser.add_argument("--roles", default="planner", help="Comma-separated roles to extract, or 'all'.")
    parser.add_argument("--vector_type", choices=["hidden", "kv", "both"], default="hidden")
    parser.add_argument(
        "--hidden_field",
        choices=["prompt_last_hidden", "latent_input_vectors", "latent_output_hidden_states", "latent_output_layer_hidden_states"],
        default="latent_output_hidden_states",
    )
    parser.add_argument("--hidden_layer", default="last", help="For layer hidden states, select all, last, or a layer index.")
    parser.add_argument("--step_pooling", choices=["first_k", "last_k", "mean_all", "step"], default="first_k")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--direction_method", choices=["mean", "diffmean", "pca"], default="mean")
    parser.add_argument("--edge_map", choices=["sequential", "none"], default="sequential")
    parser.add_argument("--kv_position", default="all")
    parser.add_argument("--clean_results_json", default="", help="Clean run JSON used to filter trace pairs.")
    parser.add_argument("--attacked_results_json", default="", help="Attacked run JSON used to filter trace pairs.")
    parser.add_argument("--pair_filter", choices=PAIR_FILTERS, default="all")
    parser.add_argument("--output_path", required=True, help="Where to save the extracted vector payload (.pt).")
    return parser


def build_vector_payload(args: argparse.Namespace) -> dict:
    from .directions import extract_hidden_vectors, mean_kv_diff

    roles = parse_roles(args.roles)
    allowed_sample_indices = load_result_filter(args.clean_results_json, args.attacked_results_json, args.pair_filter)
    trace_pairs = collect_trace_pairs(args.clean_trace_dir, args.attacked_trace_dir, roles, allowed_sample_indices)
    if not trace_pairs:
        raise SystemExit("No matched clean/attacked trace pairs found.")

    output = {
        "meta": {
            "clean_trace_dir": os.path.abspath(args.clean_trace_dir),
            "attacked_trace_dir": os.path.abspath(args.attacked_trace_dir),
            "roles": roles or "all",
            "vector_type": args.vector_type,
            "hidden_field": args.hidden_field,
            "hidden_layer": args.hidden_layer,
            "step_pooling": args.step_pooling,
            "k": args.k,
            "direction_method": args.direction_method,
            "edge_map": args.edge_map,
            "kv_position": args.kv_position,
            "pair_filter": args.pair_filter,
            "clean_results_json": os.path.abspath(args.clean_results_json) if args.clean_results_json else "",
            "attacked_results_json": os.path.abspath(args.attacked_results_json) if args.attacked_results_json else "",
            "filtered_sample_indices": sorted(allowed_sample_indices) if allowed_sample_indices is not None else None,
        },
        "roles": {},
    }

    for role, role_pairs in trace_pairs.items():
        role_payload = {"num_pairs": len(role_pairs)}
        edge_label = edge_label_for_role(role, args.edge_map)
        if edge_label:
            role_payload["edge"] = edge_label
        if args.vector_type in ("hidden", "both"):
            role_payload["hidden"] = extract_hidden_vectors(
                role_pairs,
                hidden_field=args.hidden_field,
                step_pooling=args.step_pooling,
                k=args.k,
                hidden_layer=args.hidden_layer,
                direction_method=args.direction_method,
            )
        if args.vector_type in ("kv", "both"):
            role_payload["kv"] = mean_kv_diff(
                role_pairs,
                kv_position=args.kv_position,
                direction_method=args.direction_method,
            )
        output["roles"][role] = role_payload

    return output


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    import torch

    output = build_vector_payload(args)
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(output, args.output_path)

    print(f"Saved vector payload to {args.output_path}")
    for role, role_payload in output["roles"].items():
        edge = role_payload.get("edge")
        edge_part = f" edge={edge}" if edge else ""
        print(f"- role={role}{edge_part} pairs={role_payload['num_pairs']}")


if __name__ == "__main__":
    main()
