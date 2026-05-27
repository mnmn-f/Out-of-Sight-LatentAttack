import json

from .config import parse_args, validate_args
from .runtime.runner import run_experiment


def main(argv=None) -> None:
    args = parse_args(argv)
    args.method = "latent_mas"
    validate_args(args)
    summary = run_experiment(args)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
