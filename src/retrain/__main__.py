"""Local equivalent of the scheduled retrain workflow.

Examples::

    python -m src.retrain --all
    python -m src.retrain --all --max-epochs 8 --release-dir release-assets --pull-release --publish
    python -m src.retrain --datasets weather,fx --max-epochs 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.retrain.artifacts import publish_release, pull_release
from src.retrain.discover import DatasetSpec, discover, load_dataset_file
from src.retrain.run import RunOptions, plan_document, run_many
from src.utils.paths import resolve_repo_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.retrain",
        description=(
            "Fetch each configured dataset, retrain, and promote the checkpoint "
            "only when it beats the current champion and the baselines on the "
            "latest held-out window."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Retrain every enabled config under --config-dir.",
    )
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Train this config file. Repeat to train several. Paths may be outside config/.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset ids or config paths inside --config-dir.",
    )
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--champions-dir", default="champions")
    parser.add_argument("--forecasts-dir", default="forecasts")
    parser.add_argument(
        "--release-dir",
        default=None,
        help="Import champion-*.pt from this directory before training and export assets after.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Cap training.epochs. The config value is used when it is smaller.",
    )
    parser.add_argument(
        "--dataset-timeout",
        type=float,
        default=720,
        help="Seconds allowed for each dataset when running child processes.",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Promotion metric (mae, rmse, or mse). Overrides retrain.metric.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the datasets and fetch commands as JSON and do not train.",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Train in this process instead of a child process per dataset.",
    )
    parser.add_argument(
        "--pull-release",
        action="store_true",
        help="Download the champions GitHub Release into --release-dir before training.",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Upload --release-dir to the champions GitHub Release after training.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.config is None:
        args.config = []
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be at least 1.")
    if args.dataset_timeout <= 0:
        parser.error("--dataset-timeout must be positive.")
    if not args.all and not args.config and not args.datasets:
        parser.error("Pass --all, --config, or --datasets.")
    if (args.pull_release or args.publish) and not args.release_dir:
        parser.error("--pull-release and --publish require --release-dir.")

    release_dir = resolve_repo_path(args.release_dir) if args.release_dir else None
    if args.pull_release and release_dir is not None and not args.dry_run:
        try:
            pull_release(release_dir)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1

    specs, errors = _select(args, parser)
    for error in errors:
        print(error, file=sys.stderr)
    if args.dry_run:
        print(json.dumps(plan_document(specs, errors), indent=2))
        return 1 if errors else 0

    options = RunOptions(
        champions_dir=resolve_repo_path(args.champions_dir),
        forecasts_dir=resolve_repo_path(args.forecasts_dir),
        release_dir=release_dir,
        max_epochs=args.max_epochs,
        metric=args.metric,
        dataset_timeout=args.dataset_timeout,
        dry_run=False,
        in_process=args.in_process,
    )
    results = run_many(specs, options)
    for result in results:
        print(
            f"{result.dataset_id}: decision={result.decision} promoted={result.promoted}"
        )
        if result.error:
            print(result.error, file=sys.stderr)
    failed = bool(errors) or any(result.decision == "error" for result in results)
    if args.publish and release_dir is not None:
        try:
            publish_release(release_dir)
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 1
    return 1 if failed else 0


def _select(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> tuple[list[DatasetSpec], list[str]]:
    config_dir = resolve_repo_path(args.config_dir)
    discovered, errors = discover(config_dir)
    by_id = {spec.id: spec for spec in discovered}
    by_path = {spec.path.resolve(): spec for spec in discovered}
    selected: list[DatasetSpec] = []
    seen: set[str] = set()

    def add(spec: DatasetSpec) -> None:
        if spec.id in seen:
            return
        seen.add(spec.id)
        selected.append(spec)

    if args.all or args.datasets:
        chosen = discovered
        if args.datasets:
            chosen = []
            for token in args.datasets.split(","):
                token = token.strip()
                if not token:
                    continue
                if token in by_id:
                    chosen.append(by_id[token])
                    continue
                path = _token_path(token, config_dir)
                match = by_path.get(path)
                if match is None:
                    errors.append(f"Unknown dataset or config: {token}")
                else:
                    chosen.append(match)
        for spec in chosen:
            add(spec)
    for raw in args.config:
        path = resolve_repo_path(raw)
        match = by_path.get(path.resolve())
        if match is not None:
            add(match)
            continue
        loaded, error = load_dataset_file(path)
        if error:
            errors.append(error)
        elif loaded is None:
            errors.append(f"{path} is not an enabled dataset config.")
        else:
            add(loaded)
    if not selected and not errors:
        parser.error("No datasets selected.")
    return selected, errors


def _token_path(token: str, config_dir: Path) -> Path:
    candidate = Path(token)
    if candidate.is_absolute():
        return candidate.resolve()
    direct = (config_dir / candidate).resolve()
    if direct.is_file():
        return direct
    return resolve_repo_path(token).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
