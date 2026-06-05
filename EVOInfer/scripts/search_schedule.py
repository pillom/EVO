"""Command-line entry point for EVO offline cache-schedule search.

The script loads the paper-default search settings from YAML, lets explicit
command-line options override them, and forwards the final arguments to
``EVOInfer.search``. The search itself optimizes a fixed-budget subset of the
block–timestep lattice using rollout success rate as the fitness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "search_config" / "default_search.yaml"


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load the YAML defaults for offline schedule search."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Search config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Search config must be a mapping: {config_path}")
    return payload


def _has_option(args: Iterable[str], option: str) -> bool:
    """Return whether ``option`` is already present in raw CLI tokens."""
    prefix = option + "="
    return any(item == option or item.startswith(prefix) for item in args)


def _append_default(args: List[str], option: str, value: Any) -> None:
    """Append one YAML default unless the user provided that option."""
    if value is None or _has_option(args, option):
        return
    args.extend([option, str(value)])


def _config_defaults(config: Dict[str, Any], args: List[str]) -> List[str]:
    """Merge YAML defaults with explicit EVO search arguments."""
    search = config.get("search", {}) or {}
    initialization = config.get("initialization", {}) or {}
    outputs = config.get("outputs", {}) or {}

    final_args = list(args)
    task = _get_option_value(final_args, "--task")

    if not _has_option(final_args, "--output_dir"):
        search_root = outputs.get("search_root")
        if search_root and task:
            final_args.extend(["--output_dir", str(Path(search_root) / task)])

    mappings = {
        "--k_pairs": search.get("k_pairs"),
        "--population_size": search.get("population_size"),
        "--elite_size": search.get("elite_size"),
        "--mutation_rate": search.get("mutation_rate"),
        "--crossover_rate": search.get("crossover_rate"),
        "--tournament_size": search.get("tournament_size"),
        "--max_generations": search.get("max_generations"),
        "--num_blocks": search.get("num_blocks"),
        "--num_steps": search.get("num_steps"),
        "--init_strategy": initialization.get("strategy"),
        "--init_random_fraction": initialization.get("random_fraction"),
        "--init_structure_min_pairs_per_block": initialization.get("min_pairs_per_block"),
        "--init_importance_gamma": initialization.get("importance_gamma"),
        "--init_importance_epsilon": initialization.get("importance_epsilon"),
    }
    for option, value in mappings.items():
        _append_default(final_args, option, value)

    return final_args


def _get_option_value(args: List[str], option: str) -> Optional[str]:
    """Return the value associated with a command-line option."""
    prefix = option + "="
    for index, item in enumerate(args):
        if item.startswith(prefix):
            return item.split("=", 1)[1]
        if item == option and index + 1 < len(args):
            return args[index + 1]
    return None


def main(argv: Optional[List[str]] = None) -> None:
    """Load defaults and launch rollout-driven EVO schedule search."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--help" in argv or "-h" in argv:
        print(
            "Usage: python -m EVOInfer.scripts.search_schedule [--config CONFIG] "
            "[EVO search options]\n\n"
            "Loads paper-default search settings from a YAML config and forwards "
            "all remaining options to EVOInfer.search.evo_search.\n\n"
            "Default config: EVOInfer/search_config/default_search.yaml\n\n"
            "Common options: --task, --checkpoint, --output_dir, --device, "
            "--k_pairs, --population_size, --max_generations, "
            "--init_strategy, --init_pair_importance_path"
        )
        return

    parser = argparse.ArgumentParser(
        add_help=False,
        description=(
            "Load EVO paper-default search settings from YAML and forward the "
            "resulting arguments to the rollout-driven evolutionary schedule search."
        ),
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=(
            "YAML file containing offline schedule-search defaults such as "
            "population size, refresh budget, and redundancy-aware "
            "initialization settings."
        ),
    )
    parsed, remaining = parser.parse_known_args(argv)

    config = _load_config(parsed.config)
    forwarded_args = _config_defaults(config, remaining)
    from EVOInfer.search.evo_search import main as evo_search_main

    evo_search_main.main(args=forwarded_args, standalone_mode=True)


if __name__ == "__main__":
    main()
