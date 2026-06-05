"""Rollout-driven evolutionary search for EVO cache schedules.

EVO formulates cache scheduling as a fixed-budget subset selection problem over
the block–timestep lattice. Candidate schedules are optimized with evolutionary
operators, evaluated through closed-loop rollout performance, and finally
exported as an offline schedule for residual-cache deployment.
"""

import os
import sys
import logging
import numpy as np
import torch
import json
import pickle
import random
import re
import shlex
import subprocess
import time
import multiprocessing as mp
import multiprocessing.pool as mp_pool
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Sequence
from collections import defaultdict
import copy
import dill
import hydra
from omegaconf import OmegaConf
import click
from tqdm import tqdm

# EVO cache wrapper and search utilities.
from EVOInfer.acceleration.evo_cache_wrapper import _EVOCacheWrapper
from EVOInfer.search.importance import load_activations
from EVOInfer.search.importance import (
    compute_cosine_global_pair_importance_matrix,
    compute_step_error_matrix,
    save_pair_importance_results,
)
from EVOInfer.search.activation_collection import (
    collect_block_activations_for_task,
)
from EVOInfer.search.structure_initialization import build_initial_population

# Make repository-local imports work when this file is executed as a module.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EVOInfer.utils.paths import get_checkpoint_path, get_all_available_tasks

# Logging.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


_WORKER_STATE = None


def _configure_reproducibility(
    seed: int,
    *,
    deterministic_torch: bool = True,
) -> Dict[str, Any]:
    """Seed Python, NumPy, and Torch for reproducible EVO search.

    Args:
        seed (int): Global seed used for random candidate generation and
            evaluation setup.
        deterministic_torch (bool): Whether to request deterministic Torch/CUDA
            behavior where available.

    Returns:
        Dict[str, Any]: Reproducibility metadata recorded in search outputs.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    deterministic_algorithms_enabled = None
    deterministic_error = None
    if deterministic_torch:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
            deterministic_algorithms_enabled = True
        except Exception as exc:  # pragma: no cover - defensive logging metadata
            deterministic_algorithms_enabled = False
            deterministic_error = str(exc)
    else:
        deterministic_algorithms_enabled = False

    cudnn_deterministic = None
    cudnn_benchmark = None
    if hasattr(torch.backends, "cudnn"):
        cudnn_deterministic = bool(deterministic_torch)
        cudnn_benchmark = False if deterministic_torch else bool(torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic = bool(deterministic_torch)
        torch.backends.cudnn.benchmark = False if deterministic_torch else torch.backends.cudnn.benchmark

    return {
        "seed": int(seed),
        "deterministic_torch": bool(deterministic_torch),
        "cuda_available": bool(cuda_available),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_deterministic_algorithms_enabled": deterministic_algorithms_enabled,
        "torch_deterministic_algorithms_error": deterministic_error,
        "cudnn_deterministic": cudnn_deterministic,
        "cudnn_benchmark": cudnn_benchmark,
    }


def _configure_torch_eval_seed(seed: int) -> None:
    """Reset Torch RNG for evaluation without touching Python's search RNG.

    Args:
        seed (int): Torch seed used immediately before rollout evaluation.

    Returns:
        None.
    """
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class _NoDaemonProcess(mp.Process):
    """Process subclass that permits nested multiprocessing in rollout workers."""

    @property
    def daemon(self):
        """Report non-daemon status for nested multiprocessing."""
        return False

    @daemon.setter
    def daemon(self, value):
        """Ignore daemon assignments required by multiprocessing internals."""
        pass


class _NoDaemonContext(type(mp.get_context("spawn"))):
    """Multiprocessing context backed by non-daemon worker processes."""

    Process = _NoDaemonProcess


class _NonDaemonPool(mp_pool.Pool):
    """Process pool that permits nested worker-side multiprocessing.

    Args:
        *args: Positional arguments forwarded to ``multiprocessing.Pool``.
        **kwargs: Keyword arguments forwarded to ``multiprocessing.Pool``.

    Returns:
        None.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a non-daemon process pool.

        Args:
            *args: Positional arguments forwarded to ``multiprocessing.Pool``.
            **kwargs: Keyword arguments forwarded to ``multiprocessing.Pool``.

        Returns:
            None.
        """
        kwargs["context"] = _NoDaemonContext()
        super().__init__(*args, **kwargs)


def _parse_device_list(device_spec: str) -> List[str]:
    """Parse a device spec such as ``cuda:0`` or ``cuda:0,cuda:1``.

    Args:
        device_spec (str): Comma-separated device specification.

    Returns:
        List[str]: Ordered device identifiers used by search workers.
    """
    devices = [item.strip() for item in str(device_spec).split(',') if item.strip()]
    if not devices:
        raise ValueError("At least one device must be provided.")
    return devices


def _load_policy_and_wrapper(checkpoint_path: str, device: str):
    """Load a pretrained policy and attach an EVO cache wrapper.

    Args:
        checkpoint_path (str): Path to a Diffusion Policy checkpoint.
        device (str): Torch device for the loaded policy.

    Returns:
        Tuple[object, _EVOCacheWrapper, object]: Loaded policy, cache wrapper,
        and checkpoint configuration.
    """
    logger.info("Loading checkpoint: %s on %s", checkpoint_path, device)

    payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if hasattr(cfg.training, 'use_ema') and cfg.training.use_ema:
        policy = workspace.ema_model

    policy.to(torch.device(device))
    policy.eval()

    cache_wrapper = _EVOCacheWrapper(policy)
    logger.info("Policy loaded on %s with %d cacheable blocks", device, cache_wrapper.get_num_blocks())
    return policy, cache_wrapper, cfg


def _load_pair_importance_payload(
    path: str,
) -> Tuple[Dict[Tuple[str, int], float], Dict[str, Sequence[str]]]:
    """Load pair-importance scores for redundancy-aware initialization.

    Args:
        path (str): JSON or pickle file containing ``pair_importance`` entries
            keyed by EVO block name and denoising-step key.

    Returns:
        Tuple[Dict[Tuple[str, int], float], Dict[str, Sequence[str]]]: Scores
        keyed by ``(block_name, step_idx)`` and metadata describing covered
        blocks and timesteps.
    """
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"pair importance file not found: {source_path}")

    if source_path.suffix == ".json":
        with open(source_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif source_path.suffix in {".pkl", ".pickle"}:
        with open(source_path, "rb") as f:
            payload = pickle.load(f)
    else:
        raise ValueError(f"unsupported pair importance file format: {source_path}")

    nested = payload.get("pair_importance")
    if not isinstance(nested, dict):
        raise KeyError(f"key `pair_importance` not found in {source_path}")

    pair_scores: Dict[Tuple[str, int], float] = {}
    block_names = []
    step_keys = set()

    for block_name, step_map in nested.items():
        if not isinstance(step_map, dict):
            raise TypeError(f"pair_importance[{block_name!r}] should be a dict")
        block_name = str(block_name)
        block_names.append(block_name)
        for step_key, value in step_map.items():
            step_key = str(step_key)
            if not step_key.startswith("denoise_step_"):
                raise ValueError(f"unexpected step key `{step_key}` in {source_path}")
            step_idx = int(step_key.split("_")[-1])
            pair_scores[(block_name, step_idx)] = float(value)
            step_keys.add(step_key)

    return pair_scores, {
        "block_names": sorted(block_names),
        "step_keys": sorted(step_keys, key=lambda item: int(item.split("_")[-1])),
    }


def _derive_auto_task_label(task: Optional[str], checkpoint: str) -> str:
    """Derive a filesystem-safe label for generated importance artifacts.

    Args:
        task (Optional[str]): Optional benchmark task name.
        checkpoint (str): Checkpoint path used as a fallback label source.

    Returns:
        str: Sanitized label used in output filenames.
    """
    raw_label = task or Path(checkpoint).stem or "custom_checkpoint"
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_label).strip("._-")
    return sanitized or "custom_checkpoint"


def _ensure_importance_artifacts(
    task: Optional[str],
    checkpoint: str,
    output_dir: str,
    device: str,
    activations_path: Optional[str],
    pair_importance_path: Optional[str],
    sample_steps: int,
    importance_mode: str = "l1_adjacent",
) -> Dict[str, Any]:
    """Ensure activation and pair-importance artifacts exist for search.

    Args:
        task (Optional[str]): Benchmark task name.
        checkpoint (str): Pretrained policy checkpoint path.
        output_dir (str): Search output directory.
        device (str): Torch device used if activation collection is required.
        activations_path (Optional[str]): Existing activation artifact path.
        pair_importance_path (Optional[str]): Existing pair-importance artifact
            path.
        sample_steps (int): Number of rollout steps sampled for dissimilarity.
        importance_mode (str): Pair-importance estimator, either adjacent L1 or
            global cosine dissimilarity.

    Returns:
        Dict[str, Any]: Metadata describing generated or reused artifacts and
        resolved file paths.
    """
    if pair_importance_path:
        return {
            "mode": "user_provided",
            "importance_mode": importance_mode,
            "task_label": _derive_auto_task_label(task, checkpoint),
            "importance_dir": None,
            "generated": {},
            "reused": {},
            "resolved_paths": {
                "activations_path": activations_path,
                "pair_importance_path": pair_importance_path,
            },
        }

    task_label = _derive_auto_task_label(task, checkpoint)
    importance_dir = Path(output_dir) / "importance"
    importance_dir.mkdir(parents=True, exist_ok=True)

    resolved_activations_path = importance_dir / "block_activations.pkl"
    if importance_mode == "cosine_global":
        resolved_pair_importance_path = importance_dir / f"pair_importance_{task_label}_cosine_global.json"
    elif importance_mode == "l1_adjacent":
        resolved_pair_importance_path = importance_dir / f"pair_importance_{task_label}.json"
    else:
        raise ValueError(f"unknown importance_mode: {importance_mode}")

    generated = {
        "activations": False,
        "pair_importance": False,
    }
    reused = {
        "activations": False,
        "pair_importance": False,
    }

    logger.info(
        "Using automatic pair-importance assets under: %s",
        importance_dir,
    )

    if resolved_pair_importance_path.exists():
        reused["pair_importance"] = True
        logger.info("Reusing pair-importance file: %s", resolved_pair_importance_path)

    if not resolved_pair_importance_path.exists():
        if activations_path:
            resolved_activations_path = Path(activations_path)
            if not resolved_activations_path.exists():
                raise FileNotFoundError(f"activations file not found: {resolved_activations_path}")
            reused["activations"] = True
            logger.info("Using provided activations for pair importance: %s", resolved_activations_path)
        elif resolved_activations_path.exists():
            reused["activations"] = True
            logger.info("Reusing activation file: %s", resolved_activations_path)
        else:
            logger.info("Collecting rollout block activations for pair importance...")
            activations = collect_block_activations_for_task(
                checkpoint=checkpoint,
                task_name=task_label,
                output_dir=str(importance_dir),
                device=device,
            )
            if not activations or not resolved_activations_path.exists():
                raise RuntimeError(
                    "Failed to generate block_activations.pkl for pair-importance computation."
                )
            generated["activations"] = True
            logger.info("Generated activation file: %s", resolved_activations_path)

        activations = load_activations(str(resolved_activations_path))
        logger.info("Computing pair importance with mode=%s", importance_mode)
        if importance_mode == "cosine_global":
            error_matrix, denoise_step_indices, block_names = compute_cosine_global_pair_importance_matrix(
                activations, sample_steps=sample_steps
            )
        else:
            error_matrix, denoise_step_indices, block_names = compute_step_error_matrix(
                activations, sample_steps=sample_steps
            )
        pair_json_path, _ = save_pair_importance_results(
            error_matrix=error_matrix,
            denoise_step_indices=denoise_step_indices,
            block_names=block_names,
            task_name=f"{task_label}_cosine_global" if importance_mode == "cosine_global" else task_label,
            checkpoint_path=checkpoint,
            output_dir=str(importance_dir),
        )
        resolved_pair_importance_path = Path(pair_json_path)
        generated["pair_importance"] = True

    return {
        "mode": "auto_generated",
        "importance_mode": importance_mode,
        "task_label": task_label,
        "importance_dir": str(importance_dir.resolve()),
        "generated": generated,
        "reused": reused,
        "resolved_paths": {
            "activations_path": str(resolved_activations_path.resolve()),
            "pair_importance_path": str(resolved_pair_importance_path.resolve()),
        },
    }


def _compute_pair_importance_scores(
    task: Optional[str],
    checkpoint: str,
    output_dir: str,
    device: str,
    activations_path: Optional[str],
    pair_importance_path: Optional[str],
    sample_steps: int,
    importance_mode: str = "l1_adjacent",
) -> Tuple[Dict[Tuple[str, int], float], Dict[str, Any]]:
    """Compute or load pair-importance scores for initialization.

    Args:
        task (Optional[str]): Benchmark task name.
        checkpoint (str): Pretrained policy checkpoint path.
        output_dir (str): Search output directory.
        device (str): Torch device used for automatic activation collection.
        activations_path (Optional[str]): Optional existing activation artifact.
        pair_importance_path (Optional[str]): Optional existing pair-importance
            artifact.
        sample_steps (int): Number of rollout steps sampled for dissimilarity.
        importance_mode (str): Importance estimator name.

    Returns:
        Tuple[Dict[Tuple[str, int], float], Dict[str, Any]]: Pair scores and
        metadata for the search report.
    """
    auto_asset_meta = _ensure_importance_artifacts(
        task=task,
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        activations_path=activations_path,
        pair_importance_path=pair_importance_path,
        sample_steps=sample_steps,
        importance_mode=importance_mode,
    )
    pair_importance_path = auto_asset_meta["resolved_paths"]["pair_importance_path"]

    pair_scores, pair_meta = _load_pair_importance_payload(pair_importance_path)
    return pair_scores, {
        "pair_importance_source": os.path.abspath(pair_importance_path),
        "pair_importance_meta": pair_meta,
        "auto_generated_assets": auto_asset_meta,
    }


def _build_pair_entries(
    block_names: Sequence[str],
    num_steps: int,
    pair_scores: Dict[Tuple[str, int], float],
) -> List[Dict[str, Any]]:
    """Build flattened pair entries from block-step importance scores.

    Args:
        block_names (Sequence[str]): EVO cacheable block names.
        num_steps (int): Number of denoising steps.
        pair_scores (Dict[Tuple[str, int], float]): Importance scores keyed by
            block name and timestep.

    Returns:
        List[Dict[str, Any]]: Pair entries sorted by descending importance.
    """
    entries: List[Dict[str, Any]] = []

    missing = []
    for block_name in block_names:
        for step_idx in range(num_steps):
            if (block_name, step_idx) not in pair_scores:
                missing.append((block_name, step_idx))
                if len(missing) >= 10:
                    break
        if len(missing) >= 10:
            break
    if missing:
        raise click.UsageError(
            "pair importance is missing required block-step entries: "
            f"{missing}"
        )
    logger.info(
        "Pair scoring mode: using explicit pair importance for %d blocks x %d steps.",
        len(block_names),
        num_steps,
    )

    for block_idx, block_name in enumerate(block_names):
        for step_idx in range(num_steps):
            pair_score = float(pair_scores[(block_name, step_idx)])
            entries.append(
                {
                    "pair_id": block_idx * num_steps + step_idx,
                    "block_idx": block_idx,
                    "step_idx": step_idx,
                    "block_name": block_name,
                    "score": float(pair_score),
                    "score_mode": "pair_matrix",
                    "sources": [],
                }
            )

    entries.sort(key=lambda item: item["score"], reverse=True)
    return entries


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON payload after converting NumPy values to Python types.

    Args:
        path (str): Destination JSON path.
        payload (Dict[str, Any]): Serializable or NumPy-containing metadata.

    Returns:
        None.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(payload), f, indent=2, ensure_ascii=False)


def _default_baseline_registry_path() -> str:
    """Return the default formal-baseline registry path.

    Args:
        None.

    Returns:
        str: Repository-local JSON path for canonical baseline scores.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "results" / "baselines" / "formal_baselines.json")


def _default_official_baseline_eval_path(task_label: str) -> str:
    """Return the default baseline evaluation artifact for one task.

    Args:
        task_label (str): Benchmark task label.

    Returns:
        str: Repository-local ``eval_results.json`` path for original policy
        evaluation.
    """
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "results" / task_label / "original" / "eval_results.json")


def _load_json_or_default(path: str, default: Dict[str, Any]) -> Dict[str, Any]:
    """Load JSON if present, otherwise return a deep copy of a default.

    Args:
        path (str): JSON path to load.
        default (Dict[str, Any]): Default mapping returned when the file is
            absent.

    Returns:
        Dict[str, Any]: Loaded JSON payload or copied default.
    """
    source = Path(path)
    if not source.exists():
        return copy.deepcopy(default)
    with open(source, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_file(path: str) -> Dict[str, Any]:
    """Load a JSON file.

    Args:
        path (str): Path to a JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON payload.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_official_baseline_eval_result(path: str) -> Dict[str, Any]:
    """Load and validate an original-policy evaluation result.

    Args:
        path (str): Path to ``eval_results.json`` generated in original mode.

    Returns:
        Dict[str, Any]: Validated baseline score payload.
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"official baseline eval_results not found: {source}")

    payload = _load_json_file(str(source))
    cache_mode = payload.get("cache_mode")
    if cache_mode not in (None, "original"):
        raise ValueError(
            f"official baseline eval_results cache_mode must be original, got {cache_mode!r}: {source}"
        )
    if "mean_score" not in payload:
        raise KeyError(f"official baseline eval_results is missing mean_score: {source}")

    return {
        "path": str(source.resolve()),
        "mean_score": float(payload["mean_score"]),
        "payload": payload,
    }


def _is_canonical_baseline_entry(entry: Optional[Dict[str, Any]]) -> bool:
    """Check whether a baseline registry entry is considered canonical.

    Args:
        entry (Optional[Dict[str, Any]]): Baseline registry entry.

    Returns:
        bool: ``True`` when the entry comes from an official evaluation result
        or an explicit manual override.
    """
    if not isinstance(entry, dict):
        return False
    manual_override = entry.get("manual_override")
    if isinstance(manual_override, dict) and manual_override.get("enabled"):
        return True
    if entry.get("source") == "official_eval_results":
        return True
    return bool(entry.get("official_eval_result_path"))


def _make_json_safe(value: Any) -> Any:
    """Recursively convert values into JSON-safe Python objects.

    Args:
        value (Any): Arbitrary object that may contain NumPy values or paths.

    Returns:
        Any: JSON-serializable representation.
    """
    if isinstance(value, dict):
        return {str(key): _make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


class _ParallelFitnessWorker:
    """Worker that evaluates candidate schedules on one assigned GPU.

    Args:
        checkpoint_path (str): Pretrained policy checkpoint path.
        num_blocks (int): Number of cacheable residual branches.
        num_steps (int): Number of denoising timesteps.
        device (str): Torch device assigned to this worker.
        seed (int): Worker-local reproducibility seed.
        eval_seed (int): Rollout seed for quick fitness evaluation.
        deterministic_torch (bool): Whether to request deterministic Torch/CUDA
            behavior.

    Returns:
        None.
    """

    def __init__(
        self,
        checkpoint_path: str,
        num_blocks: int,
        num_steps: int,
        device: str,
        *,
        seed: int,
        eval_seed: int,
        deterministic_torch: bool,
    ):
        """Initialize a parallel fitness worker.

        Args:
            checkpoint_path (str): Pretrained policy checkpoint path.
            num_blocks (int): Number of cacheable residual branches.
            num_steps (int): Number of denoising timesteps.
            device (str): Torch device assigned to this worker.
            seed (int): Worker-local reproducibility seed.
            eval_seed (int): Rollout seed for quick fitness evaluation.
            deterministic_torch (bool): Whether to request deterministic
                Torch/CUDA behavior.

        Returns:
            None.
        """
        self.checkpoint_path = checkpoint_path
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.device = device
        self.worker_reproducibility = _configure_reproducibility(
            seed,
            deterministic_torch=deterministic_torch,
        )
        self.eval_seed = int(eval_seed)
        self.policy, self.cache_wrapper, self.cfg = _load_policy_and_wrapper(checkpoint_path, device)

        self.ij_pairs = []
        for i in range(num_blocks):
            for j in range(num_steps):
                self.ij_pairs.append((i, j))

    def _run_environment_evaluation(self):
        """Run a one-episode quick rollout for the current worker policy.

        Args:
            None.

        Returns:
            Dict[str, Any]: Evaluation payload containing fitness and raw
            environment-runner logs, or an error record.
        """
        try:
            import tempfile
            import contextlib
            with tempfile.TemporaryDirectory() as temp_dir:
                env_runner_cfg = OmegaConf.to_container(self.cfg.task.env_runner, resolve=True)
                env_runner_cfg['n_train'] = 0
                env_runner_cfg['n_train_vis'] = 0
                env_runner_cfg['n_test'] = 1
                env_runner_cfg['n_test_vis'] = 0
                env_runner_cfg['n_envs'] = 1
                env_runner_cfg['test_start_seed'] = self.eval_seed

                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        env_runner = hydra.utils.instantiate(
                            env_runner_cfg,
                            output_dir=temp_dir
                        )
                        log_data = env_runner.run(self.cache_wrapper)

                return log_data

        except Exception as e:
            logger.error("[%s] environment evaluation failed: %s", self.device, e)
            import traceback
            logger.error("[%s] traceback: %s", self.device, traceback.format_exc())
            return {'test/mean_score': 0.0}

    def evaluate_solution(self, solution: List[int]) -> float:
        """Evaluate one candidate schedule with a quick rollout.

        Args:
            solution (List[int]): Flattened block–timestep positions selected by
                the candidate schedule.

        Returns:
            float: Quick rollout success score used as fitness.
        """
        try:
            ij_pairs = [self.ij_pairs[idx] for idx in solution]
            self.cache_wrapper.set_optimal_steps(ij_pairs)
            _configure_torch_eval_seed(self.eval_seed)
            with torch.no_grad():
                eval_results = self._run_environment_evaluation()
            return eval_results.get('test/mean_score', 0.0)
        except Exception as e:
            logger.error("[%s] solution evaluation failed: %s", self.device, e)
            import traceback
            logger.error("[%s] traceback: %s", self.device, traceback.format_exc())
            return 0.0

    def cleanup(self):
        """Release the worker-side cache wrapper.

        Args:
            None.

        Returns:
            None.
        """
        if self.cache_wrapper:
            self.cache_wrapper.cleanup()


def _init_parallel_worker(
    checkpoint_path: str,
    num_blocks: int,
    num_steps: int,
    devices: List[str],
    seed: int,
    eval_seed: int,
    deterministic_torch: bool,
):
    """Initialize a process-local worker for parallel schedule evaluation.

    Args:
        checkpoint_path (str): Pretrained policy checkpoint path.
        num_blocks (int): Number of cacheable residual branches.
        num_steps (int): Number of denoising timesteps.
        devices (List[str]): Available worker devices.
        seed (int): Base worker seed.
        eval_seed (int): Rollout seed for quick fitness evaluation.
        deterministic_torch (bool): Whether to request deterministic Torch/CUDA
            behavior.

    Returns:
        None.
    """
    global _WORKER_STATE
    identity = mp.current_process()._identity
    worker_idx = identity[0] - 1 if identity else 0
    device = devices[worker_idx % len(devices)]
    torch.set_num_threads(1)
    _WORKER_STATE = _ParallelFitnessWorker(
        checkpoint_path=checkpoint_path,
        num_blocks=num_blocks,
        num_steps=num_steps,
        device=device,
        seed=seed + worker_idx,
        eval_seed=eval_seed,
        deterministic_torch=deterministic_torch,
    )


def _evaluate_solution_in_worker(solution: List[int]) -> float:
    """Evaluate a candidate schedule in the current process worker.

    Args:
        solution (List[int]): Flattened block–timestep positions selected by the
            candidate schedule.

    Returns:
        float: Quick rollout fitness score.
    """
    if _WORKER_STATE is None:
        raise RuntimeError("Parallel worker has not been initialized.")
    return _WORKER_STATE.evaluate_solution(solution)


class Individual:
    """One candidate cache schedule in the evolutionary population.

    Args:
        solution (List[int]): Flattened block–timestep positions selected by the
            schedule.
        fitness (Optional[float]): Cached rollout fitness value.

    Returns:
        None.
    """
    
    def __init__(self, solution: List[int], fitness: Optional[float] = None):
        """Create an individual from a fixed-budget schedule.

        Args:
            solution (List[int]): Flattened block–timestep positions.
            fitness (Optional[float]): Optional rollout fitness score.

        Returns:
            None.
        """
        self.solution = solution
        self.fitness = fitness
        self.age = 0
    
    def __str__(self):
        """Format a concise individual summary for logs.

        Args:
            None.

        Returns:
            str: Human-readable fitness, age, and schedule-size summary.
        """
        fitness_str = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Individual(fitness={fitness_str}, age={self.age}, solution_size={len(self.solution)})"
    
    def copy(self):
        """Return a shallow copy of this individual.

        Args:
            None.

        Returns:
            Individual: Copy with duplicated solution list and same fitness.
        """
        cloned = Individual(self.solution.copy(), self.fitness)
        cloned.age = self.age
        return cloned


class GeneticOptimizer:
    """Evolutionary optimizer for fixed-budget EVO cache schedules.

    Args:
        checkpoint_path (str): Pretrained Diffusion Policy checkpoint path.
        num_blocks (int): Number of cacheable residual branches.
        num_steps (int): Number of denoising timesteps.
        K (int): Number of block–timestep positions refreshed by each schedule.
        task_name (Optional[str]): Benchmark task name used in logs and output
            paths.
        population_size (int): Number of schedules in each generation.
        elite_size (int): Number of top schedules preserved between generations.
        mutation_rate (float): Probability of applying mutation to a child.
        crossover_rate (float): Probability of set-level crossover.
        tournament_size (int): Number of candidates sampled for tournament
            selection.
        max_generations (int): Maximum number of evolutionary generations.
        device (str): Torch device or comma-separated devices for evaluation.
        output_dir (Optional[str]): Directory for traces and result files.
        baseline_registry_path (Optional[str]): JSON registry of formal baseline
            scores.
        baseline_allowed_drop (float): Legacy absolute baseline-drop tolerance.
        baseline_allowed_drop_ratio (float): Legacy relative baseline-drop
            tolerance.
        formal_accept_drop (float): Allowed formal-score drop from the original
            baseline.
        meaningful_improvement_eps (float): Minimum quick-fitness improvement
            treated as meaningful.
        fresh_eval_patience_after_confirm (int): Fresh-evaluation patience after
            a formal confirmation.
        formal_confirm_cooldown_fresh_evals (int): Fresh-evaluation cooldown
            between formal confirmations.
        max_fresh_evals (Optional[int]): Maximum number of quick evaluations.
        experiment_seed (int): Seed for evolutionary search operations.
        eval_seed (Optional[int]): Base seed for rollout evaluation.
        deterministic_torch (bool): Whether to request deterministic Torch/CUDA
            behavior.
        init_strategy (str): Initial population strategy.
        init_seed (Optional[int]): Seed for population initialization.
        init_profile (str): Structural profile name for guided initialization.
        init_random_fraction (float): Fraction of initial population sampled
            uniformly at random.
        init_structure_min_pairs_per_block (int): Minimum positions per block in
            guided initialization.
        init_structure_tail_min (float): Minimum random tail ratio.
        init_structure_tail_max (float): Maximum random tail ratio.
        init_pair_weights (Optional[Dict[int, float]]): Pair-importance weights
            from activation dissimilarity.
        init_importance_gamma (float): Exponent for importance-guided sampling.
        init_importance_epsilon (float): Additive sampling floor.

    Returns:
        None.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 num_blocks: int,
                 num_steps: int,
                 K: int,
                 task_name: Optional[str] = None,
                 population_size: int = 50,
                 elite_size: int = 10,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 tournament_size: int = 3,
                 max_generations: int = 100,
                 device: str = 'cuda:0',
                 output_dir: Optional[str] = None,
                 baseline_registry_path: Optional[str] = None,
                 baseline_allowed_drop: float = 0.01,
                 baseline_allowed_drop_ratio: float = 0.02,
                 formal_accept_drop: float = 0.02,
                 meaningful_improvement_eps: float = 0.005,
                 fresh_eval_patience_after_confirm: int = 40,
                 formal_confirm_cooldown_fresh_evals: int = 20,
                 max_fresh_evals: Optional[int] = 240,
                 experiment_seed: int = 20260325,
                 eval_seed: Optional[int] = None,
                 deterministic_torch: bool = True,
                 init_strategy: str = "random",
                 init_seed: Optional[int] = None,
                 init_structure_profile: str = "task",
                 init_random_fraction: float = 0.30,
                 init_structure_min_pairs_per_block: int = 4,
                 init_structure_tail_min: float = 0.0,
                 init_structure_tail_max: float = 0.0,
                 init_pair_weights: Optional[Dict[int, float]] = None,
                 init_importance_gamma: float = 1.0,
                 init_importance_epsilon: float = 0.005):
        """Initialize the EVO evolutionary optimizer and its first population.

        Args:
            checkpoint_path (str): Pretrained Diffusion Policy checkpoint path.
            num_blocks (int): Number of cacheable residual branches.
            num_steps (int): Number of denoising timesteps.
            K (int): Fixed refresh budget per candidate schedule.
            task_name (Optional[str]): Benchmark task name.
            population_size (int): Number of candidate schedules per generation.
            elite_size (int): Number of elite schedules retained.
            mutation_rate (float): Mutation probability.
            crossover_rate (float): Crossover probability.
            tournament_size (int): Tournament size for parent selection.
            max_generations (int): Maximum number of generations.
            device (str): Device specification for rollout evaluation.
            output_dir (Optional[str]): Directory for trace and result files.
            baseline_registry_path (Optional[str]): Formal baseline registry.
            baseline_allowed_drop (float): Legacy absolute drop tolerance.
            baseline_allowed_drop_ratio (float): Legacy relative drop tolerance.
            formal_accept_drop (float): Accepted formal-score drop from baseline.
            meaningful_improvement_eps (float): Minimum meaningful quick-score
                improvement.
            fresh_eval_patience_after_confirm (int): Patience after formal
                confirmation.
            formal_confirm_cooldown_fresh_evals (int): Cooldown between formal
                confirmations.
            max_fresh_evals (Optional[int]): Maximum quick evaluations.
            experiment_seed (int): Seed for evolutionary operators.
            eval_seed (Optional[int]): Seed for rollout evaluation.
            deterministic_torch (bool): Whether to request deterministic Torch
                behavior.
            init_strategy (str): Population initialization strategy.
            init_seed (Optional[int]): Initialization seed.
            init_structure_profile (str): Structural prior profile.
            init_random_fraction (float): Random fraction of initial population.
            init_structure_min_pairs_per_block (int): Per-block floor for guided
                initialization.
            init_structure_tail_min (float): Minimum random tail ratio.
            init_structure_tail_max (float): Maximum random tail ratio.
            init_pair_weights (Optional[Dict[int, float]]): Activation
                dissimilarity weights for pair-guided initialization.
            init_importance_gamma (float): Importance sampling exponent.
            init_importance_epsilon (float): Additive exploration floor.

        Returns:
            None.
        """
        
        self.checkpoint_path = checkpoint_path
        self.num_blocks = num_blocks
        self.num_steps = num_steps
        self.task_name = task_name
        self.task_label = _derive_auto_task_label(task_name, checkpoint_path)
        self.K = K
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.device_spec = device
        self.device_list = _parse_device_list(device)
        self.parallel_eval = len(self.device_list) > 1
        self.device = torch.device(self.device_list[0])
        self.pool = None
        self.output_dir = output_dir
        
        # Enumerate all candidate (block, denoise_step) pairs.
        self.ij_pairs = []
        for i in range(num_blocks):
            for j in range(num_steps):
                self.ij_pairs.append((i, j))
        
        self.N = len(self.ij_pairs)
        self.solution_universe = list(range(self.N))
        
        # Load policy locally unless evaluations are distributed across devices.
        self.policy = None
        self.cache_wrapper = None
        self.cfg = None
        if self.parallel_eval:
            logger.info("Using parallel evaluation on devices: %s", self.device_list)
        else:
            self._load_policy()
        
        # Search state.
        self.population = []
        self.best_individual = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.diversity_history = []
        self.generation_summaries: List[Dict[str, Any]] = []
        self.fitness_cache: Dict[Tuple[int, ...], float] = {}
        self.retired_signatures = set()
        self.baseline_registry_path = baseline_registry_path or _default_baseline_registry_path()
        self.baseline_allowed_drop = baseline_allowed_drop
        self.baseline_allowed_drop_ratio = baseline_allowed_drop_ratio
        self.formal_accept_drop = formal_accept_drop
        self.meaningful_improvement_eps = meaningful_improvement_eps
        self.fresh_eval_patience_after_confirm = fresh_eval_patience_after_confirm
        self.formal_confirm_cooldown_fresh_evals = formal_confirm_cooldown_fresh_evals
        self.max_fresh_evals = max_fresh_evals
        self.experiment_seed = int(experiment_seed)
        self.eval_seed = int(eval_seed) if eval_seed is not None else self.experiment_seed
        self.deterministic_torch = bool(deterministic_torch)
        self.init_strategy = init_strategy
        self.init_seed = int(init_seed) if init_seed is not None else self.experiment_seed
        self.init_structure_profile = init_structure_profile
        self.init_random_fraction = init_random_fraction
        self.init_structure_min_pairs_per_block = init_structure_min_pairs_per_block
        self.init_structure_tail_min = init_structure_tail_min
        self.init_structure_tail_max = init_structure_tail_max
        self.init_pair_weights = init_pair_weights
        self.init_importance_gamma = init_importance_gamma
        self.init_importance_epsilon = init_importance_epsilon
        self.initialization_meta: Dict[str, Any] = {}
        self.quick_eval_seed = self.eval_seed
        self.formal_eval_seed = self.eval_seed + 1000003
        self.early_stop_avg_window = 4
        self.baseline_entry: Optional[Dict[str, Any]] = None
        self.formal_baseline_score: Optional[float] = None
        self.confirm_trigger_target: Optional[float] = None
        self.formal_accept_target: Optional[float] = None
        self.hard_target: Optional[float] = None
        self.soft_target: Optional[float] = None
        self.formal_confirm_history: List[Dict[str, Any]] = []
        self.confirmed_signatures: Dict[Tuple[int, ...], Dict[str, Any]] = {}
        self.best_confirmed_candidate: Optional[Dict[str, Any]] = None
        self.best_formal_candidate: Optional[Dict[str, Any]] = None
        self.final_selected_candidate: Optional[Dict[str, Any]] = None
        self.final_search_best_formal_evaluation: Optional[Dict[str, Any]] = None
        self.total_fresh_evaluations = 0
        self.fresh_evals_since_meaningful_best = 0
        self.last_meaningful_best_fitness = float('-inf')
        self.last_meaningful_best_generation: Optional[int] = None
        self.last_meaningful_best_phase: Optional[str] = None
        self.last_confirm_fresh_eval_total: Optional[int] = None
        self.stop_reason: Optional[str] = None
        self.stop_context: Dict[str, Any] = {}
        self.timing_summary: Dict[str, Any] = {
            'initial_population_eval_seconds': 0.0,
            'evolution_loop_seconds': 0.0,
            'parallel_pool_init_seconds': 0.0,
            'optimize_total_seconds': 0.0,
            'result_save_seconds': 0.0,
            'cleanup_seconds': 0.0,
        }
        self.reproducibility_meta: Dict[str, Any] = {
            'experiment_seed': self.experiment_seed,
            'search_seed': self.experiment_seed,
            'eval_seed': self.eval_seed,
            'deterministic_torch': self.deterministic_torch,
            'quick_eval_seed': self.quick_eval_seed,
            'formal_eval_seed': self.formal_eval_seed,
            'init_strategy': self.init_strategy,
            'init_seed': self.init_seed,
            'init_structure_profile': self.init_structure_profile,
            'init_random_fraction': self.init_random_fraction,
            'init_importance_gamma': self.init_importance_gamma,
            'init_importance_epsilon': self.init_importance_epsilon,
        }
        self.best_found_generation: Optional[int] = None
        self.best_found_phase: Optional[str] = None
        self.best_update_history: List[Dict[str, Any]] = []
        self._active_generation = -1
        self._active_phase = 'initial'
        self._last_generation_build_stats: Dict[str, Any] = {}
        self._last_population_eval_stats: Dict[str, Any] = {}
        self.individual_trace_path = (
            os.path.join(self.output_dir, 'individual_eval_trace.jsonl')
            if self.output_dir is not None
            else None
        )
        self.early_stop_trace_path = (
            os.path.join(self.output_dir, 'early_stop_trace.jsonl')
            if self.output_dir is not None
            else None
        )
        if self.individual_trace_path is not None:
            Path(self.individual_trace_path).write_text('', encoding='utf-8')
        if self.early_stop_trace_path is not None:
            Path(self.early_stop_trace_path).write_text('', encoding='utf-8')
        
        
        # Initialize the population immediately so optimization can start.
        self._initialize_population()
        
    def _sample_solution(self) -> List[int]:
        """Sample a random fixed-budget cache schedule.

        Args:
            None.

        Returns:
            List[int]: Sorted flattened block–timestep positions.
        """
        return sorted(random.sample(self.solution_universe, self.K))

    def _solution_signature(self, solution: Sequence[int]) -> Tuple[int, ...]:
        """Build the canonical signature used for schedule deduplication.

        Args:
            solution (Sequence[int]): Flattened block–timestep positions.

        Returns:
            Tuple[int, ...]: Hashable schedule signature.
        """
        return tuple(solution)

    def _canonicalize_solution(self, solution: Sequence[int]) -> List[int]:
        """Repair a schedule to exactly ``K`` unique positions.

        Args:
            solution (Sequence[int]): Candidate flattened positions.

        Returns:
            List[int]: Sorted fixed-budget schedule.
        """
        unique_sorted = sorted(dict.fromkeys(int(idx) for idx in solution))
        if len(unique_sorted) > self.K:
            unique_sorted = sorted(random.sample(unique_sorted, self.K))

        if len(unique_sorted) < self.K:
            existing = set(unique_sorted)
            remaining = [idx for idx in self.solution_universe if idx not in existing]
            unique_sorted.extend(random.sample(remaining, self.K - len(unique_sorted)))
            unique_sorted.sort()
        return unique_sorted

    def _fill_solution_to_k(self, solution: List[int]) -> List[int]:
        """Alias for canonical schedule repair.

        Args:
            solution (List[int]): Candidate flattened positions.

        Returns:
            List[int]: Sorted fixed-budget schedule.
        """
        return self._canonicalize_solution(solution)

    def _sample_unique_solution(
        self,
        blocked_signatures: Optional[Sequence[Tuple[int, ...]]] = None,
        max_attempts: int = 512,
    ) -> List[int]:
        """Sample a random schedule that avoids blocked signatures.

        Args:
            blocked_signatures (Optional[Sequence[Tuple[int, ...]]]): Schedule
                signatures that should not be sampled.
            max_attempts (int): Maximum retry count.

        Returns:
            List[int]: Unique random fixed-budget schedule.
        """
        blocked = set(blocked_signatures or [])
        for _ in range(max_attempts):
            solution = self._sample_solution()
            signature = self._solution_signature(solution)
            if signature not in blocked:
                return solution
        raise RuntimeError("Failed to sample a unique solution after max_attempts.")

    def _accept_child(
        self,
        child: Individual,
        new_population: List[Individual],
        new_population_signatures: set,
    ) -> Tuple[bool, str]:
        """Accept a child schedule if it is unique and not retired.

        Args:
            child (Individual): Candidate child schedule.
            new_population (List[Individual]): Population being assembled.
            new_population_signatures (set): Signatures already present in the
                new generation.

        Returns:
            Tuple[bool, str]: Acceptance flag and reason string.
        """
        child.solution = self._canonicalize_solution(child.solution)
        signature = self._solution_signature(child.solution)
        if signature in new_population_signatures:
            return False, 'same_gen_duplicate'
        if signature in self.retired_signatures:
            return False, 'retired_duplicate'
        new_population.append(child)
        new_population_signatures.add(signature)
        return True, 'accepted'

    def _append_individual_trace(self, record: Dict[str, Any]):
        """Append one individual-evaluation record to the trace file.

        Args:
            record (Dict[str, Any]): Evaluation metadata for one candidate.

        Returns:
            None.
        """
        if self.individual_trace_path is None:
            return
        with open(self.individual_trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(_make_json_safe(record), ensure_ascii=False) + '\n')

    def _append_early_stop_trace(self, record: Dict[str, Any]):
        """Append one target-conditioned early-stopping record.

        Args:
            record (Dict[str, Any]): Early-stopping or formal-confirmation
                metadata.

        Returns:
            None.
        """
        if self.early_stop_trace_path is None:
            return
        with open(self.early_stop_trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(_make_json_safe(record), ensure_ascii=False) + '\n')

    
    def _load_policy(self):
        """Load the local policy and cache wrapper.

        Args:
            None.

        Returns:
            None.
        """
        self.policy, self.cache_wrapper, self.cfg = _load_policy_and_wrapper(
            self.checkpoint_path,
            str(self.device)
        )

    def _ensure_local_policy_loaded(self):
        """Load the local policy when a formal evaluation needs it.

        Args:
            None.

        Returns:
            None.
        """
        if self.cache_wrapper is None or self.cfg is None:
            self._load_policy()

    def _build_env_runner_cfg(self, eval_mode: str) -> Dict[str, Any]:
        """Build environment-runner settings for quick or formal rollouts.

        Args:
            eval_mode (str): ``"quick"`` for low-cost screening or ``"formal"``
                for independent target verification.

        Returns:
            Dict[str, Any]: Resolved environment-runner configuration.
        """
        self._ensure_local_policy_loaded()
        env_runner_cfg = OmegaConf.to_container(self.cfg.task.env_runner, resolve=True)
        env_runner_cfg['n_train'] = 0
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test_vis'] = 0
        if eval_mode == 'quick':
            env_runner_cfg['n_test'] = 20
            env_runner_cfg['n_envs'] = 20
            env_runner_cfg['test_start_seed'] = self.quick_eval_seed
        elif eval_mode == 'formal':
            env_runner_cfg.setdefault('n_test', 50)
            env_runner_cfg['test_start_seed'] = self.formal_eval_seed
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode}")
        return env_runner_cfg

    def _build_baseline_registry_key(self, formal_env_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Build a stable key for formal baseline lookup.

        Args:
            formal_env_cfg (Dict[str, Any]): Formal evaluation runner
                configuration.

        Returns:
            Tuple[str, Dict[str, Any]]: Serialized key and its structured
            payload.
        """
        checkpoint_abs = str(Path(self.checkpoint_path).resolve())
        key_payload = {
            'task_label': self.task_label,
            'checkpoint': checkpoint_abs,
            'env_runner_target': formal_env_cfg.get('_target_'),
            'n_test': formal_env_cfg.get('n_test'),
            'n_envs': formal_env_cfg.get('n_envs'),
            'test_start_seed': formal_env_cfg.get('test_start_seed'),
            'max_steps': formal_env_cfg.get('max_steps'),
            'cache_mode': 'original',
        }
        key = json.dumps(key_payload, ensure_ascii=False, sort_keys=True)
        return key, key_payload

    def _official_baseline_eval_path(self) -> str:
        """Return the expected original-policy evaluation result path.

        Args:
            None.

        Returns:
            str: Path to the task-level baseline ``eval_results.json``.
        """
        return _default_official_baseline_eval_path(self.task_label)

    def _ensure_official_baseline_eval_result(self) -> Dict[str, Any]:
        """Load or generate the official original-policy baseline result.

        Args:
            None.

        Returns:
            Dict[str, Any]: Validated baseline evaluation result.
        """
        baseline_eval_path = self._official_baseline_eval_path()
        baseline_eval_output_dir = str(Path(baseline_eval_path).parent)
        if Path(baseline_eval_path).exists():
            return _load_official_baseline_eval_result(baseline_eval_path)

        checkpoint_abs = str(Path(self.checkpoint_path).resolve())
        command = [
            sys.executable,
            "-m",
            "EVOInfer.scripts.eval_evo",
            "--checkpoint",
            checkpoint_abs,
            "--output_dir",
            baseline_eval_output_dir,
            "--device",
            self.device_list[0],
            "--cache_mode",
            "original",
            "--skip_video",
        ]
        logger.info(
            "Official baseline eval_results not found; launching baseline evaluation: %s",
            " ".join(shlex.quote(part) for part in command),
        )
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[2]),
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "Official baseline evaluation failed; cannot continue search."
            )
        return _load_official_baseline_eval_result(baseline_eval_path)

    def _ensure_task_baseline(self):
        """Ensure formal baseline and acceptance targets are initialized.

        Args:
            None.

        Returns:
            None.
        """
        if self.formal_baseline_score is not None:
            return

        formal_env_cfg = self._build_env_runner_cfg('formal')
        registry_key, registry_key_payload = self._build_baseline_registry_key(formal_env_cfg)
        registry = _load_json_or_default(
            self.baseline_registry_path,
            {
                'schema_version': 1,
                'entries': {},
            },
        )
        entries = registry.setdefault('entries', {})
        entry = entries.get(registry_key)
        baseline_source = 'registry'

        if not _is_canonical_baseline_entry(entry):
            previous_entry = copy.deepcopy(entry) if entry is not None else None
            baseline_eval = self._ensure_official_baseline_eval_result()
            logger.info(
                "Creating canonical baseline for %s: score=%.4f source=%s",
                self.task_label,
                baseline_eval['mean_score'],
                baseline_eval['path'],
            )
            entry = {
                'registry_key_payload': registry_key_payload,
                'task_name': self.task_name,
                'task_label': self.task_label,
                'checkpoint': str(Path(self.checkpoint_path).resolve()),
                'cache_mode': 'original',
                'formal_env_runner': formal_env_cfg,
                'baseline_score': float(baseline_eval['mean_score']),
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'source': 'official_eval_results',
                'official_eval_result_path': baseline_eval['path'],
                'official_eval_result': baseline_eval['payload'],
            }
            if previous_entry is not None:
                entry['replaced_noncanonical_entry'] = previous_entry
            entries[registry_key] = entry
            _write_json(self.baseline_registry_path, registry)
            baseline_source = 'official_eval_results'
            logger.info(
                "Canonical formal baseline saved: task=%s score=%.4f path=%s",
                self.task_label,
                entry['baseline_score'],
                self.baseline_registry_path,
            )
        else:
            logger.info(
                "Using canonical formal baseline: task=%s score=%.4f path=%s",
                self.task_label,
                entry['baseline_score'],
                self.baseline_registry_path,
            )

        self.baseline_entry = entry
        self.formal_baseline_score = float(entry['baseline_score'])
        legacy_allowed_drop = max(
            float(self.baseline_allowed_drop),
            float(self.baseline_allowed_drop_ratio) * self.formal_baseline_score,
        )
        self.confirm_trigger_target = self.formal_baseline_score
        self.formal_accept_target = self.formal_baseline_score - float(self.formal_accept_drop)
        self.hard_target = self.confirm_trigger_target
        self.soft_target = self.formal_accept_target
        self._append_early_stop_trace(
            {
                'event': 'baseline_ready',
                'task_label': self.task_label,
                'baseline_registry_path': self.baseline_registry_path,
                'source': baseline_source,
                'baseline_score': self.formal_baseline_score,
                'confirm_trigger_target': self.confirm_trigger_target,
                'formal_accept_target': self.formal_accept_target,
                'formal_accept_drop': self.formal_accept_drop,
                'hard_target': self.hard_target,
                'soft_target': self.soft_target,
                'legacy_allowed_drop': legacy_allowed_drop,
                'baseline_entry': entry,
            }
        )

    def _get_selected_candidate_payload(self) -> Optional[Dict[str, Any]]:
        """Return the candidate selected by early stopping or finalization.

        Args:
            None.

        Returns:
            Optional[Dict[str, Any]]: Selected candidate payload, if one exists.
        """
        return self.final_selected_candidate

    def _get_output_individual(self) -> Optional[Individual]:
        """Return the individual that should be written as final output.

        Args:
            None.

        Returns:
            Optional[Individual]: Final selected individual or the best quick
            individual.
        """
        selected = self._get_selected_candidate_payload()
        if selected is not None:
            return selected['individual']
        return self.best_individual

    def _get_output_quick_fitness(self) -> float:
        """Return the quick fitness associated with the final output.

        Args:
            None.

        Returns:
            float: Quick rollout fitness for the selected schedule.
        """
        selected = self._get_selected_candidate_payload()
        if selected is not None:
            return float(selected['quick_fitness'])
        return float(self.best_fitness)

    def _clone_candidate(self, candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Clone a candidate payload without sharing the individual object.

        Args:
            candidate (Optional[Dict[str, Any]]): Candidate payload to clone.

        Returns:
            Optional[Dict[str, Any]]: Deep-enough copy for search bookkeeping.
        """
        if candidate is None:
            return None
        return {
            'individual': candidate['individual'].copy(),
            'signature': candidate['signature'],
            'quick_fitness': float(candidate['quick_fitness']),
            'formal_fitness': float(candidate['formal_fitness']),
            'confirm_type': candidate['confirm_type'],
            'generation': candidate['generation'],
            'phase': candidate['phase'],
        }

    def _build_candidate_payload(
        self,
        individual: Individual,
        quick_fitness: float,
        formal_fitness: float,
        confirm_type: str,
        generation: Optional[int],
        phase: Optional[str],
    ) -> Dict[str, Any]:
        """Create a candidate payload from quick and formal evaluations.

        Args:
            individual (Individual): Candidate schedule.
            quick_fitness (float): Low-cost rollout fitness.
            formal_fitness (float): Independent formal rollout fitness.
            confirm_type (str): Label describing the confirmation outcome.
            generation (Optional[int]): Generation where the candidate appeared.
            phase (Optional[str]): Search phase where the candidate appeared.

        Returns:
            Dict[str, Any]: Candidate payload used by early stopping and final
            selection.
        """
        return {
            'individual': individual.copy(),
            'signature': self._solution_signature(individual.solution),
            'quick_fitness': float(quick_fitness),
            'formal_fitness': float(formal_fitness),
            'confirm_type': confirm_type,
            'generation': generation,
            'phase': phase,
        }

    def _is_better_formal_candidate(
        self,
        candidate: Dict[str, Any],
        current: Optional[Dict[str, Any]],
    ) -> bool:
        """Compare formal candidates using formal score, quick score, and tie-breaks.

        Args:
            candidate (Dict[str, Any]): Candidate being considered.
            current (Optional[Dict[str, Any]]): Current best formal candidate.

        Returns:
            bool: ``True`` if ``candidate`` should replace ``current``.
        """
        if current is None:
            return True
        candidate_formal = float(candidate['formal_fitness'])
        current_formal = float(current['formal_fitness'])
        if candidate_formal != current_formal:
            return candidate_formal > current_formal
        candidate_quick = float(candidate['quick_fitness'])
        current_quick = float(current['quick_fitness'])
        if candidate_quick != current_quick:
            return candidate_quick > current_quick
        return tuple(candidate['signature']) < tuple(current['signature'])

    def _update_best_formal_candidate(self, candidate: Dict[str, Any]) -> None:
        """Update the best formally evaluated candidate.

        Args:
            candidate (Dict[str, Any]): Candidate payload with formal fitness.

        Returns:
            None.
        """
        if self._is_better_formal_candidate(candidate, self.best_formal_candidate):
            self.best_formal_candidate = self._clone_candidate(candidate)

    def _set_final_selected_candidate(self, candidate: Optional[Dict[str, Any]], reason: str, details: Optional[Dict[str, Any]] = None):
        """Record the final schedule selected by early stopping or fallback.

        Args:
            candidate (Optional[Dict[str, Any]]): Candidate selected for output.
            reason (str): Stop or selection reason.
            details (Optional[Dict[str, Any]]): Additional context recorded in
                the early-stop trace.

        Returns:
            None.
        """
        if candidate is not None:
            self.final_selected_candidate = self._clone_candidate(candidate)
        self.stop_reason = reason
        self.stop_context = details or {}
        self._append_early_stop_trace(
            {
                'event': 'stop_decision',
                'reason': reason,
                'generation': self._active_generation,
                'phase': self._active_phase,
                'details': self.stop_context,
                'selected_candidate': self._candidate_to_json(self.final_selected_candidate),
            }
        )

    def _register_confirmed_candidate(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        """Register a candidate that passed formal evaluation.

        Args:
            candidate (Dict[str, Any]): Candidate payload with formal fitness.

        Returns:
            Dict[str, Any]: The same candidate payload.
        """
        if self._is_better_formal_candidate(candidate, self.best_confirmed_candidate):
            self.best_confirmed_candidate = self._clone_candidate(candidate)
        return candidate
    
    def _initialize_population(self):
        """Initialize the evolutionary-search population.

        Args:
            None.

        Returns:
            None.
        """
        logger.info("Initializing population: size=%d", self.population_size)

        initial_solutions, init_meta = build_initial_population(
            strategy=self.init_strategy,
            seed=self.init_seed,
            population_size=self.population_size,
            universe=self.solution_universe,
            num_blocks=self.num_blocks,
            num_steps=self.num_steps,
            k_pairs=self.K,
            task_name=self.task_name,
            profile=self.init_structure_profile,
            random_fraction=self.init_random_fraction,
            min_pairs_per_block=self.init_structure_min_pairs_per_block,
            random_tail_min=self.init_structure_tail_min,
            random_tail_max=self.init_structure_tail_max,
            pair_weights=self.init_pair_weights,
            importance_gamma=self.init_importance_gamma,
            importance_epsilon=self.init_importance_epsilon,
        )
        self.initialization_meta = init_meta
        self.reproducibility_meta['initialization'] = init_meta
        logger.info(
            "Initialization: strategy=%s seed=%s source_counts=%s",
            init_meta.get("strategy"),
            init_meta.get("seed"),
            init_meta.get("source_counts"),
        )
        for solution in initial_solutions:
            individual = Individual(solution)
            self.population.append(individual)
        
        logger.info("Initialized %d individuals.", len(self.population))
    
    def _run_environment_evaluation(self, eval_mode: str = 'quick'):
        """Run the environment runner for quick or formal rollout evaluation.

        Args:
            eval_mode (str): ``"quick"`` for inexpensive screening or
                ``"formal"`` for target-conditioned verification.

        Returns:
            Tuple[Dict[str, Any], Dict[str, float]]: Evaluation logs and timing
            diagnostics.
        """
        total_start = time.perf_counter()
        timing = {
            'temp_dir_seconds': 0.0,
            'env_cfg_prepare_seconds': 0.0,
            'env_runner_instantiate_seconds': 0.0,
            'env_runner_run_seconds': 0.0,
            'total_env_eval_seconds': 0.0,
        }
        try:
            # Keep environment artifacts in a temporary directory.
            import tempfile
            import contextlib
            temp_dir_start = time.perf_counter()
            with tempfile.TemporaryDirectory() as temp_dir:
                timing['temp_dir_seconds'] = time.perf_counter() - temp_dir_start
                # Prepare environment runner configuration.
                cfg_start = time.perf_counter()
                env_runner_cfg = self._build_env_runner_cfg(eval_mode)
                timing['env_cfg_prepare_seconds'] = time.perf_counter() - cfg_start

                # Instantiate environment runner while suppressing verbose output.
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        # Construct the runner.
                        instantiate_start = time.perf_counter()
                        env_runner = hydra.utils.instantiate(
                            env_runner_cfg,
                            output_dir=temp_dir
                        )
                        timing['env_runner_instantiate_seconds'] = time.perf_counter() - instantiate_start

                        # Run evaluation.
                        run_start = time.perf_counter()
                        log_data = env_runner.run(self.cache_wrapper)
                        timing['env_runner_run_seconds'] = time.perf_counter() - run_start

                timing['total_env_eval_seconds'] = time.perf_counter() - total_start
                return log_data, timing
                
        except Exception as e:
            logger.error("Environment evaluation failed: %s", e)
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            timing['total_env_eval_seconds'] = time.perf_counter() - total_start
            return {'test/mean_score': 0.0}, timing
    
    def _evaluate_solution_indices(
        self,
        solution_indices: Sequence[int],
        eval_mode: str,
        purpose: str,
        cache_mode: str = 'evo',
    ) -> Dict[str, Any]:
        """Evaluate one candidate schedule with quick or formal settings.

        Args:
            solution_indices (Sequence[int]): Flattened block–timestep positions
                selected by the schedule.
            eval_mode (str): ``"quick"`` or ``"formal"`` evaluation mode.
            purpose (str): Trace label describing why the evaluation is run.
            cache_mode (str): Cache mode label recorded in the evaluation
                payload.

        Returns:
            Dict[str, Any]: Fitness, success rate, cache statistics, timing, and
            selected block–timestep pairs.
        """
        total_start = time.perf_counter()
        timing = {
            'set_optimal_steps_seconds': 0.0,
            'env_eval_seconds': 0.0,
            'cache_stats_seconds': 0.0,
            'total_eval_seconds': 0.0,
        }
        try:
            self._ensure_local_policy_loaded()
            # Convert selected indices into (block, denoise_step) pairs.
            ij_pairs = [self.ij_pairs[idx] for idx in solution_indices]
            
            logger.debug("Evaluating pairs: %s... (total=%d)", ij_pairs[:5], len(ij_pairs))
            
            # Apply the selected EVO schedule.
            set_start = time.perf_counter()
            self.cache_wrapper.set_optimal_steps(ij_pairs)
            timing['set_optimal_steps_seconds'] = time.perf_counter() - set_start
            
            # Run environment evaluation.
            with torch.no_grad():
                env_start = time.perf_counter()
                eval_seed_for_torch = self.quick_eval_seed if eval_mode == 'quick' else self.formal_eval_seed
                _configure_torch_eval_seed(eval_seed_for_torch)
                eval_results, env_timing = self._run_environment_evaluation(eval_mode=eval_mode)
                timing['env_eval_seconds'] = time.perf_counter() - env_start
                success_rate = eval_results.get('test/mean_score', 0.0)
                
                # Extract evaluation score.
                cache_stats_start = time.perf_counter()
                cache_stats = self.cache_wrapper.get_cache_statistics()
                timing['cache_stats_seconds'] = time.perf_counter() - cache_stats_start
                hit_rate = cache_stats.get('hit_rate', 0.0)
                
                # Use task success score as fitness.
                fitness = success_rate
                timing['total_eval_seconds'] = time.perf_counter() - total_start
                
                logger.debug(
                    "Evaluation result: fitness=%.4f (score=%.4f, hit_rate=%.4f)",
                    fitness,
                    success_rate,
                    hit_rate,
                )
                
                return {
                    'fitness': fitness,
                    'success_rate': success_rate,
                    'hit_rate': hit_rate,
                    'cache_stats': cache_stats,
                    'timing': timing,
                    'env_timing': env_timing,
                    'eval_mode': eval_mode,
                    'purpose': purpose,
                    'cache_mode': cache_mode,
                    'solution_indices': list(solution_indices),
                    'ij_pairs': ij_pairs,
                }
                
        except Exception as e:
            logger.error("Solution evaluation failed: %s", str(e))
            logger.error("First five selected pairs: %s", [self.ij_pairs[idx] for idx in list(solution_indices)[:5]])
            import traceback
            logger.error("Traceback: %s", traceback.format_exc())
            timing['total_eval_seconds'] = time.perf_counter() - total_start
            return {
                'fitness': 0.0,
                'success_rate': 0.0,
                'hit_rate': 0.0,
                'cache_stats': {},
                'timing': timing,
                'env_timing': {},
                'error': str(e),
                'eval_mode': eval_mode,
                'purpose': purpose,
                'cache_mode': cache_mode,
                'solution_indices': list(solution_indices),
            }

    def _evaluate_fitness(self, individual: Individual):
        """Evaluate one individual with quick fitness settings.

        Args:
            individual (Individual): Candidate schedule to evaluate.

        Returns:
            Dict[str, Any]: Quick evaluation payload.
        """
        return self._evaluate_solution_indices(
            solution_indices=individual.solution,
            eval_mode='quick',
            purpose='population_fitness',
            cache_mode='evo',
        )
    
    def _evaluate_population(self):
        """Evaluate all unevaluated individuals in the current population.

        Args:
            None.

        Returns:
            None.
        """
        logger.debug("Evaluating population with %d individuals.", len(self.population))
        eval_start = time.perf_counter()

        signature_to_individuals: Dict[Tuple[int, ...], List[Individual]] = defaultdict(list)
        for individual in self.population:
            individual.solution = self._canonicalize_solution(individual.solution)
            signature_to_individuals[self._solution_signature(individual.solution)].append(individual)

        pending: List[Tuple[Tuple[int, ...], Individual]] = []
        cache_hits = 0
        in_memory_reuses = 0
        individual_trace_entries: List[Dict[str, Any]] = []
        for signature, individuals in signature_to_individuals.items():
            cached_fitness = self.fitness_cache.get(signature)
            if cached_fitness is not None:
                for individual in individuals:
                    individual.fitness = cached_fitness
                cache_hits += len(individuals)
                individual_trace_entries.append({
                    'generation': self._active_generation,
                    'phase': self._active_phase,
                    'source': 'fitness_cache',
                    'signature': list(signature),
                    'solution_indices': list(individuals[0].solution),
                    'num_individuals': len(individuals),
                    'fitness': cached_fitness,
                    'total_eval_seconds': 0.0,
                })
                continue

            known_fitness = next(
                (individual.fitness for individual in individuals if individual.fitness is not None),
                None,
            )
            if known_fitness is not None:
                self.fitness_cache[signature] = known_fitness
                for individual in individuals:
                    individual.fitness = known_fitness
                in_memory_reuses += len(individuals)
                individual_trace_entries.append({
                    'generation': self._active_generation,
                    'phase': self._active_phase,
                    'source': 'individual_reuse',
                    'signature': list(signature),
                    'solution_indices': list(individuals[0].solution),
                    'num_individuals': len(individuals),
                    'fitness': known_fitness,
                    'total_eval_seconds': 0.0,
                })
                continue

            pending.append((signature, individuals[0]))

        if cache_hits:
            logger.debug("Fitness-cache hits: %d individuals", cache_hits)

        if pending:
            if self.parallel_eval:
                solutions = [individual.solution for _, individual in pending]
                fitnesses = self.pool.map(_evaluate_solution_in_worker, solutions)
                for (signature, _), fitness in zip(pending, fitnesses):
                    self.fitness_cache[signature] = fitness
                    for individual in signature_to_individuals[signature]:
                        individual.fitness = fitness
                    individual_trace_entries.append({
                        'generation': self._active_generation,
                        'phase': self._active_phase,
                        'source': 'parallel_worker',
                        'signature': list(signature),
                        'solution_indices': list(signature_to_individuals[signature][0].solution),
                        'num_individuals': len(signature_to_individuals[signature]),
                        'fitness': fitness,
                        'total_eval_seconds': None,
                    })
            else:
                for signature, individual in tqdm(pending, desc="Evaluating", leave=False):
                    eval_payload = self._evaluate_fitness(individual)
                    fitness = eval_payload['fitness']
                    self.fitness_cache[signature] = fitness
                    for duplicate in signature_to_individuals[signature]:
                        duplicate.fitness = fitness
                    trace_record = {
                        'generation': self._active_generation,
                        'phase': self._active_phase,
                        'source': 'fresh_evaluation',
                        'signature': list(signature),
                        'solution_indices': list(individual.solution),
                        'num_individuals': len(signature_to_individuals[signature]),
                        'fitness': fitness,
                        'success_rate': eval_payload.get('success_rate'),
                        'hit_rate': eval_payload.get('hit_rate'),
                        'timing': eval_payload.get('timing', {}),
                        'env_timing': eval_payload.get('env_timing', {}),
                    }
                    if 'error' in eval_payload:
                        trace_record['error'] = eval_payload['error']
                    individual_trace_entries.append(trace_record)
        
        # Sort by fitness.
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update global best.
        if self.population and self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_individual = self.population[0].copy()
            self.best_found_generation = self._active_generation
            self.best_found_phase = self._active_phase
            self.best_update_history.append({
                'generation': self._active_generation,
                'phase': self._active_phase,
                'best_fitness': float(self.best_fitness),
            })
            
            best_ij_pairs = [self.ij_pairs[idx] for idx in self.population[0].solution]
            logger.info("\nNew best individual found: fitness=%.4f", self.best_fitness)
            logger.info("Best pairs preview (first 20): %s", best_ij_pairs[:20])
        
        # Log population statistics.
        fitnesses = [ind.fitness for ind in self.population]
        avg_fitness = np.mean(fitnesses)
        max_fitness = max(fitnesses)
        min_fitness = min(fitnesses)
        
        self.fitness_history.append(avg_fitness)
        generation_eval_seconds = time.perf_counter() - eval_start
        avg_seconds_per_eval = generation_eval_seconds / len(pending) if pending else 0.0
        self._last_population_eval_stats = {
            'generation': self._active_generation,
            'phase': self._active_phase,
            'population_size': len(self.population),
            'unique_signatures': len(signature_to_individuals),
            'duplicate_individuals': len(self.population) - len(signature_to_individuals),
            'new_evaluations': len(pending),
            'cache_hits': cache_hits,
            'in_memory_reuses': in_memory_reuses,
            'generation_eval_seconds': generation_eval_seconds,
            'avg_seconds_per_eval': avg_seconds_per_eval,
            'avg_fitness': float(avg_fitness),
            'max_fitness': float(max_fitness),
            'min_fitness': float(min_fitness),
        }

        for trace_record in individual_trace_entries:
            self._append_individual_trace(trace_record)
        
        logger.info("Population fitness: avg=%.4f, max=%.4f, min=%.4f", avg_fitness, max_fitness, min_fitness)
        logger.info(
            "Evaluation summary: phase=%s generation=%s new_evaluations=%d cache_hits=%d in_memory_reuses=%d "
            "generation_eval_seconds=%.2f avg_seconds_per_eval=%.2f",
            self._active_phase,
            self._active_generation,
            len(pending),
            cache_hits,
            in_memory_reuses,
            generation_eval_seconds,
            avg_seconds_per_eval,
        )
        
        return avg_fitness, max_fitness, min_fitness
    
    def _tournament_selection(self) -> Individual:
        """Select one parent by tournament selection.

        Args:
            None.

        Returns:
            Individual: Highest-fitness individual among sampled candidates.
        """
        tournament = random.sample(self.population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Recombine two fixed-budget schedules with set-level crossover.

        Args:
            parent1 (Individual): First parent schedule.
            parent2 (Individual): Second parent schedule.

        Returns:
            Tuple[Individual, Individual]: Two repaired child schedules.
        """
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        # The solution semantics are set-based, but we still want the original
        # uniform crossover's exploration behavior. Shuffle first so sorted
        # canonical order does not bias recombination.
        parent1_order = parent1.solution.copy()
        parent2_order = parent2.solution.copy()
        random.shuffle(parent1_order)
        random.shuffle(parent2_order)

        child1_solution = []
        child2_solution = []
        for idx1, idx2 in zip(parent1_order, parent2_order):
            if random.random() < 0.5:
                child1_solution.append(idx1)
                child2_solution.append(idx2)
            else:
                child1_solution.append(idx2)
                child2_solution.append(idx1)

        child1_solution = self._canonicalize_solution(child1_solution)
        child2_solution = self._canonicalize_solution(child2_solution)

        return Individual(child1_solution), Individual(child2_solution)
    
    def _mutation(self, individual: Individual):
        """Randomly replace selected positions with unselected lattice positions.

        Args:
            individual (Individual): Candidate schedule mutated in place.

        Returns:
            None.
        """
        if random.random() > self.mutation_rate:
            return

        current_set = set(individual.solution)
        remaining = [idx for idx in self.solution_universe if idx not in current_set]
        if remaining:
            max_replace = min(6, len(remaining), len(individual.solution))
            replace_count = random.randint(2, max_replace) if max_replace >= 2 else 1
            remove_indices = set(random.sample(individual.solution, replace_count))
            add_indices = random.sample(remaining, replace_count)
            new_solution = [idx for idx in individual.solution if idx not in remove_indices]
            new_solution.extend(add_indices)
            individual.solution = self._canonicalize_solution(new_solution)
            individual.fitness = None
    
    def _create_next_generation(self):
        """Create the next population from elites, crossover, and mutation.

        Args:
            None.

        Returns:
            None.
        """
        create_start = time.perf_counter()
        new_population = []
        stats = {
            'same_gen_rejects': 0,
            'retired_rejects': 0,
            'fallback_count': 0,
            'attempts_total': 0,
            'accepted_children': 0,
            'accepted_from_offspring': 0,
            'accepted_from_fallback': 0,
        }
        
        # Preserve elites.
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        elite_count = min(self.elite_size, len(self.population))
        elite_signatures = set()
        for i in range(elite_count):
            elite = self.population[i].copy()
            elite.solution = self._canonicalize_solution(elite.solution)
            elite.age += 1
            new_population.append(elite)
            elite_signatures.add(self._solution_signature(elite.solution))

        current_population_signatures = {
            self._solution_signature(self._canonicalize_solution(ind.solution))
            for ind in self.population
        }
        self.retired_signatures.update(current_population_signatures - elite_signatures)

        new_population_signatures = set(elite_signatures)

        while len(new_population) < self.population_size:
            attempts = 0
            appended = False
            while attempts < 256 and len(new_population) < self.population_size:
                stats['attempts_total'] += 1
                parent1 = self._tournament_selection()
                parent2 = self._tournament_selection()

                child1, child2 = self._crossover(parent1, parent2)
                self._mutation(child1)
                self._mutation(child2)

                appended_child = False
                for child in (child1, child2):
                    accepted, reason = self._accept_child(child, new_population, new_population_signatures)
                    if accepted:
                        appended_child = True
                        appended = True
                        stats['accepted_children'] += 1
                        stats['accepted_from_offspring'] += 1
                    elif reason == 'same_gen_duplicate':
                        stats['same_gen_rejects'] += 1
                    elif reason == 'retired_duplicate':
                        stats['retired_rejects'] += 1
                    if len(new_population) >= self.population_size:
                        break
                if appended_child:
                    break
                attempts += 1

            if appended:
                continue

            fallback_solution = self._sample_unique_solution(
                blocked_signatures=self.retired_signatures | new_population_signatures,
            )
            fallback_individual = Individual(fallback_solution, fitness=None)
            accepted, reason = self._accept_child(fallback_individual, new_population, new_population_signatures)
            if accepted:
                stats['fallback_count'] += 1
                stats['accepted_children'] += 1
                stats['accepted_from_fallback'] += 1
            elif reason == 'same_gen_duplicate':
                stats['same_gen_rejects'] += 1
            elif reason == 'retired_duplicate':
                stats['retired_rejects'] += 1
        
        # Trim any accidental overflow.
        if len(new_population) > self.population_size:
            new_population = new_population[:self.population_size]
        
        self.population = new_population
        stats['create_next_generation_seconds'] = time.perf_counter() - create_start
        accepted_from_offspring = stats['accepted_from_offspring']
        stats['avg_attempts_per_accepted_child'] = (
            stats['attempts_total'] / accepted_from_offspring
            if accepted_from_offspring > 0
            else None
        )
        self._last_generation_build_stats = stats
    
    def _calculate_diversity(self) -> float:
        """Compute average pairwise diversity among schedule sets.

        Args:
            None.

        Returns:
            float: Average number of non-overlapping selected positions between
            population members.
        """
        if len(self.population) <= 1:
            return 0.0
        
        # Compute average normalized Hamming distance.
        total_distance = 0
        count = 0
        
        for i in range(len(self.population)):
            for j in range(i + 1, len(self.population)):
                # Intersection size counts shared selected pairs.
                intersection = len(set(self.population[i].solution) & set(self.population[j].solution))
                # Distance is K minus the shared selected pairs.
                distance = self.K - intersection
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0

    def _recent_avg_improvement(self) -> Optional[float]:
        """Estimate recent improvement in average population fitness.

        Args:
            None.

        Returns:
            Optional[float]: Difference between recent and previous fitness
            windows, or ``None`` before enough history exists.
        """
        window = self.early_stop_avg_window
        if len(self.fitness_history) < (2 * window):
            return None
        recent_mean = float(np.mean(self.fitness_history[-window:]))
        previous_mean = float(np.mean(self.fitness_history[-2 * window:-window]))
        return recent_mean - previous_mean

    def _candidate_to_json(self, candidate: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Convert a candidate payload to JSON-safe schedule metadata.

        Args:
            candidate (Optional[Dict[str, Any]]): Candidate payload to serialize.

        Returns:
            Optional[Dict[str, Any]]: JSON-ready candidate summary.
        """
        if candidate is None:
            return None
        individual = candidate.get('individual')
        formal_fitness = float(candidate['formal_fitness'])
        return {
            'solution_indices': list(individual.solution) if individual is not None else [],
            'ij_pairs': [self.ij_pairs[idx] for idx in individual.solution] if individual is not None else [],
            'signature': list(candidate['signature']),
            'quick_fitness': float(candidate['quick_fitness']),
            'formal_fitness': formal_fitness,
            'confirm_type': candidate['confirm_type'],
            'generation': candidate['generation'],
            'phase': candidate['phase'],
            'confirm_trigger_target': self.confirm_trigger_target,
            'formal_accept_target': self.formal_accept_target,
            'meets_formal_requirement': self._meets_formal_requirement(formal_fitness),
        }

    def _meets_formal_requirement(self, formal_fitness: float) -> bool:
        """Check whether a formal score satisfies the acceptance target.

        Args:
            formal_fitness (float): Independent formal rollout score.

        Returns:
            bool: ``True`` if the score is above the formal acceptance target.
        """
        if self.formal_accept_target is None:
            return False
        return float(formal_fitness) >= float(self.formal_accept_target)

    def _build_pending_formal_confirm_candidates(self) -> List[Dict[str, Any]]:
        """Collect quick-passing candidates that need formal verification.

        Args:
            None.

        Returns:
            List[Dict[str, Any]]: Candidate payloads whose quick fitness reaches
            the target trigger and have not yet been confirmed.
        """
        if self.confirm_trigger_target is None:
            return []

        candidates: List[Dict[str, Any]] = []
        seen_signatures = set()

        def maybe_append(
            individual: Optional[Individual],
            quick_fitness: Optional[float],
            generation: Optional[int],
            phase: Optional[str],
            source: str,
            rank: Optional[int],
        ) -> None:
            """Append one candidate if it passes quick-screening criteria.

            Args:
                individual (Optional[Individual]): Candidate schedule.
                quick_fitness (Optional[float]): Quick rollout fitness.
                generation (Optional[int]): Generation where it appeared.
                phase (Optional[str]): Search phase label.
                source (str): Candidate source label.
                rank (Optional[int]): Rank within the source list.

            Returns:
                None.
            """
            if individual is None or quick_fitness is None:
                return
            quick_fitness = float(quick_fitness)
            if quick_fitness < float(self.confirm_trigger_target):
                return
            signature = self._solution_signature(individual.solution)
            if signature in seen_signatures or signature in self.confirmed_signatures:
                return
            seen_signatures.add(signature)
            candidates.append(
                {
                    'individual': individual.copy(),
                    'signature': signature,
                    'quick_fitness': quick_fitness,
                    'generation': generation,
                    'phase': phase,
                    'source': source,
                    'rank': rank,
                }
            )

        maybe_append(
            self.best_individual,
            self.best_fitness if self.best_individual is not None else None,
            self.best_found_generation,
            self.best_found_phase,
            source='global_best',
            rank=0,
        )
        for rank, individual in enumerate(self.population):
            maybe_append(
                individual,
                individual.fitness,
                self._active_generation,
                self._active_phase,
                source='population_rank',
                rank=rank,
            )
        return candidates

    def _update_fresh_eval_counters(self):
        """Update counters for newly evaluated candidate schedules.

        Args:
            None.

        Returns:
            None.
        """
        new_evaluations = int(self._last_population_eval_stats.get('new_evaluations', 0))
        self.total_fresh_evaluations += new_evaluations
        self.fresh_evals_since_meaningful_best += new_evaluations

    def _check_meaningful_best_update(self, previous_best: float) -> bool:
        """Check whether the quick best improved enough to matter.

        Args:
            previous_best (float): Best quick fitness before the latest
                population evaluation.

        Returns:
            bool: ``True`` when the new best clears the meaningful-improvement
            threshold.
        """
        if self.best_individual is None or self.best_fitness <= previous_best:
            return False
        improvement_from_last_meaningful = (
            float('inf')
            if not np.isfinite(self.last_meaningful_best_fitness)
            else self.best_fitness - self.last_meaningful_best_fitness
        )
        meaningful = (
            not np.isfinite(self.last_meaningful_best_fitness)
            or improvement_from_last_meaningful >= self.meaningful_improvement_eps
        )
        trace_payload = {
            'event': 'best_update',
            'generation': self._active_generation,
            'phase': self._active_phase,
            'quick_best_fitness': float(self.best_fitness),
            'previous_quick_best_fitness': float(previous_best),
            'improvement_from_last_meaningful': improvement_from_last_meaningful,
            'meaningful': meaningful,
            'fresh_evals_total': self.total_fresh_evaluations,
            'fresh_evals_since_meaningful_best_before_reset': self.fresh_evals_since_meaningful_best,
            'solution_indices': list(self.best_individual.solution),
        }
        if meaningful:
            self.last_meaningful_best_fitness = float(self.best_fitness)
            self.last_meaningful_best_generation = self._active_generation
            self.last_meaningful_best_phase = self._active_phase
            self.fresh_evals_since_meaningful_best = 0
        self._append_early_stop_trace(trace_payload)
        return meaningful

    def _should_trigger_formal_confirm(
        self,
        meaningful_best_updated: bool,
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """Decide whether a quick-passing schedule should receive formal rollout.

        Args:
            meaningful_best_updated (bool): Whether the current generation
                produced a meaningful quick-fitness improvement.

        Returns:
            Tuple[bool, str, Optional[Dict[str, Any]]]: Trigger flag, reason, and
            candidate metadata for formal confirmation.
        """
        pending_candidates = self._build_pending_formal_confirm_candidates()
        if not pending_candidates:
            return False, 'no_pending_quick_candidates', None
        if (
            self.last_confirm_fresh_eval_total is not None
            and self.total_fresh_evaluations - self.last_confirm_fresh_eval_total
            < self.formal_confirm_cooldown_fresh_evals
        ):
            return False, 'confirm_cooldown', pending_candidates[0]
        if meaningful_best_updated:
            return True, 'meaningful_best_pending_candidate', pending_candidates[0]
        return True, 'pending_quick_candidate_retry', pending_candidates[0]

    def _evaluate_formal_candidate(
        self,
        candidate_meta: Dict[str, Any],
        purpose: str,
        trace_event: str,
        history_tag: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run independent formal verification for one candidate schedule.

        Args:
            candidate_meta (Dict[str, Any]): Quick-passing candidate metadata.
            purpose (str): Evaluation-purpose label.
            trace_event (str): Event name written to the early-stop trace.
            history_tag (str): History label for reporting where the formal
                check occurred.

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Formal-confirmation record and
            candidate payload.
        """
        individual = candidate_meta['individual']
        quick_fitness = float(candidate_meta['quick_fitness'])
        logger.info(
            "Running formal confirmation: generation=%s candidate_source=%s candidate_rank=%s quick=%.4f "
            "global_best=%.4f confirm_trigger_target=%.4f formal_accept_target=%.4f",
            self._active_generation,
            candidate_meta['source'],
            candidate_meta['rank'],
            quick_fitness,
            self.best_fitness,
            self.confirm_trigger_target,
            self.formal_accept_target,
        )
        eval_payload = self._evaluate_solution_indices(
            solution_indices=individual.solution,
            eval_mode='formal',
            purpose=purpose,
            cache_mode='evo',
        )
        formal_fitness = float(eval_payload['fitness'])
        confirm_type = 'formal_accept_rejected'
        if self.confirm_trigger_target is not None and formal_fitness >= self.confirm_trigger_target:
            confirm_type = 'baseline_met'
        elif self._meets_formal_requirement(formal_fitness):
            confirm_type = 'formal_accept_met'

        candidate = self._build_candidate_payload(
            individual=individual,
            quick_fitness=quick_fitness,
            formal_fitness=formal_fitness,
            confirm_type=confirm_type,
            generation=candidate_meta['generation'],
            phase=candidate_meta['phase'],
        )
        self._update_best_formal_candidate(candidate)

        confirm_record = {
            'generation': candidate_meta['generation'],
            'phase': candidate_meta['phase'],
            'signature': list(candidate['signature']),
            'solution_indices': list(individual.solution),
            'ij_pairs': [self.ij_pairs[idx] for idx in individual.solution],
            'quick_fitness': quick_fitness,
            'formal_fitness': formal_fitness,
            'confirm_type': confirm_type,
            'confirm_trigger_target': self.confirm_trigger_target,
            'formal_accept_target': self.formal_accept_target,
            'meets_formal_requirement': self._meets_formal_requirement(formal_fitness),
            'hard_target': self.hard_target,
            'soft_target': self.soft_target,
            'fresh_evals_total': self.total_fresh_evaluations,
            'fresh_evals_since_meaningful_best': self.fresh_evals_since_meaningful_best,
            'candidate_source': candidate_meta['source'],
            'candidate_rank': candidate_meta['rank'],
            'history_tag': history_tag,
            'timing': eval_payload.get('timing', {}),
            'env_timing': eval_payload.get('env_timing', {}),
            'cache_stats': eval_payload.get('cache_stats', {}),
        }
        if 'error' in eval_payload:
            confirm_record['error'] = eval_payload['error']

        self.formal_confirm_history.append(confirm_record)
        self.confirmed_signatures[candidate['signature']] = confirm_record
        self._append_early_stop_trace(
            {
                'event': trace_event,
                **confirm_record,
            }
        )
        return confirm_record, candidate

    def _maybe_run_formal_confirm(self, meaningful_best_updated: bool) -> Dict[str, Any]:
        """Optionally run target-conditioned formal confirmation.

        Args:
            meaningful_best_updated (bool): Whether the latest quick best update
                was meaningful.

        Returns:
            Dict[str, Any]: Confirmation state including trigger reason and stop
            decision.
        """
        should_confirm, reason, candidate_meta = self._should_trigger_formal_confirm(meaningful_best_updated)
        confirm_state = {
            'triggered': False,
            'trigger_reason': reason,
            'candidate': None,
            'result': None,
            'stop': False,
        }
        if not should_confirm:
            return confirm_state

        if candidate_meta is None:
            return confirm_state

        confirm_state['candidate'] = {
            'signature': list(candidate_meta['signature']),
            'quick_fitness': float(candidate_meta['quick_fitness']),
            'candidate_source': candidate_meta['source'],
            'candidate_rank': candidate_meta['rank'],
        }
        confirm_record, candidate = self._evaluate_formal_candidate(
            candidate_meta=candidate_meta,
            purpose='formal_confirm_candidate',
            trace_event='formal_confirm',
            history_tag='optimize_loop',
        )
        self.last_confirm_fresh_eval_total = self.total_fresh_evaluations

        confirm_state['triggered'] = True
        confirm_state['result'] = confirm_record
        if confirm_record['meets_formal_requirement']:
            candidate = self._register_confirmed_candidate(candidate)
            self._set_final_selected_candidate(
                candidate,
                reason='formal_accept_target_met',
                details={
                    'confirm_trigger_target': self.confirm_trigger_target,
                    'formal_accept_target': self.formal_accept_target,
                    'confirm_record': confirm_record,
                },
            )
            confirm_state['stop'] = True
            return confirm_state

        return confirm_state

    def _finalize_search_best_with_formal_evaluation(self) -> Optional[Dict[str, Any]]:
        """Run final formal checks before exporting an offline schedule.

        Args:
            None.

        Returns:
            Optional[Dict[str, Any]]: Formal evaluation payload for the final
            search-best fallback, or ``None`` when a pending candidate is
            selected directly.
        """
        if self.best_individual is None:
            return None

        pending_candidates = self._build_pending_formal_confirm_candidates()
        if pending_candidates:
            logger.info(
                "Running final formal checks for pending quick candidates: count=%d",
                len(pending_candidates),
            )
            for candidate_meta in pending_candidates:
                confirm_record, candidate = self._evaluate_formal_candidate(
                    candidate_meta=candidate_meta,
                    purpose='final_pending_formal_confirm',
                    trace_event='final_pending_formal_confirm',
                    history_tag='final_pending_confirm',
                )
                if confirm_record['meets_formal_requirement']:
                    self._register_confirmed_candidate(candidate)
                    self.final_selected_candidate = self._clone_candidate(candidate)
                    self._append_early_stop_trace(
                        {
                            'event': 'final_pending_formal_confirm_selected',
                            'pending_candidate_count': len(pending_candidates),
                            'selected_candidate': self._candidate_to_json(candidate),
                        }
                    )
                    return None

        search_best_signature = self._solution_signature(self.best_individual.solution)
        existing_search_best_record = self.confirmed_signatures.get(search_best_signature)
        if existing_search_best_record is not None:
            formal_fitness = float(existing_search_best_record['formal_fitness'])
            meets_requirement = bool(existing_search_best_record['meets_formal_requirement'])
            confirm_type = str(existing_search_best_record['confirm_type'])
            candidate = self._build_candidate_payload(
                individual=self.best_individual,
                quick_fitness=float(existing_search_best_record['quick_fitness']),
                formal_fitness=formal_fitness,
                confirm_type=confirm_type,
                generation=self.best_found_generation,
                phase=self.best_found_phase,
            )
            eval_payload = {
                'timing': existing_search_best_record.get('timing', {}),
                'env_timing': existing_search_best_record.get('env_timing', {}),
                'cache_stats': existing_search_best_record.get('cache_stats', {}),
            }
            self.final_search_best_formal_evaluation = {
                'generation': self.best_found_generation,
                'phase': self.best_found_phase,
                'signature': list(candidate['signature']),
                'solution_indices': list(self.best_individual.solution),
                'ij_pairs': [self.ij_pairs[idx] for idx in self.best_individual.solution],
                'quick_fitness': float(existing_search_best_record['quick_fitness']),
                'formal_fitness': formal_fitness,
                'confirm_type': confirm_type,
                'confirm_trigger_target': self.confirm_trigger_target,
                'formal_accept_target': self.formal_accept_target,
                'meets_formal_requirement': meets_requirement,
                'timing': eval_payload.get('timing', {}),
                'env_timing': eval_payload.get('env_timing', {}),
                'cache_stats': eval_payload.get('cache_stats', {}),
                'reused_existing_formal_record': True,
            }
            if 'error' in existing_search_best_record:
                self.final_search_best_formal_evaluation['error'] = existing_search_best_record['error']
        else:
            logger.info(
                "No pending candidate met the formal target; evaluating search-best fallback: "
                "quick_best=%.4f formal_accept_target=%s",
                self.best_fitness,
                f"{self.formal_accept_target:.4f}" if self.formal_accept_target is not None else "None",
            )
            eval_payload = self._evaluate_solution_indices(
                solution_indices=self.best_individual.solution,
                eval_mode='formal',
                purpose='final_search_best_formal_fallback',
                cache_mode='evo',
            )
            formal_fitness = float(eval_payload['fitness'])
            meets_requirement = self._meets_formal_requirement(formal_fitness)
            confirm_type = 'final_search_best_formal_rejected'
            if self.confirm_trigger_target is not None and formal_fitness >= self.confirm_trigger_target:
                confirm_type = 'final_search_best_baseline_met'
            elif meets_requirement:
                confirm_type = 'final_search_best_formal_accept_met'

            candidate = self._build_candidate_payload(
                individual=self.best_individual,
                quick_fitness=float(self.best_fitness),
                formal_fitness=formal_fitness,
                confirm_type=confirm_type,
                generation=self.best_found_generation,
                phase=self.best_found_phase,
            )
            self._update_best_formal_candidate(candidate)
            if meets_requirement:
                self._register_confirmed_candidate(candidate)

            self.final_search_best_formal_evaluation = {
                'generation': self.best_found_generation,
                'phase': self.best_found_phase,
                'signature': list(candidate['signature']),
                'solution_indices': list(self.best_individual.solution),
                'ij_pairs': [self.ij_pairs[idx] for idx in self.best_individual.solution],
                'quick_fitness': float(self.best_fitness),
                'formal_fitness': formal_fitness,
                'confirm_type': confirm_type,
                'confirm_trigger_target': self.confirm_trigger_target,
                'formal_accept_target': self.formal_accept_target,
                'meets_formal_requirement': meets_requirement,
                'timing': eval_payload.get('timing', {}),
                'env_timing': eval_payload.get('env_timing', {}),
                'cache_stats': eval_payload.get('cache_stats', {}),
            }
            if 'error' in eval_payload:
                self.final_search_best_formal_evaluation['error'] = eval_payload['error']

        self._append_early_stop_trace(
            {
                'event': 'final_search_best_formal_evaluation',
                **self.final_search_best_formal_evaluation,
            }
        )
        selected_candidate = self.best_confirmed_candidate or self.best_formal_candidate or candidate
        self.final_selected_candidate = self._clone_candidate(selected_candidate)
        self._append_early_stop_trace(
            {
                'event': 'final_candidate_fallback_selected',
                'selection_reason': (
                    'best_confirmed_candidate'
                    if self.best_confirmed_candidate is not None
                    else 'best_formal_candidate'
                ),
                'selected_candidate': self._candidate_to_json(self.final_selected_candidate),
            }
        )
        return self.final_search_best_formal_evaluation
    
    def optimize(self):
        """Run rollout-driven evolutionary optimization.

        Args:
            None.

        Returns:
            Tuple[List[int], float]: Selected flattened block–timestep schedule
            and its quick rollout fitness.
        """
        optimize_start = time.perf_counter()
        logger.info(
            "Starting optimization: population=%d generations=%d selected_pairs=%d",
            self.population_size,
            self.max_generations,
            self.K,
        )
        self._ensure_task_baseline()
        logger.info(
            "Formal targets: baseline=%.4f confirm_trigger_target=%.4f formal_accept_target=%.4f "
            "improvement_eps=%.4f confirm_cooldown=%d max_fresh_evals=%s",
            self.formal_baseline_score,
            self.confirm_trigger_target,
            self.formal_accept_target,
            self.meaningful_improvement_eps,
            self.formal_confirm_cooldown_fresh_evals,
            self.max_fresh_evals if self.max_fresh_evals is not None else "None",
        )
        if self.parallel_eval:
            logger.info("Parallel devices: %s", self.device_list)
            pool_start = time.perf_counter()
            self.pool = _NonDaemonPool(
                processes=len(self.device_list),
                initializer=_init_parallel_worker,
                initargs=(
                    self.checkpoint_path,
                    self.num_blocks,
                    self.num_steps,
                    self.device_list,
                    self.experiment_seed,
                    self.quick_eval_seed,
                    self.deterministic_torch,
                )
            )
            self.timing_summary['parallel_pool_init_seconds'] = time.perf_counter() - pool_start
        
        # Evaluate initial population.
        logger.info("Evaluating initial population...")
        self._active_generation = -1
        self._active_phase = 'initial'
        initial_eval_start = time.perf_counter()
        self._evaluate_population()
        self.timing_summary['initial_population_eval_seconds'] = time.perf_counter() - initial_eval_start
        self._update_fresh_eval_counters()
        initial_meaningful_best = self._check_meaningful_best_update(previous_best=float('-inf'))
        initial_confirm_state = self._maybe_run_formal_confirm(initial_meaningful_best)
        initial_summary = dict(self._last_population_eval_stats)
        initial_summary['diversity'] = None
        initial_summary['hard_target'] = self.hard_target
        initial_summary['soft_target'] = self.soft_target
        initial_summary['confirm_trigger_target'] = self.confirm_trigger_target
        initial_summary['formal_accept_target'] = self.formal_accept_target
        initial_summary['fresh_evals_total'] = self.total_fresh_evaluations
        initial_summary['fresh_evals_since_meaningful_best'] = self.fresh_evals_since_meaningful_best
        initial_summary['formal_confirm_state'] = initial_confirm_state
        initial_summary['best_confirmed_candidate'] = self._candidate_to_json(self.best_confirmed_candidate)
        initial_summary['best_formal_candidate'] = self._candidate_to_json(self.best_formal_candidate)
        self.generation_summaries.append(initial_summary)
        if initial_confirm_state.get('stop'):
            logger.info("Initial formal confirmation satisfied the target; stopping search.")
            self.timing_summary['optimize_total_seconds'] = time.perf_counter() - optimize_start
            output_individual = self._get_output_individual()
            return output_individual.solution if output_individual else [], self._get_output_quick_fitness()

        evolution_loop_start = time.perf_counter()
        with tqdm(total=self.max_generations, desc="Evolution Progress", unit="gen") as pbar:
            for generation in range(self.max_generations):
                self._active_generation = generation
                self._active_phase = 'evolution'
                # Create next generation.
                generation_total_start = time.perf_counter()
                self._create_next_generation()

                # Evaluate the generation.
                previous_best = self.best_fitness
                avg_fitness, max_fitness, min_fitness = self._evaluate_population()
                self._update_fresh_eval_counters()
                meaningful_best_updated = self._check_meaningful_best_update(previous_best=previous_best)
                formal_confirm_state = self._maybe_run_formal_confirm(meaningful_best_updated)

                # Track diversity.
                diversity_start = time.perf_counter()
                diversity = self._calculate_diversity()
                diversity_calc_seconds = time.perf_counter() - diversity_start
                self.diversity_history.append(diversity)
                diversity_ratio = (diversity / float(self.K)) if self.K > 0 else 0.0

                # Update progress bar.
                pbar.set_postfix({
                    'Best': f'{self.best_fitness:.4f}',
                    'Avg': f'{avg_fitness:.4f}',
                    'Max': f'{max_fitness:.4f}',
                    'FreshNoImp': self.fresh_evals_since_meaningful_best,
                })
                pbar.update(1)

                generation_summary = {
                    'generation': generation,
                    'phase': 'evolution',
                    'best_fitness': float(self.best_fitness),
                    'avg_fitness': float(avg_fitness),
                    'max_fitness': float(max_fitness),
                    'min_fitness': float(min_fitness),
                    'diversity': float(diversity),
                    'diversity_ratio': float(diversity_ratio),
                    'diversity_calc_seconds': diversity_calc_seconds,
                    'hard_target': self.hard_target,
                    'soft_target': self.soft_target,
                    'confirm_trigger_target': self.confirm_trigger_target,
                    'formal_accept_target': self.formal_accept_target,
                    'fresh_evals_total': self.total_fresh_evaluations,
                    'fresh_evals_since_meaningful_best': self.fresh_evals_since_meaningful_best,
                    'meaningful_best_updated': meaningful_best_updated,
                    'formal_confirm_state': formal_confirm_state,
                    'best_confirmed_candidate': self._candidate_to_json(self.best_confirmed_candidate),
                    'best_formal_candidate': self._candidate_to_json(self.best_formal_candidate),
                    'generation_total_seconds': time.perf_counter() - generation_total_start,
                }
                generation_summary.update(self._last_generation_build_stats)
                generation_summary.update(self._last_population_eval_stats)
                self.generation_summaries.append(generation_summary)
                logger.info(
                    "Generation summary: generation=%d new_evaluations=%d generation_eval_seconds=%.2f "
                    "avg_seconds_per_eval=%.2f same_gen_rejects=%d retired_rejects=%d "
                    "fallback_count=%d avg_attempts_per_accepted_child=%s generation_total_seconds=%.2f",
                    generation,
                    self._last_population_eval_stats.get('new_evaluations', 0),
                    self._last_population_eval_stats.get('generation_eval_seconds', 0.0),
                    self._last_population_eval_stats.get('avg_seconds_per_eval', 0.0),
                    self._last_generation_build_stats.get('same_gen_rejects', 0),
                    self._last_generation_build_stats.get('retired_rejects', 0),
                    self._last_generation_build_stats.get('fallback_count', 0),
                    (
                        f"{self._last_generation_build_stats['avg_attempts_per_accepted_child']:.2f}"
                        if self._last_generation_build_stats.get('avg_attempts_per_accepted_child') is not None
                        else "N/A"
                    ),
                    generation_summary['generation_total_seconds'],
                )

                if formal_confirm_state.get('stop'):
                    logger.info("Formal confirmation satisfied the target; stopping search.")
                    pbar.set_description("Formal Accept Stop")
                    break
            else:
                if self.stop_reason is None:
                    self.stop_reason = 'max_generations_reached'
                    self.stop_context = {
                        'max_generations': self.max_generations,
                        'best_confirmed_candidate': self._candidate_to_json(self.best_confirmed_candidate),
                    }
        self.timing_summary['evolution_loop_seconds'] = time.perf_counter() - evolution_loop_start

        if self.final_selected_candidate is None:
            self._finalize_search_best_with_formal_evaluation()

        # Log final best schedule.
        if self.best_individual is not None:
            final_best_ij_pairs = [self.ij_pairs[idx] for idx in self.best_individual.solution]
            logger.info(f"\n{'='*60}")
            logger.info("Optimization result:")
            logger.info(f"{'='*60}")
            logger.info("Best quick fitness: %.4f", self.best_fitness)
            logger.info("Selected pair count: %d", len(final_best_ij_pairs))
            logger.info("Selected pairs: %s", final_best_ij_pairs)
            
            # Group selected denoising steps by block.
            block_groups = defaultdict(list)
            for i, j in final_best_ij_pairs:
                block_groups[i].append(j)
            
            logger.info("\nSelected steps by block:")
            for block_idx in sorted(block_groups.keys()):
                steps = sorted(block_groups[block_idx])
                logger.info("Block %s: steps %s (count=%d)", block_idx, steps, len(steps))
        output_candidate = self._get_selected_candidate_payload()
        if output_candidate is not None:
            logger.info(
                "Selected candidate: type=%s quick=%.4f formal=%.4f stop_reason=%s",
                output_candidate['confirm_type'],
                output_candidate['quick_fitness'],
                output_candidate['formal_fitness'],
                self.stop_reason,
            )
        if self.final_search_best_formal_evaluation is not None:
            logger.info(
                "Search-best formal result: quick=%.4f formal=%.4f meets_requirement=%s",
                self.final_search_best_formal_evaluation['quick_fitness'],
                self.final_search_best_formal_evaluation['formal_fitness'],
                self.final_search_best_formal_evaluation['meets_formal_requirement'],
            )
        self.timing_summary['optimize_total_seconds'] = time.perf_counter() - optimize_start
        
        output_individual = self._get_output_individual()
        return output_individual.solution if output_individual else [], self._get_output_quick_fitness()
    
    def get_best_ij_pairs(self):
        """Return the selected ``(block, denoise_step)`` refresh positions.

        Args:
            None.

        Returns:
            List[Tuple[int, int]]: Offline schedule represented as block-step
            pairs.
        """
        output_individual = self._get_output_individual()
        if output_individual is None:
            return []
        return [self.ij_pairs[idx] for idx in output_individual.solution]
    
    def save_results(self, output_path: str, extra_metadata: Optional[Dict[str, Any]] = None):
        """Save EVO search results and reproducibility metadata to JSON.

        Args:
            output_path (str): Destination JSON path.
            extra_metadata (Optional[Dict[str, Any]]): Additional metadata to
                merge into the output payload.

        Returns:
            None.
        """
        output_individual = self._get_output_individual()
        results = {
            'best_solution_indices': output_individual.solution if output_individual else [],
            'best_ij_pairs': self.get_best_ij_pairs(),
            'best_fitness': self._get_output_quick_fitness(),
            'search_best_solution_indices': self.best_individual.solution if self.best_individual else [],
            'search_best_ij_pairs': [self.ij_pairs[idx] for idx in self.best_individual.solution] if self.best_individual else [],
            'search_best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'diversity_history': self.diversity_history,
            'generation_summaries': self.generation_summaries,
            'timing_summary': self.timing_summary,
            'best_found_generation': self.best_found_generation,
            'best_found_phase': self.best_found_phase,
            'best_update_history': self.best_update_history,
            'individual_trace_path': self.individual_trace_path,
            'early_stop_trace_path': self.early_stop_trace_path,
            'stop_reason': self.stop_reason,
            'stop_context': self.stop_context,
            'total_fresh_evaluations': self.total_fresh_evaluations,
            'fresh_evals_since_meaningful_best': self.fresh_evals_since_meaningful_best,
            'last_meaningful_best_fitness': self.last_meaningful_best_fitness,
            'last_meaningful_best_generation': self.last_meaningful_best_generation,
            'last_meaningful_best_phase': self.last_meaningful_best_phase,
            'baseline_registry_path': self.baseline_registry_path,
            'baseline_entry': self.baseline_entry,
            'formal_baseline_score': self.formal_baseline_score,
            'confirm_trigger_target': self.confirm_trigger_target,
            'formal_accept_target': self.formal_accept_target,
            'formal_accept_drop': self.formal_accept_drop,
            'hard_target': self.hard_target,
            'soft_target': self.soft_target,
            'formal_confirm_history': self.formal_confirm_history,
            'best_confirmed_candidate': self._candidate_to_json(self.best_confirmed_candidate),
            'best_formal_candidate': self._candidate_to_json(self.best_formal_candidate),
            'final_selected_candidate': self._candidate_to_json(self.final_selected_candidate),
            'final_search_best_formal_evaluation': self.final_search_best_formal_evaluation,
            'reproducibility': self.reproducibility_meta,
            'parameters': {
                'task_name': self.task_name,
                'task_label': self.task_label,
                'K': self.K,
                'population_size': self.population_size,
                'elite_size': self.elite_size,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'tournament_size': self.tournament_size,
                'max_generations': self.max_generations,
                'search_space_mode': 'full_global',
                'search_space_size': len(self.solution_universe),
                'full_search_space_size': self.N,
                'baseline_allowed_drop': self.baseline_allowed_drop,
                'baseline_allowed_drop_ratio': self.baseline_allowed_drop_ratio,
                'formal_accept_drop': self.formal_accept_drop,
                'meaningful_improvement_eps': self.meaningful_improvement_eps,
                'fresh_eval_patience_after_confirm': self.fresh_eval_patience_after_confirm,
                'formal_confirm_cooldown_fresh_evals': self.formal_confirm_cooldown_fresh_evals,
                'max_fresh_evals': self.max_fresh_evals,
                'experiment_seed': self.experiment_seed,
                'search_seed': self.experiment_seed,
                'eval_seed': self.eval_seed,
                'deterministic_torch': self.deterministic_torch,
                'quick_eval_seed': self.quick_eval_seed,
                'formal_eval_seed': self.formal_eval_seed,
                'init_strategy': self.init_strategy,
                'init_seed': self.init_seed,
                'init_random_fraction': self.init_random_fraction,
                'init_structure_profile': self.init_structure_profile,
                'init_structure_min_pairs_per_block': self.init_structure_min_pairs_per_block,
                'init_structure_tail_min': self.init_structure_tail_min,
                'init_structure_tail_max': self.init_structure_tail_max,
                'init_importance_gamma': self.init_importance_gamma,
                'init_importance_epsilon': self.init_importance_epsilon,
            }
        }
        if extra_metadata:
            results.update(extra_metadata)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(_make_json_safe(results), f, indent=2, ensure_ascii=False)
        
        logger.info("Saved search results to: %s", output_path)
    
    def cleanup(self):
        """Release cache wrappers and worker pools.

        Args:
            None.

        Returns:
            None.
        """
        if self.cache_wrapper:
            self.cache_wrapper.cleanup()
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None


@click.command(help="Run rollout-driven evolutionary search for an EVO offline cache schedule.")
@click.option('-c', '--checkpoint', default=None, help='Pretrained Diffusion Policy checkpoint path. Required unless --task is given.')
@click.option('--task', default=None, help=f'Benchmark task used to resolve the default checkpoint. Available: {", ".join(get_all_available_tasks())}.')
@click.option('-o', '--output_dir', required=True, help='Directory for search traces, schedules, and JSON summaries.')
@click.option('-k', '--k_pairs', default=192, type=int, help='Fixed refresh budget: number of selected block–timestep positions.')
@click.option('-p', '--population_size', default=50, type=int, help='Number of candidate schedules in each generation.')
@click.option('-e', '--elite_size', default=10, type=int, help='Number of top schedules retained between generations.')
@click.option('-m', '--mutation_rate', default=0.1, type=float, help='Probability of mutating a child schedule.')
@click.option('-x', '--crossover_rate', default=0.8, type=float, help='Probability of applying set-level schedule crossover.')
@click.option('-t', '--tournament_size', default=3, type=int, help='Number of candidates sampled for tournament parent selection.')
@click.option('--num_blocks', default=24, type=int, help='Number of cacheable residual branches in the transformer decoder.')
@click.option('--num_steps', default=100, type=int, help='Number of denoising steps in the diffusion sampler.')
@click.option('--max_generations', default=100, type=int, help='Maximum number of evolutionary generations.')
@click.option('-d', '--device', default='cuda:0', help='Torch device spec, for example cuda:0 or cuda:0,cuda:1 for parallel quick evaluation.')
@click.option('--activations_path', default=None, help='Optional block_activations.pkl used to build pair-importance initialization.')
@click.option('--sample_steps', default=5, type=int, help='Number of rollout steps sampled when estimating activation dissimilarity.')
@click.option('--seed', default=20260325, type=int, help='Seed for evolutionary operators and default evaluation seed.')
@click.option('--eval_seed', default=None, type=int, help='Optional rollout evaluation seed. Defaults to --seed.')
@click.option('--baseline_registry_path', default=_default_baseline_registry_path(), help='JSON registry storing formal original-policy baseline scores.')
@click.option('--baseline_allowed_drop', default=0.01, type=float, help='Legacy absolute tolerance for baseline-relative acceptance.')
@click.option('--baseline_allowed_drop_ratio', default=0.02, type=float, help='Legacy relative tolerance for baseline-relative acceptance.')
@click.option('--formal_accept_drop', default=0.02, type=float, help='Allowed formal rollout score drop from the original baseline.')
@click.option('--meaningful_improvement_eps', default=0.005, type=float, help='Minimum quick-score gain treated as a meaningful best update.')
@click.option('--fresh_eval_patience_after_confirm', default=40, type=int, help='Number of fresh quick evaluations tolerated after formal confirmation.')
@click.option('--formal_confirm_cooldown_fresh_evals', default=20, type=int, help='Fresh-evaluation cooldown between independent formal confirmations.')
@click.option('--max_fresh_evals', default=240, type=int, help='Maximum number of fresh quick rollout evaluations.')
@click.option('--deterministic_torch/--no-deterministic_torch', default=True, help='Enable deterministic Torch/CUDA settings where available.')
@click.option(
    '--init_strategy',
    default='random',
    type=click.Choice(['random', 'structure', 'importance']),
    help='Initial population strategy. Use importance for activation-dissimilarity weighted initialization.',
)
@click.option('--init_seed', default=None, type=int, help='Seed for initial population construction. Defaults to --seed.')
@click.option(
    '--init_structure_profile',
    default='task',
    type=click.Choice(['task', 'balanced']),
    help='Structural prior profile used by structure-guided initialization.',
)
@click.option('--init_random_fraction', default=0.30, type=float, help='Fraction of initial schedules sampled uniformly at random.')
@click.option('--init_structure_min_pairs_per_block', default=4, type=int, help='Per-block refresh-position floor used by guided initialization.')
@click.option('--init_structure_tail_min', default=0.0, type=float, help='Lower random-tail fraction for structure-guided schedules.')
@click.option('--init_structure_tail_max', default=0.0, type=float, help='Upper random-tail fraction for structure-guided schedules.')
@click.option('--init_pair_importance_path', default=None, help='Pair-importance JSON/PKL used for activation-dissimilarity initialization.')
@click.option('--init_importance_gamma', default=1.0, type=float, help='Exponent applied to pair-importance sampling weights.')
@click.option('--init_importance_epsilon', default=0.005, type=float, help='Additive exploration floor for pair-importance sampling.')
def main(checkpoint, task, output_dir, k_pairs, population_size, 
         elite_size, mutation_rate, crossover_rate, tournament_size, num_blocks, num_steps, 
         max_generations, device, activations_path,
         sample_steps, seed, eval_seed, baseline_registry_path,
         baseline_allowed_drop, baseline_allowed_drop_ratio, formal_accept_drop,
         meaningful_improvement_eps, fresh_eval_patience_after_confirm,
         formal_confirm_cooldown_fresh_evals, max_fresh_evals, deterministic_torch,
         init_strategy, init_seed, init_structure_profile, init_random_fraction,
         init_structure_min_pairs_per_block, init_structure_tail_min,
         init_structure_tail_max, init_pair_importance_path,
         init_importance_gamma, init_importance_epsilon):
    """Run EVO search over the full global block–timestep lattice.

    Args:
        checkpoint (Optional[str]): Pretrained Diffusion Policy checkpoint path.
        task (Optional[str]): Benchmark task used for default checkpoint lookup.
        output_dir (str): Directory for schedules, traces, and summaries.
        k_pairs (int): Fixed refresh budget per schedule.
        population_size (int): Number of schedules per generation.
        elite_size (int): Number of elite schedules retained.
        mutation_rate (float): Mutation probability.
        crossover_rate (float): Crossover probability.
        tournament_size (int): Tournament size for parent selection.
        num_blocks (int): Number of cacheable residual branches.
        num_steps (int): Number of denoising timesteps.
        max_generations (int): Maximum evolutionary generations.
        device (str): Torch device specification.
        activations_path (Optional[str]): Optional activation artifact path.
        sample_steps (int): Number of rollout steps sampled for dissimilarity.
        seed (int): Search seed.
        eval_seed (Optional[int]): Rollout evaluation seed.
        baseline_registry_path (str): Formal baseline registry path.
        baseline_allowed_drop (float): Legacy absolute baseline-drop tolerance.
        baseline_allowed_drop_ratio (float): Legacy relative baseline-drop
            tolerance.
        formal_accept_drop (float): Allowed drop from original baseline in
            formal verification.
        meaningful_improvement_eps (float): Minimum meaningful quick-score gain.
        fresh_eval_patience_after_confirm (int): Patience after confirmation.
        formal_confirm_cooldown_fresh_evals (int): Cooldown between formal
            confirmations.
        max_fresh_evals (int): Maximum fresh quick evaluations.
        deterministic_torch (bool): Whether to request deterministic Torch/CUDA
            behavior.
        init_strategy (str): Initial population strategy.
        init_seed (Optional[int]): Initialization seed.
        init_structure_profile (str): Structural prior profile.
        init_random_fraction (float): Random fraction of the initial population.
        init_structure_min_pairs_per_block (int): Per-block guided floor.
        init_structure_tail_min (float): Lower random-tail fraction.
        init_structure_tail_max (float): Upper random-tail fraction.
        init_pair_importance_path (Optional[str]): Existing pair-importance file.
        init_importance_gamma (float): Importance sampling exponent.
        init_importance_epsilon (float): Importance sampling floor.

    Returns:
        None.
    """

    # Resolve checkpoint.
    if checkpoint is None and task is None:
        raise click.UsageError("Either --checkpoint or --task must be provided.")
    
    if checkpoint is None:
        # Use the default checkpoint registered for this task.
        checkpoint = get_checkpoint_path(task)
        if checkpoint is None:
            available_tasks = ", ".join(get_all_available_tasks())
            raise click.UsageError(f"Unknown task '{task}'. Available tasks: {available_tasks}")
        logger.info("Resolved checkpoint for task '%s': %s", task, checkpoint)
    else:
        logger.info("Using checkpoint: %s", checkpoint)
    
    # Create output directory.
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    reproducibility_meta = _configure_reproducibility(
        seed,
        deterministic_torch=deterministic_torch,
    )

    run_timing_summary: Dict[str, Any] = {
        'seed_setup_seconds': 0.0,
        'temp_policy_load_seconds': 0.0,
        'importance_compute_seconds': 0.0,
        'optimizer_init_seconds': 0.0,
        'result_save_seconds': 0.0,
        'cleanup_seconds': 0.0,
        'total_wall_time_seconds': 0.0,
    }
    importance_meta: Dict[str, Any] = {
        "pair_importance_source": None,
        "pair_importance_meta": None,
        "auto_generated_assets": None,
        "initialization_pair_importance": None,
    }
    run_wall_start = time.perf_counter()
    seed_setup_start = time.perf_counter()
    init_pair_weights = None
    block_names_for_scoring = None
    if init_strategy == "importance":
        probe_device = _parse_device_list(device)[0]
        logger.info("Loading cache wrapper to inspect cacheable blocks...")
        temp_policy_start = time.perf_counter()
        temp_policy, temp_cache_wrapper, _ = _load_policy_and_wrapper(checkpoint, probe_device)
        run_timing_summary['temp_policy_load_seconds'] = time.perf_counter() - temp_policy_start
        block_names_for_scoring = temp_cache_wrapper.get_block_names()
        num_blocks = len(block_names_for_scoring)
        del temp_policy
        temp_cache_wrapper.cleanup()

    if init_strategy == "importance":
        logger.info("Using importance initialization; preparing cosine-global pair importance...")
        importance_start = time.perf_counter()
        init_pair_scores, init_importance_meta = _compute_pair_importance_scores(
            task=task,
            checkpoint=checkpoint,
            output_dir=output_dir,
            device=_parse_device_list(device)[0],
            activations_path=activations_path,
            pair_importance_path=init_pair_importance_path,
            sample_steps=sample_steps,
            importance_mode="cosine_global",
        )
        run_timing_summary['importance_compute_seconds'] += time.perf_counter() - importance_start
        init_pair_entries = _build_pair_entries(
            block_names=block_names_for_scoring,
            num_steps=num_steps,
            pair_scores=init_pair_scores,
        )
        init_pair_weights = {
            int(entry["pair_id"]): float(entry["score"])
            for entry in init_pair_entries
        }
        importance_meta["initialization_pair_importance"] = init_importance_meta
    run_timing_summary['seed_setup_seconds'] = time.perf_counter() - seed_setup_start
    
    # Initialize the optimizer.
    optimizer_init_start = time.perf_counter()
    optimizer = GeneticOptimizer(
        checkpoint_path=checkpoint,
        num_blocks=num_blocks,
        num_steps=num_steps,
        task_name=task,
        K=k_pairs,
        population_size=population_size,
        elite_size=elite_size,
        mutation_rate=mutation_rate,
        crossover_rate=crossover_rate,
        tournament_size=tournament_size,
        max_generations=max_generations,
        device=device,
        output_dir=output_dir,
        baseline_registry_path=baseline_registry_path,
        baseline_allowed_drop=baseline_allowed_drop,
        baseline_allowed_drop_ratio=baseline_allowed_drop_ratio,
        formal_accept_drop=formal_accept_drop,
        meaningful_improvement_eps=meaningful_improvement_eps,
        fresh_eval_patience_after_confirm=fresh_eval_patience_after_confirm,
        formal_confirm_cooldown_fresh_evals=formal_confirm_cooldown_fresh_evals,
        max_fresh_evals=max_fresh_evals,
        experiment_seed=seed,
        eval_seed=eval_seed,
        deterministic_torch=deterministic_torch,
        init_strategy=init_strategy,
        init_seed=init_seed if init_seed is not None else seed,
        init_structure_profile=init_structure_profile,
        init_random_fraction=init_random_fraction,
        init_structure_min_pairs_per_block=init_structure_min_pairs_per_block,
        init_structure_tail_min=init_structure_tail_min,
        init_structure_tail_max=init_structure_tail_max,
        init_pair_weights=init_pair_weights,
        init_importance_gamma=init_importance_gamma,
        init_importance_epsilon=init_importance_epsilon,
    )
    optimizer.reproducibility_meta.update(reproducibility_meta)
    optimizer.reproducibility_meta.update(
        {
            'search_seed': seed,
            'eval_seed': eval_seed if eval_seed is not None else seed,
            'quick_eval_seed': optimizer.quick_eval_seed,
            'formal_eval_seed': optimizer.formal_eval_seed,
        }
    )
    run_timing_summary['optimizer_init_seconds'] = time.perf_counter() - optimizer_init_start
    
    results_path = os.path.join(output_dir, 'evo_search_results.json')
    ij_pairs_path = os.path.join(output_dir, 'evo_schedule.json')
    best_ij_pairs: List[Tuple[int, int]] = []
    block_distribution: Dict[int, int] = {}
    step_distribution: Dict[int, int] = {}
    best_fitness: Optional[float] = None

    try:
        # Run search.
        start_time = time.time()
        logger.info("\nStarting evolutionary schedule search...")
        logger.info("Population size: %d", population_size)
        logger.info("Elite size: %d", elite_size)
        logger.info("Mutation rate: %.4f", mutation_rate)
        logger.info("Crossover rate: %.4f", crossover_rate)
        logger.info("Max generations: %d", max_generations)
        logger.info("Selected pair count: %d", k_pairs)
        logger.info("Full search-space size: %d", num_blocks * num_steps)
        logger.info("Search seed: %d", seed)
        logger.info("Initialization strategy: %s", init_strategy)
        logger.info("Initialization seed: %s", optimizer.init_seed)
        logger.info("Quick eval seed: %s", optimizer.quick_eval_seed)
        logger.info("Formal eval seed: %s", optimizer.formal_eval_seed)
        logger.info("Devices: %s", _parse_device_list(device))
        
        best_solution, best_fitness = optimizer.optimize()
        optimization_time = time.time() - start_time
        
        # Convert selected indices to (block, denoise_step) pairs.
        best_ij_pairs = optimizer.get_best_ij_pairs()
        
        logger.info("\nSearch completed in %.2f seconds.", optimization_time)
        logger.info("Best fitness: %.4f", best_fitness)
        logger.info("Selected pair count: %d", len(best_ij_pairs))
        logger.info("Selected pairs: %s", best_ij_pairs)
        
        # Summarize selected pair distribution.
        block_distribution = defaultdict(int)
        step_distribution = defaultdict(int)
        for i, j in best_ij_pairs:
            block_distribution[i] += 1
            step_distribution[j] += 1
        
        logger.info("\nPair distribution:")
        logger.info("Blocks involved: %d", len(block_distribution))
        logger.info("Pairs per block: %s", dict(sorted(block_distribution.items())))
        logger.info("Denoising steps involved: %d", len(step_distribution))
        logger.info(
            "Denoising-step range: %s - %s",
            min(step_distribution.keys()) if step_distribution else 'N/A',
            max(step_distribution.keys()) if step_distribution else 'N/A',
        )
        
        # Save full search results.
        result_save_start = time.perf_counter()
        optimizer.timing_summary['result_save_seconds'] = 0.0
        optimizer.save_results(
            results_path,
            extra_metadata={
                'task': task,
                'checkpoint': checkpoint,
                'seed': seed,
                'search_seed': seed,
                'eval_seed': eval_seed if eval_seed is not None else seed,
                'search_space_mode': 'full_global',
                'full_search_space_size': num_blocks * num_steps,
                'importance_meta': importance_meta,
                'initialization': optimizer.initialization_meta,
                'run_timing_summary': run_timing_summary,
                'reproducibility': optimizer.reproducibility_meta,
            }
        )
        
        # Save compact schedule JSON.
        with open(ij_pairs_path, 'w') as f:
            json.dump({
                'selected_block_timestep_pairs': best_ij_pairs,
                'optimal_ij_pairs': best_ij_pairs,
                'fitness': best_fitness,
                'optimization_time': optimization_time,
                'search_space_mode': 'full_global',
                'run_timing_summary': run_timing_summary,
                'optimizer_timing_summary': optimizer.timing_summary,
                'initialization': optimizer.initialization_meta,
                'reproducibility': optimizer.reproducibility_meta,
                'stop_reason': optimizer.stop_reason,
                'stop_context': optimizer.stop_context,
                'formal_baseline_score': optimizer.formal_baseline_score,
                'confirm_trigger_target': optimizer.confirm_trigger_target,
                'formal_accept_target': optimizer.formal_accept_target,
                'formal_accept_drop': optimizer.formal_accept_drop,
                'hard_target': optimizer.hard_target,
                'soft_target': optimizer.soft_target,
                'best_confirmed_candidate': optimizer._candidate_to_json(optimizer.best_confirmed_candidate),
                'best_formal_candidate': optimizer._candidate_to_json(optimizer.best_formal_candidate),
                'final_selected_candidate': optimizer._candidate_to_json(optimizer.final_selected_candidate),
                'final_search_best_formal_evaluation': optimizer.final_search_best_formal_evaluation,
                'block_distribution': dict(block_distribution),
                'step_distribution': dict(step_distribution),
                'total_blocks_involved': len(block_distribution),
                'total_steps_involved': len(step_distribution)
            }, f, indent=2, ensure_ascii=False)
        run_timing_summary['result_save_seconds'] = time.perf_counter() - result_save_start
        optimizer.timing_summary['result_save_seconds'] = run_timing_summary['result_save_seconds']
        
        logger.info("\nSaved outputs:")
        logger.info("Full results: %s", results_path)
        logger.info("Selected pairs: %s", ij_pairs_path)
    
    finally:
        # Cleanup runtime wrappers.
        logger.info("\nCleaning up...")
        cleanup_start = time.perf_counter()
        optimizer.cleanup()
        run_timing_summary['cleanup_seconds'] = time.perf_counter() - cleanup_start
        optimizer.timing_summary['cleanup_seconds'] = run_timing_summary['cleanup_seconds']
        run_timing_summary['total_wall_time_seconds'] = time.perf_counter() - run_wall_start
        if best_fitness is not None:
            optimizer.save_results(
                results_path,
                extra_metadata={
                    'task': task,
                    'checkpoint': checkpoint,
                    'seed': seed,
                    'search_seed': seed,
                    'eval_seed': eval_seed if eval_seed is not None else seed,
                    'search_space_mode': 'full_global',
                    'full_search_space_size': num_blocks * num_steps,
                    'importance_meta': importance_meta,
                    'initialization': optimizer.initialization_meta,
                    'run_timing_summary': run_timing_summary,
                    'reproducibility': optimizer.reproducibility_meta,
                }
            )
            with open(ij_pairs_path, 'w') as f:
                json.dump({
                    'selected_block_timestep_pairs': best_ij_pairs,
                    'optimal_ij_pairs': best_ij_pairs,
                    'fitness': best_fitness,
                    'optimization_time': optimization_time,
                    'search_space_mode': 'full_global',
                    'run_timing_summary': run_timing_summary,
                    'optimizer_timing_summary': optimizer.timing_summary,
                    'initialization': optimizer.initialization_meta,
                    'reproducibility': optimizer.reproducibility_meta,
                    'stop_reason': optimizer.stop_reason,
                    'stop_context': optimizer.stop_context,
                    'formal_baseline_score': optimizer.formal_baseline_score,
                    'confirm_trigger_target': optimizer.confirm_trigger_target,
                    'formal_accept_target': optimizer.formal_accept_target,
                    'formal_accept_drop': optimizer.formal_accept_drop,
                    'hard_target': optimizer.hard_target,
                    'soft_target': optimizer.soft_target,
                    'best_confirmed_candidate': optimizer._candidate_to_json(optimizer.best_confirmed_candidate),
                    'best_formal_candidate': optimizer._candidate_to_json(optimizer.best_formal_candidate),
                    'final_selected_candidate': optimizer._candidate_to_json(optimizer.final_selected_candidate),
                    'final_search_best_formal_evaluation': optimizer.final_search_best_formal_evaluation,
                    'block_distribution': dict(block_distribution),
                    'step_distribution': dict(step_distribution),
                    'total_blocks_involved': len(block_distribution),
                    'total_steps_involved': len(step_distribution)
                }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main() 
