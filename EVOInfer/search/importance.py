#!/usr/bin/env python3

"""Compute activation-dissimilarity priors for EVO schedule initialization.

The final EVO objective is rollout performance, not feature similarity.
However, the paper uses residual activation dissimilarity as a lightweight prior
for redundancy-aware initialization. This module converts activations collected
from uncached rollouts into block-, step-, and pair-level scores over the
block–timestep lattice.
"""

import os
import sys
import logging
import numpy as np
import torch
import click
import pickle
import json
from pathlib import Path
from collections import defaultdict
import random

from EVOInfer.search.search_utils import canonicalize_pair_error_matrix_for_evo

# Logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_zero_one(values: np.ndarray) -> np.ndarray:
    """Normalize an array to the closed interval ``[0, 1]``.

    Args:
        values (np.ndarray): Numeric array to normalize.

    Returns:
        np.ndarray: Min-max normalized array. Constant arrays are mapped to
        ``0.5`` to avoid introducing artificial preference.
    """
    values = np.asarray(values, dtype=np.float64)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value == min_value:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def _activation_to_flat_numpy(activation):
    """Convert one activation tensor to a flat NumPy vector.

    Args:
        activation: PyTorch tensor, NumPy array, or array-like activation.

    Returns:
        np.ndarray: Flattened activation vector on CPU.
    """
    if isinstance(activation, torch.Tensor):
        return activation.detach().cpu().numpy().reshape(-1)
    if hasattr(activation, "numpy"):
        return activation.numpy().reshape(-1)
    return np.asarray(activation).reshape(-1)

def load_activations(activations_path):
    """Load activation artifacts produced by activation collection.
    
    Args:
        activations_path (str): Path to ``block_activations.pkl``.
    
    Returns:
        Dict: Rollout-step indexed activation dictionary.
    """
    logger.info("Loading activations from: %s", activations_path)
    
    with open(activations_path, 'rb') as f:
        activations = pickle.load(f)
    
    logger.info("Loaded activations for %d rollout steps.", len(activations))
    return activations


def compute_activation_errors(activations, sample_steps=5):
    """Compute legacy adjacent-step L1 statistics for diagnostic analysis.

    This helper is kept for compatibility with earlier analysis scripts. The
    main EVO initialization path uses pair-level cosine dissimilarity over the
    block–timestep lattice.

    Args:
        activations (Dict): Mapping ``rollout_step -> module_name ->
            denoise_activations``.
        sample_steps (int): Number of rollout steps sampled uniformly.

    Returns:
        Dict[str, List[float]]: Module name to averaged adjacent-step L1
        dissimilarity values across sampled rollout steps.
    """
    logger.info("Computing legacy adjacent activation dissimilarity from %d sampled rollout steps.", sample_steps)

    all_steps = list(activations.keys())
    if len(all_steps) <= sample_steps:
        sampled_steps = all_steps
        logger.info("Using all %d available rollout steps.", len(all_steps))
    else:
        sampled_steps = np.linspace(0, len(all_steps) - 1, sample_steps, dtype=int)
        sampled_steps = [all_steps[i] for i in sampled_steps]
        logger.info("Sampled %d of %d rollout steps: %s", len(sampled_steps), len(all_steps), sampled_steps)

    block_errors = defaultdict(list)

    for step in sampled_steps:
        step_data = activations[step]
        logger.info("Processing rollout step %s", step)

        for module_name, denoise_activations in step_data.items():
            if len(denoise_activations) < 2:
                continue

            step_errors = []
            for i in range(len(denoise_activations) - 1):
                activation1 = denoise_activations[i]
                activation2 = denoise_activations[i + 1]

                if hasattr(activation1, "numpy"):
                    arr1 = activation1.numpy()
                elif isinstance(activation1, torch.Tensor):
                    arr1 = activation1.detach().cpu().numpy()
                else:
                    arr1 = np.array(activation1)

                if hasattr(activation2, "numpy"):
                    arr2 = activation2.numpy()
                elif isinstance(activation2, torch.Tensor):
                    arr2 = activation2.detach().cpu().numpy()
                else:
                    arr2 = np.array(activation2)

                step_errors.append(np.mean(np.abs(arr1 - arr2)))

            if step_errors:
                block_errors[module_name].append(np.mean(step_errors))

    logger.info("Computed legacy activation dissimilarity for %d blocks.", len(block_errors))
    return block_errors


def calculate_block_importance(block_errors):
    """Compute legacy normalized block importance from activation dissimilarity.

    Args:
        block_errors (Dict[str, List[float]]): Per-module adjacent-step
            dissimilarity values.

    Returns:
        Dict[str, float]: Block scores normalized to ``[0, 1]``.
    """
    logger.info("Computing block importance.")

    block_importance = {}
    for module_name, errors in block_errors.items():
        if errors:
            avg_error = np.mean(errors)
            block_importance[module_name] = avg_error
            logger.info("Block %s: average error = %.6f", module_name, avg_error)

    if not block_importance:
        logger.warning("No block importance scores were computed.")
        return {}

    min_importance = min(block_importance.values())
    max_importance = max(block_importance.values())

    logger.info("Block importance range: [%.6f, %.6f]", min_importance, max_importance)

    if max_importance == min_importance:
        logger.info("All block scores are identical; assigning 0.5 to each block.")
        return {name: 0.5 for name in block_importance.keys()}

    normalized_importance = {}
    for name, importance in block_importance.items():
        normalized = (importance - min_importance) / (max_importance - min_importance)
        normalized_importance[name] = normalized
        logger.info("Block %s: normalized importance = %.6f", name, normalized)

    return normalized_importance


def save_block_importance_results(importance_scores, task_name, checkpoint_path, output_dir="."):
    """Save block-level importance scores as JSON and pickle artifacts.

    Args:
        importance_scores (Dict[str, float]): Normalized block scores.
        task_name (str): Benchmark task name.
        checkpoint_path (str): Source checkpoint path used for activation
            collection.
        output_dir (str): Directory where output artifacts are written.

    Returns:
        Tuple[Path, Path]: Paths to the JSON and pickle files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_python_type(obj):
        """Convert NumPy values into JSON-serializable Python values.

        Args:
            obj: Object that may contain NumPy scalars, arrays, dictionaries, or
                sequences.

        Returns:
            object: JSON-serializable equivalent.
        """
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert_to_python_type(item) for item in obj]
        return obj

    converted_scores = convert_to_python_type(importance_scores)
    results = {
        "task_name": task_name,
        "checkpoint_path": checkpoint_path,
        "block_importance": converted_scores,
        "total_blocks": len(importance_scores),
        "statistics": {
            "min_importance": float(min(importance_scores.values())) if importance_scores else 0,
            "max_importance": float(max(importance_scores.values())) if importance_scores else 0,
            "mean_importance": float(np.mean(list(importance_scores.values()))) if importance_scores else 0,
            "std_importance": float(np.std(list(importance_scores.values()))) if importance_scores else 0,
        },
    }

    json_path = output_dir / f"block_importance_{task_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pkl_path = output_dir / f"block_importance_{task_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    logger.info("Block importance JSON saved to: %s", json_path)
    logger.info("Block importance pickle saved to: %s", pkl_path)
    return json_path, pkl_path

def compute_step_error_matrix(activations, sample_steps=5):
    """Compute a legacy block–timestep dissimilarity matrix with adjacent L1.

    The returned variable is still named ``error_matrix`` for compatibility,
    but it is used as an activation-dissimilarity prior rather than as the final
    schedule objective.
    
    Args:
        activations (Dict): Mapping ``rollout_step -> module_name ->
            denoise_activations``.
        sample_steps (int): Number of rollout steps sampled uniformly.
    
    Returns:
        Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[str]]]:
        Block–timestep dissimilarity matrix, denoising-step indices, and raw
        block names.
    """
    logger.info("Computing legacy denoising-step dissimilarity matrix from %d sampled rollout steps.", sample_steps)
    
    # List rollout steps.
    all_steps = list(activations.keys())
    
    # Uniformly sample rollout steps if needed.
    if len(all_steps) <= sample_steps:
        sampled_steps = all_steps
        logger.info("Using all %d available rollout steps.", len(all_steps))
    else:
        # Uniform sampling across the rollout.
        sampled_steps = np.linspace(0, len(all_steps)-1, sample_steps, dtype=int)
        sampled_steps = [all_steps[i] for i in sampled_steps]
        logger.info("Sampled %d of %d rollout steps: %s", len(sampled_steps), len(all_steps), sampled_steps)
    
    # Accumulate one dissimilarity matrix per sampled rollout step.
    error_matrices = []
    block_names = None
    max_denoise_steps = 0
    
    for step in sampled_steps:
        step_data = activations[step]
        logger.info("Processing rollout step %s", step)
        
        # Use the first sampled step to define block ordering.
        if block_names is None:
            block_names = list(step_data.keys())
            logger.info("Detected %d blocks: %s", len(block_names), block_names)
        
        # Track maximum denoising length across blocks.
        for module_name, denoise_activations in step_data.items():
            max_denoise_steps = max(max_denoise_steps, len(denoise_activations))
        
        # Build the matrix for this rollout step.
        step_error_matrix = []
        
        for module_name in block_names:
            if module_name not in step_data:
                continue
                
            denoise_activations = step_data[module_name]
            if len(denoise_activations) < 2:
                # Not enough denoising activations; pad with zeros.
                module_errors = [0.0] * max_denoise_steps
            else:
                # Compute adjacent-step L1 dissimilarity.
                module_errors = []
                
                for i in range(len(denoise_activations) - 1):
                    activation1 = denoise_activations[i]
                    activation2 = denoise_activations[i + 1]
                    
                    # Convert activations to NumPy arrays.
                    if hasattr(activation1, 'numpy'):
                        arr1 = activation1.numpy()
                    elif isinstance(activation1, torch.Tensor):
                        arr1 = activation1.detach().cpu().numpy()
                    else:
                        arr1 = np.array(activation1)
                    
                    if hasattr(activation2, 'numpy'):
                        arr2 = activation2.numpy()
                    elif isinstance(activation2, torch.Tensor):
                        arr2 = activation2.detach().cpu().numpy()
                    else:
                        arr2 = np.array(activation2)
                    
                    # Mean L1 dissimilarity.
                    l1_error = np.mean(np.abs(arr1 - arr2))
                    module_errors.append(l1_error)
                
                # Pad shorter blocks to the maximum denoising length.
                while len(module_errors) < max_denoise_steps:
                    if module_errors:
                        module_errors.append(module_errors[-1])
                    else:
                        module_errors.append(0.0)
            
            step_error_matrix.append(module_errors)
        
        if step_error_matrix:
            error_matrices.append(np.array(step_error_matrix))
    
    if not error_matrices:
        logger.error("No error matrices were computed.")
        return None, None, None
    
    # Average across sampled rollout steps.
    avg_error_matrix = np.mean(error_matrices, axis=0)
    
    # Build denoising-step indices.
    denoise_step_indices = list(range(max_denoise_steps))
    
    logger.info("Dissimilarity matrix shape: %s", avg_error_matrix.shape)
    logger.info("Blocks: %d, denoising steps: %d", len(block_names), len(denoise_step_indices))
    
    return avg_error_matrix, denoise_step_indices, block_names


def compute_cosine_global_pair_importance_matrix(activations, sample_steps=5):
    """Compute pair-level importance using global cosine dissimilarity.

    Compute a block-by-denoising-step importance matrix using global cosine
    dissimilarity inside each block.

    For each sampled rollout step and each block, this builds the full
    denoise-step cosine matrix, then scores every denoise step by its mean
    dissimilarity to all other steps: mean(1 - cosine(step, other_step)).
    The final matrix averages these scores across sampled rollout steps.

    Args:
        activations (Dict): Mapping ``rollout_step -> module_name ->
            denoise_activations``.
        sample_steps (int): Number of rollout steps sampled uniformly.

    Returns:
        Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[str]]]:
        Pair-importance matrix, denoising-step indices, and raw block names.
    """
    logger.info(
        "Computing cosine-global pair importance from %d sampled rollout steps.",
        sample_steps,
    )

    all_steps = list(activations.keys())
    if len(all_steps) <= sample_steps:
        sampled_steps = all_steps
        logger.info("Using all %d available rollout steps.", len(all_steps))
    else:
        sampled_indices = np.linspace(0, len(all_steps) - 1, sample_steps, dtype=int)
        sampled_steps = [all_steps[i] for i in sampled_indices]
        logger.info("Sampled %d of %d rollout steps: %s", len(sampled_steps), len(all_steps), sampled_steps)

    if not sampled_steps:
        logger.error("No rollout steps found in activations.")
        return None, None, None

    first_step_data = activations[sampled_steps[0]]
    block_names = list(first_step_data.keys())
    if not block_names:
        logger.error("No block activations found.")
        return None, None, None

    max_denoise_steps = 0
    for step in sampled_steps:
        step_data = activations[step]
        for module_name in block_names:
            if module_name in step_data:
                max_denoise_steps = max(max_denoise_steps, len(step_data[module_name]))

    if max_denoise_steps <= 0:
        logger.error("No denoising steps found.")
        return None, None, None

    rollout_matrices = []
    for step in sampled_steps:
        step_data = activations[step]
        logger.info("Processing rollout step %s", step)
        step_matrix = []

        for module_name in block_names:
            denoise_activations = step_data.get(module_name, [])
            if len(denoise_activations) <= 1:
                scores = [0.0] * max_denoise_steps
                step_matrix.append(scores)
                continue

            rows = [_activation_to_flat_numpy(item).astype(np.float64) for item in denoise_activations]
            row_lengths = {row.shape[0] for row in rows}
            if len(row_lengths) != 1:
                logger.warning("Skipping %s because denoising activations have inconsistent shapes.", module_name)
                scores = [0.0] * max_denoise_steps
                step_matrix.append(scores)
                continue

            matrix = np.stack(rows, axis=0)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normalized = matrix / np.maximum(norms, 1e-12)
            cosine_matrix = np.clip(normalized @ normalized.T, -1.0, 1.0)
            dissimilarity = 1.0 - cosine_matrix

            n_steps = dissimilarity.shape[0]
            if n_steps > 1:
                sums = dissimilarity.sum(axis=1)
                scores = (sums / float(n_steps - 1)).tolist()
            else:
                scores = [0.0]

            while len(scores) < max_denoise_steps:
                scores.append(scores[-1] if scores else 0.0)
            step_matrix.append(scores[:max_denoise_steps])

        rollout_matrices.append(np.asarray(step_matrix, dtype=np.float64))

    if not rollout_matrices:
        logger.error("No cosine importance matrices were computed.")
        return None, None, None

    avg_importance_matrix = np.mean(rollout_matrices, axis=0)
    denoise_step_indices = list(range(max_denoise_steps))
    logger.info(
        "Cosine-global pair importance matrix shape: %s",
        avg_importance_matrix.shape,
    )
    return avg_importance_matrix, denoise_step_indices, block_names

def calculate_step_importance(error_matrix, denoise_step_indices, block_names):
    """Compute normalized denoising-step importance by averaging over blocks.
    
    Args:
        error_matrix (np.ndarray): Block–timestep dissimilarity matrix used as a
            legacy step-level prior.
        denoise_step_indices (List[int]): Denoising-step indices for matrix
            columns.
        block_names (List[str]): Raw block names for matrix rows.
    
    Returns:
        Dict[str, float]: Denoising-step keys mapped to normalized scores in
        ``[0, 1]``.
    """
    logger.info("Computing denoising-step importance.")
    
    if error_matrix is None:
        logger.error("Dissimilarity matrix is None.")
        return {}
    
    # Average each denoising-step column over blocks.
    step_importance = {}
    for step_idx, step_id in enumerate(denoise_step_indices):
        if step_idx < error_matrix.shape[1]:
            # Average this denoising-step column.
            column_avg = np.mean(error_matrix[:, step_idx])
            step_importance[f"denoise_step_{step_id}"] = column_avg
            logger.info("Denoising step %s: average error = %.6f", step_id, column_avg)
    
    if not step_importance:
        logger.warning("No step importance scores were computed.")
        return {}
    
    # Normalize into [0, 1].
    min_importance = min(step_importance.values())
    max_importance = max(step_importance.values())
    
    logger.info("Step importance range: [%.6f, %.6f]", min_importance, max_importance)
    
    if max_importance == min_importance:
        # All values are identical.
        normalized_importance = {name: 0.5 for name in step_importance.keys()}
        logger.info("All step scores are identical; assigning 0.5 to each step.")
    else:
        # Standard min-max normalization.
        normalized_importance = {}
        for name, importance in step_importance.items():
            normalized = (importance - min_importance) / (max_importance - min_importance)
            normalized_importance[name] = normalized
            logger.info("Step %s: normalized importance = %.6f", name, normalized)
    
    return normalized_importance

def save_pair_importance_results(
    error_matrix,
    denoise_step_indices,
    block_names,
    task_name,
    checkpoint_path,
    output_dir=".",
):
    """Save the canonical EVO pair-importance matrix.

    The pair importance keeps the full (block, denoise_step) structure instead
    of collapsing it into separable block-only and step-only marginals.

    Args:
        error_matrix (np.ndarray): Raw block–timestep activation-dissimilarity
            matrix.
        denoise_step_indices (List[int]): Denoising-step indices for columns.
        block_names (List[str]): Raw block names for rows.
        task_name (str): Benchmark task name.
        checkpoint_path (str): Source checkpoint path.
        output_dir (str): Directory where output artifacts are written.

    Returns:
        Tuple[Path, Path]: Paths to the JSON and pickle pair-importance files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical_matrix, canonical_block_names, step_keys, canonical_stats = (
        canonicalize_pair_error_matrix_for_evo(error_matrix, block_names, denoise_step_indices)
    )
    normalized_matrix = normalize_zero_one(canonical_matrix)

    pair_importance = {}
    for row_idx, block_name in enumerate(canonical_block_names):
        pair_importance[block_name] = {
            step_key: float(normalized_matrix[row_idx, col_idx])
            for col_idx, step_key in enumerate(step_keys)
        }

    results = {
        "task_name": task_name,
        "checkpoint_path": checkpoint_path,
        "pair_importance": pair_importance,
        "block_names": canonical_block_names,
        "step_keys": step_keys,
        "matrix_shape": [int(normalized_matrix.shape[0]), int(normalized_matrix.shape[1])],
        "canonicalization": canonical_stats,
        "statistics": {
            "min_importance": float(normalized_matrix.min()) if normalized_matrix.size else 0.0,
            "max_importance": float(normalized_matrix.max()) if normalized_matrix.size else 0.0,
            "mean_importance": float(normalized_matrix.mean()) if normalized_matrix.size else 0.0,
            "std_importance": float(normalized_matrix.std()) if normalized_matrix.size else 0.0,
        },
    }

    json_path = output_dir / f"pair_importance_{task_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    pkl_path = output_dir / f"pair_importance_{task_name}.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    logger.info("Pair importance JSON saved to: %s", json_path)
    logger.info("Pair importance pickle saved to: %s", pkl_path)
    logger.info(
        "Pair importance canonicalized to %d blocks x %d denoise steps",
        len(canonical_block_names),
        len(step_keys),
    )

    return json_path, pkl_path


def save_importance_results(importance_scores, task_name, checkpoint_path, output_dir="."):
    """Save normalized denoising-step importance scores.
    
    Args:
        importance_scores (Dict[str, float]): Normalized step-importance scores.
        task_name (str): Benchmark task name.
        checkpoint_path (str): Source checkpoint path.
        output_dir (str): Directory where output artifacts are written.

    Returns:
        Tuple[Path, Path]: Paths to the JSON and pickle files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert NumPy values to JSON-serializable Python types.
    def convert_to_python_type(obj):
        """Convert NumPy values into JSON-serializable Python values.

        Args:
            obj: Object that may contain NumPy scalars, arrays, dictionaries, or
                sequences.

        Returns:
            object: JSON-serializable equivalent.
        """
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_python_type(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python_type(item) for item in obj]
        else:
            return obj
    
    # Convert the payload before writing.
    converted_scores = convert_to_python_type(importance_scores)
    
    # Build result metadata.
    results = {
        'task_name': task_name,
        'checkpoint_path': checkpoint_path,
        'step_importance': converted_scores,
        'total_steps': len(importance_scores),
        'statistics': {
            'min_importance': float(min(importance_scores.values())) if importance_scores else 0,
            'max_importance': float(max(importance_scores.values())) if importance_scores else 0,
            'mean_importance': float(np.mean(list(importance_scores.values()))) if importance_scores else 0,
            'std_importance': float(np.std(list(importance_scores.values()))) if importance_scores else 0
        }
    }
    
    # Save JSON.
    json_path = output_dir / f"step_importance_{task_name}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info("Step importance JSON saved to: %s", json_path)
    
    # Save pickle for Python-side reuse.
    pkl_path = output_dir / f"step_importance_{task_name}.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info("Step importance pickle saved to: %s", pkl_path)
    
    # Log a concise summary.
    logger.info("=" * 60)
    logger.info("Step importance summary")
    logger.info("Task: %s", task_name)
    logger.info("Total steps: %d", results['total_steps'])
    logger.info("Statistics:")
    logger.info("  min: %.6f", results['statistics']['min_importance'])
    logger.info("  max: %.6f", results['statistics']['max_importance'])
    logger.info("  mean: %.6f", results['statistics']['mean_importance'])
    logger.info("  std: %.6f", results['statistics']['std_importance'])
    logger.info("=" * 60)
    
    return json_path, pkl_path

@click.command(help="Compute EVO activation-dissimilarity priors for schedule initialization.")
@click.option('-t', '--task_name', required=True, help='Benchmark task name used to label importance artifacts.')
@click.option('-c', '--checkpoint', required=True, help='Path to the pretrained Diffusion Policy checkpoint.')
@click.option('-a', '--activations_path', default=None, help='Path to block_activations.pkl from activation collection.')
@click.option('-o', '--output_dir', default='.', help='Directory for step- and pair-importance artifacts.')
@click.option('--sample_steps', default=5, type=int, help='Number of rollout steps sampled to estimate activation dissimilarity.')
def main(task_name, checkpoint, activations_path, output_dir, sample_steps):
    """CLI entry point for computing EVO importance artifacts.

    Args:
        task_name (str): Benchmark task name.
        checkpoint (str): Path to the pretrained Diffusion Policy checkpoint.
        activations_path (Optional[str]): Existing activation artifact path.
        output_dir (str): Directory where output artifacts are saved.
        sample_steps (int): Number of rollout steps sampled for dissimilarity
            estimation.

    Returns:
        None.
    """
    logger.info("Computing importance for task: %s", task_name)
    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Sampled rollout steps: %d", sample_steps)
    
    # Use the default activation path if none is supplied.
    if activations_path is None:
        activations_path = f"activations/{task_name}/block_activations.pkl"
    
    logger.info("Activation path: %s", activations_path)
    
    # Activation collection must be run before importance computation.
    if not os.path.exists(activations_path):
        logger.error("Activation file not found: %s", activations_path)
        logger.error("Run activation_collection.py first, or pass --activations_path.")
        return
    
    try:
        # 1. Load activations.
        activations = load_activations(activations_path)
        
        # 2. Compute the legacy block–timestep dissimilarity matrix.
        error_matrix, denoise_step_indices, block_names = compute_step_error_matrix(activations, sample_steps)
        
        # 3. Save legacy step-level importance.
        importance_scores = calculate_step_importance(error_matrix, denoise_step_indices, block_names)
        
        # 4. Save EVO pair-level importance.
        json_path, pkl_path = save_importance_results(
            importance_scores, task_name, checkpoint, output_dir
        )
        pair_json_path, pair_pkl_path = save_pair_importance_results(
            error_matrix=error_matrix,
            denoise_step_indices=denoise_step_indices,
            block_names=block_names,
            task_name=task_name,
            checkpoint_path=checkpoint,
            output_dir=output_dir,
        )

        logger.info("Importance computation completed.")
        logger.info("Pair importance JSON: %s", pair_json_path)

    except Exception as e:
        logger.error("Step importance computation failed: %s", e)
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
