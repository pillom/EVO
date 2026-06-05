#!/usr/bin/env python3
"""Name canonicalization helpers for EVO importance-guided search.

Activation hooks observe concrete PyTorch dropout modules, while the EVO
offline schedule is defined over residual branches in the block–timestep
lattice. This module bridges those naming conventions before pair-importance
scores are used for redundancy-aware initialization.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def map_activation_block_name_to_evo_block_name(name: str) -> Optional[str]:
    """Map activation-hook module names to EVO residual-branch names.

    Activation collection records dropout modules such as:
    - decoder.layers.0.dropout1
    - decoder.layers.0.dropout2
    - decoder.layers.0.dropout3
    - decoder.layers.0.dropout

    EVO searches over:
    - decoder.layers.0_sa_block
    - decoder.layers.0_mha_block
    - decoder.layers.0_ff_block

    Args:
        name (str): Raw module name recorded by an activation hook.

    Returns:
        Optional[str]: Canonical EVO block name, or ``None`` when the module is
        not one of the cacheable residual branches.
    """
    match = re.match(r"^(decoder\.layers\.\d+)\.(dropout|dropout1|dropout2|dropout3)$", name)
    if not match:
        return None

    layer_name, suffix = match.groups()
    if suffix == "dropout1":
        return f"{layer_name}_sa_block"
    if suffix == "dropout2":
        return f"{layer_name}_mha_block"
    if suffix in {"dropout", "dropout3"}:
        return f"{layer_name}_ff_block"
    return None


def canonicalize_block_scores_for_evo(block_scores: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Fold dropout-level block scores into EVO cache block names.

    We use max aggregation for duplicate mappings because both `dropout`
    and `dropout3` belong to the FF block and should not dilute each other.

    Args:
        block_scores (Dict[str, float]): Importance scores keyed by raw
            activation-hook module name or by already canonical EVO block name.

    Returns:
        Tuple[Dict[str, float], Dict[str, int]]: Canonical block scores and
        matching statistics for matched, passthrough, and unmatched names.
    """
    canonical_scores: Dict[str, float] = {}
    stats = {"matched": 0, "passthrough": 0, "unmatched": 0}

    for raw_name, score in block_scores.items():
        mapped_name = map_activation_block_name_to_evo_block_name(raw_name)
        if mapped_name is not None:
            stats["matched"] += 1
            previous = canonical_scores.get(mapped_name)
            canonical_scores[mapped_name] = score if previous is None else max(previous, score)
            continue

        if raw_name.endswith(("_sa_block", "_mha_block", "_ff_block")):
            stats["passthrough"] += 1
            previous = canonical_scores.get(raw_name)
            canonical_scores[raw_name] = score if previous is None else max(previous, score)
            continue

        stats["unmatched"] += 1

    return canonical_scores, stats


def canonicalize_pair_error_matrix_for_evo(
    error_matrix: np.ndarray,
    block_names: Sequence[str],
    denoise_step_indices: Sequence[int],
) -> Tuple[np.ndarray, List[str], List[str], Dict[str, int]]:
    """Canonicalize an activation dissimilarity matrix for EVO search.

    Duplicate raw modules that map to the same block are reduced with max
    on each denoising-step column so FF-related dropout nodes do not dilute the
    corresponding cache block.

    Args:
        error_matrix (np.ndarray): Raw block-by-step activation dissimilarity
            matrix.
        block_names (Sequence[str]): Raw row names associated with
            ``error_matrix``.
        denoise_step_indices (Sequence[int]): Denoising-step indices associated
            with the matrix columns.

    Returns:
        Tuple[np.ndarray, List[str], List[str], Dict[str, int]]: Canonical matrix,
        canonical EVO block names, denoising-step keys, and name-mapping
        statistics.
    """
    if error_matrix is None:
        raise ValueError("error_matrix is None")

    if len(block_names) != int(error_matrix.shape[0]):
        raise ValueError(
            f"block_names length {len(block_names)} does not match matrix rows {error_matrix.shape[0]}"
        )
    if len(denoise_step_indices) != int(error_matrix.shape[1]):
        raise ValueError(
            "denoise_step_indices length "
            f"{len(denoise_step_indices)} does not match matrix cols {error_matrix.shape[1]}"
        )

    step_keys = [f"denoise_step_{int(step_idx)}" for step_idx in denoise_step_indices]
    canonical_rows: Dict[str, np.ndarray] = {}
    stats = {"matched": 0, "passthrough": 0, "unmatched": 0, "duplicates_merged": 0}

    for row_idx, raw_name in enumerate(block_names):
        mapped_name = map_activation_block_name_to_evo_block_name(raw_name)
        if mapped_name is not None:
            stats["matched"] += 1
        elif raw_name.endswith(("_sa_block", "_mha_block", "_ff_block")):
            mapped_name = raw_name
            stats["passthrough"] += 1
        else:
            stats["unmatched"] += 1
            continue

        row = np.asarray(error_matrix[row_idx], dtype=np.float64)
        previous = canonical_rows.get(mapped_name)
        if previous is None:
            canonical_rows[mapped_name] = row.copy()
        else:
            stats["duplicates_merged"] += 1
            canonical_rows[mapped_name] = np.maximum(previous, row)

    if not canonical_rows:
        raise ValueError("no canonical blocks were found in the pair error matrix")

    canonical_block_names = sorted(canonical_rows.keys())
    canonical_matrix = np.stack([canonical_rows[name] for name in canonical_block_names], axis=0)
    return canonical_matrix, canonical_block_names, step_keys, stats
