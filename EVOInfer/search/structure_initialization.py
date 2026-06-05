"""Initial population builders for EVO offline schedule search.

This module constructs candidate cache schedules before rollout-driven
evolutionary search starts. Redundancy-aware initialization uses activation
dissimilarity as a prior over the block–timestep lattice, but it does not prune
the search space or replace rollout performance as the optimization objective.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BLOCK_TYPES = ("sa", "mha", "ff")
STEP_BINS = ("0-24", "25-49", "50-74", "75-99")
LAYER_GROUPS = ("L0-L1", "L2-L5", "L6-L7")


@dataclass(frozen=True)
class PairMeta:
    """Metadata for one block–timestep lattice position.

    Args:
        pair_id (int): Flattened lattice index ``block_idx * num_steps + step``.
        block_idx (int): Cacheable residual-branch index.
        step_idx (int): Denoising timestep index.
        block_type (str): Residual branch type, one of self-attention,
            cross-attention, or feed-forward.
        layer_idx (int): Transformer decoder layer index.
        step_bin (str): Coarse denoising-step bin used for structural quotas.
        layer_group (str): Coarse layer group used for structural quotas.

    Returns:
        None.
    """

    pair_id: int
    block_idx: int
    step_idx: int
    block_type: str
    layer_idx: int
    step_bin: str
    layer_group: str


def step_bin(step_idx: int) -> str:
    """Map a denoising step to a coarse temporal bin.

    Args:
        step_idx (int): Denoising timestep index in the diffusion sampler.

    Returns:
        str: Temporal bin label used by structure-guided initialization.
    """
    if step_idx < 25:
        return "0-24"
    if step_idx < 50:
        return "25-49"
    if step_idx < 75:
        return "50-74"
    return "75-99"


def layer_group(layer_idx: int) -> str:
    """Map a decoder layer to a coarse depth group.

    Args:
        layer_idx (int): Transformer decoder layer index.

    Returns:
        str: Layer-group label used by structure-guided initialization.
    """
    if layer_idx <= 1:
        return "L0-L1"
    if layer_idx <= 5:
        return "L2-L5"
    return "L6-L7"


def build_pair_metadata(
    *,
    num_blocks: int,
    num_steps: int,
    solution_universe: Sequence[int],
) -> Dict[int, PairMeta]:
    """Build metadata for feasible block–timestep positions.

    Args:
        num_blocks (int): Number of cacheable residual branches.
        num_steps (int): Number of denoising timesteps.
        solution_universe (Sequence[int]): Flattened lattice positions that may
            appear in a candidate schedule.

    Returns:
        Dict[int, PairMeta]: Mapping from flattened pair id to metadata.
    """
    allowed = set(int(pair_id) for pair_id in solution_universe)
    metadata: Dict[int, PairMeta] = {}
    for block_idx in range(num_blocks):
        block_type = BLOCK_TYPES[block_idx % 3]
        layer_idx = block_idx // 3
        for step_idx in range(num_steps):
            pair_id = block_idx * num_steps + step_idx
            if pair_id not in allowed:
                continue
            metadata[pair_id] = PairMeta(
                pair_id=pair_id,
                block_idx=block_idx,
                step_idx=step_idx,
                block_type=block_type,
                layer_idx=layer_idx,
                step_bin=step_bin(step_idx),
                layer_group=layer_group(layer_idx),
            )
    return metadata


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize nonnegative weights into a probability distribution.

    Args:
        weights (Dict[str, float]): Raw group weights.

    Returns:
        Dict[str, float]: Normalized weights with the same keys.
    """
    total = sum(max(0.0, float(value)) for value in weights.values())
    if total <= 0:
        uniform = 1.0 / len(weights)
        return {key: uniform for key in weights}
    return {key: max(0.0, float(value)) / total for key, value in weights.items()}


def _profile_for_task(task_name: Optional[str], profile: str) -> Dict[str, Dict[str, float]]:
    """Return broad structural centers used for initialization.

    These are deliberately task-agnostic priors. They define the shape of
    initial candidates, not fixed update pairs.

    Args:
        task_name (Optional[str]): Benchmark task name. Currently unused because
            the released initializer uses task-agnostic structural priors.
        profile (str): Requested profile name.

    Returns:
        Dict[str, Dict[str, float]]: Probability centers over branch type,
        denoising-step bin, and layer group.
    """
    del task_name

    uniform_structure = {
        "type": {"sa": 0.31, "mha": 0.29, "ff": 0.40},
        "step": {"0-24": 0.28, "25-49": 0.23, "50-74": 0.21, "75-99": 0.28},
        "layer_group": {"L0-L1": 0.31, "L2-L5": 0.43, "L6-L7": 0.26},
    }
    if profile in {"task", "balanced"}:
        return uniform_structure
    return uniform_structure


def _sample_dirichlet_like(
    rng: random.Random,
    centers: Dict[str, float],
    *,
    concentration: float,
) -> Dict[str, float]:
    """Sample a smooth categorical distribution around structural centers.

    Args:
        rng (random.Random): Random generator used for reproducible sampling.
        centers (Dict[str, float]): Mean categorical proportions.
        concentration (float): Dirichlet-like concentration controlling how
            tightly samples stay around ``centers``.

    Returns:
        Dict[str, float]: Sampled normalized proportions.
    """
    normalized = _normalize(centers)
    samples = {}
    for key, center in normalized.items():
        alpha = max(0.01, center * concentration)
        samples[key] = rng.gammavariate(alpha, 1.0)
    return _normalize(samples)


def _largest_remainder_counts(proportions: Dict[str, float], total: int) -> Dict[str, int]:
    """Convert proportions into integer quotas with largest remainders.

    Args:
        proportions (Dict[str, float]): Desired categorical proportions.
        total (int): Total number of positions to allocate.

    Returns:
        Dict[str, int]: Integer count assigned to each category.
    """
    raw = {key: max(0.0, value) * total for key, value in proportions.items()}
    counts = {key: int(value) for key, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(raw.keys(), key=lambda key: raw[key] - counts[key], reverse=True)
    for key in order[:remaining]:
        counts[key] += 1
    return counts


def _weighted_choice(rng: random.Random, items: Sequence[int], weights: Sequence[float]) -> int:
    """Sample one item from nonnegative weights.

    Args:
        rng (random.Random): Random generator used for reproducible sampling.
        items (Sequence[int]): Candidate lattice positions.
        weights (Sequence[float]): Nonnegative sampling weights aligned with
            ``items``.

    Returns:
        int: Sampled item.
    """
    total = sum(max(0.0, weight) for weight in weights)
    if total <= 0:
        return rng.choice(list(items))
    threshold = rng.random() * total
    running = 0.0
    for item, weight in zip(items, weights):
        running += max(0.0, weight)
        if running >= threshold:
            return item
    return items[-1]


def _importance_sampling_weight(
    pair_id: int,
    pair_weights: Dict[int, float],
    *,
    gamma: float,
    epsilon: float,
) -> float:
    """Compute the sampling weight for one importance-guided position.

    Args:
        pair_id (int): Flattened block–timestep lattice position.
        pair_weights (Dict[int, float]): Activation-dissimilarity prior over
            lattice positions.
        gamma (float): Exponent applied to sharpen or smooth the prior.
        epsilon (float): Additive floor that preserves exploration.

    Returns:
        float: Nonnegative sampling weight for the position.
    """
    raw_weight = max(0.0, float(pair_weights.get(int(pair_id), 0.0)))
    return float((max(0.0, epsilon) + raw_weight) ** max(0.0, gamma))


def _counts_for_solution(solution: Iterable[int], metadata: Dict[int, PairMeta]) -> Dict[str, Counter]:
    """Count structural categories represented by a candidate schedule.

    Args:
        solution (Iterable[int]): Flattened block–timestep positions in one
            candidate schedule.
        metadata (Dict[int, PairMeta]): Metadata for each feasible position.

    Returns:
        Dict[str, Counter]: Counts by branch type, timestep bin, and layer group.
    """
    counts = {
        "type": Counter(),
        "step": Counter(),
        "layer_group": Counter(),
    }
    for pair_id in solution:
        meta = metadata[int(pair_id)]
        counts["type"][meta.block_type] += 1
        counts["step"][meta.step_bin] += 1
        counts["layer_group"][meta.layer_group] += 1
    return counts


def _candidate_weight(
    pair_id: int,
    metadata: Dict[int, PairMeta],
    counts: Dict[str, Counter],
    quotas: Dict[str, Dict[str, int]],
) -> float:
    """Score a candidate position by unmet structural quotas.

    Args:
        pair_id (int): Candidate flattened lattice position.
        metadata (Dict[int, PairMeta]): Metadata for feasible positions.
        counts (Dict[str, Counter]): Current category counts in the partial
            schedule.
        quotas (Dict[str, Dict[str, int]]): Desired category quotas.

    Returns:
        float: Sampling weight that favors categories still below quota.
    """
    meta = metadata[pair_id]
    type_need = max(0, quotas["type"].get(meta.block_type, 0) - counts["type"][meta.block_type]) + 1
    step_need = max(0, quotas["step"].get(meta.step_bin, 0) - counts["step"][meta.step_bin]) + 1
    group_need = (
        max(0, quotas["layer_group"].get(meta.layer_group, 0) - counts["layer_group"][meta.layer_group])
        + 1
    )
    return float(type_need * step_need * group_need)


def _repair_to_k(
    *,
    rng: random.Random,
    solution: Sequence[int],
    universe: Sequence[int],
    k_pairs: int,
) -> List[int]:
    """Repair a candidate schedule so it contains exactly ``k_pairs`` positions.

    Args:
        rng (random.Random): Random generator used for trimming or refilling.
        solution (Sequence[int]): Candidate flattened lattice positions.
        universe (Sequence[int]): Feasible position universe.
        k_pairs (int): Required refresh budget.

    Returns:
        List[int]: Sorted schedule with exactly ``k_pairs`` unique positions.
    """
    repaired = sorted(dict.fromkeys(int(pair_id) for pair_id in solution))
    if len(repaired) > k_pairs:
        repaired = sorted(rng.sample(repaired, k_pairs))
    if len(repaired) < k_pairs:
        existing = set(repaired)
        remaining = [int(pair_id) for pair_id in universe if int(pair_id) not in existing]
        repaired.extend(rng.sample(remaining, k_pairs - len(repaired)))
    return sorted(repaired)


def sample_structure_solution(
    *,
    rng: random.Random,
    universe: Sequence[int],
    metadata: Dict[int, PairMeta],
    k_pairs: int,
    num_blocks: int,
    task_name: Optional[str],
    profile: str,
    min_pairs_per_block: int,
    random_tail_ratio: float,
    concentration: float = 80.0,
) -> Tuple[List[int], Dict[str, Any]]:
    """Sample one structure-guided initial schedule.

    Args:
        rng (random.Random): Random generator used for reproducible sampling.
        universe (Sequence[int]): Feasible flattened lattice positions.
        metadata (Dict[int, PairMeta]): Metadata for feasible positions.
        k_pairs (int): Fixed refresh budget.
        num_blocks (int): Number of cacheable residual branches.
        task_name (Optional[str]): Benchmark task name used for metadata only.
        profile (str): Structural profile name.
        min_pairs_per_block (int): Floor on positions sampled per block before
            quota-based filling.
        random_tail_ratio (float): Fraction of the candidate replaced by random
            positions to preserve exploration.
        concentration (float): Dirichlet-like concentration around structural
            profile centers.

    Returns:
        Tuple[List[int], Dict[str, Any]]: Sampled schedule and summary metadata.
    """
    centers = _profile_for_task(task_name, profile)
    proportions = {
        "type": _sample_dirichlet_like(rng, centers["type"], concentration=concentration),
        "step": _sample_dirichlet_like(rng, centers["step"], concentration=concentration),
        "layer_group": _sample_dirichlet_like(rng, centers["layer_group"], concentration=concentration),
    }
    quotas = {
        group_name: _largest_remainder_counts(group_props, k_pairs)
        for group_name, group_props in proportions.items()
    }

    by_block: Dict[int, List[int]] = defaultdict(list)
    for pair_id in universe:
        meta = metadata[int(pair_id)]
        by_block[meta.block_idx].append(int(pair_id))

    selected = set()
    for block_idx in range(num_blocks):
        block_candidates = by_block.get(block_idx, [])
        if not block_candidates:
            continue
        take = min(int(min_pairs_per_block), len(block_candidates), max(0, k_pairs - len(selected)))
        if take > 0:
            selected.update(rng.sample(block_candidates, take))

    while len(selected) < k_pairs:
        counts = _counts_for_solution(selected, metadata)
        candidates = [int(pair_id) for pair_id in universe if int(pair_id) not in selected]
        if not candidates:
            break
        weights = [_candidate_weight(pair_id, metadata, counts, quotas) for pair_id in candidates]
        selected.add(_weighted_choice(rng, candidates, weights))

    solution = _repair_to_k(rng=rng, solution=sorted(selected), universe=universe, k_pairs=k_pairs)

    tail_count = int(round(max(0.0, min(1.0, random_tail_ratio)) * k_pairs))
    if tail_count > 0:
        keep_count = max(0, k_pairs - tail_count)
        kept = set(rng.sample(solution, keep_count))
        remaining = [int(pair_id) for pair_id in universe if int(pair_id) not in kept]
        solution = _repair_to_k(
            rng=rng,
            solution=list(kept) + rng.sample(remaining, min(tail_count, len(remaining))),
            universe=universe,
            k_pairs=k_pairs,
        )

    final_counts = _counts_for_solution(solution, metadata)
    summary = {
        "profile": profile,
        "profile_centers": centers,
        "sampled_proportions": proportions,
        "sampled_quotas": quotas,
        "min_pairs_per_block": int(min_pairs_per_block),
        "random_tail_ratio": float(random_tail_ratio),
        "final_counts": {key: dict(counter) for key, counter in final_counts.items()},
    }
    return solution, summary


def sample_importance_solution(
    *,
    rng: random.Random,
    universe: Sequence[int],
    metadata: Dict[int, PairMeta],
    pair_weights: Dict[int, float],
    k_pairs: int,
    num_blocks: int,
    min_pairs_per_block: int,
    gamma: float,
    epsilon: float,
) -> Tuple[List[int], Dict[str, Any]]:
    """Sample one importance-guided initial schedule.

    Args:
        rng (random.Random): Random generator used for reproducible sampling.
        universe (Sequence[int]): Feasible flattened lattice positions.
        metadata (Dict[int, PairMeta]): Metadata for feasible positions.
        pair_weights (Dict[int, float]): Activation-dissimilarity prior over
            block–timestep positions.
        k_pairs (int): Fixed refresh budget.
        num_blocks (int): Number of cacheable residual branches.
        min_pairs_per_block (int): Floor on positions sampled per block.
        gamma (float): Exponent controlling concentration on high-dissimilarity
            positions.
        epsilon (float): Additive sampling floor that keeps every position
            reachable.

    Returns:
        Tuple[List[int], Dict[str, Any]]: Sampled schedule and summary metadata.
    """
    by_block: Dict[int, List[int]] = defaultdict(list)
    for pair_id in universe:
        meta = metadata[int(pair_id)]
        by_block[meta.block_idx].append(int(pair_id))

    selected = set()
    block_floor_counts: Dict[int, int] = {}
    for block_idx in range(num_blocks):
        block_candidates = [pair_id for pair_id in by_block.get(block_idx, []) if pair_id not in selected]
        if not block_candidates:
            block_floor_counts[block_idx] = 0
            continue
        take = min(int(min_pairs_per_block), len(block_candidates), max(0, k_pairs - len(selected)))
        block_floor_counts[block_idx] = take
        for _ in range(take):
            candidates = [pair_id for pair_id in block_candidates if pair_id not in selected]
            if not candidates:
                break
            weights = [
                _importance_sampling_weight(
                    pair_id,
                    pair_weights,
                    gamma=gamma,
                    epsilon=epsilon,
                )
                for pair_id in candidates
            ]
            selected.add(_weighted_choice(rng, candidates, weights))

    while len(selected) < k_pairs:
        candidates = [int(pair_id) for pair_id in universe if int(pair_id) not in selected]
        if not candidates:
            break
        weights = [
            _importance_sampling_weight(
                pair_id,
                pair_weights,
                gamma=gamma,
                epsilon=epsilon,
            )
            for pair_id in candidates
        ]
        selected.add(_weighted_choice(rng, candidates, weights))

    solution = _repair_to_k(rng=rng, solution=sorted(selected), universe=universe, k_pairs=k_pairs)
    final_counts = _counts_for_solution(solution, metadata)
    used_weights = [float(pair_weights.get(int(pair_id), 0.0)) for pair_id in solution]
    summary = {
        "profile": "importance",
        "importance_gamma": float(gamma),
        "importance_epsilon": float(epsilon),
        "min_pairs_per_block": int(min_pairs_per_block),
        "block_floor_counts": block_floor_counts,
        "selected_importance_stats": {
            "min": min(used_weights) if used_weights else 0.0,
            "max": max(used_weights) if used_weights else 0.0,
            "mean": sum(used_weights) / len(used_weights) if used_weights else 0.0,
        },
        "final_counts": {key: dict(counter) for key, counter in final_counts.items()},
    }
    return solution, summary


def build_initial_population(
    *,
    strategy: str,
    seed: int,
    population_size: int,
    universe: Sequence[int],
    num_blocks: int,
    num_steps: int,
    k_pairs: int,
    task_name: Optional[str] = None,
    profile: str = "task",
    random_fraction: float = 0.30,
    min_pairs_per_block: int = 4,
    random_tail_min: float = 0.0,
    random_tail_max: float = 0.0,
    pair_weights: Optional[Dict[int, float]] = None,
    importance_gamma: float = 1.0,
    importance_epsilon: float = 0.005,
) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Build the initial population for EVO evolutionary search.

    Args:
        strategy (str): Initialization strategy: ``"random"``, ``"structure"``,
            or ``"importance"``.
        seed (int): Random seed for reproducible population construction.
        population_size (int): Number of candidate schedules to generate.
        universe (Sequence[int]): Feasible flattened block–timestep positions.
        num_blocks (int): Number of cacheable residual branches.
        num_steps (int): Number of denoising timesteps.
        k_pairs (int): Fixed refresh budget per schedule.
        task_name (Optional[str]): Benchmark task name.
        profile (str): Structural profile name for structure-guided sampling.
        random_fraction (float): Fraction of the population initialized with
            fully random schedules.
        min_pairs_per_block (int): Minimum refresh positions assigned per block
            during guided initialization.
        random_tail_min (float): Lower bound on random replacement ratio for
            structure-guided schedules.
        random_tail_max (float): Upper bound on random replacement ratio for
            structure-guided schedules.
        pair_weights (Optional[Dict[int, float]]): Activation-dissimilarity
            weights used by importance-guided initialization.
        importance_gamma (float): Exponent applied to importance weights.
        importance_epsilon (float): Additive floor for importance sampling.

    Returns:
        Tuple[List[List[int]], Dict[str, Any]]: Initial population and metadata
        describing the sampling sources and priors.
    """
    if strategy not in {"random", "structure", "importance"}:
        raise ValueError(f"unknown initialization strategy: {strategy}")
    if population_size <= 0:
        return [], {"strategy": strategy, "seed": int(seed)}
    if len(universe) < k_pairs:
        raise ValueError(f"solution universe has {len(universe)} pairs, cannot sample K={k_pairs}")
    if strategy == "importance" and not pair_weights:
        raise ValueError("importance initialization requires pair_weights")

    rng = random.Random(int(seed))
    metadata = build_pair_metadata(
        num_blocks=num_blocks,
        num_steps=num_steps,
        solution_universe=universe,
    )
    if len(metadata) != len(set(int(pair_id) for pair_id in universe)):
        raise ValueError("solution universe contains pair ids outside num_blocks x num_steps")

    population: List[List[int]] = []
    signatures = set()
    per_individual_summaries: List[Dict[str, Any]] = []

    random_count = population_size
    if strategy in {"structure", "importance"}:
        random_count = int(round(max(0.0, min(1.0, random_fraction)) * population_size))
        random_count = min(population_size, max(0, random_count))

    def add_solution(solution: Sequence[int], source: str, summary: Optional[Dict[str, Any]] = None) -> bool:
        """Insert a unique repaired schedule into the population.

        Args:
            solution (Sequence[int]): Candidate flattened lattice positions.
            source (str): Label describing how the candidate was generated.
            summary (Optional[Dict[str, Any]]): Optional per-candidate metadata.

        Returns:
            bool: ``True`` if the repaired candidate was added, otherwise
            ``False`` when it duplicated an existing schedule.
        """
        repaired = _repair_to_k(rng=rng, solution=solution, universe=universe, k_pairs=k_pairs)
        signature = tuple(repaired)
        if signature in signatures:
            return False
        population.append(repaired)
        signatures.add(signature)
        per_individual_summaries.append({"source": source, **(summary or {})})
        return True

    attempts = 0
    while len(population) < random_count and attempts < population_size * 64:
        attempts += 1
        add_solution(rng.sample(list(universe), k_pairs), "random")

    attempts = 0
    while strategy == "structure" and len(population) < population_size and attempts < population_size * 128:
        attempts += 1
        tail_ratio = rng.uniform(float(random_tail_min), float(random_tail_max))
        solution, summary = sample_structure_solution(
            rng=rng,
            universe=universe,
            metadata=metadata,
            k_pairs=k_pairs,
            num_blocks=num_blocks,
            task_name=task_name,
            profile=profile,
            min_pairs_per_block=min_pairs_per_block,
            random_tail_ratio=tail_ratio,
        )
        add_solution(solution, "structure", summary)

    attempts = 0
    while strategy == "importance" and len(population) < population_size and attempts < population_size * 128:
        attempts += 1
        solution, summary = sample_importance_solution(
            rng=rng,
            universe=universe,
            metadata=metadata,
            pair_weights=pair_weights or {},
            k_pairs=k_pairs,
            num_blocks=num_blocks,
            min_pairs_per_block=min_pairs_per_block,
            gamma=importance_gamma,
            epsilon=importance_epsilon,
        )
        add_solution(solution, "importance", summary)

    attempts = 0
    while len(population) < population_size and attempts < population_size * 64:
        attempts += 1
        add_solution(rng.sample(list(universe), k_pairs), "random_fallback")

    if len(population) != population_size:
        raise RuntimeError(
            f"failed to build unique initial population: {len(population)} / {population_size}"
        )

    source_counts = Counter(item["source"] for item in per_individual_summaries)
    meta = {
        "strategy": strategy,
        "seed": int(seed),
        "population_size": int(population_size),
        "k_pairs": int(k_pairs),
        "universe_size": len(universe),
        "profile": profile,
        "random_fraction": float(random_fraction),
        "random_count": int(random_count),
        "min_pairs_per_block": int(min_pairs_per_block),
        "random_tail_min": float(random_tail_min),
        "random_tail_max": float(random_tail_max),
        "importance_gamma": float(importance_gamma),
        "importance_epsilon": float(importance_epsilon),
        "source_counts": dict(source_counts),
        "structure_profile_centers": _profile_for_task(task_name, profile),
    }
    if pair_weights:
        values = [float(pair_weights.get(int(pair_id), 0.0)) for pair_id in universe]
        meta["importance_weight_stats"] = {
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "mean": sum(values) / len(values) if values else 0.0,
        }
    return population, meta
