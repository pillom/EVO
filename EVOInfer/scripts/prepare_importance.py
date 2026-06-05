"""Prepare activation-dissimilarity priors for EVO initialization.

EVO uses feature similarity only as a search-time prior. This script collects
residual activations from the uncached Diffusion Policy and converts them into
pair-level scores over the block–timestep lattice. The resulting artifact can
bias part of the initial population toward less redundant positions while the
final schedule is still selected by rollout fitness.
"""

import logging
from pathlib import Path

import click

from EVOInfer.utils.paths import get_all_available_tasks, get_checkpoint_path


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command(help="Prepare activation-dissimilarity priors for EVO schedule search.")
@click.option("--task", required=True, help=f"Benchmark task name. Available: {', '.join(get_all_available_tasks())}.")
@click.option("--checkpoint", default="auto", help="Checkpoint path, or auto to use the default task checkpoint.")
@click.option("--output_dir", required=True, help="Directory for activation and pair-importance artifacts.")
@click.option("--device", default="cuda:0", help="Torch device used to run activation-collection rollouts.")
@click.option("--sample_steps", default=5, type=int, help="Number of rollout steps used to estimate activation dissimilarity.")
@click.option("--activations_path", default=None, help="Optional existing block_activations.pkl file to reuse instead of collecting activations.")
def main(task, checkpoint, output_dir, device, sample_steps, activations_path):
    """Create pair-level priors for redundancy-aware EVO initialization.

    Args:
        task (str): Benchmark task used to resolve the default checkpoint and
            label the output artifacts.
        checkpoint (str): Explicit checkpoint path, or ``"auto"`` to use the
            task-specific default.
        output_dir (str): Directory where activations and pair-importance files
            are written.
        device (str): Torch device used for activation collection.
        sample_steps (int): Number of rollout steps used when estimating
            cross-step activation dissimilarity.
        activations_path (Optional[str]): Existing ``block_activations.pkl`` to
            reuse. When omitted, the script first collects activations from the
            uncached policy.
    """
    from EVOInfer.search.activation_collection import collect_block_activations_for_task
    from EVOInfer.search.importance import (
        compute_cosine_global_pair_importance_matrix,
        load_activations,
        save_pair_importance_results,
    )

    if checkpoint == "auto":
        checkpoint = get_checkpoint_path(task)
        if checkpoint is None:
            raise click.UsageError(f"Unknown task: {task}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if activations_path is None:
        collect_block_activations_for_task(
            checkpoint=checkpoint,
            task_name=task,
            output_dir=str(output_dir),
            device=device,
        )
        activations_path = output_dir / "block_activations.pkl"
    else:
        activations_path = Path(activations_path)

    activations = load_activations(activations_path)
    error_matrix, denoise_step_indices, block_names = compute_cosine_global_pair_importance_matrix(
        activations,
        sample_steps=sample_steps,
    )
    pair_json_path, _ = save_pair_importance_results(
        error_matrix=error_matrix,
        denoise_step_indices=denoise_step_indices,
        block_names=block_names,
        task_name=task,
        checkpoint_path=checkpoint,
        output_dir=str(output_dir),
    )
    logger.info("Pair-importance prior saved to %s", pair_json_path)


if __name__ == "__main__":
    main()
