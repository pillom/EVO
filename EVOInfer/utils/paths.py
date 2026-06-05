"""Default checkpoint paths used by EVO reproduction scripts.

The command-line entry points accept ``--checkpoint auto`` so experiments can be
run by task name. These helpers keep that task-to-checkpoint mapping in one
place for prior preparation, offline schedule search, and rollout evaluation.
"""


TASK_CHECKPOINT_PATHS = {
    # Native low-dimensional tasks.
    "kitchen": "checkpoint/low_dim/kitchen/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "pusht": "checkpoint/pusht/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "blockpush": "checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",

    # Image-based Robomimic tasks.
    "can_mh": "checkpoint/can_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "can_ph": "checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "lift_mh": "checkpoint/lift_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "lift_ph": "checkpoint/lift_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "square_mh": "checkpoint/square_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "square_ph": "checkpoint/square_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "transport_mh": "checkpoint/transport_mh/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "transport_ph": "checkpoint/transport_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
    "tool_hang_ph": "checkpoint/tool_hang_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt",
}


def get_checkpoint_path(task_name):
    """Return the default checkpoint path for a benchmark task."""
    return TASK_CHECKPOINT_PATHS.get(task_name)


def get_all_available_tasks():
    """Return task names that support ``--checkpoint auto``."""
    return list(TASK_CHECKPOINT_PATHS.keys())
