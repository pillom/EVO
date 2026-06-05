#!/usr/bin/env python3

"""Collect residual activations for EVO redundancy-aware initialization.

EVO uses activation dissimilarity as a lightweight prior when constructing the
initial population for offline cache-schedule search. This module runs a short
uncached Diffusion Policy rollout, records cacheable residual branches across
denoising steps, and saves the tensors used to estimate block–timestep
dissimilarity.
"""

import os
import sys
import logging
import numpy as np
import torch
import click
import pickle
import dill
import hydra
from pathlib import Path
from collections import defaultdict
from omegaconf import OmegaConf

# Make repository-local imports work when launched as a module.
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiffusionFeatureHookTuple:
    """Forward hook that stores tensor outputs from cacheable residual branches.

    Args:
        module: PyTorch module whose forward outputs should be collected.

    Returns:
        None.
    """

    def __init__(self, module):
        """Register the hook on a module.

        Args:
            module: PyTorch module to observe during policy inference.

        Returns:
            None.
        """
        self.features = []
        self.handle = module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        """Store outputs emitted by the hooked module.

        Args:
            module: Hooked PyTorch module.
            inputs: Positional inputs received by the module.
            output: Module output, either a tensor or a tuple/list containing
                tensors.

        Returns:
            None.
        """
        if isinstance(output, (tuple, list)):
            for item in output:
                if hasattr(item, "detach"):
                    self.features.append(item)
        else:
            self.features.append(output)

    def close(self):
        """Remove the PyTorch hook.

        Args:
            None.

        Returns:
            None.
        """
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

def collect_block_activations_for_task(
    checkpoint,
    task_name,
    output_dir,
    device='cuda:0'
):
    """Collect residual activations across rollout steps for one task.
    
    Args:
        checkpoint (str): Path to the pretrained Diffusion Policy checkpoint.
        task_name (str): Benchmark task name, such as ``"kitchen"`` or
            ``"can_ph"``.
        output_dir (str): Directory for activation and diagnostic artifacts.
        device (str): Torch device used for the short uncached rollout.
    
    Returns:
        Optional[Dict[int, Dict[str, List[torch.Tensor]]]]: Rollout-step to
        block-name to denoising activations, or ``None`` if collection fails.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"=" * 60)
    logger.info("Collecting block activations for task: %s", task_name)
    logger.info(f"=" * 60)
    
    # Run a rollout with forward hooks installed.
    rollout_activations = run_rollout_with_activation_hooks(
        checkpoint, output_dir, device
    )
    
    if rollout_activations:
        # Persist tensors used to build the redundancy-aware initialization prior.
        activations_path = output_dir / 'block_activations.pkl'
        with open(activations_path, 'wb') as f:
            pickle.dump(rollout_activations, f)
        
        logger.info("Saved activations to: %s", activations_path)
        
        # Store compact diagnostics for checking activation collection quality.
        stats = analyze_activation_statistics(rollout_activations)
        stats_path = output_dir / 'activation_stats.pkl'
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        
        logger.info("Saved activation statistics to: %s", stats_path)
    
    logger.info("Finished activation collection for task: %s", task_name)
    return rollout_activations

def run_rollout_with_activation_hooks(
    checkpoint_path, output_dir, device
):
    """Run one rollout while recording transformer residual activations.

    Args:
        checkpoint_path (str): Path to the pretrained Diffusion Policy
            checkpoint.
        output_dir (str): Directory used by the environment runner.
        device (str): Torch device used for the rollout.

    Returns:
        Optional[Dict[int, Dict[str, List[torch.Tensor]]]]: Rollout-step indexed
        activation dictionary, or ``None`` if collection fails.
    """
    rollout_activations = defaultdict(lambda: defaultdict(list))
    
    # Restore the caller's working directory after rollout construction.
    original_cwd = os.getcwd()
    
    try:
        # Diffusion Policy configs expect to run from the repository root.
        os.chdir(root_dir)
        logger.info("Working directory set to repository root: %s", root_dir)
        
        # Load checkpoint.
        payload = torch.load(open(checkpoint_path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        
        # Recreate the Diffusion Policy workspace.
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=output_dir)
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        
        # Select the policy.
        policy = workspace.model
        if hasattr(cfg.training, 'use_ema') and cfg.training.use_ema:
            policy = workspace.ema_model
        policy.to(device)
        policy.eval()
        
        logger.info("Policy loaded and set to eval mode.")
        
        # Register hooks.
        feature_hooks = {}
        
        # Locate Transformer decoder layers.
        if hasattr(policy, 'model') and hasattr(policy.model, 'decoder'):
            decoder = policy.model.decoder
            
            # Hook dropout outputs inside decoder layers.
            if hasattr(decoder, 'layers'):
                for layer_idx, layer in enumerate(decoder.layers):
                    for module_name, module in layer.named_modules():
                        if 'dropout' in module_name:
                            full_name = f"decoder.layers.{layer_idx}.{module_name}"
                            hook = DiffusionFeatureHookTuple(module)
                            feature_hooks[full_name] = hook
        
        if not feature_hooks:
            logger.warning("No activation hooks were registered.")
            return None
        
        logger.info("Registered %d activation hooks.", len(feature_hooks))
        
        # Build an environment runner for a short activation rollout.
        cfg = workspace.cfg
        env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
        
        # Keep the activation rollout small and deterministic.
        env_runner_cfg['n_train'] = 1
        env_runner_cfg['n_train_vis'] = 0  
        env_runner_cfg['n_test'] = 0
        env_runner_cfg['n_test_vis'] = 0
        env_runner_cfg['n_envs'] = 1
        
        # Instantiate the environment runner.
        env_runner = hydra.utils.instantiate(
            env_runner_cfg,
            output_dir=str(output_dir)
        )
        
        # Wrap the runner so each policy call stores hook outputs.
        class ActivationCollectingEnvRunner:
            """Environment-runner adapter that snapshots hook outputs per action.

            Args:
                base_runner: Original Diffusion Policy environment runner.
                feature_hooks (Dict[str, DiffusionFeatureHookTuple]): Hooks
                    keyed by raw module name.

            Returns:
                None.
            """

            def __init__(self, base_runner, feature_hooks):
                """Create an activation-collecting runner wrapper.

                Args:
                    base_runner: Environment runner used for the short rollout.
                    feature_hooks (Dict[str, DiffusionFeatureHookTuple]): Hooks
                        that store denoising-step activations.

                Returns:
                    None.
                """
                self.base_runner = base_runner
                self.feature_hooks = feature_hooks
                self.rollout_activations = defaultdict(lambda: defaultdict(list))
                self.step_count = 0
            
            def run(self, policy):
                """Run the base environment while collecting policy activations.

                Args:
                    policy: Diffusion Policy object whose ``predict_action``
                        method will be temporarily wrapped.

                Returns:
                    object: Result returned by the base environment runner.
                """
                # Hook into the policy's predict_action method.
                original_predict_action = policy.predict_action
                
                def hooked_predict_action(*args, **kwargs):
                    """Collect hook outputs from one policy action query.

                    Args:
                        *args: Positional arguments forwarded to the original
                            policy.
                        **kwargs: Keyword arguments forwarded to the original
                            policy.

                    Returns:
                        object: Action prediction returned by the original
                        policy.
                    """
                    # Clear features from the previous predict_action call.
                    for hook in self.feature_hooks.values():
                        hook.features.clear()
                    
                    # Execute the original predict_action.
                    result = original_predict_action(*args, **kwargs)
                    
                    # Store block activations for this rollout step.
                    step_activations = {}
                    for module_name, hook in self.feature_hooks.items():
                        # Preserve all denoising-step activations.
                        all_features = []
                        for feature in hook.features:
                            if hasattr(feature, 'clone'):
                                all_features.append(feature.clone().detach().cpu())
                            else:
                                all_features.append(feature)
                        step_activations[module_name] = all_features
                    
                    self.rollout_activations[self.step_count] = step_activations
                    self.step_count += 1
                    
                    return result
                
                # Temporarily replace policy.predict_action.
                policy.predict_action = hooked_predict_action
                
                try:
                    # Run rollout.
                    logger.info("Running activation rollout...")
                    result = self.base_runner.run(policy)
                    logger.info("Rollout collected %d policy steps.", self.step_count)
                    return result
                finally:
                    # Restore original predict_action.
                    policy.predict_action = original_predict_action
        
        # Run the wrapped environment runner.
        collecting_runner = ActivationCollectingEnvRunner(
            env_runner, feature_hooks
        )
        
        # Run rollout.
        log_data = collecting_runner.run(policy)
        
        # Return collected activations.
        return dict(collecting_runner.rollout_activations)
        
    except Exception as e:
        logger.error("Activation rollout failed: %s", e)
        import traceback
        traceback.print_exc()
        return None
    finally:
        for hook in locals().get("feature_hooks", {}).values():
            try:
                hook.close()
            except Exception:
                pass
        # Restore original cwd.
        os.chdir(original_cwd)

def analyze_activation_statistics(rollout_activations):
    """Compute lightweight diagnostics for collected activations.

    Args:
        rollout_activations (Dict): Rollout-step indexed activation dictionary
            produced by ``run_rollout_with_activation_hooks``.

    Returns:
        Dict[str, Any]: Counts, average denoising-step coverage, and activation
        shapes for each hooked module.
    """
    stats = {
        'total_rollout_steps': len(rollout_activations),
        'modules': {},
        'denoise_steps_per_rollout': {}
    }
    
    # Summarize per-module activation counts and shapes.
    if rollout_activations:
        first_step_data = list(rollout_activations.values())[0]
        
        for module_name in first_step_data.keys():
            module_stats = {
                'total_appearances': 0,
                'avg_denoise_steps': 0,
                'activation_shapes': []
            }
            
            total_denoise_steps = 0
            for step, step_data in rollout_activations.items():
                if module_name in step_data:
                    module_stats['total_appearances'] += 1
                    denoise_steps_count = len(step_data[module_name])
                    total_denoise_steps += denoise_steps_count
                    
                    # Record activation shapes.
                    for activation in step_data[module_name]:
                        if hasattr(activation, 'shape'):
                            module_stats['activation_shapes'].append(activation.shape)
            
            if module_stats['total_appearances'] > 0:
                module_stats['avg_denoise_steps'] = total_denoise_steps / module_stats['total_appearances']
            
            stats['modules'][module_name] = module_stats
        
        # Record per-rollout denoising-step counts.
        for step, step_data in rollout_activations.items():
            denoise_counts = {}
            for module_name, activations in step_data.items():
                denoise_counts[module_name] = len(activations)
            stats['denoise_steps_per_rollout'][step] = denoise_counts
    
    return stats

@click.command(help="Collect residual activations from uncached Diffusion Policy rollouts.")
@click.option('-c', '--checkpoint', required=True, help='Path to a pretrained Diffusion Policy checkpoint.')
@click.option('-t', '--task_name', required=True, help='Benchmark task name used to label activation artifacts.')
@click.option('-o', '--output_dir', default=None, help='Directory for activation artifacts. Defaults to activations/{task_name}.')
@click.option('-d', '--device', default='cuda:0', help='Torch device used for the activation rollout.')
def main(checkpoint, task_name, output_dir, device):
    """CLI entry point for collecting EVO activation artifacts.

    Args:
        checkpoint (str): Path to a pretrained Diffusion Policy checkpoint.
        task_name (str): Benchmark task name used to label output artifacts.
        output_dir (Optional[str]): Directory where activation artifacts are
            saved. Defaults to ``activations/{task_name}``.
        device (str): Torch device used for rollout collection.

    Returns:
        None.
    """
    if output_dir is None:
        output_dir = f"activations/{task_name}"
    
    logger.info("Task: %s", task_name)
    logger.info("Checkpoint: %s", checkpoint)
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)
    
    # Collect activations.
    results = collect_block_activations_for_task(
        checkpoint=checkpoint,
        task_name=task_name,
        output_dir=output_dir,
        device=device
    )
    
    logger.info("=" * 60)
    logger.info("Activation collection completed.")
    logger.info("Task: %s", task_name)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 60)

if __name__ == '__main__':
    main()
