"""Evaluate baseline and EVO-accelerated Diffusion Policy inference.

This script loads a pretrained transformer-based Diffusion Policy checkpoint,
optionally installs an offline-optimized EVO residual-cache schedule such as
``evo_schedule.json``, and reports closed-loop rollout performance together
with inference-cost diagnostics.
"""
import sys
import os
import pathlib
import click
import hydra
import torch
import dill
import numpy as np
import time
import json
import logging
from omegaconf import OmegaConf
from copy import deepcopy

from thop import profile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from EVOInfer.acceleration.evo_cache_wrapper import FastDiffusionPolicyEVO
from diffusion_policy.common.pytorch_util import dict_apply
from EVOInfer.utils.paths import get_checkpoint_path, get_all_available_tasks

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("eval_evo_script")

@click.command(help="Evaluate original or EVO-accelerated Diffusion Policy inference.")
@click.option('-c', '--checkpoint', default=None, help='Explicit pretrained Diffusion Policy checkpoint path.')
@click.option('-t', '--task', default=None, help=f'Benchmark task name used to resolve --checkpoint auto. Available: {", ".join(get_all_available_tasks())}.')
@click.option('-o', '--output_dir', required=True, help='Directory for rollout metrics, cache configuration, and benchmark outputs.')
@click.option('-d', '--device', default='cuda:0', help='Torch device used for policy inference and benchmarking.')
@click.option('--pairs_path', default=None, help='JSON file containing the EVO offline schedule, for example evo_schedule.json.')
@click.option('--num_inference_steps', default=100, type=int, help='Number of denoising steps used by the diffusion sampler.')
@click.option('--cache_mode', default='evo', type=click.Choice(['original', 'evo']), help='Evaluation mode: original baseline or EVO residual-cache reuse.')
@click.option('--eval_seed', default=None, type=int, help='Optional test_start_seed override for closed-loop rollout evaluation.')
@click.option('--n_test', default=None, type=int, help='Optional number of test rollouts for closed-loop evaluation.')
@click.option('--skip_benchmark', is_flag=True, help='Skip both THOP FLOPs profiling and wall-clock speed benchmarking.')
@click.option('--skip_flops', is_flag=True, help='Skip THOP FLOPs profiling while still allowing wall-clock speed benchmarking.')
@click.option('--skip_speed', is_flag=True, help='Skip wall-clock speed benchmarking while keeping FLOPs profiling unless --skip_benchmark is set.')
@click.option('--benchmark_only', is_flag=True, help='Run only dummy-input speed/FLOPs diagnostics and skip environment rollouts.')
@click.option('--skip_video', is_flag=True, help='Disable video rendering during rollout evaluation.')

def main(checkpoint, task, output_dir, device, pairs_path,
         num_inference_steps, cache_mode, eval_seed, n_test,
         skip_benchmark, skip_flops, skip_speed, benchmark_only, skip_video):
    """Evaluate a baseline or EVO-accelerated policy."""
    if checkpoint is None and task is None:
        raise click.UsageError("Either --checkpoint or --task must be specified")
    
    if checkpoint is None:
        checkpoint = get_checkpoint_path(task)
        if checkpoint is None:
            available_tasks = ", ".join(get_all_available_tasks())
            raise click.UsageError(f"Unknown task '{task}'. Available tasks: {available_tasks}")
        logger.info(f"Using task '{task}' with checkpoint: {checkpoint}")
    else:
        logger.info(f"Using direct checkpoint path: {checkpoint}")
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    logger.info(f"Configuration loaded: {cfg._target_}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    logger.info("Workspace loaded successfully")

    original_policy = workspace.model
    original_policy.to(device)
    original_policy.eval()

    has_model = hasattr(original_policy, 'model')
    logger.info(f"Policy type: {type(original_policy).__name__}")
    if has_model:
        logger.info(f"Model type: {type(original_policy.model).__name__}")

    actions_per_inference = original_policy.n_action_steps
    logger.info(f"Number of action frames per inference: {actions_per_inference}")

    logger.info("Creating policy copy for cache acceleration...")
    fast_policy = deepcopy(original_policy)

    if cache_mode == 'evo':
        logger.info(f"Applying EVO cache acceleration, pairs_path={pairs_path}")
        fast_policy = FastDiffusionPolicyEVO.apply_cache(
            policy=fast_policy,
            cache_mode='evo',
            pairs_path=pairs_path,
            num_inference_steps=num_inference_steps
        )
    else:
        logger.info(f"Using original policy (no caching)")
        fast_policy = FastDiffusionPolicyEVO.apply_cache(
            policy=fast_policy,
            cache_mode='original'
        )
    
    B = 1
    To = cfg.n_obs_steps
    
    if hasattr(cfg, 'shape_meta'):
        logger.info("Detected image model configuration")
        obs_dict = {}
        for key, shape in cfg.shape_meta['obs'].items():
            tensor_shape = [B, To] + list(shape['shape'])
            obs_dict[key] = torch.zeros(tensor_shape, device=device, dtype=torch.float32)
    else:
        logger.info("Detected low-dimensional model configuration")
        obs_dim = cfg.obs_dim if hasattr(cfg, 'obs_dim') else None
        
        if obs_dim is None and hasattr(cfg, 'task'):
            task_cfg = OmegaConf.to_container(cfg.task, resolve=True)
            if 'obs_dim' in task_cfg:
                obs_dim = task_cfg['obs_dim']
                logger.info(f"Using obs_dim from configuration: {obs_dim}")
        
        obs_dict = {
            'obs': torch.zeros((B, To, obs_dim), device=device, dtype=torch.float32)
        }
        
        if hasattr(cfg, 'use_past_action') and cfg.use_past_action:
            action_dim = cfg.action_dim
            obs_dict['past_action'] = torch.zeros((B, To, action_dim), device=device, dtype=torch.float32)

    logger.info(f"Created input dictionary with keys: {list(obs_dict.keys())}")
    for key, tensor in obs_dict.items():
        logger.info(f"  {key}: shape={tensor.shape}")
    
    logger.info("Warming up...")
    try:
        with torch.no_grad():
            original_policy.predict_action(obs_dict)
            fast_policy.predict_action(obs_dict)
        logger.info("Warmup complete")
    except Exception as e:
        logger.error(f"Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        return
    
    flops_value = 0.0
    avg_original = None
    frequency_original = None
    avg_fast = None
    frequency_fast = None
    speedup_time = None

    if skip_benchmark:
        logger.info("Skipping thop FLOPs profiling and wall-clock speed benchmark")
    elif skip_flops:
        logger.info("Skipping thop FLOPs profiling; wall-clock speed benchmark will be kept")
        if skip_speed:
            logger.info("Skipping wall-clock speed benchmark")
        else:
            logger.info(f"\n=== Performance Testing (action frames/inference: {actions_per_inference}) ===")
            num_trials = 10

            logger.info("\n----- Original Policy -----")
            original_durations = []

            for i in range(num_trials):
                torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    original_policy.predict_action(obs_dict)

                torch.cuda.synchronize()
                duration = time.time() - start_time
                original_durations.append(duration)
                logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")

            avg_original = np.mean(original_durations)
            frequency_original = actions_per_inference / avg_original
            logger.info(f"Average time: {avg_original:.4f} seconds")
            logger.info(f"Action frequency: {frequency_original:.2f} actions/second")

            logger.info("\n----- Cached Policy -----")
            fast_durations = []

            for i in range(num_trials):
                torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    fast_policy.predict_action(obs_dict)

                torch.cuda.synchronize()
                duration = time.time() - start_time
                fast_durations.append(duration)
                logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")

            avg_fast = np.mean(fast_durations)
            frequency_fast = actions_per_inference / avg_fast
            speedup_time = avg_original / avg_fast

            logger.info(f"Average time: {avg_fast:.4f} seconds")
            logger.info(f"Action frequency: {frequency_fast:.2f} actions/second")
            logger.info(f"Speedup: {speedup_time:.2f}x")
    else:
        logger.info("\nComputing policy FLOPs using thop...")
        try:
            obs_dict_copy = {}
            for key, value in obs_dict.items():
                obs_dict_copy[key] = value.clone()

            with torch.no_grad():
                nobs = fast_policy.normalizer.normalize(obs_dict_copy)
                value = next(iter(nobs.values()))
                B, To = value.shape[:2]

                if hasattr(fast_policy, 'action_dim'):
                    Da = fast_policy.action_dim
                elif hasattr(cfg, 'action_dim'):
                    Da = cfg.action_dim
                else:
                    Da = cfg.task.action_dim

                if hasattr(fast_policy, 'obs_feature_dim'):
                    Do = fast_policy.obs_feature_dim
                else:
                    if hasattr(cfg, 'obs_dim'):
                        Do = cfg.obs_dim
                    else:
                        Do = cfg.task.obs_dim

                logger.info(f"Action dimension: {Da}, Observation feature dimension: {Do}")

                device = fast_policy.device
                dtype = torch.float32
                timestep = torch.zeros(B, dtype=torch.long, device=device)
                cond = None
                sample = None

                action_dim = fast_policy.action_dim
                obs_feature_dim = fast_policy.obs_dim if hasattr(fast_policy, 'obs_dim') else Do

                logger.info(f"Action dimension: {action_dim}, Observation feature dimension: {obs_feature_dim}")

                if hasattr(fast_policy, 'obs_encoder'):
                    if fast_policy.obs_as_cond:
                        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                        nobs_features = fast_policy.obs_encoder(this_nobs)
                        cond = nobs_features.reshape(B, To, -1)
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                    else:
                        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
                        nobs_features = fast_policy.obs_encoder(this_nobs)
                        nobs_features = nobs_features.reshape(B, To, -1)
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                        sample[:,:To,action_dim:] = nobs_features
                else:
                    if hasattr(fast_policy, 'obs_as_cond') and fast_policy.obs_as_cond:
                        cond = obs_dict['obs'][:,:To]
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim), device=device, dtype=dtype)
                        if hasattr(fast_policy, 'pred_action_steps_only') and fast_policy.pred_action_steps_only:
                            sample = torch.zeros(size=(B, fast_policy.n_action_steps, action_dim), device=device, dtype=dtype)
                    else:
                        sample = torch.zeros(size=(B, fast_policy.horizon, action_dim+obs_feature_dim), device=device, dtype=dtype)
                        sample[:,:To,action_dim:] = obs_dict['obs'][:,:To]

                try:
                    fast_policy.eval()
                    macs, params = profile(fast_policy.model, inputs=(sample, timestep, cond), verbose=False)
                    flops_value = macs * 2

                    logger.info(f"Model parameter count: {params/1e6:.2f} M")
                    logger.info(f"Model MACs: {macs/1e9:.4f} G")
                    logger.info(f"Model FLOPs: {flops_value/1e9:.4f} G")
                except Exception as e:
                    logger.error(f"Error computing FLOPs: {e}")
                    flops_value = 0.0
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            logger.error(f"Error computing FLOPs: {e}")
            flops_value = 0.0
            import traceback
            traceback.print_exc()

        if skip_speed:
            logger.info("Skipping wall-clock speed benchmark; FLOPs profiling was kept")
        else:
            logger.info(f"\n=== Performance Testing (action frames/inference: {actions_per_inference}) ===")
            num_trials = 10

            logger.info("\n----- Original Policy -----")
            original_durations = []

            for i in range(num_trials):
                torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    original_policy.predict_action(obs_dict)

                torch.cuda.synchronize()
                duration = time.time() - start_time
                original_durations.append(duration)
                logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")

            avg_original = np.mean(original_durations)
            frequency_original = actions_per_inference / avg_original
            logger.info(f"Average time: {avg_original:.4f} seconds")
            logger.info(f"Action frequency: {frequency_original:.2f} actions/second")

            logger.info("\n----- Cached Policy -----")
            fast_durations = []

            for i in range(num_trials):
                torch.cuda.synchronize()
                start_time = time.time()

                with torch.no_grad():
                    fast_policy.predict_action(obs_dict)

                torch.cuda.synchronize()
                duration = time.time() - start_time
                fast_durations.append(duration)
                logger.info(f"Run {i+1}/{num_trials}: {duration:.4f} seconds")

            avg_fast = np.mean(fast_durations)
            frequency_fast = actions_per_inference / avg_fast
            speedup_time = avg_original / avg_fast

            logger.info(f"Average time: {avg_fast:.4f} seconds")
            logger.info(f"Action frequency: {frequency_fast:.2f} actions/second")
            logger.info(f"Speedup: {speedup_time:.2f}x")
    
    cache_config = {
        "device": str(device),
        "mode": cache_mode,
        "actions_per_inference": int(actions_per_inference),
        "num_inference_steps": int(num_inference_steps),
        "eval_seed": int(eval_seed) if eval_seed is not None else None,
        "n_test": int(n_test) if n_test is not None else None,
        "skip_benchmark": bool(skip_benchmark),
        "skip_flops": bool(skip_flops),
        "skip_speed": bool(skip_speed),
        "benchmark_only": bool(benchmark_only),
    }

    if cache_mode == 'evo':
        cache_config.update({
            "pairs_path": pairs_path
        })
    
    benchmark_results = {
        **cache_config,
        "original": {
            "avg_time": None if avg_original is None else float(avg_original),
            "frequency": None if frequency_original is None else float(frequency_original),
        },
        "fast": {
            "avg_time": None if avg_fast is None else float(avg_fast),
            "frequency": None if frequency_fast is None else float(frequency_fast),
        },
        "speedup": None if speedup_time is None else float(speedup_time),
        "flops": float(flops_value),
        "config": OmegaConf.to_container(cfg, resolve=True)
    }
    
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(benchmark_results, f, indent=2)
    
    logger.info(f"Benchmark results saved to {output_dir}/benchmark_results.json")
    if benchmark_only:
        eval_results_path = os.path.join(output_dir, 'eval_results.json')
        if os.path.exists(eval_results_path):
            with open(eval_results_path, 'r') as f:
                existing_eval_results = json.load(f)
            existing_eval_results.update({
                "speedup": None if speedup_time is None else float(speedup_time),
                "flops": existing_eval_results.get("flops", 0.0) if skip_flops else float(flops_value),
                "skip_benchmark": bool(skip_benchmark),
                "skip_flops": bool(skip_flops),
                "skip_speed": bool(skip_speed),
                "benchmark_only": bool(benchmark_only),
                "benchmark_only_updated": True,
            })
            with open(eval_results_path, 'w') as f:
                json.dump(existing_eval_results, f, indent=2)
            logger.info(f"Updated existing evaluation results with benchmark fields: {eval_results_path}")
        logger.info("Benchmark-only mode enabled; skipping environment evaluation.")
        return
    
    env_runner_cfg = OmegaConf.to_container(cfg.task.env_runner, resolve=True)
    if eval_seed is not None:
        env_runner_cfg['test_start_seed'] = int(eval_seed)
        logger.info(f"Override env_runner test_start_seed={eval_seed}")
    if n_test is not None:
        env_runner_cfg['n_test'] = int(n_test)
        logger.info(f"Override env_runner n_test={n_test}")
    if skip_video:
        env_runner_cfg['n_train_vis'] = 0
        env_runner_cfg['n_test_vis'] = 0
        logger.info("Skip video rendering")
    
    try:
        env_runner = hydra.utils.instantiate(env_runner_cfg, output_dir=output_dir)
    except Exception as e:
        logger.error(f"Error instantiating environment runner: {e}")
        try:
            import inspect
            from diffusion_policy.env_runner.kitchen_lowdim_runner import KitchenLowdimRunner
            from diffusion_policy.env_runner.block_push_lowdim_runner import BlockPushLowdimRunner
            
            runner_class = None
            if "KitchenLowdimRunner" in str(env_runner_cfg):
                runner_class = KitchenLowdimRunner
                logger.info("Detected KitchenLowdimRunner")
            elif "BlockPushLowdimRunner" in str(env_runner_cfg):
                runner_class = BlockPushLowdimRunner
                logger.info("Detected BlockPushLowdimRunner")
            
            if runner_class:
                sig = inspect.signature(runner_class.__init__)
                logger.info(f"Runner initialization parameters: {list(sig.parameters.keys())}")
                
                required_params = set(sig.parameters.keys()) - {'self'}
                for param in list(env_runner_cfg.keys()):
                    if param not in required_params and param != '_target_':
                        logger.info(f"Remove unneeded parameter: {param}")
                        del env_runner_cfg[param]
                
                env_runner_cfg['output_dir'] = output_dir
                env_runner = hydra.utils.instantiate(env_runner_cfg)
            else:
                raise e
        except Exception as inner_e:
            logger.error(f"Attempt to fix environment runner instantiation failed: {inner_e}")
            raise
    
    logger.info("\nRunning environment evaluation with cached policy...")
    fast_runner_log = env_runner.run(fast_policy)
    
    test_mean_score = fast_runner_log.get('test/mean_score', 0.0)
    
    eval_results = {
        "mean_score": float(test_mean_score),
        "speedup": None if speedup_time is None else float(speedup_time),
        "flops": float(flops_value),
        "cache_mode": cache_mode,
        "pairs_path": pairs_path if cache_mode == 'evo' else None,
        "eval_seed": int(eval_seed) if eval_seed is not None else None,
        "n_test": int(n_test) if n_test is not None else None,
        "skip_benchmark": bool(skip_benchmark),
        "skip_flops": bool(skip_flops),
        "skip_speed": bool(skip_speed),
        "benchmark_only": bool(benchmark_only),
        "env_runner_cfg": env_runner_cfg,
    }
    
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    logger.info(f"Evaluation results saved to {output_dir}/eval_results.json")
    if speedup_time is None:
        logger.info(f"Closed-loop rollout performance: {test_mean_score:.4f}, Speedup: skipped")
    else:
        logger.info(f"Closed-loop rollout performance: {test_mean_score:.4f}, Speedup: {speedup_time:.2f}x")
    if flops_value > 0:
        logger.info(f"FLOPs: {flops_value/1e9:.4f} GFLOPs")
    
    json_log = {k: v._path if hasattr(v, '_path') else v for k, v in fast_runner_log.items()}
    out_path = os.path.join(output_dir, 'fast_eval_log.json')
    with open(out_path, 'w') as f:
        json.dump(json_log, f, indent=2, sort_keys=True)
    logger.info(f"Detailed evaluation logs saved to {out_path}")

if __name__ == "__main__":
    main()
