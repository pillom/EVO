"""Runtime wrapper for deploying offline EVO cache schedules.

EVO fixes the schedule after evolutionary search and keeps the optimizer out of
the inference loop. At test time, selected positions in the block-timestep
lattice evaluate the original residual branch and refresh the cache, while
unselected positions reuse the latest cached residual from the same computation
unit.
"""

import torch
import logging
import os
import types
import json
from typing import List, Tuple, Optional, Dict
from pathlib import Path

logger = logging.getLogger(__name__)


class FastDiffusionPolicyEVO:
    """Factory interface for attaching EVO cache acceleration to a policy."""

    @staticmethod
    def apply_cache(
        policy,
        cache_mode: str = "evo",
        pairs_path: Optional[str] = None,
        num_inference_steps: int = 100,
    ):
        """Install a verified EVO cache schedule on a Diffusion Policy instance.

        Args:
            policy: Diffusion Policy object whose transformer denoiser will be
                wrapped for residual-cache reuse.
            cache_mode (str): ``"evo"`` loads an offline-optimized schedule;
                ``"original"`` returns the policy unchanged.
            pairs_path (Optional[str]): JSON file containing selected
                ``(block_idx, denoise_step)`` positions in the block-timestep
                lattice. Required when ``cache_mode`` is ``"evo"``.
            num_inference_steps (int): Number of reverse denoising steps used by
                the diffusion sampler.

        Returns:
            object: The original policy in ``"original"`` mode, or an
            ``_EVOCacheWrapper`` that delegates policy calls while applying the
            fixed EVO schedule.
        """
        if cache_mode == "original":
            return policy
        if cache_mode != "evo":
            raise ValueError(f"Unknown cache mode: {cache_mode}. Use 'original' or 'evo'.")
        if pairs_path is None:
            raise ValueError("pairs_path is required for EVO cache mode")

        pairs_path = Path(pairs_path)
        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        with pairs_path.open("r", encoding="utf-8") as f:
            pairs_data = json.load(f)

        if isinstance(pairs_data, list):
            pairs = pairs_data
        elif isinstance(pairs_data, dict):
            pairs = (
                pairs_data.get("selected_block_timestep_pairs")
                or pairs_data.get("optimal_ij_pairs")
                or pairs_data.get("pairs")
            )
        else:
            pairs = None
        if pairs is None:
            raise ValueError(f"Invalid pairs file format: {pairs_path}")

        wrapper = _EVOCacheWrapper(policy, num_inference_steps=num_inference_steps)
        wrapper.set_optimal_steps([(int(block_idx), int(step)) for block_idx, step in pairs])
        logger.info("EVO cache applied with %d block-timestep pairs", len(pairs))
        return wrapper


class _EVOCacheWrapper:
    """Executor for fixed EVO schedules over the block-timestep lattice.

    The wrapper intercepts transformer decoder residual branches, refreshes the
    positions selected by the offline schedule, and reuses cached residuals at
    skipped positions. Policy weights, sampler behavior, and the action interface
    are left unchanged.
    """
    
    def __init__(self, policy, num_inference_steps=100):
        """Initialize cache state and install the model-level forward wrapper."""
        self.policy = policy
        self.model = policy.model
        self.num_inference_steps = num_inference_steps
        self.cache = None
        self.cacheable_blocks = []
        self.original_forwards = {}
        self._setup_cache()
        
        # Mirror common policy methods so callers can use the wrapper like the original policy.
        self.device = getattr(policy, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        for attr in ['to', 'eval', 'train', 'named_parameters', 'parameters', 'state_dict']:
            if hasattr(policy, attr) and not hasattr(self, attr):
                setattr(self, attr, getattr(policy, attr))
        
        self._setup_auto_reset_predict_action()
    
    def _setup_auto_reset_predict_action(self):
        """Install a prediction wrapper that resets residual caches per action query."""
        self._original_predict_action = self.policy.predict_action
        self._original_reset = self.policy.reset if hasattr(self.policy, 'reset') else (lambda: None)
        
        def predict_action_with_auto_reset(*args, **kwargs):
            """Run ``predict_action`` with a fresh residual cache."""
            self.reset_cache()
            return self._original_predict_action(*args, **kwargs)
        
        self.predict_action = predict_action_with_auto_reset
    
    def _setup_cache(self):
        """Initialize cache bookkeeping and prepare the transformer wrapper."""
        self.cache = {
            'mode': 'optimal',
            'num_steps': self.num_inference_steps,
            'current_step': -1,
            'block_steps': {},
            'block_cache': {},
            'clone_cache': os.environ.get('EVO_CACHE_CLONE', '0') == '1',
            'cache_hits': {'total_calls': 0, 'cache_hits': 0},
            'steps_lookup': {}
        }
        
        self._find_cacheable_blocks()
        self._wrap_model_forward()
    
    def _find_cacheable_blocks(self):
        """Enumerate self-attention, cross-attention, and feed-forward residual branches."""
        import torch.nn as nn
        
        cacheable_blocks = []
        
        if isinstance(self.model.decoder, nn.TransformerDecoder):
            for i, layer in enumerate(self.model.decoder.layers):
                if isinstance(layer, nn.TransformerDecoderLayer):
                    layer_name = f'decoder.layers.{i}'
                    
                    cacheable_blocks.append((f"{layer_name}_sa_block", layer, 'sa_block', i, 0))
                    cacheable_blocks.append((f"{layer_name}_mha_block", layer, 'mha_block', i, 1))
                    cacheable_blocks.append((f"{layer_name}_ff_block", layer, 'ff_block', i, 2))
        
        cacheable_blocks.sort(key=lambda x: (x[3], x[4]))
        self.cacheable_blocks = cacheable_blocks
        
        logger.info(f"Found {len(cacheable_blocks)} cacheable blocks")
    
    def _wrap_model_forward(self):
        """Wrap ``model.forward`` to advance the denoising-step index."""
        if hasattr(self.model, '_cache_wrapped'):
            return
        
        original_forward = self.model.forward
        
        def forward_with_cache(sample, timestep, cond=None, **kwargs):
            """Update schedule flags before one denoising model call."""
            self.cache['current_step'] += 1
            current_step = self.cache['current_step']
            self._update_cache_flags(current_step)
            return original_forward(sample, timestep, cond, **kwargs)
        
        self.model._original_forward = original_forward
        self.model.forward = forward_with_cache
        self.model._cache_wrapped = True
    
    def _add_cache_to_transformer_layer(self, layer, layer_name):
        """Attach residual-cache refresh and reuse logic to one decoder layer."""
        import torch.nn as nn
        
        if hasattr(layer, '_cache_wrapped'):
            return
        
        original_forward = layer.forward
        self.original_forwards[layer_name] = original_forward
        cache = self.cache
        
        sa_block_key = f"{layer_name}_sa_block"
        mha_block_key = f"{layer_name}_mha_block"
        ff_block_key = f"{layer_name}_ff_block"
        
        # Backfill helper blocks for TransformerDecoderLayer implementations.
        if not hasattr(layer, '_sa_block'):
            def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
                """Evaluate the self-attention residual branch."""
                # PyTorch attention returns (output, weights); EVO caches only the residual output.
                x = self.self_attn(x, x, x, 
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=False)[0]
                return self.dropout1(x)
            layer._sa_block = types.MethodType(_sa_block, layer)
        
        if not hasattr(layer, '_mha_block'):
            def _mha_block(self, x, mem, attn_mask=None, key_padding_mask=None):
                """Evaluate the cross-attention residual branch."""
                # PyTorch attention returns (output, weights); EVO caches only the residual output.
                x = self.multihead_attn(x, mem, mem,
                                       attn_mask=attn_mask,
                                       key_padding_mask=key_padding_mask,
                                       need_weights=False)[0]
                return self.dropout2(x)
            layer._mha_block = types.MethodType(_mha_block, layer)
        
        if not hasattr(layer, '_ff_block'):
            def _ff_block(self, x):
                """Evaluate the feed-forward residual branch."""
                # Feed-forward residual branch cached as one lattice unit.
                x = self.linear2(self.dropout(self.activation(self.linear1(x))))
                return self.dropout3(x)
            layer._ff_block = types.MethodType(_ff_block, layer)
        
        def forward_with_cache(self, tgt, memory, tgt_mask=None, memory_mask=None,
                              tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
            """Run a decoder layer under the fixed EVO refresh schedule."""
            current_step = cache['current_step']
            
            block_should_cache = cache.get('should_cache', {})
            should_cache_sa = block_should_cache.get(sa_block_key, False)
            should_cache_mha = block_should_cache.get(mha_block_key, False)
            should_cache_ff = block_should_cache.get(ff_block_key, False)
            
            can_use_sa_cache = sa_block_key in cache['block_cache'] and not should_cache_sa
            can_use_mha_cache = mha_block_key in cache['block_cache'] and not should_cache_mha
            can_use_ff_cache = ff_block_key in cache['block_cache'] and not should_cache_ff 
            
            cache['cache_hits']['total_calls'] += 3
            
            x = tgt
            
            # Reuse or refresh each residual branch independently at this timestep.
            if can_use_sa_cache:
                cache['cache_hits']['cache_hits'] += 1
                x = x + cache['block_cache'][sa_block_key]
            
            else:
                sa_result = self._sa_block(self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
                if should_cache_sa:
                    cached = sa_result.detach()
                    cache['block_cache'][sa_block_key] = cached.clone() if cache.get('clone_cache', False) else cached
                x = x + sa_result
            
            if can_use_mha_cache:
                cache['cache_hits']['cache_hits'] += 1
                x = x + cache['block_cache'][mha_block_key]
            else:
                mha_result = self._mha_block(self.norm2(x), memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask)
                if should_cache_mha:
                    cached = mha_result.detach()
                    cache['block_cache'][mha_block_key] = cached.clone() if cache.get('clone_cache', False) else cached
                x = x + mha_result
            
            if can_use_ff_cache:
                cache['cache_hits']['cache_hits'] += 1
                x = x + cache['block_cache'][ff_block_key]
            else:
                ff_result = self._ff_block(self.norm3(x))
                if should_cache_ff:
                    cached = ff_result.detach()
                    cache['block_cache'][ff_block_key] = cached.clone() if cache.get('clone_cache', False) else cached
                x = x + ff_result
            
            return x
        
        layer.forward = types.MethodType(forward_with_cache, layer)
        layer._cache_wrapped = True
    
    def _remove_cache_from_transformer_layer(self, layer, layer_name):
        """Restore the original forward method for one wrapped decoder layer."""
        if hasattr(layer, '_cache_wrapped') and layer_name in self.original_forwards:
            layer.forward = self.original_forwards[layer_name]
            delattr(layer, '_cache_wrapped')
    
    def _update_cache_flags(self, current_step: int):
        """Compute which residual branches should be refreshed at a timestep."""
        if current_step == 0 or current_step == 99:
            should_cache = {block_key: True for block_key in self.cache.get('block_steps', {}).keys()}
            self.cache['should_cache'] = should_cache
            return
        
        if 'steps_lookup' not in self.cache or not self.cache['steps_lookup']:
            steps_lookup = {}
            for block_key, steps in self.cache['block_steps'].items():
                steps_lookup[block_key] = set() if not steps else set(steps)
            self.cache['steps_lookup'] = steps_lookup
        
        should_cache = {}
        for block_key, step_set in self.cache['steps_lookup'].items():
            should_cache[block_key] = True if not step_set else current_step in step_set
        
        self.cache['should_cache'] = should_cache
    
    def set_optimal_steps(self, block_step_pairs: List[Tuple[int, int]]):
        """Load the selected schedule and wrap all affected layers.

        Args:
            block_step_pairs (List[Tuple[int, int]]): Offline schedule encoded
                as ``(block_idx, denoise_step)`` positions in the
                block-timestep lattice.
        """
        from collections import defaultdict
        
        self.cache['block_steps'] = {}
        self.cache['block_cache'] = {}
        self.cache['steps_lookup'] = {}
        
        block_steps_dict = defaultdict(list)
        
        for block_idx, step in block_step_pairs:
            if 0 <= block_idx < len(self.cacheable_blocks):
                block_name = self.cacheable_blocks[block_idx][0]
                block_steps_dict[block_name].append(step)
        
        for block_name, steps in block_steps_dict.items():
            self.cache['block_steps'][block_name] = sorted(list(set(steps)))
        
        layers_to_cache = set()
        for block_name, layer, block_type, layer_idx, block_type_idx in self.cacheable_blocks:
            if block_name in self.cache['block_steps']:
                layers_to_cache.add(layer)
        
        for layer in layers_to_cache:
            layer_name = None
            for block_name, l, _, _, _ in self.cacheable_blocks:
                if l == layer:
                    for suffix in ['_sa_block', '_mha_block', '_ff_block']:
                        if block_name.endswith(suffix):
                            layer_name = block_name.replace(suffix, '')
                            break
                    if layer_name:
                        break
            
            if layer_name:
                self._add_cache_to_transformer_layer(layer, layer_name)
    
    def reset(self):
        """Reset the wrapped policy and clear EVO residual-cache state."""
        if hasattr(self, '_original_reset'):
            self._original_reset()
        self.reset_cache()
        return self
    
    def reset_cache(self):
        """Clear residual caches before a new action query."""
        if self.cache is not None:
            self.cache['current_step'] = -1
            self.cache['block_cache'] = {}
            self.cache['steps_lookup'] = {}
        return self
    
    def reset_statistics(self):
        """Reset counters used to report residual-cache hit statistics."""
        if self.cache is not None:
            self.cache['cache_hits'] = {'total_calls': 0, 'cache_hits': 0}
        return self
    
    def get_cache_statistics(self):
        """Return cacheable branch calls, cache hits, and hit rate."""
        total_calls = self.cache['cache_hits']['total_calls']
        cache_hits = self.cache['cache_hits']['cache_hits']
        hit_rate = cache_hits / total_calls if total_calls > 0 else 0
        return {'total_calls': total_calls, 'cache_hits': cache_hits, 'hit_rate': hit_rate}
    
    def cleanup(self):
        """Remove EVO wrappers and restore the original model implementation."""
        layers_to_restore = set()
        for block_name, layer, _, _, _ in self.cacheable_blocks:
            layers_to_restore.add(layer)
        
        for layer in layers_to_restore:
            layer_name = None
            for block_name, l, _, _, _ in self.cacheable_blocks:
                if l == layer:
                    for suffix in ['_sa_block', '_mha_block', '_ff_block']:
                        if block_name.endswith(suffix):
                            layer_name = block_name.replace(suffix, '')
                            break
                    if layer_name:
                        break
            
            if layer_name and layer_name in self.original_forwards:
                self._remove_cache_from_transformer_layer(layer, layer_name)
        
        if hasattr(self.model, '_cache_wrapped') and hasattr(self.model, '_original_forward'):
            self.model.forward = self.model._original_forward
            delattr(self.model, '_cache_wrapped')
            delattr(self.model, '_original_forward')
        
        self.cache = None
        self.original_forwards = {}
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the wrapped Diffusion Policy object."""
        if hasattr(self.policy, name):
            return getattr(self.policy, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def get_num_blocks(self):
        """Return the number of cacheable residual branches."""
        return len(self.cacheable_blocks)
    
    def get_block_names(self):
        """Return stable names for cacheable residual branches."""
        return [name for name, _, _, _, _ in self.cacheable_blocks]
    
    def get_block_info(self):
        """Return branch type and decoder-layer metadata for cacheable units."""
        return [(name, block_type, layer_idx, block_type_idx) 
                for name, _, block_type, layer_idx, block_type_idx in self.cacheable_blocks] 
