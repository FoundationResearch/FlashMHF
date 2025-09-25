# coding=utf-8
# Copyright 2025 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial
from typing import Callable, Optional, Tuple, Union

import math
import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...generation import GenerationMixin
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    is_torch_flex_attn_available,
    logging,
    replace_return_docstrings,
)
from .configuration_mhffnmoe import MHFFNMoEConfig


if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from ...integrations.flex_attention import make_flex_block_causal_mask

from ...integrations import use_kernel_forward_from_hub


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = ""
_CONFIG_FOR_DOC = "MHFFNMoEConfig"


@use_kernel_forward_from_hub("RMSNorm")
# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->MHFFNMoE
class MHFFNMoERMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        MHFFNMoERMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(MHFFNMoERMSNorm)


# Copied from transformers.models.mhffn.modeling_mhffn.MHFFNMultiHeadRMSNorm with MHFFN->MHFFNMoE
class MHFFNMoEMultiHeadRMSNorm(nn.Module):
    """
    Multi-Head RMSNorm that applies different normalization parameters for each attention head.
    This is more efficient than using ModuleList as it supports batched computation.
    """
    def __init__(self, num_heads, head_dim, eps=1e-6):
        """
        Args:
            num_heads (int): Number of attention heads
            head_dim (int): Dimension of each head
            eps (float): Small value to prevent division by zero
        """
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.variance_epsilon = eps
        # Weight parameter for each head: shape (num_heads, head_dim)
        self.weight = nn.Parameter(torch.ones(num_heads, head_dim))

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Tensor of shape (batch_size, seq_len, num_heads, head_dim)
        
        Returns:
            Tensor of the same shape with per-head normalization applied
        """
        input_dtype = hidden_states.dtype
        # Convert to float32 for numerical stability
        hidden_states = hidden_states.to(torch.float32)
        # Compute variance along the head_dim (last dimension)
        # Shape: (batch_size, seq_len, num_heads, 1)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # Apply RMS normalization for each head
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Apply per-head weight scaling
        # Broadcasting: (batch_size, seq_len, num_heads, head_dim) * (num_heads, head_dim)
        hidden_states = hidden_states * self.weight.unsqueeze(0).unsqueeze(0)
        return hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(MHFFNMoEMultiHeadRMSNorm)

# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->MHFFNMoE
class MHFFNMoERotaryEmbedding(nn.Module):
    def __init__(self, config: MHFFNMoEConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed



def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def repeat_kuv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is a helper function to repeat the key/up/value states in Multi-Head FFN naive pytorch implementation.
    Unlike repeat_kv in attention, k/u/v here have no batch dimension.
    So, this is also the equivalent of torch.repeat_interleave(x, dim=0, repeats=n_rep).
    The hidden states go from (num_key_value_heads, intermediate_size, head_dim) to (num_attention_heads, intermediate_size, head_dim)
    For MoE format, it goes from (num_kuv_heads, num_experts, experts_dim, head_dim) to (num_kuv_heads * n_rep, num_experts, experts_dim, head_dim)

    Args:
        hidden_states (`torch.Tensor`): The key/up/value tensor of shape (num_kuv_heads, slen, head_dim) or (num_kuv_heads, num_experts, experts_dim, head_dim)
        n_rep (`int`): The number of times to repeat the tensor

    Returns:
        `torch.Tensor`: The repeated tensor of shape (num_kuv_heads * n_rep, slen, head_dim) or (num_kuv_heads * n_rep, num_experts, experts_dim, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    
    if hidden_states.dim() == 3:
        # Original format: (num_kuv_heads, intermediate_size, head_dim)
        num_kuv_heads, intermediate_size, head_dim = hidden_states.shape
        hidden_states = hidden_states[None, :, None, :, :].expand(1, num_kuv_heads, n_rep, intermediate_size, head_dim)
        return hidden_states.reshape(num_kuv_heads * n_rep, intermediate_size, head_dim)
    elif hidden_states.dim() == 4:
        # MoE format: (num_kuv_heads, num_experts, experts_dim, head_dim)
        num_kuv_heads, num_experts, experts_dim, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, None, :, :, :].expand(num_kuv_heads, n_rep, num_experts, experts_dim, head_dim)
        return hidden_states.reshape(num_kuv_heads * n_rep, num_experts, experts_dim, head_dim)
    else:
        raise ValueError(f"Unsupported tensor dimension: {hidden_states.dim()}. Expected 3 or 4 dimensions.")

# Copied from transformers.models.mhffn.modeling_mhffn.MHFFNMultiHeadFFNFlash with MHFFN->MHFFNMoE,mhffn->mhffnmoe
class MHFFNMoEMultiHeadFFNFlash(nn.Module):
    """Multi Head Feed Forward Network, the core logic is implemented in triton kernel, like flash attention"""
    def __init__(self, config, use_flash=True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mhffnmoe_num_heads = config.mhffnmoe_num_heads
        self.mhffnmoe_num_kuv_heads = config.mhffnmoe_num_kuv_heads
        self.mhffnmoe_num_kuv_groups = self.mhffnmoe_num_heads // self.mhffnmoe_num_kuv_heads
        self.head_dim = self.hidden_size // self.mhffnmoe_num_heads
        self.mhffnmoe_num_experts = config.mhffnmoe_num_experts
        self.mhffnmoe_experts_dim = self.intermediate_size // self.mhffnmoe_num_experts # we expect a division without remainder

        if self.config.mhffnmoe_apply_dot_scaling:
            self.head_scaling, self.up_scaling = math.sqrt(self.mhffnmoe_num_heads), math.sqrt(self.mhffnmoe_num_heads)
            if self.config.mhffnmoe_apply_dot_scaling_theory:
                self.head_scaling, self.up_scaling = self.mhffnmoe_num_heads, self.mhffnmoe_num_heads
        else:
            self.head_scaling, self.up_scaling = 1.0, 1.0
        if self.config.mhffnmoe_apply_final_scaling:
            self.final_scaling = math.sqrt(3./8.)
        else:
            self.final_scaling = 1.0

        self.q_norm = MHFFNMoEMultiHeadRMSNorm(self.mhffnmoe_num_heads, self.head_dim, eps=config.rms_norm_eps)
        # self.k_norm = MHFFNMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # self.u_norm = MHFFNMoERMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # V 的 norm 是可选的，但加上可能更稳定
        # self.v_norm = MHFFNMoERMSNorm(self.head_dim, elementwise_affine=True) 
        
        # ------------------Triton/CUDA Related Code---------------
        self.flash = use_flash
        if self.flash and self.head_dim > 256:
            # raise ValueError("FlashMLP is not supported for head_dim > 256")
            print("FlashMLP is not supported for head_dim > 256. Falling back to pytorch version")
            self.flash = False
        # Import flash MLP kernel - try CUDA version first, then Triton fallback
        self.flash_mlp = None
        self.use_cuda_flash_mlp = False
        # Try CUDA version first
        if self.flash:
            try:
                raise ImportError("nocuda")
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../../csrc/flash_ffn_moe'))
                from flash_ffn_moe_torch import FlashFFNMoE as FlashFFNMoECuda
                self.flash_mlp = FlashFFNMoECuda()
                self.use_cuda_flash_mlp = True
                print("Using Flash MLP CUDA implementation")
            except ImportError:
                # Fallback to Triton version
                try:
                    # import tilelang version
                    raise ImportError("notilelang")
                    from ops.flashffn_tilelang import FlashFFNTileLang as FlashMLPTileLang
                    self.flash_mlp = FlashMLPTileLang(enable_warnings=False)
                    self.use_cuda_flash_mlp = False
                    print("Using Flash MLP TileLang implementation")
                except ImportError:
                    try:
                        import sys
                        import os
                        sys.path.append(os.path.join(os.path.dirname(__file__), '../../../../..'))
                        if self.config.mhffnmoe_use_legacy_flash:
                            from ops.flash_mlp import FlashMLP
                            self.flash_mlp = FlashMLP()
                            print("Using Flash MLP Legacy(moe) Triton implementation")
                        else:
                            from ops.flash_mlp_moe import FlashMLPMoE
                            self.flash_mlp = FlashMLPMoE()
                            print("Using Flash MLP MoE Triton implementation")
                        self.use_cuda_flash_mlp = False
                    except ImportError as e:
                        print(f"Warning: FlashMLP not available, falling back to standard implementation. Error: {e}")
                        self.flash = False
                        self.flash_mlp = None
        else:
            print("Using Python FlashFFN Implementation")
        
        # Self projection layers
        self.w_self, self.w_self_2 = None, None
        if not self.config.mhffnmoe_skip_linear:
            self.w_self = nn.Linear(self.hidden_size, self.hidden_size, bias=config.mlp_bias)
            self.w_self_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=config.mlp_bias)
        # Multi-head gate, up and down projections as raw parameter tensors (no Linear wrapper)
        self.gate_weight = nn.Parameter(torch.empty(self.intermediate_size, self.head_dim * self.mhffnmoe_num_kuv_heads))  # (d_inter, d_model)
        self.up_weight = nn.Parameter(torch.empty(self.intermediate_size, self.head_dim * self.mhffnmoe_num_kuv_heads))
        self.down_weight = nn.Parameter(torch.empty(self.intermediate_size, self.head_dim * self.mhffnmoe_num_kuv_heads))
        self.router = nn.Parameter(torch.empty(self.mhffnmoe_num_heads, self.head_dim, self.mhffnmoe_num_experts))
        self.act_fn = ACT2FN[config.hidden_act]

        if config.mhffnmoe_custom_init:
            self._init_mhffnmoe_weights()
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        batch_size, seq_len, hidden_size = x.shape
        # First self projection: h' = h . W_self
        if self.config.mhffnmoe_skip_linear:
            Q = x
        else:
            Q = self.w_self(x)  # (batch_size, seq_len, hidden_size)
        
        # Split into heads: h' = [h'1, h'2, ..., h'mhffnmoe_num_heads]
        Q_multihead = Q.view(batch_size, seq_len, self.mhffnmoe_num_heads, self.head_dim)
        K_multihead = self.gate_weight.view(self.intermediate_size, self.mhffnmoe_num_kuv_heads, self.head_dim)
        U_multihead = self.up_weight.view(self.intermediate_size, self.mhffnmoe_num_kuv_heads, self.head_dim)
        V_multihead = self.down_weight.view(self.intermediate_size, self.mhffnmoe_num_kuv_heads, self.head_dim)
        
        if self.config.mhffnmoe_apply_rmsnorm:
            Q_multihead = self.q_norm(Q_multihead)
            # K_multihead = self.k_norm(K_multihead)
            # U_multihead = self.u_norm(U_multihead)
            # V_multihead = self.v_norm(V_multihead)

        # Transpose, swap the intermediate_size and mhffnmoe_num_heads dimensions
        Q_multihead = Q_multihead.transpose(1, 2).contiguous()   # (batch_size, mhffnmoe_num_heads, seq_len, head_dim)
        K_multihead = K_multihead.transpose(0, 1)   # (mhffnmoe_num_kuv_heads, intermediate_size, head_dim)
        U_multihead = U_multihead.transpose(0, 1)   # (mhffnmoe_num_kuv_heads, intermediate_size, head_dim)
        V_multihead = V_multihead.transpose(0, 1)   # (mhffnmoe_num_kuv_heads, intermediate_size, head_dim)

        # View for Experts
        K_multihead = K_multihead.view(self.mhffnmoe_num_kuv_heads, self.mhffnmoe_num_experts, self.mhffnmoe_experts_dim, self.head_dim).contiguous()
        U_multihead = U_multihead.view(self.mhffnmoe_num_kuv_heads, self.mhffnmoe_num_experts, self.mhffnmoe_experts_dim, self.head_dim).contiguous()
        V_multihead = V_multihead.view(self.mhffnmoe_num_kuv_heads, self.mhffnmoe_num_experts, self.mhffnmoe_experts_dim, self.head_dim).contiguous()

        R_multihead = (Q_multihead.to(torch.float32) @ self.router.to(torch.float32)).contiguous().sigmoid()  # (batch_size, mhffnmoe_num_heads, seq_len, mhffnmoe_num_experts)
        denominator = R_multihead.sum(dim=-1, keepdim=True)  # (batch_size, mhffnmoe_num_heads, seq_len, 1)
        R_multihead = R_multihead / denominator.clamp(min=1e-20)

        assert Q_multihead.shape == (batch_size, self.mhffnmoe_num_heads, seq_len, self.head_dim)
        assert K_multihead.shape == (self.mhffnmoe_num_kuv_heads, self.mhffnmoe_num_experts, self.mhffnmoe_experts_dim, self.head_dim)
        assert U_multihead.shape == (self.mhffnmoe_num_kuv_heads, self.mhffnmoe_num_experts, self.mhffnmoe_experts_dim, self.head_dim)
        assert V_multihead.shape == (self.mhffnmoe_num_kuv_heads, self.mhffnmoe_num_experts, self.mhffnmoe_experts_dim, self.head_dim)
        assert R_multihead.shape == (batch_size, self.mhffnmoe_num_heads, seq_len, self.mhffnmoe_num_experts)
        
        if self.flash and self.flash_mlp is not None:
            if self.config.mhffnmoe_use_legacy_flash:
                output = torch.zeros(Q_multihead.shape, dtype=torch.float32, device=x.device)  # (batch_size, mhffnmoe_num_heads, seq_len, head_dim)
                for i in range(self.mhffnmoe_num_experts):
                    v_expert = self.flash_mlp(
                        Q_multihead, K_multihead[:, i, :, :], U_multihead[:, i, :, :], V_multihead[:, i, :, :],
                        self.config.mhffnmoe_apply_dot_scaling, 
                        self.head_scaling, self.up_scaling, self.config.hidden_act
                    )  # (batch_size, mhffnmoe_num_heads, seq_len, head_dim)
                    output += (v_expert * self.final_scaling).to(torch.float32) * (R_multihead[:, :, :, i:i+1]) # (batch_size, mhffnmoe_num_heads, seq_len, head_dim)
                output = output.to(x.dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size) # (batch_size, seq_len, mhffnmoe_num_heads, head_dim) -> (batch_size, seq_len, hidden_size)
            else:
                output = self.flash_mlp(
                    Q_multihead, K_multihead, U_multihead, V_multihead, R_multihead.contiguous()
                ).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
            if not self.config.mhffnmoe_skip_linear:
                output = self.w_self_2(output)  # (batch_size, seq_len, hidden_size)
            
        else:
            # We only need to repeat kv for naive pytorch implementation
            K_multihead = repeat_kuv(K_multihead, self.mhffnmoe_num_kuv_groups) # (mhffnmoe_num_heads, mhffnmoe_num_experts, experts_dim, head_dim)
            U_multihead = repeat_kuv(U_multihead, self.mhffnmoe_num_kuv_groups) # (mhffnmoe_num_heads, mhffnmoe_num_experts, experts_dim, head_dim)
            V_multihead = repeat_kuv(V_multihead, self.mhffnmoe_num_kuv_groups) # (mhffnmoe_num_heads, mhffnmoe_num_experts, experts_dim, head_dim)

            output = torch.zeros(Q_multihead.shape, dtype=torch.float32, device=x.device)  # (batch_size, mhffnmoe_num_heads, seq_len, head_dim)
            for i in range(self.mhffnmoe_num_experts):
                weights = Q_multihead @ K_multihead[:, i, :, :].transpose(1, 2) * self.head_scaling # (batch_size, mhffnmoe_num_heads, seq_len, experts_dim)
                weights = self.act_fn(weights)
                weights = weights * (Q_multihead @ U_multihead[:, i, :, :].transpose(1, 2) * self.up_scaling) # (batch_size, mhffnmoe_num_heads, seq_len, experts_dim)
                # print(f"weights.shape: {weights.shape}, V_multihead[:, i, :, :].shape: {V_multihead[:, i, :, :].shape}, R_multihead[:, :, :, i:i+1].shape: {R_multihead[:, :, :, i:i+1].shape}")
                # print(f"w@V.shape: {(weights @ V_multihead[:, i, :, :]).shape}, R.shape: {R_multihead[:, :, :, i:i+1].shape}")
                output += (weights @ V_multihead[:, i, :, :] * self.final_scaling).to(torch.float32) * (R_multihead[:, :, :, i:i+1]) # (batch_size, mhffnmoe_num_heads, seq_len, head_dim)
            
            output = output.to(x.dtype).transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size) # (batch_size, seq_len, mhffnmoe_num_heads, head_dim) -> (batch_size, seq_len, hidden_size)
            
            if not self.config.mhffnmoe_skip_linear:
                output = self.w_self_2(output)  # (batch_size, seq_len, hidden_size)
        return output

    def _init_mhffnmoe_weights(self):
        """Custom initialization for the Multi Head FFN"""
        std = self.config.initializer_range
        # Use the same initializer as default nn.Linear in Transformers (normal with std)
        if self.w_self is not None:
            if self.config.mhffnmoe_init_wself_identity:
                self.w_self.weight.data = torch.eye(self.hidden_size, self.hidden_size, dtype=self.w_self.weight.dtype, device=self.w_self.weight.device)
            else:
                torch.nn.init.xavier_uniform_(self.w_self.weight, gain=self.config.mhffnmoe_init_w_self_gain)
                # torch.nn.init.xavier_normal_(self.w_self.weight, gain=self.config.mhffnmoe_init_w_self_gain)
                # self.w_self.weight.data.normal_(mean=0.0, std=std * self.config.mhffnmoe_init_w_self_gain)
            if self.w_self.bias is not None:
                torch.nn.init.zeros_(self.w_self.bias)
            self.w_self._mhffnmoe_custom_init = True
        if self.w_self_2 is not None:
            if self.config.mhffnmoe_init_wself_identity:
                self.w_self_2.weight.data = torch.eye(self.hidden_size, self.hidden_size, dtype=self.w_self_2.weight.dtype, device=self.w_self_2.weight.device)
            else:
                torch.nn.init.xavier_uniform_(self.w_self_2.weight, gain=self.config.mhffnmoe_init_w_self_2_gain)
                # torch.nn.init.xavier_normal_(self.w_self_2.weight, gain=self.config.mhffnmoe_init_w_self_2_gain)
                # self.w_self_2.weight.data.normal_(mean=0.0, std=std * self.config.mhffnmoe_init_w_self_2_gain)
            if self.w_self_2.bias is not None:
                torch.nn.init.zeros_(self.w_self_2.bias)
            self.w_self_2._mhffnmoe_custom_init = True

        # use normal distribution to initialize the weights just like the default nn.Linear in Transformers
        # or alternatively, try to use xavier_uniform_ if the results are not good
        if self.config.mhffnmoe_apply_dot_scaling_theory:
            self.gate_weight.data.normal_(mean=0.0, std=std * self.config.mhffnmoe_init_gate_gain / math.sqrt(self.head_scaling))
            self.up_weight.data.normal_(mean=0.0, std=std * self.config.mhffnmoe_init_up_gain / math.sqrt(self.up_scaling))
        else:
            self.gate_weight.data.normal_(mean=0.0, std=std * self.config.mhffnmoe_init_gate_gain)
            self.up_weight.data.normal_(mean=0.0, std=std * self.config.mhffnmoe_init_up_gain)
        self.down_weight.data.normal_(mean=0.0, std=std * self.config.mhffnmoe_init_down_gain / self.final_scaling) # LeCun init, we divide by final_scaling here as we multiply it in forward

        nn.init.trunc_normal_(self.router, mean=0.0, std=std * self.config.mhffnmoe_init_router_gain)

        # Mark all layers as custom-initialized so that PreTrainedModel does not override them
        self.gate_weight._mhffnmoe_custom_init = True
        self.up_weight._mhffnmoe_custom_init = True
        self.down_weight._mhffnmoe_custom_init = True
        self.router._mhffnmoe_custom_init = True

MHFFNMoEMultiHeadFFN = MHFFNMoEMultiHeadFFNFlash

# Legacy MLP for backward compatibility
# Copied from transformers.models.llama.modeling_llama.LlamaMLP with Llama->MHFFNMoE
class MHFFNMoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->MHFFNMoE
class MHFFNMoEAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: MHFFNMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# Copied from transformers.models.llama.modeling_llama.LlamaDecoderLayer with Llama->MHFFNMoE
class MHFFNMoEDecoderLayer(nn.Module):
    def __init__(self, config: MHFFNMoEConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mhffnmoe_disable_original_mlp_rmsnorm = config.mhffnmoe_disable_original_mlp_rmsnorm
        self.self_attn = MHFFNMoEAttention(config=config, layer_idx=layer_idx)
        self.mlp = MHFFNMoEMultiHeadFFNFlash(config, use_flash=config.mhffnmoe_use_flash)
        self.input_layernorm = MHFFNMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if not self.mhffnmoe_disable_original_mlp_rmsnorm:
            self.post_attention_layernorm = MHFFNMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        if not self.mhffnmoe_disable_original_mlp_rmsnorm:
            hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


MHFFNMOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MHFFNMoEConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare MHFFNMOE Model outputting raw hidden-states without any specific head on top.",
    MHFFNMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel with Llama->MHFFNMoE
class MHFFNMoEPreTrainedModel(PreTrainedModel):
    config_class = MHFFNMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MHFFNMoEDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        # Skip modules that explicitly declare they've handled their own init
        if getattr(module, "_mhffnmoe_custom_init", False):
            # print(f"Skipping default initialization for {module}")
            return
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # print(f"Initializing {module} with normal distribution")
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, MHFFNMoERMSNorm):
            module.weight.data.fill_(1.0)

MHFFNMOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length) or `BlockMask`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If the model is configured to use flex_attention, it will attempt to convert the mask Tensor into a BlockMask,
            but you can also pass a `BlockMask` object directly here.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare MHFFNMOE Model outputting raw hidden-states without any specific head on top.",
    MHFFNMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaModel with LLAMA->MHFFNMOE,Llama->MHFFNMOE
class MHFFNMoEModel(MHFFNMoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`MHFFNMoEDecoderLayer`]

    Args:
        config: MHFFNMoEConfig
    """

    def __init__(self, config: MHFFNMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [MHFFNMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MHFFNMoERMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MHFFNMoERotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(MHFFNMOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    partial(decoder_layer.__call__, **flash_attn_kwargs),
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: Union[torch.Tensor, "BlockMask"],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool = False,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None
        if self.config._attn_implementation == "flex_attention":
            if isinstance(attention_mask, torch.Tensor):
                attention_mask = make_flex_block_causal_mask(attention_mask)
            return attention_mask

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu", "npu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


# Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM with LLAMA->MHFFNMOE,Llama->MHFFNMOE,meta-llama/Llama-2-7b-hf->
class MHFFNMoEForCausalLM(MHFFNMoEPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = MHFFNMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    @add_start_docstrings_to_model_forward(MHFFNMOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MHFFNMoEForCausalLM

        >>> model = MHFFNMoEForCausalLM.from_pretrained("")
        >>> tokenizer = AutoTokenizer.from_pretrained("")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        # print(input_ids.shape)
        # logging.log(input_ids.shape)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The LLaMa Model transformer with a sequence classification head on top (linear layer).

    [`MHFFNMoEForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    MHFFNMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with LLAMA->MHFFNMOE,Llama->MHFFNMOE
class MHFFNMoEForSequenceClassification(MHFFNMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MHFFNMoEModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(MHFFNMOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> SequenceClassifierOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        transformer_outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden_states = transformer_outputs.last_hidden_state
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            last_non_pad_token = -1
        elif input_ids is not None:
            # To handle both left- and right- padding, we take the rightmost token that is not equal to pad_token_id
            non_pad_mask = (input_ids != self.config.pad_token_id).to(logits.device, torch.int32)
            token_indices = torch.arange(input_ids.shape[-1], device=logits.device, dtype=torch.int32)
            last_non_pad_token = (token_indices * non_pad_mask).argmax(-1)
        else:
            last_non_pad_token = -1
            logger.warning_once(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), last_non_pad_token]

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, pooled_logits=pooled_logits, config=self.config)

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
The MHFFNMoE Model transformer with a span classification head on top for extractive question-answering tasks like
SQuAD (a linear layer on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MHFFNMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForQuestionAnswering with LLAMA->MHFFNMOE,Llama->MHFFNMOE
class MHFFNMoEForQuestionAnswering(MHFFNMoEPreTrainedModel):
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.transformer = MHFFNMoEModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.transformer.embed_tokens

    def set_input_embeddings(self, value):
        self.transformer.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(MHFFNMOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> QuestionAnsweringModelOutput:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """

        outputs: BaseModelOutputWithPast = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs.last_hidden_state

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        loss = None
        if start_positions is not None and end_positions is not None:
            loss = self.loss_function(start_logits, end_logits, start_positions, end_positions, **kwargs)

        return QuestionAnsweringModelOutput(
            loss=loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    The MHFFNMOE Model transformer with a token classification head on top (a linear layer on top of the hidden-states
    output) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    MHFFNMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForTokenClassification with LLAMA->MHFFNMOE,Llama->MHFFNMOE
class MHFFNMoEForTokenClassification(MHFFNMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MHFFNMoEModel(config)
        if getattr(config, "classifier_dropout", None) is not None:
            classifier_dropout = config.classifier_dropout
        elif getattr(config, "hidden_dropout", None) is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.score = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @can_return_tuple
    @add_start_docstrings_to_model_forward(MHFFNMOE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> TokenClassifierOutput:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        outputs: BaseModelOutputWithPast = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.score(sequence_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.config)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "MHFFNMoEForCausalLM",
    "MHFFNMoEModel",
    "MHFFNMoEPreTrainedModel",
    "MHFFNMoEForSequenceClassification",
    "MHFFNMoEForQuestionAnswering",
    "MHFFNMoEForTokenClassification",
    "MHFFNMoEMultiHeadFFN",
    "MHFFNMoEMLP",
]
