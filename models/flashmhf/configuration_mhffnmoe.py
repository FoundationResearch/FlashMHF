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
"""MHFFNMoE model configuration"""

from ...configuration_utils import PretrainedConfig
from ...modeling_rope_utils import rope_config_validation


class MHFFNMoEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MHFFNMoEModel`]. It is used to instantiate an MHFFNMoE
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the MHFFNMoE-7B.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the MHFFNMoE model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`MHFFNMoEModel`]
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 11008):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 32):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 32):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When
            converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed
            by meanpooling all the original heads within that group. For more details checkout [this
            paper](https://arxiv.org/pdf/2305.13245.pdf). If it is not specified, will default to
            `num_attention_heads`.
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. MHFFNMoE 1 supports up to 2048 tokens,
            MHFFNMoE 2 up to 4096, CodeMHFFNMoE up to 16384.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        pad_token_id (`int`, *optional*):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
        pretraining_tp (`int`, *optional*, defaults to 1):
            Experimental feature. Tensor parallelism rank used during pretraining. Please refer to [this
            document](https://huggingface.co/docs/transformers/main/perf_train_gpu_many#tensor-parallelism) to
            understand more about it. This value is necessary to ensure exact reproducibility of the pretraining
            results. Please refer to [this issue](https://github.com/pytorch/pytorch/issues/76232).
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie weight embeddings
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period of the RoPE embeddings.
        rope_scaling (`Dict`, *optional*):
            Dictionary containing the scaling configuration for the RoPE embeddings. NOTE: if you apply new rope type
            and you expect the model to work on longer `max_position_embeddings`, we recommend you to update this value
            accordingly.
            Expected contents:
                `rope_type` (`str`):
                    The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
                    'mhffnmoe3'], with 'default' being the original RoPE implementation.
                `factor` (`float`, *optional*):
                    Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
                    most scaling types, a `factor` of x will enable the model to handle sequences of length x *
                    original maximum pre-trained length.
                `original_max_position_embeddings` (`int`, *optional*):
                    Used with 'dynamic', 'longrope' and 'mhffnmoe3'. The original max position embeddings used during
                    pretraining.
                `attention_factor` (`float`, *optional*):
                    Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
                    computation. If unspecified, it defaults to value recommended by the implementation, using the
                    `factor` field to infer the suggested value.
                `beta_fast` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
                    ramp function. If unspecified, it defaults to 32.
                `beta_slow` (`float`, *optional*):
                    Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
                    ramp function. If unspecified, it defaults to 1.
                `short_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to short contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `long_factor` (`List[float]`, *optional*):
                    Only used with 'longrope'. The scaling factor to be applied to long contexts (<
                    `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
                    size divided by the number of attention heads divided by 2
                `low_freq_factor` (`float`, *optional*):
                    Only used with 'mhffnmoe3'. Scaling factor applied to low frequency components of the RoPE
                `high_freq_factor` (`float`, *optional*):
                    Only used with 'mhffnmoe3'. Scaling factor applied to high frequency components of the RoPE
        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        mlp_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.
        head_dim (`int`, *optional*):
            The attention head dimension. If None, it will default to hidden_size // num_attention_heads
        mhffnmoe_use_flash (`bool`, *optional*, defaults to `True`):
            Whether to use flash triton/cuda implementation in the Multi Head FFN.
        mhffnmoe_use_legacy_flash (`bool`, *optional*, defaults to `False`):
            Whether to use legacy flash triton/cuda(not specified for MoE flash) implementation in the Multi Head FFN.
            Note: You must set mhffnmoe_use_flash as well as mhffnmoe_use_legacy_flash to be true to enable legacy flash implementation.
        mhffnmoe_num_heads (`int`, *optional*, defaults to 1):
            Number of heads for Multi Head FFN. The hidden_size should be divisible by mhffnmoe_num_heads.
        mhffnmoe_num_kuv_heads(`int`, *optional*):
            Number of 1 Q heads corresponds to K,U,V heads. E.g. if set to 2, when Q have k heads, the K,U,V will only have k//2 heads.
            It's required that mhffnmoe_num_heads % mhffnmoe_num_kuv_heads == 0, otherwise severe bugs will occur.
            If it is not specified, will default to `num_attention_heads`.
        mhffnmoe_num_experts (`int`, *optional*, defaults to 1):
            Number of experts for the Mixture of Experts in the Multi Head FFN MoE. The intermediate_size should be divisible by mhffnmoe_num_experts.
        mhffnmoe_skip_linear (`bool`, *optional*, defaults to `False`):
            Whether to skip the linear projection in the Multi Head FFN.
        mhffnmoe_apply_dot_scaling (`bool`, *optional*, defaults to `False`):
            Whether to apply dot scaling to the Multi Head FFN. If True, the dot product of the Multi Head FFN will be scaled by sqrt(mhffnmoe_num_heads).
        mhffnmoe_apply_dot_scaling_theory (`bool`, *optional*, defaults to `False`):
            Whether to apply the theoretically derived dot scaling to the Multi Head FFN. If True, the dot product of the Multi Head FFN will be scaled by mhffnmoe_num_heads, and std will be adjusted accordingly during initialization.
        mhffnmoe_apply_final_scaling (`bool`, *optional*, defaults to `False`):
            Whether to apply final scaling to the MHFFNMoE. If True, the output of the MHFFNMoE will be scaled by mhffnmoe_num_heads
        mhffnmoe_apply_rmsnorm (`bool`, *optional*, defaults to `False`):
            Whether to apply learnable rmsnorm to MHFFNMoE's q,k,u. If True, q=rmsnorm(q), same applies to k, v.
        mhffnmoe_disable_original_mlp_rmsnorm (`bool`, *optional*, defaults to `False`):
            Whether to disable the original MLP rmsnorm in the MHFFNMoE. If True, the original MLP rmsnorm will not be applied.
            This might be useful when using the new MHFFNMoE with new pre-use multihead rmsnorm.
        mhffnmoe_custom_init (`bool`, *optional*, defaults to `False`):
            Whether to use custom initialization for the Multi Head FFN.
        mhffnmoe_init_wself_identity (`bool`, *optional*, defaults to `False`):
            Whether initialize wself and wself2 as identity matrix instead of xavier init.
            This will only work when mhffnmoe_custom_init is set to true.
        mhffnmoe_init_w_self_gain (`float`, *optional*, defaults to 1.0):
            Gain for the self-attention weight.
        mhffnmoe_init_w_self_2_gain (`float`, *optional*, defaults to 1.0):
            Gain for the self-attention weight squared.
        mhffnmoe_init_gate_gain (`float`, *optional*, defaults to 1.0):
            Gain for the gate projection.
        mhffnmoe_init_up_gain (`float`, *optional*, defaults to 1.0):
            Gain for the up projection.
        mhffnmoe_init_down_gain (`float`, *optional*, defaults to 1.0):
            Gain for the down projection.
        mhffnmoe_init_router_gain (`float`, *optional*, defaults to 1.0):
            Gain for the router projection.
    ```python
    >>> from transformers import MHFFNMoEModel, MHFFNMoEConfig

    >>> # Initializing a MHFFNMoE mhffnmoe-7b style configuration
    >>> configuration = MHFFNMoEConfig()

    >>> # Initializing a model from the mhffnmoe-7b style configuration
    >>> model = MHFFNMoEModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "mhffnmoe"
    keys_to_ignore_at_inference = ["past_key_values"]
    # Default tensor parallel plan for base model `MHFFNMoEModel`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=None,
        mhffnmoe_use_flash=True,
        mhffnmoe_use_legacy_flash=False,
        mhffnmoe_num_heads=1,
        mhffnmoe_num_kuv_heads=None,
        mhffnmoe_num_experts=1,
        mhffnmoe_skip_linear=False,
        mhffnmoe_apply_dot_scaling=False,
        mhffnmoe_apply_dot_scaling_theory=False,
        mhffnmoe_apply_final_scaling=False,
        mhffnmoe_apply_rmsnorm=False,
        mhffnmoe_disable_original_mlp_rmsnorm=False,
        mhffnmoe_custom_init=False,
        mhffnmoe_init_wself_identity=False,
        mhffnmoe_init_w_self_gain=1.0,
        mhffnmoe_init_w_self_2_gain=1.0,
        mhffnmoe_init_gate_gain=1.0,
        mhffnmoe_init_up_gain=1.0,
        mhffnmoe_init_down_gain=1.0,
        mhffnmoe_init_router_gain=1.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        # set to MHA if not specified num of kv heads
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim if head_dim is not None else self.hidden_size // self.num_attention_heads
        
        self.mhffnmoe_use_flash = mhffnmoe_use_flash
        self.mhffnmoe_use_legacy_flash = mhffnmoe_use_legacy_flash
        self.mhffnmoe_num_heads = mhffnmoe_num_heads
        # sets to MHA(MHFFNMoE) if not specified num of kuv heads
        if mhffnmoe_num_kuv_heads is None:
            mhffnmoe_num_kuv_heads = mhffnmoe_num_heads
        self.mhffnmoe_num_kuv_heads = mhffnmoe_num_kuv_heads
        self.mhffnmoe_num_experts = mhffnmoe_num_experts
        self.mhffnmoe_skip_linear = mhffnmoe_skip_linear
        self.mhffnmoe_apply_dot_scaling = mhffnmoe_apply_dot_scaling
        self.mhffnmoe_apply_dot_scaling_theory = mhffnmoe_apply_dot_scaling_theory
        self.mhffnmoe_apply_final_scaling = mhffnmoe_apply_final_scaling
        self.mhffnmoe_apply_rmsnorm = mhffnmoe_apply_rmsnorm
        self.mhffnmoe_disable_original_mlp_rmsnorm = mhffnmoe_disable_original_mlp_rmsnorm
        self.mhffnmoe_custom_init = mhffnmoe_custom_init
        self.mhffnmoe_init_wself_identity = mhffnmoe_init_wself_identity
        self.mhffnmoe_init_w_self_gain = mhffnmoe_init_w_self_gain
        self.mhffnmoe_init_w_self_2_gain = mhffnmoe_init_w_self_2_gain
        self.mhffnmoe_init_gate_gain = mhffnmoe_init_gate_gain
        self.mhffnmoe_init_up_gain = mhffnmoe_init_up_gain
        self.mhffnmoe_init_down_gain = mhffnmoe_init_down_gain
        self.mhffnmoe_init_router_gain = mhffnmoe_init_router_gain

        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, copy it it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


__all__ = ["MHFFNMoEConfig"]
