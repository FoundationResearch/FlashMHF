"""
Flash MLP Implementation using Triton

Similar to Flash Attention, this kernel performs the MHFFN computation:
Q @ K^T -> activation -> (optional: * U) -> @ V

The key insight is to avoid materializing the large intermediate tensor
(batch_size, mhffn_num_heads, seq_len, intermediate_size) in HBM by using
tiling and online computation.
"""
# TODO:
# 1. investigate into block ptr order param, maybe change it to (1, 0) will results in better mem alignment
# 2. investigate into tensor descriptor
# 3. investigate into warp/num_stage number functionality.

# fwd: online autotune is fast
# bwd: dhead=64, B_SEQ, B_INTER = 64,64; num_stages=1; num_warps=4
# bwd: dhead=128?


import torch
import triton
import triton.language as tl
import math
import os


############################################
# Autotuning configuration for Flash MLP
############################################

# Candidate block sizes (power-of-two ≥16) We are adding a wider range for hopper architecture
# _BLOCK1_OPTIONS = [32, 64, 128, 256, 512, 1024]
# _BLOCK2_OPTIONS = [16, 32, 64, 128, 256]
# _STAGE_OPTIONS = [1,2,3,4]
# _WARP_OPTIONS = [2,4,8,16]

# bf16
# 1group:
# always 128 - 16, 3 - 4

# 2 group:
# 128 hdim: 128 - 16, 1/3 - 4
# 64 hdim:
#   3090: 128 - 16 - 2 - 4
#   a6000: 128 - 16 - 3 - 4
# 64 - 32, 3 - 4

# ------HOPPER-------
# BF16, hdim128, 1grps, bigmodel, 0.97xfuse - BLOCK_SEQ: 128, BLOCK_INTER: 128, num_warps: 8, num_ctas: 1, num_stages: 2
# BF16, hdim128, 2grps, bigmodel, 1.04xfuse - BLOCK_SEQ: 128, BLOCK_INTER: 32, num_warps: 16, num_ctas: 1, num_stages: 3
# ===================

_BLOCK1_OPTIONS = [128]
_BLOCK2_OPTIONS = [16]
_STAGE_OPTIONS = [3]
_WARP_OPTIONS = [4]

# Shared-memory limit – Ampere smem is 99 KB, Hopper doubles
_SMEM_LIMIT = 200 * 1024  # bytes (per-SM partition)


def _estimate_smem(block_seq: int, block_inter: int, head_dim: int, has_up: bool) -> int:
    """Return required shared memory in bytes for given tile sizes (bf16=2 bytes)."""
    num_elems = block_seq * head_dim + 2 * block_inter * head_dim + (has_up * block_inter * head_dim)
    return num_elems * 2 * 2  # bf16 element size


# TODO: TMA, sync worker on loading and computing(warp specialization)
# Build search space
_CONFIGS = [
    triton.Config(
        {'BLOCK_SEQ': bs, 'BLOCK_INTER': bi, "LOG2_BLOCK_SEQ": int(math.log2(bs))},
        num_warps=w,
        num_stages=st,
    )
    for bs in _BLOCK1_OPTIONS
    for bi in _BLOCK2_OPTIONS
    for st in _STAGE_OPTIONS
    for w in _WARP_OPTIONS
]


def _prune_configs(configs, named_args, **meta):
    """Remove configs that (1) exceed smem or (2) tile sizes larger than actual dims."""
    seq_len = named_args["seq_len"]
    intermediate_size = named_args["intermediate_size"]
    head_dim = named_args["head_dim"]
    has_up = True  # up projection always present
    
    valid = []
    for conf in configs:
        bs = conf.kwargs["BLOCK_SEQ"]
        bi = conf.kwargs["BLOCK_INTER"]
        # st = conf.kwargs["num_stages"]
        if (bs > seq_len and seq_len > 16) or (bi > intermediate_size and intermediate_size > 16):
            continue
        if _estimate_smem(bs, bi, head_dim, has_up) > _SMEM_LIMIT:
            print("not skipping configs although it may exceed the smem limit")
            # continue
        valid.append(conf)
    print(f"Valid configs: {len(valid)}", flush=True)
    return valid


@triton.autotune(configs=_CONFIGS,
                 key=["seq_len", "intermediate_size", "head_dim"],
                 prune_configs_by={"early_config_prune": _prune_configs})
@triton.jit
def flash_mlp_forward_kernel(
    # Input tensors
    Q_ptr, K_ptr, U_ptr, V_ptr,
    # Output tensor
    O_ptr,
    # Dimensions
    batch_size, num_heads, seq_len, head_dim, intermediate_size,
    # Strides for Q (batch_size, num_heads, seq_len, head_dim)
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    # Strides for K (num_heads, intermediate_size, head_dim)
    k_head_stride, k_inter_stride, k_head_dim_stride,
    # Strides for U (num_heads, intermediate_size, head_dim) - optional
    u_head_stride, u_inter_stride, u_head_dim_stride,
    # Strides for V (num_heads, intermediate_size, head_dim)
    v_head_stride, v_inter_stride, v_head_dim_stride,
    # Strides for O (batch_size, num_heads, seq_len, head_dim)
    o_batch_stride, o_head_stride, o_seq_stride, o_head_dim_stride,
    # Block sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_INTER: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KUV_GROUPS: tl.constexpr,
    LOG2_BLOCK_SEQ: tl.constexpr
):
    """
    Flash MLP Forward Kernel
    
    Key insight: Instead of materializing the full (seq_len, intermediate_size) tensor,
    we compute it in blocks and immediately apply the V projection.
    
    Full Forward Formula:
        Init WQ, WK, WU, WV matrix, scalar s0, s1. Given h shape (batch_size, seq_len, d_model)
        WQ shape (dmodel, dmodel), WK shape (intermediate_size, dmodel), WU shape (intermediate_size, dmodel), WV shape (intermediate_size, dmodel)
        Q = h @ WQ       ---  (batch_size, seq_len, dmodel)
        K = WK           ---  (intermediate_size, dmodel)
        U = WU           ---  (intermediate_size, dmodel)
        V = WV           ---  (intermediate_size, dmodel)
        Q, K, U, V reshape & transpose to split heads
        ===From here, is the input of flash kernel===
        Q                ---  (batch_size, num_heads, seq_len, head_dim)
        K                ---  (num_heads, intermediate_size, head_dim)
        U                ---  (num_heads, intermediate_size, head_dim)
        V                ---  (num_heads, intermediate_size, head_dim)
        M = Q @ K^T * s0 ---  (batch_size, num_heads, seq_len, intermediate_size)
        N = Q @ U^T * s1 ---  (batch_size, num_heads, seq_len, intermediate_size)
        A = actfn(M) * N ---  (batch_size, num_heads, seq_len, intermediate_size)
        S = A @ V        ---  (batch_size, num_heads, seq_len, head_dim)
        S transpose and reshape to merge heads
        S                ---  (batch_size, seq_len, dmodel)
        ===To here, is the output of flash kernel===
        O = S @ WO       ---  (batch_size, seq_len, dmodel)
    
    Memory complexity: O(BLOCK_SEQ * BLOCK_INTER) instead of O(seq_len * intermediate_size)
    """
    # Static assertions for block size constraints
    tl.assume(BLOCK_SEQ >= 16)
    tl.assume(BLOCK_INTER >= 16)
    tl.assume(HEAD_DIM >= 16)
    tl.assume(BLOCK_SEQ % 16 == 0)
    tl.assume(BLOCK_INTER % 16 == 0)
    tl.assume(q_head_dim_stride == 1)                      # contiguous head dim in Q
    tl.assume(k_head_dim_stride == 1)                      # contiguous head dim in K
    tl.assume(u_head_dim_stride == 1)
    tl.assume(v_head_dim_stride == 1)
    tl.assume(q_head_stride % 16 == 0)
    tl.assume(k_head_stride % 16 == 0)
    tl.assume(u_head_stride % 16 == 0)
    tl.assume(v_head_stride % 16 == 0)
    
    # Constants
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    
    # Program IDs (Flash-Attention style grid)
    #   pid 0 -> sequence block index
    #   pid 1 -> head index
    #   pid 2 -> batch index
    seq_block_id = tl.program_id(0)
    kuv_head_id = tl.program_id(1)
    q_head_id = kuv_head_id * NUM_KUV_GROUPS
    batch_id = tl.program_id(2)
    
    # Calculate sequence range for this block
    seq_start = seq_block_id * BLOCK_SEQ
    
    col_ids = tl.arange(0, HEAD_DIM)[None, :]   # [H]
    
    # Pointers for this batch and kuv head
    q_ptr = Q_ptr + batch_id * q_batch_stride + q_head_id * q_head_stride
    k_ptr = (
        K_ptr + kuv_head_id * k_head_stride
        + col_ids * k_head_dim_stride # remember to comment those lines out if option 2.1
    )
    u_ptr = (
        U_ptr + kuv_head_id * u_head_stride
        + col_ids * u_head_dim_stride # remember to comment those lines out if option 2.1
    )
    v_ptr = (
        V_ptr + kuv_head_id * v_head_stride
        + col_ids * v_head_dim_stride # remember to comment those lines out if option 2.1
    )
    o_ptr = O_ptr + batch_id * o_batch_stride + q_head_id * o_head_stride

    # ----------Option 1 - Make Block PTR -----------
    # q_ptr_block = tl.make_block_ptr(
    #     base=q_ptr,
    #     shape=(NUM_KUV_GROUPS, seq_len, HEAD_DIM),
    #     strides=(q_head_stride, q_seq_stride, q_head_dim_stride),
    #     offsets=(0, seq_start, 0),
    #     block_shape=(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM),
    #     order=(0, 1, 2)  # head idx dim is the slowest changing one so the order is like this
    # )
    # q_block = tl.load(q_ptr_block, boundary_check=(1,))  # (NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM)
    # q_block = q_block.reshape((NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM)) # (NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM)
    # ----------Option 2 - Make Tile PTR -----------
    # Here we can use log accelerate the computation, will be implemented later
    row_ids = tl.arange(0, NUM_KUV_GROUPS * BLOCK_SEQ)
    g = (row_ids >> LOG2_BLOCK_SEQ)[:, None]     # row_ids // BLOCK_SEQ
    s_local = row_ids & (BLOCK_SEQ - 1)# row_ids % BLOCK_SEQ
    s = seq_start + s_local
    tl.multiple_of(seq_start, BLOCK_SEQ)
    q_ptr_block = (
        q_ptr
        + g * q_head_stride
        + s[:, None] * q_seq_stride
        + col_ids * q_head_dim_stride
    )
    mask_rows = (s < seq_len)[:, None]
    q_block = tl.load(q_ptr_block, mask=mask_rows)
    output_acc = tl.zeros((NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM), dtype=tl.float32)
    
    # We'll use tl.make_block_ptr inside the loop for better coalesced accesses.
    # Loop over intermediate dimension blocks
    row_inter = tl.arange(0, BLOCK_INTER)[:, None]
    k_ptr_block = k_ptr + row_inter * k_inter_stride - BLOCK_INTER * k_inter_stride          # option 2.2, prevent first loop
    u_ptr_block = u_ptr + row_inter * u_inter_stride - BLOCK_INTER * u_inter_stride          # option 2.2, prevent first loop
    v_ptr_block = v_ptr + row_inter * v_inter_stride - BLOCK_INTER * v_inter_stride          # option 2.2, prevent first loop
    for inter_start in tl.range(0, intermediate_size, BLOCK_INTER):
        inter_start = tl.multiple_of(inter_start, BLOCK_INTER)
        # ----------Option 2.1 - Make Block PTR -----------
        # Build block pointers for K, U, V using tl.make_block_ptr for efficient access
        # k_ptr_block = tl.make_block_ptr(
        #     base=k_ptr,
        #     shape=(HEAD_DIM, intermediate_size),
        #     strides=(k_head_dim_stride, k_inter_stride),
        #     offsets=(0, inter_start),
        #     block_shape=(HEAD_DIM, BLOCK_INTER),
        #     order=(0, 1),  # load as (HEAD_DIM, BLOCK_INTER)
        # )

        # u_ptr_block = tl.make_block_ptr(
        #     base=u_ptr,
        #     shape=(HEAD_DIM, intermediate_size),
        #     strides=(u_head_dim_stride, u_inter_stride),
        #     offsets=(0, inter_start),
        #     block_shape=(HEAD_DIM, BLOCK_INTER),
        #     order=(0, 1),  # load as (HEAD_DIM, BLOCK_INTER)
        # )

        # v_ptr_block = tl.make_block_ptr(
        #     base=v_ptr,
        #     shape=(intermediate_size, HEAD_DIM),
        #     strides=(v_inter_stride, v_head_dim_stride),
        #     offsets=(inter_start, 0),
        #     block_shape=(BLOCK_INTER, HEAD_DIM),
        #     order=(0, 1),  # load as (BLOCK_INTER, HEAD_DIM)
        # )
        # k_block = tl.load(k_ptr_block)  # (HEAD_DIM, BLOCK_INTER), no bdry check
        # u_block = tl.load(u_ptr_block)  # (HEAD_DIM, BLOCK_INTER), no bdry check
        # v_block = tl.load(v_ptr_block)  # (BLOCK_INTER, HEAD_DIM), no bdry check
        # ----------Option 2.2 - Make Tile PTR -----------
        k_ptr_block += BLOCK_INTER * k_inter_stride
        u_ptr_block += BLOCK_INTER * u_inter_stride
        v_ptr_block += BLOCK_INTER * u_inter_stride
        k_block = tl.trans(tl.load(k_ptr_block))  # (HEAD_DIM, BLOCK_INTER), no bdry check
        u_block = tl.trans(tl.load(u_ptr_block))  # (HEAD_DIM, BLOCK_INTER), no bdry check
        v_block = tl.load(v_ptr_block)            # (BLOCK_INTER, HEAD_DIM), no bdry check
        
        # Compute Q @ K^T: (BLOCK_SEQ, BLOCK_INTER)
        qk_scores = tl.dot(q_block, k_block)
        
        # Activation: SiLU (x * sigmoid(x)) – the only supported option
        activated_scores = (qk_scores * (1.0/(tl.exp2(-qk_scores*RCP_LN2) + 1.0)))
        
        # Compute Q @ U^T
        qu_scores = tl.dot(q_block, u_block)
        activated_scores = (activated_scores * qu_scores).to(tl.bfloat16)
        
        # fuse accumulation: acc argument of tl.dot
        output_acc = tl.dot(activated_scores, v_block, output_acc)
    # ----------Option 1 - Make Block PTR -----------
    # output_acc = output_acc.reshape((NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM))
    # o_block_ptr = tl.make_block_ptr(
    #     base=o_ptr,
    #     shape=(NUM_KUV_GROUPS, seq_len, HEAD_DIM),
    #     strides=(o_head_stride, o_seq_stride, o_head_dim_stride),
    #     offsets=(0, seq_start, 0),
    #     block_shape=(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM),
    #     order=(0, 1, 2)
    # )
    # tl.store(o_block_ptr, output_acc.to(tl.bfloat16), boundary_check=(1,))
    # ----------Option 2 - Make Tile PTR -----------
    o_ptr_block = (
        o_ptr
        + g * o_head_stride
        + s[:, None] * o_seq_stride
        + col_ids * o_head_dim_stride
    )
    tl.store(o_ptr_block, output_acc.to(tl.bfloat16), mask=mask_rows)
    


# ---------------------------
# Backward autotune configs
# ---------------------------

# _BWD_BLOCK_SEQ_OPTIONS = [16,32,64,128]
# _BWD_BLOCK_INTER_OPTIONS = [16,32,64]
# _BWD_STAGE_OPTIONS = [2,3]
# _BWD_WARP_OPTIONS = [4]
# _BWD_BATCHLOOP_NUMSTAGE = [1,2]
# _BWD_INNERLOOP_NUMSTAGE = [1,2]

# ------RTX3090-------
# SEQ: [32,64], INTER: [32, 64], STAGE: [2,3], WARP: [4]
# ===================

# ------HOPPER-------
# WARP: [16] # TODO
# ===================

_BWD_BLOCK_SEQ_OPTIONS = [32,64]
_BWD_BLOCK_INTER_OPTIONS = [32,64]
_BWD_STAGE_OPTIONS = [2,3] # if 64, seems this need to change to 2, while 128, change to 3
_BWD_WARP_OPTIONS = [4]
_BWD_BATCHLOOP_NUMSTAGE = [1]
_BWD_INNERLOOP_NUMSTAGE = [2]


def _estimate_smem_bwd(block_seq: int, block_inter: int, head_dim: int, has_up: bool) -> int:
    """Rough shared-memory footprint for one CTA of backward pass (bytes).

    Layout (all fp32):
        • dq_acc (BLOCK_SEQ × D)
        • q_block & do_block (2 × BLOCK_SEQ × D)  – fp16 but upcast ⇒ fp32 in SRAM
        • dk, dv (2 × block_inter × D)
        • k_block, v_block, u_block (<=3 × block_inter × D) – upcast fp32
    Simplify为 3·BLOCK_SEQ + (4 + has_up)·BLOCK_INTER elements.
    """
    elems = head_dim * (3 * block_seq + (4 + (1 if has_up else 0)) * block_inter)
    return elems * 4

_BWD_CONFIGS = [
    triton.Config(
        {'BLOCK_SEQ': bs,'BLOCK_INTER': bi, "LOG2_BLOCK_SEQ": int(math.log2(bs)),
        "BATCHLOOP_NUMSTAGE": blst, "INNERLOOP_NUMSTAGE": ilst},
        num_stages=st, num_warps = wa
    )
    for bs in _BWD_BLOCK_SEQ_OPTIONS
    for bi in _BWD_BLOCK_INTER_OPTIONS
    for st in _BWD_STAGE_OPTIONS
    for wa in _BWD_WARP_OPTIONS
    for blst in _BWD_BATCHLOOP_NUMSTAGE
    for ilst in _BWD_INNERLOOP_NUMSTAGE
]

def _prune_bwd_configs(configs, named_args, **meta):
    seq_len = named_args["seq_len"]
    intermediate_size = named_args["intermediate_size"]
    head_dim = named_args["head_dim"]
    has_up = True  # up projection always present
    valid = []
    for conf in configs:
        bs = conf.kwargs["BLOCK_SEQ"]
        bi = conf.kwargs["BLOCK_INTER"]
        # Tile must fit dims
        if (bs > seq_len and seq_len > 16) or (bi > intermediate_size and intermediate_size > 16):
            continue
        if _estimate_smem_bwd(bs, bi, head_dim, has_up) > _SMEM_LIMIT:
            print(f"Skipping config: {conf.kwargs}, smem_bwd: {_estimate_smem_bwd(bs, bi, head_dim, has_up)}", flush=True)
            # continue # we don't prune base on estimation of smem right now, it's not accurate
        valid.append(conf)
    print(f"Valid configs: {len(valid)}", flush=True)
    return valid

@triton.autotune(configs=_BWD_CONFIGS,
                 key=["seq_len", "intermediate_size", "head_dim"],
                 prune_configs_by={"early_config_prune": _prune_bwd_configs})
@triton.jit
def flash_mlp_backward_kernel_dq(
    # Input tensors
    Q_ptr, K_ptr, U_ptr, V_ptr, dS_ptr,
    # Output gradients
    dQ_ptr,
    # Dimensions
    batch_size, num_heads, seq_len, head_dim, intermediate_size,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_head_stride, k_inter_stride, k_head_dim_stride,
    u_head_stride, u_inter_stride, u_head_dim_stride,
    v_head_stride, v_inter_stride, v_head_dim_stride,
    ds_batch_stride, ds_head_stride, ds_seq_stride, ds_head_dim_stride,
    # # Grad strides (same as forward, so we omit the parameter to save space)
    # dq_batch_stride, dq_head_stride, dq_seq_stride, dq_head_dim_stride,
    # dk_head_stride, dk_inter_stride, dk_head_dim_stride,
    # du_head_stride, du_inter_stride, du_head_dim_stride,
    # dv_head_stride, dv_inter_stride, dv_head_dim_stride,
    # Block sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_INTER: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KUV_GROUPS: tl.constexpr,
    LOG2_BLOCK_SEQ: tl.constexpr,
    BATCHLOOP_NUMSTAGE: tl.constexpr,
    INNERLOOP_NUMSTAGE: tl.constexpr,
):
    """Flash MLP Backward Kernel with non-atomic updates for dQ and atomic updates for dK/dU/dV
    
    Full Backward Formula:
        Given dS shape (batch_size, num_heads, seq_len, head_dim), and previous saved Q,K,U,V
        Q                                           ---  (batch_size, num_heads, seq_len, head_dim)
        K                                           ---  (num_heads, intermediate_size, head_dim)
        U                                           ---  (num_heads, intermediate_size, head_dim)
        V                                           ---  (num_heads, intermediate_size, head_dim)
        # we need to recompute M and N since we're not holding them during forward pass
        M = Q @ K^T * s0                            ---  (batch_size, num_heads, seq_len, intermediate_size)
        N = Q @ U^T * s1                            ---  (batch_size, num_heads, seq_len, intermediate_size)
        A = actfn(M) * N                            ---  (batch_size, num_heads, seq_len, intermediate_size)
        dV = A^T @ dS                               ---  (intermediate_size, head_dim)
        dK = s0 * V @ dS^T * N^T * actfn'(M^T) @ Q  ---  (intermediate_size, head_dim)
        dU = s1 * V @ dS^T * actfn(M^T) @ Q         ---  (intermediate_size, head_dim)
        dQ1 = s0 * dS @ V^T * N * silu'(M) @ K      ---  (batch_size, num_heads, seq_len, head_dim)
        dQ2 = s1 * dS @ V^T * silu(M) @ U           ---  (batch_size, num_heads, seq_len, head_dim)
        dQ = dQ1 + dQ2                              ---  (batch_size, num_heads, seq_len, head_dim)
        
    If we want to reuse the calculated result(this may require more smem), we can do:
        # Recompute M, N and A
        M = Q @ K^T * s0                            ---  (batch_size, num_heads, seq_len, intermediate_size)
        N = Q @ U^T * s1                            ---  (batch_size, num_heads, seq_len, intermediate_size)
        A = actfn(M) * N                            ---  (batch_size, num_heads, seq_len, intermediate_size)
        # Backward chain rule
        dA = dS @ V^T                               ---  (batch_size, num_heads, seq_len, intermediate_size)
        dN = dA * actfn(M)                          ---  (batch_size, num_heads, seq_len, intermediate_size)
        dM = (dA * N) * actfn'(M)                   ---  (batch_size, num_heads, seq_len, intermediate_size)
        # Parameter gradients (remember to accumulate over batch_size and seq_len dimensions)
        dV = A^T @ dS                               ---  (intermediate_size, head_dim)
        dK = s0 * dM^T @ Q                          ---  (intermediate_size, head_dim)
        dU = s1 * dN^T @ Q                          ---  (intermediate_size, head_dim)
        dQ = s0 * dM @ K + s1 * dN @ U              ---  (batch_size, num_heads, seq_len, head_dim)
    """
    # tl.max_constancy(head_scaling, 1)
    # tl.max_constancy(up_scaling,   1)
    tl.assume((BLOCK_SEQ & (BLOCK_SEQ - 1)) == 0)          # power-of-2
    tl.assume((BLOCK_INTER & (BLOCK_INTER - 1)) == 0)      # power-of-2
    tl.assume(BLOCK_SEQ % 16 == 0)
    tl.assume(BLOCK_INTER % 16 == 0)
    tl.assume(HEAD_DIM % 64 == 0)                          # or 16/32;
    tl.assume(q_head_dim_stride == 1)                      # contiguous head dim in Q/dQ
    tl.assume(ds_head_dim_stride == 1)                     # contiguous head dim in dS
    tl.assume(k_head_dim_stride == 1)                      # contiguous head dim in K/dK
    tl.assume(u_head_dim_stride == 1)
    tl.assume(v_head_dim_stride == 1)
    tl.assume(q_head_stride % 16 == 0)
    tl.assume(k_head_stride % 16 == 0)
    tl.assume(u_head_stride % 16 == 0)
    tl.assume(v_head_stride % 16 == 0)
    tl.assume(ds_head_stride % 16 == 0)
    tl.assume(k_inter_stride % 16 == 0)
    tl.assume(u_inter_stride % 16 == 0)
    tl.assume(v_inter_stride % 16 == 0)
    tl.assume(q_batch_stride % 16 == 0)
    tl.assume(ds_batch_stride % 16 == 0)
    
    # Constants
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    
    # Program IDs
    block_id = tl.program_id(0)
    kuv_head_id = tl.program_id(1)
    q_head_id = kuv_head_id * NUM_KUV_GROUPS
    batch_id = tl.program_id(2)
    # seq may exceed seqlen
    seq_start = block_id * BLOCK_SEQ
    if (seq_start > seq_len):
        return
    
    # Arange reuse for loading q and ds. Here we can use log accelerate the computation, will be implemented later
    row_ids = tl.arange(0, NUM_KUV_GROUPS * BLOCK_SEQ)
    col_ids = tl.arange(0, HEAD_DIM)[None, :] # [H]
    g = (row_ids >> LOG2_BLOCK_SEQ)[:, None]  # row_ids // BLOCK_SEQ
    s_local = row_ids & (BLOCK_SEQ - 1)       # row_ids % BLOCK_SEQ
    s = (seq_start + s_local)[:, None]
    tl.multiple_of(seq_start, BLOCK_SEQ)
    
    row_inter = tl.arange(0, BLOCK_INTER)[:, None]
    
    # Pointers for current block
    q_ptr = Q_ptr + batch_id * q_batch_stride + q_head_id * q_head_stride
    k_ptr = (
        K_ptr + kuv_head_id * k_head_stride
        + col_ids * k_head_dim_stride # remember to comment those lines out if option 2.1
    )
    u_ptr = (
        U_ptr + kuv_head_id * u_head_stride
        + col_ids * u_head_dim_stride # remember to comment those lines out if option 2.1
    )
    v_ptr = (
        V_ptr + kuv_head_id * v_head_stride
        + col_ids * v_head_dim_stride # remember to comment those lines out if option 2.1
    )
    
    ds_ptr = dS_ptr + batch_id * ds_batch_stride + q_head_id * ds_head_stride
    dq_ptr = dQ_ptr + batch_id * q_batch_stride + q_head_id * q_head_stride
    
    # triton compiler hints
    tl.multiple_of(q_ptr, 16)
    # tl.multiple_of(k_ptr, 16)         # option 2.1
    # tl.multiple_of(u_ptr, 16)         # option 2.1
    # tl.multiple_of(v_ptr, 16)         # option 2.1
    tl.multiple_of(k_ptr, [16, 16])   # option 2.2
    tl.multiple_of(u_ptr, [16, 16])   # option 2.2
    tl.multiple_of(v_ptr, [16, 16])   # option 2.2
    tl.multiple_of(ds_ptr, 16)
    tl.multiple_of(dq_ptr, 16)

    # ----------Option 1 - Make Block PTR -----------
    # Block pointers for loading, for Q block we need Q and Q^T while computing, so loading is normal
    # q_block_ptr_1 = tl.make_block_ptr(
    #     base=q_ptr,
    #     shape=(NUM_KUV_GROUPS, seq_len, HEAD_DIM),
    #     strides=(q_head_stride, q_seq_stride, q_head_dim_stride),
    #     offsets=(0, seq_start, 0),
    #     block_shape=(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM), # (NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM)
    #     order=(0, 1, 2)
    # )
    # q_block_1 = tl.load(q_block_ptr_1, boundary_check=(1,)).reshape(NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM) # (NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM)
    # # for ds, we load as normal, and load outside of loop(inside loop no change)
    # ds_ptr_block_1 = tl.make_block_ptr(
    #     base=ds_ptr,
    #     shape=(NUM_KUV_GROUPS, seq_len, HEAD_DIM),
    #     strides=(ds_head_stride, ds_seq_stride, ds_head_dim_stride),
    #     offsets=(0, seq_start, 0),
    #     block_shape=(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM),
    #     order=(0, 1, 2),  # load as (NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM)
    # )
    # ds_block_1 = tl.load(ds_ptr_block_1, boundary_check=(1,)).reshape(NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM)  # (NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM)
    # ----------Option 2 - Make Tile PTR -----------(experimentted that this is slightly faster)
    mask_rows = ((seq_start + s_local) < seq_len)[:, None]
    q_block_ptr_1 = (
        q_ptr
        + g * q_head_stride
        + s * q_seq_stride
        + col_ids * q_head_dim_stride
    )
    ds_ptr_block_1 = (
        ds_ptr
        + g * ds_head_stride
        + s * ds_seq_stride
        + col_ids * ds_head_dim_stride
    )
    tl.multiple_of(q_block_ptr_1, [16, 16])
    tl.multiple_of(ds_ptr_block_1, [16, 16])
    q_block_1 = tl.load(q_block_ptr_1, mask=mask_rows)
    ds_block_1 = tl.load(ds_ptr_block_1, mask=mask_rows)
    # dq accumulator
    dq_acc = tl.zeros((NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM), dtype=tl.float32)
    # ----------Option 2.1 - Make Block PTR -----------
    # Build block pointers for K, U, V using tl.make_block_ptr for efficient access
    # for k, u, we both need K, K^T and U, U^T, so load normally
    # k_ptr_block_1 = tl.make_block_ptr(
    #     base=k_ptr,
    #     shape=(intermediate_size, HEAD_DIM),
    #     strides=(k_inter_stride, k_head_dim_stride),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_INTER, HEAD_DIM),
    #     order=(0, 1),  # load as (BLOCK_INTER, HEAD_DIM)
    # )
    # u_ptr_block_1 = tl.make_block_ptr(
    #     base=u_ptr,
    #     shape=(intermediate_size, HEAD_DIM),
    #     strides=(u_inter_stride, u_head_dim_stride),
    #     offsets=(0, 0),
    #     block_shape=(BLOCK_INTER, HEAD_DIM),
    #     order=(0, 1),  # load as (BLOCK_INTER, HEAD_DIM)
    # )
    # # for v, we need only need V^T, so load as transpose
    # v_ptr_block_1 = tl.make_block_ptr(
    #     base=v_ptr,
    #     shape=(HEAD_DIM, intermediate_size),
    #     strides=(v_head_dim_stride, v_inter_stride),
    #     offsets=(0, 0),
    #     block_shape=(HEAD_DIM, BLOCK_INTER),
    #     order=(1, 0),  # load as (HEAD_DIM, BLOCK_INTER)
    # )
    # ----------Option 2.2 - Make Tile PTR -----------
    k_ptr_block_1 = k_ptr + row_inter * k_inter_stride
    u_ptr_block_1 = u_ptr + row_inter * u_inter_stride
    v_ptr_block_1 = v_ptr + row_inter * v_inter_stride
    for inter_start_ in tl.range(0, intermediate_size, BLOCK_INTER, num_stages=None):
        tl.multiple_of(inter_start_, BLOCK_INTER)
        tl.multiple_of(k_ptr_block_1, [16, 16])
        tl.multiple_of(u_ptr_block_1, [16, 16])
        tl.multiple_of(v_ptr_block_1, [16, 16])
        k_block_1 = tl.load(k_ptr_block_1, cache_modifier='.cg') # (BLOCK_INTER, HEAD_DIM), no bdry check
        u_block_1 = tl.load(u_ptr_block_1, cache_modifier='.cg') # (BLOCK_INTER, HEAD_DIM), no bdry check
        # vT_block_1 = tl.load(v_ptr_block_1, cache_modifier='.cg')  # (HEAD_DIM, BLOCK_INTER), no bdry check option 2.1
        vT_block_1 = tl.trans(tl.load(v_ptr_block_1, cache_modifier='.cg'))                                   # option 2.2
        
        # ----------Option 2.1 - Make Block PTR -----------
        # k_ptr_block_1 = tl.advance(k_ptr_block_1, (BLOCK_INTER, 0))
        # u_ptr_block_1 = tl.advance(u_ptr_block_1, (BLOCK_INTER, 0)) 
        # v_ptr_block_1 = tl.advance(v_ptr_block_1, (0, BLOCK_INTER))
        # ----------Option 2.2 - Make Tile PTR -----------
        k_ptr_block_1 += BLOCK_INTER * k_inter_stride
        u_ptr_block_1 += BLOCK_INTER * u_inter_stride
        v_ptr_block_1 += BLOCK_INTER * v_inter_stride

        # recompute M = Q @ K.T * s0, N = Q @ U.T * s1, A = silu(M) * N (also, we only need A^T)
        M = tl.dot(q_block_1, tl.trans(k_block_1))
        N = tl.dot(q_block_1, tl.trans(u_block_1)).to(tl.bfloat16)
        sigM = (1.0/(tl.exp2(-M * RCP_LN2)+1.0)).to(tl.bfloat16)
        M = M.to(tl.bfloat16)
        siluM = M * sigM
        # dsiluM = (siluM + sigM*(1-siluM)) # silu'(x) = sig + x*sig*(1-sig) = silu+sig(1-silu)
        # backpropagate dA = dS @ V^T, dN = dA * silu(M), dM = dA * N * silu'(M). silu'(x) = sigx+x*sigx*(1-sigx)
        dA = tl.dot(ds_block_1, vT_block_1).to(tl.bfloat16)
        dM = dA * N * (siluM + sigM*(1.0-siluM))
        dN = dA * siluM
        
        # originally dQ = tl.dot(dM, k_block) + tl.dot(dN, u_block)    # (BLOCK_SEQ, HEAD_DIM)
        dq_acc = tl.dot(dM, k_block_1, dq_acc)
        dq_acc = tl.dot(dN, u_block_1, dq_acc)
        
    # write accumulated dQ for this (batch, head, seq_block)
    # ----------Option 1 - Make Block PTR -----------
    # dq_block_ptr = tl.make_block_ptr(
    #     base=dq_ptr,
    #     shape=(NUM_KUV_GROUPS, seq_len, HEAD_DIM),
    #     strides=(q_head_stride, q_seq_stride, q_head_dim_stride),
    #     offsets=(0, seq_start, 0),
    #     block_shape=(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM),
    #     order=(0, 1, 2)
    # )
    # tl.store(dq_block_ptr, dq_acc.reshape(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM), boundary_check=(1,))
    # ----------Option 2 - Make Tile PTR -----------
    dq_block_ptr = (
        dq_ptr
        + g * q_head_stride
        + s * q_seq_stride
        + col_ids * q_head_dim_stride
    )
    tl.store(dq_block_ptr, dq_acc.to(tl.bfloat16), mask=mask_rows, cache_modifier=".cg", eviction_policy="")


@triton.autotune(configs=_BWD_CONFIGS,
                 key=["seq_len", "intermediate_size", "head_dim"],
                 prune_configs_by={"early_config_prune": _prune_bwd_configs})
@triton.jit
def flash_mlp_backward_kernel_dkdudv(
    # Input tensors
    Q_ptr, K_ptr, U_ptr, V_ptr, dS_ptr,
    # Output gradients
    dK_ptr, dU_ptr, dV_ptr,
    # Dimensions
    batch_size, num_heads, seq_len, head_dim, intermediate_size,
    q_batch_stride, q_head_stride, q_seq_stride, q_head_dim_stride,
    k_head_stride, k_inter_stride, k_head_dim_stride,
    u_head_stride, u_inter_stride, u_head_dim_stride,
    v_head_stride, v_inter_stride, v_head_dim_stride,
    ds_batch_stride, ds_head_stride, ds_seq_stride, ds_head_dim_stride,
    # # Grad strides (same as forward, so we omit the parameter to save space)
    # dq_batch_stride, dq_head_stride, dq_seq_stride, dq_head_dim_stride,
    # dk_head_stride, dk_inter_stride, dk_head_dim_stride,
    # du_head_stride, du_inter_stride, du_head_dim_stride,
    # dv_head_stride, dv_inter_stride, dv_head_dim_stride,
    # Configuration
    head_scaling: tl.constexpr,
    up_scaling: tl.constexpr,
    # Block sizes
    BLOCK_SEQ: tl.constexpr,
    BLOCK_INTER: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_KUV_GROUPS: tl.constexpr,
    LOG2_BLOCK_SEQ: tl.constexpr,
    BATCH_SIZE: tl.constexpr,
    BATCHLOOP_NUMSTAGE: tl.constexpr,
    INNERLOOP_NUMSTAGE: tl.constexpr,
):
    # ----
    # tl.max_constancy(head_scaling, 1)
    # tl.max_constancy(up_scaling,   1)
    tl.assume(head_scaling == up_scaling)
    tl.assume((BLOCK_SEQ & (BLOCK_SEQ - 1)) == 0)          # power-of-2
    tl.assume((BLOCK_INTER & (BLOCK_INTER - 1)) == 0)      # power-of-2
    tl.assume(BLOCK_SEQ % 16 == 0)
    tl.assume(BLOCK_INTER % 16 == 0)
    tl.assume(HEAD_DIM % 64 == 0)                          # or 16/32;
    tl.assume(q_head_dim_stride == 1)                      # contiguous head dim in Q/dQ
    tl.assume(ds_head_dim_stride == 1)                     # contiguous head dim in dS
    tl.assume(k_head_dim_stride == 1)                      # contiguous head dim in K/dK
    tl.assume(u_head_dim_stride == 1)
    tl.assume(v_head_dim_stride == 1)
    tl.assume(q_head_stride % 16 == 0)
    tl.assume(k_head_stride % 16 == 0)
    tl.assume(u_head_stride % 16 == 0)
    tl.assume(v_head_stride % 16 == 0)
    tl.assume(ds_head_stride % 16 == 0)
    tl.assume(k_inter_stride % 16 == 0)
    tl.assume(u_inter_stride % 16 == 0)
    tl.assume(v_inter_stride % 16 == 0)
    tl.assume(q_batch_stride % 16 == 0)
    tl.assume(ds_batch_stride % 16 == 0)
    
    # Constants
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    
    # Program IDs
    block_id = tl.program_id(0)
    kuv_head_id = tl.program_id(1)
    # no batch id becuase it's always 0
    q_head_id = kuv_head_id * NUM_KUV_GROUPS
    
    # Calculate ranges, we merge seq and inter calc into one kernel
    inter_start = block_id * BLOCK_INTER
    
    # Arange reuse for loading q and ds. Here we can use log accelerate the computation, will be implemented later
    row_ids = tl.arange(0, NUM_KUV_GROUPS * BLOCK_SEQ)
    col_ids = tl.arange(0, HEAD_DIM)[None, :] # [H]
    g = (row_ids >> LOG2_BLOCK_SEQ)[:, None]  # row_ids // BLOCK_SEQ
    s_local = row_ids & (BLOCK_SEQ - 1)       # row_ids % BLOCK_SEQ
    
    # Pointers for current block
    q_ptr = (
        Q_ptr + q_head_id * q_head_stride
        + g * q_head_stride           # remember to comment those lines out if option 1
        + col_ids * q_head_dim_stride # remember to comment those lines out if option 1
    )
    k_ptr = K_ptr + kuv_head_id * k_head_stride
    u_ptr = U_ptr + kuv_head_id * u_head_stride
    v_ptr = V_ptr + kuv_head_id * v_head_stride
    
    ds_ptr = (
        dS_ptr + q_head_id * ds_head_stride
        + g * ds_head_stride           # remember to comment those lines out if option 1
        + col_ids * ds_head_dim_stride # remember to comment those lines out if option 1
    )
    
    dk_ptr = dK_ptr + kuv_head_id * k_head_stride
    du_ptr = dU_ptr + kuv_head_id * u_head_stride
    dv_ptr = dV_ptr + kuv_head_id * v_head_stride
    
    # triton compiler hints
    tl.multiple_of(q_ptr, [16,16])
    tl.multiple_of(k_ptr, 16)
    tl.multiple_of(u_ptr, 16)
    tl.multiple_of(v_ptr, 16)
    tl.multiple_of(ds_ptr, [16,16])
    tl.multiple_of(dk_ptr, 16)
    tl.multiple_of(du_ptr, 16)
    tl.multiple_of(dv_ptr, 16)
    
    k_block_ptr_2 = tl.make_block_ptr(
        base=k_ptr,
        shape=(intermediate_size, HEAD_DIM),
        strides=(k_inter_stride, k_head_dim_stride),
        offsets=(inter_start, 0),
        block_shape=(BLOCK_INTER, HEAD_DIM),
        order=(0, 1),  # load as (BLOCK_INTER, HEAD_DIM)
    )
    u_block_ptr_2 = tl.make_block_ptr(
        base=u_ptr,
        shape=(intermediate_size, HEAD_DIM),
        strides=(u_inter_stride, u_head_dim_stride),
        offsets=(inter_start, 0),
        block_shape=(BLOCK_INTER, HEAD_DIM),
        order=(0, 1),  # load as (BLOCK_INTER, HEAD_DIM)
    )
    # For V, we need V^T, so load as transpose
    vT_block_ptr_2 = tl.make_block_ptr(
        base=v_ptr,
        shape=(HEAD_DIM, intermediate_size),
        strides=(v_head_dim_stride, v_inter_stride),
        offsets=(0, inter_start),
        block_shape=(HEAD_DIM, BLOCK_INTER),
        order=(1, 0),  # load as (HEAD_DIM, BLOCK_INTER)
    )
    k_block_2 = tl.load(k_block_ptr_2, cache_modifier='.ca')
    u_block_2 = tl.load(u_block_ptr_2, cache_modifier='.ca')
    vT_block_2 = tl.load(vT_block_ptr_2, cache_modifier='.ca')
    # dk, du, dv accumulator
    dk_acc = tl.zeros((BLOCK_INTER, HEAD_DIM), dtype=tl.float32)
    du_acc = tl.zeros((BLOCK_INTER, HEAD_DIM), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_INTER, HEAD_DIM), dtype=tl.float32)
    q_ptr_ = q_ptr - q_batch_stride         # prepare q_ptr_
    ds_ptr_ = ds_ptr - ds_batch_stride      # prepare ds_ptr_
    # loop over seq_len
    for _ in tl.range(0, BATCH_SIZE, 1, num_stages=None, loop_unroll_factor=None): # loop_unroll_factor=BATCH_SIZE makes it even slower?
        q_ptr_ += q_batch_stride
        ds_ptr_ += ds_batch_stride
        tl.multiple_of(q_ptr_, [16,16])
        tl.multiple_of(ds_ptr_, [16,16])
        for seq_start_ in tl.range(0, seq_len, BLOCK_SEQ, num_stages=None):
            tl.multiple_of(seq_start_, BLOCK_SEQ)
            tl.multiple_of(q_ptr_, [16,16])
            tl.multiple_of(ds_ptr_, [16,16])
            # ----------Option 1 - Make Block PTR -----------
            # Build block pointers for Q and dS, both load normally
            # q_block_ptr_2 = tl.make_block_ptr(
            #     base=q_ptr_,
            #     shape=(NUM_KUV_GROUPS, seq_len, HEAD_DIM),
            #     strides=(q_head_stride, q_seq_stride, q_head_dim_stride),
            #     offsets=(0, seq_start_, 0),
            #     block_shape=(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM), # load as (NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM)
            #     order=(0, 1, 2)
            # )
            # ds_block_ptr_2 = tl.make_block_ptr(
            #     base=ds_ptr_,
            #     shape=(NUM_KUV_GROUPS, seq_len, HEAD_DIM),
            #     strides=(ds_head_stride, ds_seq_stride, ds_head_dim_stride),
            #     offsets=(0, seq_start_, 0),
            #     block_shape=(NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM), # load as (NUM_KUV_GROUPS, BLOCK_SEQ, HEAD_DIM)
            #     order=(0, 1, 2)
            # )
            # q_block_2 = tl.load(q_block_ptr_2, boundary_check=(1,)).reshape(NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM)
            # ds_block_2 = tl.load(ds_block_ptr_2, boundary_check=(1,)).reshape(NUM_KUV_GROUPS * BLOCK_SEQ, HEAD_DIM)
            # ----------Option 2 - Tile PTR -----------(experimented that this is faster)
            s_ = seq_start_ + s_local
            mask_rows = (s_ < seq_len)[:, None]
            q_block_ptr_2 = (
                q_ptr_
                + s_[:, None] * q_seq_stride
            )
            ds_block_ptr_2 = (
                ds_ptr_
                + s_[:, None] * ds_seq_stride
            )
            tl.multiple_of(q_block_ptr_2, [16,16])
            tl.multiple_of(ds_block_ptr_2, [16,16])
            q_block_2 = tl.load(q_block_ptr_2, mask=mask_rows, cache_modifier=".cg", eviction_policy='')
            ds_block_2 = tl.load(ds_block_ptr_2, mask=mask_rows, cache_modifier=".cg", eviction_policy='')

            # recompute M = Q @ K.T * s0, N = Q @ U.T * s1, A = silu(M) * N (also, we only need A^T)
            M = tl.dot(q_block_2, tl.trans(k_block_2)) # (BLOCK_SEQ, BLOCK_INTER) dtype float32
            N = tl.dot(q_block_2, tl.trans(u_block_2)).to(tl.bfloat16)
            sigM = (1.0/(tl.exp2(-M * RCP_LN2)+1.0)).to(tl.bfloat16)
            M = M.to(tl.bfloat16)
            siluM = M * sigM
            # dsiluM = (siluM + sigM*(1-siluM)) # silu'(x) = sig + x*sig*(1-sig) = silu+sig(1-silu)
            AT = tl.trans(siluM * N)
            # backpropagate dA = dS @ V^T, dN = dA * silu(M), dM = dA * N * silu'(M). silu'(x) = sigx+x*sigx*(1-sigx)
            dA = tl.dot(ds_block_2, vT_block_2).to(tl.bfloat16)
            dM = dA * N * (siluM + sigM*(1.0-siluM))
            dN = dA * siluM
        
            dv_acc = tl.dot(AT, ds_block_2, dv_acc)                   # (BLOCK_INTER, HEAD_DIM)
            dk_acc = tl.dot(tl.trans(dM) * head_scaling, q_block_2, dk_acc) # (BLOCK_INTER, HEAD_DIM)
            du_acc = tl.dot(tl.trans(dN) * up_scaling, q_block_2, du_acc)   # (BLOCK_INTER, HEAD_DIM)
    
    # write accumulated dK, du, dv for this (inter_block, head_dim). all load normally, no transpose load
    dk_block_ptr = tl.make_block_ptr(
        base=dk_ptr,
        shape=(intermediate_size, HEAD_DIM),
        strides=(k_inter_stride, k_head_dim_stride),
        offsets=(inter_start, 0),
        block_shape=(BLOCK_INTER, HEAD_DIM), # load as (BLOCK_INTER, HEAD_DIM)
        order=(0, 1)
    )
    du_block_ptr = tl.make_block_ptr(
        base=du_ptr,
        shape=(intermediate_size, HEAD_DIM),
        strides=(u_inter_stride, u_head_dim_stride),
        offsets=(inter_start, 0),
        block_shape=(BLOCK_INTER, HEAD_DIM),
        order=(0, 1)
    )
    dv_block_ptr = tl.make_block_ptr(
        base=dv_ptr,
        shape=(intermediate_size, HEAD_DIM),
        strides=(v_inter_stride, v_head_dim_stride),
        offsets=(inter_start, 0),
        block_shape=(BLOCK_INTER, HEAD_DIM),
        order=(0, 1)
    )
    tl.store(dk_block_ptr, dk_acc.to(tl.bfloat16), cache_modifier=".cg", eviction_policy='')
    tl.store(du_block_ptr, du_acc.to(tl.bfloat16), cache_modifier=".cg", eviction_policy='')
    tl.store(dv_block_ptr, dv_acc.to(tl.bfloat16), cache_modifier=".cg", eviction_policy='')

class FlashMLPFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, U, V, apply_dot_scaling, head_scaling, up_scaling):
        """
        Flash MLP Forward Function
        
        Args:
            Q: (batch_size, num_heads, seq_len, head_dim)
            K: (num_heads, intermediate_size, head_dim)
            U: (num_heads, intermediate_size, head_dim) or None
            V: (num_heads, intermediate_size, head_dim)
            apply_dot_scaling: bool
            head_scaling: float
            up_scaling: float
        
        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, num_heads, seq_len, head_dim = Q.shape
        num_kuv_heads, intermediate_size, _ = K.shape
        num_kuv_groups = num_heads // num_kuv_heads
        
        expected_dtype = torch.bfloat16
        if Q.dtype != expected_dtype or K.dtype != expected_dtype or V.dtype != expected_dtype or U.dtype != expected_dtype:
            raise TypeError(
                f"FlashMLPFunction expects all inputs in {expected_dtype} for optimal performance, got"
                f" Q={Q.dtype}, K={K.dtype}, V={V.dtype}, U={U.dtype}"
            )
        
        # Output tensor
        output = torch.empty_like(Q)

        # Autotuned kernel uses META to determine optimal block sizes.
        def grid(META):
            # to implement GQFFN, we launch kernel at num_kuv_heads dim and in each thread load num_kuv_groups times q_block
            return (triton.cdiv(seq_len, META['BLOCK_SEQ']), num_kuv_heads, batch_size)

        # Launch kernel
        flash_mlp_forward_kernel[grid](
            Q, K, U, V, output,
            batch_size, num_heads, seq_len, head_dim, intermediate_size,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            K.stride(0), K.stride(1), K.stride(2),
            U.stride(0), U.stride(1), U.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            HEAD_DIM=head_dim,  # compile-time constant; BLOCK sizes provided by autotune config
            NUM_KUV_GROUPS=num_kuv_groups
        )

        # Save for backward (keep heuristic guess for backward tiling)
        ctx.save_for_backward(Q, K, U, V)
        ctx.apply_dot_scaling = apply_dot_scaling
        ctx.head_scaling = head_scaling
        ctx.up_scaling = up_scaling

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass with optimized Triton kernel"""
        Q, K, U, V = ctx.saved_tensors
        # K *= ctx.head_scaling  # kblock involved computation always need * headscaling
        # U *= ctx.up_scaling  # ublock involved computation always need * upscaling
        assert grad_output.shape == Q.shape, "grad_output shape should be the same as Q"
        batch_size, num_heads, seq_len, head_dim = Q.shape
        num_kuv_heads, intermediate_size, _ = K.shape
        num_kuv_groups = num_heads // num_kuv_heads
        
        # Initialize gradients
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dU = torch.empty_like(U)
        dV = torch.empty_like(V)

        # Grid configuration for backward kernel
        def grid(meta):
            BLOCK_SEQ = meta['BLOCK_SEQ']
            grid_seq = triton.cdiv(seq_len, BLOCK_SEQ)
            return (grid_seq, num_kuv_heads, batch_size)
        
        def grid2(meta):
            BLOCK_INTER = meta['BLOCK_INTER']
            grid_inter = triton.cdiv(intermediate_size, BLOCK_INTER)
            return (grid_inter, num_kuv_heads, 1)
        
        
        # print(f"strides: {Q.stride()}, {K.stride()}, {U.stride()}, {V.stride()} \n {grad_output.stride()}, {grad_output.shape} {grad_output}")
        
        kernel = flash_mlp_backward_kernel_dq[grid]
        kernel2 = flash_mlp_backward_kernel_dkdudv[grid2]
        # Launch backward kernel
        kernel(
            Q, K, U, V, grad_output,
            dQ,
            batch_size, num_heads, seq_len, head_dim, intermediate_size,
            *Q.stride(),
            *K.stride(),
            *U.stride(),
            *V.stride(),
            *grad_output.stride(),
            HEAD_DIM=head_dim,
            NUM_KUV_GROUPS=num_kuv_groups,
            # BLOCK_SEQ=64,
            # BLOCK_INTER=64,
        )
        kernel2(
            Q, K, U, V, grad_output,
            dK, dU, dV,
            batch_size, num_heads, seq_len, head_dim, intermediate_size,
            *Q.stride(),
            *K.stride(),
            *U.stride(),
            *V.stride(),
            *grad_output.stride(),
            ctx.head_scaling, ctx.up_scaling,
            HEAD_DIM=head_dim,
            NUM_KUV_GROUPS=num_kuv_groups,
            BATCH_SIZE=batch_size,
            # BLOCK_SEQ=64,
            # BLOCK_INTER=64,
        )
        
        return dQ, dK, dU, dV, None, None, None


class FlashMLP(torch.nn.Module):
    """
    Flash MLP: Memory-efficient implementation of Multi-Head FFN computation
    
    This implementation uses the same memory optimization principles as Flash Attention:
    1. Tiling: Compute in blocks to fit in SRAM
    2. Recomputation: Recompute forward values during backward pass
    3. Online computation: Avoid materializing large intermediate tensors
    
    Memory complexity: O(batch_size * num_heads * seq_len * head_dim) 
    instead of O(batch_size * num_heads * seq_len * intermediate_size)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, U, V, apply_dot_scaling, head_scaling, up_scaling, activation_fn):
        """
        Forward pass of Flash MLP
        
        Args:
            Q: (batch_size, num_heads, seq_len, head_dim)
            K: (num_heads, intermediate_size, head_dim)  
            U: (num_heads, intermediate_size, head_dim) or None
            V: (num_heads, intermediate_size, head_dim)
            skip_linear: bool (not used in this kernel)
            apply_dot_scaling: bool
            head_scaling: float
            up_scaling: float
        
        Returns:
            output: (batch_size, num_heads, seq_len, head_dim)
        """
        if activation_fn != 'silu':
            raise ValueError("FlashMLP currently supports only SiLU activation")
        if apply_dot_scaling or head_scaling != 1.0 or up_scaling != 1.0:
            raise ValueError("FlashMLP does not support dot scaling(this is proved to be a bad design, if you stick using it, please use torch implementation)")
        
        # For SiLU gated MLP we always expect an up projection matrix U.
        if U is None:
            raise ValueError("FlashMLP SiLU variant requires up-projection tensor U (got None)")
        return FlashMLPFunction.apply(
            Q, K, U, V, apply_dot_scaling, head_scaling, up_scaling
        ) 