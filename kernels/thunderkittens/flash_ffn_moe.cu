/**
* FlashFFN-MoE: Memory-Efficient Multi-Head Feed-Forward Networks with Mixture of Experts
* Optimized for H100 using ThunderKittens framework
* This version only supports KUV Groups = 1. 
* This version only supports seqlen % (BLOCK_SEQ*CON_WARPGRPS) == 0. THIS IS A SEVERE PROBLEM. Will be fixed later.
* 
* Algorithm: O = Σ_e R_e * ((SiLU(Q @ K_e^T) ⊙ (Q @ U_e^T)) @ V_e)
* 
* Full implementation for HEAD_DIM=128
*/
// todo: dqdr can be initialized to empty rather than zero
#include "kittens.cuh"
#include "ops/group/group.cuh"
#include "ops/warp/memory/vec/shared_to_register.cuh"
#include "utils.cuh"
#include <cooperative_groups.h>

using namespace kittens;
namespace cg = cooperative_groups;
#define USE_FAST_SIGMOID
// #define COMPUTE_DKDUDV
// #define COPMUTE_DQDR
// #define USE_DIRECT_SILU
constexpr int HEAD_DIM = 128;
// constexpr float RCP_LN2 = 1.4426950408889634f;  // = 1.0 / ln(2)
// ============================================================================
// Forward Pass Configuration for HEAD_DIM=128 (0bytes spill load)
// ============================================================================
// 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
// ptxas info     : Used 128 registers, 128 bytes smem
// nvlink info    : 0 bytes gmem
struct fwd_config {
    static constexpr int BLOCK_SEQ = (4*16);
    static constexpr int BLOCK_INTER = 64;
    static constexpr int NUM_STAGES = 3;
    static constexpr int CONSUMER_WARPGROUPS = 3;
    static constexpr int PRODUCER_WARPGROUPS = 1; // TODO: current code only supports producer=1
    static constexpr int NUM_WARPGROUPS = (CONSUMER_WARPGROUPS + PRODUCER_WARPGROUPS);
    static constexpr int NUM_WORKERS = (NUM_WARPGROUPS * kittens::WARPGROUP_WARPS);
};

struct fwd_globals {
    using cfg = fwd_config;
    using block_q = st_bf<cfg::BLOCK_SEQ, HEAD_DIM>;
    using block_k = st_bf<cfg::BLOCK_INTER, HEAD_DIM>;
    using block_u = st_bf<cfg::BLOCK_INTER, HEAD_DIM>;
    using block_v = st_bf<cfg::BLOCK_INTER, HEAD_DIM>;
    using block_o = st_bf<cfg::BLOCK_SEQ, HEAD_DIM>;    // still use st_bf for saving smem
    using vector_r = sv_fl<cfg::BLOCK_SEQ>;

    using q_gl = gl<bf16, -1, -1, -1, -1, block_q>; // (batchsize, num_heads, seq_len, head_dim)
    using k_gl = gl<bf16, -1, -1, -1, -1, block_k>; // (kuv_heads, num_experts, expert_dim, head_dim)
    using u_gl = gl<bf16, -1, -1, -1, -1, block_u>; // (kuv_heads, num_experts, expert_dim, head_dim)
    using v_gl = gl<bf16, -1, -1, -1, -1, block_v>; // (kuv_heads, num_experts, expert_dim, head_dim)
    using o_gl = gl<bf16, -1, -1, -1, -1, block_o>; // (batchsize, num_heads, seq_len, head_dim)
    using r_gl = gl<float, -1, -1, -1, -1, vector_r>; // (batchsize, num_heads, seq_len, num_experts)

    q_gl Q;
    k_gl K;
    u_gl U;
    v_gl V;
    o_gl O;
    r_gl R;
    // bf16* R;
    
    // some call-defined constants
    const int batch_size;
    const int num_heads;
    const int num_kuv_heads;
    const int seq_len;
    const int num_experts;
    const int expert_dim;

};

// ============================================================================
// Forward Kernel
// ============================================================================

__global__ __launch_bounds__(fwd_config::NUM_WORKERS * kittens::WARP_THREADS, 1)
void flashffn_moe_fwd_kernel_head128(const __grid_constant__ fwd_globals globals) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(); int warpgroupid = warpid/kittens::WARPGROUP_WARPS;
    using cfg = fwd_config;
    using q_tile = fwd_globals::block_q;
    using k_tile = fwd_globals::block_k;
    using u_tile = fwd_globals::block_u;
    using v_tile = fwd_globals::block_v;
    using o_tile = fwd_globals::block_o;
    using r_vect = fwd_globals::vector_r;

    q_tile (&q_smem)[fwd_config::CONSUMER_WARPGROUPS]    = al.allocate<q_tile, fwd_config::CONSUMER_WARPGROUPS>();
    r_vect (&r_smem)[fwd_config::CONSUMER_WARPGROUPS]    = al.allocate<r_vect, fwd_config::CONSUMER_WARPGROUPS>();
    k_tile (&k_smem)[fwd_config::NUM_STAGES] = al.allocate<k_tile, fwd_config::NUM_STAGES>();
    u_tile (&u_smem)[fwd_config::NUM_STAGES] = al.allocate<u_tile, fwd_config::NUM_STAGES>();
    v_tile (&v_smem)[fwd_config::NUM_STAGES] = al.allocate<v_tile, fwd_config::NUM_STAGES>();
    auto (*o_smem)                 = reinterpret_cast<o_tile(*)>(q_smem); // reuse q_smem for o_smem

    const int tiles_per_expert = globals.expert_dim / fwd_config::BLOCK_INTER;
    const int total_inter_blocks = tiles_per_expert * globals.num_experts;
    const int head_id = blockIdx.y; const int kuv_head_id = blockIdx.y;
    const int seq_id = blockIdx.x * fwd_config::CONSUMER_WARPGROUPS;
    const int batch_id = blockIdx.z;

    __shared__ kittens::semaphore qsmem_semaphore, r_arrived;
    __shared__ kittens::semaphore k_smem_arrived[fwd_config::NUM_STAGES], u_smem_arrived[fwd_config::NUM_STAGES], v_smem_arrived[fwd_config::NUM_STAGES];
    __shared__ kittens::semaphore compute_done[fwd_config::NUM_STAGES], r_used;

    // semaphore is on shared memory, we only need to initialize once
    if (threadIdx.x == 0) { 
        // qsmem only need to be fetched once
        init_semaphore(qsmem_semaphore, 0, 1);
        init_semaphore(r_arrived, 0, 1);
        init_semaphore(r_used, fwd_config::CONSUMER_WARPGROUPS, 0);
        for (int j = 0; j < fwd_config::NUM_STAGES; ++j) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(u_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], fwd_config::CONSUMER_WARPGROUPS, 0);
        }
        // pipeline warmup: first fetch is done after initialization of semaphores
        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));
        for (int warpgroup = 0; warpgroup != fwd_config::CONSUMER_WARPGROUPS; ++warpgroup) {
            coord<q_tile> q_tile_idx = {batch_id, head_id, seq_id + warpgroup, 0};
            tma::load_async(q_smem[warpgroup], globals.Q, q_tile_idx, qsmem_semaphore);
        }
        const int warmup_expert_id = 0; // warmup prefetch expert 0
        tma::expect_bytes(r_arrived, sizeof(r_smem));
        for (int warpgroup = 0; warpgroup != fwd_config::CONSUMER_WARPGROUPS; ++warpgroup) {
            coord<r_vect> r_vec_idx = {batch_id, head_id, warmup_expert_id, seq_id + warpgroup};
            tma::load_async(r_smem[warpgroup], globals.R, r_vec_idx, r_arrived);
        }
        const int warmup_tiles = min(total_inter_blocks, fwd_config::NUM_STAGES);
        for (int inter_tile = 0; inter_tile != warmup_tiles; ++inter_tile) {
            int expert_id = inter_tile / tiles_per_expert;
            int real_inter_tile = inter_tile % tiles_per_expert;
            // during warmup, inter_tile=stage is just a coincidence, for later staging pipeline work fluently
            int stage = inter_tile;
            coord<k_tile> kuv_tile_idx = {kuv_head_id, expert_id, real_inter_tile, 0};
            tma::expect_bytes(k_smem_arrived[stage], sizeof(k_tile));
            tma::load_async(k_smem[stage], globals.K, kuv_tile_idx, k_smem_arrived[stage]);
            tma::expect_bytes(u_smem_arrived[stage], sizeof(u_tile));
            tma::load_async(u_smem[stage], globals.U, kuv_tile_idx, u_smem_arrived[stage]);
            tma::expect_bytes(v_smem_arrived[stage], sizeof(v_tile));
            tma::load_async(v_smem[stage], globals.V, kuv_tile_idx, v_smem_arrived[stage]);
        }
    }
    __syncthreads();

    // warp 0 for K,U,V loading
    if(warpgroupid == fwd_config::NUM_WARPGROUPS-1) { 
        warpgroup::decrease_registers<32>();
        // warp 0 for K,V loading
        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            const int warmup_tiles = min(total_inter_blocks, fwd_config::NUM_STAGES);
            for (int inter_tile = warmup_tiles; inter_tile != total_inter_blocks; ++inter_tile) {
                int expert_id = inter_tile / tiles_per_expert;
                int real_inter_tile = inter_tile % tiles_per_expert;
                // prefetch the next stage
                int stage_next = inter_tile % fwd_config::NUM_STAGES;
                // OPTION -- 1 -- Directly wait for the slot we are launching new load, this is the most straightforward 
                wait(compute_done[stage_next], ((inter_tile / fwd_config::NUM_STAGES)-1) & 1);
                coord<k_tile> kuv_tile_idx = {kuv_head_id, expert_id, real_inter_tile, 0};
                tma::expect_bytes(k_smem_arrived[stage_next], sizeof(k_tile));
                tma::load_async(k_smem[stage_next], globals.K, kuv_tile_idx, k_smem_arrived[stage_next]);
                tma::expect_bytes(v_smem_arrived[stage_next], sizeof(v_tile));
                tma::load_async(v_smem[stage_next], globals.V, kuv_tile_idx, v_smem_arrived[stage_next]);
                // OPTION -- 2 -- Use the same waiting logic in H100.cu -- ABANDONED
            }
        // warp 1 for U loading
        } else if (warpid % kittens::WARPGROUP_WARPS == 1) {
            const int warmup_tiles = min(total_inter_blocks, fwd_config::NUM_STAGES);
            for (int inter_tile = warmup_tiles; inter_tile != total_inter_blocks; ++inter_tile) {
                int expert_id = inter_tile / tiles_per_expert;
                int real_inter_tile = inter_tile % tiles_per_expert;
                // prefetch the next stage
                int stage_next = inter_tile % fwd_config::NUM_STAGES;
                // OPTION -- 1 -- Directly wait for the slot we are launching new load, this is the most straightforward 
                wait(compute_done[stage_next], ((inter_tile / fwd_config::NUM_STAGES)-1) & 1);
                coord<k_tile> kuv_tile_idx = {kuv_head_id, expert_id, real_inter_tile, 0};
                tma::expect_bytes(u_smem_arrived[stage_next], sizeof(u_tile));
                tma::load_async(u_smem[stage_next], globals.U, kuv_tile_idx, u_smem_arrived[stage_next]);
                // OPTION -- 2 -- Use the same waiting logic in H100.cu -- ABANDONED
            }
        // warp 2 for R loading
        } else if (warpid % kittens::WARPGROUP_WARPS == 2) {
            for (int expert_id = 1; expert_id != globals.num_experts; ++expert_id) {
                wait(r_used, (expert_id - 1) & 1);
                tma::expect_bytes(r_arrived, sizeof(r_smem));
                for (int warpgroup = 0; warpgroup != fwd_config::CONSUMER_WARPGROUPS; ++warpgroup) {
                    coord<r_vect> r_vec_idx = {batch_id, head_id, expert_id, seq_id + warpgroup};
                    tma::load_async(r_smem[warpgroup], globals.R, r_vec_idx, r_arrived);
                }
                
            }
        }
    // consumer warp
    } else {
        warpgroup::increase_registers<160>();

        
        rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, HEAD_DIM> o_block;
        rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, HEAD_DIM> q_reg;
        // rt_fl<16, HEAD_DIM> o_e; // o for each expert
        rv_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> r_vec;
        rv_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> r_vec_bf;
        zero(o_block);
        // zero(o_e);
        wait(qsmem_semaphore, 0);
        warpgroup::load(q_reg, q_smem[warpgroupid]);

        rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, fwd_config::BLOCK_INTER> R1_bf;
        for (int inter_tile = 0; inter_tile != total_inter_blocks; ++inter_tile) {
            const int stage = inter_tile % fwd_config::NUM_STAGES;
            const int phase = (inter_tile / fwd_config::NUM_STAGES) & 1;
        
            // compute m = Q @ K.T and n = Q @ U.T asynchronously
            {
                rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, fwd_config::BLOCK_INTER> R1;
                wait(k_smem_arrived[stage], phase);
                warpgroup::mm_ABt(R1, q_reg, k_smem[stage]);
            
                // if on the edge of expert
                if (inter_tile % tiles_per_expert == 0) {
                    const int expert_id = inter_tile / tiles_per_expert;
                    wait(r_arrived, expert_id & 1);
                    warpgroup::load(r_vec, r_smem[warpgroupid]);
                    if (warpgroup::laneid() == 0) {
                        arrive(r_used, 1);
                    }
                    copy(r_vec_bf, r_vec);
                    // during the next all tiles corresponding to this expert, the expert weight is in r_vec
                }

                warpgroup::mma_async_wait();
                copy(R1_bf, R1);
            }
            #ifdef USE_DIRECT_SILU
            {
                rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, fwd_config::BLOCK_INTER> R2;
                rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, fwd_config::BLOCK_INTER> R2_bf;
                wait(u_smem_arrived[stage], phase);
                warpgroup::mm_ABt(R2, q_reg, u_smem[stage]);
                warpgroup::mma_commit_group();
                fast_silu_bf16(R1_bf,R1_bf);
                mul_row(R1_bf,R1_bf,r_vec_bf); // R1 multiply by expert weight, it's mathematically equivalent
                warpgroup::mma_async_wait();
                copy(R2_bf, R2);
                mul(R1_bf, R1_bf, R2_bf); // R1 = Re * silu(q.k) * (q.u), shape (BLOCK_SEQ, BLOCK_INTER)
            }
            #else
            {
                rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, fwd_config::BLOCK_INTER> R2;
                rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, fwd_config::BLOCK_INTER> R2_bf;
                wait(u_smem_arrived[stage], phase);
                warpgroup::mm_ABt(R2, q_reg, u_smem[stage]);
                warpgroup::mma_commit_group();
                #ifdef USE_FAST_SIGMOID
                    fast_sigmoid_bf16(R2_bf,R1_bf); // R2=sigmoid(R1)
                #else
                    mul(R2, R1, -RCP_LN2);  // R2=(-q.k * rcpln2), ready to get exp2
                    exp2(R2, R2); // R2 = exp(-q.k)
                    add(R2, R2, 1.0f); // R2 = exp(-q.k)+1
                    rcp_inplace_approx(R2); // R2 = 1 / (exp(-q.k)+1)=sigmoid(R1)
                #endif
                mul(R1_bf, R1_bf, R2_bf); // R1 = q.k * 1 / (exp(-q.k)+1) = silu(q.k)
                mul_row(R1_bf,R1_bf,r_vec_bf); // R1 multiply by expert weight, it's mathematically equivalent
                warpgroup::mma_async_wait();
                copy(R2_bf, R2);
                mul(R1_bf, R1_bf, R2_bf); // R1 = Re * silu(q.k) * (q.u), shape (BLOCK_SEQ, BLOCK_INTER)
            }
            #endif
            {
                wait(v_smem_arrived[stage], phase);
                warpgroup::mma_AB(o_block, R1_bf, v_smem[stage]); // use mma_AB here for accumulation
                warpgroup::mma_async_wait();
            }
            if (warpgroup::laneid() == 0) {
                arrive(compute_done[stage], 1);
            }

        }
        warpgroup::store(o_smem[warpgroupid], o_block);
        // warpgroup::sync(warpgroupid + kittens::WARPGROUP_WARPS); // wait for all warpgroups to finish storing
        
        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            coord<o_tile> o_tile_idx = {batch_id, head_id, seq_id + warpgroupid, 0};
            tma::store_async(globals.O, o_smem[warpgroupid], o_tile_idx);
        }
        tma::store_async_wait();
    }
}



// ============================================================================
// Backward Pass Configuration for HEAD_DIM=128
// ============================================================================

struct bwd_intermediate_noqr_config {
    // Force requirement: BLOCK_SEQ=BLOCK_INTER, NUM_STAGES=BWD_CONSUMER_WARPGROUPS=2
    constexpr static int BLOCK_SEQ = (4*16);
    constexpr static int BLOCK_INTER = (4*16);
    constexpr static int BLOCKS_SM = 1;
    // WE USE TIC-TOC ON Q/R RESROUCES SO NUM_STAGES IS FIXED AT 2
    constexpr static int NUM_STAGES = 2;
    constexpr static int BWD_CONSUMER_WARPGROUPS = 1;
    constexpr static int BWD_PRODUCER_WARPGROUPS = 1;
    constexpr static int BWD_NUM_WARPGROUPS = (BWD_CONSUMER_WARPGROUPS + BWD_PRODUCER_WARPGROUPS);
    constexpr static int BWD_NUM_WORKERS = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS);
};

struct bwd_intermediate_noqr_globals {
    using cfg = bwd_intermediate_noqr_config;
    using q_tile = st_bf<cfg::BLOCK_SEQ, HEAD_DIM>;
    using k_tile = st_bf<cfg::BLOCK_INTER, HEAD_DIM>;
    using u_tile = st_bf<cfg::BLOCK_INTER, HEAD_DIM>;
    using v_tile = st_bf<cfg::BLOCK_INTER, HEAD_DIM>;
    using r_vect = sv_fl<cfg::BLOCK_SEQ>;

    using do_tile = st_bf<cfg::BLOCK_SEQ, HEAD_DIM>; // ds is still bf, meaning upward gradients
    using dq_tile = st_fl<cfg::BLOCK_SEQ, HEAD_DIM>; // dq,dk,du,dv are bf from global memory. reg is fl, need conversion
    using dk_tile = st_fl<cfg::BLOCK_INTER, HEAD_DIM>;
    using du_tile = st_fl<cfg::BLOCK_INTER, HEAD_DIM>;
    using dv_tile = st_fl<cfg::BLOCK_INTER, HEAD_DIM>;
    using middle_tile = st_bf<cfg::BLOCK_SEQ, cfg::BLOCK_INTER>;
    using dr_vect = sv_fl<cfg::BLOCK_SEQ>;

    using q_gl = gl<bf16, -1,-1,-1,-1, q_tile>;
    using k_gl = gl<bf16, -1,-1,-1,-1, k_tile>;
    using u_gl = gl<bf16, -1,-1,-1,-1, u_tile>;
    using v_gl = gl<bf16, -1,-1,-1,-1, v_tile>;
    using r_gl = gl<float, -1,-1,-1,-1, r_vect>;

    using do_gl = gl<bf16, -1,-1,-1,-1, do_tile>;
    using dq_gl = gl<float, -1,-1,-1,-1, dq_tile>;
    using dk_gl = gl<float, -1,-1,-1,-1, dk_tile>;
    using du_gl = gl<float, -1,-1,-1,-1, du_tile>;
    using dv_gl = gl<float, -1,-1,-1,-1, dv_tile>;
    using dr_gl = gl<float, -1,-1,-1,-1, dr_vect>; 

    q_gl Q;
    k_gl K;
    u_gl U;
    v_gl V;
    r_gl R;

    do_gl dO;
    dq_gl dQ;
    dk_gl dK;
    du_gl dU;
    dv_gl dV;
    dr_gl dR;

    const int batch_size;
    const int num_heads;
    const int num_kuv_heads;
    const int seq_len;
    const int num_experts;
    const int expert_dim;
};

#ifdef COMPUTE_DKDUDV
// Backward outline: compute dN, A, dM; accumulate dQ, dK, dU, dV, dR.
// Backward kernel
__global__ __launch_bounds__(bwd_intermediate_noqr_config::BWD_NUM_WORKERS * kittens::WARP_THREADS, 1)
void flashffn_moe_bwd_kernel_head128_intermediate_noqr(const __grid_constant__ bwd_intermediate_noqr_globals globals) {
    // asm volatile (".pragma \"enable_smem_spilling\";");
    using cfg = bwd_intermediate_noqr_config;
    using glbs = bwd_intermediate_noqr_globals;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using q_tile = glbs::q_tile;
    using k_tile = glbs::k_tile;
    using u_tile = glbs::u_tile;
    using v_tile = glbs::v_tile;
    using r_vect = glbs::r_vect;
    using middle_tile = glbs::middle_tile;

    using do_tile = glbs::do_tile;
    using dq_tile = glbs::dq_tile;
    using dk_tile = glbs::dk_tile;
    using du_tile = glbs::du_tile;
    using dv_tile = glbs::dv_tile;
    using dr_vect = glbs::dr_vect;

    // Shared memory layout depends on BLOCK_SEQ=BLOCK_INTER and NUM_STAGES=2
    k_tile  (&k_smem) [cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<k_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    q_tile  (&q_smem) [cfg::NUM_STAGES] = al.allocate<q_tile,  cfg::NUM_STAGES>();
    u_tile  (&u_smem) [cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<u_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    do_tile (&do_smem)[cfg::NUM_STAGES] = al.allocate<do_tile, cfg::NUM_STAGES>();
    v_tile  (&v_smem) [cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<v_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    // dq_tile (&dq_smem) = al.allocate<dq_tile>();
    
    middle_tile (&m_smem)[cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<middle_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    middle_tile (&spill_smem)[cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<middle_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    middle_tile (&spill_smem2)[cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<middle_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    
    using acc_tile = st_bf<cfg::BLOCK_INTER, HEAD_DIM>;

    acc_tile (&dk_offload)[cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<acc_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    acc_tile (&du_offload)[cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<acc_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    acc_tile (&dv_offload)[cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<acc_tile, cfg::BWD_CONSUMER_WARPGROUPS>();

    r_vect  (&r_smem) [cfg::NUM_STAGES] = al.allocate<r_vect, cfg::NUM_STAGES>();
    // dr_vect (&dr_smem) = al.allocate<dr_vect>();

    dk_tile (*dk_smem) = reinterpret_cast<dk_tile*>(&k_smem[0].data[0]);
    du_tile (*du_smem) = reinterpret_cast<du_tile*>(&u_smem[0].data[0]);
    dv_tile (*dv_smem) = reinterpret_cast<dv_tile*>(&v_smem[0].data[0]);
    
    const int batch_id = blockIdx.z;
    const int warp_id = kittens::warpid();
    const int warpgroup_id = warp_id / kittens::WARPGROUP_WARPS;
    const int tiles_per_expert = globals.expert_dim / cfg::BLOCK_INTER;
    const int total_seq_blocks = globals.seq_len / (cfg::BLOCK_SEQ);
    const int inter_tile_idx = (blockIdx.x);
    const int head_idx = (blockIdx.y);
    const int expert_idx = inter_tile_idx / tiles_per_expert;

    __shared__ kittens::semaphore kuv_smem_arrived;
    __shared__ kittens::semaphore q_smem_arrived[cfg::NUM_STAGES], r_smem_arrived[cfg::NUM_STAGES], do_smem_arrived[cfg::NUM_STAGES];
    __shared__ kittens::semaphore compute_done[cfg::NUM_STAGES];
    // __shared__ kittens::semaphore dqdr_smem_ready_to_write;

    // warmup prefetch
    const int warmup_seq_idx = 0;
    if (threadIdx.x == 0) {
        int tic = 0;
        init_semaphore(kuv_smem_arrived, 0, 1);
        for (int stage = 0; stage != cfg::NUM_STAGES; ++stage) {
            init_semaphore(q_smem_arrived[stage], 0, 1);
            init_semaphore(r_smem_arrived[stage], 0, 1);
            init_semaphore(do_smem_arrived[stage], 0, 1);
            init_semaphore(compute_done[stage], 1, 0);
        }
        // init_semaphore(dqdr_smem_ready_to_write, 1, 0);
        tma::expect_bytes(kuv_smem_arrived, sizeof(k_smem)+sizeof(u_smem)+sizeof(v_smem));
        for (int warpgroup = 0; warpgroup != cfg::BWD_CONSUMER_WARPGROUPS; ++warpgroup) {
            int current_inter_tile_idx = (blockIdx.x * cfg::BWD_CONSUMER_WARPGROUPS) + warpgroup;
            int current_expert_idx = current_inter_tile_idx / tiles_per_expert;
            int current_real_inter_tile_idx = current_inter_tile_idx % tiles_per_expert;
            coord<k_tile> kuv_tile_idx = {head_idx, current_expert_idx, current_real_inter_tile_idx, 0};
            tma::load_async(k_smem[warpgroup], globals.K, kuv_tile_idx, kuv_smem_arrived);
            tma::load_async(u_smem[warpgroup], globals.U, kuv_tile_idx, kuv_smem_arrived);
            tma::load_async(v_smem[warpgroup], globals.V, kuv_tile_idx, kuv_smem_arrived);
        }
        coord<q_tile> qdo_tile_idx = {batch_id, head_idx, warmup_seq_idx, 0};
        tma::expect_bytes(q_smem_arrived[tic], sizeof(q_smem[0]));
        tma::load_async(q_smem[tic], globals.Q, qdo_tile_idx, q_smem_arrived[tic]);
        tma::expect_bytes(do_smem_arrived[tic], sizeof(do_smem[0]));
        tma::load_async(do_smem[tic], globals.dO, qdo_tile_idx, do_smem_arrived[tic]);

        coord<r_vect> router_idx = {batch_id, head_idx, expert_idx, warmup_seq_idx};
        tma::expect_bytes(r_smem_arrived[tic], sizeof(r_smem[0]));
        tma::load_async(r_smem[tic], globals.R, router_idx, r_smem_arrived[tic]);
    }
    __syncthreads();
    // producers
    if (warpgroup_id == cfg::BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<32>();
        // load q/do/r
        if (warp_id % kittens::WARPGROUP_WARPS == 0) {
            int tic = 0, toc = 1;
            for (int seq_idx = 0; seq_idx != total_seq_blocks; ++seq_idx, tic ^= 1, toc ^= 1) {
                if (seq_idx + 1 < total_seq_blocks) {
                    coord<q_tile> qdo_tile_idx = {batch_id, head_idx, seq_idx + 1, 0};
                    coord<r_vect> router_idx = {batch_id, head_idx, expert_idx, seq_idx + 1};
                    tma::expect_bytes(q_smem_arrived[toc], sizeof(q_smem[0]));
                    tma::load_async(q_smem[toc], globals.Q, qdo_tile_idx, q_smem_arrived[toc]);
                    tma::expect_bytes(do_smem_arrived[toc], sizeof(do_smem[0]));
                    tma::load_async(do_smem[toc], globals.dO, qdo_tile_idx, do_smem_arrived[toc]);
                    tma::expect_bytes(r_smem_arrived[toc], sizeof(r_smem[0]));
                    tma::load_async(r_smem[toc], globals.R, router_idx, r_smem_arrived[toc]);
                }
                wait(compute_done[tic], (seq_idx / cfg::NUM_STAGES) & 1);
            }
        } 
    // consumers
    } else {
        warpgroup::increase_registers<256>();
        rt_bf<cfg::BLOCK_INTER/kittens::WARPGROUP_WARPS, HEAD_DIM> common_acc_reg;
        rt_fl<cfg::BLOCK_INTER/kittens::WARPGROUP_WARPS, HEAD_DIM> tmp_acc_reg;
        rt_bf<cfg::BLOCK_INTER/kittens::WARPGROUP_WARPS, HEAD_DIM> tmp_bf_reg;
        
        rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER> R1,R2,R3,R4;
        rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER> tmp_mma_fl;

        // rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER, kittens::ducks::rt_layout::row> R_bf16;
        rv_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> router_reg;
        rv_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> router_reg_bf;

        // ==================== 概览：本分支（消费者）负责核心反向计算 ====================
        // 寄存器累加器（FP32）：
        //   - common_acc_reg：通过 smem offload 复用，分别累计 dK / dU / dV
        //   - dq_reg：累计本 warpgroup 的 dQ（随后与另一组合并）
        // 中间方块（BF16，节省寄存器）：
        //   - R1：M 或 silu(M)
        //   - R2：silu'(M)（通过 σ 与 silu 组合得到）
        //   - R3：在不同阶段作为 1-silu(M) / dN / N / expertN 等中间量
        //   - R4：dA = dO @ V^T
        // 共享内存中转（BF16）：
        //   - R_bf16：已移除，直接使用 BF16 寄存器写入

        zero(common_acc_reg); 
        warpgroup::store(dk_offload[warpgroup_id], common_acc_reg);
        warpgroup::store(du_offload[warpgroup_id], common_acc_reg);
        warpgroup::store(dv_offload[warpgroup_id], common_acc_reg);

        int tic = 0, toc = 1;
        wait(kuv_smem_arrived, 0);
        for (int seq_idx = 0; seq_idx != total_seq_blocks; ++seq_idx, tic^=1, toc ^= 1) {
            const int phase = (seq_idx / cfg::NUM_STAGES) & 1;
            // ==================== 第 1 部分：重算 M 与 dA，两条主分支并行 ====================
            // 输入：Q/K/V/dO（本轮 seq 块）
            // 输出：
            //   - R1 = M = Q @ K^T（随后转为 silu(M)）
            //   - R4 = dA = dO @ V^T
            one(R3);
            // zero(dr_reg);
            wait(q_smem_arrived[tic], phase);
            warpgroup::mm_ABt(tmp_mma_fl, q_smem[tic], k_smem[warpgroup_id]);
            warpgroup::mma_async_wait();
            copy(R1, tmp_mma_fl);

            wait(do_smem_arrived[tic], phase);
            warpgroup::mm_ABt(tmp_mma_fl, do_smem[tic], v_smem[warpgroup_id]);
            warpgroup::mma_commit_group();
            // ==================== 第 2 部分：激活与导数 ====================
            // R2 = σ(M)；R1 = silu(M) = M*σ(M)；R2 最终变为 silu'(M)
            // R3 暂存 1 - silu(M)，用于构造 silu'(M)
            fast_sigmoid_bf16(R2, R1);
            mul(R1,R1,R2);
            sub(R3,R3,R1);
            mul(R2,R2,R3);
            add(R2,R1,R2);
            warpgroup::store(spill_smem[warpgroup_id], R2);
            // ==================== 第 3 部分：构造 dN 并暂存 ====================
            // dN = router ⊙ (dA ⊙ silu(M)) ；写入 m_smem 作为后续 AB/AtB 的输入
            warpgroup::mma_async_wait();
            copy(R4, tmp_mma_fl);
            mul(R3,R4,R1);
            warpgroup::store(spill_smem2[warpgroup_id], R4);
            wait(r_smem_arrived[tic],phase);
            warpgroup::load(router_reg, r_smem[tic]);
            copy(router_reg_bf, router_reg);
            mul_row(R3,R3,router_reg_bf);
            // copy(R_bf16, R3);
            warpgroup::store(m_smem[warpgroup_id], R3);
            // ==================== 第 4 部分：重算 N，累计 dR，启动 dQ 与 dU ====================
            // N = Q @ U^T； dR 累加：sum_row(dA ⊙ silu(M) ⊙ N)
            // dQ 第一项：dQ += dN @ U； dU：dU += dN^T @ Q
            warpgroup::mm_ABt(tmp_mma_fl, q_smem[tic], u_smem[warpgroup_id]);
            warpgroup::mma_async_wait();
            copy(R3, tmp_mma_fl);
            // accumulate_dr_rowsum_triple(dr_reg, R4, R1, R3);
            // warpgroup::mm_AB(dq_reg, R_bf16, u_smem[warpgroup_id]);
            // warpgroup::mma_commit_group();
            zero(tmp_acc_reg);
            warpgroup::mma_AtB(tmp_acc_reg, m_smem[warpgroup_id], q_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            copy(tmp_bf_reg, tmp_acc_reg);
            warpgroup::load(common_acc_reg, du_offload[warpgroup_id]);
            add(common_acc_reg, common_acc_reg, tmp_bf_reg);
            warpgroup::store(du_offload[warpgroup_id], common_acc_reg);
            // ==================== 第 5 部分：构造 A，累计 dV ====================
            // expertN = N ⊙ router； A = silu(M) ⊙ expertN； dV += A^T @ dO
            mul_row(R3,R3,router_reg_bf);
            mul(R1,R1,R3);
            warpgroup::mma_async_wait();
            // copy(R_bf16, R1);
            warpgroup::store(m_smem[warpgroup_id], R1);
            zero(tmp_acc_reg);
            warpgroup::mma_AtB(tmp_acc_reg, m_smem[warpgroup_id], do_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            copy(tmp_bf_reg, tmp_acc_reg);
            warpgroup::load(common_acc_reg, dv_offload[warpgroup_id]);
            add(common_acc_reg, common_acc_reg, tmp_bf_reg);
            warpgroup::store(dv_offload[warpgroup_id], common_acc_reg);
            // ==================== 第 6 部分：构造 dM，补齐 dQ，累计 dK ====================
            // dM = expertN ⊙ dA ⊙ silu'(M)； dQ += dM @ K； dK += dM^T @ Q
            warpgroup::load(R4, spill_smem2[warpgroup_id]);
            mul(R4,R3,R4);
            warpgroup::load(R2, spill_smem[warpgroup_id]);
            mul(R2,R4,R2);
            // copy(R_bf16, R2);
            warpgroup::mma_async_wait();
            warpgroup::store(m_smem[warpgroup_id], R2);
            // warpgroup::mma_AB(dq_reg, R_bf16, k_smem[warpgroup_id]);
            // warpgroup::mma_commit_group();
            zero(tmp_acc_reg);
            warpgroup::mma_AtB(tmp_acc_reg, m_smem[warpgroup_id], q_smem[tic]);
            warpgroup::mma_commit_group();
            warpgroup::mma_async_wait();
            copy(tmp_bf_reg, tmp_acc_reg);
            warpgroup::load(common_acc_reg, dk_offload[warpgroup_id]);
            add(common_acc_reg, common_acc_reg, tmp_bf_reg);
            warpgroup::store(dk_offload[warpgroup_id], common_acc_reg);
            warpgroup::mma_async_wait();
            // ==================== 第 7 部分：与另一 warpgroup 合并 dQ/dR，并写回 ====================
            if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
        }
        // ==================== 第 8 部分：写回 dK/dU/dV 到全局内存（原位累加） ====================
        int current_inter_tile_idx = (blockIdx.x);
        int current_expert_idx = current_inter_tile_idx / tiles_per_expert;
        int current_real_inter_tile_idx = current_inter_tile_idx % tiles_per_expert;
        
        warpgroup::load(common_acc_reg, dk_offload[warpgroup_id]);
        copy(tmp_acc_reg, common_acc_reg);
        warpgroup::store(dk_smem[warpgroup_id], tmp_acc_reg);
        coord<dk_tile> dkuv_tile_idx = {head_idx, current_expert_idx, current_real_inter_tile_idx, 0};
        group<4>::sync(warpgroup::groupid()+4);
        if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
            tma::store_add_async(globals.dK, dk_smem[warpgroup_id], dkuv_tile_idx);
            tma::store_commit_group();
        }

        warpgroup::load(common_acc_reg, du_offload[warpgroup_id]);
        copy(tmp_acc_reg, common_acc_reg);
        warpgroup::store(du_smem[warpgroup_id], tmp_acc_reg);
        group<4>::sync(warpgroup::groupid()+4);
        if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
            tma::store_add_async(globals.dU, du_smem[warpgroup_id], dkuv_tile_idx);
            tma::store_commit_group();
        }
        // wait(dqdr_smem_ready_to_write, toc);

        warpgroup::load(common_acc_reg, dv_offload[warpgroup_id]);
        copy(tmp_acc_reg, common_acc_reg);
        warpgroup::store(dv_smem[warpgroup_id], tmp_acc_reg);
        group<4>::sync(warpgroup::groupid()+4);
        if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
            tma::store_add_async(globals.dV, dv_smem[warpgroup_id], dkuv_tile_idx);
            tma::store_commit_group();
        }
        tma::store_async_wait();
    }
}
#endif

// ============================================================================
// Backward Pass (dQ/dR) — QR prefetch, loop over K/U/V intermediate dimension
// ============================================================================
struct bwd_qrfirst_dqdr_config {
    // Force requirement: BLOCK_SEQ=BLOCK_INTER, NUM_STAGES fixed at 2 for tic-toc
    constexpr static int BLOCK_SEQ = (4*16);
    constexpr static int BLOCK_INTER = (4*16);
    constexpr static int BLOCKS_SM = 1;
    constexpr static int NUM_STAGES = 2;
    // 1 consumer warpgroup + 1 producer warpgroup
    constexpr static int BWD_CONSUMER_WARPGROUPS = 1;
    constexpr static int BWD_PRODUCER_WARPGROUPS = 1;
    constexpr static int BWD_NUM_WARPGROUPS = (BWD_CONSUMER_WARPGROUPS + BWD_PRODUCER_WARPGROUPS);
    constexpr static int BWD_NUM_WORKERS = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS);
};

struct bwd_qrfirst_dqdr_globals {
    using cfg = bwd_qrfirst_dqdr_config;
    using q_tile  = st_bf<cfg::BLOCK_SEQ,  HEAD_DIM>;
    using k_tile  = st_bf<cfg::BLOCK_INTER,HEAD_DIM>;
    using u_tile  = st_bf<cfg::BLOCK_INTER,HEAD_DIM>;
    using v_tile  = st_bf<cfg::BLOCK_INTER,HEAD_DIM>;
    using r_vect  = sv_fl<cfg::BLOCK_SEQ>;

    using do_tile = st_bf<cfg::BLOCK_SEQ,  HEAD_DIM>;
    using dq_tile = st_fl<cfg::BLOCK_SEQ,  HEAD_DIM>;
    using dr_vect = sv_fl<cfg::BLOCK_SEQ>;
    using middle_tile = st_bf<cfg::BLOCK_SEQ, cfg::BLOCK_INTER>;

    using q_gl  = gl<bf16, -1,-1,-1,-1, q_tile>;
    using k_gl  = gl<bf16, -1,-1,-1,-1, k_tile>;
    using u_gl  = gl<bf16, -1,-1,-1,-1, u_tile>;
    using v_gl  = gl<bf16, -1,-1,-1,-1, v_tile>;
    using r_gl  = gl<float,-1,-1,-1,-1, r_vect>;

    using do_gl = gl<bf16, -1,-1,-1,-1, do_tile>;
    using dq_gl = gl<float,-1,-1,-1,-1, dq_tile>;
    using dr_gl = gl<float,-1,-1,-1,-1, dr_vect>;

    q_gl Q;
    k_gl K;
    u_gl U;
    v_gl V;
    r_gl R;

    do_gl dO;
    dq_gl dQ;
    dr_gl dR;

    const int batch_size;
    const int num_heads;
    const int num_kuv_heads;
    const int seq_len;
    const int num_experts;
    const int expert_dim;
};
#ifdef COPMUTE_DQDR
__global__ __launch_bounds__(bwd_qrfirst_dqdr_config::BWD_NUM_WORKERS * kittens::WARP_THREADS, bwd_qrfirst_dqdr_config::BLOCKS_SM)
void flashffn_moe_bwd_kernel_head128_qr_prefetch_interloop(const __grid_constant__ bwd_qrfirst_dqdr_globals globals) {
    using cfg = bwd_qrfirst_dqdr_config;
    using glbs = bwd_qrfirst_dqdr_globals;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    using q_tile  = glbs::q_tile;
    using k_tile  = glbs::k_tile;
    using u_tile  = glbs::u_tile;
    using v_tile  = glbs::v_tile;
    using r_vect  = glbs::r_vect;
    using do_tile = glbs::do_tile;
    using dq_tile = glbs::dq_tile;
    using dr_vect = glbs::dr_vect;
    using middle_tile = glbs::middle_tile;

    // Shared memory
    k_tile  (&k_smem)[cfg::NUM_STAGES] = al.allocate<k_tile,  cfg::NUM_STAGES>();
    u_tile  (&u_smem)[cfg::NUM_STAGES] = al.allocate<u_tile,  cfg::NUM_STAGES>();
    v_tile  (&v_smem)[cfg::NUM_STAGES] = al.allocate<v_tile,  cfg::NUM_STAGES>();
    q_tile  (&q_smem) = al.allocate<q_tile>();
    do_tile (&do_smem)= al.allocate<do_tile>();
    r_vect  (&r_smem) = al.allocate<r_vect>();
    middle_tile (&m_smem) = al.allocate<middle_tile>();
    // reuse smem
    dq_tile (&dq_smem) = al.allocate<dq_tile>();
    dr_vect (&dr_smem) = al.allocate<dr_vect>();

    const int batch_id = blockIdx.z;
    const int head_idx = blockIdx.y;
    const int seq_block = blockIdx.x;
    const int warp_id = kittens::warpid();
    const int warpgroup_id = warp_id / kittens::WARPGROUP_WARPS;
    const int tiles_per_expert = globals.expert_dim / cfg::BLOCK_INTER;
    const int total_inter_blocks = tiles_per_expert * globals.num_experts;

    // Semaphores
    __shared__ kittens::semaphore k_smem_arrived[cfg::NUM_STAGES], u_smem_arrived[cfg::NUM_STAGES], v_smem_arrived[cfg::NUM_STAGES];
    __shared__ kittens::semaphore q_smem_arrived, do_smem_arrived;
    __shared__ kittens::semaphore r_smem_arrived, r_used;
    __shared__ kittens::semaphore compute_done[cfg::NUM_STAGES];

    // Warmup prefetch for this (batch, head, seq_block)
    if (threadIdx.x == 0) {
        for (int s = 0; s < cfg::NUM_STAGES; ++s) {
            init_semaphore(k_smem_arrived[s], 0, 1);
            init_semaphore(u_smem_arrived[s], 0, 1);
            init_semaphore(v_smem_arrived[s], 0, 1);
            init_semaphore(compute_done[s], 1, 0);
        }
        init_semaphore(q_smem_arrived, 0, 1);
        init_semaphore(r_smem_arrived, 0, 1);
        init_semaphore(do_smem_arrived, 0, 1);
        init_semaphore(r_used, cfg::BWD_CONSUMER_WARPGROUPS, 0);

        // Stage warmup for K/U/V at inter_tile=0 (expert=0, tile=0)
        coord<k_tile> kuv_tile_idx = {head_idx, 0, 0, 0};
        tma::expect_bytes(k_smem_arrived[0], sizeof(k_tile));
        tma::load_async(k_smem[0], globals.K, kuv_tile_idx, k_smem_arrived[0]);
        tma::expect_bytes(u_smem_arrived[0], sizeof(u_tile));
        tma::load_async(u_smem[0], globals.U, kuv_tile_idx, u_smem_arrived[0]);
        tma::expect_bytes(v_smem_arrived[0], sizeof(v_tile));
        tma::load_async(v_smem[0], globals.V, kuv_tile_idx, v_smem_arrived[0]);

        // Warmup Q/dO/R for current seq_block, expert=0
        coord<q_tile> q_tile_idx = {batch_id, head_idx, seq_block, 0};
        tma::expect_bytes(q_smem_arrived, sizeof(q_tile));
        tma::load_async(q_smem, globals.Q, q_tile_idx, q_smem_arrived);
        tma::expect_bytes(do_smem_arrived, sizeof(do_tile));
        tma::load_async(do_smem, globals.dO, q_tile_idx, do_smem_arrived);

        coord<r_vect> r_vec_idx  = {batch_id, head_idx, 0, seq_block};
        tma::expect_bytes(r_smem_arrived, sizeof(r_vect));
        tma::load_async(r_smem, globals.R, r_vec_idx, r_smem_arrived);
    }
    __syncthreads();

    // Producer warpgroup: stream K/U/V across intermediate, stream R across experts
    if (warpgroup_id == cfg::BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();
        if (warp_id % kittens::WARPGROUP_WARPS == 0) {
            // K/V streamer
            for (int inter_tile = 1; inter_tile < total_inter_blocks; ++inter_tile) {
                const int expert_id = inter_tile / tiles_per_expert;
                const int real_inter_tile = inter_tile % tiles_per_expert;
                const int stage = inter_tile % cfg::NUM_STAGES;
                wait(compute_done[stage], ((inter_tile / cfg::NUM_STAGES) - 1) & 1);
                coord<k_tile> kuv_tile_idx = {head_idx, expert_id, real_inter_tile, 0};
                tma::expect_bytes(k_smem_arrived[stage], sizeof(k_tile));
                tma::load_async(k_smem[stage], globals.K, kuv_tile_idx, k_smem_arrived[stage]);
                tma::expect_bytes(v_smem_arrived[stage], sizeof(v_tile));
                tma::load_async(v_smem[stage], globals.V, kuv_tile_idx, v_smem_arrived[stage]);
            }
        } else if (warp_id % kittens::WARPGROUP_WARPS == 1) {
            // U streamer
            for (int inter_tile = 1; inter_tile < total_inter_blocks; ++inter_tile) {
                const int expert_id = inter_tile / tiles_per_expert;
                const int real_inter_tile = inter_tile % tiles_per_expert;
                const int stage = inter_tile % cfg::NUM_STAGES;
                wait(compute_done[stage], ((inter_tile / cfg::NUM_STAGES) - 1) & 1);
                coord<u_tile> u_idx = {head_idx, expert_id, real_inter_tile, 0};
                tma::expect_bytes(u_smem_arrived[stage], sizeof(u_tile));
                tma::load_async(u_smem[stage], globals.U, u_idx, u_smem_arrived[stage]);
            }
        } else if (warp_id % kittens::WARPGROUP_WARPS == 2) {
            // R streamer: double-buffer across experts
            for (int expert_id = 1; expert_id != globals.num_experts; ++expert_id) {
                wait(r_used, (expert_id - 1) & 1);
                coord<r_vect> r_vec_idx  = {batch_id, head_idx, expert_id, seq_block};
                tma::expect_bytes(r_smem_arrived, sizeof(r_vect));
                tma::load_async(r_smem, globals.R, r_vec_idx, r_smem_arrived);
            }
        }
    // Consumer warpgroup: computes dQ/dR for this seq block, loops inter-dimension
    } else {
        warpgroup::increase_registers<248>();
        // Middle tiles and registers
        rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER> one_minus_silu;
        rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER> R1, R2, R3, R4;
        rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER, kittens::ducks::rt_layout::row> R_bf16;
        rv_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> router_reg;
        rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, HEAD_DIM> dq_reg;
        rv_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> dr_reg;

        // Ensure Q/dO ready once
        wait(q_smem_arrived, 0);
        wait(do_smem_arrived, 0);

        zero(dq_reg);
        zero(dr_reg);

        for (int expert_id = 0; expert_id < globals.num_experts; ++expert_id) {
            // Router for this expert
            wait(r_smem_arrived, expert_id & 1);
            warpgroup::load(router_reg, r_smem);
            group<4>::sync(warpgroup::groupid()+4); // necessary?
            if (warpgroup::laneid() == 0) arrive(r_used);

            for (int inter = 0; inter < tiles_per_expert; ++inter) {
                const int inter_tile = expert_id * tiles_per_expert + inter;
                const int stage = inter_tile % cfg::NUM_STAGES;
                const int phase = (inter_tile / cfg::NUM_STAGES) & 1;
                one(one_minus_silu);
                wait(k_smem_arrived[stage], phase);
                wait(u_smem_arrived[stage], phase);
                wait(v_smem_arrived[stage], phase);

                // R1 = M = Q @ K^T
                warpgroup::mm_ABt(R1, q_smem, k_smem[stage]);
                warpgroup::mma_async_wait();
                // R4 = dA = dO @ V^T
                warpgroup::mm_ABt(R4, do_smem, v_smem[stage]);
                warpgroup::mma_commit_group();
                // sigmoid/silu
                fast_sigmoid(R2, R1);
                mul(R1, R1, R2);     // R1 = silu(M)
                // N
                warpgroup::mma_async_wait();
                warpgroup::mm_ABt(R3, q_smem, u_smem[stage]);
                warpgroup::mma_async_wait();

                // dR rowsum
                accumulate_dr_rowsum_triple(dr_reg, R4, R1, R3);

                // dN = dA * silu(M) * Router
                mul(R3, R4, R1);
                mul_row(R3, R3, router_reg);
                copy(R_bf16, R3);
                warpgroup::store(m_smem, R_bf16);
                warpgroup::mma_AB(dq_reg, m_smem, u_smem[stage]); // dQ += dN @ U
                warpgroup::mma_commit_group();

                // silu'(M)
                sub(one_minus_silu, one_minus_silu, R1);
                mul(R2, R2, one_minus_silu);
                add(R2, R1, R2);

                // expertN = Router * N
                // R3 has been clobbered to dN above; recompute N = Q @ U^T before forming expertN
                warpgroup::mm_ABt(R3, q_smem, u_smem[stage]);
                warpgroup::mma_async_wait();
                mul_row(R3, R3, router_reg);
                // dM
                mul(R4, R4, R3);   // R4 = dA * expertN
                mul(R2, R4, R2);   // R2 = dM
                copy(R_bf16, R2);
                warpgroup::mma_async_wait();
                warpgroup::store(m_smem, R_bf16);
                warpgroup::mma_AB(dq_reg, m_smem, k_smem[stage]); // dQ += dM @ K
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait();

                if (warpgroup::laneid() == 0) {
                    arrive(compute_done[stage]);
                }
            } // inter tiles
            // zero(dr_reg);
            warpgroup::store(dr_smem, dr_reg);
            group<4>::sync(warpgroup::groupid()+4);
            // write dR for this expert/seq block (no atomics)
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                coord<dr_vect> dr_idx = {batch_id, head_idx, expert_id, seq_block};
                tma::store_async(globals.dR, dr_smem, dr_idx);
                tma::store_commit_group();
            }
            // if ((expert_id & 7) == 7) tma::store_async_wait();
            zero(dr_reg);
        } // experts

        // warpgroup::mma_async_wait();
        // write dQ for this seq block (no atomics)
        warpgroup::store(dq_smem, dq_reg);
        group<4>::sync(warpgroup::groupid()+4);
        if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
            coord<dq_tile> dq_idx = {batch_id, head_idx, seq_block, 0};
            tma::store_async(globals.dQ, dq_smem, dq_idx);
            tma::store_commit_group();
        }
        tma::store_async_wait();
    }
}
#endif

#define TORCH_COMPILE

#ifdef TORCH_COMPILE

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>
#include <cstdio>

// ============================================================================
// Forward Python Binding
// ============================================================================
void flashffn_moe_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor U, torch::Tensor V, torch::Tensor R, torch::Tensor O) {
    /*
        We require the dtype to be bf16, and all tensors need to be contiguous.
        We require R to be of shape (batchsize, num_heads, num_experts, seq_len)
    */
    TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "Q must be bf16");
    TORCH_CHECK(K.scalar_type() == at::kBFloat16 && U.scalar_type() == at::kBFloat16 && V.scalar_type() == at::kBFloat16, "K/U/V must be bf16");
    TORCH_CHECK(R.scalar_type() == at::kFloat, "R must be float32");

    auto batch_size = Q.size(0);
    auto num_heads = Q.size(1);
    auto seq_len = Q.size(2);
    auto head_dim = Q.size(3);

    auto kuv_heads = K.size(0);
    auto num_experts = K.size(1);
    auto expert_dim = K.size(2);

    assert(head_dim == HEAD_DIM);
    assert(num_heads == kuv_heads); // This version we only support num_heads == kuv_heads
    assert(expert_dim % fwd_config::BLOCK_INTER == 0); // For the loop to work, we must have expert_dim % BLOCK_INTER == 0
    // assert(seq_len % (fwd_config::BLOCK_SEQ * fwd_config::CONSUMER_WARPGROUPS) == 0); // For the loop to work, we must have seq_len % BLOCK_SEQ == 0. TODO: remove this and fix the kernel
    static_assert(fwd_config::BLOCK_INTER % 16 == 0, "BLOCK_INTER must be divisible by 16");
    static_assert(fwd_config::BLOCK_SEQ % 16 == 0, "BLOCK_SEQ must be divisible by 16");
    static_assert(fwd_config::PRODUCER_WARPGROUPS == 1, "Currently only support producer=1");

    using q_gl = fwd_globals::q_gl;
    using k_gl = fwd_globals::k_gl; 
    using u_gl = fwd_globals::u_gl;
    using v_gl = fwd_globals::v_gl;
    using o_gl = fwd_globals::o_gl;
    using r_gl = fwd_globals::r_gl;

    // auto Rt = R.permute({0,1,3,2}).contiguous(); // (batchsize, num_heads, seq_len, num_experts) -> (batchsize, num_heads, num_experts, seq_len)
    
    q_gl qg{reinterpret_cast<bf16*>(Q.data_ptr<c10::BFloat16>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    k_gl kg{reinterpret_cast<bf16*>(K.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads, (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    u_gl ug{reinterpret_cast<bf16*>(U.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads, (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    v_gl vg{reinterpret_cast<bf16*>(V.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads, (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    o_gl og{reinterpret_cast<bf16*>(O.data_ptr<c10::BFloat16>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    r_gl rg{reinterpret_cast<float*>(R.data_ptr<float>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)num_experts, (unsigned)seq_len};

    fwd_globals g{
        qg, kg, ug, vg, og, rg,
        
        (int)batch_size,
        (int)num_heads,
        (int)kuv_heads,
        (int)seq_len,
        (int)num_experts,
        (int)expert_dim,
    };

    auto stream   = at::cuda::getCurrentCUDAStream().stream();
    auto threads  = fwd_config::NUM_WORKERS * kittens::WARP_THREADS;
    auto smem_size = 200000;
    auto tile = fwd_config::BLOCK_SEQ * fwd_config::CONSUMER_WARPGROUPS;
    dim3 grid(
        // don't forget to do seq_masking inside kernel
        (unsigned)((seq_len + tile - 1) / tile),
        (unsigned)num_heads,
        (unsigned)batch_size
    );
    
    // printf("[C++] Launching kernel.\n");
    cudaFuncSetAttribute(flashffn_moe_fwd_kernel_head128, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    flashffn_moe_fwd_kernel_head128<<<grid, threads, smem_size, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    // cudaStreamSynchronize(stream);
}

// ============================================================================
// Backward Python Binding
// ============================================================================
void flashffn_moe_backward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor U, torch::Tensor V, torch::Tensor R,
    torch::Tensor dQ, torch::Tensor dK, torch::Tensor dU, torch::Tensor dV, torch::Tensor dR,
    torch::Tensor dO
) {
    TORCH_CHECK(Q.scalar_type() == at::kBFloat16, "Q must be bf16");
    TORCH_CHECK(K.scalar_type() == at::kBFloat16 && U.scalar_type() == at::kBFloat16 && V.scalar_type() == at::kBFloat16, "K/U/V must be bf16");
    TORCH_CHECK(R.scalar_type() == at::kFloat, "R must be float32");
    TORCH_CHECK(dO.scalar_type() == at::kBFloat16, "dO (grad_output) must be bf16");
    TORCH_CHECK(dQ.scalar_type() == at::kFloat && dK.scalar_type() == at::kFloat && dU.scalar_type() == at::kFloat && dV.scalar_type() == at::kFloat, "dQ/dK/dU/dV must be fp32");
    TORCH_CHECK(dR.scalar_type() == at::kFloat, "dR must be float32");

    using bwd_config = bwd_intermediate_noqr_config;

    auto batch_size = Q.size(0);
    auto num_heads  = Q.size(1);
    auto seq_len    = Q.size(2);
    auto head_dim   = Q.size(3);
    auto kuv_heads   = K.size(0);
    auto num_experts = K.size(1);
    auto expert_dim  = K.size(2);
    auto intermediate_size = num_experts * expert_dim;

    TORCH_CHECK(head_dim == HEAD_DIM, "Only HEAD_DIM=128 is supported");
    TORCH_CHECK(kuv_heads == num_heads, "num_heads must equal kuv_heads (current impl)");
    TORCH_CHECK(expert_dim % (2*bwd_config::BLOCK_INTER) == 0, "expert_dim % (2*BLOCK_INTER) == 0 required");
    TORCH_CHECK(seq_len % (bwd_config::BLOCK_SEQ) == 0, "seq_len % (BLOCK_SEQ) == 0 required");

#ifdef COMPUTE_DKDUDV
    using q_gl = bwd_intermediate_noqr_globals::q_gl;
    using k_gl = bwd_intermediate_noqr_globals::k_gl;
    using u_gl = bwd_intermediate_noqr_globals::u_gl;
    using v_gl = bwd_intermediate_noqr_globals::v_gl;
    using r_gl = bwd_intermediate_noqr_globals::r_gl;

    using do_gl = bwd_intermediate_noqr_globals::do_gl;
    using dq_gl = bwd_intermediate_noqr_globals::dq_gl;
    using dk_gl = bwd_intermediate_noqr_globals::dk_gl;
    using du_gl = bwd_intermediate_noqr_globals::du_gl;
    using dv_gl = bwd_intermediate_noqr_globals::dv_gl;
    using dr_gl = bwd_intermediate_noqr_globals::dr_gl;

    q_gl qg{reinterpret_cast<bf16*>(Q.data_ptr<c10::BFloat16>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    k_gl kg{reinterpret_cast<bf16*>(K.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    u_gl ug{reinterpret_cast<bf16*>(U.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    v_gl vg{reinterpret_cast<bf16*>(V.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    r_gl rg{reinterpret_cast<float*>(R.data_ptr<float>()),       (unsigned)batch_size, (unsigned)num_heads,    (unsigned)num_experts, (unsigned)seq_len};

    do_gl dsg{reinterpret_cast<bf16*>(dO.data_ptr<c10::BFloat16>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    dq_gl dqg{reinterpret_cast<float*>(dQ.data_ptr<float>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    dk_gl dkg{reinterpret_cast<float*>(dK.data_ptr<float>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    du_gl dug{reinterpret_cast<float*>(dU.data_ptr<float>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    dv_gl dvg{reinterpret_cast<float*>(dV.data_ptr<float>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    dr_gl drg{reinterpret_cast<float*>(dR.data_ptr<float>()),  (unsigned)batch_size, (unsigned)num_heads, (unsigned)num_experts, (unsigned)seq_len};

    bwd_intermediate_noqr_globals g{
        qg, kg, ug, vg, rg,
        dsg, dqg, dkg, dug, dvg, drg,
        (int)batch_size,
        (int)num_heads,
        (int)kuv_heads,
        (int)seq_len,
        (int)num_experts,
        (int)expert_dim,
    };
    
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto threads = bwd_config::BWD_NUM_WORKERS * kittens::WARP_THREADS;
    auto smem_size = kittens::MAX_SHARED_MEMORY;
    // auto smem_size = 194000; // from h100.cu
    // size_t smem;
    // cudaFuncGetAttributes(&attr, flashffn_moe_bwd_kernel_head128_intermediate_noqr);
    // printf("sharedMem = %zu\n", attr.maxDynamicSharedSizeBytes);
    auto tile = bwd_config::BLOCK_INTER;
    dim3 grid(
        (unsigned)((intermediate_size + tile - 1) / tile),
        (unsigned)num_heads,
        (unsigned)batch_size
    );
    cudaFuncSetAttribute(flashffn_moe_bwd_kernel_head128_intermediate_noqr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(flashffn_moe_bwd_kernel_head128_intermediate_noqr, cudaFuncAttributePreferredSharedMemoryCarveout, 95); // from h100.cu
    flashffn_moe_bwd_kernel_head128_intermediate_noqr<<<grid, threads, smem_size, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
#endif
#ifdef COPMUTE_DQDR
    // -------- Run seq-loop dq/dr kernel after dK/dU/dV --------
    using dqdr_cfg = bwd_qrfirst_dqdr_config;
    using dqdr_gl  = bwd_qrfirst_dqdr_globals;
    dqdr_gl::q_gl  qg2{reinterpret_cast<bf16*>(Q.data_ptr<c10::BFloat16>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    dqdr_gl::k_gl  kg2{reinterpret_cast<bf16*>(K.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    dqdr_gl::u_gl  ug2{reinterpret_cast<bf16*>(U.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    dqdr_gl::v_gl  vg2{reinterpret_cast<bf16*>(V.data_ptr<c10::BFloat16>()), (unsigned)kuv_heads,  (unsigned)num_experts, (unsigned)expert_dim, (unsigned)head_dim};
    dqdr_gl::r_gl  rg2{reinterpret_cast<float*>(R.data_ptr<float>()),       (unsigned)batch_size, (unsigned)num_heads, (unsigned)num_experts, (unsigned)seq_len};
    dqdr_gl::do_gl dsg2{reinterpret_cast<bf16*>(dO.data_ptr<c10::BFloat16>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    dqdr_gl::dq_gl dqg2{reinterpret_cast<float*>(dQ.data_ptr<float>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)seq_len, (unsigned)head_dim};
    dqdr_gl::dr_gl drg2{reinterpret_cast<float*>(dR.data_ptr<float>()), (unsigned)batch_size, (unsigned)num_heads, (unsigned)num_experts, (unsigned)seq_len};

    bwd_qrfirst_dqdr_globals g2{
        qg2, kg2, ug2, vg2, rg2,
        dsg2, dqg2, drg2,
        (int)batch_size,
        (int)num_heads,
        (int)kuv_heads,
        (int)seq_len,
        (int)num_experts,
        (int)expert_dim,
    };
    auto threads2 = dqdr_cfg::BWD_NUM_WORKERS * kittens::WARP_THREADS;
    auto smem_size2 = 184000;
    auto total_seq_blocks2 = (unsigned)(seq_len / dqdr_cfg::BLOCK_SEQ);
    dim3 grid2(
        (unsigned)total_seq_blocks2,
        (unsigned)num_heads,
        (unsigned)batch_size
    );
    cudaFuncSetAttribute(flashffn_moe_bwd_kernel_head128_qr_prefetch_interloop, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size2);
    flashffn_moe_bwd_kernel_head128_qr_prefetch_interloop<<<grid2, threads2, smem_size2, stream>>>(g2);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
#endif
}

PYBIND11_MODULE(flash_ffn_moe, m) {
m.def("flashffn_moe_forward", &flashffn_moe_forward, "FlashFFN-MoE forward (H100/TK)");
m.def("flashffn_moe_backward", &flashffn_moe_backward, "FlashFFN-MoE backward. Loop over intermediate and do atomic add on dQ, dR(H100/TK)");
}

#endif