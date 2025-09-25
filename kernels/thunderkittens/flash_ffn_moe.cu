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

#include "kittens.cuh"
#include "utils.cuh"
#include <cooperative_groups.h>

using namespace kittens;
namespace cg = cooperative_groups;
#define USE_FAST_SIGMOID
#define USE_DIRECT_SILU
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
        warpgroup::sync(warpgroupid + kittens::WARPGROUP_WARPS); // wait for all warpgroups to finish storing
        
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

struct bwd_intermediate_atomicqr_config {
    // Force requirement: BLOCK_SEQ=BLOCK_INTER, NUM_STAGES=BWD_CONSUMER_WARPGROUPS=2
    constexpr static int BLOCK_SEQ = (4*16);
    constexpr static int BLOCK_INTER = (4*16);
    constexpr static int BLOCKS_SM = 1;
    // WE USE TIC-TOC ON Q/R RESROUCES SO NUM_STAGES IS FIXED AT 2
    constexpr static int NUM_STAGES = 2;
    constexpr static int BWD_CONSUMER_WARPGROUPS = 2;
    constexpr static int BWD_PRODUCER_WARPGROUPS = 1;
    constexpr static int BWD_NUM_WARPGROUPS = (BWD_CONSUMER_WARPGROUPS + BWD_PRODUCER_WARPGROUPS);
    constexpr static int BWD_NUM_WORKERS = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS);
};

struct bwd_intermediate_atomicqr_globals {
    using cfg = bwd_intermediate_atomicqr_config;
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
// ============================================================================
// Backward Helper Functions
// ============================================================================
// TODO: kuv_store
// TODO: bwd_loop
template<int BLOCK_SEQ, int BLOCK_INTER>
__device__ static inline void
compute_bwd_loop_head128_intermediate_atomicqr(
    kittens::semaphore *q_smem_arrived, kittens::semaphore *r_smem_arrived, kittens::semaphore *do_smem_arrived,
    auto &q_smem, auto &k_smem, auto &u_smem, auto &v_smem, auto &r_smem, auto &do_smem,
    int seq_idx, int tic, int toc
    // more register files
) {
/*
    Algorithm that only takes 4 registers blocks of <BLOCKSEQ, BLOCKINTER>
    R1 <- Q.K^T=M
    R2 <- sigmoid(R1)=sigM
    R1 <- R1*R2=siluM
    R3 <- 1.0-R1=1-siluM
    R2 <- R2*R3 = sigM(1-siluM)
    R2 <- R1+R2=siluM+sigM(1-siluM)=silu'M
    R3 <- dO.V^T=dA
    R4 <- R3*R1*Router=dN
    dU = R4^T.Q --- dU complete
    R4 <- Q.U^T=N
    R4 <- R4*Router=expertN
    R1 <- R1*R4=A
    dV = R1^T.dO --- dV complete
    R1 <- R3*R4*R2=dM
    dK = R1^T.Q --- dK complete
    ---All dK, dU, dV complete---
*/
    using cfg = bwd_intermediate_atomicqr_config;
    
}
// ============================================================================
// Backward Kernel(currently suffers from register spilling by 3.5K bytes, need move some reg into smem)
// ============================================================================
// TODO: consider change all gradients to bf16
__global__ __launch_bounds__(bwd_intermediate_atomicqr_config::BWD_NUM_WORKERS * kittens::WARP_THREADS, bwd_intermediate_atomicqr_config::BLOCKS_SM)
void flashffn_moe_bwd_kernel_head128_intermediate_atomicqr(const __grid_constant__ bwd_intermediate_atomicqr_globals globals) {
    using cfg = bwd_intermediate_atomicqr_config;
    using glbs = bwd_intermediate_atomicqr_globals;
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

    // MAGIC: the smem must be place as such order for later reinterpretation
    // It also requires BLOCK_SEQ=BLOCK_INTER, BWD_CONSUMER_WARPGROUPS=NUM_STAGES=2
    k_tile  (&k_smem) [cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<k_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    q_tile  (&q_smem) [cfg::NUM_STAGES] = al.allocate<q_tile,  cfg::NUM_STAGES>(); // in this order to overwrite
    u_tile  (&u_smem) [cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<u_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    do_tile (&do_smem)[cfg::NUM_STAGES] = al.allocate<do_tile, cfg::NUM_STAGES>(); // in this order to overwrite
    v_tile  (&v_smem) [cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<v_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    dq_tile (&dq_smem) = al.allocate<dq_tile>(); // in this order to overwrite
    
    middle_tile (&m_smem)[cfg::BWD_CONSUMER_WARPGROUPS] = al.allocate<middle_tile, cfg::BWD_CONSUMER_WARPGROUPS>();
    r_vect  (&r_smem) [cfg::NUM_STAGES] = al.allocate<r_vect, cfg::NUM_STAGES>();
    dr_vect (&dr_smem) = al.allocate<dr_vect>();

    dk_tile (*dk_smem) = reinterpret_cast<dk_tile*>(&k_smem[0].data[0]);
    du_tile (*du_smem) = reinterpret_cast<du_tile*>(&u_smem[0].data[0]);
    dv_tile (*dv_smem) = reinterpret_cast<dv_tile*>(&v_smem[0].data[0]);
    
    const int batch_id = blockIdx.z;
    const int warp_id = kittens::warpid();
    const int warpgroup_id = warp_id / kittens::WARPGROUP_WARPS;
    const int tiles_per_expert = globals.expert_dim / cfg::BLOCK_INTER;
    // we don't do  globals.seq_len / (cfg::BLOCK_SEQ * CONSUMER_NUM) because we implement individual logic for each consumer groups.
    const int total_seq_blocks = globals.seq_len / (cfg::BLOCK_SEQ);
    const int inter_tile_idx = (blockIdx.x * cfg::BWD_CONSUMER_WARPGROUPS);
    const int head_idx = (blockIdx.y);
    const int expert_idx = inter_tile_idx / tiles_per_expert;
    const int real_inter_tile_idx = inter_tile_idx % tiles_per_expert;

    __shared__ kittens::semaphore kuv_smem_arrived; // only 1, we will wait until all consumer groups' kuv_smem are ready
    __shared__ kittens::semaphore q_smem_arrived[cfg::NUM_STAGES], r_smem_arrived[cfg::NUM_STAGES], do_smem_arrived[cfg::NUM_STAGES];
    __shared__ kittens::semaphore compute_done[cfg::NUM_STAGES]; // TODO: is it num_stage or num_consumer_groups? confused.
    __shared__ kittens::semaphore dqdr_smem_ready_to_write;

    // ----- warmup prefetch -----
    const int warmup_seq_idx = 0;
    if (threadIdx.x == 0) {
        int tic = 0, toc = 1;
        init_semaphore(kuv_smem_arrived, 0, 1);
        for (int stage = 0; stage != cfg::NUM_STAGES; ++stage) {
            init_semaphore(q_smem_arrived[stage], 0, 1);
            init_semaphore(r_smem_arrived[stage], 0, 1);
            init_semaphore(do_smem_arrived[stage], 0, 1);
            init_semaphore(compute_done[stage], 1, 0);
        }
        init_semaphore(dqdr_smem_ready_to_write, 1, 0);
        // arrive(dqdr_smem_ready_to_write); // it's already ready. ACTUALLY, this is not needed, why?
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
        // only warmup prefetch the [tic] stage
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
    // producer warpgroup(2) - we stick to the pattern of using last warpgroup as producers
    if (warpgroup_id == cfg::BWD_NUM_WARPGROUPS - 1) {
        warpgroup::decrease_registers<24>();
        // warp0 for loading q,do,r
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
        // warp1 for writing dqdr
        } else if (warp_id % kittens::WARPGROUP_WARPS == 1) {
            int tic = 0, toc = 1;
            for (int seq_idx = 0; seq_idx != total_seq_blocks; ++seq_idx, tic^=1, toc^=1) {
                const int phase = (seq_idx / cfg::NUM_STAGES) & 1;
                wait(compute_done[tic], phase);
                coord<dq_tile> dq_tile_idx = {batch_id, head_idx, seq_idx, 0};
                coord<dr_vect> dr_vect_idx = {batch_id, head_idx, expert_idx, seq_idx};
                tma::store_add_async(globals.dQ, dq_smem, dq_tile_idx);
                tma::store_add_async(globals.dR, dr_smem, dr_vect_idx);
                tma::store_async_wait();
                if ((warpgroup::laneid() % kittens::WARP_THREADS) == 0) arrive(dqdr_smem_ready_to_write);
            }
        }
    // consumer warpgroup(0,1) - the first two warpgroups are consumers
    } else {
        // gradient accumulation registers
        rt_fl<cfg::BLOCK_INTER/kittens::WARPGROUP_WARPS, HEAD_DIM> dk_acc_reg, du_acc_reg, dv_acc_reg;
        // rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, HEAD_DIM> q_reg, do_reg;
        rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, HEAD_DIM> dq_reg;
        
        // i designed a very delicate and beautiful algorithms that only takes 4 registers to compute dK,dU,dV
        // also the algorithm elegantly use dr_smem to store result of dr of warp1, while warp0 only store to dr_reg
        // and later wrap0 merge two result by adding dr_reg+dr_smem. it takes no more space and is still very fast.
        // i am a pure genius
        rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER> R1,R2,R3,R4;
        rt_bf<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, cfg::BLOCK_INTER, kittens::ducks::rt_layout::row> R_bf16;
        rv_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> router_reg;
        rv_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> dr_reg;

        zero(dk_acc_reg); zero(du_acc_reg); zero(dv_acc_reg); 
        // consumer warpgroup 0
        if (warpgroup_id == 0) {
            warpgroup::increase_registers<256>(); //maybe change to 240
            rv_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, kittens::ducks::rv_layout::ortho> dr_reg_1;
            rt_fl<cfg::BLOCK_SEQ/kittens::WARPGROUP_WARPS, HEAD_DIM> dq_reg_1;
            int tic = 0, toc = 1;
            wait(kuv_smem_arrived, 0);
            for (int seq_idx = 0; seq_idx != total_seq_blocks; ++seq_idx, tic^=1, toc ^= 1) {
                // compute dk,du,dv of warpgroup 0
                const int phase = (seq_idx / cfg::NUM_STAGES) & 1; // TODO: why? consider change to (seq_idx / cfg::NUM_STAGES) & 1
                one(R3); // always reset R3 to full 1.0
                zero(dr_reg); // zero dr_reg for later accumualtion
                wait(q_smem_arrived[tic], phase);
                warpgroup::mm_ABt(R1, q_smem[tic], k_smem[warpgroup_id]); // R1=Q.K^T=M
                warpgroup::mma_async_wait(); // wait for R1
                wait(do_smem_arrived[tic], phase);
                warpgroup::mm_ABt(R4, do_smem[tic], v_smem[warpgroup_id]); // R4=dO.V^T=dA
                warpgroup::mma_commit_group(); // commit R4
                #ifdef USE_FAST_SIGMOID
                    fast_sigmoid(R2, R1);
                #else
                    mul(R2, R1, -RCP_LN2);
                    exp2(R2,R2);
                    add(R2,R2,1.0f);
                    rcp_inplace_approx(R2); // R2=sigmoid(R1)=sigM
                #endif
                mul(R1,R1,R2); // R1=R1*R2=M*sigM=siluM
                sub(R3,R3,R1); //R3=1.0-R1=1-siluM
                mul(R2,R2,R3); //R2=R2*R3=sigM(1-siluM)
                add(R2,R1,R2); //R2=R1+R2=siluM+sigM(1-siluM)=silu'M
                // TODO: consider R1 commit, R4 commit, mma_async_wait<1>(). might work
                warpgroup::mma_async_wait(); // wait for R4=dA
                mul(R3,R4,R1); // R3=R4*R1=dA*siluM
                wait(r_smem_arrived[tic],phase);
                warpgroup::load(router_reg, r_smem[tic]);
                mul_row(R3,R3,router_reg); // R3=R3*router=dA*siluM*router=dN
                copy(R_bf16, R3); // conversion from f32->bf16 for saving into m_smem
                warpgroup::store(m_smem[warpgroup_id], R_bf16); // m_smem=R3=dN
                warpgroup::mm_ABt(R3, q_smem[tic], u_smem[warpgroup_id]); // R3=Q.U^T=N
                warpgroup::mma_async_wait(); // wait for R3=Q.U^T=N
                // right now, we have R1=siluM, R2=silu'M, R3=N, R4=dA, we can compute dR=sum_row(dA*siluM*N)
                accumulate_dr_rowsum_triple(dr_reg, R4, R1, R3); // up to now, dR is correct => R1,R3,R4 is correct
                warpgroup::mm_AB(dq_reg, R_bf16, u_smem[warpgroup_id]); // dQ=dN.U
                warpgroup::mma_commit_group(); // async compute dQ=dN.U
                group<4>::sync(warpgroup::groupid()+4); // wait for m_smem writing complete
                warpgroup::mma_AtB(du_acc_reg, m_smem[warpgroup_id], q_smem[tic]); // dU+=dN^T.Q
                warpgroup::mma_commit_group(); // async compute dU+=dN^T.Q, don't need to wait
                mul_row(R3,R3,router_reg); //R3=R3*router=N*router=expertN
                mul(R1,R1,R3); //R1=R1*R3=siluM*expertN=A
                // TODO: consider change to mma_async_wait<1> so that still allow dq to compute
                warpgroup::mma_async_wait(); // wait for previous dU,dQ complete so that we can overwrite m_smem
                copy(R_bf16, R1); // conversion from f32->bf16 for saving into m_smem
                warpgroup::store(m_smem[warpgroup_id], R_bf16); // m_smem=A
                group<4>::sync(warpgroup::groupid()+4); // wait for m_smem writing complete
                warpgroup::mma_AtB(dv_acc_reg, m_smem[warpgroup_id], do_smem[tic]); // dV+=A^T.dO
                warpgroup::mma_commit_group();
                mul(R4,R3,R4); // R4=R3*R4=expertN*dA
                mul(R2,R4,R2); // R2=R4*R2=expertN*dA*silu'M=dM
                copy(R_bf16, R2); // conversion from f32->bf16 for saving into m_smem
                warpgroup::mma_async_wait(); // wait for previous dK complete so that we can overwrite m_smem
                warpgroup::store(m_smem[warpgroup_id], R_bf16); // m_smem=dM
                warpgroup::mma_AB(dq_reg, R_bf16, k_smem[warpgroup_id]); // dQ+=dM.K
                warpgroup::mma_commit_group();
                group<4>::sync(warpgroup::groupid()+4); // wait for m_smem writing complete
                warpgroup::mma_AtB(dk_acc_reg, m_smem[warpgroup_id], q_smem[tic]); // dK+=dM^T.Q
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // dK,dU,dV all accumulates correctly to reg files, also, m_smem stores dM
                
                // TODO: merge warp0's dq_reg, dr_reg with warp1's dq_smem, dr_smem
                group<8>::sync(10); // sync across groups, dq_smem and dr_smem should contains the result from warpgroup1 now
                warpgroup::load(dq_reg_1, dq_smem);
                warpgroup::load(dr_reg_1, dr_smem);
                add(dq_reg,dq_reg,dq_reg_1);
                add(dr_reg,dr_reg,dr_reg_1);
                wait(dqdr_smem_ready_to_write, toc); // wait until smem is written into gmem
                if (seq_idx > 0) tma::store_async_wait();
                warpgroup::store(dq_smem,dq_reg);
                warpgroup::store(dr_smem,dr_reg);
                group<4>::sync(warpgroup::groupid()+4); // wait for all writing done.

                if (warpgroup::laneid() == 0) arrive(compute_done[tic]);
                
            }
            // store kuv using tma::store_add_async
            int current_inter_tile_idx = (blockIdx.x * cfg::BWD_CONSUMER_WARPGROUPS) + warpgroup_id;
            int current_expert_idx = current_inter_tile_idx / tiles_per_expert;
            int current_real_inter_tile_idx = current_inter_tile_idx % tiles_per_expert;
            group<8>::sync(10);
            warpgroup::store(dk_smem[warpgroup_id], dk_acc_reg);
            coord<dk_tile> dkuv_tile_idx = {head_idx, current_expert_idx, current_real_inter_tile_idx, 0};
            group<4>::sync(warpgroup::groupid()+4);
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                tma::store_add_async(globals.dK, dk_smem[warpgroup_id], dkuv_tile_idx);
                tma::store_commit_group();
            }
            warpgroup::store(du_smem[warpgroup_id], du_acc_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                tma::store_add_async(globals.dU, du_smem[warpgroup_id], dkuv_tile_idx);
                tma::store_commit_group();
            }
            wait(dqdr_smem_ready_to_write, toc); // this might not be requires since dk_smem[0] does not need dq_smem
            warpgroup::store(dv_smem[warpgroup_id], dv_acc_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                tma::store_add_async(globals.dV, dv_smem[warpgroup_id], dkuv_tile_idx);
                tma::store_commit_group();
            }
            tma::store_async_wait();

        // consumer warpgroup 1
        } else if (warpgroup_id == 1) {
            warpgroup::increase_registers<224>(); //maybe change to 240
            int tic = 0, toc = 1;
            wait(kuv_smem_arrived, 0);
            for (int seq_idx = 0; seq_idx != total_seq_blocks; ++seq_idx, tic^=1, toc ^= 1) {
                // compute dk,du,dv of warpgroup 0
                const int phase = (seq_idx / cfg::NUM_STAGES) & 1; // TODO: why? consider change to (seq_idx / cfg::NUM_STAGES) & 1
                one(R3); // always reset R3 to full 1.0
                zero(dr_reg); // zero dr_reg for later accumualtion
                wait(q_smem_arrived[tic], phase);
                warpgroup::mm_ABt(R1, q_smem[tic], k_smem[warpgroup_id]); // R1=Q.K^T=M
                warpgroup::mma_async_wait(); // wait for R1
                wait(do_smem_arrived[tic], phase);
                warpgroup::mm_ABt(R4, do_smem[tic], v_smem[warpgroup_id]); // R4=dO.V^T=dA
                warpgroup::mma_commit_group(); // commit R4
                #ifdef USE_FAST_SIGMOID
                    fast_sigmoid(R2, R1);
                #else
                    mul(R2, R1, -RCP_LN2);
                    exp2(R2,R2);
                    add(R2,R2,1.0f);
                    rcp_inplace_approx(R2); // R2=sigmoid(R1)=sigM
                #endif
                mul(R1,R1,R2); // R1=R1*R2=M*sigM=siluM
                sub(R3,R3,R1); //R3=1.0-R1=1-siluM
                mul(R2,R2,R3); //R2=R2*R3=sigM(1-siluM)
                add(R2,R1,R2); //R2=R1+R2=siluM+sigM(1-siluM)=silu'M
                // TODO: consider R1 commit, R4 commit, mma_async_wait<1>(). might work
                warpgroup::mma_async_wait(); // wait for R4=dA
                mul(R3,R4,R1); // R3=R4*R1=dA*siluM
                wait(r_smem_arrived[tic],phase);
                warpgroup::load(router_reg, r_smem[tic]);
                mul_row(R3,R3,router_reg); // R3=R3*router=dA*siluM*router=dN
                copy(R_bf16, R3); // conversion from f32->bf16 for saving into m_smem
                warpgroup::store(m_smem[warpgroup_id], R_bf16); // m_smem=R3=dN
                warpgroup::mm_ABt(R3, q_smem[tic], u_smem[warpgroup_id]); // R3=Q.U^T=N
                warpgroup::mma_async_wait(); // wait for R3=Q.U^T=N
                // right now, we have R1=siluM, R2=silu'M, R3=N, R4=dA, we can compute dR=sum_row(dA*siluM*N)
                accumulate_dr_rowsum_triple(dr_reg, R4, R1, R3); // up to now, dR is correct => R1,R3,R4 is correct
                warpgroup::mm_AB(dq_reg, R_bf16, u_smem[warpgroup_id]); // dQ=dN.U
                warpgroup::mma_commit_group(); // async compute dQ=dN.U
                group<4>::sync(warpgroup::groupid()+4); // wait for m_smem writing complete
                warpgroup::mma_AtB(du_acc_reg, m_smem[warpgroup_id], q_smem[tic]); // dU+=dN^T.Q
                warpgroup::mma_commit_group(); // async compute dU+=dN^T.Q, don't need to wait
                mul_row(R3,R3,router_reg); //R3=R3*router=N*router=expertN
                mul(R1,R1,R3); //R1=R1*R3=siluM*expertN=A
                // TODO: consider change to mma_async_wait<1> so that still allow dq to compute
                warpgroup::mma_async_wait(); // wait for previous dU,dQ complete so that we can overwrite m_smem
                copy(R_bf16, R1); // conversion from f32->bf16 for saving into m_smem
                warpgroup::store(m_smem[warpgroup_id], R_bf16); // m_smem=A
                group<4>::sync(warpgroup::groupid()+4); // wait for m_smem writing complete
                warpgroup::mma_AtB(dv_acc_reg, m_smem[warpgroup_id], do_smem[tic]); // dV+=A^T.dO
                warpgroup::mma_commit_group();
                mul(R4,R3,R4); // R4=R3*R4=expertN*dA
                mul(R2,R4,R2); // R2=R4*R2=expertN*dA*silu'M=dM
                copy(R_bf16, R2); // conversion from f32->bf16 for saving into m_smem
                warpgroup::mma_async_wait(); // wait for previous dK complete so that we can overwrite m_smem
                warpgroup::store(m_smem[warpgroup_id], R_bf16); // m_smem=dM
                warpgroup::mma_AB(dq_reg, R_bf16, k_smem[warpgroup_id]); // dQ+=dM.K
                warpgroup::mma_commit_group();
                group<4>::sync(warpgroup::groupid()+4); // wait for m_smem writing complete
                warpgroup::mma_AtB(dk_acc_reg, m_smem[warpgroup_id], q_smem[tic]); // dK+=dM^T.Q
                warpgroup::mma_commit_group();
                warpgroup::mma_async_wait(); // dK,dU,dV all accumulates correctly to reg files, also, m_smem stores dM

                // store dq_reg and dr_reg into dq_smem and dr_smem waiting for warp0 to do the merge
                warpgroup::store(dr_smem, dr_reg); // TODO: later move this in upper code to advance overlapping
                warpgroup::store(dq_smem, dq_reg);
                group<4>::sync(warpgroup::groupid()+4); // sync for writing smem. (TODO: maybe not needed since later whole warpgroup sync)
                group<8>::sync(10); // sync across groups
            }
            // store kuv using tma::store_add_async
            int current_inter_tile_idx = (blockIdx.x * cfg::BWD_CONSUMER_WARPGROUPS) + warpgroup_id;
            int current_expert_idx = current_inter_tile_idx / tiles_per_expert;
            int current_real_inter_tile_idx = current_inter_tile_idx % tiles_per_expert;
            group<8>::sync(10);
            warpgroup::store(dk_smem[warpgroup_id], dk_acc_reg);
            coord<dk_tile> dkuv_tile_idx = {head_idx, current_expert_idx, current_real_inter_tile_idx, 0};
            group<4>::sync(warpgroup::groupid()+4);
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                tma::store_add_async(globals.dK, dk_smem[warpgroup_id], dkuv_tile_idx);
                tma::store_commit_group();
            }
            warpgroup::store(du_smem[warpgroup_id], du_acc_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                tma::store_add_async(globals.dU, du_smem[warpgroup_id], dkuv_tile_idx);
                tma::store_commit_group();
            }
            wait(dqdr_smem_ready_to_write, toc); // wait for dq to be able to get overwritten
            warpgroup::store(dv_smem[warpgroup_id], dv_acc_reg);
            group<4>::sync(warpgroup::groupid()+4);
            if (kittens::warpid() % kittens::WARPGROUP_WARPS == 0) {
                tma::store_add_async(globals.dV, dv_smem[warpgroup_id], dkuv_tile_idx);
                tma::store_commit_group();
            }
            tma::store_async_wait();
        }
    }
}


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
    assert(seq_len % (fwd_config::BLOCK_SEQ * fwd_config::CONSUMER_WARPGROUPS) == 0); // For the loop to work, we must have seq_len % BLOCK_SEQ == 0. TODO: remove this and fix the kernel
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
    auto smem_size = kittens::MAX_SHARED_MEMORY;
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
    cudaStreamSynchronize(stream);
}

// ============================================================================
// Backward Python Binding
// ============================================================================
void flashffn_moe_backward_intermediate_atomicqr(
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

    using bwd_config = bwd_intermediate_atomicqr_config;

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
    TORCH_CHECK(seq_len % (bwd_config::BLOCK_SEQ * bwd_config::BWD_CONSUMER_WARPGROUPS) == 0, "seq_len % (BLOCK_SEQ * CONS_WARPGRPS) == 0 required");

    using q_gl = bwd_intermediate_atomicqr_globals::q_gl;
    using k_gl = bwd_intermediate_atomicqr_globals::k_gl;
    using u_gl = bwd_intermediate_atomicqr_globals::u_gl;
    using v_gl = bwd_intermediate_atomicqr_globals::v_gl;
    using r_gl = bwd_intermediate_atomicqr_globals::r_gl;

    using do_gl = bwd_intermediate_atomicqr_globals::do_gl;
    using dq_gl = bwd_intermediate_atomicqr_globals::dq_gl;
    using dk_gl = bwd_intermediate_atomicqr_globals::dk_gl;
    using du_gl = bwd_intermediate_atomicqr_globals::du_gl;
    using dv_gl = bwd_intermediate_atomicqr_globals::dv_gl;
    using dr_gl = bwd_intermediate_atomicqr_globals::dr_gl;

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

    bwd_intermediate_atomicqr_globals g{
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
    auto tile = bwd_config::BLOCK_INTER * bwd_config::BWD_CONSUMER_WARPGROUPS;
    dim3 grid(
        (unsigned)((intermediate_size + tile - 1) / tile),
        (unsigned)num_heads,
        (unsigned)batch_size
    );
    cudaFuncSetAttribute(flashffn_moe_bwd_kernel_head128_intermediate_atomicqr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    cudaFuncSetAttribute(flashffn_moe_bwd_kernel_head128_intermediate_atomicqr, cudaFuncAttributePreferredSharedMemoryCarveout, 95); // from h100.cu
    flashffn_moe_bwd_kernel_head128_intermediate_atomicqr<<<grid, threads, smem_size, stream>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
}

PYBIND11_MODULE(flash_ffn_moe, m) {
  m.def("flashffn_moe_forward", &flashffn_moe_forward, "FlashFFN-MoE forward (H100/TK)");
  m.def("flashffn_moe_backward_intermediate_atomicqr", &flashffn_moe_backward_intermediate_atomicqr, "FlashFFN-MoE backward. Loop over intermediate and do atomic add on dQ, dR(H100/TK)");
}

#endif