#pragma once

// #include <math_functions.h>
#include <cuda_runtime.h>
#include "kittens.cuh"

constexpr float RCP_LN2 = 1.4426950408889634f;  // = 1.0 / ln(2)

// ============================================================================
// RCP(1/x)
// ============================================================================

// I'm not sure this is fast or not, will test later
__device__ __forceinline__ float2 rcp2_elem(float2 a) {
    return float2{__frcp_rd(a.x), __frcp_rd(a.y)};
}

__device__ __forceinline__ float rcp_fast(float x) {
    float y;
    asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__forceinline__ __device__ float tanh_fast (float x) {
    float r;
    asm ("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

template<int H, int W>
__device__ inline void fast_sigmoid(kittens::rt_fl<H, W> &dst, kittens::rt_fl<H, W> &src) {
    #pragma unroll
    for (int i = 0; i != dst.height; ++i) {
        #pragma unroll
        for (int j = 0; j != dst.width; ++j) {
            #pragma unroll
            for (int k = 0; k != dst.packed_per_tile; ++k) {
                dst.tiles[i][j].data[k].x = fmaf (0.5, tanh_fast (0.5f * src.tiles[i][j].data[k].x), 0.5f);
                dst.tiles[i][j].data[k].y = fmaf (0.5, tanh_fast (0.5f * src.tiles[i][j].data[k].y), 0.5f);
            }
        }
    }
}

template<int H, int W>
__device__ inline void fast_silu(kittens::rt_fl<H, W> &dst, kittens::rt_fl<H, W> &src) {
    #pragma unroll
    for (int i = 0; i != dst.height; ++i) {
        #pragma unroll
        for (int j = 0; j != dst.width; ++j) {
            #pragma unroll
            for (int k = 0; k != dst.packed_per_tile; ++k) {
                dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x*fmaf (0.5, tanh_fast (0.5f * src.tiles[i][j].data[k].x), 0.5f);
                dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y*fmaf (0.5, tanh_fast (0.5f * src.tiles[i][j].data[k].y), 0.5f);
            }
        }
    }
}

__forceinline__ __device__ kittens::bf16_2 tanh_fast_bf16 (kittens::bf16_2 x) {
    uint32_t in = *reinterpret_cast<const uint32_t*>(&x);
    uint32_t r;
    asm ("tanh.approx.bf16x2 %0,%1; \n\t" : "=r"(r) : "r"(in));
    return *reinterpret_cast<__nv_bfloat162*>(&r);
}

template<int H, int W>
__device__ inline void fast_sigmoid_bf16(kittens::rt_bf<H, W> &dst, kittens::rt_bf<H, W> &src) {
    const __nv_bfloat162 half = __float2bfloat162_rn(0.5f);
    #pragma unroll
    for (int i = 0; i != dst.height; ++i) {
        #pragma unroll
        for (int j = 0; j != dst.width; ++j) {
            #pragma unroll
            for (int k = 0; k != dst.packed_per_tile; ++k) {
                dst.tiles[i][j].data[k] = __hfma2(half, tanh_fast_bf16(__hmul2(half,src.tiles[i][j].data[k])), half);
            }
        }
    }
}

template<int H, int W>
__device__ inline void fast_silu_bf16(kittens::rt_bf<H, W> &dst, kittens::rt_bf<H, W> &src) {
    const __nv_bfloat162 half = __float2bfloat162_rn(0.5f);
    #pragma unroll
    for (int i = 0; i != dst.height; ++i) {
        #pragma unroll
        for (int j = 0; j != dst.width; ++j) {
            #pragma unroll
            for (int k = 0; k != dst.packed_per_tile; ++k) {
                __nv_bfloat162 x  = src.tiles[i][j].data[k];        // bf16x2
                __nv_bfloat162 hx = __hmul2(half, x);               // 0.5 * x (逐lane)
                __nv_bfloat162 th = tanh_fast_bf16(hx);             // tanh(0.5 * x) (逐lane)
                // SiLU(x) = 0.5*x*tanh(0.5*x) + 0.5*x  ==>  out = hfma2(hx, th, hx)
                __nv_bfloat162 out = __hfma2(hx, th, hx);
                dst.tiles[i][j].data[k] = out;
            }
        }
    }
}

template<int H, int W>
__device__ inline void rcp_inplace(kittens::rt_fl<H, W> &t) {
    #pragma unroll
    for (int i = 0; i != t.height; ++i) {
        #pragma unroll
        for (int j = 0; j != t.width; ++j) {
            #pragma unroll
            for (int k = 0; k != t.packed_per_tile; ++k) {
                t.tiles[i][j].data[k].x = __frcp_rd(t.tiles[i][j].data[k].x);
                t.tiles[i][j].data[k].y = __frcp_rd(t.tiles[i][j].data[k].y);
            }
        }
    }
}

template<int H, int W>
__device__ inline void rcp_inplace_approx(kittens::rt_fl<H, W> &t) {
    #pragma unroll
    for (int i = 0; i != t.height; ++i) {
        #pragma unroll
        for (int j = 0; j != t.width; ++j) {
            #pragma unroll
            for (int k = 0; k != t.packed_per_tile; ++k) {
                float2 v = t.tiles[i][j].data[k];
                float rx, ry;
                asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(rx) : "f"(v.x));
                asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(ry) : "f"(v.y));
                t.tiles[i][j].data[k] = make_float2(rx, ry);
            }
        }
    }
}

template<int H, int W>
__device__ inline void rcp_inplace_approx_realinplace(kittens::rt_fl<H, W> &t) {
    #pragma unroll
    for (int i = 0; i != t.height; ++i) {
        #pragma unroll
        for (int j = 0; j != t.width; ++j) {
            #pragma unroll
            for (int k = 0; k != t.packed_per_tile; ++k) {
                asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(t.tiles[i][j].data[k].x) : "f"(t.tiles[i][j].data[k].x));
                asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(t.tiles[i][j].data[k].y) : "f"(t.tiles[i][j].data[k].y));
            }
        }
    }
}

template<int H, int W>
__device__ inline void silu_inplace_approx(kittens::rt_fl<H, W> &t) {
    #pragma unroll
    for (int i = 0; i != t.height; ++i) {
        #pragma unroll
        for (int j = 0; j != t.width; ++j) {
            #pragma unroll
            for (int k = 0; k != t.packed_per_tile; ++k) {
                float2 v = t.tiles[i][j].data[k];

                // Compute sigmoid(x) ≈ 1 / (1 + exp(-x)) using fast ex2 + rcp, then y = x * sigmoid(x)
                // float tx = -v.x * RCP_LN2;
                // float ty = -v.y * RCP_LN2;

                // float f1, f2;
                // asm volatile ("ex2.approx.f32 %0, %1;" : "=f"(f1) : "f"(tx));
                // asm volatile ("ex2.approx.f32 %0, %1;" : "=f"(f2) : "f"(ty));

                // f1 = 1.0f + f1;
                // f2 = 1.0f + f2;
                float f1 = 1.0f + exp2f(-v.x * RCP_LN2);
                float f2 = 1.0f + exp2f(-v.y * RCP_LN2);

                float f3, f4;
                asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(f3) : "f"(f1));
                asm volatile ("rcp.approx.f32 %0, %1;" : "=f"(f4) : "f"(f2));

                t.tiles[i][j].data[k] = make_float2(v.x * f3, v.y * f4);
            }
        }
    }
}

template<int H, int W>
__device__ inline void silu_inplace_approx_fancy(kittens::rt_fl<H, W> &t) {
    #pragma unroll
    for (int i = 0; i != t.height; ++i) {
        #pragma unroll
        for (int j = 0; j != t.width; ++j) {
            #pragma unroll
            for (int k = 0; k != t.packed_per_tile; ++k) {
                float2 &p = t.tiles[i][j].data[k];

                float tmp = -p.x * RCP_LN2;
                float tmp2 = -p.y * RCP_LN2;
                asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(tmp));
                asm volatile ("ex2.approx.f32 %0, %0;" : "+f"(tmp2));
                tmp = 1.0f + tmp;
                tmp2 = 1.0f + tmp2;
                asm volatile ("rcp.approx.f32 %0, %0;" : "+f"(tmp));
                asm volatile ("rcp.approx.f32 %0, %0;" : "+f"(tmp2));
                p.x *= tmp;
                p.y *= tmp2;
            }
        }
    }
}

// ============================================================================
// row_sum(A*B*C) -> vector_register
// ============================================================================
// Minimal-register row-sum of triple product: dr_reg += sum_row(dA * siluM * N).
template<typename RV, typename RT>
__device__ inline void accumulate_dr_rowsum_triple(RV &dr_reg, const RT &dA, const RT &siluM, const RT &N) {
    using dtype = typename RT::dtype; // packed type, e.g., float2
    constexpr int H = RT::height;
    constexpr int W = RT::width;

    // Row-major path: leader is the first lane among the 4 lanes that own a row pack
    const int leader = threadIdx.x & 0x1C;

    #pragma unroll
    for (int i = 0; i < H; ++i) {
        dtype accum_top  = kittens::base_ops::zero::op<dtype>();
        dtype accum_bot  = kittens::base_ops::zero::op<dtype>();

        #pragma unroll
        for (int j = 0; j < W; ++j) {
            #pragma unroll
            for (int k = 0; k < RT::packed_per_tile; k += 2) {
                // top half
                accum_top = kittens::base_ops::sum::op<dtype>(
                    accum_top,
                    kittens::base_ops::mul::op<dtype>(
                        kittens::base_ops::mul::op<dtype>(dA.tiles[i][j].data[k+0], siluM.tiles[i][j].data[k+0]),
                        N.tiles[i][j].data[k+0]
                    )
                );
                // bottom half
                accum_bot = kittens::base_ops::sum::op<dtype>(
                    accum_bot,
                    kittens::base_ops::mul::op<dtype>(
                        kittens::base_ops::mul::op<dtype>(dA.tiles[i][j].data[k+1], siluM.tiles[i][j].data[k+1]),
                        N.tiles[i][j].data[k+1]
                    )
                );
            }
        }

        // Pack each half-row into a single float per half
        dtype accum_packed;
        accum_packed.x = accum_top.x + accum_top.y;
        accum_packed.y = accum_bot.x + accum_bot.y;

        // Intra-warp reduction across the 4 lanes that own this row
        accum_packed = kittens::base_ops::sum::op<dtype>(accum_packed, kittens::packed_shfl_down_sync(kittens::MASK_ALL, accum_packed, 2));
        accum_packed = kittens::base_ops::sum::op<dtype>(accum_packed, kittens::packed_shfl_down_sync(kittens::MASK_ALL, accum_packed, 1));

        // Broadcast from leader lane, then accumulate onto dr_reg
        accum_packed   = kittens::packed_shfl_sync(kittens::MASK_ALL, accum_packed, leader);
        dr_reg[i][0]   = kittens::base_ops::sum::op<typename RV::dtype>(dr_reg[i][0], accum_packed);
    }
}
