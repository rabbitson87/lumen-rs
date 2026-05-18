// Standalone SiLU(gate) * up kernel — Phase 17.D-1d.
//
// Replaces the production 5-dispatch chain
//   combined_bf16.to_dtype(F32) → narrow(0) → narrow(inter) → silu(gate)*up → bf16 cast
// with a single bf16-in / bf16-out dispatch.
//
// This is *not* the previously-NEGATIVE fused gate_up_silu_mul (Phase 12/13)
// which folded the matmul into the same kernel and incurred prohibitive
// register pressure. Here gate_up_proj remains its own qmv_fast dispatch;
// this kernel only does the elementwise SiLU+mul, keeping each dispatch's
// register footprint minimal.
//
// Math (per output element):
//   y[row, col] = bfloat( silu( float(combined[row, col]) ) *
//                         float(combined[row, inter + col]) )
//   where silu(x) = x / (1 + exp(-x))   (numerically stable near 0)

#include <metal_stdlib>
using namespace metal;

struct SiluMulDims {
    uint inter;  // half of combined's last-dim
};

// One thread per output element. Grid is m × inter (rows × columns).
//
//   row = thread_position_in_grid.y
//   col = thread_position_in_grid.x
//
// `combined` is [m, 2*inter] row-major: gate occupies [m, 0..inter) and
// up occupies [m, inter..2*inter).
kernel void silu_mul_bf16in_bf16out(
    device const bfloat*        combined  [[buffer(0)]],
    device       bfloat*        y         [[buffer(1)]],
    constant SiluMulDims&       dims      [[buffer(2)]],
    uint2                       gid       [[thread_position_in_grid]]
) {
    uint inter = dims.inter;
    uint row = gid.y;
    uint col = gid.x;

    if (col >= inter) return;

    uint base = row * (inter * 2u);
    float gate = float(combined[base + col]);
    float up   = float(combined[base + inter + col]);

    // silu(gate) = gate / (1 + exp(-gate)) — equivalent to gate * sigmoid(gate)
    float silu_g = gate / (1.0f + metal::exp(-gate));
    y[row * inter + col] = bfloat(silu_g * up);
}
