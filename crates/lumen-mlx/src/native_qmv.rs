//! MLX's quantized matvec, transcribed for the decode kernels that must add what MLX adds in the
//! order MLX adds it.

/// MLX's own quantized matvec, transcribed (`kernels/quantized.h`, MLX 0.32): the helpers its
/// `qmv_fast` runs, so a kernel built on them adds in the order `quantized_matmul` and `gather_qmm`
/// do. A matvec row is one simdgroup's lanes, each 16 values of every 512 (4-bit weights), summed
/// per lane in `qdot`'s order and then across lanes with `simd_sum`; changing that partition or
/// order changes the low bits.
pub(crate) const QMV_HEADER: &str = r#"
#define RB(v) static_cast<float>(static_cast<bfloat16_t>(v))

// load_vector over bf16 activations. The composed path casts the activation to bf16 before the
// gather, so `x` (f32 here) is rounded first; the quad sum then rounds to bf16 after every add, as
// `sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3]` does on bf16 operands.
inline float load_vector_bf16(const device float* x, thread float* x_thread) {
    float sum = 0;
    for (int i = 0; i < 16; i += 4) {
        const float a = RB(x[i]);
        const float b = RB(x[i + 1]);
        const float c = RB(x[i + 2]);
        const float d = RB(x[i + 3]);
        sum += RB(RB(RB(a + b) + c) + d);
        x_thread[i] = a;
        x_thread[i + 1] = b / 16.0f;
        x_thread[i + 2] = c / 256.0f;
        x_thread[i + 3] = d / 4096.0f;
    }
    return sum;
}

// load_vector over f32 activations.
inline float load_vector_f32(const device float* x, thread float* x_thread) {
    float sum = 0;
    for (int i = 0; i < 16; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
    }
    return sum;
}

// qdot for 4-bit weights, 16 values.
inline float qdot4(
    const device uint16_t* ws,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
    float accum = 0;
    for (int i = 0; i < 4; i++) {
        accum +=
            (x_thread[4 * i] * (ws[i] & 0x000f) +
             x_thread[4 * i + 1] * (ws[i] & 0x00f0) +
             x_thread[4 * i + 2] * (ws[i] & 0x0f00) +
             x_thread[4 * i + 3] * (ws[i] & 0xf000));
    }
    return scale * accum + sum * bias;
}

// load_vector and qdot for 8-bit weights over f32 activations, `V` values a lane (`V` is 8 under
// `qmv_fast`, 4 under `qmv`).
template <int V>
inline float load_vector8(const device float* x, thread float* x_thread) {
    float sum = 0;
    for (int i = 0; i < V; i++) {
        sum += x[i];
        x_thread[i] = x[i];
    }
    return sum;
}
template <int V>
inline float qdot8(
    const device uint8_t* w,
    const thread float* x_thread,
    float scale,
    float bias,
    float sum) {
    float accum = 0;
    for (int i = 0; i < V; i++) {
        accum += x_thread[i] * w[i];
    }
    return scale * accum + sum * bias;
}
"#;
