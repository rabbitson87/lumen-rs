//! A stacked tensor with CPU-writable slots.
//!
//! An expert cache needs the routed experts' weights in one `[n_slots, ...]`
//! array so a single `gather_qmm` can pick experts by slot index, and it needs
//! to replace a slot's bytes with another expert's when the router asks for one
//! that is not resident. On Apple silicon the GPU reads the same memory the CPU
//! writes, so a miss costs one read from disk into that slot and nothing else.
//!
//! MLX arrays are treated as immutable by everything above this module, so the
//! write is the one place that has to be careful:
//!
//! * the pool's buffer is allocated once, as a real contiguous allocation (not a
//!   broadcast of a scalar, which would have no storage to write into);
//! * [`PoolTensor::slot_ptr`] is `unsafe`: the caller promises that no GPU
//!   work that reads that slot is still running;
//! * the array is only ever used as an *input* to ops while the pool holds its
//!   own handle to it, so MLX never sees it as donatable.

use anyhow::{Result, anyhow, bail};
use mlx_rs::{Array, Dtype};

/// Bytes per element of the dtypes an expert pool holds (quantized weights are
/// packed `u32`, scales and biases `bf16`/`f16`, MX scales `u8`); `None` for
/// anything else, including complex.
fn elem_bytes(dtype: Dtype) -> Option<usize> {
    Some(match dtype {
        Dtype::Bool | Dtype::Uint8 | Dtype::Int8 => 1,
        Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
        Dtype::Uint32 | Dtype::Int32 | Dtype::Float32 => 4,
        Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 => 8,
        _ => return None,
    })
}

/// `[n_slots, ...slot_shape]` of one dtype, materialized and writable per slot.
pub(crate) struct PoolTensor {
    array: Array,
    base: *mut u8,
    n_slots: usize,
    slot_bytes: usize,
}

// SAFETY: `base` points into the buffer owned by `array`, which lives exactly as
// long as the `PoolTensor`; the type adds no aliasing of its own. Callers
// serialize writes against GPU reads themselves (see `slot_ptr`).
unsafe impl Send for PoolTensor {}

impl PoolTensor {
    /// Allocate and materialize a zero-filled pool.
    pub(crate) fn new(n_slots: usize, slot_shape: &[i32], dtype: Dtype) -> Result<Self> {
        if n_slots == 0 {
            bail!("PoolTensor::new: a pool needs at least one slot");
        }
        let elem = elem_bytes(dtype)
            .ok_or_else(|| anyhow!("PoolTensor::new: unsupported dtype {dtype:?}"))?;
        let slot_elems: usize = slot_shape.iter().map(|&d| d.max(0) as usize).product();
        let slot_bytes = slot_elems * elem;
        if slot_bytes == 0 {
            bail!("PoolTensor::new: empty slot shape {slot_shape:?}");
        }
        let mut shape = Vec::with_capacity(slot_shape.len() + 1);
        shape.push(n_slots as i32);
        shape.extend_from_slice(slot_shape);

        let array = mlx_rs::ops::zeros_dtype(&shape, dtype)
            .map_err(|e| anyhow!("PoolTensor::new: zeros: {e}"))?;
        array
            .eval()
            .map_err(|e| anyhow!("PoolTensor::new: eval: {e}"))?;

        // `zeros` could in principle come back as a stride-0 view; writing
        // through the base pointer would then hit shared storage. Refuse.
        // (MLX strides count elements, not bytes.)
        let strides = array.strides();
        let mut expect = 1usize;
        for (axis, &dim) in shape.iter().enumerate().rev() {
            if dim > 1 && strides[axis] != expect {
                bail!(
                    "PoolTensor::new: array is not contiguous row-major (strides {strides:?} for shape {shape:?})"
                );
            }
            expect *= dim.max(1) as usize;
        }

        // SAFETY: the array is evaluated, so its buffer exists; the byte view
        // of the data pointer does not depend on the element type.
        let base = unsafe { mlx_sys::mlx_array_data_uint8(array.as_ptr()) } as *mut u8;
        if base.is_null() {
            bail!("PoolTensor::new: evaluated array has no data pointer");
        }
        Ok(Self {
            array,
            base,
            n_slots,
            slot_bytes,
        })
    }

    /// The stacked array, for use as an op input. Clone the handle to keep it
    /// alive in a graph; never take ownership of the pool's own.
    pub(crate) fn array(&self) -> &Array {
        &self.array
    }

    /// Slots `base..base + len` as an array of their own, sharing the pool's storage: a gather
    /// over it sees `len` experts, not the whole pool.
    ///
    /// The sorted `gather_qmm` the prefill takes costs more the more experts the stacked tensor
    /// holds (a 512-expert layer in a 2,400-slot pool: the down projection 30 ms instead of 13,
    /// measured), so a prefill chunk gathers over a view of exactly its layer's slots.
    pub(crate) fn view(&self, base: usize, len: usize) -> Result<Array> {
        use mlx_rs::ops::indexing::IndexOp;
        if len == 0 || base + len > self.n_slots {
            bail!(
                "PoolTensor::view: slots {base}..{} out of {}",
                base + len,
                self.n_slots
            );
        }
        Ok(self.array.index(base as i32..(base + len) as i32))
    }

    #[cfg(test)]
    pub(crate) fn n_slots(&self) -> usize {
        self.n_slots
    }

    pub(crate) fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Overwrite `slot` with `bytes` (exactly one slot's worth); the tests' way
    /// to fill a pool without the IO pool.
    ///
    /// # Safety
    /// No GPU work that reads this slot may be queued or running. Ops that read
    /// *other* slots of the same array are fine. The protocol that makes this
    /// true belongs to the cache above: a slot is unmapped before it is written
    /// and only published again afterwards.
    #[cfg(test)]
    pub(crate) unsafe fn write_slot(&self, slot: usize, bytes: &[u8]) {
        assert!(slot < self.n_slots, "slot {slot} out of {}", self.n_slots);
        assert_eq!(
            bytes.len(),
            self.slot_bytes,
            "a slot is {} bytes",
            self.slot_bytes
        );
        // SAFETY: in bounds by the asserts above, and `bytes` cannot overlap the
        // pool because it is an ordinary borrowed slice and the pool's memory is
        // only ever reachable through this type's raw pointer.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.base.add(slot * self.slot_bytes),
                self.slot_bytes,
            );
        }
    }

    /// Where `slot` starts, for an IO worker that fills it directly (it reads
    /// into its own buffer first and copies here; see [`super::io`]).
    ///
    /// # Safety
    /// No GPU work that reads the slot may be queued or running while the
    /// pointer is being written through, and the pointer must not be used past
    /// the pool's lifetime.
    pub(crate) unsafe fn slot_ptr(&self, slot: usize) -> *mut u8 {
        assert!(slot < self.n_slots, "slot {slot} out of {}", self.n_slots);
        // SAFETY: in bounds by the assert.
        unsafe { self.base.add(slot * self.slot_bytes) }
    }

    /// The address of the pool's storage, to check that it never moves.
    #[cfg(test)]
    pub(crate) fn base_addr(&self) -> usize {
        self.base as usize
    }

    /// One slot read back from the live buffer (test only).
    #[cfg(test)]
    pub(crate) fn read_slot(&self, slot: usize) -> Vec<u8> {
        assert!(slot < self.n_slots);
        // SAFETY: in bounds; test code, no concurrent writer.
        unsafe {
            std::slice::from_raw_parts(self.base.add(slot * self.slot_bytes), self.slot_bytes)
                .to_vec()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::ops::indexing::take_axis;

    /// Deterministic filler, different per `seed`.
    fn pattern(len: usize, seed: u64) -> Vec<u8> {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 24) as u8
            })
            .collect()
    }

    /// The bytes of `a[rows]` after the GPU has gathered them.
    fn gathered_bytes(pool: &PoolTensor, rows: &[i32]) -> Vec<u8> {
        let idx = Array::from_slice(rows, &[rows.len() as i32]);
        let out = take_axis(pool.array(), &idx, 0).expect("take");
        out.eval().expect("eval");
        let view = out.view::<u8>().expect("byte view");
        view.eval().expect("eval view");
        view.try_as_slice::<u8>().expect("slice").to_vec()
    }

    #[test]
    #[ignore = "needs a Metal device"]
    fn a_new_pool_is_zeroed_contiguous_and_sized_as_asked() {
        let pool = PoolTensor::new(6, &[4, 32], Dtype::Uint32).expect("pool");
        assert_eq!(pool.n_slots(), 6);
        assert_eq!(pool.slot_bytes(), 4 * 32 * 4);
        assert_eq!(pool.array().shape(), &[6, 4, 32]);
        for slot in 0..6 {
            assert!(pool.read_slot(slot).iter().all(|&b| b == 0), "slot {slot}");
        }
    }

    /// (A) What the CPU writes is what the next GPU read sees, a rewrite
    /// replaces it rather than being served from a stale copy, and untouched
    /// slots are never disturbed.
    #[test]
    #[ignore = "needs a Metal device"]
    fn cpu_writes_are_seen_by_later_gpu_reads_and_rewrites_replace_them() {
        let pool = PoolTensor::new(8, &[1024], Dtype::Uint32).expect("pool");
        let n = pool.slot_bytes();
        let (a, b) = (pattern(n, 1), pattern(n, 2));
        assert_ne!(a, b);

        // SAFETY: nothing is running on the GPU.
        unsafe { pool.write_slot(3, &a) };
        let got = gathered_bytes(&pool, &[3, 5]);
        assert_eq!(
            &got[..n],
            a.as_slice(),
            "slot 3 must read back what was written"
        );
        assert!(got[n..].iter().all(|&x| x == 0), "slot 5 was never written");

        // Rewrite the same slot, after the GPU has already read the old bytes.
        // SAFETY: the previous gather was evaluated to completion above.
        unsafe { pool.write_slot(3, &b) };
        let got = gathered_bytes(&pool, &[3, 3, 0]);
        assert_eq!(
            &got[..n],
            b.as_slice(),
            "the rewrite must replace the old bytes"
        );
        assert_eq!(&got[n..2 * n], b.as_slice());
        assert!(
            got[2 * n..].iter().all(|&x| x == 0),
            "slot 0 was never written"
        );
    }

    /// One expert's bytes out of a stacked, contiguous tensor `[E, ...]`.
    fn expert_bytes(stacked: &Array, e: usize) -> Vec<u8> {
        let n_experts = stacked.shape()[0] as usize;
        let flat = stacked.view::<u8>().expect("byte view");
        flat.eval().expect("eval");
        let all = flat.try_as_slice::<u8>().expect("slice");
        let per = all.len() / n_experts;
        all[e * per..(e + 1) * per].to_vec()
    }

    /// A pool shaped like `stacked` but with `n_slots` slots, expert `e` of
    /// `stacked` copied into slot `slot_of(e)`.
    fn pool_like(stacked: &Array, n_slots: usize, slot_of: impl Fn(usize) -> usize) -> PoolTensor {
        let n_experts = stacked.shape()[0] as usize;
        let pool = PoolTensor::new(n_slots, &stacked.shape()[1..], stacked.dtype()).expect("pool");
        for e in 0..n_experts {
            // SAFETY: nothing is running on the GPU while the pool is filled.
            unsafe { pool.write_slot(slot_of(e), &expert_bytes(stacked, e)) };
        }
        pool
    }

    /// (C) The property the expert cache rests on: quantized expert bytes
    /// copied into scrambled slots give exactly the output the original stacked
    /// tensors give, for each quantization mode the checkpoints use.
    #[test]
    #[ignore = "needs a Metal device"]
    fn gather_qmm_over_copied_slots_matches_the_stacked_original_bit_for_bit() {
        use crate::native_quant::{
            MODE_AFFINE, MODE_MXFP4, gather_qmm_with_mode, quantize_with_mode,
        };
        use mlx_rs::random;

        const E: usize = 8;
        const OUT: i32 = 128;
        const IN: i32 = 256;
        for (mode, group_size, has_biases) in [(MODE_MXFP4, 32, false), (MODE_AFFINE, 64, true)] {
            // Random bf16 weights for E experts, quantized as a stack.
            let w = random::normal::<f32>(&[E as i32, OUT, IN], None, None, None)
                .expect("normal")
                .as_dtype(Dtype::Bfloat16)
                .expect("bf16");
            let (wq, sc, bi) = quantize_with_mode(&w, group_size, 4, mode).expect("quantize");
            assert_eq!(bi.is_some(), has_biases, "{mode:?} bias presence");

            // Experts 0..8 land in slots (3e+1) mod 8: a bijection that scrambles order.
            let slot_of = |e: usize| (3 * e + 1) % E;
            let p_w = pool_like(&wq, E, slot_of);
            let p_s = pool_like(&sc, E, slot_of);
            let p_b = bi.as_ref().map(|b| pool_like(b, E, slot_of));

            let x = random::normal::<f32>(&[1, 1, 1, 1, IN], None, None, None)
                .expect("normal")
                .as_dtype(Dtype::Bfloat16)
                .expect("bf16");
            let wanted: [u32; 2] = [5, 2];
            let by_expert = Array::from_slice(&wanted, &[1, 1, 2]);
            let by_slot =
                Array::from_slice(&wanted.map(|e| slot_of(e as usize) as u32), &[1, 1, 2]);

            let run = |w: &Array, s: &Array, b: Option<&Array>, idx: &Array| -> Vec<u8> {
                let y = gather_qmm_with_mode(
                    &x,
                    w,
                    s,
                    b,
                    None,
                    Some(idx),
                    true,
                    group_size,
                    4,
                    mode,
                    false,
                )
                .expect("gather_qmm");
                let bytes = y
                    .as_dtype(Dtype::Float32)
                    .expect("f32")
                    .view::<u8>()
                    .expect("view");
                bytes.eval().expect("eval");
                bytes.try_as_slice::<u8>().expect("slice").to_vec()
            };
            let reference = run(&wq, &sc, bi.as_ref(), &by_expert);
            let from_pool = run(
                p_w.array(),
                p_s.array(),
                p_b.as_ref().map(|b| b.array()),
                &by_slot,
            );
            assert_eq!(reference.len(), 2 * OUT as usize * 4);
            assert_eq!(
                from_pool, reference,
                "{mode:?}: gather over copied slots differs from the stacked original"
            );
            // The test would pass trivially if both outputs were zeros.
            assert!(
                reference.iter().any(|&b| b != 0),
                "{mode:?}: reference output is all zero"
            );

            // Now overwrite one slot with a different expert's bytes and check the
            // gather follows: asking for slot_of(5) must now give expert 6's result.
            // SAFETY: the gathers above were evaluated to completion.
            unsafe { p_w.write_slot(slot_of(5), &expert_bytes(&wq, 6)) };
            unsafe { p_s.write_slot(slot_of(5), &expert_bytes(&sc, 6)) };
            if let (Some(pb), Some(b)) = (p_b.as_ref(), bi.as_ref()) {
                unsafe { pb.write_slot(slot_of(5), &expert_bytes(b, 6)) };
            }
            let swapped = run(
                p_w.array(),
                p_s.array(),
                p_b.as_ref().map(|b| b.array()),
                &by_slot,
            );
            let want = run(
                &wq,
                &sc,
                bi.as_ref(),
                &Array::from_slice(&[6u32, 2], &[1, 1, 2]),
            );
            assert_eq!(
                swapped, want,
                "{mode:?}: a rewritten slot must act as the new expert"
            );
            assert_ne!(swapped, reference, "{mode:?}: the rewrite changed nothing");
        }
    }

    /// (B) Using the pool as an op input must not move or recycle its storage.
    #[test]
    #[ignore = "needs a Metal device"]
    fn using_the_pool_as_an_input_never_moves_its_storage() {
        let pool = PoolTensor::new(4, &[2048], Dtype::Uint32).expect("pool");
        let before = pool.base_addr();
        let n = pool.slot_bytes();
        for round in 0..20u64 {
            let bytes = pattern(n, 100 + round);
            // SAFETY: each round's gather is evaluated before the next write.
            unsafe { pool.write_slot((round % 4) as usize, &bytes) };
            let got = gathered_bytes(&pool, &[(round % 4) as i32]);
            assert_eq!(got, bytes, "round {round}");
            // Also feed it through an elementwise op, the kind MLX may run in place.
            let doubled = pool.array().add(pool.array()).expect("add");
            doubled.eval().expect("eval");
            assert_eq!(pool.array().strides(), &[2048, 1], "round {round}");
        }
        assert_eq!(pool.base_addr(), before, "the pool's buffer moved");
        // SAFETY: asks MLX where the evaluated array's data lives now.
        let live = unsafe { mlx_sys::mlx_array_data_uint8(pool.array().as_ptr()) } as usize;
        assert_eq!(
            live, before,
            "MLX's own view of the buffer disagrees with the cached pointer"
        );
    }
}
