//! Native state caches for the GatedDeltaNet linear-attention block (Phase A.8-C).
//!
//! Two persistent buffers per layer carry SSM and conv state across decode
//! calls without bridging through Candle:
//!
//!   - [`NativeSsmState`]: `[B, num_v_heads, head_dim_v, head_dim_k]` —
//!     the running outer product accumulator updated by `ssm_step`. The
//!     production `GatedDeltaNet` keeps this as `Option<Tensor>`; we
//!     mirror that lifecycle here but stay resident as a Metal buffer.
//!
//!   - [`NativeConvState`]: `[B, conv_kernel - 1, qkv_dim]` — the last
//!     `kernel-1` rows of the qkv tensor needed by the depthwise causal
//!     conv1d to compose its current window. Smaller than the SSM state
//!     (4480 × 3 × 4 = ~54 KB for Qwen3.5) but lives on the same
//!     allocation discipline.
//!
//! Both states reset to zeros at construction and on `reset()`. The Metal
//! buffer is retained across resets so subsequent generations reuse it.

use anyhow::{Result, anyhow};

use super::context::NativeContext;
use super::tensor::{NativeDType, NativeTensor};

/// Persistent SSM state buffer for a single GatedDeltaNet layer.
///
/// Shape: `[B, num_v_heads, head_dim_v, head_dim_k]` F32, contiguous.
/// `head_dim_v == head_dim_k == head_dim` for Qwen3.5 — kept separate in the
/// signature so future variants with rectangular SSM kernels stay supported.
pub struct NativeSsmState {
    state: NativeTensor,
    populated: bool,
    b: usize,
    num_v_heads: usize,
    head_dim_v: usize,
    head_dim_k: usize,
}

impl NativeSsmState {
    pub fn new(
        ctx: &NativeContext,
        b: usize,
        num_v_heads: usize,
        head_dim_v: usize,
        head_dim_k: usize,
    ) -> Result<Self> {
        if b == 0 || num_v_heads == 0 || head_dim_v == 0 || head_dim_k == 0 {
            return Err(anyhow!(
                "NativeSsmState::new: dims must be > 0 (got b={b}, hv={num_v_heads}, dv={head_dim_v}, dk={head_dim_k})"
            ));
        }
        let state = ctx.zeros(
            vec![b, num_v_heads, head_dim_v, head_dim_k],
            NativeDType::F32,
        )?;
        Ok(Self {
            state,
            populated: false,
            b,
            num_v_heads,
            head_dim_v,
            head_dim_k,
        })
    }

    /// Drop the running state. Subsequent forward calls will see fresh zeros.
    /// The buffer itself stays allocated so the next prefill writes into it
    /// without re-allocating.
    pub fn reset(&mut self, ctx: &NativeContext) -> Result<()> {
        // Zero the buffer in place so the SSM step kernel reads zero state on
        // the very first iteration (matches `state := zeros` at the top of
        // `forward_ssm_loop`).
        let bytes = self.state.byte_len();
        unsafe {
            std::ptr::write_bytes(self.state.buffer().contents() as *mut u8, 0, bytes);
        }
        let _ = ctx;
        self.populated = false;
        Ok(())
    }

    /// True once at least one `ssm_step` has been dispatched against the
    /// state buffer. Used by the dispatcher to decide whether to seed the
    /// state with zeros (first call) or trust the existing contents.
    pub fn is_populated(&self) -> bool {
        self.populated
    }

    pub fn mark_populated(&mut self) {
        self.populated = true;
    }

    pub fn buf(&self) -> &NativeTensor {
        &self.state
    }

    pub fn batch(&self) -> usize {
        self.b
    }

    pub fn num_v_heads(&self) -> usize {
        self.num_v_heads
    }

    pub fn head_dim_v(&self) -> usize {
        self.head_dim_v
    }

    pub fn head_dim_k(&self) -> usize {
        self.head_dim_k
    }

    /// Capture the current SSM buffer contents plus `populated` flag for
    /// later [`restore`]. Used by speculative decoding to roll back state
    /// after a verify-batch forward.
    pub fn snapshot(&self) -> Result<NativeSsmSnapshot> {
        let n = self.state.numel();
        let ptr = self.state.buffer().contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(NativeSsmSnapshot {
            data: slice.to_vec(),
            populated: self.populated,
        })
    }

    /// Restore a previously captured [`NativeSsmSnapshot`]. Snapshot must
    /// match the state's shape (numel) — typically the same instance that
    /// produced it.
    pub fn restore(&mut self, snap: &NativeSsmSnapshot) -> Result<()> {
        let n = self.state.numel();
        if snap.data.len() != n {
            return Err(anyhow!(
                "NativeSsmState::restore numel mismatch: snap={} state={}",
                snap.data.len(),
                n
            ));
        }
        let dst = self.state.buffer().contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(snap.data.as_ptr(), dst, n);
        }
        self.populated = snap.populated;
        Ok(())
    }
}

/// Captured contents of a [`NativeSsmState`].
#[derive(Clone)]
pub struct NativeSsmSnapshot {
    data: Vec<f32>,
    populated: bool,
}

impl NativeSsmSnapshot {
    pub fn populated(&self) -> bool {
        self.populated
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

/// Persistent conv1d state — last `conv_kernel - 1` rows of the qkv tensor.
///
/// Shape: `[B, conv_kernel - 1, qkv_dim]` F32, contiguous. Entries past the
/// active window remain zero from the initial allocation; conv1d treats them
/// as the "before-prompt" context.
pub struct NativeConvState {
    state: NativeTensor,
    /// Number of valid rows currently held. Starts at 0; capped at
    /// `conv_kernel - 1`.
    valid_rows: usize,
    b: usize,
    conv_pad: usize,
    qkv_dim: usize,
}

impl NativeConvState {
    pub fn new(ctx: &NativeContext, b: usize, conv_kernel: usize, qkv_dim: usize) -> Result<Self> {
        if b == 0 || conv_kernel < 2 || qkv_dim == 0 {
            return Err(anyhow!(
                "NativeConvState::new: invalid dims (b={b}, kernel={conv_kernel}, qkv_dim={qkv_dim})"
            ));
        }
        let conv_pad = conv_kernel - 1;
        let state = ctx.zeros(vec![b, conv_pad, qkv_dim], NativeDType::F32)?;
        Ok(Self {
            state,
            valid_rows: 0,
            b,
            conv_pad,
            qkv_dim,
        })
    }

    pub fn reset(&mut self, _ctx: &NativeContext) -> Result<()> {
        let bytes = self.state.byte_len();
        unsafe {
            std::ptr::write_bytes(self.state.buffer().contents() as *mut u8, 0, bytes);
        }
        self.valid_rows = 0;
        Ok(())
    }

    pub fn buf(&self) -> &NativeTensor {
        &self.state
    }

    pub fn valid_rows(&self) -> usize {
        self.valid_rows
    }

    /// Mark `n` newly-pushed rows as valid (capped at `conv_pad`). Called by
    /// the conv1d update kernel after writing the new tail of the window.
    pub fn set_valid_rows(&mut self, n: usize) {
        self.valid_rows = n.min(self.conv_pad);
    }

    pub fn batch(&self) -> usize {
        self.b
    }

    pub fn conv_pad(&self) -> usize {
        self.conv_pad
    }

    pub fn qkv_dim(&self) -> usize {
        self.qkv_dim
    }

    /// Capture conv1d window contents + `valid_rows` for [`restore`].
    pub fn snapshot(&self) -> Result<NativeConvSnapshot> {
        let n = self.state.numel();
        let ptr = self.state.buffer().contents() as *const f32;
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(NativeConvSnapshot {
            data: slice.to_vec(),
            valid_rows: self.valid_rows,
        })
    }

    pub fn restore(&mut self, snap: &NativeConvSnapshot) -> Result<()> {
        let n = self.state.numel();
        if snap.data.len() != n {
            return Err(anyhow!(
                "NativeConvState::restore numel mismatch: snap={} state={}",
                snap.data.len(),
                n
            ));
        }
        let dst = self.state.buffer().contents() as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(snap.data.as_ptr(), dst, n);
        }
        self.valid_rows = snap.valid_rows;
        Ok(())
    }
}

/// Captured contents of a [`NativeConvState`].
#[derive(Clone)]
pub struct NativeConvSnapshot {
    data: Vec<f32>,
    valid_rows: usize,
}

impl NativeConvSnapshot {
    pub fn valid_rows(&self) -> usize {
        self.valid_rows
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ssm_state_alloc_and_reset() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut s = NativeSsmState::new(&ctx, 1, 4, 8, 8).unwrap();
        assert_eq!(s.batch(), 1);
        assert_eq!(s.num_v_heads(), 4);
        assert_eq!(s.head_dim_v(), 8);
        assert_eq!(s.head_dim_k(), 8);
        assert!(!s.is_populated());

        let v = s.buf().to_vec_f32().unwrap();
        assert_eq!(v.len(), 1 * 4 * 8 * 8);
        for x in &v {
            assert_eq!(*x, 0.0);
        }

        s.mark_populated();
        assert!(s.is_populated());
        s.reset(&ctx).unwrap();
        assert!(!s.is_populated());
    }

    #[test]
    fn conv_state_alloc_and_reset() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut s = NativeConvState::new(&ctx, 1, 4, 64).unwrap();
        assert_eq!(s.conv_pad(), 3);
        assert_eq!(s.qkv_dim(), 64);
        assert_eq!(s.valid_rows(), 0);

        let v = s.buf().to_vec_f32().unwrap();
        assert_eq!(v.len(), 1 * 3 * 64);
        for x in &v {
            assert_eq!(*x, 0.0);
        }

        s.set_valid_rows(2);
        assert_eq!(s.valid_rows(), 2);
        s.set_valid_rows(10); // cap at conv_pad=3
        assert_eq!(s.valid_rows(), 3);

        s.reset(&ctx).unwrap();
        assert_eq!(s.valid_rows(), 0);
    }

    #[test]
    fn ssm_state_zero_dim_rejected() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        assert!(NativeSsmState::new(&ctx, 0, 4, 8, 8).is_err());
        assert!(NativeSsmState::new(&ctx, 1, 0, 8, 8).is_err());
    }

    #[test]
    fn ssm_state_snapshot_restore_roundtrip() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut s = NativeSsmState::new(&ctx, 1, 4, 8, 8).unwrap();
        // Write a known pattern into the buffer.
        let n = s.buf().numel();
        let dst = s.buf().buffer().contents() as *mut f32;
        for i in 0..n {
            unsafe {
                *dst.add(i) = i as f32 * 0.5;
            }
        }
        s.mark_populated();

        let snap = s.snapshot().unwrap();
        assert_eq!(snap.numel(), n);
        assert!(snap.populated());

        // Mutate and reset, then restore.
        s.reset(&ctx).unwrap();
        assert!(!s.is_populated());
        let v0 = s.buf().to_vec_f32().unwrap();
        for x in &v0 {
            assert_eq!(*x, 0.0);
        }

        s.restore(&snap).unwrap();
        assert!(s.is_populated());
        let v1 = s.buf().to_vec_f32().unwrap();
        for i in 0..n {
            assert_eq!(v1[i], i as f32 * 0.5);
        }
    }

    #[test]
    fn conv_state_snapshot_restore_roundtrip() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let mut s = NativeConvState::new(&ctx, 1, 4, 64).unwrap();
        let n = s.buf().numel();
        let dst = s.buf().buffer().contents() as *mut f32;
        for i in 0..n {
            unsafe {
                *dst.add(i) = (i as f32) - 100.0;
            }
        }
        s.set_valid_rows(2);

        let snap = s.snapshot().unwrap();
        assert_eq!(snap.valid_rows(), 2);
        assert_eq!(snap.numel(), n);

        s.reset(&ctx).unwrap();
        assert_eq!(s.valid_rows(), 0);

        s.restore(&snap).unwrap();
        assert_eq!(s.valid_rows(), 2);
        let v = s.buf().to_vec_f32().unwrap();
        for i in 0..n {
            assert_eq!(v[i], (i as f32) - 100.0);
        }
    }

    #[test]
    fn conv_state_invalid_kernel_rejected() {
        let ctx = match NativeContext::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        // kernel=1 → conv_pad=0 → reject (no valid window state)
        assert!(NativeConvState::new(&ctx, 1, 1, 64).is_err());
        assert!(NativeConvState::new(&ctx, 0, 4, 64).is_err());
    }
}
