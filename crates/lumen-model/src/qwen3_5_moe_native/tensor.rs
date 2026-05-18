//! `NativeTensor`: contiguous Metal-Buffer-backed tensor with shape + dtype.
//!
//! Key differences vs `candle_core::Tensor`:
//!   - No autograd, no per-op command buffer, no implicit `wait_until_completed`.
//!   - Shape is row-major contiguous (no general strides). Slicing returns a
//!     view with an offset into the same buffer.
//!   - dtype is one of `{F32, BF16, U32, U8}`. The forward path keeps everything
//!     in F32 for parity with the Candle baseline; BF16 lands in a later phase
//!     once parity is established.
//!   - Buffer is ref-counted (`metal::Buffer` wraps an objc id), so cloning a
//!     `NativeTensor` is cheap and shares storage.
//!
//! The intent is for this type to be the *only* activation carrier on the hot
//! forward path, replacing every `Tensor` round-trip through Candle's metal
//! storage layer.

use anyhow::{anyhow, Result};
use lumen_metal::metal::Buffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeDType {
    F32,
    BF16,
    U32,
    U8,
}

impl NativeDType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::F32 | Self::U32 => 4,
            Self::BF16 => 2,
            Self::U8 => 1,
        }
    }
}

/// Row-major contiguous tensor view over a `metal::Buffer` slice.
///
/// `offset_bytes` is the start offset within `buffer`; the tensor occupies
/// `numel() * dtype.size_in_bytes()` bytes from that offset.
///
/// The optional `_anchor` keeps a Candle tensor alive when this `NativeTensor`
/// was adopted from one via `bridge::from_candle_tensor`. Candle's `metal`
/// crate version (`objc2-metal`) and ours (`metal-rs`) do not share an ABI for
/// the buffer wrapper struct, so the transmuted `Buffer` we hold here cannot
/// safely retain the underlying `MTLBuffer` on its own. By holding the source
/// Candle tensor for the lifetime of the `NativeTensor`, Candle's own storage
/// keeps the `MTLBuffer` ObjC object alive — the transmuted view stays valid
/// regardless of whether `Buffer::clone()` (de)increments refcount correctly.
#[derive(Clone)]
pub struct NativeTensor {
    buffer: Buffer,
    offset_bytes: usize,
    shape: Vec<usize>,
    dtype: NativeDType,
    _anchor: Option<candle_core::Tensor>,
}

impl NativeTensor {
    /// Wrap an existing buffer view. Caller guarantees the slice
    /// `[offset_bytes, offset_bytes + numel*dtype.bytes())` is valid.
    pub fn from_buffer(
        buffer: Buffer,
        offset_bytes: usize,
        shape: Vec<usize>,
        dtype: NativeDType,
    ) -> Result<Self> {
        Self::from_buffer_with_anchor(buffer, offset_bytes, shape, dtype, None)
    }

    /// Same as `from_buffer`, but optionally retains a Candle tensor for the
    /// lifetime of this view. Used by the zero-copy bridge.
    pub fn from_buffer_with_anchor(
        buffer: Buffer,
        offset_bytes: usize,
        shape: Vec<usize>,
        dtype: NativeDType,
        anchor: Option<candle_core::Tensor>,
    ) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let needed = numel * dtype.size_in_bytes();
        if offset_bytes + needed > buffer.length() as usize {
            return Err(anyhow!(
                "NativeTensor::from_buffer: slice [{offset_bytes}, {}) exceeds buffer length {}",
                offset_bytes + needed,
                buffer.length()
            ));
        }
        Ok(Self {
            buffer,
            offset_bytes,
            shape,
            dtype,
            _anchor: anchor,
        })
    }

    /// View accessor: same buffer, new shape and offset. No allocation. The
    /// anchor (if any) is propagated so the new view inherits the Candle
    /// storage's lifetime guarantee.
    pub fn slice(&self, offset_bytes: usize, shape: Vec<usize>) -> Result<Self> {
        Self::from_buffer_with_anchor(
            self.buffer.clone(),
            offset_bytes,
            shape,
            self.dtype,
            self._anchor.clone(),
        )
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn offset_bytes(&self) -> usize {
        self.offset_bytes
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn dtype(&self) -> NativeDType {
        self.dtype
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn byte_len(&self) -> usize {
        self.numel() * self.dtype.size_in_bytes()
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Reshape: same numel, new shape. Returns a view (no copy).
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(anyhow!(
                "reshape numel mismatch: {} vs {}",
                self.numel(),
                new_numel
            ));
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            offset_bytes: self.offset_bytes,
            shape: new_shape,
            dtype: self.dtype,
            _anchor: self._anchor.clone(),
        })
    }

    /// Narrow on the first axis: `new_shape[0] = len`, offset advances by
    /// `start * stride0 * dtype.bytes()`. Returns a view.
    pub fn narrow_axis0(&self, start: usize, len: usize) -> Result<Self> {
        if self.shape.is_empty() {
            return Err(anyhow!("narrow_axis0: scalar tensor"));
        }
        let dim0 = self.shape[0];
        if start + len > dim0 {
            return Err(anyhow!(
                "narrow_axis0: range [{start}, {}) exceeds axis-0 length {dim0}",
                start + len
            ));
        }
        let inner: usize = self.shape[1..].iter().product();
        let new_offset = self.offset_bytes + start * inner * self.dtype.size_in_bytes();
        let mut new_shape = self.shape.clone();
        new_shape[0] = len;
        Ok(Self {
            buffer: self.buffer.clone(),
            offset_bytes: new_offset,
            shape: new_shape,
            dtype: self.dtype,
            _anchor: self._anchor.clone(),
        })
    }

    /// Read F32 contents to host (copy). Used for parity checks; not on hot path.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype != NativeDType::F32 {
            return Err(anyhow!(
                "to_vec_f32 requires F32, got {:?}",
                self.dtype
            ));
        }
        let n = self.numel();
        let ptr = unsafe {
            (self.buffer.contents() as *const u8).add(self.offset_bytes)
                as *const f32
        };
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(slice.to_vec())
    }

    /// Read U32 contents to host (copy).
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        if self.dtype != NativeDType::U32 {
            return Err(anyhow!(
                "to_vec_u32 requires U32, got {:?}",
                self.dtype
            ));
        }
        let n = self.numel();
        let ptr = unsafe {
            (self.buffer.contents() as *const u8).add(self.offset_bytes)
                as *const u32
        };
        let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
        Ok(slice.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_metal::metal::{Device, MTLResourceOptions};

    fn make_buffer(bytes: usize) -> Option<Buffer> {
        Device::system_default()
            .and_then(|d| d.new_buffer(bytes, MTLResourceOptions::StorageModeShared).ok())
    }

    #[test]
    fn from_buffer_ok() {
        let Some(buf) = make_buffer(64) else { return };
        let t = NativeTensor::from_buffer(buf, 0, vec![4, 4], NativeDType::F32).unwrap();
        assert_eq!(t.numel(), 16);
        assert_eq!(t.byte_len(), 64);
        assert_eq!(t.rank(), 2);
    }

    #[test]
    fn from_buffer_overflow_rejected() {
        let Some(buf) = make_buffer(32) else { return };
        let result = NativeTensor::from_buffer(buf, 16, vec![8], NativeDType::F32);
        assert!(result.is_err());
    }

    #[test]
    fn reshape_preserves_view() {
        let Some(buf) = make_buffer(64) else { return };
        let t = NativeTensor::from_buffer(buf, 0, vec![4, 4], NativeDType::F32).unwrap();
        let r = t.reshape(vec![16]).unwrap();
        assert_eq!(r.numel(), 16);
        assert_eq!(r.shape(), &[16]);
    }

    #[test]
    fn reshape_numel_mismatch_rejected() {
        let Some(buf) = make_buffer(64) else { return };
        let t = NativeTensor::from_buffer(buf, 0, vec![4, 4], NativeDType::F32).unwrap();
        assert!(t.reshape(vec![5, 4]).is_err());
    }

    #[test]
    fn narrow_axis0_advances_offset() {
        let Some(buf) = make_buffer(128) else { return };
        let t = NativeTensor::from_buffer(buf, 0, vec![8, 4], NativeDType::F32).unwrap();
        let v = t.narrow_axis0(2, 3).unwrap();
        assert_eq!(v.shape(), &[3, 4]);
        // 2 rows × 4 cols × 4 bytes = 32
        assert_eq!(v.offset_bytes(), 32);
    }

    #[test]
    fn dtype_byte_sizes() {
        assert_eq!(NativeDType::F32.size_in_bytes(), 4);
        assert_eq!(NativeDType::BF16.size_in_bytes(), 2);
        assert_eq!(NativeDType::U32.size_in_bytes(), 4);
        assert_eq!(NativeDType::U8.size_in_bytes(), 1);
    }
}
