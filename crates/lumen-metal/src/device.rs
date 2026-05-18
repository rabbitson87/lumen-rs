use anyhow::{anyhow, Result};
use candle_metal_kernels::metal::{Commands, ResidencySet};
use crate::metal::{CommandQueue, Device, MTLResourceOptions};

/// Wrapper around Metal device and command queue.
///
/// Provides buffer allocation and command submission for TurboQuant kernels.
///
/// The `commands` field is cmk's scheduler. All dispatch helpers should route
/// through it (via `commands.command_encoder()?`) instead of opening their own
/// raw `compute_command_encoder_no_fence()` — that's the only way cmk's
/// cross-encoder fence + global output map can correctly order overlapping
/// reads/writes between dispatches sharing buffers. The legacy
/// "one-encoder-per-dispatch + own command buffer" pattern leaks dependencies
/// between encoders and causes numerical drift on long-form generation.
pub struct MetalContext {
    pub device: Device,
    pub queue: CommandQueue,
    pub commands: Commands,
    /// Residency set kept alive so cmk's command-buffer factory can reuse it.
    /// Buffers allocated through `buffer_with_data` / `buffer_zeroed` are not
    /// auto-inserted into the residency set; callers needing residency
    /// guarantees should `insert` explicitly via the public field.
    pub residency_set: ResidencySet,
}

impl MetalContext {
    /// Create a new Metal context using the system default GPU.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or_else(|| anyhow!("no Metal GPU"))?;
        let queue = device
            .new_command_queue()
            .map_err(|e| anyhow!("new_command_queue: {e}"))?;
        eprintln!(
            "TurboQuant Metal: {} ({}MB)",
            device.architecture_name(),
            device.recommended_max_working_set_size() / (1024 * 1024)
        );
        let residency_set = ResidencySet::new(&device);
        let commands = Commands::new(queue.clone(), &residency_set)
            .map_err(|e| anyhow!("Commands::new: {e:?}"))?;
        Ok(Self {
            device,
            queue,
            commands,
            residency_set,
        })
    }

    /// Allocate a GPU buffer initialized with data.
    pub fn buffer_with_data<T: Copy>(&self, data: &[T]) -> crate::metal::Buffer {
        let byte_len = std::mem::size_of_val(data);
        self.device
            .new_buffer_with_data(
                data.as_ptr() as *const _,
                byte_len,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("Metal buffer alloc failed")
    }

    /// Allocate a zero-initialized GPU buffer of `byte_len` bytes.
    pub fn buffer_zeroed(&self, byte_len: u64) -> crate::metal::Buffer {
        self.device
            .new_buffer(byte_len as usize, MTLResourceOptions::StorageModeShared)
            .expect("Metal buffer alloc failed")
    }

    /// Allocate a GPU buffer for `count` elements of type T (zeroed).
    pub fn buffer_for<T>(&self, count: usize) -> crate::metal::Buffer {
        let byte_len = (count * std::mem::size_of::<T>()) as u64;
        self.buffer_zeroed(byte_len)
    }

    /// Read buffer contents back to CPU.
    pub fn read_buffer<T: Copy>(&self, buffer: &crate::metal::Buffer, count: usize) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.to_vec()
    }

    /// Check if device supports non-uniform threadgroups (Apple Silicon always does).
    pub fn supports_non_uniform_threadgroups(&self) -> bool {
        // Apple Silicon (Apple GPU family 7+) always supports this.
        true
    }

    /// construct an `IndirectCommandBuffer` from this
    /// context's device without exposing the underlying objc2 types to
    /// callers (lumen-model and other crates that don't depend on
    /// objc2 directly).
    pub fn new_indirect_command_buffer(
        &self,
        max_commands: usize,
        max_kernel_buffer_bind_count: usize,
    ) -> Result<crate::metal::IndirectCommandBuffer> {
        use objc2::rc::Retained;
        use objc2::runtime::ProtocolObject;
        use objc2_metal::MTLDevice;
        let raw: Retained<ProtocolObject<dyn MTLDevice>> =
            Retained::from(self.device.as_ref());
        crate::metal::IndirectCommandBuffer::new(
            &raw,
            max_commands,
            max_kernel_buffer_bind_count,
        )
        .map_err(|e| anyhow!("ICB alloc: {e:?}"))
    }
}

/// `MTLResourceUsage::Read | Write` constant for ICB
/// resource declarations, exposed without forcing callers to depend on
/// objc2_metal directly.
pub fn icb_read_write_usage() -> objc2_metal::MTLResourceUsage {
    objc2_metal::MTLResourceUsage(
        objc2_metal::MTLResourceUsage::Read.0 | objc2_metal::MTLResourceUsage::Write.0,
    )
}
