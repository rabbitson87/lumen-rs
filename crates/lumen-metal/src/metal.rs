//! Metal type shim — re-exports `candle_metal_kernels::metal` types with
//! aliases that match the names our codebase historically used (which came
//! from `metal v0.31`). This is the foundation of the ABI-M migration: by
//! routing through the same Metal wrappers Candle uses, every Candle
//! `Tensor` buffer is *directly* compatible with our kernels — no transmute,
//! no garbage-memory bridge round-trip.
//!
//! ## Naming map (metal v0.31 → candle-metal-kernels::metal)
//! | Old (metal v0.31)              | New (this module)            |
//! |--------------------------------|------------------------------|
//! | `Buffer`                       | `Buffer`                     |
//! | `Device`                       | `Device`                     |
//! | `CommandQueue`                 | `CommandQueue` (alias to `Retained<...>`) |
//! | `CommandBuffer`                | `CommandBuffer`              |
//! | `ComputePipelineState`         | `ComputePipeline`            |
//! | `ComputePipelineState` (alias) | `ComputePipelineState`       |
//! | `ComputeCommandEncoderRef`     | `ComputeCommandEncoder`      |
//! | `BlitCommandEncoderRef`        | `BlitCommandEncoder`         |
//! | `MTLSize`                      | `MTLSize` (objc2_metal)      |
//! | `MTLResourceOptions`           | `MTLResourceOptions`         |
//! | `MTLBlitOption`                | `MTLBlitOption`              |
//! | `MTLLanguageVersion`           | `MTLLanguageVersion`         |
//! | `CompileOptions`               | `CompileOptions` (re-exported from objc2_metal) |
//!
//! Old call patterns that still need surgical replacement at the call site:
//! - `enc.set_bytes(idx, len, ptr)` → `enc.set_bytes_directly(idx, len, ptr)`
//! - `cmd.new_compute_command_encoder()` (returns owned, drops auto-end)
//! - `Device::new_buffer(size, opts)` (different signature; use `Device::new_buffer_with_data` or owned wrapper)

// ── Direct re-exports from candle-metal-kernels ─────────────────────────
pub use candle_metal_kernels::metal::{
    BlitCommandEncoder, Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder,
    ComputePipeline, Device, IndirectCommandBuffer, Library, MTLResourceOptions,
};

// `ComputePipelineState` was the metal v0.31 name; cmk uses `ComputePipeline`.
// Keep both names available so a gradual migration can proceed without a
// flag-day rename.
pub use candle_metal_kernels::metal::ComputePipeline as ComputePipelineState;
pub use candle_metal_kernels::metal::ComputePipeline as ComputePipelineStateRef;

// metal v0.31 had `*Ref` borrowed-encoder types; in objc2 we just borrow
// the owned encoder. Provide aliases so `&ComputeCommandEncoderRef` call
// sites mechanically map to `&ComputeCommandEncoder`.
pub use candle_metal_kernels::metal::BlitCommandEncoder as BlitCommandEncoderRef;
pub use candle_metal_kernels::metal::CommandBuffer as CommandBufferRef;
pub use candle_metal_kernels::metal::CommandQueue as CommandQueueRef;
pub use candle_metal_kernels::metal::ComputeCommandEncoder as ComputeCommandEncoderRef;

// ── objc2 / objc2-metal pass-throughs ────────────────────────────────────
pub use objc2_metal::{
    MTLBlitOption, MTLCompileOptions as CompileOptions, MTLLanguageVersion, MTLResourceUsage,
    MTLSize,
};

// Foreign primitive used in older metal v0.31 APIs; our migrated code
// should prefer `usize` directly. Kept as a type alias for any straggling
// signatures.
pub type NSUInteger = usize;

// ── Extension traits — bridge metal v0.31 method ergonomics ─────────────

/// `metal v0.31`'s `ComputeCommandEncoderRef::set_buffer(idx, Some(buf), offset_u64)`
/// took a `u64` offset; cmk's `ComputeCommandEncoder::set_buffer` takes
/// `usize`. Add an extension method that accepts the older `u64` offset
/// shape so existing dispatch code compiles unchanged.
pub trait ComputeEncoderCompat {
    fn set_buffer_at(&self, index: usize, buffer: Option<&Buffer>, offset: u64);
    /// Backward-compat shim for callers that haven't migrated to the
    /// input/output split yet. Defaults to `set_input_buffer`. Most kernel
    /// bindings *read* their buffers, so input is the safe default; the
    /// minority that *write* should call `set_output_buffer` explicitly to
    /// get the auto-barrier hint.
    fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize);
}

// Blanket impl over anything that derefs to a `ComputeCommandEncoder`. Picks up
// `&ComputeCommandEncoder` itself, `WrappedEncoder<'_>` from cmk's
// `EncoderProvider`, and `CommandsGuard<'_>` from the batched command path —
// all of which need the same backward-compat shims.
impl<T: AsRef<ComputeCommandEncoder>> ComputeEncoderCompat for T {
    #[inline]
    fn set_buffer_at(&self, index: usize, buffer: Option<&Buffer>, offset: u64) {
        // Use the output-buffer entry point so the auto-barrier engine treats
        // every binding as a potential write target. This is the conservative
        // safe default for callers that haven't migrated to the
        // input/output split — see the trait doc for the rationale.
        self.as_ref()
            .set_output_buffer(index, buffer, offset as usize);
    }
    #[inline]
    fn set_buffer(&self, index: usize, buffer: Option<&Buffer>, offset: usize) {
        self.as_ref().set_output_buffer(index, buffer, offset);
    }
}

/// Extension trait — surfaces the rest of `ComputeCommandEncoder`'s inherent
/// API through `AsRef<ComputeCommandEncoder>` wrappers. cmk's `CommandsGuard`
/// has `AsRef<ComputeCommandEncoder>` but only re-exposes `set_label` and
/// `set_compute_pipeline_state` as inherent methods — everything else (bytes
/// upload, dispatch, threadgroup memory, ICB) lives behind `.as_ref()`. This
/// trait + blanket impl makes those callable directly on a `CommandsGuard`
/// without rewriting every call site.
pub trait BatchedEncoderExt {
    fn set_threadgroup_memory_length(&self, index: usize, length: usize);
    fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const std::ffi::c_void);
    fn dispatch_threads(&self, threads_per_grid: MTLSize, threads_per_threadgroup: MTLSize);
    fn dispatch_thread_groups(
        &self,
        threadgroups_per_grid: MTLSize,
        threads_per_threadgroup: MTLSize,
    );
    fn use_buffers_for_icb(&self, buffers: &[&Buffer], usage: MTLResourceUsage);
    fn execute_commands_in_buffer(
        &self,
        icb: &candle_metal_kernels::metal::IndirectCommandBuffer,
        count: usize,
    );
    fn execute_commands_in_buffer_range(
        &self,
        icb: &candle_metal_kernels::metal::IndirectCommandBuffer,
        start: usize,
        length: usize,
    );
    /// Generic `set_bytes` — typed wrapper that uploads a single struct by
    /// reference. Matches cmk's inherent `set_bytes<T>` shape, so call sites
    /// can write `enc.set_bytes(idx, &val)` regardless of whether `enc` is a
    /// `&ComputeCommandEncoder` (inherent wins) or a `&CommandsGuard` (trait
    /// resolves).
    fn set_bytes<U>(&self, index: usize, data: &U);
}

impl<T: AsRef<ComputeCommandEncoder>> BatchedEncoderExt for T {
    #[inline]
    fn set_threadgroup_memory_length(&self, index: usize, length: usize) {
        self.as_ref().set_threadgroup_memory_length(index, length);
    }
    #[inline]
    fn set_bytes_directly(&self, index: usize, length: usize, bytes: *const std::ffi::c_void) {
        self.as_ref().set_bytes_directly(index, length, bytes);
    }
    #[inline]
    fn dispatch_threads(&self, grid: MTLSize, tpt: MTLSize) {
        self.as_ref().dispatch_threads(grid, tpt);
    }
    #[inline]
    fn dispatch_thread_groups(&self, threadgroups: MTLSize, tpt: MTLSize) {
        self.as_ref().dispatch_thread_groups(threadgroups, tpt);
    }
    #[inline]
    fn use_buffers_for_icb(&self, buffers: &[&Buffer], usage: MTLResourceUsage) {
        self.as_ref().use_buffers_for_icb(buffers, usage);
    }
    #[inline]
    fn execute_commands_in_buffer(
        &self,
        icb: &candle_metal_kernels::metal::IndirectCommandBuffer,
        count: usize,
    ) {
        self.as_ref().execute_commands_in_buffer(icb, count);
    }
    #[inline]
    fn execute_commands_in_buffer_range(
        &self,
        icb: &candle_metal_kernels::metal::IndirectCommandBuffer,
        start: usize,
        length: usize,
    ) {
        self.as_ref()
            .execute_commands_in_buffer_range(icb, start, length);
    }
    #[inline]
    fn set_bytes<U>(&self, index: usize, data: &U) {
        self.as_ref().set_bytes(index, data);
    }
}

/// Same idea for `BlitCommandEncoder::copy_from_buffer` whose newer
/// signature uses `usize` for byte offsets/lengths.
pub trait BlitEncoderCompat {
    fn copy_from_buffer_u64(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        size: u64,
    );
}

impl BlitEncoderCompat for BlitCommandEncoder {
    #[inline]
    fn copy_from_buffer_u64(
        &mut self,
        src: &Buffer,
        src_offset: u64,
        dst: &Buffer,
        dst_offset: u64,
        size: u64,
    ) {
        self.copy_from_buffer(
            src,
            src_offset as usize,
            dst,
            dst_offset as usize,
            size as usize,
        );
    }
}

// ── API smoothers (replace metal v0.31 patterns) ─────────────────────────

/// `metal v0.31` had `MTLSize::new(w, h, d)` accepting `u64`; objc2-metal
/// exposes only the repr-C struct with `usize` fields. This macro accepts
/// any integer expression and casts to `usize`, which keeps every old
/// call site (`mtl_size!(n as u64, 1, 1)`, `mtl_size!(rows as u32, 1, 1)`,
/// `mtl_size!(1, 1, 1)`) compiling without surgical rewrites.
#[macro_export]
macro_rules! mtl_size {
    ($w:expr, $h:expr, $d:expr) => {
        $crate::metal::MTLSize {
            width: ($w) as usize,
            height: ($h) as usize,
            depth: ($d) as usize,
        }
    };
}

/// Auto-end wrapper around `ComputeCommandEncoder`. cmk's
/// `ComputeCommandEncoder` does **not** call `end_encoding()` on `Drop`, so a
/// `let enc = cmd.auto_compute_encoder();` followed by another
/// encoder open on the same command buffer triggers a Metal assert. This
/// wrapper restores the auto-end-on-drop semantics that the call sites
/// migrated from `metal v0.31` were relying on.
pub struct AutoEndCe {
    inner: Option<ComputeCommandEncoder>,
}

impl AutoEndCe {
    #[inline]
    pub fn new(inner: ComputeCommandEncoder) -> Self {
        Self { inner: Some(inner) }
    }
    /// Take ownership of the inner encoder, skipping the auto-end-on-drop.
    /// Useful when the caller wants to call `end_encoding()` explicitly.
    #[inline]
    pub fn into_inner(mut self) -> ComputeCommandEncoder {
        self.inner.take().expect("AutoEndCe inner already taken")
    }
    /// Explicitly end encoding. After this the wrapper is inert; `Drop` is a no-op.
    #[inline]
    pub fn end_encoding(mut self) {
        if let Some(e) = self.inner.take() {
            e.end_encoding();
        }
    }
}

impl Drop for AutoEndCe {
    fn drop(&mut self) {
        if let Some(e) = self.inner.take() {
            e.end_encoding();
        }
    }
}

impl AsRef<ComputeCommandEncoder> for AutoEndCe {
    fn as_ref(&self) -> &ComputeCommandEncoder {
        self.inner
            .as_ref()
            .expect("AutoEndCe inner already taken — use after end_encoding/into_inner")
    }
}

impl std::ops::Deref for AutoEndCe {
    type Target = ComputeCommandEncoder;
    fn deref(&self) -> &ComputeCommandEncoder {
        self.as_ref()
    }
}

/// Trait helper to open an auto-ending compute encoder. Call sites can write
/// `let enc = cmd.auto_compute_encoder();` and rely on `Drop` to insert the
/// matching `end_encoding()` even when the body errors out early.
pub trait CommandBufferExt {
    fn auto_compute_encoder(&self) -> AutoEndCe;
}

impl CommandBufferExt for CommandBuffer {
    #[inline]
    fn auto_compute_encoder(&self) -> AutoEndCe {
        AutoEndCe::new(self.compute_command_encoder_no_fence())
    }
}

/// cmk's `CommandBuffer::blit_command_encoder(fence, outputs)` takes two
/// ancillary args that the call sites we migrated from `metal v0.31` don't
/// thread through. This helper provides the same zero-arg ergonomics that
/// `compute_command_encoder_no_fence` offers — fresh fence + standalone
/// output map, sufficient for the small number of blit dispatches we issue.
#[inline]
pub fn blit_command_encoder_no_fence(cmd: &CommandBuffer) -> BlitCommandEncoder {
    use objc2_metal::MTLCommandBuffer as _;
    use std::sync::{Arc, Mutex};
    let device = Device::new(cmd.as_ref().device());
    let fence = Arc::new(candle_metal_kernels::metal::Fence::new(&device));
    let outputs = Arc::new(Mutex::new(std::collections::HashMap::new()));
    cmd.blit_command_encoder(&fence, &outputs)
}

/// Process-wide cmk `Commands` scheduler. All dispatch helpers should route
/// through this singleton so cmk's cross-encoder fence + `prev_ce_outputs`
/// global map can correctly order overlapping reads/writes between
/// dispatches sharing buffers.
///
/// Initialized lazily from the system default Metal device. On Apple Silicon
/// (single GPU) this is the same device every other path uses, so the
/// scheduler's view of dependencies is consistent.
pub fn process_commands() -> &'static candle_metal_kernels::metal::Commands {
    use std::sync::OnceLock;
    static C: OnceLock<candle_metal_kernels::metal::Commands> = OnceLock::new();
    C.get_or_init(|| {
        let device = Device::system_default().expect("Metal device available");
        let queue = device
            .new_command_queue()
            .expect("new_command_queue for process Commands");
        let residency = candle_metal_kernels::metal::ResidencySet::new(&device);
        candle_metal_kernels::metal::Commands::new(queue, &residency)
            .expect("Commands::new for process scheduler")
    })
}

/// Create a fresh command buffer from a cmk `CommandQueue` (which is a
/// `Retained<ProtocolObject<dyn MTLCommandQueue>>` rather than a struct
/// with `new_command_buffer()` like metal v0.31). Wraps the cmk
/// `CommandBuffer::new` constructor.
#[inline]
pub fn new_command_buffer(queue: &CommandQueue) -> CommandBuffer {
    use objc2_metal::MTLCommandQueue as _;
    let raw = queue
        .commandBuffer()
        .expect("MTLCommandQueue::commandBuffer returned nil");
    CommandBuffer::new(raw)
}

/// `metal v0.31` had `CompileOptions::new()` returning an owned options
/// struct with `set_language_version` / `set_fast_math_enabled` methods.
/// objc2-metal has the trait methods on `MTLCompileOptions` but the cmk
/// re-export is `Retained<MTLCompileOptions>`. Returns a wrapper that
/// exposes the same setter API as metal v0.31.
pub fn new_compile_options() -> CompileOptionsBuilder {
    CompileOptionsBuilder::new()
}

/// Setter-friendly wrapper around `Retained<MTLCompileOptions>` that
/// mirrors the metal v0.31 `CompileOptions` API surface our migration
/// touches (set_language_version, set_fast_math_enabled). Deref into the
/// underlying retained object so downstream `Device::new_library_with_source`
/// can accept it.
pub struct CompileOptionsBuilder {
    inner: objc2::rc::Retained<objc2_metal::MTLCompileOptions>,
}

impl CompileOptionsBuilder {
    pub fn new() -> Self {
        let inner = objc2_metal::MTLCompileOptions::new();
        Self { inner }
    }

    pub fn set_language_version(&self, version: MTLLanguageVersion) {
        self.inner.setLanguageVersion(version)
    }

    pub fn set_fast_math_enabled(&self, enabled: bool) {
        // objc2-metal exposes this as `mathMode` (modern) or
        // `setFastMathEnabled` (deprecated). Use the deprecated setter
        // for parity with our metal v0.31 call sites.
        #[allow(deprecated)]
        self.inner.setFastMathEnabled(enabled)
    }
}

impl AsRef<objc2_metal::MTLCompileOptions> for CompileOptionsBuilder {
    fn as_ref(&self) -> &objc2_metal::MTLCompileOptions {
        &self.inner
    }
}

impl std::ops::Deref for CompileOptionsBuilder {
    type Target = objc2_metal::MTLCompileOptions;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
