use crate::metal::{ComputePipelineState, Device, Library};
use anyhow::{Context, Result};
use std::collections::HashMap;

/// Compiled Metal shader library + cached compute pipelines.
///
/// Compiles the TurboQuant .metal source once, then caches each kernel
/// function's pipeline state by name.
pub struct ShaderPipelines {
    #[allow(dead_code)]
    library: Library,
    pipelines: HashMap<&'static str, ComputePipelineState>,
}

/// All kernel function names in the TurboQuant Metal shader.
pub const KERNEL_NAMES: &[&str] = &[
    "tq_rotate_and_normalize",
    "tq_lloyd_max_quantize",
    "tq_bitpack_and_residual",
    "tq_qjl_project_signs",
    "tq_compressed_attention_scores",
    "tq_compressed_value_gather",
    "tq_rotate_query",
    "tq_softmax",
    // Fused kernels
    "tq_rotate_normalize_quantize",
    "tq_scores_and_softmax",
    // V2 kernels: precomputed QJL query projection
    "tq_qjl_project_query",
    "tq_compressed_attention_scores_v2",
    "tq_scores_and_softmax_v2",
    // V3 kernels: V2 + float4-vectorized Stage 1
    "tq_compressed_attention_scores_v3",
    "tq_scores_and_softmax_v3",
    // V4 kernels: V3 + centroids cached in threadgroup memory
    "tq_compressed_attention_scores_v4",
    "tq_scores_and_softmax_v4",
    // V5 kernels: V4 + fp16 Stage 1 (half precision compute, f32 accumulation)
    "tq_compressed_attention_scores_v5",
    "tq_scores_and_softmax_v5",
    // V6 kernel: SIMD-group cooperative reduction over dim (32 threads / kv_idx)
    "tq_compressed_attention_scores_v6",
    // Parallel softmax (single threadgroup, arbitrary n)
    "tq_softmax_parallel",
    // GQA fan-out value gather (one compute, fan out to gqa_ratio Q heads)
    "tq_compressed_value_gather_multi",
];

impl ShaderPipelines {
    /// Compile the embedded Metal source and build pipeline states for all kernels.
    pub fn new(device: &Device) -> Result<Self> {
        let source = include_str!("shaders/turboquant.metal");

        let options = crate::metal::new_compile_options();
        options.set_language_version(crate::metal::MTLLanguageVersion::Version3_0);
        options.set_fast_math_enabled(true);

        let library = device
            .new_library_with_source(source, Some(options.as_ref()))
            .map_err(|e| anyhow::anyhow!("Metal shader compile error: {e}"))?;

        let mut pipelines = HashMap::new();
        for &name in KERNEL_NAMES {
            let func = library
                .get_function(name, None)
                .map_err(|e| anyhow::anyhow!("kernel '{name}' not found: {e}"))?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| anyhow::anyhow!("pipeline '{name}' failed: {e}"))?;
            pipelines.insert(name, pipeline);
        }

        eprintln!("TurboQuant Metal: compiled {} kernels", pipelines.len());
        Ok(Self { library, pipelines })
    }

    /// Get a cached pipeline state by kernel name.
    pub fn get(&self, name: &str) -> Result<&ComputePipelineState> {
        self.pipelines
            .get(name)
            .with_context(|| format!("unknown kernel: {name}"))
    }

    /// Max threads per threadgroup for a given kernel.
    pub fn max_threads(&self, name: &str) -> Result<u64> {
        let pipeline = self.get(name)?;
        Ok(pipeline.max_total_threads_per_threadgroup() as u64)
    }
}
