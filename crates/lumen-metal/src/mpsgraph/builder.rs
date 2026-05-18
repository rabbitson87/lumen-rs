//! Graph builder templates.
//!
//! Each builder returns the input `MPSGraphTensor` placeholders, a target
//! tensor (output), and a feed dictionary so the caller can drive
//! `MPSGraphExecutable::compile`.

use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSDictionary, NSNumber};
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphShapedType, MPSGraphTensor, MPSGraphTensorNamedDataLayout,
};

use super::tensor::shape_from_dims;

/// 2D matrix multiplication template: `A [m × k] · B [k × n] → C [m × n]`.
pub struct MatmulGraph2D {
    pub graph: Retained<MPSGraph>,
    pub a: Retained<MPSGraphTensor>,
    pub b: Retained<MPSGraphTensor>,
    pub c: Retained<MPSGraphTensor>,
    /// Feed dictionary keyed by placeholders, valued with shaped types.
    /// Pass directly to `compile`.
    pub feeds: Retained<NSDictionary<MPSGraphTensor, MPSGraphShapedType>>,
}

impl MatmulGraph2D {
    pub fn build(graph: Retained<MPSGraph>, m: usize, k: usize, n: usize) -> Self {
        let a_shape = shape_from_dims(&[m, k]);
        let b_shape = shape_from_dims(&[k, n]);

        let a = unsafe {
            graph.placeholderWithShape_dataType_name(
                Some(&a_shape),
                MPSDataType::Float32,
                None,
            )
        };
        let b = unsafe {
            graph.placeholderWithShape_dataType_name(
                Some(&b_shape),
                MPSDataType::Float32,
                None,
            )
        };
        let c = unsafe {
            graph.matrixMultiplicationWithPrimaryTensor_secondaryTensor_name(&a, &b, None)
        };

        let a_shaped = shaped_type(&a_shape, MPSDataType::Float32);
        let b_shaped = shaped_type(&b_shape, MPSDataType::Float32);
        let feeds = NSDictionary::from_slices(&[&*a, &*b], &[&*a_shaped, &*b_shaped]);

        Self { graph, a, b, c, feeds }
    }
}

fn shaped_type(
    shape: &NSArray<NSNumber>,
    dtype: MPSDataType,
) -> Retained<MPSGraphShapedType> {
    use objc2::AllocAnyThread;
    let alloc = MPSGraphShapedType::alloc();
    unsafe { MPSGraphShapedType::initWithShape_dataType(alloc, Some(shape), dtype) }
}

// `MPSGraphTensorNamedDataLayout` is only re-exported here to silence the
// "unused import" lint when builders for layout-aware ops are added later.
#[allow(dead_code)]
fn _layout_witness() -> MPSGraphTensorNamedDataLayout {
    MPSGraphTensorNamedDataLayout::CHW
}

/// RmsNorm template: `y = x · rsqrt(mean(x², axis=-1) + eps) · weight`.
///
/// MPSGraph has no fused RmsNorm op; we decompose into 6 nodes (square →
/// mean → add → rsqrt → mul → mul). Apple's compiler stitches them into a
/// single Metal shader at `compile()` time, so dispatch overhead = 1.
///
/// - `x`: `[m, hidden]` placeholder
/// - `weight`: `[hidden]` placeholder (per-feature gain)
/// - `y`: `[m, hidden]` output
pub struct RmsNormGraph {
    pub graph: Retained<MPSGraph>,
    pub x: Retained<MPSGraphTensor>,
    pub weight: Retained<MPSGraphTensor>,
    pub y: Retained<MPSGraphTensor>,
    pub feeds: Retained<NSDictionary<MPSGraphTensor, MPSGraphShapedType>>,
}

impl RmsNormGraph {
    pub fn build(graph: Retained<MPSGraph>, m: usize, hidden: usize, eps: f32) -> Self {
        let x_shape = shape_from_dims(&[m, hidden]);
        let w_shape = shape_from_dims(&[hidden]);

        let x = unsafe {
            graph.placeholderWithShape_dataType_name(
                Some(&x_shape),
                MPSDataType::Float32,
                None,
            )
        };
        let weight = unsafe {
            graph.placeholderWithShape_dataType_name(
                Some(&w_shape),
                MPSDataType::Float32,
                None,
            )
        };

        // axes = [1] → reduce along hidden (last dim of [m, hidden])
        let axes_arr: Vec<Retained<NSNumber>> = vec![NSNumber::new_i64(1)];
        let axes = NSArray::from_retained_slice(&axes_arr);

        let x_sq = unsafe { graph.squareWithTensor_name(&x, None) };
        let var = unsafe { graph.meanOfTensor_axes_name(&x_sq, &axes, None) };
        let eps_t = unsafe {
            graph.constantWithScalar_dataType(eps as f64, MPSDataType::Float32)
        };
        let var_eps = unsafe {
            graph.additionWithPrimaryTensor_secondaryTensor_name(&var, &eps_t, None)
        };
        let inv = unsafe { graph.reciprocalSquareRootWithTensor_name(&var_eps, None) };
        // x · inv  (broadcast inv [m,1] onto x [m,hidden])
        let x_norm = unsafe {
            graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&x, &inv, None)
        };
        // · weight  (broadcast weight [hidden] onto [m,hidden])
        let y = unsafe {
            graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&x_norm, &weight, None)
        };

        let x_shaped = shaped_type(&x_shape, MPSDataType::Float32);
        let w_shaped = shaped_type(&w_shape, MPSDataType::Float32);
        let feeds = NSDictionary::from_slices(&[&*x, &*weight], &[&*x_shaped, &*w_shaped]);

        Self { graph, x, weight, y, feeds }
    }
}

/// RmsNorm template with bf16 output.
///
/// Identical compute to [`RmsNormGraph`] (square → mean → add → rsqrt → mul →
/// mul); the final node casts the f32 result to bf16 inside the same fused
/// shader. Reduction stays in f32 — only the store is narrowed.
///
/// Lever B Stage 2 staging vehicle: pairs with a bf16-accepting downstream
/// matmul to halve the activation read bandwidth at the qkv_proj boundary
/// without the Phase A.1.5 cast-back trap.
pub struct RmsNormBf16OutGraph {
    pub graph: Retained<MPSGraph>,
    pub x: Retained<MPSGraphTensor>,
    pub weight: Retained<MPSGraphTensor>,
    pub y: Retained<MPSGraphTensor>,
    pub feeds: Retained<NSDictionary<MPSGraphTensor, MPSGraphShapedType>>,
}

impl RmsNormBf16OutGraph {
    pub fn build(graph: Retained<MPSGraph>, m: usize, hidden: usize, eps: f32) -> Self {
        let x_shape = shape_from_dims(&[m, hidden]);
        let w_shape = shape_from_dims(&[hidden]);

        let x = unsafe {
            graph.placeholderWithShape_dataType_name(
                Some(&x_shape),
                MPSDataType::Float32,
                None,
            )
        };
        let weight = unsafe {
            graph.placeholderWithShape_dataType_name(
                Some(&w_shape),
                MPSDataType::Float32,
                None,
            )
        };

        let axes_arr: Vec<Retained<NSNumber>> = vec![NSNumber::new_i64(1)];
        let axes = NSArray::from_retained_slice(&axes_arr);

        let x_sq = unsafe { graph.squareWithTensor_name(&x, None) };
        let var = unsafe { graph.meanOfTensor_axes_name(&x_sq, &axes, None) };
        let eps_t = unsafe {
            graph.constantWithScalar_dataType(eps as f64, MPSDataType::Float32)
        };
        let var_eps = unsafe {
            graph.additionWithPrimaryTensor_secondaryTensor_name(&var, &eps_t, None)
        };
        let inv = unsafe { graph.reciprocalSquareRootWithTensor_name(&var_eps, None) };
        let x_norm = unsafe {
            graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&x, &inv, None)
        };
        let y_f32 = unsafe {
            graph.multiplicationWithPrimaryTensor_secondaryTensor_name(&x_norm, &weight, None)
        };
        let y = unsafe {
            graph.castTensor_toType_name(&y_f32, MPSDataType::BFloat16, None)
        };

        let x_shaped = shaped_type(&x_shape, MPSDataType::Float32);
        let w_shaped = shaped_type(&w_shape, MPSDataType::Float32);
        let feeds = NSDictionary::from_slices(&[&*x, &*weight], &[&*x_shaped, &*w_shaped]);

        Self { graph, x, weight, y, feeds }
    }
}
