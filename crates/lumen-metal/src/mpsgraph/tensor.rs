//! MTLBuffer ↔ MPSGraphTensorData bridging + shape helpers.

use objc2::AllocAnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSNumber};
use objc2_metal::MTLBuffer;
use objc2_metal_performance_shaders::MPSDataType;
use objc2_metal_performance_shaders_graph::MPSGraphTensorData;

/// Build an `MPSShape` (== `NSArray<NSNumber>`) from a `usize` slice.
pub fn shape_from_dims(dims: &[usize]) -> Retained<NSArray<NSNumber>> {
    let numbers: Vec<Retained<NSNumber>> =
        dims.iter().map(|&d| NSNumber::new_u64(d as u64)).collect();
    NSArray::from_retained_slice(&numbers)
}

/// Wrap a Metal buffer as an `MPSGraphTensorData` view.
///
/// The buffer is borrowed (objc retain), so `tensor_data` must outlive the
/// command buffer encoded with it. Callers control the buffer lifetime.
pub fn tensor_data_from_buffer(
    buffer: &ProtocolObject<dyn MTLBuffer>,
    shape: &NSArray<NSNumber>,
    dtype: MPSDataType,
) -> Retained<MPSGraphTensorData> {
    let alloc = MPSGraphTensorData::alloc();
    unsafe { MPSGraphTensorData::initWithMTLBuffer_shape_dataType(alloc, buffer, shape, dtype) }
}
