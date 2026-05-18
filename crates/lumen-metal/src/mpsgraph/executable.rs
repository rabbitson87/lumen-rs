//! Compile + run helpers around `MPSGraphExecutable`.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSArray, NSDictionary};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use objc2_metal_performance_shaders::MPSCommandBuffer;
use objc2_metal_performance_shaders_graph::{
    MPSGraph, MPSGraphDevice, MPSGraphExecutable, MPSGraphShapedType, MPSGraphTensor,
    MPSGraphTensorData,
};

/// Compile a graph against a device + feed dictionary.
///
/// bit-identical with this default
/// (`compilationDescriptor=None` → Apple's `Level1` + `reducedPrecisionFastMath=None`).
/// We tried explicitly forcing `Level0 + None` (Option 1, see
/// `mpsgraph_phase3_strict_math_negative.md`): drift in 80-callsite Phase 3.2
/// stayed unchanged, AND `Level0` actually broke Phase 3.1 bit-identity
/// (warmup tok=82254 vs 9419 / 0 baseline) — `Level1`'s additional passes
/// are what produced the parity in the first place. Default reverted.
pub fn compile(
    graph: &MPSGraph,
    device: &MPSGraphDevice,
    feeds: &NSDictionary<MPSGraphTensor, MPSGraphShapedType>,
    targets: &[&MPSGraphTensor],
) -> Retained<MPSGraphExecutable> {
    let targets_arr = NSArray::from_slice(targets);
    unsafe {
        graph.compileWithDevice_feeds_targetTensors_targetOperations_compilationDescriptor(
            Some(device),
            feeds,
            &targets_arr,
            None,
            None,
        )
    }
}

/// Synchronously run an executable on the given command queue.
///
/// `inputs` must match the placeholder order used at compile time.
/// `outputs` must match the target tensor order.
pub fn run_sync(
    exe: &MPSGraphExecutable,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    inputs: &[&MPSGraphTensorData],
    outputs: &[&MPSGraphTensorData],
) -> Retained<NSArray<MPSGraphTensorData>> {
    let inputs_arr = NSArray::from_slice(inputs);
    let outputs_arr = NSArray::from_slice(outputs);
    unsafe {
        exe.runWithMTLCommandQueue_inputsArray_resultsArray_executionDescriptor(
            queue,
            &inputs_arr,
            Some(&outputs_arr),
            None,
        )
    }
}

/// Encode an executable into a fresh command buffer on the given queue
/// and `commit()` (no wait). The caller can then either rely on the queue's
/// in-order execution (Candle's downstream ops will sync naturally) or
/// `wait_until_completed()` if the result is needed immediately on CPU.
///
/// This is the "Phase 3 best path": when `queue` is Candle's own command
/// queue, this avoids cross-queue races without the `device.synchronize()`
/// overhead that the Phase 2 PoC needed.
pub fn encode_and_commit(
    exe: &MPSGraphExecutable,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    inputs: &[&MPSGraphTensorData],
    outputs: &[&MPSGraphTensorData],
) -> Retained<ProtocolObject<dyn MTLCommandBuffer>> {
    let inputs_arr = NSArray::from_slice(inputs);
    let outputs_arr = NSArray::from_slice(outputs);
    let mtl_buf = queue
        .commandBuffer()
        .expect("MTLCommandQueue::commandBuffer returned nil");
    let mps_buf = unsafe { MPSCommandBuffer::commandBufferWithCommandBuffer(&mtl_buf) };
    unsafe {
        exe.encodeToCommandBuffer_inputsArray_resultsArray_executionDescriptor(
            &mps_buf,
            &inputs_arr,
            Some(&outputs_arr),
            None,
        );
    }
    mtl_buf.commit();
    mtl_buf
}
