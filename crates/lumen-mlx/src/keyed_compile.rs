//! MLX `compile` with the constants of a call in the cache key.
//!
//! The mlx-rs fork's compile wrappers hand `mlx_detail_compile` an empty list of constants and
//! name the function by the `TypeId` of what they are given. MLX keeps one traced graph per
//! function id and per input shapes and dtypes, so a closure that bakes a constant in at trace time
//! (a head count, a clip threshold, a scale) is replayed for every later call with the same shapes,
//! whatever that call's constant is. Callers had to make sure no two calls differed, with a
//! process-global claim of the first constants, and fall back to uncompiled code for a layer whose
//! constants did not match.
//!
//! [`KeyedCompiled`] calls `mlx_detail_compile` itself, with the constants the caller states: the
//! numbers (a head count, a clip threshold's bits) the traced function builds its graph from. MLX
//! then keeps one graph per distinct constants as it does per distinct shapes. Its function id is a
//! number of its own, so two compiled functions can never share one.
//!
//! The traced function is a plain `fn` that is handed those same constants next to its arguments, so
//! the key MLX looks graphs up by is, by construction, everything the function reads besides the
//! arrays: a constant it did not read from there could not be forgotten in the key, because it could
//! not be read at all. It may return an error or panic: neither crosses the C boundary, both come
//! back as an `Err`.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::cell::{Cell, RefCell};
    use std::ffi::c_void;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use anyhow::{Result, anyhow, bail};
    use mlx_rs::Array;
    use mlx_rs::error::Exception;

    /// The function MLX traces: the arrays and the constants of the call in, arrays out.
    pub type TracedFn = fn(&[Array], &[u64]) -> std::result::Result<Vec<Array>, Exception>;

    /// Function ids start far from the small numbers and from the type-id hashes the fork uses.
    static NEXT_ID: AtomicUsize = AtomicUsize::new(0x4c55_4d45_4e00_0000);

    thread_local! {
        /// What the traced function reported when it failed (MLX only sees a non-zero status).
        static TRACE_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
        /// The constants of the call being applied, for the trampoline MLX calls back into.
        static CONSTANTS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
        static HANDLER_READY: Cell<bool> = const { Cell::new(false) };
    }

    /// MLX's own error handler ends the process. mlx-rs replaces it with one that keeps the error
    /// for the caller, but only when its first operation on a thread runs; a compiled call that fails
    /// before that would take the process with it, so one operation is run first.
    fn ensure_error_handler() {
        if !HANDLER_READY.with(|ready| ready.replace(true)) {
            drop(mlx_rs::ops::zeros::<f32>(&[1]));
        }
    }

    /// A compiled function whose cache key includes the constants of each call.
    pub struct KeyedCompiled {
        f: TracedFn,
        shapeless: bool,
        id: usize,
    }

    impl KeyedCompiled {
        pub fn new(f: TracedFn, shapeless: bool) -> Self {
            Self {
                f,
                shapeless,
                id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            }
        }

        /// Runs the function on `args`, tracing it first when MLX has no graph for these shapes,
        /// dtypes and `constants` (which the function is handed when it is traced).
        pub fn call(&self, args: &[&Array], constants: &[u64]) -> Result<Vec<Array>> {
            ensure_error_handler();
            TRACE_ERROR.with(|e| *e.borrow_mut() = None);
            let outer = CONSTANTS.with(|c| c.replace(constants.to_vec()));
            let result = self.apply(args, constants);
            CONSTANTS.with(|c| *c.borrow_mut() = outer);
            result
        }

        fn apply(&self, args: &[&Array], constants: &[u64]) -> Result<Vec<Array>> {
            // SAFETY: every handle made below is freed on every path; the payload is this
            // function's own pointer, which `trampoline` turns back into a `TracedFn`.
            unsafe {
                let inner = mlx_sys::mlx_closure_new_func_payload(
                    Some(trampoline),
                    self.f as usize as *mut c_void,
                    None,
                );
                let mut compiled = mlx_sys::mlx_closure_new();
                let status = mlx_sys::mlx_detail_compile(
                    &mut compiled,
                    inner,
                    self.id,
                    self.shapeless,
                    constants.as_ptr(),
                    constants.len(),
                );
                mlx_sys::mlx_closure_free(inner);
                if status != 0 {
                    mlx_sys::mlx_closure_free(compiled);
                    return Err(failure("compile"));
                }
                let input = mlx_sys::mlx_vector_array_new();
                for a in args {
                    if mlx_sys::mlx_vector_array_append_value(input, a.as_ptr()) != 0 {
                        mlx_sys::mlx_vector_array_free(input);
                        mlx_sys::mlx_closure_free(compiled);
                        bail!("could not pass an argument to the compiled function");
                    }
                }
                let mut out = mlx_sys::mlx_vector_array_new();
                let status = mlx_sys::mlx_closure_apply(&mut out, compiled, input);
                mlx_sys::mlx_vector_array_free(input);
                mlx_sys::mlx_closure_free(compiled);
                if status != 0 {
                    mlx_sys::mlx_vector_array_free(out);
                    return Err(failure("apply"));
                }
                let taken = take_arrays(out);
                mlx_sys::mlx_vector_array_free(out);
                taken
            }
        }
    }

    impl Drop for KeyedCompiled {
        fn drop(&mut self) {
            // SAFETY: erases this function's graphs from MLX's cache; the id is unique to `self`.
            unsafe {
                mlx_sys::mlx_detail_compile_erase(self.id);
            }
        }
    }

    fn failure(stage: &str) -> anyhow::Error {
        match TRACE_ERROR.with(|e| e.borrow_mut().take()) {
            Some(message) => anyhow!("compiled function failed: {message}"),
            None => anyhow!("MLX could not {stage} the compiled function"),
        }
    }

    /// The arrays of an MLX vector, each as an owned handle.
    ///
    /// # Safety
    /// `vector` must be a live `mlx_vector_array`.
    unsafe fn take_arrays(vector: mlx_sys::mlx_vector_array) -> Result<Vec<Array>> {
        // SAFETY: the caller's contract; each handle from `get` is a new reference we own.
        unsafe {
            let size = mlx_sys::mlx_vector_array_size(vector);
            let mut arrays = Vec::with_capacity(size);
            for i in 0..size {
                let mut a = mlx_sys::mlx_array_new();
                if mlx_sys::mlx_vector_array_get(&mut a, vector, i) != 0 {
                    mlx_sys::mlx_array_free(a);
                    bail!("could not read result {i} of the compiled function");
                }
                arrays.push(Array::from_ptr(a));
            }
            Ok(arrays)
        }
    }

    /// MLX's callback: runs the traced function. Nothing escapes: a failure is reported by status
    /// and a message in [`TRACE_ERROR`], and a panic is caught here.
    unsafe extern "C" fn trampoline(
        ret: *mut mlx_sys::mlx_vector_array,
        input: mlx_sys::mlx_vector_array,
        payload: *mut c_void,
    ) -> i32 {
        // SAFETY: `payload` is the `TracedFn` `KeyedCompiled::call` passed, as an address.
        let f: TracedFn = unsafe { std::mem::transmute::<usize, TracedFn>(payload as usize) };
        let traced = std::panic::catch_unwind(|| -> std::result::Result<_, String> {
            // SAFETY: `input` is live for the duration of the callback.
            let arrays = unsafe { take_arrays(input) }.map_err(|e| e.to_string())?;
            let constants = CONSTANTS.with(|c| c.borrow().clone());
            let outputs = f(&arrays, &constants).map_err(|e| e.what().to_string())?;
            // SAFETY: a new vector we fill, then hand to MLX, which frees it.
            unsafe {
                let vector = mlx_sys::mlx_vector_array_new();
                for a in &outputs {
                    if mlx_sys::mlx_vector_array_append_value(vector, a.as_ptr()) != 0 {
                        mlx_sys::mlx_vector_array_free(vector);
                        return Err("could not return an array".to_string());
                    }
                }
                Ok(vector)
            }
        });
        match traced {
            Ok(Ok(vector)) => {
                // SAFETY: `ret` is the callback's result slot.
                unsafe { *ret = vector };
                0
            }
            Ok(Err(message)) => {
                TRACE_ERROR.with(|e| *e.borrow_mut() = Some(message));
                1
            }
            Err(_) => {
                TRACE_ERROR.with(|e| *e.borrow_mut() = Some("the function panicked".to_string()));
                1
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use std::cell::Cell;
        use std::sync::OnceLock;

        use super::*;

        thread_local! {
            /// How many times the function below has been traced on this thread.
            static TRACES: Cell<usize> = const { Cell::new(0) };
        }

        /// `x * f32::from_bits(constants[0])`; `constants[1]` of 1 fails and of 2 panics.
        fn scaled(args: &[Array], constants: &[u64]) -> std::result::Result<Vec<Array>, Exception> {
            TRACES.with(|n| n.set(n.get() + 1));
            match constants.get(1) {
                Some(1) => Err(Exception::custom("planned failure")),
                Some(2) => panic!("planned panic"),
                _ => Ok(vec![mlx_rs::ops::multiply(
                    &args[0],
                    Array::from_f32(f32::from_bits(constants[0] as u32)),
                )?]),
            }
        }

        fn scaled_twin(
            args: &[Array],
            _constants: &[u64],
        ) -> std::result::Result<Vec<Array>, Exception> {
            Ok(vec![mlx_rs::ops::add(&args[0], Array::from_f32(1.0))?])
        }

        fn host(a: &Array) -> Vec<f32> {
            a.eval().unwrap();
            a.as_slice::<f32>().to_vec()
        }

        fn bits(x: f32) -> u64 {
            u64::from(x.to_bits())
        }

        #[test]
        fn a_call_with_other_constants_is_traced_again_and_one_with_the_same_is_replayed() {
            static SLOT: OnceLock<KeyedCompiled> = OnceLock::new();
            let compiled = SLOT.get_or_init(|| KeyedCompiled::new(scaled, false));
            let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3]);
            let run = |scale: f32| host(&compiled.call(&[&x], &[bits(scale)]).unwrap()[0]);
            let traces = || TRACES.with(Cell::get);
            // Each constant is served its own graph, in whatever order they come.
            for scale in [2.0, 3.0, 2.0, 3.0, 5.0, 2.0, 5.0, 3.0] {
                assert_eq!(
                    run(scale),
                    [scale, 2.0 * scale, 3.0 * scale],
                    "scale {scale}"
                );
            }
            // And once each is traced, the same constants replay a graph instead of tracing again.
            let settled = traces();
            for scale in [2.0, 3.0, 5.0, 3.0, 2.0] {
                assert_eq!(
                    run(scale),
                    [scale, 2.0 * scale, 3.0 * scale],
                    "scale {scale}"
                );
            }
            assert_eq!(
                traces(),
                settled,
                "a call with known constants traced again"
            );
        }

        #[test]
        fn two_compiled_functions_never_share_an_identity() {
            let a = KeyedCompiled::new(scaled, false);
            let b = KeyedCompiled::new(scaled_twin, false);
            let c = KeyedCompiled::new(scaled, false);
            assert!(a.id != b.id && b.id != c.id && a.id != c.id);
            let x = Array::from_slice(&[1.0f32, 2.0], &[2]);
            // Same shapes, dtypes and constants: only the identity tells them apart.
            assert_eq!(host(&a.call(&[&x], &[bits(2.0)]).unwrap()[0]), [2.0, 4.0]);
            assert_eq!(host(&b.call(&[&x], &[bits(2.0)]).unwrap()[0]), [2.0, 3.0]);
        }

        #[test]
        fn a_failure_or_a_panic_in_the_traced_function_is_an_error_not_an_abort() {
            let failing = KeyedCompiled::new(scaled, false);
            let x = Array::from_slice(&[1.0f32], &[1]);
            let err = failing
                .call(&[&x], &[bits(1.0), 1])
                .unwrap_err()
                .to_string();
            assert!(err.contains("planned failure"), "{err}");
            let err = failing
                .call(&[&x], &[bits(1.0), 2])
                .unwrap_err()
                .to_string();
            assert!(err.contains("panicked"), "{err}");
            assert_eq!(host(&failing.call(&[&x], &[bits(4.0)]).unwrap()[0]), [4.0]);
        }
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::*;
