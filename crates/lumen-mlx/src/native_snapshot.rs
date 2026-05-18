//! Track A1 / A2 — native prompt-cache snapshot + fork lifecycle.
//!
//! Mirrors `mlx_runner.py::_snapshot_cache` / `_restore_cache` for the
//! native (mlx-rs) runner. Two modes:
//!
//! * **Shallow** (`PromptCacheSnapshot::capture_shallow`) — `Array::clone()`
//!   refcount bumps only. Used for same-seq rollback (Track A2 spec-decode
//!   verify-loop). `restore_into` reassigns the captured Arrays back into
//!   the cache; the snapshot is then conceptually consumed by the caller.
//!
//! * **Deep** (`PromptCacheSnapshot::capture_deep`) — `Array::eval()` +
//!   `Array::deep_clone()` materialize independent buffers. Used for
//!   multi-fork from a master snapshot (Track A1 prefix caching). The
//!   master is reusable; `install_into_fresh` re-deep-clones at each
//!   install so sibling forks remain mutually isolated.
//!
//! See `track_a1_prefix_cache_landed.md` and `track_a2_spec_decode_pyo3_landed.md`
//! for the PyO3-side semantics this module mirrors.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // public surface used by runner_native.rs once snapshot stubs are wired
mod imp {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::Array;

    use crate::native_cache::{NativeLayerCache, NativePromptCache};

    /// Per-layer captured state. Both Full and Linear variants hold the
    /// captured Arrays directly; the `is_deep` flag on the enclosing
    /// `PromptCacheSnapshot` decides whether they are refcount-clones
    /// (shallow) or independent buffers (deep).
    pub enum LayerSnapshot {
        Full {
            keys: Option<Array>,
            values: Option<Array>,
            offset: usize,
        },
        Linear {
            slots: Vec<Option<Array>>,
            offset: usize,
            lengths: Option<Array>,
            left_padding: Option<Array>,
        },
    }

    /// Captured prompt-cache state. Construct via `capture_shallow` or
    /// `capture_deep`; apply via `restore_into` (rollback) or
    /// `install_into_fresh` (fork).
    pub struct PromptCacheSnapshot {
        layers: Vec<LayerSnapshot>,
        position: usize,
        is_deep: bool,
    }

    /// Produce a fresh contiguous Array with independent buffer, mirroring
    /// PyO3's `mx.array(a) + mx.eval(a)` deep-copy semantics.
    ///
    /// **Why not `Array::deep_clone()`** — mlx-rs 0.25.3's `Array::deep_clone`
    /// reads `mlx_array_data_*(self.as_ptr())` and constructs a new Array via
    /// `mlx_array_new_data(data, shape, dim, dtype)` passing only the shape.
    /// Strides are NOT propagated. For Arrays produced by `concatenate_axis`,
    /// `reshape`, or `transpose` whose physical layout is non-contiguous (a
    /// strided view over a parent buffer), `deep_clone` copies the wrong
    /// memory range — the resulting Array reads valid bytes from the parent
    /// but interprets them as if contiguous, yielding silently corrupt data.
    /// Empirically: A1 fork on Qwen3.6-35B mxfp4 hybrid produced divergent
    /// decode trajectory (axis 1+2 fail, axis 3+4 pass) when using
    /// `deep_clone`; the same path passes when `deep_clone` is replaced by
    /// `add(a, scalar 0)` which forces a contiguous evaluation.
    ///
    /// `add(a, 0)` is a no-op semantically but materializes through MLX's
    /// standard graph evaluator, producing a new contiguous Array in the
    /// same dtype + shape with bit-identical values. This avoids the broken
    /// FFI-level deep_clone path entirely.
    fn deep_clone_array(a: &Array) -> Result<Array> {
        let zero = Array::from_f32(0.0)
            .as_dtype(a.dtype())
            .context("native_snapshot: cast zero scalar to source dtype")?;
        let cloned = mlx_rs::ops::add(a, &zero)
            .context("native_snapshot: add(a, 0) for deep clone failed")?;
        cloned
            .eval()
            .context("native_snapshot: eval after add(a, 0)")?;
        Ok(cloned)
    }

    fn deep_clone_option(a: Option<&Array>) -> Result<Option<Array>> {
        a.map(deep_clone_array).transpose()
    }

    impl PromptCacheSnapshot {
        pub fn position(&self) -> usize {
            self.position
        }

        pub fn is_deep(&self) -> bool {
            self.is_deep
        }

        pub fn layer_count(&self) -> usize {
            self.layers.len()
        }

        /// Cheap snapshot — `Array::clone()` is a refcount bump per
        /// `native_cache.rs` invariants. Suitable for same-seq rollback
        /// only. NOT safe to install into a different seq's cache because
        /// subsequent in-place lazy-graph evaluations on the source could
        /// surface through the shared buffer.
        pub fn capture_shallow(cache: &NativePromptCache, position: usize) -> Result<Self> {
            let layers = cache
                .layers()
                .iter()
                .map(|l| -> Result<LayerSnapshot> {
                    match l {
                        NativeLayerCache::Full(c) => Ok(LayerSnapshot::Full {
                            // Capture logical-length view, not the padded
                            // buffer — restore path will refill via
                            // update_and_fetch + grow on next step.
                            keys: c.keys_view()?,
                            values: c.values_view()?,
                            offset: c.offset(),
                        }),
                        NativeLayerCache::Linear(c) => Ok(LayerSnapshot::Linear {
                            slots: c.slots().to_vec(),
                            offset: c.offset(),
                            lengths: c.lengths().cloned(),
                            left_padding: c.left_padding().cloned(),
                        }),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                layers,
                position,
                is_deep: false,
            })
        }

        /// Materializes independent Array buffers via `eval` + `deep_clone`.
        /// Required for fork-to-new-seq usage (Track A1 prefix cache).
        pub fn capture_deep(cache: &NativePromptCache, position: usize) -> Result<Self> {
            let layers = cache
                .layers()
                .iter()
                .map(|l| -> Result<LayerSnapshot> {
                    match l {
                        NativeLayerCache::Full(c) => Ok(LayerSnapshot::Full {
                            keys: deep_clone_option(c.keys_view()?.as_ref())?,
                            values: deep_clone_option(c.values_view()?.as_ref())?,
                            offset: c.offset(),
                        }),
                        NativeLayerCache::Linear(c) => {
                            let mut slots = Vec::with_capacity(c.slots().len());
                            for slot in c.slots() {
                                slots.push(deep_clone_option(slot.as_ref())?);
                            }
                            Ok(LayerSnapshot::Linear {
                                slots,
                                offset: c.offset(),
                                lengths: deep_clone_option(c.lengths())?,
                                left_padding: deep_clone_option(c.left_padding())?,
                            })
                        }
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self {
                layers,
                position,
                is_deep: true,
            })
        }

        /// Apply this snapshot in-place — same-seq rollback. Reassigns each
        /// layer's captured state. The snapshot itself is not modified, but
        /// the caller should release it after restore (single-shot semantics
        /// from `runner_native::restore_state`).
        pub fn restore_into(&self, cache: &mut NativePromptCache) -> Result<()> {
            if cache.len() != self.layers.len() {
                return Err(anyhow!(
                    "PromptCacheSnapshot::restore_into: layer count mismatch (cache={}, snap={})",
                    cache.len(),
                    self.layers.len()
                ));
            }
            for (i, snap) in self.layers.iter().enumerate() {
                let layer = cache
                    .layer_mut(i)
                    .ok_or_else(|| anyhow!("PromptCacheSnapshot::restore_into: layer {i} missing"))?;
                apply_layer(layer, snap, /* fork = */ false, i)?;
            }
            Ok(())
        }

        /// Install this (deep) snapshot into a freshly-constructed cache.
        /// Each Array is **re-deep-cloned** so the destination cache's
        /// buffers are independent of both the master snapshot and any
        /// sibling forks. Mirrors `mlx_runner.py::_restore_cache`'s
        /// `arrays_deep` / `kv_deep` branches.
        ///
        /// For shallow snapshots, falls back to `restore_into` (caller
        /// accepts reduced isolation — usually used for same-seq paths
        /// where this is fine).
        pub fn install_into_fresh(&self, cache: &mut NativePromptCache) -> Result<()> {
            if !self.is_deep {
                return self.restore_into(cache);
            }
            if cache.len() != self.layers.len() {
                return Err(anyhow!(
                    "PromptCacheSnapshot::install_into_fresh: layer count mismatch (cache={}, snap={})",
                    cache.len(),
                    self.layers.len()
                ));
            }
            for (i, snap) in self.layers.iter().enumerate() {
                let layer = cache.layer_mut(i).ok_or_else(|| {
                    anyhow!("PromptCacheSnapshot::install_into_fresh: layer {i} missing")
                })?;
                apply_layer(layer, snap, /* fork = */ true, i)?;
            }
            Ok(())
        }
    }

    /// Common write path — reassigns captured state into a layer. When
    /// `fork == true`, each Array is re-deep-cloned for sibling isolation.
    fn apply_layer(
        layer: &mut NativeLayerCache,
        snap: &LayerSnapshot,
        fork: bool,
        layer_idx: usize,
    ) -> Result<()> {
        match (layer, snap) {
            (
                NativeLayerCache::Full(c),
                LayerSnapshot::Full {
                    keys,
                    values,
                    offset,
                },
            ) => {
                let (kk, vv) = if fork {
                    (
                        deep_clone_option(keys.as_ref())?,
                        deep_clone_option(values.as_ref())?,
                    )
                } else {
                    (keys.clone(), values.clone())
                };
                c.set_state(kk, vv, *offset);
                Ok(())
            }
            (
                NativeLayerCache::Linear(c),
                LayerSnapshot::Linear {
                    slots,
                    offset,
                    lengths,
                    left_padding,
                },
            ) => {
                if c.len() != slots.len() {
                    return Err(anyhow!(
                        "PromptCacheSnapshot apply_layer: linear layer {layer_idx} slot count mismatch (cache={}, snap={})",
                        c.len(),
                        slots.len()
                    ));
                }
                for (idx, s) in slots.iter().enumerate() {
                    let cloned = if fork {
                        deep_clone_option(s.as_ref())?
                    } else {
                        s.clone()
                    };
                    c.set_slot(idx, cloned)?;
                }
                c.set_offset(*offset);
                let (l, lp) = if fork {
                    (
                        deep_clone_option(lengths.as_ref())?,
                        deep_clone_option(left_padding.as_ref())?,
                    )
                } else {
                    (lengths.clone(), left_padding.clone())
                };
                c.set_lengths(l);
                c.set_left_padding(lp);
                Ok(())
            }
            (NativeLayerCache::Full(_), LayerSnapshot::Linear { .. })
            | (NativeLayerCache::Linear(_), LayerSnapshot::Full { .. }) => Err(anyhow!(
                "PromptCacheSnapshot apply_layer: layer kind mismatch at {layer_idx}"
            )),
        }
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)]
pub(crate) use imp::{LayerSnapshot, PromptCacheSnapshot};

#[cfg(all(test, feature = "mlx-native"))]
mod lifecycle_tests {
    use super::imp::PromptCacheSnapshot;
    use crate::native_cache::NativePromptCache;
    use mlx_rs::Array;

    fn make_kv_block(b: i32, h: i32, s: i32, d: i32, fill: f32) -> Array {
        let n = (b * h * s * d) as usize;
        let data: Vec<f32> = (0..n).map(|i| fill + i as f32 * 0.001).collect();
        Array::from_slice(&data, &[b, h, s, d])
    }

    /// Hybrid layer kinds matching Qwen3.5-MoE structure: 3 linear + 1 full,
    /// repeated. Slot count 2 (conv_state + ssm_state).
    fn make_hybrid_cache() -> NativePromptCache {
        let kinds = [true, true, true, false, true, true, true, false];
        NativePromptCache::new(&kinds, 2)
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn shallow_capture_then_restore_round_trips_full_layer() {
        let mut cache = make_hybrid_cache();
        let full = cache.layer_mut(3).unwrap().as_full_mut().unwrap();
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 4, 8, 1.0);
        full.update_and_fetch(&k, &v).unwrap();
        assert_eq!(cache.full_attn_offset(), 4);

        let snap = PromptCacheSnapshot::capture_shallow(&cache, 4).unwrap();

        // Advance cache further.
        let full = cache.layer_mut(3).unwrap().as_full_mut().unwrap();
        let k2 = make_kv_block(1, 2, 1, 8, 2.0);
        let v2 = make_kv_block(1, 2, 1, 8, 3.0);
        full.update_and_fetch(&k2, &v2).unwrap();
        assert_eq!(cache.full_attn_offset(), 5);

        // Restore — offset should rewind to 4.
        snap.restore_into(&mut cache).unwrap();
        assert_eq!(cache.full_attn_offset(), 4);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn shallow_capture_then_restore_round_trips_linear_layer() {
        let mut cache = make_hybrid_cache();
        let lin = cache.layer_mut(0).unwrap().as_linear_mut().unwrap();
        lin.set(0, make_kv_block(1, 1, 1, 4, 0.5)).unwrap();
        lin.advance(7);
        assert_eq!(cache.layer(0).unwrap().offset(), 7);

        let snap = PromptCacheSnapshot::capture_shallow(&cache, 7).unwrap();

        let lin = cache.layer_mut(0).unwrap().as_linear_mut().unwrap();
        lin.set(0, make_kv_block(1, 1, 1, 4, 9.0)).unwrap();
        lin.advance(3);
        assert_eq!(cache.layer(0).unwrap().offset(), 10);

        snap.restore_into(&mut cache).unwrap();
        assert_eq!(cache.layer(0).unwrap().offset(), 7);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn deep_capture_then_install_into_fresh_seeds_independent_cache() {
        let mut master = make_hybrid_cache();
        let full = master.layer_mut(3).unwrap().as_full_mut().unwrap();
        let k = make_kv_block(1, 2, 4, 8, 0.0);
        let v = make_kv_block(1, 2, 4, 8, 1.0);
        full.update_and_fetch(&k, &v).unwrap();
        let lin = master.layer_mut(0).unwrap().as_linear_mut().unwrap();
        lin.set(0, make_kv_block(1, 1, 1, 4, 0.5)).unwrap();
        lin.advance(4);

        let snap = PromptCacheSnapshot::capture_deep(&master, 4).unwrap();
        assert!(snap.is_deep());
        assert_eq!(snap.position(), 4);

        let mut fork = master.clone_structure_only();
        snap.install_into_fresh(&mut fork).unwrap();

        assert_eq!(fork.full_attn_offset(), 4);
        assert_eq!(fork.layer(0).unwrap().offset(), 4);
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn capture_layer_count_matches_cache() {
        let cache = make_hybrid_cache();
        let snap = PromptCacheSnapshot::capture_shallow(&cache, 0).unwrap();
        assert_eq!(snap.layer_count(), cache.len());
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn restore_into_rejects_layer_count_mismatch() {
        let cache = make_hybrid_cache();
        let snap = PromptCacheSnapshot::capture_shallow(&cache, 0).unwrap();

        let kinds_short = [true, true];
        let mut shorter = NativePromptCache::new(&kinds_short, 2);
        let err = snap.restore_into(&mut shorter).unwrap_err();
        assert!(err.to_string().contains("layer count mismatch"));
    }
}
