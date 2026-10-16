//! Splicing vision soft tokens over their placeholder rows.
//!
//! Both image towers land on the same problem once encoding is done: the prompt
//! carries a run of placeholder tokens per image, the tower produced a
//! `[run_len, hidden]` block for each, and the embedding stream has to be
//! rebuilt with those blocks in place of the placeholders' own embeddings.
//!
//! The wrinkle is chunked prefill. Runs are addressed against the **whole**
//! prompt, but a forward pass only ever sees one window of it — so a run can
//! fall outside the window entirely, or straddle either edge and contribute
//! only some of its rows. That arithmetic is where the off-by-ones live, so it
//! sits here as pure functions with no MLX dependency, unit-testable on any
//! machine. The few MLX-side helpers both towers also share are in the
//! `mlx-native`-gated [`tower`] submodule.

/// Contiguous `(start, len)` runs of `image_token` in a prompt — one per image,
/// in prompt order.
///
/// The chat templates emit each image as a sentinel, a run of placeholders, and
/// a closing sentinel, so the sentinels break the runs apart and two adjacent
/// images never merge into one.
pub(crate) fn image_token_runs(ids: &[u32], image_token: u32) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut i = 0usize;
    while i < ids.len() {
        if ids[i] != image_token {
            i += 1;
            continue;
        }
        let start = i;
        while i < ids.len() && ids[i] == image_token {
            i += 1;
        }
        runs.push((start, i - start));
    }
    runs
}

/// One image's soft tokens clipped to a prefill window.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct SoftSlice {
    /// Index into the `runs` / `soft` vectors.
    pub image: usize,
    /// Rows `[row_start, row_end)` of that image's `[len_i, hidden]` block.
    pub row_start: i32,
    pub row_end: i32,
    /// Where those rows land in the window: `[local_start, local_end)`.
    pub local_start: i32,
    pub local_end: i32,
}

/// Intersect prompt-global placeholder runs with the prefill window
/// `[span_start, span_start + len)`.
///
/// Single-pass prefill passes the whole prompt and every run maps onto itself.
/// Chunked prefill calls this once per chunk: runs before the window are
/// dropped, runs after it are dropped, and a run straddling an edge contributes
/// only its overlapping rows — the rest arrive with the neighbouring chunk.
pub(crate) fn clip_runs_to_window(
    span_start: i32,
    len: i32,
    runs: &[(usize, usize)],
) -> anyhow::Result<Vec<SoftSlice>> {
    let span_end = span_start + len;
    let mut out = Vec::new();
    let mut cursor = 0i32;
    for (image, &(start, run_len)) in runs.iter().enumerate() {
        let (start, run_len) = (start as i32, run_len as i32);
        let g0 = start.max(span_start);
        let g1 = (start + run_len).min(span_end);
        if g1 <= g0 {
            continue;
        }
        let local_start = g0 - span_start;
        if local_start < cursor {
            return Err(anyhow::anyhow!("image-token runs overlap or are unsorted"));
        }
        let local_end = g1 - span_start;
        out.push(SoftSlice {
            image,
            row_start: g0 - start,
            row_end: g1 - start,
            local_start,
            local_end,
        });
        cursor = local_end;
    }
    Ok(out)
}

/// What the two image towers share besides the splice arithmetic: weight
/// lookup and the two env knobs their forwards read.
#[cfg(feature = "mlx-native")]
pub(crate) mod tower {
    use anyhow::{Context, Result, anyhow};
    use mlx_rs::{Array, Dtype};
    use std::collections::HashMap;
    use std::sync::OnceLock;

    /// Run a tower in float32 instead of the checkpoint's bf16. Used by the
    /// golden-tensor parity tests, which compare against float32 PyTorch
    /// references; production leaves this off.
    pub(crate) fn vision_f32_enabled() -> bool {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        *ENABLED.get_or_init(|| {
            std::env::var("LUMEN_VISION_F32")
                .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes"))
                .unwrap_or(false)
        })
    }

    /// Drain the lazy graph every N encoder layers. `0` disables draining
    /// (fastest, highest peak memory). Without it every layer's activations
    /// stay live until the first eval and peak memory climbs by GBs on a long
    /// patch grid; the default 4 keeps the live set to a few layers', which
    /// matters on 36 GB machines where the text weights already take ~15 GB.
    pub(crate) fn vision_eval_every() -> usize {
        static EVERY: OnceLock<usize> = OnceLock::new();
        *EVERY.get_or_init(|| {
            std::env::var("LUMEN_VISION_EVAL_EVERY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(4)
        })
    }

    /// Look up a weight and cast it to the working dtype.
    pub(crate) fn take(
        weights: &HashMap<String, Array>,
        name: &str,
        dtype: Dtype,
    ) -> Result<Array> {
        weights
            .get(name)
            .ok_or_else(|| anyhow!("vision weight `{name}` not found"))?
            .as_dtype(dtype)
            .with_context(|| format!("cast `{name}` to {dtype:?}"))
    }

    /// Look up an `[out, in]` projection and return it transposed to `[in, out]`.
    pub(crate) fn take_t(
        weights: &HashMap<String, Array>,
        name: &str,
        dtype: Dtype,
    ) -> Result<Array> {
        let w = take(weights, name, dtype)?;
        mlx_rs::ops::transpose(&w).with_context(|| format!("transpose(`{name}`)"))
    }
}

#[cfg(test)]
mod run_tests {
    use super::image_token_runs;

    const IMG: u32 = 258880;

    #[test]
    fn finds_a_single_contiguous_run() {
        let ids = [2u32, 105, IMG, IMG, IMG, 107, 106];
        assert_eq!(image_token_runs(&ids, IMG), vec![(2, 3)]);
    }

    #[test]
    fn finds_multiple_runs_in_prompt_order() {
        let ids = [IMG, IMG, 7, 8, IMG, 9, IMG, IMG, IMG];
        assert_eq!(image_token_runs(&ids, IMG), vec![(0, 2), (4, 1), (6, 3)]);
    }

    #[test]
    fn text_only_prompt_has_no_runs() {
        let ids = [2u32, 105, 2364, 107, 106];
        assert!(image_token_runs(&ids, IMG).is_empty());
    }

    #[test]
    fn run_at_the_very_end_is_closed() {
        let ids = [1u32, 2, IMG, IMG];
        assert_eq!(image_token_runs(&ids, IMG), vec![(2, 2)]);
    }

    #[test]
    fn empty_prompt_has_no_runs() {
        assert!(image_token_runs(&[], IMG).is_empty());
    }
}

/// Window clipping is what lets chunked prefill (Gemma 4 streaming, and every
/// Qwen prefill) splice images at all: the runs are prompt-global but each
/// forward pass only sees one window of the prompt.
#[cfg(test)]
mod window_tests {
    use super::clip_runs_to_window;

    /// `(image, row_start, row_end, local_start, local_end)` for terseness.
    fn plan(
        span_start: i32,
        len: i32,
        runs: &[(usize, usize)],
    ) -> Vec<(usize, i32, i32, i32, i32)> {
        clip_runs_to_window(span_start, len, runs)
            .expect("clip")
            .into_iter()
            .map(|s| (s.image, s.row_start, s.row_end, s.local_start, s.local_end))
            .collect()
    }

    /// Single-pass prefill: the window is the whole prompt, so every run maps
    /// onto itself. This is the path the non-streaming code takes and it must
    /// stay exactly what it was before windowing existed.
    #[test]
    fn whole_prompt_window_maps_runs_onto_themselves() {
        let runs = [(4usize, 280usize), (300, 280)];
        assert_eq!(
            plan(0, 700, &runs),
            vec![(0, 0, 280, 4, 284), (1, 0, 280, 300, 580)]
        );
    }

    #[test]
    fn runs_outside_the_window_are_dropped() {
        let runs = [(0usize, 10usize), (500, 10)];
        assert!(plan(100, 100, &runs).is_empty());
    }

    /// A run straddling the trailing edge contributes only its head; the tail
    /// belongs to the next chunk. The two halves must tile the run exactly —
    /// no gap, no overlap.
    #[test]
    fn run_straddling_a_chunk_boundary_splits_without_gaps() {
        let runs = [(90usize, 20usize)];
        // chunk 0 covers [0,100): rows 0..10 land at local 90..100
        assert_eq!(plan(0, 100, &runs), vec![(0, 0, 10, 90, 100)]);
        // chunk 1 covers [100,200): rows 10..20 land at local 0..10
        assert_eq!(plan(100, 100, &runs), vec![(0, 10, 20, 0, 10)]);
    }

    /// A run wholly inside a later chunk is addressed relative to that chunk,
    /// not the prompt.
    #[test]
    fn interior_run_is_window_relative() {
        let runs = [(150usize, 20usize)];
        assert_eq!(plan(100, 100, &runs), vec![(0, 0, 20, 50, 70)]);
    }

    /// An image larger than one chunk shows up in all three windows it spans,
    /// each taking its own slice of rows.
    #[test]
    fn run_spanning_several_windows_is_covered_exactly_once_per_window() {
        let runs = [(10usize, 250usize)];
        assert_eq!(plan(0, 100, &runs), vec![(0, 0, 90, 10, 100)]);
        assert_eq!(plan(100, 100, &runs), vec![(0, 90, 190, 0, 100)]);
        assert_eq!(plan(200, 100, &runs), vec![(0, 190, 250, 0, 60)]);
    }

    #[test]
    fn no_runs_is_empty_not_an_error() {
        assert!(plan(0, 100, &[]).is_empty());
    }

    #[test]
    fn unsorted_runs_are_rejected() {
        let runs = [(50usize, 10usize), (10, 10)];
        assert!(clip_runs_to_window(0, 100, &runs).is_err());
    }
}
