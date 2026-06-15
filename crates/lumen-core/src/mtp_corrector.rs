//! Per-depth diagonal-affine hidden-state corrector for recursive MTP drift.
//!
//! Self-speculative MTP drafts K tokens by **chaining** its head: the hidden
//! produced at draft step `k` is fed back as the input to step `k+1`. That fed
//! hidden is the head's *estimate* of the trunk's true hidden at that position;
//! the estimate drifts off-distribution as `k` grows, so deep drafts get worse
//! and acceptance collapses (lumen Qwen3.6: accept 0.57→0.36→0.24→0.18 for
//! K=1→4). The corrector counteracts the drift with a tiny **offline-fit**
//! per-depth, per-channel affine map
//!
//! ```text
//!   h_corr = scale_d * h + bias_d        (elementwise, depth d)
//! ```
//!
//! pulling the recursive draft hidden `x` toward the trunk's teacher-forced
//! hidden `y` at the same position. `scale_d`/`bias_d` are the closed-form
//! per-channel least-squares fit of `y ≈ scale·x + bias`:
//!
//! ```text
//!   scale_c = cov(x_c, y_c) / (var(x_c) + eps)
//!   bias_c  = mean(y_c) - scale_c * mean(x_c)
//! ```
//!
//! This is NOT head retraining — it's a fitted adapter (`depth × hidden` floats)
//! computed from calibration activations. Mirrors MTPLX
//! `mtplx/correctors/diagonal_affine.py` (`fit_c1_diagonal`). The acceptance
//! rule is unchanged (still exact/greedy or rejection sampling); the corrector
//! only improves the *draft quality* (reduces TV(p,q)).

/// Online per-(depth, channel) regression accumulator. Streams calibration
/// pairs `(x = recursive draft hidden, y = trunk teacher-forced hidden)` at a
/// given 1-based `depth` and accumulates the sufficient statistics for the
/// closed-form diagonal-affine fit — never stores the raw pairs.
#[derive(Debug, Clone)]
pub struct CorrectorStats {
    depth_count: usize,
    hidden: usize,
    /// Per-depth sample count (rows seen at that depth).
    n: Vec<u64>,
    /// Per-depth per-channel sums, each `depth_count * hidden` long.
    sum_x: Vec<f64>,
    sum_y: Vec<f64>,
    sum_xx: Vec<f64>,
    sum_xy: Vec<f64>,
    sum_yy: Vec<f64>,
}

impl CorrectorStats {
    pub fn new(depth_count: usize, hidden: usize) -> Self {
        let n_cells = depth_count * hidden;
        Self {
            depth_count,
            hidden,
            n: vec![0u64; depth_count],
            sum_x: vec![0.0; n_cells],
            sum_y: vec![0.0; n_cells],
            sum_xx: vec![0.0; n_cells],
            sum_xy: vec![0.0; n_cells],
            sum_yy: vec![0.0; n_cells],
        }
    }

    pub fn depth_count(&self) -> usize {
        self.depth_count
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Record one calibration row at 1-based `depth`. `x`/`y` must both be
    /// length `hidden`. Out-of-range depth or wrong length is ignored
    /// (defensive — calibration should never feed bad shapes).
    pub fn observe(&mut self, depth: usize, x: &[f32], y: &[f32]) {
        if depth == 0 || depth > self.depth_count {
            return;
        }
        if x.len() != self.hidden || y.len() != self.hidden {
            return;
        }
        let base = (depth - 1) * self.hidden;
        self.n[depth - 1] += 1;
        for c in 0..self.hidden {
            let xc = x[c] as f64;
            let yc = y[c] as f64;
            self.sum_x[base + c] += xc;
            self.sum_y[base + c] += yc;
            self.sum_xx[base + c] += xc * xc;
            self.sum_xy[base + c] += xc * yc;
            self.sum_yy[base + c] += yc * yc;
        }
    }

    /// Per-depth drift diagnostic: returns, for each 1-based depth,
    /// `(n, rel_mae_pre, rel_mae_post)` where rel_mae = sqrt(E[(Δ)²]) /
    /// sqrt(E[y²]). `pre` is the raw recursive-vs-target gap; `post` is the
    /// residual AFTER the fitted diagonal-affine correction. If `pre` is small,
    /// there is little drift to correct; if `post ≈ pre`, the affine model
    /// can't capture the drift (try low-rank / nonlinear).
    pub fn drift_report(&self, eps: f32) -> Vec<(u64, f32, f32)> {
        let eps = eps as f64;
        let mut out = Vec::with_capacity(self.depth_count);
        for d in 0..self.depth_count {
            let cnt = self.n[d];
            if cnt == 0 {
                out.push((0, 0.0, 0.0));
                continue;
            }
            let inv = 1.0 / cnt as f64;
            let base = d * self.hidden;
            let mut sse_pre = 0.0;
            let mut sse_post = 0.0;
            let mut sy2 = 0.0;
            for c in 0..self.hidden {
                let mx = self.sum_x[base + c] * inv;
                let my = self.sum_y[base + c] * inv;
                let exx = self.sum_xx[base + c] * inv;
                let eyy = self.sum_yy[base + c] * inv;
                let exy = self.sum_xy[base + c] * inv;
                // E[(x-y)^2] = E[x^2] - 2E[xy] + E[y^2]
                sse_pre += exx - 2.0 * exy + eyy;
                // residual of best affine y≈s·x+b: var(y) - cov^2/var(x)
                let varx = exx - mx * mx;
                let vary = eyy - my * my;
                let cov = exy - mx * my;
                let resid = vary - (cov * cov) / (varx + eps);
                sse_post += resid.max(0.0);
                sy2 += eyy;
            }
            let denom = (sy2.max(1e-12)).sqrt();
            out.push((
                cnt,
                (sse_pre.max(0.0).sqrt() / denom) as f32,
                (sse_post.max(0.0).sqrt() / denom) as f32,
            ));
        }
        out
    }

    /// Merge another accumulator (same shape) — for multi-shard calibration.
    pub fn merge(&mut self, other: &CorrectorStats) {
        if other.depth_count != self.depth_count || other.hidden != self.hidden {
            return;
        }
        for d in 0..self.depth_count {
            self.n[d] += other.n[d];
        }
        for i in 0..self.sum_x.len() {
            self.sum_x[i] += other.sum_x[i];
            self.sum_y[i] += other.sum_y[i];
            self.sum_xx[i] += other.sum_xx[i];
            self.sum_xy[i] += other.sum_xy[i];
            self.sum_yy[i] += other.sum_yy[i];
        }
    }

    /// Closed-form diagonal-affine fit. Depths with no samples get the identity
    /// map (scale 1, bias 0) so applying the corrector there is a no-op.
    pub fn finalize(&self, eps: f32) -> MtpCorrector {
        let eps = eps as f64;
        let mut scale = vec![1.0f32; self.depth_count * self.hidden];
        let mut bias = vec![0.0f32; self.depth_count * self.hidden];
        for d in 0..self.depth_count {
            let cnt = self.n[d];
            if cnt == 0 {
                continue;
            }
            let inv = 1.0 / cnt as f64;
            let base = d * self.hidden;
            for c in 0..self.hidden {
                let mx = self.sum_x[base + c] * inv;
                let my = self.sum_y[base + c] * inv;
                let var = self.sum_xx[base + c] * inv - mx * mx;
                let cov = self.sum_xy[base + c] * inv - mx * my;
                let s = cov / (var + eps);
                let b = my - s * mx;
                let s = if s.is_finite() { s as f32 } else { 1.0 };
                let b = if b.is_finite() { b as f32 } else { 0.0 };
                scale[base + c] = s;
                bias[base + c] = b;
            }
        }
        MtpCorrector {
            depth_count: self.depth_count,
            hidden: self.hidden,
            scale,
            bias,
        }
    }
}

/// Fitted per-depth diagonal-affine corrector. `scale`/`bias` are flat
/// `depth_count * hidden` row-major arrays (depth-major).
#[derive(Debug, Clone)]
pub struct MtpCorrector {
    depth_count: usize,
    hidden: usize,
    scale: Vec<f32>,
    bias: Vec<f32>,
}

const MTPC_MAGIC: &[u8; 4] = b"MTPC";

impl MtpCorrector {
    pub fn depth_count(&self) -> usize {
        self.depth_count
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Apply `h = scale_d * h + bias_d` in place at 1-based `depth`. No-op if
    /// depth is out of range or `h` has the wrong length (defensive).
    pub fn apply(&self, h: &mut [f32], depth: usize) {
        if depth == 0 || depth > self.depth_count || h.len() != self.hidden {
            return;
        }
        let base = (depth - 1) * self.hidden;
        for c in 0..self.hidden {
            h[c] = self.scale[base + c] * h[c] + self.bias[base + c];
        }
    }

    /// Serialize: magic `MTPC` + depth(u32 LE) + hidden(u32 LE) + scale f32[] +
    /// bias f32[] (all little-endian).
    pub fn to_bytes(&self) -> Vec<u8> {
        let n = self.scale.len();
        let mut out = Vec::with_capacity(12 + n * 8);
        out.extend_from_slice(MTPC_MAGIC);
        out.extend_from_slice(&(self.depth_count as u32).to_le_bytes());
        out.extend_from_slice(&(self.hidden as u32).to_le_bytes());
        for &v in &self.scale {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.bias {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Parse the [`to_bytes`] format. Returns `None` on bad magic / truncation.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 12 || &buf[0..4] != MTPC_MAGIC {
            return None;
        }
        let depth = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
        let hidden = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
        let n = depth * hidden;
        let need = 12 + n * 8;
        if buf.len() < need {
            return None;
        }
        let read_f32s = |start: usize| -> Vec<f32> {
            (0..n)
                .map(|i| {
                    let o = start + i * 4;
                    f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
                })
                .collect()
        };
        let scale = read_f32s(12);
        let bias = read_f32s(12 + n * 4);
        Some(Self {
            depth_count: depth,
            hidden,
            scale,
            bias,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_known_affine() {
        // Synthetic: y_c = a_c * x_c + b_c with per-channel a,b. The closed-form
        // diagonal fit must recover (a, b) from noiseless pairs.
        let hidden = 4;
        let depth_count = 2;
        let a = [2.0f32, -1.0, 0.5, 3.0];
        let b = [0.1f32, 0.2, -0.3, 1.0];
        let mut stats = CorrectorStats::new(depth_count, hidden);
        // Vary x across rows so var(x) > 0.
        for row in 0..64 {
            let x: Vec<f32> = (0..hidden)
                .map(|c| (row as f32) * 0.13 + c as f32 * 0.7 - 5.0)
                .collect();
            for depth in 1..=depth_count {
                // depth 2 uses a different affine to prove per-depth separation.
                let (aa, bb): (&[f32], &[f32]) = if depth == 1 {
                    (&a, &b)
                } else {
                    (&[1.5, 1.5, 1.5, 1.5], &[0.0, 0.0, 0.0, 0.0])
                };
                let y: Vec<f32> = (0..hidden).map(|c| aa[c] * x[c] + bb[c]).collect();
                stats.observe(depth, &x, &y);
            }
        }
        let corr = stats.finalize(1e-8);
        // Apply to a fresh x and check it maps to the true y at each depth.
        let x = [1.0f32, 2.0, 3.0, 4.0];
        let mut h1 = x;
        corr.apply(&mut h1, 1);
        for c in 0..hidden {
            let want = a[c] * x[c] + b[c];
            assert!((h1[c] - want).abs() < 1e-3, "d1 c{c}: {} vs {want}", h1[c]);
        }
        let mut h2 = x;
        corr.apply(&mut h2, 2);
        for c in 0..hidden {
            let want = 1.5 * x[c];
            assert!((h2[c] - want).abs() < 1e-3, "d2 c{c}: {} vs {want}", h2[c]);
        }
    }

    #[test]
    fn empty_depth_is_identity() {
        let corr = CorrectorStats::new(3, 5).finalize(1e-6);
        let mut h = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let orig = h;
        corr.apply(&mut h, 2);
        assert_eq!(h, orig, "no samples → identity map");
    }

    #[test]
    fn roundtrip_serialize() {
        let mut stats = CorrectorStats::new(2, 3);
        for row in 0..10 {
            let x = [row as f32, row as f32 + 1.0, row as f32 - 2.0];
            let y = [2.0 * x[0], 2.0 * x[1], 2.0 * x[2]];
            stats.observe(1, &x, &y);
            stats.observe(2, &x, &y);
        }
        let corr = stats.finalize(1e-8);
        let bytes = corr.to_bytes();
        let back = MtpCorrector::from_bytes(&bytes).expect("parse");
        assert_eq!(back.depth_count(), 2);
        assert_eq!(back.hidden(), 3);
        let mut a = [3.0f32, 4.0, 5.0];
        let mut b = a;
        corr.apply(&mut a, 1);
        back.apply(&mut b, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn merge_equiv_to_single_pass() {
        let mk = |seed: f32| {
            let mut s = CorrectorStats::new(1, 2);
            for r in 0..20 {
                let x = [r as f32 + seed, r as f32 * 0.5];
                let y = [1.3 * x[0] + 0.2, 0.7 * x[1] - 0.1];
                s.observe(1, &x, &y);
            }
            s
        };
        let mut a = mk(0.0);
        let b = mk(100.0);
        let mut single = mk(0.0);
        single.merge(&mk(100.0));
        a.merge(&b);
        let ca = a.finalize(1e-8);
        let cs = single.finalize(1e-8);
        assert_eq!(ca.to_bytes(), cs.to_bytes());
    }
}
