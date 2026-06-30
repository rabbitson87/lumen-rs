//! Per-depth **orthogonal-Procrustes** hidden-state corrector for recursive MTP.
//!
//! The diagonal-affine corrector ([`crate::mtp_corrector`]) failed on this
//! single-head recursive MTP: it reduced the raw L2 gap between the chained
//! draft hidden `x` and the trunk's teacher-forced hidden `y` but *lowered*
//! acceptance. Two mechanisms explain that:
//!
//! 1. The MTP head **RMSNorms its input**, so a uniform per-channel *scale*
//!    correction is largely washed out — only the *direction* (rotation) of
//!    the hidden survives the norm into the head's logits.
//! 2. The depth-1 rel-MAE is already huge (≈0.99 on a dense 9B, ≈6.8 on the
//!    35B MoE) — `x` and `y` are not the same representation, they differ by
//!    a (near-)rotation, which a diagonal map structurally cannot express.
//!
//! This corrector fits, per depth, the best **scaled rotation** mapping the
//! centered draft hidden onto the centered trunk hidden:
//!
//! ```text
//!   h_corr = s_d · (x - mean_x_d) · R_d + mean_y_d     (R_d orthogonal)
//! ```
//!
//! `R_d`, `s_d`, `mean_x_d`, `mean_y_d` are the closed-form orthogonal
//! Procrustes solution of `y ≈ s·x·R`:
//!
//! ```text
//!   M       = Xc^T Yc            (d×d centered cross-covariance)
//!   M       = U Σ V^T            (SVD)
//!   R       = U V^T              (nearest orthogonal matrix)
//!   s       = tr(Σ) / ‖Xc‖_F^2   (optimal isotropic scale)
//! ```
//!
//! Like the diagonal corrector this is NOT head retraining — it is a fitted
//! `depth × (d×d)` adapter computed from calibration activations. The
//! acceptance rule is unchanged; only draft quality (TV(p,q)) is affected.
//!
//! The `finalize`/`residual_report` pair is itself decisive: the Procrustes
//! residual is the floor any orthogonal correction can reach. If it stays
//! near the raw rel-MAE, `x` and `y` are nearly uncorrelated (only end-to-end
//! retraining, C4, can help); if it drops far below the diagonal residual,
//! the rotation is the missing capability and wiring `apply` is worthwhile.

use faer::Mat;

/// Streaming accumulator of the per-(depth) sufficient statistics for the
/// orthogonal-Procrustes fit. Stores, per depth: sample count, the per-channel
/// sums of `x` and `y`, the **full `d×d` cross-product** `Σ x_iᵀ y_i`, and the
/// scalar traces `Σ‖x_i‖²` / `Σ‖y_i‖²`. Never stores raw rows.
///
/// Memory: `depth_count · hidden²` f64 for the cross-product (the dominant
/// term — e.g. 3 · 4096² · 8 B ≈ 402 MB). Calibration-only; not resident at
/// serve time.
#[derive(Clone)]
pub struct ProcrustesStats {
    depth_count: usize,
    hidden: usize,
    n: Vec<u64>,
    sx: Vec<f64>,     // depth · hidden
    sy: Vec<f64>,     // depth · hidden
    sxy: Vec<f64>,    // depth · hidden · hidden  (row-major: [i*hidden + j] = Σ x_i·y_j)
    sxx_tr: Vec<f64>, // per depth: Σ_ic x_ic²
    syy_tr: Vec<f64>, // per depth: Σ_ic y_ic²
}

impl ProcrustesStats {
    pub fn new(depth_count: usize, hidden: usize) -> Self {
        Self {
            depth_count,
            hidden,
            n: vec![0u64; depth_count],
            sx: vec![0.0; depth_count * hidden],
            sy: vec![0.0; depth_count * hidden],
            sxy: vec![0.0; depth_count * hidden * hidden],
            sxx_tr: vec![0.0; depth_count],
            syy_tr: vec![0.0; depth_count],
        }
    }

    pub fn depth_count(&self) -> usize {
        self.depth_count
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Record one calibration row at 1-based `depth`. `x` = recursive draft
    /// hidden, `y` = trunk teacher-forced hidden. The `d×d` outer-product
    /// accumulation is the hot cost (`hidden²` FMA per call).
    pub fn observe(&mut self, depth: usize, x: &[f32], y: &[f32]) {
        if depth == 0 || depth > self.depth_count {
            return;
        }
        if x.len() != self.hidden || y.len() != self.hidden {
            return;
        }
        let d = self.hidden;
        let vbase = (depth - 1) * d;
        let mbase = (depth - 1) * d * d;
        self.n[depth - 1] += 1;
        let mut xx = 0.0f64;
        let mut yy = 0.0f64;
        for i in 0..d {
            let xi = x[i] as f64;
            self.sx[vbase + i] += xi;
            xx += xi * xi;
            let row = mbase + i * d;
            // Σ x_i · y_j  — the cross-product outer product.
            for j in 0..d {
                self.sxy[row + j] += xi * (y[j] as f64);
            }
        }
        for j in 0..d {
            let yj = y[j] as f64;
            self.sy[vbase + j] += yj;
            yy += yj * yj;
        }
        self.sxx_tr[depth - 1] += xx;
        self.syy_tr[depth - 1] += yy;
    }

    /// Merge another accumulator of identical shape (multi-shard calibration).
    pub fn merge(&mut self, other: &ProcrustesStats) {
        if other.depth_count != self.depth_count || other.hidden != self.hidden {
            return;
        }
        for d in 0..self.depth_count {
            self.n[d] += other.n[d];
            self.sxx_tr[d] += other.sxx_tr[d];
            self.syy_tr[d] += other.syy_tr[d];
        }
        for i in 0..self.sx.len() {
            self.sx[i] += other.sx[i];
            self.sy[i] += other.sy[i];
        }
        for i in 0..self.sxy.len() {
            self.sxy[i] += other.sxy[i];
        }
    }

    /// Centered cross-covariance `M = Xc^T Yc / n` (d×d) for `depth` (1-based),
    /// plus `varx_tr = ‖Xc‖_F²/n` and `vary_tr = ‖Yc‖_F²/n`. Returns `None` if
    /// the depth has no samples.
    fn centered(&self, depth: usize) -> Option<(Vec<f64>, Vec<f64>, Vec<f64>, f64, f64)> {
        let cnt = self.n[depth - 1];
        if cnt == 0 {
            return None;
        }
        let d = self.hidden;
        let inv = 1.0 / cnt as f64;
        let vbase = (depth - 1) * d;
        let mbase = (depth - 1) * d * d;
        let mx: Vec<f64> = (0..d).map(|i| self.sx[vbase + i] * inv).collect();
        let my: Vec<f64> = (0..d).map(|j| self.sy[vbase + j] * inv).collect();
        // M[i,j] = E[x_i y_j] - mx_i my_j
        let mut m = vec![0.0f64; d * d];
        for i in 0..d {
            let row = mbase + i * d;
            let mri = i * d;
            let mxi = mx[i];
            for j in 0..d {
                m[mri + j] = self.sxy[row + j] * inv - mxi * my[j];
            }
        }
        let mx2: f64 = mx.iter().map(|v| v * v).sum();
        let my2: f64 = my.iter().map(|v| v * v).sum();
        let varx_tr = (self.sxx_tr[depth - 1] * inv - mx2).max(0.0);
        let vary_tr = (self.syy_tr[depth - 1] * inv - my2).max(0.0);
        Some((m, mx, my, varx_tr, vary_tr))
    }

    /// Solve the orthogonal Procrustes fit for one depth from the centered
    /// cross-covariance. Returns `(R row-major d×d, s, sum_sigma)`.
    fn fit_depth(m: &[f64], d: usize, varx_tr: f64) -> (Vec<f64>, f64, f64) {
        // M (d×d) → SVD → R = U Vᵀ, s = ΣΣ / ‖Xc‖².
        let mat = Mat::from_fn(d, d, |i, j| m[i * d + j]);
        let svd = match mat.thin_svd() {
            Ok(s) => s,
            Err(_) => {
                // Degenerate → identity rotation, unit scale.
                let mut r = vec![0.0f64; d * d];
                for i in 0..d {
                    r[i * d + i] = 1.0;
                }
                return (r, 1.0, 0.0);
            }
        };
        let u = svd.U();
        let v = svd.V();
        let sdiag = svd.S();
        let sum_sigma: f64 = (0..d).map(|i| sdiag[i]).sum();
        // R = U Vᵀ  →  R[i,j] = Σ_k U[i,k] V[j,k]
        let mut r = vec![0.0f64; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut acc = 0.0f64;
                for k in 0..d {
                    acc += u[(i, k)] * v[(j, k)];
                }
                r[i * d + j] = acc;
            }
        }
        let s = if varx_tr > 1e-12 {
            sum_sigma / varx_tr
        } else {
            1.0
        };
        (r, s, sum_sigma)
    }

    /// Per-depth Procrustes residual diagnostic: `(n, rel_raw, rel_proc)` where
    /// `rel_raw = ‖x-y‖/‖y‖` (centered) and `rel_proc = ‖y - s·x·R‖/‖y‖` is the
    /// residual after the best scaled-rotation fit. `rel_proc ≪ rel_raw` ⇒ a
    /// rotation aligns the representations (wiring `apply` is worthwhile);
    /// `rel_proc ≈ rel_raw` ⇒ `x`,`y` are near-uncorrelated (only C4 helps).
    pub fn residual_report(&self) -> Vec<(u64, f32, f32)> {
        let d = self.hidden;
        let mut out = Vec::with_capacity(self.depth_count);
        for depth in 1..=self.depth_count {
            let cnt = self.n[depth - 1];
            let Some((m, _mx, _my, varx_tr, vary_tr)) = self.centered(depth) else {
                out.push((0, 0.0, 0.0));
                continue;
            };
            // raw centered gap: ‖Xc-Yc‖²/n = varx_tr - 2·tr(M) + vary_tr
            let tr_m: f64 = (0..d).map(|i| m[i * d + i]).sum();
            let raw = (varx_tr - 2.0 * tr_m + vary_tr).max(0.0);
            let (_r, s, sum_sigma) = Self::fit_depth(&m, d, varx_tr);
            // proc residual: ‖Yc - s·Xc·R‖²/n = vary_tr - 2·s·ΣΣ + s²·varx_tr
            let proc = (vary_tr - 2.0 * s * sum_sigma + s * s * varx_tr).max(0.0);
            let denom = vary_tr.max(1e-12).sqrt();
            out.push((
                cnt,
                (raw.sqrt() / denom) as f32,
                (proc.sqrt() / denom) as f32,
            ));
        }
        out
    }

    /// Closed-form fit → resident corrector. Depths with no samples get the
    /// identity rotation + unit scale + zero offset (apply is a no-op there).
    pub fn finalize(&self) -> ProcrustesCorrector {
        let d = self.hidden;
        let mut scale = vec![1.0f32; self.depth_count];
        let mut mean_x = vec![0.0f32; self.depth_count * d];
        let mut mean_y = vec![0.0f32; self.depth_count * d];
        let mut rot = vec![0.0f32; self.depth_count * d * d];
        // default rotations = identity
        for depth in 0..self.depth_count {
            let base = depth * d * d;
            for i in 0..d {
                rot[base + i * d + i] = 1.0;
            }
        }
        for depth in 1..=self.depth_count {
            let Some((m, mx, my, varx_tr, _vary_tr)) = self.centered(depth) else {
                continue;
            };
            let (r, s, _) = Self::fit_depth(&m, d, varx_tr);
            let vbase = (depth - 1) * d;
            let rbase = (depth - 1) * d * d;
            scale[depth - 1] = s as f32;
            for i in 0..d {
                mean_x[vbase + i] = mx[i] as f32;
                mean_y[vbase + i] = my[i] as f32;
            }
            for i in 0..d * d {
                rot[rbase + i] = r[i] as f32;
            }
        }
        ProcrustesCorrector {
            depth_count: self.depth_count,
            hidden: d,
            scale,
            mean_x,
            mean_y,
            rot,
        }
    }
}

/// Fitted per-depth scaled-rotation corrector. `rot` is `depth_count` stacked
/// row-major `d×d` orthogonal matrices; `scale`/`mean_x`/`mean_y` are per-depth.
#[derive(Clone)]
pub struct ProcrustesCorrector {
    depth_count: usize,
    hidden: usize,
    scale: Vec<f32>,
    mean_x: Vec<f32>,
    mean_y: Vec<f32>,
    rot: Vec<f32>,
}

const MPRC_MAGIC: &[u8; 4] = b"MPRC";

impl ProcrustesCorrector {
    pub fn depth_count(&self) -> usize {
        self.depth_count
    }
    pub fn hidden(&self) -> usize {
        self.hidden
    }
    /// Per-depth scalar scale (1-based depth → `scale[depth-1]`).
    pub fn scale(&self, depth: usize) -> f32 {
        self.scale
            .get(depth.saturating_sub(1))
            .copied()
            .unwrap_or(1.0)
    }
    /// Row-major `d×d` rotation slice for 1-based `depth` (empty if OOR).
    pub fn rotation(&self, depth: usize) -> &[f32] {
        if depth == 0 || depth > self.depth_count {
            return &[];
        }
        let base = (depth - 1) * self.hidden * self.hidden;
        &self.rot[base..base + self.hidden * self.hidden]
    }
    pub fn mean_x(&self, depth: usize) -> &[f32] {
        let base = (depth - 1) * self.hidden;
        &self.mean_x[base..base + self.hidden]
    }
    pub fn mean_y(&self, depth: usize) -> &[f32] {
        let base = (depth - 1) * self.hidden;
        &self.mean_y[base..base + self.hidden]
    }

    /// Host apply: `h ← s·(h - mean_x)·R + mean_y` in place at 1-based `depth`.
    /// `d×d` matvec — heavy (`hidden²` FMA); for accept A/B only. The serve
    /// path should apply `rot`/`scale` as a GPU matmul instead.
    pub fn apply(&self, h: &mut [f32], depth: usize) {
        if depth == 0 || depth > self.depth_count || h.len() != self.hidden {
            return;
        }
        let d = self.hidden;
        let s = self.scale[depth - 1];
        let mx = self.mean_x(depth);
        let my = self.mean_y(depth);
        let r = self.rotation(depth);
        // centered input
        let xc: Vec<f32> = (0..d).map(|i| h[i] - mx[i]).collect();
        // out_j = s · Σ_i xc_i · R[i,j] + my_j
        for j in 0..d {
            let mut acc = 0.0f32;
            for i in 0..d {
                acc += xc[i] * r[i * d + j];
            }
            h[j] = s * acc + my[j];
        }
    }

    /// Serialize: `MPRC` + depth(u32) + hidden(u32) + scale[depth] +
    /// mean_x[depth·d] + mean_y[depth·d] + rot[depth·d·d] (all f32 LE).
    pub fn to_bytes(&self) -> Vec<u8> {
        let d = self.hidden;
        let dc = self.depth_count;
        let mut out = Vec::with_capacity(12 + (dc + 2 * dc * d + dc * d * d) * 4);
        out.extend_from_slice(MPRC_MAGIC);
        out.extend_from_slice(&(dc as u32).to_le_bytes());
        out.extend_from_slice(&(d as u32).to_le_bytes());
        for &v in &self.scale {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.mean_x {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.mean_y {
            out.extend_from_slice(&v.to_le_bytes());
        }
        for &v in &self.rot {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Parse the [`to_bytes`] format. `None` on bad magic / truncation.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < 12 || &buf[0..4] != MPRC_MAGIC {
            return None;
        }
        let dc = u32::from_le_bytes(buf[4..8].try_into().ok()?) as usize;
        let d = u32::from_le_bytes(buf[8..12].try_into().ok()?) as usize;
        let n_scale = dc;
        let n_mean = dc * d;
        let n_rot = dc * d * d;
        let need = 12 + (n_scale + 2 * n_mean + n_rot) * 4;
        if buf.len() < need {
            return None;
        }
        let mut off = 12usize;
        let read = |count: usize, off: &mut usize| -> Vec<f32> {
            let v: Vec<f32> = (0..count)
                .map(|i| {
                    let o = *off + i * 4;
                    f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
                })
                .collect();
            *off += count * 4;
            v
        };
        let scale = read(n_scale, &mut off);
        let mean_x = read(n_mean, &mut off);
        let mean_y = read(n_mean, &mut off);
        let rot = read(n_rot, &mut off);
        Some(Self {
            depth_count: dc,
            hidden: d,
            scale,
            mean_x,
            mean_y,
            rot,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build a small known orthogonal rotation, generate y = s·x·R + b, and
    // confirm the fit recovers a corrector that maps a fresh x back onto y.
    #[test]
    fn recovers_known_rotation() {
        let d = 4;
        // 90° rotation in the (0,1) plane, identity on (2,3): row-major R.
        // y_row = x_row · R.
        let r_true = [
            0.0f32, 1.0, 0.0, 0.0, //
            -1.0, 0.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];
        let s_true = 1.0f32;
        let b = [0.5f32, -0.3, 0.2, 0.1];
        let apply_true = |x: &[f32]| -> Vec<f32> {
            (0..d)
                .map(|j| {
                    let mut acc = 0.0f32;
                    for i in 0..d {
                        acc += x[i] * r_true[i * d + j];
                    }
                    s_true * acc + b[j]
                })
                .collect()
        };
        let mut stats = ProcrustesStats::new(1, d);
        for row in 0..256 {
            let x: Vec<f32> = (0..d)
                .map(|c| ((row * 7 + c * 13) % 19) as f32 * 0.21 - 2.0)
                .collect();
            let y = apply_true(&x);
            stats.observe(1, &x, &y);
        }
        let rep = stats.residual_report();
        // Rotation must drive the residual to ~0 (it's an exact scaled rotation).
        assert!(rep[0].2 < 1e-2, "proc residual not small: {:?}", rep[0]);
        let corr = stats.finalize();
        let x = [1.0f32, -2.0, 3.0, 0.5];
        let want = apply_true(&x);
        let mut got = x;
        corr.apply(&mut got, 1);
        for c in 0..d {
            assert!(
                (got[c] - want[c]).abs() < 1e-2,
                "c{c}: {} vs {}",
                got[c],
                want[c]
            );
        }
    }

    #[test]
    fn empty_depth_is_identity() {
        let corr = ProcrustesStats::new(2, 3).finalize();
        let mut h = [1.0f32, 2.0, 3.0];
        let orig = h;
        corr.apply(&mut h, 1);
        assert_eq!(h, orig, "no samples → identity + zero offset = no-op");
    }

    #[test]
    fn roundtrip_serialize() {
        let mut stats = ProcrustesStats::new(1, 3);
        for row in 0..32 {
            let x = [row as f32, row as f32 * 0.5 - 1.0, 2.0 - row as f32];
            // y = rotate(x) by swapping 0<->2 + scale 1.5
            let y = [1.5 * x[2], 1.5 * x[1], 1.5 * x[0]];
            stats.observe(1, &x, &y);
        }
        let corr = stats.finalize();
        let bytes = corr.to_bytes();
        let back = ProcrustesCorrector::from_bytes(&bytes).expect("parse");
        let mut a = [3.0f32, 4.0, 5.0];
        let mut b = a;
        corr.apply(&mut a, 1);
        back.apply(&mut b, 1);
        assert_eq!(a, b);
    }
}
