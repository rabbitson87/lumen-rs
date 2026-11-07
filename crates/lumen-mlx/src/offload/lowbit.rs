//! A cheaper copy of the routed experts: the same affine layout at fewer bits per weight.
//!
//! A checkpoint's expert is `scale * code + bias` per group of weights. Here each group of the
//! source is decoded back to floats and fitted again at `bits` bits. The source is the
//! checkpoint's own reconstruction, not the float original: what the copy has to agree with is
//! the model as it is served.
//!
//! A min-max fit spreads the levels from the group's smallest weight to its largest, so a single
//! outlier decides the step and at two bits two of the four levels go to the extremes. The
//! least-squares fit lets the tails clip: it alternates between assigning codes and solving
//! `(scale, bias)` for those codes, which cannot raise the squared error, and starts from both
//! the min-max fit and one sized for the group's spread.
//!
//! This is lossy by design, and it is not what the exact path serves.

use anyhow::{Result, bail};

/// How a group's `(scale, bias)` are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Recipe {
    /// From the group's smallest weight to its largest.
    MinMax,
    /// Alternating least squares.
    LeastSquares,
}

impl Recipe {
    pub fn name(self) -> &'static str {
        match self {
            Self::MinMax => "minmax",
            Self::LeastSquares => "lsq",
        }
    }

    pub fn parse(raw: &str) -> Result<Self> {
        match raw {
            "minmax" => Ok(Self::MinMax),
            "lsq" => Ok(Self::LeastSquares),
            other => bail!("recipe {other:?} is not `minmax` or `lsq`"),
        }
    }
}

/// A low-bit copy's quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LowbitSpec {
    pub bits: u32,
    pub group_size: usize,
    pub recipe: Recipe,
}

impl LowbitSpec {
    /// `bits,group,recipe`, like `3,64,lsq`.
    pub fn parse(raw: &str) -> Result<Self> {
        let parts: Vec<&str> = raw.trim().split(',').map(str::trim).collect();
        let [bits, group, recipe] = parts.as_slice() else {
            bail!("{raw:?} is not bits,group,recipe, like 3,64,lsq");
        };
        let number = |what: &str, v: &str| -> Result<usize> {
            v.parse()
                .map_err(|_| anyhow::anyhow!("{raw:?}: {what} {v:?} is not a number"))
        };
        Self::new(
            number("bits", bits)? as u32,
            number("group", group)?,
            Recipe::parse(recipe)?,
        )
    }

    pub fn new(bits: u32, group_size: usize, recipe: Recipe) -> Result<Self> {
        if !(2..=8).contains(&bits) {
            bail!("{bits} bits is outside what the affine kernels take (2 to 8)");
        }
        if group_size == 0 {
            bail!("a group of 0 weights");
        }
        Ok(Self {
            bits,
            group_size,
            recipe,
        })
    }
}

/// The float type of a checkpoint's scales and biases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScaleType {
    Bf16,
    F32,
}

impl ScaleType {
    pub(crate) fn parse(safetensors: &str) -> Result<Self> {
        match safetensors {
            "BF16" => Ok(Self::Bf16),
            "F32" => Ok(Self::F32),
            other => bail!("scales of type {other} are not supported"),
        }
    }

    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }

    /// `x` as this type stores it.
    fn held(self, x: f32) -> f32 {
        match self {
            Self::Bf16 => bf16_to_f32(f32_to_bf16(x)),
            Self::F32 => x,
        }
    }

    fn read(self, b: &[u8]) -> f32 {
        match self {
            Self::Bf16 => bf16_to_f32(u16::from_le_bytes(b.try_into().expect("2 bytes"))),
            Self::F32 => f32::from_le_bytes(b.try_into().expect("4 bytes")),
        }
    }

    fn write(self, x: f32, to: &mut [u8]) {
        match self {
            Self::Bf16 => to.copy_from_slice(&f32_to_bf16(x).to_le_bytes()),
            Self::F32 => to.copy_from_slice(&x.to_le_bytes()),
        }
    }
}

/// A projection of one expert, as packed `u32` words and floats for its scales and biases, little
/// endian.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PackedProj<'a> {
    pub weight: &'a [u8],
    pub scales: &'a [u8],
    pub biases: &'a [u8],
    pub scale_type: ScaleType,
    /// Rows of the weight matrix.
    pub rows: usize,
    /// Weights in a row.
    pub cols: usize,
    pub bits: u32,
    pub group_size: usize,
}

/// Where a projection's copy goes.
#[derive(Debug)]
pub(crate) struct PackedProjMut<'a> {
    pub weight: &'a mut [u8],
    pub scales: &'a mut [u8],
    pub biases: &'a mut [u8],
}

/// Squared error of a copy against its source, and the source's energy: `10 log10(energy / err)`
/// is the copy's signal-to-noise ratio.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct Fidelity {
    pub err: f64,
    pub energy: f64,
}

impl Fidelity {
    pub(crate) fn add(&mut self, other: Fidelity) {
        self.err += other.err;
        self.energy += other.energy;
    }

    pub(crate) fn snr_db(&self) -> f64 {
        10.0 * (self.energy / self.err.max(f64::MIN_POSITIVE)).log10()
    }
}

/// Bytes of a packed row of `cols` weights at `bits` bits (whole `u32` words).
pub(crate) fn row_bytes(cols: usize, bits: u32) -> Result<usize> {
    let total = cols * bits as usize;
    if !total.is_multiple_of(32) {
        bail!("{cols} weights at {bits} bits do not fill whole 32-bit words");
    }
    Ok(total / 8)
}

fn bf16_to_f32(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// To the nearest `bf16`, ties to even.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

/// Code `i` of a stream packed low bits first, the order MLX reads.
#[inline]
fn read_code(words: &[u32], i: usize, bits: u32) -> u32 {
    let bit = i * bits as usize;
    let (word, off) = (bit / 32, (bit % 32) as u32);
    let mut v = words[word] >> off;
    if off + bits > 32 {
        v |= words[word + 1] << (32 - off);
    }
    v & ((1u32 << bits) - 1)
}

#[inline]
fn write_code(words: &mut [u32], i: usize, bits: u32, code: u32) {
    let bit = i * bits as usize;
    let (word, off) = (bit / 32, (bit % 32) as u32);
    words[word] |= code << off;
    if off + bits > 32 {
        words[word + 1] |= code >> (32 - off);
    }
}

#[inline]
fn code_of(x: f32, scale: f32, bias: f32, top: f32) -> f32 {
    if scale > 0.0 {
        ((x - bias) / scale).round().clamp(0.0, top)
    } else {
        0.0
    }
}

fn squared_error(xs: &[f32], scale: f32, bias: f32, top: f32) -> f64 {
    xs.iter()
        .map(|&x| {
            let d = x - (scale * code_of(x, scale, bias, top) + bias);
            f64::from(d * d)
        })
        .sum()
}

/// `(scale, bias)` from the group's smallest weight to its largest.
fn fit_min_max(xs: &[f32], top: f32) -> (f32, f32) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &x in xs {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    ((hi - lo) / top, lo)
}

/// The step of a uniform quantizer of a unit Gaussian with this many levels, the distortion
/// minimum (Jayant & Noll, table 4.1).
fn gaussian_step(levels: u32) -> f32 {
    match levels {
        2 => 1.596,
        4 => 0.996,
        8 => 0.586,
        16 => 0.335,
        32 => 0.188,
        _ => 4.0 / levels as f32,
    }
}

/// `(scale, bias)` centred on the group, stepped for its spread.
fn fit_spread(xs: &[f32], top: f32) -> (f32, f32) {
    let n = xs.len() as f32;
    let mean = xs.iter().sum::<f32>() / n;
    let var = xs.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / n;
    let scale = var.sqrt() * gaussian_step(top as u32 + 1);
    (scale, mean - scale * top * 0.5)
}

/// Rounds of assign-then-solve. Two levels' worth of a 64-weight group settles in a few.
const REFINE_ROUNDS: usize = 8;

/// Alternate between the codes for `(scale, bias)` and the least-squares `(scale, bias)` for those
/// codes, until the codes stop changing.
fn refine(xs: &[f32], top: f32, start: (f32, f32), codes: &mut [f32]) -> (f32, f32) {
    let (mut scale, mut bias) = start;
    for round in 0..REFINE_ROUNDS {
        let mut changed = round == 0;
        for (c, &x) in codes.iter_mut().zip(xs) {
            let q = code_of(x, scale, bias, top);
            changed |= *c != q;
            *c = q;
        }
        if !changed {
            break;
        }
        let n = xs.len() as f64;
        let (mut sq, mut sx, mut sqq, mut sqx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for (&q, &x) in codes.iter().zip(xs) {
            let (q, x) = (f64::from(q), f64::from(x));
            sq += q;
            sx += x;
            sqq += q * q;
            sqx += q * x;
        }
        let d = n * sqq - sq * sq;
        if d <= 0.0 {
            // Every weight took one code: nothing to fit a slope to.
            return (0.0, (sx / n) as f32);
        }
        scale = ((n * sqx - sq * sx) / d) as f32;
        bias = ((sx - f64::from(scale) * sq) / n) as f32;
    }
    (scale, bias)
}

/// A group's `(scale, bias)`, as the values the kernels will read.
fn fit_group(
    xs: &[f32],
    top: f32,
    recipe: Recipe,
    scale_type: ScaleType,
    codes: &mut [f32],
) -> (f32, f32) {
    let held = |(s, b): (f32, f32)| (scale_type.held(s), scale_type.held(b));
    let min_max = fit_min_max(xs, top);
    match recipe {
        Recipe::MinMax => held(min_max),
        Recipe::LeastSquares => {
            let spread = fit_spread(xs, top);
            // A candidate is judged as stored: that is what gets served.
            [min_max, spread]
                .into_iter()
                .flat_map(|start| [held(start), held(refine(xs, top, start, codes))])
                .map(|c| (c, squared_error(xs, c.0, c.1, top)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map_or((0.0, 0.0), |(c, _)| c)
        }
    }
}

fn words_of(bytes: &[u8]) -> impl Iterator<Item = u32> + '_ {
    bytes
        .chunks_exact(4)
        .map(|w| u32::from_le_bytes(w.try_into().expect("4 bytes")))
}

fn floats_of(bytes: &[u8], t: ScaleType) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(t.bytes()).map(move |v| t.read(v))
}

/// Fit one projection of one expert again at `spec`, into `dst`.
pub(crate) fn requantize(
    src: PackedProj<'_>,
    spec: LowbitSpec,
    dst: PackedProjMut<'_>,
) -> Result<Fidelity> {
    let src_row = row_bytes(src.cols, src.bits)?;
    let dst_row = row_bytes(src.cols, spec.bits)?;
    if !src.cols.is_multiple_of(src.group_size) || !src.cols.is_multiple_of(spec.group_size) {
        bail!(
            "{} weights do not divide into groups of {} and {}",
            src.cols,
            src.group_size,
            spec.group_size
        );
    }
    let t = src.scale_type;
    let src_groups = src.cols / src.group_size;
    let dst_groups = src.cols / spec.group_size;
    if src.weight.len() != src.rows * src_row
        || src.scales.len() != src.rows * src_groups * t.bytes()
        || src.biases.len() != src.rows * src_groups * t.bytes()
    {
        bail!("the source does not match {} x {}", src.rows, src.cols);
    }
    if dst.weight.len() != src.rows * dst_row
        || dst.scales.len() != src.rows * dst_groups * t.bytes()
        || dst.biases.len() != src.rows * dst_groups * t.bytes()
    {
        bail!("the destination does not match {} x {}", src.rows, src.cols);
    }

    let top = ((1u32 << spec.bits) - 1) as f32;
    let mut fidelity = Fidelity::default();
    let mut src_words = vec![0u32; src_row / 4];
    let mut dst_words = vec![0u32; dst_row / 4];
    let mut row = vec![0f32; src.cols];
    let mut codes = vec![0f32; spec.group_size];

    for r in 0..src.rows {
        let weight = &src.weight[r * src_row..(r + 1) * src_row];
        for (w, v) in src_words.iter_mut().zip(words_of(weight)) {
            *w = v;
        }
        let (at, len) = (r * src_groups * t.bytes(), src_groups * t.bytes());
        let scales = floats_of(&src.scales[at..at + len], t);
        let biases = floats_of(&src.biases[at..at + len], t);
        for (g, (s, b)) in scales.zip(biases).enumerate() {
            let start = g * src.group_size;
            for (i, x) in row[start..start + src.group_size].iter_mut().enumerate() {
                *x = s * read_code(&src_words, start + i, src.bits) as f32 + b;
            }
        }

        dst_words.fill(0);
        for (g, xs) in row.chunks_exact(spec.group_size).enumerate() {
            let (fs, fb) = fit_group(xs, top, spec.recipe, t, &mut codes);
            for (i, &x) in xs.iter().enumerate() {
                let q = code_of(x, fs, fb, top);
                write_code(&mut dst_words, g * spec.group_size + i, spec.bits, q as u32);
                let d = f64::from(x) - (f64::from(fs) * f64::from(q) + f64::from(fb));
                fidelity.err += d * d;
                fidelity.energy += f64::from(x) * f64::from(x);
            }
            let at = (r * dst_groups + g) * t.bytes();
            t.write(fs, &mut dst.scales[at..at + t.bytes()]);
            t.write(fb, &mut dst.biases[at..at + t.bytes()]);
        }
        for (i, w) in dst_words.iter().enumerate() {
            let at = r * dst_row + i * 4;
            dst.weight[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
    }
    Ok(fidelity)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Xorshift, so these tests need no RNG crate and no MLX.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }

        /// A unit Gaussian (Box-Muller).
        fn gauss(&mut self) -> f32 {
            let (u, v) = (self.next().max(1e-6), self.next());
            (-2.0 * u.ln()).sqrt() * (std::f32::consts::TAU * v).cos()
        }
    }

    fn lsq(bits: u32, gs: usize) -> LowbitSpec {
        LowbitSpec::new(bits, gs, Recipe::LeastSquares).unwrap()
    }

    #[test]
    fn a_spec_reads_as_bits_group_recipe() {
        let spec = LowbitSpec::parse(" 3, 64 ,lsq ").unwrap();
        assert_eq!(spec, lsq(3, 64));
        for bad in [
            "",
            "3",
            "3,64",
            "3,64,lsq,1",
            "x,64,lsq",
            "3,y,lsq",
            "3,64,best",
            "1,64,lsq",
        ] {
            assert!(LowbitSpec::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn codes_are_packed_low_bits_first_across_word_boundaries() {
        // Two 4-bit codes share a word from the bottom up.
        let mut w = [0u32; 1];
        write_code(&mut w, 0, 4, 0xa);
        write_code(&mut w, 1, 4, 0x5);
        assert_eq!(w[0], 0x5a);
        // The eleventh 3-bit code starts at bit 30 and runs into the next word.
        let mut w = [0u32; 2];
        write_code(&mut w, 10, 3, 0b101);
        assert_eq!(w, [0b01 << 30, 0b1]);
        assert_eq!(read_code(&w, 10, 3), 0b101);
    }

    #[test]
    fn every_code_of_every_width_reads_back() {
        for bits in [2, 3, 4, 5, 6, 8] {
            let n = 32 * 8;
            let mut w = vec![0u32; n * bits as usize / 32];
            let code = |i: usize| (i as u32).wrapping_mul(2_654_435_761) >> (32 - bits);
            for i in 0..n {
                write_code(&mut w, i, bits, code(i));
            }
            for i in 0..n {
                assert_eq!(read_code(&w, i, bits), code(i), "bits {bits} code {i}");
            }
        }
    }

    #[test]
    fn bf16_rounds_to_nearest_and_keeps_what_it_can_hold() {
        for x in [0.0f32, 1.0, -1.0, 0.5, 1024.0, -0.00390625] {
            assert_eq!(bf16_to_f32(f32_to_bf16(x)), x);
        }
        // 1 + 2^-7 is the next bf16 after 1: a bit past the midpoint rounds up, a bit short down.
        let step = 1.0f32 / 128.0;
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + step * 0.51)), 1.0 + step);
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + step * 0.49)), 1.0);
    }

    #[test]
    fn least_squares_never_loses_to_min_max_and_gains_most_on_heavy_tails() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut codes = vec![0f32; 64];
        let (mut gauss, mut tails) = (Fidelity::default(), Fidelity::default());
        for trial in 0..400 {
            let heavy = trial % 2 == 1;
            let xs: Vec<f32> = (0..64)
                .map(|i| {
                    let x = rng.gauss() * 0.01;
                    if heavy && i % 31 == 0 { x * 8.0 } else { x }
                })
                .collect();
            let mut err = |r| {
                let (s, b) = fit_group(&xs, 3.0, r, ScaleType::Bf16, &mut codes);
                squared_error(&xs, s, b, 3.0)
            };
            let (mm, ls) = (err(Recipe::MinMax), err(Recipe::LeastSquares));
            assert!(ls <= mm, "trial {trial}: least squares {ls} > min-max {mm}");
            let t = if heavy { &mut tails } else { &mut gauss };
            t.err += ls;
            t.energy += mm;
        }
        // Here `energy` holds the min-max error: the ratio is the gain in dB.
        assert!(gauss.snr_db() > 1.0, "Gaussian gain {}", gauss.snr_db());
        assert!(tails.snr_db() > 4.0, "heavy-tail gain {}", tails.snr_db());
    }

    /// A random expert projection at 4 bits, `rows` x `cols`, groups of `gs`.
    fn source(rows: usize, cols: usize, gs: usize, seed: u64) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut rng = Rng(seed);
        let groups = cols / gs;
        let mut weight = vec![0u8; rows * cols / 2];
        for b in &mut weight {
            *b = (rng.next() * 256.0) as u8;
        }
        let bf = |x: f32| f32_to_bf16(x).to_le_bytes();
        let scales: Vec<u8> = (0..rows * groups).flat_map(|_| bf(0.002)).collect();
        let biases: Vec<u8> = (0..rows * groups).flat_map(|_| bf(-0.015)).collect();
        (weight, scales, biases)
    }

    fn proj<'a>(
        (w, s, b): &'a (Vec<u8>, Vec<u8>, Vec<u8>),
        rows: usize,
        cols: usize,
        gs: usize,
    ) -> PackedProj<'a> {
        PackedProj {
            weight: w,
            scales: s,
            biases: b,
            scale_type: ScaleType::Bf16,
            rows,
            cols,
            bits: 4,
            group_size: gs,
        }
    }

    fn requantized(src: PackedProj<'_>, spec: LowbitSpec) -> (Vec<u8>, Vec<u8>, Vec<u8>, Fidelity) {
        let groups = src.cols / spec.group_size;
        let mut w = vec![0u8; src.rows * row_bytes(src.cols, spec.bits).unwrap()];
        let mut s = vec![0u8; src.rows * groups * 2];
        let mut b = vec![0u8; src.rows * groups * 2];
        let f = requantize(
            src,
            spec,
            PackedProjMut {
                weight: &mut w,
                scales: &mut s,
                biases: &mut b,
            },
        )
        .unwrap();
        (w, s, b, f)
    }

    #[test]
    fn a_copy_at_the_source_width_loses_little() {
        let src = source(4, 128, 32, 11);
        let (_, _, _, f) = requantized(proj(&src, 4, 128, 32), lsq(4, 32));
        // Not exact: a group that misses a level of its range is fitted on a slightly other grid.
        assert!(f.snr_db() > 30.0, "{} dB", f.snr_db());
    }

    /// What MLX itself reads out of packed words, scales and biases.
    fn mlx_dequantized(
        (w, s, b): (&[u8], &[u8], &[u8]),
        rows: usize,
        cols: usize,
        gs: usize,
        bits: u32,
    ) -> Vec<f32> {
        use crate::native_quant::{MODE_AFFINE, dequantize_with_mode};
        use mlx_rs::{Array, Dtype};
        let words: Vec<u32> = words_of(w).collect();
        let groups = (cols / gs) as i32;
        let w = Array::from_slice(&words, &[rows as i32, (words.len() / rows) as i32]);
        let bf = |b: &[u8]| {
            let v: Vec<f32> = floats_of(b, ScaleType::Bf16).collect();
            Array::from_slice(&v, &[rows as i32, groups])
                .as_dtype(Dtype::Bfloat16)
                .unwrap()
        };
        let (s, b) = (bf(s), bf(b));
        let x = dequantize_with_mode(&w, &s, Some(&b), gs as i32, bits as i32, MODE_AFFINE)
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
        x.eval().unwrap();
        x.as_slice::<f32>().to_vec()
    }

    /// This file's own reading of packed words, scales and biases.
    fn own_reading(
        (w, s, b): (&[u8], &[u8], &[u8]),
        cols: usize,
        gs: usize,
        bits: u32,
    ) -> Vec<f32> {
        let words: Vec<u32> = words_of(w).collect();
        let row_words = cols * bits as usize / 32;
        let scales: Vec<f32> = floats_of(s, ScaleType::Bf16).collect();
        let biases: Vec<f32> = floats_of(b, ScaleType::Bf16).collect();
        let groups = cols / gs;
        let mut out = Vec::new();
        for (r, row) in words.chunks_exact(row_words).enumerate() {
            for i in 0..cols {
                let g = r * groups + i / gs;
                out.push(scales[g] * read_code(row, i, bits) as f32 + biases[g]);
            }
        }
        out
    }

    /// Within what `bf16` can tell apart: MLX computes in the scale's dtype.
    fn same_value(a: f32, b: f32) -> bool {
        (a - b).abs() <= a.abs().max(b.abs()).max(1e-4) / 128.0
    }

    #[test]
    fn mlx_reads_every_weight_where_this_file_wrote_it() {
        // `Fidelity` compares a copy with this file's own reading of the source, so a wrong bit
        // order would score perfectly. Only MLX's reading can catch that.
        let (rows, cols, gs) = (8, 256, 64);
        let src = source(rows, cols, gs, 14);
        let parts = (&src.0[..], &src.1[..], &src.2[..]);
        let mlx = mlx_dequantized(parts, rows, cols, gs, 4);
        let own = own_reading(parts, cols, gs, 4);
        assert_eq!(mlx.len(), own.len());
        for (i, (m, o)) in mlx.iter().zip(&own).enumerate() {
            assert!(same_value(*m, *o), "source weight {i}: MLX {m}, here {o}");
        }
        for bits in [2, 3] {
            let (w, s, b, f) = requantized(proj(&src, rows, cols, gs), lsq(bits, gs));
            let parts = (&w[..], &s[..], &b[..]);
            let mlx = mlx_dequantized(parts, rows, cols, gs, bits);
            let copy = own_reading(parts, cols, gs, bits);
            assert_eq!(mlx.len(), copy.len());
            for (i, (m, o)) in mlx.iter().zip(&copy).enumerate() {
                assert!(
                    same_value(*m, *o),
                    "{bits} bits, weight {i}: MLX {m}, here {o}"
                );
            }
            // The error this file reports is the error between the two models MLX would serve.
            let err: f64 = own
                .iter()
                .zip(&copy)
                .map(|(x, y)| f64::from(x - y) * f64::from(x - y))
                .sum();
            assert!((err - f.err).abs() <= 1e-6 * f.err, "{err} vs {}", f.err);
        }
    }

    #[test]
    fn fewer_bits_cost_about_what_they_should() {
        let src = source(16, 256, 64, 12);
        let snr = |bits| {
            requantized(proj(&src, 16, 256, 64), lsq(bits, 64))
                .3
                .snr_db()
        };
        let (two, three) = (snr(2), snr(3));
        assert!(three > two + 3.0, "3 bits {three} dB, 2 bits {two} dB");
        assert!(two > 3.0, "2 bits {two} dB");
    }

    #[test]
    fn a_constant_group_is_reproduced_exactly() {
        // All weights equal: code 0 of every word, one scale and bias.
        let rows = 2;
        let cols = 64;
        let src = (
            vec![0u8; rows * cols / 2],
            f32_to_bf16(0.25)
                .to_le_bytes()
                .repeat(rows * 2)
                .into_iter()
                .collect::<Vec<u8>>(),
            f32_to_bf16(0.5)
                .to_le_bytes()
                .repeat(rows * 2)
                .into_iter()
                .collect::<Vec<u8>>(),
        );
        let (_, _, _, f) = requantized(proj(&src, rows, cols, 32), lsq(2, 64));
        assert_eq!(f.err, 0.0);
        assert!(f.energy > 0.0);
    }

    #[test]
    fn f32_scales_stay_f32_and_fit_at_least_as_well() {
        let (w, s, b) = source(4, 128, 32, 15);
        let widen = |bytes: &[u8]| -> Vec<u8> {
            floats_of(bytes, ScaleType::Bf16)
                .flat_map(f32::to_le_bytes)
                .collect()
        };
        let (s32, b32) = (widen(&s), widen(&b));
        let src = PackedProj {
            weight: &w,
            scales: &s32,
            biases: &b32,
            scale_type: ScaleType::F32,
            rows: 4,
            cols: 128,
            bits: 4,
            group_size: 32,
        };
        let mut out = (
            vec![0u8; 4 * 32],
            vec![0u8; 4 * 4 * 4],
            vec![0u8; 4 * 4 * 4],
        );
        let f = requantize(
            src,
            lsq(2, 32),
            PackedProjMut {
                weight: &mut out.0,
                scales: &mut out.1,
                biases: &mut out.2,
            },
        )
        .unwrap();
        let bf = requantized(proj(&(w.clone(), s, b), 4, 128, 32), lsq(2, 32)).3;
        // Scales that are not rounded to bf16 cannot fit worse, beyond float noise.
        assert!(f.err <= bf.err * 1.001, "{} vs {}", f.err, bf.err);
    }

    #[test]
    fn shapes_that_cannot_be_packed_are_errors() {
        assert!(row_bytes(100, 3).is_err());
        assert_eq!(row_bytes(2560, 2).unwrap(), 640);
        assert!(LowbitSpec::new(1, 64, Recipe::MinMax).is_err());
        assert!(LowbitSpec::new(9, 64, Recipe::MinMax).is_err());
        assert!(LowbitSpec::new(2, 0, Recipe::MinMax).is_err());
        let src = source(1, 128, 32, 13);
        let mut d = (vec![0u8; 8], vec![0u8; 2], vec![0u8; 2]);
        let r = requantize(
            proj(&src, 1, 128, 32),
            lsq(2, 64),
            PackedProjMut {
                weight: &mut d.0,
                scales: &mut d.1,
                biases: &mut d.2,
            },
        );
        assert!(r.is_err());
    }
}
