//! The binary layout the codebook, rotation and QJL tables are saved in.
//!
//! Every file is a 4-byte magic, a few `u32` header fields, then flat arrays
//! of floats — all little-endian, nothing between them. Each table chooses its
//! own magic, header fields and float width; only the encoding is shared.

use std::io::{self, Read, Write};

/// Write `magic`, then each of `fields` as a little-endian `u32`.
pub(crate) fn write_header(w: &mut impl Write, magic: &[u8; 4], fields: &[u32]) -> io::Result<()> {
    w.write_all(magic)?;
    for &v in fields {
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}

/// Read a header written by [`write_header`]: `None` when the file does not
/// start with `magic`, else its `N` header fields.
pub(crate) fn read_header<const N: usize>(
    r: &mut impl Read,
    magic: &[u8; 4],
) -> io::Result<Option<[u32; N]>> {
    let mut found = [0u8; 4];
    r.read_exact(&mut found)?;
    if &found != magic {
        return Ok(None);
    }
    let mut fields = [0u32; N];
    let mut buf4 = [0u8; 4];
    for v in &mut fields {
        r.read_exact(&mut buf4)?;
        *v = u32::from_le_bytes(buf4);
    }
    Ok(Some(fields))
}

pub(crate) fn write_f32s(w: &mut impl Write, xs: &[f32]) -> io::Result<()> {
    for &x in xs {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

pub(crate) fn read_f32s(r: &mut impl Read, n: usize) -> io::Result<Vec<f32>> {
    let mut buf4 = [0u8; 4];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut buf4)?;
        out.push(f32::from_le_bytes(buf4));
    }
    Ok(out)
}

pub(crate) fn write_f64s(w: &mut impl Write, xs: &[f64]) -> io::Result<()> {
    for &x in xs {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

pub(crate) fn read_f64s(r: &mut impl Read, n: usize) -> io::Result<Vec<f64>> {
    let mut buf8 = [0u8; 8];
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        r.read_exact(&mut buf8)?;
        out.push(f64::from_le_bytes(buf8));
    }
    Ok(out)
}
