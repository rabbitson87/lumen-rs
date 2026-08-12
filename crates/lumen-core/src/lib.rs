/// `always!()` / `never!()` — defensive conditions that leave the coverage
/// denominator. See the module docs for the three-build behaviour.
pub mod defensive;

pub mod bitpack;
pub mod compressor;
pub mod config;
pub mod dry;
pub mod lloyd_max;
pub mod mtp_corrector;
pub mod mtp_procrustes;
pub mod qjl;
pub mod rotation;
pub mod runaway;
pub mod sampling;
pub mod stop;
