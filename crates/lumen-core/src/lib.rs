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

/// Scratch paths for the save/load round-trip tests.
///
/// Three of them used to write to a fixed name under `/tmp` — `tq_codebook_test.bin`
/// and friends. Two checkouts of this repo testing at once, or one `cargo test`
/// racing a `cargo xtask gate`, write the same file and read back each other's
/// bytes; the failure is a corrupted-looking codebook in whichever process
/// loses, and it does not reproduce when you run that test alone.
///
/// `TempPath` gives each writer a name unique to the process and the call site,
/// and deletes it on drop so a panicking assert does not leave the file behind
/// for the next run to trip over.
#[cfg(test)]
pub(crate) mod testpath {
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU32, Ordering};

    static NEXT: AtomicU32 = AtomicU32::new(0);

    /// A unique path under the system temp dir, removed when dropped.
    pub(crate) struct TempPath(PathBuf);

    impl TempPath {
        pub(crate) fn new(stem: &str) -> Self {
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            Self(
                std::env::temp_dir()
                    .join(format!("lumen-core-{stem}-{}-{n}.bin", std::process::id())),
            )
        }

        pub(crate) fn as_str(&self) -> &str {
            self.0.to_str().expect("temp dir is not valid UTF-8")
        }

        pub(crate) fn path(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TempPath {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn two_temp_paths_never_collide() {
        let a = TempPath::new("same");
        let b = TempPath::new("same");
        assert_ne!(
            a.path(),
            b.path(),
            "same stem must still yield distinct paths"
        );
    }

    #[test]
    fn the_file_is_gone_after_drop() {
        let p = {
            let t = TempPath::new("dropme");
            std::fs::write(t.path(), b"x").expect("write scratch file");
            assert!(t.path().exists());
            t.path().to_path_buf()
        };
        assert!(!p.exists(), "TempPath must clean up on drop");
    }
}
