//! FLUX.2-klein text-to-image denoise loop (pipeline glue), Phase 3.
//!
//! Wires the parity-validated DiT ([`crate::dit`]) + VAE ([`crate::vae`]) into a
//! full image-generation loop:
//!
//! ```text
//! for t in 0..steps:
//!     noise   = dit.forward(latents, text, timesteps[t], img_ids, txt_ids)
//!     latents = latents + (sigmas[t+1] - sigmas[t]) * noise   # Euler
//! packed_cf = latents.reshape(B,lh,lw,128).transpose(0,3,1,2)
//! image     = vae.decode_packed_latents(packed_cf)
//! ```
//!
//! The caller supplies the prompt embeds, init latents, ids, and the (already
//! computed + verified) schedule — this module owns only the loop + decode, so
//! the glue is isolated from the text encoder and RNG.

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result, bail};
    use mlx_rs::Array;

    use crate::dit::{Dit, DitConfig};
    use crate::scheduler::Schedule;
    use crate::vae::VaeDecoder;

    /// Loaded klein pipeline: DiT + VAE decoder.
    pub struct KleinPipeline {
        dit: Dit,
        vae: VaeDecoder,
    }

    impl KleinPipeline {
        /// Load DiT + VAE from their diffusers BF16 safetensors.
        pub fn load(dit_path: &str, vae_path: &str) -> Result<Self> {
            let dit = Dit::load(dit_path, DitConfig::klein_4b()).context("load DiT")?;
            let vae = VaeDecoder::load(vae_path).context("load VAE")?;
            Ok(Self { dit, vae })
        }

        /// Run the denoise loop and return both the per-step latents (for parity
        /// debugging) and the final decoded image `[B, Ho, Wo, 3]`.
        ///
        /// - `prompt_embeds`: `[1, txt_seq, 7680]`
        /// - `text_ids`/`img_ids`: row-major `[seq, 4]` i32
        /// - `init_latents`: packed `[1, img_seq, 128]`
        /// - `schedule`: verified sigmas/timesteps
        #[allow(clippy::too_many_arguments)]
        pub fn generate_with_trace(
            &self,
            prompt_embeds: &Array,
            text_ids: &[i32],
            init_latents: &Array,
            img_ids: &[i32],
            schedule: &Schedule,
            lat_h: i32,
            lat_w: i32,
        ) -> Result<(Vec<Array>, Array)> {
            let steps = schedule.timesteps.len();
            if steps == 0 {
                bail!("empty schedule");
            }
            let sh = init_latents.shape().to_vec();
            if sh.len() != 3 || sh[2] != 128 {
                bail!("init_latents expects [1,img_seq,128], got {sh:?}");
            }
            let b = sh[0];

            let mut latents = init_latents.clone();
            let mut per_step: Vec<Array> = Vec::with_capacity(steps);

            for t in 0..steps {
                let timestep = schedule.timesteps[t];
                // DiT noise prediction. timestep is already ×1000-scaled
                // (>1), so the DiT's "scale if <=1" branch is a no-op.
                let noise = self
                    .dit
                    .forward(&latents, prompt_embeds, timestep, None, img_ids, text_ids)
                    .with_context(|| format!("DiT forward step {t}"))?;

                // Euler step: latents += dt * noise.
                let dt = schedule.dt(t);
                let dt_arr = Array::from_slice(&[dt], &[1]);
                latents = noise
                    .multiply(&dt_arr)
                    .and_then(|d| latents.add(&d))
                    .with_context(|| format!("euler step {t}"))?;
                latents
                    .eval()
                    .with_context(|| format!("eval latents step {t}"))?;
                per_step.push(latents.clone());
            }

            // Final reshape → channels-first packed [B,128,lh,lw], then decode.
            let packed_cf = latents
                .reshape(&[b, lat_h, lat_w, 128])
                .and_then(|x| x.transpose_axes(&[0, 3, 1, 2]))
                .context("final pack reshape failed")?;
            let image = self
                .vae
                .decode_packed_latents(&packed_cf)
                .context("decode_packed_latents failed")?;
            image.eval().context("eval final image")?;
            Ok((per_step, image))
        }

        /// Run the loop and return only the final decoded image.
        #[allow(clippy::too_many_arguments)]
        pub fn generate(
            &self,
            prompt_embeds: &Array,
            text_ids: &[i32],
            init_latents: &Array,
            img_ids: &[i32],
            schedule: &Schedule,
            lat_h: i32,
            lat_w: i32,
        ) -> Result<Array> {
            let (_steps, image) = self.generate_with_trace(
                prompt_embeds,
                text_ids,
                init_latents,
                img_ids,
                schedule,
                lat_h,
                lat_w,
            )?;
            Ok(image)
        }
    }
}

#[cfg(feature = "mlx-native")]
pub use imp::KleinPipeline;

// ── end-to-end parity test against the mflux klein dump (/tmp/klein_*.bin) ────
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::KleinPipeline;
    use crate::scheduler;
    use mlx_rs::Array;

    // Klein DiT + VAE (bf16), resolved from the local HF cache.
    fn dit_path() -> String {
        crate::hf_cache::snapshot_path_or_rel(
            crate::repos::VAE,
            "transformer/diffusion_pytorch_model.safetensors",
        )
        .to_string_lossy()
        .into_owned()
    }
    fn vae_path() -> String {
        crate::hf_cache::snapshot_path_or_rel(
            crate::repos::VAE,
            "vae/diffusion_pytorch_model.safetensors",
        )
        .to_string_lossy()
        .into_owned()
    }

    fn read_f32_bin(path: &str) -> Vec<f32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not f32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_i32_bin(path: &str) -> Vec<i32> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(bytes.len() % 4, 0, "{path} not i32-aligned");
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_dims() -> (i32, i32, i32, i32) {
        let d = read_i32_bin("/tmp/klein_dims.bin");
        (d[0], d[1], d[2], d[3]) // lat_h, lat_w, img_seq, txt_seq
    }

    use lumen_testkit::cosine;

    /// Reorder channels-first `[B,C,H,W]` flat → channels-last `[B,H,W,C]`.
    fn cf_to_cl(cf: &[f32], b: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; cf.len()];
        for bi in 0..b {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        let src = ((bi * c + ci) * h + hi) * w + wi;
                        let dst = ((bi * h + hi) * w + wi) * c + ci;
                        out[dst] = cf[src];
                    }
                }
            }
        }
        out
    }

    /// Standalone VAE packed-decode parity: feed mflux's FINAL packed latents,
    /// compare decoded image to mflux's image.
    #[test]
    #[ignore = "needs a Metal device AND /tmp/klein_{packed_cf,unpatch_f32,image}.bin \
                reference dumps; no committed script produces them — see \
                crates/lumen-mlx/tests/golden/README.md"]
    fn vae_decode_packed_matches_reference() {
        use crate::vae::VaeDecoder;
        let (lat_h, lat_w, _img_seq, _txt) = read_dims();
        // mflux dumped the channels-first packed latents [1,128,lh,lw].
        let packed = read_f32_bin("/tmp/klein_packed_cf.bin");
        let packed_arr = Array::from_slice(&packed, &[1, 128, lat_h, lat_w]);
        packed_arr.eval().unwrap();

        let vae = VaeDecoder::load(vae_path()).expect("load VAE");

        // Glue-only check: bn-denorm + unpatchify vs an f32 reference computed
        // identically (channels-first [1,32,2lh,2lw]). This isolates the
        // reshape/transpose order + bn math from bf16 noise — mflux's own
        // bf16 path differs from f32 here by ~1.7e-2, so f32-vs-f32 is the
        // exactness criterion for the glue.
        let unpatch = vae.unpatchify_debug(&packed_arr).expect("unpatchify");
        unpatch.eval().unwrap();
        let up_obs: &[f32] = unpatch.as_slice();
        let up_exp = read_f32_bin("/tmp/klein_unpatch_f32.bin");
        assert_eq!(up_obs.len(), up_exp.len(), "unpatch size");
        let mut up_max = 0.0f32;
        for (&g, &e) in up_obs.iter().zip(up_exp.iter()) {
            up_max = up_max.max((g - e).abs());
        }
        println!("bn-denorm+unpatchify glue (f32-vs-f32) max_abs = {up_max:.6e}");
        assert!(
            up_max < 1e-4,
            "unpatchify glue max_abs {up_max:.6e} >= 1e-4"
        );

        let img_cl = vae
            .decode_packed_latents(&packed_arr)
            .expect("decode_packed");
        img_cl.eval().unwrap();
        let h = lat_h * 16; // 2*lat * 8 vae upsample = 16x
        let w = lat_w * 16;
        assert_eq!(img_cl.shape(), &[1, h, w, 3], "image shape (channels-last)");
        let obs: &[f32] = img_cl.as_slice();

        // mflux image is channels-first [1,3,H,W]; reorder to channels-last.
        let exp_cf = read_f32_bin("/tmp/klein_image.bin");
        let exp = cf_to_cl(&exp_cf, 1, 3, h as usize, w as usize);
        let (mut max_abs, mut mse) = (0.0f32, 0.0f64);
        for (&g, &e) in obs.iter().zip(exp.iter()) {
            let d = (g - e).abs();
            max_abs = max_abs.max(d);
            mse += (d as f64) * (d as f64);
        }
        mse /= obs.len() as f64;
        // images in [-1,1] → peak 2.0. NOTE: the mflux reference decode runs in
        // bf16 and is non-deterministic to ~0.12 max-abs run-to-run, so a tight
        // max-abs gate is not meaningful here; PSNR is the robust criterion.
        let psnr = 10.0 * (4.0 / mse).log10();
        println!("vae_decode_packed max_abs = {max_abs:.6e}  psnr = {psnr:.2} dB");
        assert!(
            max_abs < 5e-2 || psnr > 30.0,
            "vae packed decode max_abs {max_abs:.6e} >= 5e-2 and psnr {psnr:.2} <= 30dB"
        );
    }

    /// Full e2e glue parity: run the Rust denoise loop on mflux's init latents,
    /// compare per-step latents (cosine) + final image (PSNR / max-abs).
    #[test]
    // /tmp/klein_sigmas.bin is the same dump the `flux-scheduler-invariants`
    // defect was recorded against: a dev-session artifact that had ceased to
    // exist, so the test failed everywhere but the machine that made it. That
    // one was fixed by reading the schedule from the code; this test still
    // needs the whole set, and nothing committed regenerates them.
    #[ignore = "needs a Metal device AND the full /tmp/klein_*.bin dump set \
                (sigmas, prompt_embeds, init_latents, img_ids, text_ids, per_step, \
                image); no committed script produces them — see \
                crates/lumen-mlx/tests/golden/README.md"]
    fn klein_e2e_matches_reference() {
        let (lat_h, lat_w, img_seq, txt_seq) = read_dims();
        let steps = 4usize;

        // 0. scheduler — assert it matches mflux first (defense in depth).
        let sched = scheduler::compute((lat_h * lat_w) as usize, steps);
        let exp_sigmas = read_f32_bin("/tmp/klein_sigmas.bin");
        let mut sig_max = 0.0f32;
        for (&g, &e) in sched.sigmas.iter().zip(exp_sigmas.iter()) {
            sig_max = sig_max.max((g - e).abs());
        }
        println!("scheduler sigma max_abs = {sig_max:.3e}");
        assert!(sig_max < 1e-4, "scheduler mismatch {sig_max:.3e}");

        // 1. load mflux dumps.
        let prompt_embeds = Array::from_slice(
            &read_f32_bin("/tmp/klein_prompt_embeds.bin"),
            &[1, txt_seq, 7680],
        );
        let init_latents = Array::from_slice(
            &read_f32_bin("/tmp/klein_init_latents.bin"),
            &[1, img_seq, 128],
        );
        let img_ids = read_i32_bin("/tmp/klein_img_ids.bin");
        let text_ids = read_i32_bin("/tmp/klein_text_ids.bin");
        prompt_embeds.eval().unwrap();
        init_latents.eval().unwrap();
        assert_eq!(img_ids.len(), (img_seq * 4) as usize, "img_ids len");
        assert_eq!(text_ids.len(), (txt_seq * 4) as usize, "text_ids len");

        // 2. run the Rust pipeline.
        let pipe = KleinPipeline::load(&dit_path(), &vae_path()).expect("load pipeline");
        let (per_step, image) = pipe
            .generate_with_trace(
                &prompt_embeds,
                &text_ids,
                &init_latents,
                &img_ids,
                &sched,
                lat_h,
                lat_w,
            )
            .expect("generate");

        // 3. per-step latent parity (cosine > 0.999).
        let ref_per_step = read_f32_bin("/tmp/klein_per_step.bin"); // [steps,1,img_seq,128]
        let step_elems = (img_seq * 128) as usize;
        for (t, latents) in per_step.iter().enumerate() {
            latents.eval().unwrap();
            let obs: &[f32] = latents.as_slice();
            let exp = &ref_per_step[t * step_elems..(t + 1) * step_elems];
            let cos = cosine(obs, exp);
            println!("step {t}: latents cosine = {cos:.6}");
            assert!(cos > 0.999, "step {t} cosine {cos:.6} <= 0.999");
        }

        // 4. final image parity (max-abs / PSNR).
        let h = lat_h * 16;
        let w = lat_w * 16;
        image.eval().unwrap();
        assert_eq!(image.shape(), &[1, h, w, 3], "image shape");
        let obs: &[f32] = image.as_slice();
        let exp_cf = read_f32_bin("/tmp/klein_image.bin");
        let exp = cf_to_cl(&exp_cf, 1, 3, h as usize, w as usize);
        let (mut max_abs, mut mse) = (0.0f32, 0.0f64);
        for (&g, &e) in obs.iter().zip(exp.iter()) {
            let d = (g - e).abs();
            max_abs = max_abs.max(d);
            mse += (d as f64) * (d as f64);
        }
        mse /= obs.len() as f64;
        // images are in [-1,1] range → peak 2.0.
        let psnr = 10.0 * (4.0 / mse).log10();
        println!("final image: max_abs = {max_abs:.6e}  psnr = {psnr:.2} dB");
        assert!(
            max_abs < 5e-2 || psnr > 30.0,
            "final image max_abs {max_abs:.6e} >= 5e-2 and psnr {psnr:.2} <= 30dB"
        );
    }
}
