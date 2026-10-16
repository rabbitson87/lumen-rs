//! Embedding service — channel-actor wrapper around
//! [`lumen_mlx::embedding::EmbeddingModel`] (in-process Qwen3 on MLX).
//! Mirrors the `EngineHandle` pattern used for chat/completions so the
//! route handler stays free of locking concerns.
//!
//! The model loads inside the service task once at startup and serves requests
//! sequentially: `embed` takes `&mut self`, and concurrent forward calls
//! sharing one MLX stream would race. Batching happens *within* a request —
//! the model pads a whole `texts` slice into one forward pass — so serializing
//! at this layer costs nothing for the common case of one POST with many
//! inputs.
//!
//! Only [`EmbeddingService`] needs the `mlx-native` feature; the handle and
//! message types stay available either way, so `main.rs` can hold an
//! `Option<EmbeddingHandle>` that is simply `None` in a build without it —
//! the same state as `EMBEDDING_MODEL_ID` being unset, which `/v1/embeddings`
//! already answers with a 503.

use crate::engine::request_reply;
use anyhow::Result;
use lumen_mlx::embedding::EmbeddingBatch;
use tokio::sync::{mpsc, oneshot};

pub enum EmbeddingRequest {
    Embed {
        texts: Vec<String>,
        reply: oneshot::Sender<Result<EmbeddingBatch>>,
    },
    Info {
        reply: oneshot::Sender<EmbeddingInfo>,
    },
}

#[derive(Debug, Clone)]
pub struct EmbeddingInfo {
    pub model_id: String,
    pub dim: usize,
    pub max_seq_len: usize,
}

#[derive(Clone)]
pub struct EmbeddingHandle {
    tx: mpsc::Sender<EmbeddingRequest>,
}

impl EmbeddingHandle {
    pub fn new(tx: mpsc::Sender<EmbeddingRequest>) -> Self {
        Self { tx }
    }

    /// [`request_reply`] to the embedding service.
    async fn request<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> EmbeddingRequest,
    ) -> Result<T> {
        request_reply(
            &self.tx,
            make,
            "embedding service channel closed",
            "embedding service dropped reply",
        )
        .await
    }

    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingBatch> {
        self.request(|reply| EmbeddingRequest::Embed { texts, reply })
            .await?
    }

    pub async fn info(&self) -> Option<EmbeddingInfo> {
        self.request(|reply| EmbeddingRequest::Info { reply })
            .await
            .ok()
    }
}

#[cfg(feature = "mlx-native")]
pub struct EmbeddingService {
    model: lumen_mlx::embedding::EmbeddingModel,
}

#[cfg(feature = "mlx-native")]
impl EmbeddingService {
    pub fn load(model_id: &str) -> Result<Self> {
        Ok(Self {
            model: lumen_mlx::embedding::EmbeddingModel::load(model_id)?,
        })
    }

    /// Drain the request channel until the sender side is dropped.
    /// Runs on a dedicated OS thread (see `main.rs`) because the forward pass
    /// is synchronous GPU work and would otherwise block a tokio worker.
    pub fn run(mut self, mut rx: mpsc::Receiver<EmbeddingRequest>) {
        while let Some(req) = rx.blocking_recv() {
            match req {
                EmbeddingRequest::Embed { texts, reply } => {
                    let res = self.model.embed(&texts);
                    let _ = reply.send(res);
                }
                EmbeddingRequest::Info { reply } => {
                    let _ = reply.send(EmbeddingInfo {
                        model_id: self.model.model_id().to_string(),
                        dim: self.model.dim(),
                        max_seq_len: self.model.max_seq_len(),
                    });
                }
            }
        }
    }
}
