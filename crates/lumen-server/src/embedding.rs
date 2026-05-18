//! Embedding service — channel-actor wrapper around
//! `lumen_model::embedding::EmbeddingModel` (in-process candle Qwen3).
//! Mirrors the `EngineHandle` pattern used for chat/completions so the
//! route handler stays free of locking concerns.
//!
//! The model loads inside the service task once at startup and serves
//! requests sequentially — candle's KV cache is per-model and must be
//! cleared between independent sequences (the model itself handles
//! that internally), and concurrent forward calls into the same
//! `&mut Model` would be unsound.

use anyhow::Result;
use lumen_model::embedding::{EmbeddingBatch, EmbeddingModel};
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

    pub async fn embed(&self, texts: Vec<String>) -> Result<EmbeddingBatch> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbeddingRequest::Embed {
                texts,
                reply: reply_tx,
            })
            .await
            .map_err(|_| anyhow::anyhow!("embedding service channel closed"))?;
        reply_rx
            .await
            .map_err(|_| anyhow::anyhow!("embedding service dropped reply"))?
    }

    pub async fn info(&self) -> Option<EmbeddingInfo> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EmbeddingRequest::Info { reply: reply_tx })
            .await
            .ok()?;
        reply_rx.await.ok()
    }
}

pub struct EmbeddingService {
    model: EmbeddingModel,
}

impl EmbeddingService {
    pub fn load(model_id: &str) -> Result<Self> {
        Ok(Self {
            model: EmbeddingModel::load(model_id)?,
        })
    }

    /// Drain the request channel until the sender side is dropped.
    /// Runs on a dedicated OS thread (see `main.rs`) because candle's
    /// forward call is synchronous CPU/GPU work and would otherwise
    /// block a tokio worker.
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
