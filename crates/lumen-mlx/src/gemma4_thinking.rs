//! Thinking-channel budget tracker for Gemma 4.
//!
//! Gemma 4 emits a `<|channel>thought\n…<channel|>` block for chain-of-
//! thought reasoning before producing visible tokens. The model is
//! supposed to close the channel on its own once reasoning is done,
//! but reasoning-first variants (notably fit-korean) sometimes loop
//! forever inside the thought channel — the token distribution stays
//! varied (so [`lumen_core::runaway`] doesn't trip and [`lumen_core::dry`]
//! sees no n-gram cycle), yet the model never emits the close-channel
//! token and the user sees zero visible output.
//!
//! This module tracks the in-thought state across a decode loop and
//! signals the caller to stop when the thought budget is exceeded.
//! Analogous to llama.cpp's `--reasoning-budget` knob (see GitHub
//! issue ggml-org/llama.cpp#21141), but applied at the token level
//! inside our decode loop rather than via a separate truncation step.

// Token IDs are duplicated here (not imported) so this module compiles
// without the `mlx-native` feature flag — it's pure state-machine
// logic, and pulling in `gemma4_chat::imp` would gate the whole
// runaway-defence layer behind a feature.
pub const TOK_CHANNEL_OPEN: u32 = 100;
pub const TOK_CHANNEL_CLOSE: u32 = 101;
/// `<turn|>` — closes the model turn. Member of the EOS set
/// `[1, 50, 106]`, so emitting it triggers normal decode termination
/// on the next iteration.
pub const TOK_TURN_CLOSE: u32 = 106;

/// State machine that watches each emitted token and flips a flag
/// when the model has spent more than `max_thinking_tokens` inside
/// an open `<|channel>thought…` block.
///
/// Recovery strategy when the budget is exceeded:
///
/// 1. **First trip** → [`Self::try_force_close`] returns
///    [`TOK_CHANNEL_CLOSE`]. The caller swaps the sampled token for
///    the close-channel id and pushes it into the cache. The model
///    sees a syntactically-valid channel close and typically
///    transitions to the visible channel on its own.
///
/// 2. **Second trip** (model re-opens thought and overruns again) →
///    [`Self::should_hard_break`] returns `true`. Caller breaks the
///    decode loop. We don't keep retrying because models that
///    re-enter thought after a forced close are in a degenerate state
///    that further force-closes won't recover.
#[derive(Debug, Clone, Copy)]
pub struct ChannelBudget {
    pub max_thinking_tokens: usize,
    pub max_force_close_attempts: usize,
    in_thought: bool,
    thought_count: usize,
    force_close_attempts: usize,
}

impl ChannelBudget {
    /// `max_thinking_tokens == 0` disables the budget (default).
    pub fn new(max_thinking_tokens: usize) -> Self {
        Self {
            max_thinking_tokens,
            max_force_close_attempts: 1,
            in_thought: false,
            thought_count: 0,
            force_close_attempts: 0,
        }
    }

    /// Read from `LUMEN_MAX_THINKING_TOKENS` (budget) and
    /// `LUMEN_MAX_FORCE_CLOSE_ATTEMPTS` (default 1).
    pub fn from_env() -> Self {
        let n = std::env::var("LUMEN_MAX_THINKING_TOKENS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(0);
        let attempts = std::env::var("LUMEN_MAX_FORCE_CLOSE_ATTEMPTS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(1);
        let mut b = Self::new(n);
        b.max_force_close_attempts = attempts;
        b
    }

    /// Update state after a token is emitted. Toggles `in_thought` on
    /// `<|channel>` / `<channel|>` and increments the counter for
    /// every token inside the channel.
    pub fn observe(&mut self, token: u32) {
        if token == TOK_CHANNEL_OPEN {
            self.in_thought = true;
            self.thought_count = 0;
        } else if token == TOK_CHANNEL_CLOSE {
            self.in_thought = false;
            self.thought_count = 0;
        } else if self.in_thought {
            self.thought_count += 1;
        }
    }

    /// True once the model has spent more than `max_thinking_tokens`
    /// continuously inside an open thought channel.
    pub fn exceeded(&self) -> bool {
        self.max_thinking_tokens > 0
            && self.in_thought
            && self.thought_count > self.max_thinking_tokens
    }

    /// If the budget is exceeded AND we haven't burned our force-close
    /// quota, return the channel-close token id and update internal
    /// state as if the model emitted it. Caller must replace the
    /// just-sampled token with the returned id, push it into the
    /// generated trail, and feed it as the next decode input — that
    /// keeps the KV cache consistent with the visible token stream.
    pub fn try_force_close(&mut self) -> Option<u32> {
        if !self.exceeded() {
            return None;
        }
        if self.force_close_attempts >= self.max_force_close_attempts {
            return None;
        }
        self.in_thought = false;
        self.thought_count = 0;
        self.force_close_attempts += 1;
        Some(TOK_CHANNEL_CLOSE)
    }

    /// True after we've exhausted force-close attempts and the model
    /// still won't leave the thought channel. Caller should abort the
    /// decode loop — further force-closes are unlikely to help.
    pub fn should_hard_break(&self) -> bool {
        self.exceeded() && self.force_close_attempts >= self.max_force_close_attempts
    }

    #[allow(dead_code)] // no caller for some accessors: the streaming path reads the state machine
    // through `step()` instead.
    pub fn is_active(&self) -> bool {
        self.max_thinking_tokens > 0
    }

    #[allow(dead_code)] // no caller: the streaming path reads state through `step()`.
    pub fn in_thought(&self) -> bool {
        self.in_thought
    }

    pub fn thought_count(&self) -> usize {
        self.thought_count
    }

    #[allow(dead_code)] // no caller: `should_hard_break()` is the read that matters.
    pub fn force_close_attempts(&self) -> usize {
        self.force_close_attempts
    }

    /// After at least one force-close has occurred, the model
    /// shouldn't be allowed to re-enter the thought channel —
    /// fit-korean and similar reasoning-first variants will loop
    /// indefinitely back into `<|channel>thought` if given the
    /// chance. Callers should swap a sampled
    /// [`TOK_CHANNEL_OPEN`] for [`TOK_TURN_CLOSE`] when this
    /// returns `true`.
    pub fn should_block_channel_open(&self) -> bool {
        self.force_close_attempts > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_never_trips() {
        let mut b = ChannelBudget::new(0);
        b.observe(TOK_CHANNEL_OPEN);
        for _ in 0..10_000 {
            b.observe(42);
        }
        assert!(!b.exceeded());
    }

    #[test]
    fn counts_only_inside_thought() {
        let mut b = ChannelBudget::new(100);
        // Tokens before any open are ignored.
        for _ in 0..200 {
            b.observe(42);
        }
        assert!(!b.exceeded());

        b.observe(TOK_CHANNEL_OPEN);
        for _ in 0..50 {
            b.observe(42);
        }
        assert!(!b.exceeded()); // 50 <= 100
        for _ in 0..51 {
            b.observe(42);
        }
        assert!(b.exceeded()); // 101 > 100
    }

    #[test]
    fn close_resets_counter() {
        let mut b = ChannelBudget::new(10);
        b.observe(TOK_CHANNEL_OPEN);
        for _ in 0..8 {
            b.observe(42);
        }
        b.observe(TOK_CHANNEL_CLOSE);
        // Re-open: counter is fresh.
        b.observe(TOK_CHANNEL_OPEN);
        for _ in 0..8 {
            b.observe(42);
        }
        assert!(!b.exceeded());
    }

    #[test]
    fn force_close_first_attempt_then_hard_break() {
        let mut b = ChannelBudget::new(5);
        b.observe(TOK_CHANNEL_OPEN);
        for _ in 0..6 {
            b.observe(42);
        }
        assert!(b.exceeded());
        // First force-close: returns the close token, resets state.
        let forced = b.try_force_close();
        assert_eq!(forced, Some(TOK_CHANNEL_CLOSE));
        assert!(!b.exceeded());
        assert!(!b.should_hard_break());
        // Model re-opens thought and overruns again.
        b.observe(TOK_CHANNEL_OPEN);
        for _ in 0..6 {
            b.observe(42);
        }
        assert!(b.exceeded());
        // Second attempt blocked — quota exhausted.
        assert_eq!(b.try_force_close(), None);
        assert!(b.should_hard_break());
    }

    #[test]
    fn try_force_close_noop_when_under_budget() {
        let mut b = ChannelBudget::new(100);
        b.observe(TOK_CHANNEL_OPEN);
        b.observe(42);
        assert!(!b.exceeded());
        assert_eq!(b.try_force_close(), None);
        assert_eq!(b.force_close_attempts(), 0);
    }
}
