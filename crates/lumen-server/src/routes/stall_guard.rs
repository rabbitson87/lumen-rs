//! A response writer that gives up on a client that has stopped reading.
//!
//! A write to a connected client that takes no bytes waits for as long as the
//! connection lives. An SSE route that wrote with `tcp.write_all(..).await?`
//! therefore never found out: it sat on the socket, and the queue of events the
//! engine had produced for it, until the peer read or the OS dropped the
//! connection. [`StallGuard`] turns a write that makes no progress for
//! [`WRITE_STALL`] into an error, which ends the route, drops the receiver
//! (freeing the queue) and tells the engine its client is gone.

use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;

use tokio::io::AsyncWrite;
use tokio::time::{Sleep, sleep};

/// How long a write may wait without a byte being taken before the client is
/// treated as gone. The kernel buffers a good deal before a write has to wait
/// at all, so a client that is merely slow never gets here.
pub const WRITE_STALL: Duration = Duration::from_secs(120);

pub struct StallGuard<W> {
    inner: W,
    limit: Duration,
    /// Set while a write is waiting: the deadline from when it first had to.
    waiting: Option<Pin<Box<Sleep>>>,
}

impl<W> StallGuard<W> {
    pub fn new(inner: W) -> Self {
        Self::with_limit(inner, WRITE_STALL)
    }

    pub fn with_limit(inner: W, limit: Duration) -> Self {
        Self {
            inner,
            limit,
            waiting: None,
        }
    }

    /// One poll of the inner writer under the deadline.
    fn guarded<T>(
        &mut self,
        cx: &mut Context<'_>,
        poll: impl FnOnce(Pin<&mut W>, &mut Context<'_>) -> Poll<io::Result<T>>,
    ) -> Poll<io::Result<T>>
    where
        W: Unpin,
    {
        match poll(Pin::new(&mut self.inner), cx) {
            Poll::Ready(result) => {
                self.waiting = None;
                Poll::Ready(result)
            }
            Poll::Pending => {
                let limit = self.limit;
                let timer = self.waiting.get_or_insert_with(|| Box::pin(sleep(limit)));
                match timer.as_mut().poll(cx) {
                    Poll::Ready(()) => {
                        self.waiting = None;
                        Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            "the client stopped reading",
                        )))
                    }
                    Poll::Pending => Poll::Pending,
                }
            }
        }
    }
}

impl<W: AsyncWrite + Unpin> AsyncWrite for StallGuard<W> {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        self.guarded(cx, |w, cx| w.poll_write(cx, buf))
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.guarded(cx, |w, cx| w.poll_flush(cx))
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        self.guarded(cx, |w, cx| w.poll_shutdown(cx))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::AsyncWriteExt;

    /// A peer that never takes a byte.
    struct Stalled;

    impl AsyncWrite for Stalled {
        fn poll_write(
            self: Pin<&mut Self>,
            _: &mut Context<'_>,
            _: &[u8],
        ) -> Poll<io::Result<usize>> {
            Poll::Pending
        }
        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// A peer that takes one byte every so often, as a slow client does.
    struct Trickle {
        every: Duration,
        next: Pin<Box<Sleep>>,
        taken: usize,
    }

    impl AsyncWrite for Trickle {
        fn poll_write(
            mut self: Pin<&mut Self>,
            cx: &mut Context<'_>,
            buf: &[u8],
        ) -> Poll<io::Result<usize>> {
            if self.next.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            let every = self.every;
            self.next = Box::pin(sleep(every));
            self.taken += 1;
            Poll::Ready(Ok(buf.len().min(1)))
        }
        fn poll_flush(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
        fn poll_shutdown(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<io::Result<()>> {
            Poll::Ready(Ok(()))
        }
    }

    /// RED before: the write waited for ever.
    #[tokio::test]
    async fn a_client_that_takes_no_bytes_times_out() {
        let mut tcp = StallGuard::with_limit(Stalled, Duration::from_millis(60));
        // Bounded here so a guard that never gives up fails this test rather
        // than hanging it.
        let outcome = tokio::time::timeout(Duration::from_secs(2), tcp.write_all(b"data: x\n\n"));
        let err = outcome
            .await
            .expect("the guard must give up on its own")
            .expect_err("a write nobody takes is an error");
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    /// The deadline is per stall, not per response: a client that keeps taking
    /// bytes, however slowly, is never cut off. Twelve bytes take about 240 ms,
    /// four times the limit; each one arrives well inside it.
    #[tokio::test]
    async fn a_slow_client_is_not_cut_off() {
        let slow = Trickle {
            every: Duration::from_millis(20),
            next: Box::pin(sleep(Duration::from_millis(20))),
            taken: 0,
        };
        let mut tcp = StallGuard::with_limit(slow, Duration::from_millis(60));
        tcp.write_all(&[7; 12])
            .await
            .expect("each byte beats the deadline");
        assert_eq!(tcp.inner.taken, 12);
    }
}
