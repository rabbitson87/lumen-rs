//! A streaming response has to reach the client while the engine generates.
//!
//! Every stream used to arrive in one burst when generation finished: the
//! engine ran as a tokio task that never yields during a request, and the SSE
//! writer it woke did not run until the request was done (task 019). An
//! in-process test of the engine did not reproduce it — the stall came from how
//! the server's own runtime scheduled the two — so this drives the real binary
//! over HTTP and times what a client receives.
//!
//! Needs a Gemma 4 checkpoint; without one it prints a skip line and passes:
//!
//! ```sh
//! LUMEN_GEMMA4_MODEL_DIR=/path/to/gemma-4 \
//!   cargo test -p lumen-server --release --test streaming_delivery -- --ignored
//! ```

use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The server process, killed on drop so a failed assertion never leaves it
/// holding the GPU.
struct Server(Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .expect("a free port")
        .port()
}

/// Starts the server on `port` and returns once it accepts requests (after its
/// own warm-up, so the weights are already materialized).
fn start(model_dir: &str, port: u16) -> Server {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lumen-server"))
        .env("MODEL_ID", model_dir)
        .env("PORT", port.to_string())
        .env("LUMEN_HOST", "127.0.0.1")
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn lumen-server");
    let stderr = child.stderr.take().expect("server stderr");
    let server = Server(child);
    let (ready_tx, ready_rx) = mpsc::channel();
    // Keeps draining after the ready line too: a full pipe would block the
    // server on its next log line.
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines().map_while(Result::ok) {
            if line.contains("serving on") {
                let _ = ready_tx.send(());
            }
        }
    });
    ready_rx
        .recv_timeout(Duration::from_secs(600))
        .expect("the server never started listening");
    server
}

/// Streams one chat completion and returns when each non-empty content delta
/// arrived, measured from the request.
fn content_delta_arrivals(port: u16) -> Vec<Duration> {
    let body = r#"{"model":"gemma-4","messages":[{"role":"user","content":"Count from 1 to 60, separated by spaces."}],"max_tokens":64,"temperature":0,"stream":true,"chat_template_kwargs":{"enable_thinking":false}}"#;
    let mut tcp = TcpStream::connect(("127.0.0.1", port)).expect("connect");
    let t0 = Instant::now();
    write!(
        tcp,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
    .expect("send the request");
    let mut arrivals = Vec::new();
    for line in BufReader::new(tcp).lines().map_while(Result::ok) {
        if line.starts_with("data: [DONE]") {
            break;
        }
        if line.contains("\"delta\":{\"content\":\"") && !line.contains("\"content\":\"\"") {
            arrivals.push(t0.elapsed());
        }
    }
    arrivals
}

#[test]
#[ignore = "requires a Gemma 4 checkpoint; set LUMEN_GEMMA4_MODEL_DIR"]
fn the_server_streams_tokens_while_it_generates() {
    let Ok(dir) = std::env::var("LUMEN_GEMMA4_MODEL_DIR") else {
        eprintln!("skip: set LUMEN_GEMMA4_MODEL_DIR to a Gemma 4 checkpoint");
        return;
    };
    let port = free_port();
    let _server = start(&dir, port);
    let _ = content_delta_arrivals(port);
    let arrivals = content_delta_arrivals(port);
    assert!(
        arrivals.len() >= 16,
        "only {} content deltas",
        arrivals.len()
    );
    let (first, last) = (arrivals[0], arrivals[arrivals.len() - 1]);
    assert!(
        first < last / 2,
        "the first token arrived at {first:?} of a {last:?} stream: delivered in one burst"
    );
}
