//! Who may reach the server, and from where.
//!
//! lumen-app has always passed three settings to the server — `LUMEN_HOST`,
//! `LUMEN_API_KEY` and `LUMEN_CORS` — and the server read none of them. It
//! bound every interface and answered anyone, whatever the app's
//! "localhost (127.0.0.1)" scope or API key said, while the README promised a
//! loopback default. This module is where those settings take effect.

use atomic_http::external::http::HeaderMap;
use std::net::IpAddr;

/// The address the listener binds.
///
/// `LUMEN_HOST` (what lumen-app sets) wins. `HOST` (what the docs used to
/// promise) is honoured only as an IP literal: shells and CI runners export
/// `HOST` as the machine's *name*, and binding to that would put the server on
/// the LAN interface and take it off loopback. The default is loopback — an API
/// that may have no key must not face the network unless someone asked it to.
pub fn listen_addr(lumen_host: Option<&str>, host: Option<&str>, port: u16) -> String {
    let host = lumen_host
        .map(str::trim)
        .filter(|h| !h.is_empty())
        .or_else(|| host.map(str::trim).filter(|h| h.parse::<IpAddr>().is_ok()))
        .unwrap_or("127.0.0.1");
    // An IPv6 literal needs brackets before the port: `[::]:41110`.
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]:{port}")
    } else {
        format!("{host}:{port}")
    }
}

/// `LUMEN_CORS`: which browser origins may read responses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cors {
    /// No CORS headers, so browsers block cross-origin reads. The default.
    Off,
    /// Pages served from this machine: `localhost`, `127.0.0.1` or `[::1]`, on
    /// any port, over http or https.
    Localhost,
    /// Any origin.
    All,
}

impl Cors {
    /// Unset or unrecognised is `Off` — the setting can only open access.
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("localhost") => Self::Localhost,
            Some("all") | Some("*") => Self::All,
            _ => Self::Off,
        }
    }

    /// The `Access-Control-Allow-Origin` value for a request from `origin`.
    pub fn allow_origin(self, origin: Option<&str>) -> Option<String> {
        match self {
            Self::Off => None,
            Self::All => Some("*".to_string()),
            Self::Localhost => origin.filter(|o| is_local_origin(o)).map(str::to_string),
        }
    }
}

/// An `Origin` is scheme, host and optional port — nothing else — so the host
/// is matched exactly: `http://localhost.example.com` is not local.
fn is_local_origin(origin: &str) -> bool {
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let (host, port) = match rest.strip_prefix('[') {
        Some(v6) => match v6.split_once(']') {
            Some((host, port)) => (host, port),
            None => return false,
        },
        None => match rest.split_once(':') {
            Some((host, port)) => (host, port),
            None => (rest, ""),
        },
    };
    let port_ok = port.is_empty()
        || port
            .strip_prefix(':')
            .unwrap_or(port)
            .bytes()
            .all(|b| b.is_ascii_digit());
    port_ok && matches!(host, "localhost" | "127.0.0.1" | "::1")
}

/// The access settings, resolved once at startup.
#[derive(Debug, Clone)]
pub struct Access {
    api_key: Option<String>,
    cors: Cors,
}

impl Access {
    pub fn new(api_key: Option<&str>, cors: Cors) -> Self {
        Self {
            api_key: api_key
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .map(str::to_string),
            cors,
        }
    }

    pub fn from_env() -> Self {
        Self::new(
            std::env::var("LUMEN_API_KEY").ok().as_deref(),
            Cors::parse(std::env::var("LUMEN_CORS").ok().as_deref()),
        )
    }

    pub fn requires_key(&self) -> bool {
        self.api_key.is_some()
    }

    pub fn cors(&self) -> Cors {
        self.cors
    }

    /// Whether a request may proceed.
    ///
    /// With `LUMEN_API_KEY` set, every route needs the key except `/health`
    /// (liveness probes) and CORS preflight (browsers never attach
    /// credentials to it). OpenAI clients send it as `Authorization: Bearer`,
    /// Anthropic clients as `x-api-key`; either is accepted.
    pub fn authorized(&self, method: &str, path: &str, headers: &HeaderMap) -> bool {
        let Some(key) = self.api_key.as_deref() else {
            return true;
        };
        if path == "/health" || method == "OPTIONS" {
            return true;
        }
        let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
        let bearer = header("authorization").and_then(|v| {
            let (scheme, token) = v.trim().split_once(' ')?;
            scheme.eq_ignore_ascii_case("bearer").then_some(token)
        });
        [bearer, header("x-api-key")]
            .into_iter()
            .flatten()
            .any(|given| constant_time_eq(given.trim().as_bytes(), key.as_bytes()))
    }
}

/// Compares without an early exit on the first differing byte, so the time a
/// rejection takes says nothing about how much of a guessed key was right.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomic_http::external::http::HeaderValue;

    fn headers(pairs: &[(&'static str, &str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_str(v).expect("header value"));
        }
        h
    }

    #[test]
    fn the_listener_defaults_to_loopback() {
        assert_eq!(listen_addr(None, None, 41110), "127.0.0.1:41110");
        assert_eq!(listen_addr(Some("  "), None, 41110), "127.0.0.1:41110");
    }

    #[test]
    fn the_listener_binds_what_the_app_asked_for() {
        assert_eq!(listen_addr(Some("0.0.0.0"), None, 8), "0.0.0.0:8");
        assert_eq!(
            listen_addr(Some("192.168.1.20"), Some("10.0.0.1"), 8),
            "192.168.1.20:8"
        );
        assert_eq!(listen_addr(Some("::"), None, 8), "[::]:8");
        assert_eq!(listen_addr(Some("[::1]"), None, 8), "[::1]:8");
    }

    #[test]
    fn host_counts_only_as_an_ip_literal() {
        assert_eq!(listen_addr(None, Some("0.0.0.0"), 8), "0.0.0.0:8");
        // zsh and CI runners export HOST as the machine name.
        assert_eq!(
            listen_addr(None, Some("Mac-Studio.local"), 8),
            "127.0.0.1:8"
        );
    }

    #[test]
    fn without_a_key_every_request_is_authorized() {
        let open = Access::new(None, Cors::Off);
        assert!(!open.requires_key());
        assert!(open.authorized("POST", "/v1/chat/completions", &HeaderMap::new()));
        assert!(Access::new(Some("  "), Cors::Off).authorized(
            "GET",
            "/v1/models",
            &HeaderMap::new()
        ));
    }

    #[test]
    fn a_configured_key_is_required_everywhere_but_health_and_preflight() {
        let locked = Access::new(Some("s3cret"), Cors::Off);
        assert!(locked.requires_key());
        for path in [
            "/v1/chat/completions",
            "/v1/messages",
            "/v1/models",
            "/v1/loads",
        ] {
            assert!(
                !locked.authorized("POST", path, &HeaderMap::new()),
                "{path}"
            );
        }
        assert!(locked.authorized("GET", "/health", &HeaderMap::new()));
        assert!(locked.authorized("OPTIONS", "/v1/messages", &HeaderMap::new()));
    }

    #[test]
    fn the_key_is_accepted_the_way_each_sdk_sends_it() {
        let locked = Access::new(Some("s3cret"), Cors::Off);
        let ok = |h: HeaderMap| locked.authorized("POST", "/v1/messages", &h);
        assert!(ok(headers(&[("authorization", "Bearer s3cret")])));
        assert!(ok(headers(&[("authorization", "bearer s3cret")])));
        assert!(ok(headers(&[("x-api-key", "s3cret")])));
        assert!(!ok(headers(&[("authorization", "Bearer s3cre")])));
        assert!(!ok(headers(&[("authorization", "Bearer s3cret!")])));
        assert!(!ok(headers(&[("authorization", "Basic s3cret")])));
        assert!(!ok(headers(&[("authorization", "s3cret")])));
        assert!(!ok(headers(&[("x-api-key", "nope")])));
    }

    #[test]
    fn cors_parses_the_apps_three_modes() {
        assert_eq!(Cors::parse(None), Cors::Off);
        assert_eq!(Cors::parse(Some("off")), Cors::Off);
        assert_eq!(Cors::parse(Some("Localhost")), Cors::Localhost);
        assert_eq!(Cors::parse(Some("all")), Cors::All);
        assert_eq!(Cors::parse(Some("everything")), Cors::Off);
    }

    #[test]
    fn localhost_cors_admits_only_this_machines_origins() {
        let allow = |o: &str| Cors::Localhost.allow_origin(Some(o));
        for local in [
            "http://localhost:5173",
            "http://localhost",
            "https://127.0.0.1:3000",
            "http://[::1]:8080",
        ] {
            assert_eq!(allow(local).as_deref(), Some(local), "{local}");
        }
        for remote in [
            "http://localhost.example.com",
            "http://127.0.0.1.nip.io",
            "http://evil.com",
            "null",
            "file://",
            "http://localhost:80/path",
        ] {
            assert_eq!(allow(remote), None, "{remote}");
        }
        assert_eq!(Cors::Localhost.allow_origin(None), None);
    }

    #[test]
    fn all_and_off_ignore_the_origin() {
        assert_eq!(
            Cors::All.allow_origin(Some("http://evil.com")).as_deref(),
            Some("*")
        );
        assert_eq!(Cors::All.allow_origin(None).as_deref(), Some("*"));
        assert_eq!(Cors::Off.allow_origin(Some("http://localhost:5173")), None);
    }
}
