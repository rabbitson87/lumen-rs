//! `LUMEN_STRIP_CLIENT_META_WRAPPERS` — the escape hatch (005 Phase 4.1).
//!
//! Own test binary, for the reason `gemma4_config_env.rs` is: the flag is read
//! from the process environment, so a test that sets it races every sibling
//! that strips. Here that would silently turn
//! `chat_io_meta_wrapper.rs`'s strip assertions into no-ops — passing tests
//! that no longer test anything, which is worse than a failure.
//!
//! The switch is worth its own coverage because it is the only way out when the
//! heuristic is wrong about a real user's message, and it is spelled
//! independently in **three** functions. One of them silently ignoring the flag
//! leaves an operator with no way to turn the deletion off.

use lumen_mlx::chat_io::{
    ChatTurn, strip_client_meta_wrappers, strip_client_meta_wrappers_flat,
    strip_client_meta_wrappers_flat_indexed, strip_client_meta_wrappers_indexed,
};

const ENV: &str = "LUMEN_STRIP_CLIENT_META_WRAPPERS";

fn wrapper() -> String {
    "If you have completed the work, call task_complete now. Thanks.".to_string()
}

struct Restore;
impl Drop for Restore {
    fn drop(&mut self) {
        // SAFETY: single-threaded within this binary; the value is read on the
        // next strip call rather than cached.
        unsafe { std::env::remove_var(ENV) };
    }
}

#[test]
fn the_disable_switch_is_honoured_by_every_strip_entry_point() {
    let _r = Restore;
    // SAFETY: as above.
    let set = |v: &str| unsafe { std::env::set_var(ENV, v) };
    let clear = || unsafe { std::env::remove_var(ENV) };
    let w = wrapper();

    // Baseline: unset means ON, so the wrapper goes. Establish that first, or
    // an "off" assertion could pass against a heuristic that never fires.
    clear();
    let mut turns = vec![ChatTurn::User(&w), ChatTurn::User("real")];
    strip_client_meta_wrappers(&mut turns);
    assert_eq!(turns.len(), 1, "default is ON");

    // Every documented off-spelling, against all four entry points.
    for off in ["0", "false", "FALSE", "off", "OFF", "Off"] {
        set(off);

        let mut turns = vec![ChatTurn::User(&w), ChatTurn::User("real")];
        strip_client_meta_wrappers(&mut turns);
        assert_eq!(
            turns.len(),
            2,
            "{off:?} must disable strip_client_meta_wrappers"
        );

        let mut turns = vec![ChatTurn::User(&w), ChatTurn::User("real")];
        let kept = strip_client_meta_wrappers_indexed(&mut turns);
        assert_eq!(turns.len(), 2, "{off:?} must disable the indexed variant");
        assert_eq!(
            kept,
            vec![0, 1],
            "a disabled strip must still report every index, or a caller's side \
             table is filtered down to nothing"
        );

        let mut flat = vec![
            ("user".to_string(), w.clone()),
            ("user".to_string(), "real".to_string()),
        ];
        strip_client_meta_wrappers_flat(&mut flat);
        assert_eq!(flat.len(), 2, "{off:?} must disable the flat variant");

        let mut flat = vec![
            ("user".to_string(), w.clone()),
            ("user".to_string(), "real".to_string()),
        ];
        let kept = strip_client_meta_wrappers_flat_indexed(&mut flat);
        assert_eq!(
            flat.len(),
            2,
            "{off:?} must disable the flat indexed variant"
        );
        assert_eq!(kept, vec![0, 1]);
    }

    // Anything else means ON — the flag is opt-OUT, so a typo must not silently
    // disable a default-on behaviour.
    for on in ["1", "true", "yes", "", "no", "disabled", "0 "] {
        set(on);
        let mut turns = vec![ChatTurn::User(&w), ChatTurn::User("real")];
        strip_client_meta_wrappers(&mut turns);
        assert_eq!(
            turns.len(),
            1,
            "{on:?} is not a recognised off-value, so stripping must stay ON"
        );
    }
}
