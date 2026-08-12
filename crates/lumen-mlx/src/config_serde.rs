//! Serde helpers shared by the per-family `config.json` parsers.
//!
//! Ungated so both parsers can use one copy, and so the behaviour is
//! unit-testable without the GPU stack.

use serde::Deserialize;

/// `#[serde(default)]` covers a **missing** key; it does not cover an explicit
/// `null`. Upstream exporters routinely spell an inapplicable field as `null`
/// rather than omitting it — a dense checkpoint carrying `"num_experts": null`
/// is the JGOS-31B shape — and a plain `#[serde(default)] usize` hard-fails on
/// it with `invalid type: null, expected usize at line N column M`, an error
/// that names neither the field nor the file.
///
/// Apply to every field that is legitimately absent on *some* architecture (the
/// MoE group on a dense config, the dense group on a MoE one) so that null and
/// missing mean the same thing: the default.
pub fn null_as_default<'de, D, T>(d: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(d)?.unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize, Debug, PartialEq)]
    struct Probe {
        #[serde(default, deserialize_with = "null_as_default")]
        n: usize,
        #[serde(default, deserialize_with = "null_as_default")]
        f: f32,
        #[serde(default, deserialize_with = "null_as_default")]
        s: String,
    }

    #[test]
    fn null_missing_and_present_all_behave() {
        // The three cases that must agree, and the one that must not change.
        let missing: Probe = serde_json::from_str("{}").expect("missing keys");
        let nulled: Probe =
            serde_json::from_str(r#"{"n":null,"f":null,"s":null}"#).expect("explicit nulls");
        assert_eq!(missing, nulled);
        assert_eq!(missing.n, 0);

        let present: Probe = serde_json::from_str(r#"{"n":7,"f":1.5,"s":"x"}"#).expect("present");
        assert_eq!(present.n, 7);
        assert_eq!(present.s, "x");
    }

    #[test]
    fn a_wrong_type_is_still_rejected() {
        // Null-tolerance must not become type-tolerance: a string where a
        // number belongs is a corrupt config, not an absent field.
        assert!(serde_json::from_str::<Probe>(r#"{"n":"twelve"}"#).is_err());
        assert!(serde_json::from_str::<Probe>(r#"{"n":-1}"#).is_err());
    }
}
