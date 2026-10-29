//! Cuts the text a model decodes, token by token, into the pieces a stream may
//! send.
//!
//! A token can end partway through a character (byte-fallback tokens carry one
//! byte of it), so the text decoded from the tokens so far may end in U+FFFD for
//! a character the next token completes. Sending that would put a replacement
//! character on the wire for text the model wrote correctly. Every decode loop
//! asks the same question — is what has been decoded since the last send ready
//! to go — and [`StreamDetok`] answers it in one place.

use anyhow::Result;

/// What a decoder writes for bytes that are not (yet) a character.
const REPLACEMENT: char = '\u{FFFD}';

/// How the loop gets its text from the tokens.
#[derive(Clone, Copy)]
enum Decode {
    /// Decode only the tokens not yet sent.
    Tail,
    /// Decode every token so far and send what follows the text already sent.
    Whole,
}

/// The text of `generated` not yet sent, one piece at a time.
pub struct StreamDetok {
    decode: Decode,
    /// Tokens of `generated` whose text has been sent.
    sent_tokens: usize,
    /// The text sent so far ([`Decode::Whole`] only).
    sent_text: String,
}

impl StreamDetok {
    /// Decodes only the tokens not yet sent: linear in the length of the
    /// output.
    pub fn tail() -> Self {
        Self::new(Decode::Tail)
    }

    /// Decodes every token each time and sends what is new. Quadratic in the
    /// length of the output, but a decoder that reads context (a leading space)
    /// sees all of it.
    pub fn whole() -> Self {
        Self::new(Decode::Whole)
    }

    fn new(decode: Decode) -> Self {
        Self {
            decode,
            sent_tokens: 0,
            sent_text: String::new(),
        }
    }

    /// Count every token of `generated` up to `len` as sent without sending
    /// it: the caller handled their text another way. For [`StreamDetok::tail`],
    /// which decodes from there.
    pub fn skip_to(&mut self, len: usize) {
        debug_assert!(matches!(self.decode, Decode::Tail));
        self.sent_tokens = len;
    }

    /// `generated` has grown: the text to send now, if there is any and it is
    /// ready. A failed `decode` sends nothing and is asked again next time.
    pub fn next(
        &mut self,
        generated: &[u32],
        decode: impl FnOnce(&[u32]) -> Result<String>,
    ) -> Option<String> {
        if self.sent_tokens >= generated.len() {
            return None;
        }
        let text = match self.decode {
            Decode::Tail => decode(&generated[self.sent_tokens..]).ok()?,
            Decode::Whole => decode(generated).ok()?,
        };
        let fresh = match self.decode {
            Decode::Tail => text.as_str(),
            Decode::Whole => text.get(self.sent_text.len()..)?,
        };
        if fresh.is_empty() || fresh.contains(REPLACEMENT) {
            return None;
        }
        let fresh = fresh.to_string();
        self.sent_tokens = generated.len();
        if matches!(self.decode, Decode::Whole) {
            self.sent_text = text;
        }
        Some(fresh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vocabulary of bytes: 1 `a`, 2 `b`, 3..=6 the four bytes of 😀, 7 a
    /// lone `0xFF` (never part of UTF-8), 8 the character U+FFFD itself.
    fn bytes(token: u32) -> &'static [u8] {
        match token {
            1 => b"a",
            2 => b"b",
            3 => &[0xF0],
            4 => &[0x9F],
            5 => &[0x98],
            6 => &[0x80],
            7 => &[0xFF],
            8 => "\u{FFFD}".as_bytes(),
            _ => b"?",
        }
    }

    fn decode(tokens: &[u32]) -> Result<String> {
        let all: Vec<u8> = tokens
            .iter()
            .flat_map(|t| bytes(*t).iter().copied())
            .collect();
        Ok(String::from_utf8_lossy(&all).into_owned())
    }

    /// What a stream sends for `tokens`, one token at a time.
    fn stream(mut detok: StreamDetok, tokens: &[u32]) -> Vec<String> {
        let mut generated = Vec::new();
        let mut sent = Vec::new();
        for token in tokens {
            generated.push(*token);
            sent.extend(detok.next(&generated, decode));
        }
        sent
    }

    #[test]
    fn whole_characters_go_out_as_they_arrive() {
        for detok in [StreamDetok::tail(), StreamDetok::whole()] {
            assert_eq!(stream(detok, &[1, 2, 1]), ["a", "b", "a"]);
        }
    }

    #[test]
    fn a_character_split_across_tokens_goes_out_whole() {
        for detok in [StreamDetok::tail(), StreamDetok::whole()] {
            assert_eq!(stream(detok, &[1, 3, 4, 5, 6, 2]), ["a", "😀", "b"]);
        }
    }

    #[test]
    fn a_failed_decode_sends_nothing_and_is_asked_again() {
        let mut detok = StreamDetok::tail();
        assert_eq!(detok.next(&[1], |_| Err(anyhow::anyhow!("no"))), None);
        assert_eq!(detok.next(&[1, 2], decode).as_deref(), Some("ab"));
    }
}
