//! Cuts the text a model decodes, token by token, into the pieces a stream may
//! send.
//!
//! A token can end partway through a character (byte-fallback tokens carry one
//! byte of it), so the text decoded from the tokens so far may end in U+FFFD for
//! a character the next token completes. Sending that would put a replacement
//! character on the wire for text the model wrote correctly. Every decode loop
//! asks the same question — is what has been decoded since the last send ready
//! to go — and [`StreamDetok`] answers it in one place.
//!
//! The text is held back only while it *ends* in U+FFFD, and only for as long as
//! the character could still be completed: a UTF-8 character is at most four
//! bytes and a byte-fallback token carries one. A replacement character the
//! model really wrote (an echoed file, a stray byte) is in the middle of text
//! soon enough, or comes out after four tokens, and is sent like any other. A
//! stream that held every tail containing one would send nothing after it.

use anyhow::Result;

/// What a decoder writes for bytes that are not (yet) a character.
const REPLACEMENT: char = '\u{FFFD}';

/// The most tokens an incomplete character can span: UTF-8 is at most four
/// bytes, and a token carries at least one of them.
const MAX_HELD_TOKENS: usize = 4;

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
    /// How many times in a row the text has been held for a trailing U+FFFD:
    /// counted per hold rather than per token so that a stream with two
    /// channels, whose tokens mostly belong to the other one, is not made to
    /// send half a character by a count of tokens that were never its own.
    held: usize,
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
            held: 0,
        }
    }

    /// Count every token of `generated` up to `len` as sent without sending
    /// it: the caller handled their text another way. For [`StreamDetok::tail`],
    /// which decodes from there.
    pub fn skip_to(&mut self, len: usize) {
        debug_assert!(matches!(self.decode, Decode::Tail));
        self.sent_tokens = len;
        self.held = 0;
    }

    /// `generated` has grown: the text to send now, if there is any and it is
    /// ready. A failed `decode` sends nothing and is asked again next time.
    pub fn next(
        &mut self,
        generated: &[u32],
        decode: impl FnOnce(&[u32]) -> Result<String>,
    ) -> Option<String> {
        self.take(generated, decode, true)
    }

    /// The stream is over: whatever is still held, replacement characters and
    /// all. Without this the last few tokens of a reply that ends in one never
    /// go out.
    pub fn finish(
        &mut self,
        generated: &[u32],
        decode: impl FnOnce(&[u32]) -> Result<String>,
    ) -> Option<String> {
        self.take(generated, decode, false)
    }

    fn take(
        &mut self,
        generated: &[u32],
        decode: impl FnOnce(&[u32]) -> Result<String>,
        may_hold: bool,
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
        if fresh.is_empty() {
            return None;
        }
        if may_hold && self.held + 1 < MAX_HELD_TOKENS && fresh.ends_with(REPLACEMENT) {
            self.held += 1;
            return None;
        }
        self.held = 0;
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

    /// A whole tool call, as one token.
    const CALL: &str = "<tool_call>\n<function=read>\n<parameter=path>\nsrc/lib.rs\n\
                        </parameter>\n</function>\n</tool_call>";

    /// A vocabulary of bytes: 1 `a`, 2 `b`, 3..=6 the four bytes of 😀, 7 a
    /// lone `0xFF` (never part of UTF-8), 8 the character U+FFFD itself, 100 a tool
    /// call.
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
            100 => CALL.as_bytes(),
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

    /// A replacement character the model really wrote — an echoed file, a
    /// stray byte — is text like any other. It used to end the stream: the
    /// tail that held it never stopped containing it, so nothing after it was
    /// ever sent.
    #[test]
    fn a_genuine_replacement_character_does_not_end_the_stream() {
        for detok in [StreamDetok::tail(), StreamDetok::whole()] {
            let sent = stream(detok, &[1, 8, 2, 1, 2, 1]);
            assert_eq!(sent.concat(), "a\u{FFFD}baba", "sent {sent:?}");
        }
        for detok in [StreamDetok::tail(), StreamDetok::whole()] {
            let sent = stream(detok, &[1, 7, 2, 1, 2, 1]);
            assert_eq!(sent.concat(), "a\u{FFFD}baba", "sent {sent:?}");
        }
    }

    /// No run of invalid bytes holds the stream for longer than the four
    /// tokens an incomplete character can span.
    #[test]
    fn a_run_of_invalid_bytes_is_not_held_for_ever() {
        for detok in [StreamDetok::tail(), StreamDetok::whole()] {
            let sent = stream(detok, &[7, 7, 7, 7, 7, 7, 1]);
            assert_eq!(
                sent.concat(),
                "\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}\u{FFFD}a"
            );
            assert!(sent.len() >= 2, "held to the end: {sent:?}");
        }
    }

    #[test]
    fn the_end_of_the_stream_sends_what_was_held() {
        for detok in [StreamDetok::tail(), StreamDetok::whole()] {
            let mut detok = detok;
            let mut generated = Vec::new();
            let mut sent = Vec::new();
            for token in [1, 3, 4] {
                generated.push(token);
                sent.extend(detok.next(&generated, decode));
            }
            assert_eq!(sent, ["a"], "the half character is held");
            assert_eq!(
                detok.finish(&generated, decode).as_deref(),
                Some("\u{FFFD}")
            );
            assert_eq!(detok.finish(&generated, decode), None, "once");
        }
    }

    /// What the tools loop does with it: the text goes to the Qwen parser, and
    /// a call written after a genuine replacement character still arrives.
    #[test]
    fn a_tool_call_after_a_replacement_character_reaches_the_parser() {
        use crate::qwen3_5_tools::Qwen35ResponseParser;
        let mut detok = StreamDetok::tail();
        let mut parser = Qwen35ResponseParser::new();
        let mut generated = Vec::new();
        for token in [1, 8, 2, 100] {
            generated.push(token);
            if let Some(text) = detok.next(&generated, decode) {
                parser.feed(&text);
            }
        }
        let response = parser.finish();
        assert_eq!(
            response.tool_calls.len(),
            1,
            "visible: {:?}",
            response.visible
        );
        assert_eq!(response.tool_calls[0].name, "read");
    }
}
