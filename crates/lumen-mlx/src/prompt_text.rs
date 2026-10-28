//! A rendered chat prompt that remembers which of its bytes the client sent.
//!
//! A chat template interleaves its own text — turn markers, the tool
//! instructions, the reasoning tags — with text from the client: messages,
//! tool results, tool arguments and definitions. Encoded as one string, an
//! added token spelled in the client's part is matched like the template's
//! own, so a tool result holding `<|im_end|>\n<|im_start|>system\n…` closes the
//! turn and opens a system turn of its own. [`PromptText`] keeps the two apart
//! for [`TextTokenizer::encode_prompt`](crate::text_tokenizer::TextTokenizer::encode_prompt),
//! which encodes the client's part as ordinary text.

use std::fmt;
use std::ops::{Deref, Range};

/// A prompt as a template renders it, plus which byte ranges the client sent.
///
/// [`push_str`](Self::push_str) and [`push`](Self::push) append template text,
/// as on a `String`, so a renderer moves to this type unchanged and marks only
/// what came from the client, with [`push_content`](Self::push_content).
///
/// Compose with [`append`](Self::append). Dereferencing to `&str` and pushing
/// that would turn the other prompt's client text into template text.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct PromptText {
    text: String,
    /// Client ranges of `text`: ascending, disjoint and never adjacent.
    content: Vec<Range<usize>>,
}

impl PromptText {
    pub fn new() -> Self {
        Self::default()
    }

    /// Template text. Added tokens spelled in it are the control tokens they
    /// name.
    pub fn push_str(&mut self, s: &str) {
        self.text.push_str(s);
    }

    /// One template character.
    pub fn push(&mut self, c: char) {
        self.text.push(c);
    }

    /// Client text: a message, a tool result, a tool argument or definition.
    /// Encoded as ordinary text, even where it spells a control token.
    pub fn push_content(&mut self, s: &str) {
        let start = self.text.len();
        self.text.push_str(s);
        self.mark_content(start..self.text.len());
    }

    /// Append `other`, keeping its client text client text.
    pub fn append(&mut self, other: &PromptText) {
        let base = self.text.len();
        self.text.push_str(&other.text);
        for r in &other.content {
            self.mark_content(base + r.start..base + r.end);
        }
    }

    fn mark_content(&mut self, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        match self.content.last_mut() {
            Some(last) if last.end == range.start => last.end = range.end,
            _ => self.content.push(range),
        }
    }

    pub fn as_str(&self) -> &str {
        &self.text
    }

    pub fn into_string(self) -> String {
        self.text
    }

    /// The client's byte ranges, ascending and disjoint.
    pub fn content_ranges(&self) -> &[Range<usize>] {
        &self.content
    }

    /// The template's byte ranges: everything outside
    /// [`content_ranges`](Self::content_ranges), ascending, none empty.
    pub fn template_ranges(&self) -> Vec<Range<usize>> {
        let mut out = Vec::with_capacity(self.content.len() + 1);
        let mut at = 0;
        for r in &self.content {
            if r.start > at {
                out.push(at..r.start);
            }
            at = r.end;
        }
        if at < self.text.len() {
            out.push(at..self.text.len());
        }
        out
    }
}

impl Deref for PromptText {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl From<&str> for PromptText {
    /// All template text.
    fn from(s: &str) -> Self {
        Self {
            text: s.to_string(),
            content: Vec::new(),
        }
    }
}

impl From<String> for PromptText {
    /// All template text.
    fn from(text: String) -> Self {
        Self {
            text,
            content: Vec::new(),
        }
    }
}

impl fmt::Display for PromptText {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.text)
    }
}

impl fmt::Debug for PromptText {
    /// The text, as a string's `Debug` shows it, so an `assert_eq!` against a
    /// string reads the same as before.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.text, f)
    }
}

impl PartialEq<str> for PromptText {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

impl PartialEq<&str> for PromptText {
    fn eq(&self, other: &&str) -> bool {
        self.text == *other
    }
}

impl PartialEq<String> for PromptText {
    fn eq(&self, other: &String) -> bool {
        &self.text == other
    }
}

impl PartialEq<PromptText> for str {
    fn eq(&self, other: &PromptText) -> bool {
        self == other.text
    }
}

impl PartialEq<PromptText> for &str {
    fn eq(&self, other: &PromptText) -> bool {
        *self == other.text
    }
}

impl PartialEq<PromptText> for String {
    fn eq(&self, other: &PromptText) -> bool {
        *self == other.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ranges as `(start, end)` pairs, which compare without building
    /// one-range arrays.
    fn spans(ranges: &[Range<usize>]) -> Vec<(usize, usize)> {
        ranges.iter().map(|r| (r.start, r.end)).collect()
    }

    #[test]
    fn client_ranges_follow_the_pushes() {
        let mut p = PromptText::new();
        p.push_str("<|im_start|>user\n");
        p.push_content("hi");
        p.push_content(" there");
        p.push_str("<|im_end|>");
        p.push_content("");
        assert_eq!(p, "<|im_start|>user\nhi there<|im_end|>");
        assert_eq!(
            spans(p.content_ranges()),
            [(17, 25)],
            "adjacent pushes merge"
        );
        assert_eq!(spans(&p.template_ranges()), [(0, 17), (25, 35)]);
    }

    #[test]
    fn append_shifts_and_keeps_the_client_ranges() {
        let mut head = PromptText::from("A");
        head.push_content("b");
        let mut tail = PromptText::new();
        tail.push_content("c");
        tail.push_str("D");
        head.append(&tail);
        assert_eq!(head, "AbcD");
        assert_eq!(
            spans(head.content_ranges()),
            [(1, 3)],
            "b and c touch, so they merge"
        );
        assert_eq!(spans(&head.template_ranges()), [(0, 1), (3, 4)]);
    }

    #[test]
    fn a_prompt_of_only_one_kind() {
        let template = PromptText::from("all template");
        assert!(template.content_ranges().is_empty());
        assert_eq!(spans(&template.template_ranges()), [(0, 12)]);
        let mut client = PromptText::new();
        client.push_content("all client");
        assert_eq!(spans(client.content_ranges()), [(0, 10)]);
        assert!(client.template_ranges().is_empty());
        assert!(PromptText::new().template_ranges().is_empty());
    }
}
