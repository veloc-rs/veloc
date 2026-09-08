//! Borrowed, on-demand tokens. Newlines delimit statements; nested grammar is
//! consumed by the parser, never by delimiter-counting string splitters.
use super::parser::ParseError;
use alloc::format;
use core::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    Word,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Less,
    Greater,
    Comma,
    Colon,
    Equal,
    Arrow,
    Newline,
    Eof,
}

#[derive(Clone, Debug)]
struct Token {
    kind: Kind,
    joined: bool,
    span: Range<usize>,
}

#[derive(Clone)]
pub(super) struct Cursor<'a> {
    source: &'a str,
    end: usize,
    token: Token,
}

impl<'a> Cursor<'a> {
    pub fn new(source: &'a str) -> Self {
        Self::range(source, 0..source.len())
    }

    pub fn range(source: &'a str, range: Range<usize>) -> Self {
        let mut cursor = Self {
            source,
            end: range.end,
            token: Token {
                kind: Kind::Eof,
                joined: false,
                span: range.start..range.start,
            },
        };
        cursor.advance();
        cursor
    }

    pub fn remaining(&self) -> Range<usize> {
        self.offset()..self.end
    }
    pub fn slice(&self, range: Range<usize>) -> Self {
        Self::range(self.source, range)
    }

    pub fn joined(&self) -> bool {
        self.token.joined
    }

    pub fn kind(&self) -> Kind {
        self.token.kind
    }
    pub fn text(&self) -> &'a str {
        &self.source[self.token.span.clone()]
    }
    pub fn offset(&self) -> usize {
        self.token.span.start
    }
    pub fn is(&self, word: &str) -> bool {
        self.kind() == Kind::Word && self.text() == word
    }

    pub fn advance(&mut self) {
        let mut start = self.token.span.end;
        let bytes = self.source.as_bytes();
        while start < self.end {
            let ch = self.source[start..self.end].chars().next().unwrap();
            if ch != '\n' && ch.is_whitespace() {
                start += ch.len_utf8();
            } else if self.source[start..self.end].starts_with("//") {
                start += self.source[start..self.end]
                    .find('\n')
                    .unwrap_or(self.end - start);
            } else {
                break;
            }
        }
        if start == self.end {
            self.token = Token {
                kind: Kind::Eof,
                joined: false,
                span: start..start,
            };
            return;
        }
        let kind = match bytes[start] {
            b'(' => Kind::LParen,
            b')' => Kind::RParen,
            b'[' => Kind::LBracket,
            b']' => Kind::RBracket,
            b'<' => Kind::Less,
            b'>' => Kind::Greater,
            b',' => Kind::Comma,
            b':' => Kind::Colon,
            b'=' => Kind::Equal,
            b'\n' => Kind::Newline,
            b'-' if self.source[start..self.end].starts_with("->") => Kind::Arrow,
            _ => Kind::Word,
        };
        let mut end = start + if kind == Kind::Arrow { 2 } else { 1 };
        if kind == Kind::Word {
            end = start;
            for (offset, ch) in self.source[start..self.end].char_indices() {
                if ch.is_whitespace()
                    || "()[]<>,:=".contains(ch)
                    || self.source[start + offset..self.end].starts_with("->")
                    || self.source[start + offset..self.end].starts_with("//")
                {
                    break;
                }
                end = start + offset + ch.len_utf8();
            }
        }
        self.token = Token {
            kind,
            joined: start == self.token.span.end,
            span: start..end,
        };
    }

    pub fn eat(&mut self, kind: Kind) -> bool {
        if self.kind() != kind {
            return false;
        }
        self.advance();
        true
    }

    pub fn expect(&mut self, kind: Kind) -> Result<(), ParseError> {
        if self.eat(kind) {
            Ok(())
        } else {
            let expected = match kind {
                Kind::Comma => "`,` before the next operand",
                Kind::Equal => "`=` in named field",
                Kind::LBracket => "`[` to start a [] list",
                _ => {
                    return Err(ParseError(format!(
                        "expected {kind:?}, found `{}`",
                        self.text()
                    )));
                }
            };
            Err(ParseError(format!(
                "expected {expected}, found `{}`",
                self.text()
            )))
        }
    }

    pub fn word(&mut self) -> Result<&'a str, ParseError> {
        self.atom(Ok)
    }

    /// Decode a single token before advancing, so bad literals are diagnosed
    /// at their own span rather than at the following comma or end of line.
    pub fn atom<T>(
        &mut self,
        decode: impl FnOnce(&'a str) -> Result<T, ParseError>,
    ) -> Result<T, ParseError> {
        if self.kind() != Kind::Word {
            return Err(ParseError(format!(
                "expected name or operand, found {:?}",
                self.kind()
            )));
        }
        let value = decode(self.text())?;
        self.advance();
        Ok(value)
    }

    pub fn keyword(&mut self, word: &str) -> Result<(), ParseError> {
        if !self.is(word) {
            return Err(ParseError(format!("expected `{word}`")));
        }
        self.advance();
        Ok(())
    }

    pub fn finish(&self) -> Result<(), ParseError> {
        if self.kind() == Kind::Eof {
            Ok(())
        } else {
            Err(ParseError(format!(
                "unexpected operand or trailing text `{}`",
                self.text()
            )))
        }
    }

    /// The grammar deliberately keeps physical newlines significant. These
    /// cursors borrow ranges of the original source, including absolute spans.
    pub fn statement(&mut self) -> Option<Self> {
        while self.eat(Kind::Newline) {}
        if self.kind() == Kind::Eof {
            return None;
        }
        let start = self.offset();
        let end = start
            + self.source[start..self.end]
                .find('\n')
                .unwrap_or(self.end - start);
        let statement = Self::range(self.source, start..end);
        self.token.span = end..end;
        self.advance();
        Some(statement)
    }

    pub fn named(&self) -> bool {
        let mut next = self.clone();
        next.kind() == Kind::Word && {
            next.advance();
            next.kind() == Kind::Equal
        }
    }

    /// Alternate layouts have required top-level named fields. Nested argument
    /// lists cannot select an alternate layout.
    pub fn has_named(&self) -> bool {
        let mut next = self.clone();
        let mut depth = 0usize;
        loop {
            match next.kind() {
                Kind::Eof | Kind::Newline => return false,
                Kind::Equal if depth == 0 => return true,
                Kind::LParen | Kind::LBracket | Kind::Less => depth += 1,
                Kind::RParen | Kind::RBracket | Kind::Greater => {
                    let Some(inner) = depth.checked_sub(1) else {
                        return false;
                    };
                    depth = inner;
                }
                _ => {}
            }
            next.advance();
        }
    }

    pub fn locate(&self, error: ParseError) -> ParseError {
        let prefix = &self.source[..self.offset()];
        let line = prefix.bytes().filter(|&b| b == b'\n').count() + 1;
        let start = prefix.rfind('\n').map_or(0, |i| i + 1);
        let column = self.source[start..self.offset()].chars().count() + 1;
        ParseError(format!("line {line}, column {column}: {}", error.0))
    }
}
