//! Borrowed, on-demand tokens. Newlines delimit statements; nested grammar is
//! consumed by the parser, never by delimiter-counting string splitters.
use super::parser::ParseError;
use alloc::{collections::VecDeque, format, string::String};
use core::ops::Range;

/// Structural tokens only; keywords, types and literal values are decoded by
/// the parser and atom codecs, keeping the lexer independent of the opcode set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind {
    /// An uninterpreted name, mnemonic, keyword or literal (e.g. `iadd`, `-1`).
    Word,
    /// `(` / `)`: parameter, argument, result and type lists.
    LParen,
    RParen,
    /// `[` / `]`: successor lists.
    LBracket,
    RBracket,
    /// `<` / `>`: vector shapes such as `i32<scalable 4>`, not comparisons.
    Less,
    Greater,
    /// `,`: separates list entries, operands and named fields.
    Comma,
    /// `:`: type/signature annotations and block or stack-slot declarations.
    Colon,
    /// `=`: result definitions and named operand fields.
    Equal,
    /// `->`: separates signature parameters from return types.
    Arrow,
    /// A physical LF ends a statement; unlike other whitespace, it is retained.
    Newline,
    /// End of source, also permitting a final statement without a newline.
    Eof,
}

/// One-based source position. Columns count Unicode scalar values, not UTF-8
/// bytes or rendered display cells; a tab counts as one column.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Location {
    pub line: usize,
    pub column: usize,
}

impl Location {
    pub fn error(self, message: impl Into<String>) -> ParseError {
        ParseError {
            location: self,
            message: message.into(),
        }
    }
}

#[derive(Debug)]
struct Token {
    kind: Kind,
    /// UTF-8 byte range used to borrow the token spelling from the source.
    span: Range<usize>,
    location: Location,
}

/// Lookahead caches tokens; consuming them never scans their source again.
pub(super) struct Cursor<'a> {
    source: &'a str,
    // Scanner position: lookahead may place this beyond the current token.
    offset: usize,
    location: Location,
    // Parser position: advancing consumes cached tokens before scanning more.
    token: Token,
    ahead: VecDeque<Token>,
}

impl<'a> Cursor<'a> {
    pub fn new(source: &'a str) -> Self {
        let location = Location { line: 1, column: 1 };
        let mut cursor = Self {
            source,
            offset: 0,
            location,
            token: Token {
                kind: Kind::Eof,
                span: 0..0,
                location,
            },
            ahead: VecDeque::new(),
        };
        cursor.advance();
        cursor
    }

    pub fn kind(&self) -> Kind {
        self.token.kind
    }
    pub fn text(&self) -> &'a str {
        &self.source[self.token.span.clone()]
    }
    pub fn location(&self) -> Location {
        self.token.location
    }
    pub fn is(&self, word: &str) -> bool {
        self.kind() == Kind::Word && self.text() == word
    }

    fn bump(&mut self) {
        let ch = self.source[self.offset..].chars().next().expect("not EOF");
        self.offset += ch.len_utf8();
        if ch == '\n' {
            self.location.line += 1;
            self.location.column = 1;
        } else {
            self.location.column += 1;
        }
    }

    fn scan(&mut self) -> Token {
        // Skip spaces and comments, but leave LF for the statement parser.
        // In CRLF input the CR is skipped as whitespace and the LF ends the line.
        while self.offset < self.source.len() {
            let rest = &self.source[self.offset..];
            let ch = rest.chars().next().unwrap();
            if ch != '\n' && ch.is_whitespace() {
                self.bump();
            } else if rest.starts_with("//") {
                while self.offset < self.source.len()
                    && !self.source[self.offset..].starts_with('\n')
                {
                    self.bump();
                }
            } else {
                break;
            }
        }
        let start = self.offset;
        let location = self.location;
        if start == self.source.len() {
            return Token {
                kind: Kind::Eof,
                span: start..start,
                location,
            };
        }
        let kind = match self.source.as_bytes()[start] {
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
            b'-' if self.source[start..].starts_with("->") => Kind::Arrow,
            _ => Kind::Word,
        };
        if kind == Kind::Word {
            // Dots and hyphens belong to words (mnemonics, flags, SSA hints and
            // negative literals); only `->` gives a hyphen structural meaning.
            while self.offset < self.source.len() {
                let rest = &self.source[self.offset..];
                let ch = rest.chars().next().unwrap();
                if ch.is_whitespace()
                    || "()[]<>,:=".contains(ch)
                    || rest.starts_with("->")
                    || rest.starts_with("//")
                {
                    break;
                }
                self.bump();
            }
        } else {
            self.bump();
            if kind == Kind::Arrow {
                self.bump();
            }
        }
        Token {
            kind,
            span: start..self.offset,
            location,
        }
    }

    pub fn advance(&mut self) {
        self.token = self.ahead.pop_front().unwrap_or_else(|| self.scan());
    }

    /// Zero is the current token. Scanning ahead never advances parser position.
    fn peek(&mut self, distance: usize) -> &Token {
        if distance == 0 {
            return &self.token;
        }
        while self.ahead.len() < distance {
            let token = self.scan();
            self.ahead.push_back(token);
        }
        &self.ahead[distance - 1]
    }

    pub fn peek_kind(&mut self, distance: usize) -> Kind {
        self.peek(distance).kind
    }

    pub fn peek_is(&mut self, distance: usize, word: &str) -> bool {
        let token = self.peek(distance);
        let span = token.span.clone();
        token.kind == Kind::Word && &self.source[span] == word
    }

    pub fn at_end(&self) -> bool {
        matches!(self.kind(), Kind::Newline | Kind::Eof)
    }

    pub fn skip_newlines(&mut self) {
        while self.eat(Kind::Newline) {}
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
                    return Err(self.error(format!("expected {kind:?}, found `{}`", self.text())));
                }
            };
            Err(self.error(format!("expected {expected}, found `{}`", self.text())))
        }
    }

    pub fn word(&mut self) -> Result<&'a str, ParseError> {
        self.atom(Ok)
    }

    /// Decode a single token before advancing, so bad literals are diagnosed
    /// at their own span rather than at the following comma or end of line.
    pub fn atom<T>(
        &mut self,
        decode: impl FnOnce(&'a str) -> Result<T, String>,
    ) -> Result<T, ParseError> {
        if self.kind() != Kind::Word {
            return Err(self.error(format!("expected name or operand, found {:?}", self.kind())));
        }
        let value = decode(self.text()).map_err(|message| self.error(message))?;
        self.advance();
        Ok(value)
    }

    pub fn keyword(&mut self, word: &str) -> Result<(), ParseError> {
        if !self.is(word) {
            return Err(self.error(format!("expected `{word}`")));
        }
        self.advance();
        Ok(())
    }

    /// Check the statement boundary without consuming it; the outer parser
    /// loop owns newline consumption for declarations and instructions alike.
    pub fn finish(&self) -> Result<(), ParseError> {
        if self.at_end() {
            Ok(())
        } else {
            Err(self.error(format!(
                "unexpected operand or trailing text `{}`",
                self.text()
            )))
        }
    }

    pub fn named(&mut self) -> bool {
        self.named_at(0)
    }

    pub fn named_at(&mut self, distance: usize) -> bool {
        self.peek_kind(distance) == Kind::Word && self.peek_kind(distance + 1) == Kind::Equal
    }

    /// Cache alternate-layout lookahead up to the first top-level named field
    /// or statement boundary. Nested fields do not select a layout.
    /// Depth tracking is only a hint; the parser validates matching delimiters.
    pub fn has_named(&mut self) -> bool {
        let mut depth = 0usize;
        let mut distance = 0;
        loop {
            match self.peek_kind(distance) {
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
            distance += 1;
        }
    }

    pub fn error(&self, message: impl Into<String>) -> ParseError {
        self.location().error(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookahead_reuses_tokens_and_stops_at_the_newline() {
        let mut input = Cursor::new("v0, v1\nnext, mask=v2");
        assert!(!input.has_named());
        let scanned = input.offset;
        assert!(!input.has_named());
        for kind in [Kind::Comma, Kind::Word, Kind::Newline] {
            input.advance();
            assert_eq!(input.kind(), kind);
            assert_eq!(
                input.offset, scanned,
                "cached tokens must not rescan source"
            );
        }
        input.skip_newlines();
        assert_eq!(input.text(), "next");
        assert!(input.has_named());
        let scanned = input.offset;
        assert!(input.named_at(2));
        assert_eq!(input.offset, scanned);
        assert_eq!(input.error("test").to_string(), "line 2, column 1: test");
    }
}
