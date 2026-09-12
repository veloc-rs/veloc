//! Tokens borrow names and numbers; string literals are decoded once.
//! Byte offsets always refer to the original file, including comments/imports.
use crate::Error;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Kind<'a> {
    Name(&'a str),
    Number(&'a str),
    Text(String),
    Symbol(&'static str),
    Eof,
}

pub(super) struct Token<'a> {
    pub offset: usize,
    pub kind: Kind<'a>,
}

pub(super) struct Lexer<'a> {
    source: &'a str,
    offset: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self { source, offset: 0 }
    }

    pub fn next(&mut self) -> Result<Token<'a>, Error> {
        let bytes = self.source.as_bytes();
        loop {
            while bytes.get(self.offset).is_some_and(u8::is_ascii_whitespace) {
                self.offset += 1;
            }
            if !self.source[self.offset..].starts_with("//") {
                break;
            }
            while bytes.get(self.offset).is_some_and(|&b| b != b'\n') {
                self.offset += 1;
            }
        }
        let offset = self.offset;
        let Some(&byte) = bytes.get(offset) else {
            return Ok(Token {
                offset,
                kind: Kind::Eof,
            });
        };
        let kind = match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                self.offset += 1;
                while bytes
                    .get(self.offset)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.'))
                {
                    self.offset += 1;
                }
                Kind::Name(&self.source[offset..self.offset])
            }
            b'0'..=b'9' => {
                self.offset += 1;
                while bytes.get(self.offset).is_some_and(u8::is_ascii_digit) {
                    self.offset += 1;
                }
                Kind::Number(&self.source[offset..self.offset])
            }
            b'"' => Kind::Text(self.string()?),
            _ => {
                // Maximal munch makes ->, ||, && and comparisons indivisible.
                let tail = &self.source[offset..];
                let symbol = [
                    "->", "||", "&&", "==", "!=", "<=", ">=", "{", "}", "(", ")", "[", "]", "<",
                    ">", ":", ",", ";", "=", "@", "|", "&", "!", "-", "+", "*", "?",
                ]
                .into_iter()
                .find(|s| tail.starts_with(s))
                .ok_or_else(|| Error::at(self.source, offset, "unexpected character"))?;
                self.offset += symbol.len();
                Kind::Symbol(symbol)
            }
        };
        Ok(Token { offset, kind })
    }

    fn string(&mut self) -> Result<String, Error> {
        let start = self.offset;
        self.offset += 1;
        let mut text = String::new();
        loop {
            let Some(ch) = self.source[self.offset..].chars().next() else {
                return Err(Error::at(self.source, start, "unterminated string"));
            };
            self.offset += ch.len_utf8();
            match ch {
                '"' => return Ok(text),
                '\\' => {
                    let ch = match self.source.as_bytes().get(self.offset) {
                        Some(b'"') => '"',
                        Some(b'\\') => '\\',
                        Some(b'n') => '\n',
                        Some(b'r') => '\r',
                        Some(b't') => '\t',
                        _ => {
                            return Err(Error::at(
                                self.source,
                                self.offset,
                                "invalid string escape",
                            ));
                        }
                    };
                    self.offset += 1;
                    text.push(ch);
                }
                '\n' | '\r' => return Err(Error::at(self.source, start, "newline in string")),
                _ => text.push(ch),
            }
        }
    }
}
