//! Tokens borrow names and numbers; string literals are decoded once.
//! Byte offsets always refer to the original file, including comments/imports.
use crate::Error;

#[derive(Debug, PartialEq, Eq)]
pub(super) enum Kind<'a> {
    Fn,
    Const,
    Import,
    Type,
    TypeSet,
    Op,
    SelfValue,
    Let,
    Name(&'a str),
    Number(&'a str),
    Text(String),
    ColonColon,
    Arrow,
    FatArrow,
    PipePipe,
    AmpAmp,
    EqEq,
    NotEq,
    Le,
    Ge,
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Lt,
    Gt,
    Colon,
    Comma,
    Semi,
    Eq,
    At,
    Pipe,
    Amp,
    Bang,
    Minus,
    Plus,
    Star,
    Question,
    Dot,
    Eof,
}

impl<'a> Kind<'a> {
    fn word(text: &'a str) -> Self {
        match text {
            "fn" => Self::Fn,
            "const" => Self::Const,
            "import" => Self::Import,
            "type" => Self::Type,
            "typeset" => Self::TypeSet,
            "op" => Self::Op,
            "self" => Self::SelfValue,
            "let" => Self::Let,
            _ => Self::Name(text),
        }
    }

    /// Fixed token spelling for diagnostics and AST operator names.
    pub fn spelling(&self) -> &'static str {
        match self {
            Self::ColonColon => "::",
            Self::Arrow => "->",
            Self::FatArrow => "=>",
            Self::PipePipe => "||",
            Self::AmpAmp => "&&",
            Self::EqEq => "==",
            Self::NotEq => "!=",
            Self::Le => "<=",
            Self::Ge => ">=",
            Self::LBrace => "{",
            Self::RBrace => "}",
            Self::LParen => "(",
            Self::RParen => ")",
            Self::LBracket => "[",
            Self::RBracket => "]",
            Self::Lt => "<",
            Self::Gt => ">",
            Self::Colon => ":",
            Self::Comma => ",",
            Self::Semi => ";",
            Self::Eq => "=",
            Self::At => "@",
            Self::Pipe => "|",
            Self::Amp => "&",
            Self::Bang => "!",
            Self::Minus => "-",
            Self::Plus => "+",
            Self::Star => "*",
            Self::Question => "?",
            Self::Dot => ".",
            Self::Fn => "fn",
            Self::Const => "const",
            Self::Import => "import",
            Self::Type => "type",
            Self::TypeSet => "typeset",
            Self::Op => "op",
            Self::SelfValue => "self",
            Self::Let => "let",
            Self::Name(_) => "name",
            Self::Number(_) => "number",
            Self::Text(_) => "string",
            Self::Eof => "end of file",
        }
    }

    /// Keywords remain usable as names outside their grammatical positions.
    /// Property names are ordinary names, not lexer keywords.
    pub fn name(&self) -> Option<&'a str> {
        Some(match self {
            Self::Fn
            | Self::Const
            | Self::Import
            | Self::Type
            | Self::TypeSet
            | Self::Op
            | Self::SelfValue
            | Self::Let => self.spelling(),
            Self::Name(name) => name,
            _ => return None,
        })
    }
}

pub(super) struct Token<'a> {
    pub offset: usize,
    pub kind: Kind<'a>,
}

#[derive(Clone)]
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
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
                {
                    self.offset += 1;
                }
                Kind::word(&self.source[offset..self.offset])
            }
            b'0'..=b'9' => {
                self.offset += 1;
                while bytes
                    .get(self.offset)
                    .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_')
                {
                    self.offset += 1;
                }
                Kind::Number(&self.source[offset..self.offset])
            }
            b'"' => Kind::Text(self.string()?),
            _ => {
                // Recognize two-byte tokens before their one-byte prefixes.
                let kind = match (byte, bytes.get(offset + 1).copied()) {
                    (b':', Some(b':')) => Kind::ColonColon,
                    (b'-', Some(b'>')) => Kind::Arrow,
                    (b'=', Some(b'>')) => Kind::FatArrow,
                    (b'|', Some(b'|')) => Kind::PipePipe,
                    (b'&', Some(b'&')) => Kind::AmpAmp,
                    (b'=', Some(b'=')) => Kind::EqEq,
                    (b'!', Some(b'=')) => Kind::NotEq,
                    (b'<', Some(b'=')) => Kind::Le,
                    (b'>', Some(b'=')) => Kind::Ge,
                    (b'{', _) => Kind::LBrace,
                    (b'}', _) => Kind::RBrace,
                    (b'(', _) => Kind::LParen,
                    (b')', _) => Kind::RParen,
                    (b'[', _) => Kind::LBracket,
                    (b']', _) => Kind::RBracket,
                    (b'<', _) => Kind::Lt,
                    (b'>', _) => Kind::Gt,
                    (b':', _) => Kind::Colon,
                    (b',', _) => Kind::Comma,
                    (b';', _) => Kind::Semi,
                    (b'=', _) => Kind::Eq,
                    (b'@', _) => Kind::At,
                    (b'|', _) => Kind::Pipe,
                    (b'&', _) => Kind::Amp,
                    (b'!', _) => Kind::Bang,
                    (b'-', _) => Kind::Minus,
                    (b'+', _) => Kind::Plus,
                    (b'*', _) => Kind::Star,
                    (b'?', _) => Kind::Question,
                    (b'.', _) => Kind::Dot,
                    _ => return Err(Error::at(self.source, offset, "unexpected character")),
                };
                self.offset += kind.spelling().len();
                kind
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
