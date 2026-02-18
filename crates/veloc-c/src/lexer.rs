//! C Language Lexer
//!
//! Tokenizes C source code into a stream of tokens.

use crate::error::{Error, Result};
use std::format;
use std::string::String;
use std::vec::Vec;

/// Token kind enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TokenKind {
    // Literals
    Integer(i64),
    UnsignedInteger(u64),
    FloatLit(f64),
    CharLit(char),
    StringLit,
    Identifier,

    // Keywords
    Auto,
    Break,
    Case,
    CharKw,
    Const,
    Continue,
    Default,
    Do,
    Double,
    Else,
    Enum,
    Extern,
    FloatKw,
    For,
    Goto,
    If,
    Inline,
    Int,
    Long,
    Register,
    Restrict,
    Return,
    Short,
    Signed,
    Sizeof,
    Static,
    Struct,
    Switch,
    Typedef,
    Union,
    Unsigned,
    Void,
    Volatile,
    While,

    // Operators
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Percent,     // %
    Increment,   // ++
    Decrement,   // --
    Assign,      // =
    PlusAssign,  // +=
    MinusAssign, // -=
    MulAssign,   // *=
    DivAssign,   // /=
    ModAssign,   // %=
    Eq,          // ==
    Ne,          // !=
    Lt,          // <
    Le,          // <=
    Gt,          // >
    Ge,          // >=
    And,         // &
    Or,          // |
    Xor,         // ^
    Not,         // !
    LogicalAnd,  // &&
    LogicalOr,   // ||
    ShiftLeft,   // <<
    ShiftRight,  // >>
    Tilde,       // ~
    Arrow,       // ->
    Dot,         // .
    Comma,       // ,
    Colon,       // :
    Semicolon,   // ;
    Question,    // ?

    // Delimiters
    LParen,   // (
    RParen,   // )
    LBrace,   // {
    RBrace,   // }
    LBracket, // [
    RBracket, // ]

    // Preprocessor
    Hash, // #

    // Special
    Eof,
    NewLine,
    Whitespace,
    Comment,
}

impl TokenKind {
    /// Check if this token kind is a type specifier keyword
    pub fn is_type_specifier(&self) -> bool {
        matches!(
            self,
            TokenKind::Void
                | TokenKind::CharKw
                | TokenKind::Short
                | TokenKind::Int
                | TokenKind::Long
                | TokenKind::FloatKw
                | TokenKind::Double
                | TokenKind::Signed
                | TokenKind::Unsigned
        )
    }

    /// Check if this is a storage class specifier
    pub fn is_storage_class(&self) -> bool {
        matches!(
            self,
            TokenKind::Typedef
                | TokenKind::Extern
                | TokenKind::Static
                | TokenKind::Auto
                | TokenKind::Register
        )
    }
}

/// Token structure
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub lexeme: String,
    pub line: u32,
    pub column: u32,
}

impl Token {
    pub fn new(kind: TokenKind, lexeme: impl Into<String>, line: u32, column: u32) -> Self {
        Token {
            kind,
            lexeme: lexeme.into(),
            line,
            column,
        }
    }
}

/// C Language Lexer
pub struct Lexer<'a> {
    source: &'a str,
    chars: std::str::Chars<'a>,
    current: Option<char>,
    line: u32,
    column: u32,
    peeked: Vec<Token>,
}

impl<'a> Lexer<'a> {
    /// Create a new lexer from source code
    pub fn new(source: &'a str) -> Self {
        let mut chars = source.chars();
        let current = chars.next();
        Lexer {
            source,
            chars,
            current,
            line: 1,
            column: 1,
            peeked: Vec::new(),
        }
    }

    /// Get the next token
    pub fn next_token(&mut self) -> Result<Token> {
        if let Some(token) = self.peeked.pop() {
            return Ok(token);
        }
        self.skip_whitespace();
        self.read_token()
    }

    /// Peek at the next token without consuming it
    pub fn peek_token(&mut self) -> Result<Token> {
        if self.peeked.is_empty() {
            let token = self.next_token()?;
            self.peeked.push(token);
        }
        Ok(self.peeked.last().unwrap().clone())
    }

    /// Peek at the nth token ahead
    pub fn peek_nth(&mut self, n: usize) -> Result<Token> {
        while self.peeked.len() <= n {
            let token = self.next_token()?;
            self.peeked.push(token);
        }
        Ok(self.peeked[n].clone())
    }

    /// Check if we've reached the end of file
    pub fn is_eof(&self) -> bool {
        self.current.is_none()
    }

    /// Get current line number
    pub fn line(&self) -> u32 {
        self.line
    }

    /// Get current column number
    pub fn column(&self) -> u32 {
        self.column
    }

    /// Advance to the next character
    fn advance(&mut self) -> Option<char> {
        let prev = self.current;
        if let Some(c) = prev {
            if c == '\n' {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
        }
        self.current = self.chars.next();
        prev
    }

    /// Peek at the next character without consuming
    fn peek_char(&self) -> Option<char> {
        self.current
    }

    /// Peek at the second next character
    fn peek_char2(&self) -> Option<char> {
        self.chars.clone().next()
    }

    /// Skip whitespace characters
    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_ascii_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    /// Read the next token
    fn read_token(&mut self) -> Result<Token> {
        let start_line = self.line;
        let start_col = self.column;

        match self.peek_char() {
            None => Ok(Token::new(TokenKind::Eof, "", start_line, start_col)),
            Some(c) => {
                match c {
                    // Single character tokens
                    '+' => {
                        self.advance();
                        match self.peek_char() {
                            Some('+') => {
                                self.advance();
                                Ok(Token::new(
                                    TokenKind::Increment,
                                    "++",
                                    start_line,
                                    start_col,
                                ))
                            }
                            Some('=') => {
                                self.advance();
                                Ok(Token::new(
                                    TokenKind::PlusAssign,
                                    "+=",
                                    start_line,
                                    start_col,
                                ))
                            }
                            _ => Ok(Token::new(TokenKind::Plus, "+", start_line, start_col)),
                        }
                    }
                    '-' => {
                        self.advance();
                        match self.peek_char() {
                            Some('-') => {
                                self.advance();
                                Ok(Token::new(
                                    TokenKind::Decrement,
                                    "--",
                                    start_line,
                                    start_col,
                                ))
                            }
                            Some('=') => {
                                self.advance();
                                Ok(Token::new(
                                    TokenKind::MinusAssign,
                                    "-=",
                                    start_line,
                                    start_col,
                                ))
                            }
                            Some('>') => {
                                self.advance();
                                Ok(Token::new(TokenKind::Arrow, "->", start_line, start_col))
                            }
                            _ => Ok(Token::new(TokenKind::Minus, "-", start_line, start_col)),
                        }
                    }
                    '*' => {
                        self.advance();
                        if self.peek_char() == Some('=') {
                            self.advance();
                            Ok(Token::new(
                                TokenKind::MulAssign,
                                "*=",
                                start_line,
                                start_col,
                            ))
                        } else {
                            Ok(Token::new(TokenKind::Star, "*", start_line, start_col))
                        }
                    }
                    '/' => {
                        self.advance();
                        match self.peek_char() {
                            Some('/') => self.read_line_comment(start_line, start_col),
                            Some('*') => self.read_block_comment(start_line, start_col),
                            Some('=') => {
                                self.advance();
                                Ok(Token::new(
                                    TokenKind::DivAssign,
                                    "/=",
                                    start_line,
                                    start_col,
                                ))
                            }
                            _ => Ok(Token::new(TokenKind::Slash, "/", start_line, start_col)),
                        }
                    }
                    '%' => {
                        self.advance();
                        if self.peek_char() == Some('=') {
                            self.advance();
                            Ok(Token::new(
                                TokenKind::ModAssign,
                                "%=",
                                start_line,
                                start_col,
                            ))
                        } else {
                            Ok(Token::new(TokenKind::Percent, "%", start_line, start_col))
                        }
                    }
                    '=' => {
                        self.advance();
                        if self.peek_char() == Some('=') {
                            self.advance();
                            Ok(Token::new(TokenKind::Eq, "==", start_line, start_col))
                        } else {
                            Ok(Token::new(TokenKind::Assign, "=", start_line, start_col))
                        }
                    }
                    '!' => {
                        self.advance();
                        if self.peek_char() == Some('=') {
                            self.advance();
                            Ok(Token::new(TokenKind::Ne, "!=", start_line, start_col))
                        } else {
                            Ok(Token::new(TokenKind::Not, "!", start_line, start_col))
                        }
                    }
                    '<' => {
                        self.advance();
                        match self.peek_char() {
                            Some('=') => {
                                self.advance();
                                Ok(Token::new(TokenKind::Le, "<=", start_line, start_col))
                            }
                            Some('<') => {
                                self.advance();
                                Ok(Token::new(
                                    TokenKind::ShiftLeft,
                                    "<<",
                                    start_line,
                                    start_col,
                                ))
                            }
                            _ => Ok(Token::new(TokenKind::Lt, "<", start_line, start_col)),
                        }
                    }
                    '>' => {
                        self.advance();
                        match self.peek_char() {
                            Some('=') => {
                                self.advance();
                                Ok(Token::new(TokenKind::Ge, ">=", start_line, start_col))
                            }
                            Some('>') => {
                                self.advance();
                                Ok(Token::new(
                                    TokenKind::ShiftRight,
                                    ">>",
                                    start_line,
                                    start_col,
                                ))
                            }
                            _ => Ok(Token::new(TokenKind::Gt, ">", start_line, start_col)),
                        }
                    }
                    '&' => {
                        self.advance();
                        if self.peek_char() == Some('&') {
                            self.advance();
                            Ok(Token::new(
                                TokenKind::LogicalAnd,
                                "&&",
                                start_line,
                                start_col,
                            ))
                        } else {
                            Ok(Token::new(TokenKind::And, "&", start_line, start_col))
                        }
                    }
                    '|' => {
                        self.advance();
                        if self.peek_char() == Some('|') {
                            self.advance();
                            Ok(Token::new(
                                TokenKind::LogicalOr,
                                "||",
                                start_line,
                                start_col,
                            ))
                        } else {
                            Ok(Token::new(TokenKind::Or, "|", start_line, start_col))
                        }
                    }
                    '^' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Xor, "^", start_line, start_col))
                    }
                    '~' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Tilde, "~", start_line, start_col))
                    }
                    '.' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Dot, ".", start_line, start_col))
                    }
                    ',' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Comma, ",", start_line, start_col))
                    }
                    ':' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Colon, ":", start_line, start_col))
                    }
                    ';' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Semicolon, ";", start_line, start_col))
                    }
                    '?' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Question, "?", start_line, start_col))
                    }
                    '(' => {
                        self.advance();
                        Ok(Token::new(TokenKind::LParen, "(", start_line, start_col))
                    }
                    ')' => {
                        self.advance();
                        Ok(Token::new(TokenKind::RParen, ")", start_line, start_col))
                    }
                    '{' => {
                        self.advance();
                        Ok(Token::new(TokenKind::LBrace, "{", start_line, start_col))
                    }
                    '}' => {
                        self.advance();
                        Ok(Token::new(TokenKind::RBrace, "}", start_line, start_col))
                    }
                    '[' => {
                        self.advance();
                        Ok(Token::new(TokenKind::LBracket, "[", start_line, start_col))
                    }
                    ']' => {
                        self.advance();
                        Ok(Token::new(TokenKind::RBracket, "]", start_line, start_col))
                    }
                    '#' => {
                        self.advance();
                        Ok(Token::new(TokenKind::Hash, "#", start_line, start_col))
                    }
                    '"' => self.read_string(start_line, start_col),
                    '\'' => self.read_char(start_line, start_col),
                    c if c.is_ascii_alphabetic() || c == '_' => {
                        self.read_identifier(start_line, start_col)
                    }
                    c if c.is_ascii_digit() => self.read_number(start_line, start_col),
                    _ => Err(Error::lexical(
                        format!("Unexpected character: {}", c),
                        start_line,
                        start_col,
                    )),
                }
            }
        }
    }

    /// Read an identifier or keyword
    fn read_identifier(&mut self, start_line: u32, start_col: u32) -> Result<Token> {
        let mut lexeme = String::new();
        while let Some(c) = self.peek_char() {
            if c.is_ascii_alphanumeric() || c == '_' {
                lexeme.push(self.advance().unwrap());
            } else {
                break;
            }
        }

        let kind = Self::keyword_or_identifier(&lexeme);
        Ok(Token::new(kind, lexeme, start_line, start_col))
    }

    /// Read a number literal (integer or float)
    fn read_number(&mut self, start_line: u32, start_col: u32) -> Result<Token> {
        let mut lexeme = String::new();
        let mut is_float = false;
        let mut is_hex = false;

        // Check for hexadecimal prefix
        if self.peek_char() == Some('0') {
            lexeme.push(self.advance().unwrap());
            if let Some(c) = self.peek_char() {
                if c == 'x' || c == 'X' {
                    is_hex = true;
                    lexeme.push(self.advance().unwrap());
                    while let Some(c) = self.peek_char() {
                        if c.is_ascii_hexdigit() {
                            lexeme.push(self.advance().unwrap());
                        } else {
                            break;
                        }
                    }
                }
            }
        }

        if !is_hex {
            while let Some(c) = self.peek_char() {
                if c.is_ascii_digit() {
                    lexeme.push(self.advance().unwrap());
                } else if c == '.' && !is_float {
                    is_float = true;
                    lexeme.push(self.advance().unwrap());
                } else if c == 'e' || c == 'E' {
                    is_float = true;
                    lexeme.push(self.advance().unwrap());
                    if let Some(sign) = self.peek_char() {
                        if sign == '+' || sign == '-' {
                            lexeme.push(self.advance().unwrap());
                        }
                    }
                } else {
                    break;
                }
            }
        }

        // Check for suffixes (u, U, l, L, f, F)
        while let Some(c) = self.peek_char() {
            if "ulULfF".contains(c) {
                lexeme.push(self.advance().unwrap());
            } else {
                break;
            }
        }

        let kind =
            if is_float || lexeme.contains('.') || lexeme.contains('e') || lexeme.contains('E') {
                match lexeme.parse::<f64>() {
                    Ok(val) => TokenKind::FloatLit(val),
                    Err(_) => {
                        return Err(Error::lexical(
                            "Invalid float literal",
                            start_line,
                            start_col,
                        ));
                    }
                }
            } else {
                if is_hex {
                    match i64::from_str_radix(&lexeme[2..], 16) {
                        Ok(val) => TokenKind::Integer(val),
                        Err(_) => {
                            return Err(Error::lexical(
                                "Invalid hexadecimal literal",
                                start_line,
                                start_col,
                            ));
                        }
                    }
                } else {
                    match lexeme.parse::<i64>() {
                        Ok(val) => TokenKind::Integer(val),
                        Err(_) => {
                            return Err(Error::lexical(
                                "Invalid integer literal",
                                start_line,
                                start_col,
                            ));
                        }
                    }
                }
            };

        Ok(Token::new(kind, lexeme, start_line, start_col))
    }

    /// Read a string literal
    fn read_string(&mut self, start_line: u32, start_col: u32) -> Result<Token> {
        let mut lexeme = String::new();
        lexeme.push(self.advance().unwrap()); // opening "

        while let Some(c) = self.peek_char() {
            if c == '"' {
                lexeme.push(self.advance().unwrap());
                return Ok(Token::new(
                    TokenKind::StringLit,
                    lexeme,
                    start_line,
                    start_col,
                ));
            } else if c == '\\' {
                lexeme.push(self.advance().unwrap());
                if let Some(escaped) = self.advance() {
                    lexeme.push(escaped);
                }
            } else if c == '\n' {
                return Err(Error::lexical(
                    "Unterminated string literal",
                    start_line,
                    start_col,
                ));
            } else {
                lexeme.push(self.advance().unwrap());
            }
        }

        Err(Error::lexical(
            "Unterminated string literal",
            start_line,
            start_col,
        ))
    }

    /// Read a character literal
    fn read_char(&mut self, start_line: u32, start_col: u32) -> Result<Token> {
        let mut lexeme = String::new();
        lexeme.push(self.advance().unwrap()); // opening '

        let ch = match self.peek_char() {
            Some('\\') => {
                lexeme.push(self.advance().unwrap());
                match self.advance() {
                    Some('n') => {
                        lexeme.push('n');
                        '\n'
                    }
                    Some('t') => {
                        lexeme.push('t');
                        '\t'
                    }
                    Some('r') => {
                        lexeme.push('r');
                        '\r'
                    }
                    Some('\\') => {
                        lexeme.push('\\');
                        '\\'
                    }
                    Some('\'') => {
                        lexeme.push('\'');
                        '\''
                    }
                    Some('0') => {
                        lexeme.push('0');
                        '\0'
                    }
                    Some(c) => {
                        lexeme.push(c);
                        c
                    }
                    None => {
                        return Err(Error::lexical(
                            "Unterminated char literal",
                            start_line,
                            start_col,
                        ));
                    }
                }
            }
            Some(c) if c != '\'' => {
                lexeme.push(self.advance().unwrap());
                c
            }
            _ => {
                return Err(Error::lexical(
                    "Empty character literal",
                    start_line,
                    start_col,
                ));
            }
        };

        if self.peek_char() != Some('\'') {
            return Err(Error::lexical(
                "Unterminated char literal",
                start_line,
                start_col,
            ));
        }
        lexeme.push(self.advance().unwrap()); // closing '

        Ok(Token::new(
            TokenKind::CharLit(ch),
            lexeme,
            start_line,
            start_col,
        ))
    }

    /// Read a line comment
    fn read_line_comment(&mut self, start_line: u32, start_col: u32) -> Result<Token> {
        let mut lexeme = String::new();
        lexeme.push(self.advance().unwrap()); // second /

        while let Some(c) = self.peek_char() {
            if c == '\n' {
                break;
            }
            lexeme.push(self.advance().unwrap());
        }

        Ok(Token::new(
            TokenKind::Comment,
            lexeme,
            start_line,
            start_col,
        ))
    }

    /// Read a block comment
    fn read_block_comment(&mut self, start_line: u32, start_col: u32) -> Result<Token> {
        let mut lexeme = String::new();
        lexeme.push(self.advance().unwrap()); // *

        loop {
            match self.peek_char() {
                Some('*') => {
                    lexeme.push(self.advance().unwrap());
                    if self.peek_char() == Some('/') {
                        lexeme.push(self.advance().unwrap());
                        return Ok(Token::new(
                            TokenKind::Comment,
                            lexeme,
                            start_line,
                            start_col,
                        ));
                    }
                }
                Some(c) => {
                    lexeme.push(self.advance().unwrap());
                }
                None => {
                    return Err(Error::lexical(
                        "Unterminated block comment",
                        start_line,
                        start_col,
                    ));
                }
            }
        }
    }

    /// Check if a string is a keyword or identifier
    fn keyword_or_identifier(s: &str) -> TokenKind {
        match s {
            "auto" => TokenKind::Auto,
            "break" => TokenKind::Break,
            "case" => TokenKind::Case,
            "char" => TokenKind::CharKw,
            "const" => TokenKind::Const,
            "continue" => TokenKind::Continue,
            "default" => TokenKind::Default,
            "do" => TokenKind::Do,
            "double" => TokenKind::Double,
            "else" => TokenKind::Else,
            "enum" => TokenKind::Enum,
            "extern" => TokenKind::Extern,
            "float" => TokenKind::FloatKw,
            "for" => TokenKind::For,
            "goto" => TokenKind::Goto,
            "if" => TokenKind::If,
            "inline" => TokenKind::Inline,
            "int" => TokenKind::Int,
            "long" => TokenKind::Long,
            "register" => TokenKind::Register,
            "restrict" => TokenKind::Restrict,
            "return" => TokenKind::Return,
            "short" => TokenKind::Short,
            "signed" => TokenKind::Signed,
            "sizeof" => TokenKind::Sizeof,
            "static" => TokenKind::Static,
            "struct" => TokenKind::Struct,
            "switch" => TokenKind::Switch,
            "typedef" => TokenKind::Typedef,
            "union" => TokenKind::Union,
            "unsigned" => TokenKind::Unsigned,
            "void" => TokenKind::Void,
            "volatile" => TokenKind::Volatile,
            "while" => TokenKind::While,
            _ => TokenKind::Identifier,
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Result<Token>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_token() {
            Ok(Token {
                kind: TokenKind::Eof,
                ..
            }) => None,
            result => Some(result),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_tokens() {
        let lexer = Lexer::new("+ - * / = == != < >");
        let tokens: Vec<_> = lexer.filter_map(|t| t.ok()).collect();
        assert_eq!(tokens.len(), 9);
    }

    #[test]
    fn test_keywords() {
        let lexer = Lexer::new("int main(void) { return 0; }");
        let tokens: Vec<_> = lexer.filter_map(|t| t.ok()).collect();
        assert_eq!(tokens[0].kind, TokenKind::Int);
        assert_eq!(tokens[1].kind, TokenKind::Identifier);
        assert_eq!(tokens[2].kind, TokenKind::LParen);
        assert_eq!(tokens[3].kind, TokenKind::Void);
    }

    #[test]
    fn test_numbers() {
        let lexer = Lexer::new("42 3.14 0xFF");
        let tokens: Vec<_> = lexer.filter_map(|t| t.ok()).collect();
        assert_eq!(tokens[0].kind, TokenKind::Integer(42));
        assert!(matches!(tokens[1].kind, TokenKind::FloatLit(_)));
        assert_eq!(tokens[2].kind, TokenKind::Integer(255));
    }
}
