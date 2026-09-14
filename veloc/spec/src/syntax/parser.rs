//! Recursive descent for declarations and set expressions; precedence climbing
//! for pure expressions. A single lookahead token keeps lexing independent of grammar.
use std::collections::BTreeMap;

use super::lexer::{Kind as TokenKind, Lexer, Token};
use super::{
    Decl, DeclKind, File, FunctionBody, Import, Kind, Node, Parameter, ResultType, Results,
    Signature,
};
use crate::Error;

pub(crate) fn parse_file(source: &str) -> Result<File, Error> {
    Parser::new(source)?.file()
}

pub fn parse(source: &str) -> Result<Vec<Decl>, Error> {
    let file = parse_file(source)?;
    if let Some(import) = file.imports.first() {
        return Err(Error::at(
            source,
            import.offset,
            "imports require Source::load and must precede declarations",
        ));
    }
    Ok(file.declarations)
}

/// Result types stop before the operation body's opening brace. Constraints
/// have their own operators and integer range, but share literal/call parsing.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Context {
    Value,
    Type,
    Expr,
}

struct Parser<'a> {
    source: &'a str,
    lexer: Lexer<'a>,
    token: Token<'a>,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Result<Self, Error> {
        let mut lexer = Lexer::new(source);
        let token = lexer.next()?;
        Ok(Self {
            source,
            lexer,
            token,
        })
    }

    fn file(&mut self) -> Result<File, Error> {
        let mut file = File {
            imports: Vec::new(),
            declarations: Vec::new(),
        };
        while self.token.kind != TokenKind::Eof {
            let Token { offset, kind } = self.bump()?;
            let declaration = match kind {
                TokenKind::Import => {
                    if !file.declarations.is_empty() {
                        return Err(self.error(offset, "imports must precede declarations"));
                    }
                    let TokenKind::Text(path) = self.bump()?.kind else {
                        return Err(self.error(offset, "import requires a quoted relative path"));
                    };
                    self.expect(TokenKind::Semi)?;
                    file.imports.push(Import { offset, path });
                    continue;
                }
                TokenKind::Fn => self.function(offset, None, false)?,
                TokenKind::Const => {
                    if self.bump()?.kind != TokenKind::Fn {
                        return Err(self.error(offset, "const must qualify a function"));
                    }
                    self.function(offset, None, true)?
                }
                _ => self.declaration(offset, kind)?,
            };
            file.declarations.push(declaration);
        }
        Ok(file)
    }

    // Type methods share the ordinary function parser and checker.
    fn methods(
        &mut self,
        owner: &str,
        fields: &mut BTreeMap<String, Node>,
    ) -> Result<Vec<Decl>, Error> {
        self.expect(TokenKind::LBrace)?;
        let mut records = Vec::new();
        while !self.at(TokenKind::RBrace) {
            let Token { offset, kind } = self.bump()?;
            match kind {
                TokenKind::Fn => records.push(self.function(offset, Some(owner), false)?),
                TokenKind::Const => {
                    if self.token.kind == TokenKind::Fn {
                        self.bump()?;
                        records.push(self.function(offset, Some(owner), true)?);
                    } else {
                        records.push(self.associated_constant(offset)?);
                    }
                }
                _ => {
                    let name = kind
                        .name()
                        .ok_or_else(|| self.error(offset, "expected a name"))?;
                    self.expect(TokenKind::Colon)?;
                    let value = self.expression(0, Context::Value)?;
                    self.expect(TokenKind::Comma)?;
                    if fields.insert(name.to_owned(), value).is_some() {
                        return Err(self.error(offset, format!("duplicate type field `{name}`")));
                    }
                }
            }
        }
        self.expect(TokenKind::RBrace)?;
        Ok(records)
    }

    fn associated_constant(&mut self, offset: usize) -> Result<Decl, Error> {
        let name = self.name()?;
        self.expect(TokenKind::Colon)?;
        let ty = self.expression(0, Context::Type)?;
        self.expect(TokenKind::Semi)?;
        Ok(Decl {
            offset,
            name,
            kind: DeclKind::Constant(ty),
            fields: BTreeMap::new(),
        })
    }

    fn function(
        &mut self,
        offset: usize,
        owner: Option<&str>,
        is_const: bool,
    ) -> Result<Decl, Error> {
        let name = self.name()?;
        let signature = self.signature(owner, is_const)?;
        let body = if owner.is_some() && self.eat(TokenKind::Semi)? {
            FunctionBody::Rust { offset, path: None }
        } else {
            self.function_body()?
        };
        Ok(Decl {
            offset,
            name,
            kind: DeclKind::Function { signature, body },
            fields: BTreeMap::new(),
        })
    }

    fn function_body(&mut self) -> Result<FunctionBody, Error> {
        if self.eat(TokenKind::Eq)? {
            let offset = self.token.offset;
            if self.name()? != "rust" {
                return Err(self.error(offset, "expected rust binding"));
            }
            self.expect(TokenKind::LParen)?;
            let TokenKind::Text(path) = self.bump()?.kind else {
                return Err(self.error(offset, "rust binding requires a qualified path"));
            };
            self.expect(TokenKind::RParen)?;
            self.expect(TokenKind::Semi)?;
            Ok(FunctionBody::Rust {
                offset,
                path: Some(path),
            })
        } else {
            let offset = self.token.offset;
            let mut fields = self.fields(0, Context::Expr, false)?;
            if let Some((field, node)) = fields.iter().find(|(name, _)| name.as_str() != "value") {
                return Err(self.error(node.offset, format!("unknown function field `{field}`")));
            }
            let value = fields
                .remove("value")
                .ok_or_else(|| self.error(offset, "missing function field `value`"))?;
            Ok(FunctionBody::Value(value))
        }
    }

    fn declaration(&mut self, offset: usize, kind: TokenKind<'a>) -> Result<Decl, Error> {
        let declaration_name = kind
            .name()
            .ok_or_else(|| self.error(offset, "expected a name"))?;
        let name = self.name()?;
        let mut fields = BTreeMap::new();
        let kind = match kind {
            TokenKind::Type => {
                self.expect(TokenKind::Eq)?;
                let binding = self.expression(0, Context::Type)?;
                let members = if self.at(TokenKind::LBrace) {
                    if !matches!(&binding.kind, Kind::Call(name, _) if name == "rust") {
                        return Err(self.error(offset, "methods require a Rust-bound type"));
                    }
                    self.methods(&name, &mut fields)?
                } else {
                    self.expect(TokenKind::Semi)?;
                    Vec::new()
                };
                DeclKind::Type { binding, members }
            }
            TokenKind::TypeSet => {
                self.expect(TokenKind::Eq)?;
                let set = self.expression(0, Context::Type)?;
                self.expect(TokenKind::Semi)?;
                DeclKind::TypeSet(set)
            }
            TokenKind::Op => {
                let signature = self.signature(None, false)?;
                fields = self.fields(0, Context::Value, false)?;
                DeclKind::Op(signature)
            }
            _ => {
                fields = self.fields(0, Context::Value, false)?;
                DeclKind::Fields(declaration_name.to_owned())
            }
        };
        Ok(Decl {
            offset,
            name,
            fields,
            kind,
        })
    }

    fn signature(&mut self, owner: Option<&str>, is_const: bool) -> Result<Signature, Error> {
        let generics = if self.eat(TokenKind::Lt)? {
            self.sequence(TokenKind::Gt, Self::parameter)?
        } else {
            Vec::new()
        };
        self.expect(TokenKind::LParen)?;
        let params = self.sequence(TokenKind::RParen, |p| {
            let borrowed = owner.is_some() && p.eat(TokenKind::Amp)?;
            if let Some(owner) = owner
                && p.token.kind == TokenKind::SelfValue
            {
                let offset = p.bump()?.offset;
                let mut ty = Node {
                    offset,
                    kind: Kind::Name(owner.into()),
                };
                if borrowed {
                    ty = Node {
                        offset,
                        kind: Kind::Ref(Box::new(ty)),
                    };
                }
                Ok(Parameter {
                    offset,
                    name: "self".into(),
                    moves: false,
                    ty,
                })
            } else {
                if borrowed {
                    return Err(p.error(p.token.offset, "expected self after &"));
                }
                p.parameter()
            }
        })?;
        self.expect(TokenKind::Arrow)?;
        let results = if self.eat(TokenKind::LParen)? {
            Results::Fixed(self.sequence(TokenKind::RParen, Self::result)?)
        } else {
            let ty = self.expression(0, Context::Type)?;
            if matches!(&ty.kind, Kind::Name(name) if name == "signature") {
                Results::Signature
            } else {
                Results::Fixed(vec![ResultType {
                    offset: ty.offset,
                    name: None,
                    ty,
                }])
            }
        };
        Ok(Signature {
            is_const,
            generics,
            params,
            results,
        })
    }

    fn parameter(&mut self) -> Result<Parameter, Error> {
        let offset = self.token.offset;
        let mut name = self.name()?;
        let moves = name == "move" && !self.at(TokenKind::Colon);
        if moves {
            name = self.name()?;
        }
        self.expect(TokenKind::Colon)?;
        let ty = self.expression(0, Context::Type)?;
        Ok(Parameter {
            offset,
            name,
            moves,
            ty,
        })
    }

    fn result(&mut self) -> Result<ResultType, Error> {
        let first = self.expression(0, Context::Type)?;
        let offset = first.offset;
        let (name, ty) = if self.eat(TokenKind::Colon)? {
            let Kind::Name(name) = first.kind else {
                return Err(self.error(offset, "expected a result name before ':'"));
            };
            (Some(name), self.expression(0, Context::Type)?)
        } else {
            (None, first)
        };
        Ok(ResultType { offset, name, ty })
    }

    fn fields(
        &mut self,
        depth: u8,
        context: Context,
        shorthand: bool,
    ) -> Result<BTreeMap<String, Node>, Error> {
        self.check_depth(depth, context)?;
        self.expect(TokenKind::LBrace)?;
        let mut fields = BTreeMap::new();
        while !self.at(TokenKind::RBrace) {
            let offset = self.token.offset;
            let name = self.name()?;
            if name == "query" && !self.at(TokenKind::Colon) {
                let query = self.name()?;
                let param = if self.eat(TokenKind::LParen)? {
                    let offset = self.token.offset;
                    let name = self.name()?;
                    self.expect(TokenKind::Colon)?;
                    let ty = self.expression(depth + 1, Context::Type)?;
                    self.expect(TokenKind::RParen)?;
                    Some(Parameter {
                        offset,
                        name,
                        moves: false,
                        ty,
                    })
                } else {
                    None
                };
                self.expect(TokenKind::Arrow)?;
                let result = self.name()?;
                let body = Node {
                    offset,
                    kind: Kind::Object(result, self.fields(depth + 1, Context::Expr, true)?),
                };
                let body = if let Some(param) = param {
                    Node {
                        offset,
                        kind: Kind::Scoped(Box::new(param), Box::new(body)),
                    }
                } else {
                    body
                };
                let entry = fields.entry("queries".into()).or_insert_with(|| Node {
                    offset,
                    kind: Kind::List(Vec::new()),
                });
                let Kind::List(queries) = &mut entry.kind else {
                    return Err(self.error(offset, "queries are declared with query blocks"));
                };
                queries.push(Node {
                    offset,
                    kind: Kind::Query(query, Box::new(body)),
                });
                self.eat(TokenKind::Comma)?;
                continue;
            }
            if name == "queries" {
                return Err(self.error(offset, "use named query blocks instead of a queries field"));
            }
            if name == "constraints" {
                return Err(self.error(offset, "use a verify block instead of constraints"));
            }
            let context_param = if name == "verify" && self.eat(TokenKind::LParen)? {
                let offset = self.token.offset;
                let name = self.name()?;
                self.expect(TokenKind::Colon)?;
                let ty = self.expression(depth + 1, Context::Expr)?;
                self.expect(TokenKind::RParen)?;
                Some(Parameter {
                    offset,
                    name,
                    moves: false,
                    ty,
                })
            } else {
                None
            };
            let block = name == "verify" && self.at(TokenKind::LBrace);
            let mut node = if block {
                self.expect(TokenKind::LBrace)?;
                let mut statements = Vec::new();
                while !self.at(TokenKind::RBrace) {
                    let offset = self.token.offset;
                    let statement = if self.token.kind == TokenKind::Let {
                        self.bump()?;
                        let name = self.name()?;
                        self.expect(TokenKind::Eq)?;
                        Node {
                            offset,
                            kind: Kind::Let(
                                name,
                                Box::new(self.expression(depth + 1, Context::Expr)?),
                            ),
                        }
                    } else {
                        self.expression(depth + 1, Context::Expr)?
                    };
                    statements.push(statement);
                    self.expect(TokenKind::Semi)?;
                }
                self.expect(TokenKind::RBrace)?;
                Node {
                    offset,
                    kind: Kind::List(statements),
                }
            } else if shorthand && (self.at(TokenKind::Comma) || self.at(TokenKind::RBrace)) {
                Node {
                    offset,
                    kind: Kind::Name(name.clone()),
                }
            } else {
                self.expect(TokenKind::Colon)?;
                self.expression(
                    depth,
                    if name == "meta" {
                        Context::Expr
                    } else {
                        context
                    },
                )?
            };
            if let Some(param) = context_param {
                node = Node {
                    offset,
                    kind: Kind::Scoped(Box::new(param), Box::new(node)),
                };
            }
            if fields.insert(name.clone(), node).is_some() {
                return Err(self.error(offset, format!("duplicate field `{name}`")));
            }
            if block {
                self.eat(TokenKind::Comma)?;
            } else if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }
        self.expect(TokenKind::RBrace)?;
        Ok(fields)
    }

    fn expression(&mut self, depth: u8, context: Context) -> Result<Node, Error> {
        if context == Context::Expr {
            self.binary(depth, 0)
        } else {
            self.union(depth, context)
        }
    }

    // Flat sets use n-ary nodes so a long union does not create a deep AST.
    // Intersection binds more tightly than union.
    fn union(&mut self, depth: u8, context: Context) -> Result<Node, Error> {
        let first = self.intersection(depth, context)?;
        if !self.eat(TokenKind::Pipe)? {
            return Ok(first);
        }
        let offset = first.offset;
        let mut nodes = vec![first];
        loop {
            nodes.push(self.intersection(depth, context)?);
            if !self.eat(TokenKind::Pipe)? {
                break;
            }
        }
        Ok(Node {
            offset,
            kind: Kind::Union(nodes),
        })
    }

    fn intersection(&mut self, depth: u8, context: Context) -> Result<Node, Error> {
        let first = self.atom(depth, context)?;
        let first = self.postfix(first, depth, context)?;
        if !self.eat(TokenKind::Amp)? {
            return Ok(first);
        }
        let offset = first.offset;
        let mut nodes = vec![first];
        loop {
            let node = self.atom(depth, context)?;
            nodes.push(self.postfix(node, depth, context)?);
            if !self.eat(TokenKind::Amp)? {
                break;
            }
        }
        Ok(Node {
            offset,
            kind: Kind::Intersection(nodes),
        })
    }

    fn binary(&mut self, depth: u8, precedence: u8) -> Result<Node, Error> {
        self.check_depth(depth, Context::Expr)?;
        let offset = self.token.offset;
        let kind = match self.token.kind {
            TokenKind::Bang | TokenKind::Minus => {
                let op = self.token.kind.spelling();
                self.bump()?;
                Kind::Unary(op, Box::new(self.binary(depth + 1, 9)?))
            }
            TokenKind::Pipe => {
                self.bump()?;
                let mut names = vec![self.name()?];
                while self.eat(TokenKind::Comma)? {
                    names.push(self.name()?);
                }
                self.expect(TokenKind::Pipe)?;
                Kind::Lambda(names, Box::new(self.binary(depth + 1, 0)?))
            }
            _ => self.atom(depth, Context::Expr)?.kind,
        };
        let mut lhs = self.postfix(Node { offset, kind }, depth, Context::Expr)?;
        let mut chain = 0;
        while let Some((op, level)) = self.binary_operator() {
            if level < precedence {
                break;
            }
            chain += 1;
            self.check_depth(depth + chain, Context::Expr)?;
            self.bump()?;
            let rhs = self.binary(depth + 1, level + 1)?;
            lhs = Node {
                offset,
                kind: Kind::Binary(op, Box::new(lhs), Box::new(rhs)),
            };
        }
        Ok(lhs)
    }

    fn postfix(&mut self, mut node: Node, depth: u8, context: Context) -> Result<Node, Error> {
        let offset = node.offset;

        let mut postfix = 0;
        while self.at(TokenKind::Question)
            || self.at(TokenKind::Dot)
            || self.at(TokenKind::ColonColon)
        {
            postfix += 1;
            self.check_depth(depth + postfix, Context::Expr)?;
            if self.eat(TokenKind::Question)? {
                node = Node {
                    offset,
                    kind: Kind::Try(Box::new(node)),
                };
            } else if self.eat(TokenKind::ColonColon)? {
                let Kind::Name(owner) = node.kind else {
                    return Err(self.error(offset, ":: requires a type or namespace path"));
                };
                let name = format!("{owner}::{}", self.name()?);
                let kind = if self.eat(TokenKind::LParen)? {
                    Kind::Call(
                        name,
                        self.sequence(TokenKind::RParen, |p| {
                            p.expression(depth + postfix, context)
                        })?,
                    )
                } else {
                    Kind::Name(name)
                };
                node = Node { offset, kind };
            } else {
                self.expect(TokenKind::Dot)?;
                let member = self.name()?;
                let kind = if self.eat(TokenKind::LParen)? {
                    let args = self.sequence(TokenKind::RParen, |p| {
                        p.expression(depth + postfix, context)
                    })?;
                    Kind::Method(Box::new(node), member, args)
                } else {
                    Kind::Member(Box::new(node), member)
                };
                node = Node { offset, kind };
            }
        }
        Ok(node)
    }

    fn binary_operator(&self) -> Option<(&'static str, u8)> {
        let level = match self.token.kind {
            TokenKind::PipePipe => 1,
            TokenKind::AmpAmp => 2,
            TokenKind::Pipe => 3,
            TokenKind::Amp => 4,
            TokenKind::EqEq | TokenKind::NotEq => 5,
            TokenKind::Le | TokenKind::Ge | TokenKind::Lt | TokenKind::Gt => 6,
            TokenKind::Plus | TokenKind::Minus => 7,
            TokenKind::Star => 8,
            _ => return None,
        };
        Some((self.token.kind.spelling(), level))
    }

    fn atom(&mut self, depth: u8, context: Context) -> Result<Node, Error> {
        self.check_depth(depth, context)?;
        let Token { offset, kind } = self.bump()?;
        let kind = match kind {
            TokenKind::Amp => Kind::Ref(Box::new(self.atom(depth + 1, context)?)),
            TokenKind::LParen => {
                let node = self.expression(depth + 1, context)?;
                self.expect(TokenKind::RParen)?;
                return Ok(node);
            }
            TokenKind::LBracket => Kind::List(
                self.sequence(TokenKind::RBracket, |p| p.expression(depth + 1, context))?,
            ),
            TokenKind::Text(text) => Kind::Text(text),
            TokenKind::Number(text) if context == Context::Expr => Kind::Integer(
                text.parse()
                    .map_err(|_| self.error(offset, "expression integer is out of range"))?,
            ),
            TokenKind::Number(text) => Kind::Number(
                text.parse()
                    .map_err(|_| self.error(offset, "integer is out of range"))?,
            ),
            word if word.name().is_some() => {
                let name = word.name().unwrap().to_owned();
                if self.eat(TokenKind::LParen)? {
                    let arguments = if context == Context::Expr {
                        context
                    } else {
                        Context::Value
                    };
                    Kind::Call(
                        name,
                        self.sequence(TokenKind::RParen, |p| p.expression(depth + 1, arguments))?,
                    )
                } else if context != Context::Type && self.at(TokenKind::LBrace) {
                    Kind::Object(name, self.fields(depth + 1, context, true)?)
                } else {
                    Kind::Name(name)
                }
            }
            _ => return Err(self.error(offset, "expected a name or value")),
        };
        Ok(Node { offset, kind })
    }

    fn sequence<T>(
        &mut self,
        end: TokenKind<'a>,
        mut item: impl FnMut(&mut Self) -> Result<T, Error>,
    ) -> Result<Vec<T>, Error> {
        let mut items = Vec::new();
        while self.token.kind != end {
            items.push(item(self)?);
            if self.token.kind != end {
                self.expect(TokenKind::Comma)?;
            }
        }
        self.expect(end)?;
        Ok(items)
    }

    fn check_depth(&self, depth: u8, context: Context) -> Result<(), Error> {
        if depth < 64 {
            return Ok(());
        }
        let kind = if context == Context::Expr {
            "expression"
        } else {
            "definition"
        };
        Err(self.error(
            self.token.offset,
            format!("{kind} nesting exceeds 64 levels"),
        ))
    }

    fn name(&mut self) -> Result<String, Error> {
        let Token { offset, kind } = self.bump()?;
        kind.name()
            .map(str::to_owned)
            .ok_or_else(|| self.error(offset, "expected a name"))
    }

    fn at(&self, kind: TokenKind<'a>) -> bool {
        self.token.kind == kind
    }

    fn eat(&mut self, kind: TokenKind<'a>) -> Result<bool, Error> {
        if !self.at(kind) {
            return Ok(false);
        }
        self.bump()?;
        Ok(true)
    }

    fn expect(&mut self, kind: TokenKind<'a>) -> Result<(), Error> {
        let spelling = kind.spelling();
        if self.eat(kind)? {
            return Ok(());
        }
        Err(self.error(self.token.offset, format!("expected `{spelling}`")))
    }

    fn bump(&mut self) -> Result<Token<'a>, Error> {
        let next = self.lexer.next()?;
        Ok(std::mem::replace(&mut self.token, next))
    }

    fn error(&self, offset: usize, message: impl Into<String>) -> Error {
        Error::at(self.source, offset, message)
    }
}
