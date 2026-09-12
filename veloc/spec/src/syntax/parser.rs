//! Recursive descent for declarations and set expressions; precedence climbing
//! for pure expressions. A single lookahead token keeps lexing independent of grammar.
use std::collections::BTreeMap;

use super::lexer::{Kind as TokenKind, Lexer, Token};
use super::{
    File, FunctionBody, Import, Kind, Node, Parameter, Record, ResultType, Results, Signature,
};
use crate::Error;

pub(crate) fn parse_file(source: &str) -> Result<File, Error> {
    Parser::new(source)?.file()
}

pub(crate) fn parse(source: &str) -> Result<Vec<Record>, Error> {
    let file = parse_file(source)?;
    if let Some(import) = file.imports.first() {
        return Err(Error::at(
            source,
            import.offset,
            "imports require Source::load and must precede declarations",
        ));
    }
    Ok(file.records)
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
            records: Vec::new(),
        };
        while self.token.kind != TokenKind::Eof {
            let offset = self.token.offset;
            let kind = self.name()?;
            if kind == "import" {
                if !file.records.is_empty() {
                    return Err(self.error(offset, "imports must precede declarations"));
                }
                let TokenKind::Text(path) = self.bump()?.kind else {
                    return Err(self.error(offset, "import requires a quoted relative path"));
                };
                self.expect(";")?;
                file.imports.push(Import { offset, path });
            } else if kind == "extern" {
                file.records.extend(self.external(offset)?);
            } else {
                let mut record = self.declaration(offset, kind)?;
                let owner = (record.kind == "type" && self.at("{")).then(|| record.name.clone());
                let methods = if let Some(owner) = owner {
                    self.methods(&owner, &mut record.fields)?
                } else {
                    Vec::new()
                };
                file.records.push(record);
                file.records.extend(methods);
            }
        }
        Ok(file)
    }

    // Methods share the ordinary function checker and expansion mechanism.
    fn methods(
        &mut self,
        owner: &str,
        fields: &mut BTreeMap<String, Node>,
    ) -> Result<Vec<Record>, Error> {
        self.expect("{")?;
        let mut records = Vec::new();
        while !self.at("}") {
            let offset = self.token.offset;
            let kind = self.name()?;
            if kind != "fn" {
                self.expect(":")?;
                let value = self.expression(0, Context::Value)?;
                self.expect(",")?;
                if fields.insert(kind.clone(), value).is_some() {
                    return Err(self.error(offset, format!("duplicate type field `{kind}`")));
                }
                continue;
            }
            let name = format!("{owner}.{}", self.name()?);
            let signature = self.method_signature(Some(owner))?;
            let body = if self.eat(";")? {
                FunctionBody::Rust { offset, path: None }
            } else {
                self.function_body()?
            };
            records.push(Record {
                offset,
                kind: "fn".into(),
                name,
                fields: BTreeMap::new(),
                body: Some(body),
                signature: Some(signature),
            });
        }
        self.expect("}")?;
        Ok(records)
    }

    fn function_body(&mut self) -> Result<FunctionBody, Error> {
        if self.eat("=")? {
            let offset = self.token.offset;
            if self.name()? != "rust" {
                return Err(self.error(offset, "expected rust binding"));
            }
            self.expect("(")?;
            let TokenKind::Text(path) = self.bump()?.kind else {
                return Err(self.error(offset, "rust binding requires a qualified path"));
            };
            self.expect(")")?;
            self.expect(";")?;
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

    fn external(&mut self, offset: usize) -> Result<Vec<Record>, Error> {
        if self.name()? != "interface" {
            return Err(self.error(offset, "expected extern interface"));
        }
        let name = self.name()?;
        self.expect("{")?;
        let mut records = vec![Record {
            offset,
            kind: "extern-interface".into(),
            name: name.clone(),
            fields: BTreeMap::new(),
            signature: None,
            body: None,
        }];
        while !self.at("}") {
            let offset = self.token.offset;
            if self.name()? != "fn" {
                return Err(self.error(offset, "expected extern method"));
            }
            let method = self.name()?;
            let signature = self.signature()?;
            self.expect(";")?;
            records.push(Record {
                offset,
                kind: "extern-fn".into(),
                name: format!("{name}.{method}"),
                fields: BTreeMap::new(),
                signature: Some(signature),
                body: None,
            });
        }
        self.expect("}")?;
        Ok(records)
    }

    fn declaration(&mut self, offset: usize, kind: String) -> Result<Record, Error> {
        let name = self.name()?;
        let signature = if matches!(kind.as_str(), "op" | "fn") {
            Some(self.signature()?)
        } else {
            None
        };
        let body = if kind == "fn" {
            Some(self.function_body()?)
        } else {
            None
        };
        let fields = match kind.as_str() {
            "type" | "typeset" | "predicate" => {
                self.expect("=")?;
                let node = self.expression(0, Context::Type)?;
                if kind != "type" || !self.at("{") {
                    self.expect(";")?;
                } else if !matches!(&node.kind, Kind::Call(name, _) if name == "rust") {
                    return Err(self.error(offset, "methods require a Rust-bound type"));
                }
                BTreeMap::from([(if kind == "type" { "expr" } else { "set" }.into(), node)])
            }
            "fn" => BTreeMap::new(),
            _ => self.fields(0, Context::Value, false)?,
        };
        Ok(Record {
            offset,
            kind,
            name,
            fields,
            signature,
            body,
        })
    }

    fn signature(&mut self) -> Result<Signature, Error> {
        self.method_signature(None)
    }

    fn method_signature(&mut self, owner: Option<&str>) -> Result<Signature, Error> {
        let generics = if self.eat("<")? {
            self.sequence(">", Self::parameter)?
        } else {
            Vec::new()
        };
        self.expect("(")?;
        let params = self.sequence(")", |p| {
            if let Some(owner) = owner
                && matches!(p.token.kind, TokenKind::Name("self"))
            {
                let offset = p.bump()?.offset;
                Ok(Parameter {
                    offset,
                    name: "self".into(),
                    moves: false,
                    ty: Node {
                        offset,
                        kind: Kind::Name(owner.into()),
                    },
                })
            } else {
                p.parameter()
            }
        })?;
        self.expect("->")?;
        let results = if self.eat("(")? {
            Results::Fixed(self.sequence(")", Self::result)?)
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
            generics,
            params,
            results,
        })
    }

    fn parameter(&mut self) -> Result<Parameter, Error> {
        let offset = self.token.offset;
        let mut name = self.name()?;
        let moves = name == "move" && !self.at(":");
        if moves {
            name = self.name()?;
        }
        self.expect(":")?;
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
        let (name, ty) = if self.eat(":")? {
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
        self.expect("{")?;
        let mut fields = BTreeMap::new();
        while !self.at("}") {
            let offset = self.token.offset;
            let name = self.name()?;
            if name == "constraints" {
                return Err(self.error(offset, "use a verify block instead of constraints"));
            }
            let block = name == "verify" && self.at("{");
            let node = if block {
                self.expect("{")?;
                let mut statements = Vec::new();
                while !self.at("}") {
                    statements.push(self.expression(depth + 1, Context::Expr)?);
                    self.expect(";")?;
                }
                self.expect("}")?;
                Node {
                    offset,
                    kind: Kind::List(statements),
                }
            } else if shorthand && (self.at(",") || self.at("}")) {
                Node {
                    offset,
                    kind: Kind::Name(name.clone()),
                }
            } else {
                self.expect(":")?;
                self.expression(depth, context)?
            };
            if fields.insert(name.clone(), node).is_some() {
                return Err(self.error(offset, format!("duplicate field `{name}`")));
            }
            if block {
                self.eat(",")?;
            } else if !self.at("}") {
                self.expect(",")?;
            }
        }
        self.expect("}")?;
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
        if !self.eat("|")? {
            return Ok(first);
        }
        let offset = first.offset;
        let mut nodes = vec![first];
        loop {
            nodes.push(self.intersection(depth, context)?);
            if !self.eat("|")? {
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
        if !self.eat("&")? {
            return Ok(first);
        }
        let offset = first.offset;
        let mut nodes = vec![first];
        loop {
            let node = self.atom(depth, context)?;
            nodes.push(self.postfix(node, depth, context)?);
            if !self.eat("&")? {
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
            TokenKind::Symbol(op @ ("!" | "-")) => {
                self.bump()?;
                Kind::Unary(op, Box::new(self.binary(depth + 1, 9)?))
            }
            TokenKind::Symbol("|") => {
                self.bump()?;
                let name = self.name()?;
                self.expect("|")?;
                Kind::Lambda(name, Box::new(self.binary(depth + 1, 0)?))
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
        while self.at("?") || self.at(".") {
            postfix += 1;
            self.check_depth(depth + postfix, Context::Expr)?;
            if self.eat("?")? {
                node = Node {
                    offset,
                    kind: Kind::Try(Box::new(node)),
                };
            } else {
                self.expect(".")?;
                let member = self.name()?;
                let kind = if self.eat("(")? {
                    let args = self.sequence(")", |p| p.expression(depth + postfix, context))?;
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
        let TokenKind::Symbol(op) = self.token.kind else {
            return None;
        };
        let level = match op {
            "||" => 1,
            "&&" => 2,
            "|" => 3,
            "&" => 4,
            "==" | "!=" => 5,
            "<=" | ">=" | "<" | ">" => 6,
            "+" | "-" => 7,
            "*" => 8,
            _ => return None,
        };
        Some((op, level))
    }

    fn atom(&mut self, depth: u8, context: Context) -> Result<Node, Error> {
        self.check_depth(depth, context)?;
        let Token { offset, kind } = self.bump()?;
        let kind = match kind {
            TokenKind::Symbol("(") => {
                let node = self.expression(depth + 1, context)?;
                self.expect(")")?;
                return Ok(node);
            }
            TokenKind::Symbol("[") => {
                Kind::List(self.sequence("]", |p| p.expression(depth + 1, context))?)
            }
            TokenKind::Text(text) => Kind::Text(text),
            TokenKind::Number(text) if context == Context::Expr => Kind::Integer(
                text.parse()
                    .map_err(|_| self.error(offset, "expression integer is out of range"))?,
            ),
            TokenKind::Number(text) => Kind::Number(
                text.parse()
                    .map_err(|_| self.error(offset, "integer is out of range"))?,
            ),
            TokenKind::Name(name) => {
                let name = name.to_owned();
                if self.eat("(")? {
                    let arguments = if context == Context::Expr {
                        context
                    } else {
                        Context::Value
                    };
                    Kind::Call(
                        name,
                        self.sequence(")", |p| p.expression(depth + 1, arguments))?,
                    )
                } else if context != Context::Type && self.at("{") {
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
        end: &str,
        mut item: impl FnMut(&mut Self) -> Result<T, Error>,
    ) -> Result<Vec<T>, Error> {
        let mut items = Vec::new();
        while !self.at(end) {
            items.push(item(self)?);
            if !self.at(end) {
                self.expect(",")?;
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
        match kind {
            TokenKind::Name(name) => Ok(name.to_owned()),
            _ => Err(self.error(offset, "expected a name")),
        }
    }

    fn at(&self, symbol: &str) -> bool {
        matches!(self.token.kind, TokenKind::Symbol(s) if s == symbol)
    }

    fn eat(&mut self, symbol: &str) -> Result<bool, Error> {
        if !self.at(symbol) {
            return Ok(false);
        }
        self.bump()?;
        Ok(true)
    }

    fn expect(&mut self, symbol: &str) -> Result<(), Error> {
        if self.eat(symbol)? {
            return Ok(());
        }
        Err(self.error(self.token.offset, format!("expected `{symbol}`")))
    }

    fn bump(&mut self) -> Result<Token<'a>, Error> {
        let next = self.lexer.next()?;
        Ok(std::mem::replace(&mut self.token, next))
    }

    fn error(&self, offset: usize, message: impl Into<String>) -> Error {
        Error::at(self.source, offset, message)
    }
}
