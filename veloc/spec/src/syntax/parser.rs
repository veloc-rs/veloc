//! Recursive descent for declarations and set expressions; precedence climbing
//! for constraints. A single lookahead token keeps lexing independent of grammar.
use std::collections::BTreeMap;

use super::lexer::{Kind as TokenKind, Lexer, Token};
use super::{File, Import, Kind, Node, Parameter, Record, ResultType, Results, Signature};
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
    Constraint,
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
            } else {
                file.records.push(self.declaration(offset, kind)?);
            }
        }
        Ok(file)
    }

    fn declaration(&mut self, offset: usize, kind: String) -> Result<Record, Error> {
        let name = self.name()?;
        let signature = if matches!(kind.as_str(), "op" | "fn") {
            Some(self.signature()?)
        } else {
            None
        };
        let fields = match kind.as_str() {
            "type" | "predicate" => {
                self.expect("=")?;
                let node = self.expression(0, Context::Type)?;
                self.expect(";")?;
                BTreeMap::from([(if kind == "type" { "expr" } else { "set" }.into(), node)])
            }
            _ => self.fields(0)?,
        };
        Ok(Record {
            offset,
            kind,
            name,
            fields,
            signature,
        })
    }

    fn signature(&mut self) -> Result<Signature, Error> {
        let generics = if self.eat("<")? {
            self.sequence(">", Self::parameter)?
        } else {
            Vec::new()
        };
        self.expect("(")?;
        let params = self.sequence(")", Self::parameter)?;
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
        let property = self.eat("@")?;
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
            property,
            moves,
            ty,
        })
    }

    fn result(&mut self) -> Result<ResultType, Error> {
        if self.at("@") {
            return Err(self.error(self.token.offset, "results cannot be properties"));
        }
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

    fn fields(&mut self, depth: u8) -> Result<BTreeMap<String, Node>, Error> {
        self.expect("{")?;
        let mut fields = BTreeMap::new();
        self.sequence("}", |parser| {
            let offset = parser.token.offset;
            let name = parser.name()?;
            parser.expect(":")?;
            let node = if name == "constraints" {
                let offset = parser.token.offset;
                parser.expect("[")?;
                Node {
                    offset,
                    kind: Kind::List(
                        parser.sequence("]", |p| p.expression(depth + 1, Context::Constraint))?,
                    ),
                }
            } else {
                parser.expression(depth, Context::Value)?
            };
            if fields.insert(name.clone(), node).is_some() {
                return Err(parser.error(offset, format!("duplicate field `{name}`")));
            }
            Ok(())
        })?;
        Ok(fields)
    }

    fn expression(&mut self, depth: u8, context: Context) -> Result<Node, Error> {
        if context == Context::Constraint {
            self.constraint(depth, 0)
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
        if !self.eat("&")? {
            return Ok(first);
        }
        let offset = first.offset;
        let mut nodes = vec![first];
        loop {
            nodes.push(self.atom(depth, context)?);
            if !self.eat("&")? {
                break;
            }
        }
        Ok(Node {
            offset,
            kind: Kind::Intersection(nodes),
        })
    }

    fn constraint(&mut self, depth: u8, precedence: u8) -> Result<Node, Error> {
        self.check_depth(depth, Context::Constraint)?;
        let offset = self.token.offset;
        let kind = match self.token.kind {
            TokenKind::Symbol(op @ ("!" | "-")) => {
                self.bump()?;
                Kind::Unary(op, Box::new(self.constraint(depth + 1, 6)?))
            }
            TokenKind::Symbol("|") => {
                self.bump()?;
                let name = self.name()?;
                self.expect("|")?;
                Kind::Lambda(name, Box::new(self.constraint(depth + 1, 0)?))
            }
            _ => self.atom(depth, Context::Constraint)?.kind,
        };
        let mut lhs = Node { offset, kind };
        let mut chain = 0;
        while let Some((op, level)) = self.binary_operator() {
            if level < precedence {
                break;
            }
            chain += 1;
            self.check_depth(depth + chain, Context::Constraint)?;
            self.bump()?;
            let rhs = self.constraint(depth + 1, level + 1)?;
            lhs = Node {
                offset,
                kind: Kind::Binary(op, Box::new(lhs), Box::new(rhs)),
            };
        }
        Ok(lhs)
    }

    fn binary_operator(&self) -> Option<(&'static str, u8)> {
        let TokenKind::Symbol(op) = self.token.kind else {
            return None;
        };
        let level = match op {
            "||" => 1,
            "&&" => 2,
            "==" | "!=" | "<=" | ">=" | "<" | ">" => 3,
            "+" | "-" => 4,
            "*" => 5,
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
            TokenKind::Symbol("[") if context != Context::Constraint => {
                Kind::List(self.sequence("]", |p| p.expression(depth + 1, Context::Value))?)
            }
            TokenKind::Text(text) => Kind::Text(text),
            TokenKind::Number(text) if context == Context::Constraint => Kind::Integer(
                text.parse()
                    .map_err(|_| self.error(offset, "constraint integer is out of range"))?,
            ),
            TokenKind::Number(text) => Kind::Number(
                text.parse()
                    .map_err(|_| self.error(offset, "integer is out of range"))?,
            ),
            TokenKind::Name(name) => {
                let name = name.to_owned();
                if self.eat("(")? {
                    let arguments = if context == Context::Constraint {
                        context
                    } else {
                        Context::Value
                    };
                    Kind::Call(
                        name,
                        self.sequence(")", |p| p.expression(depth + 1, arguments))?,
                    )
                } else if context == Context::Value && self.at("{") {
                    Kind::Object(name, self.fields(depth + 1)?)
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
        let kind = if context == Context::Constraint {
            "constraint"
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
