use std::collections::BTreeMap;

use crate::Error;

#[derive(Debug, Clone)]
pub(crate) struct Node {
    pub offset: usize,
    pub kind: Kind,
}

#[derive(Debug, Clone)]
pub(crate) enum Kind {
    Name(String),
    Text(String),
    Number(u32),
    List(Vec<Node>),
    Call(String, Vec<Node>),
    Object(String, BTreeMap<String, Node>),
    Union(Vec<Node>),
    Intersection(Vec<Node>),
}

#[derive(Debug, Clone)]
pub(crate) struct Parameter {
    pub offset: usize,
    pub name: String,
    pub property: bool,
    pub ty: Node,
}

#[derive(Debug, Clone)]
pub(crate) struct ResultType {
    pub offset: usize,
    pub name: Option<String>,
    pub ty: Node,
}

#[derive(Debug, Clone)]
pub(crate) enum Results {
    Fixed(Vec<ResultType>),
    Signature,
}

#[derive(Debug, Clone)]
pub(crate) struct Signature {
    pub generics: Vec<Parameter>,
    pub params: Vec<Parameter>,
    pub results: Results,
}

#[derive(Debug, Clone)]
pub(crate) struct Record {
    pub offset: usize,
    pub kind: String,
    pub name: String,
    pub fields: BTreeMap<String, Node>,
    pub signature: Option<Signature>,
}

pub(crate) fn parse(source: &str) -> Result<Vec<Record>, Error> {
    let mut parser = Parser { source, offset: 0 };
    let mut records = Vec::new();
    while parser.peek().is_some() {
        let offset = parser.offset;
        let kind = parser.name()?;
        let name = parser.name()?;
        let signature = if kind == "op" {
            Some(parser.signature()?)
        } else {
            None
        };
        let fields = if matches!(kind.as_str(), "predicate" | "type") {
            parser.expect(b'=')?;
            let set = parser.type_node()?;
            parser.expect(b';')?;
            BTreeMap::from([(if kind == "type" { "expr" } else { "set" }.into(), set)])
        } else {
            parser.fields(0)?
        };
        records.push(Record {
            offset,
            kind,
            name,
            fields,
            signature,
        });
    }
    Ok(records)
}

struct Parser<'a> {
    source: &'a str,
    offset: usize,
}

impl Parser<'_> {
    fn signature(&mut self) -> Result<Signature, Error> {
        let generics = if self.peek() == Some(b'<') {
            self.offset += 1;
            self.parameters(b'>')?
        } else {
            Vec::new()
        };
        self.expect(b'(')?;
        let params = self.parameters(b')')?;
        self.expect(b'-')?;
        self.expect(b'>')?;
        let results = if self.peek() == Some(b'(') {
            self.offset += 1;
            let mut results = Vec::new();
            while self.peek() != Some(b')') {
                results.push(self.result()?);
                if self.peek() != Some(b')') {
                    self.expect(b',')?;
                }
            }
            self.expect(b')')?;
            Results::Fixed(results)
        } else {
            let ty = self.type_node()?;
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

    fn result(&mut self) -> Result<ResultType, Error> {
        if self.peek() == Some(b'@') {
            return Err(self.error(self.offset, "results cannot be properties"));
        }
        let first = self.type_node()?;
        let offset = first.offset;
        let (name, ty) = if self.peek() == Some(b':') {
            let Kind::Name(name) = first.kind else {
                return Err(self.error(offset, "expected a result name before ':'"));
            };
            self.offset += 1;
            (Some(name), self.type_node()?)
        } else {
            (None, first)
        };
        Ok(ResultType { offset, name, ty })
    }

    // Unlike a general field value, a type cannot consume the following `{`:
    // in `-> T { ... }` it starts the operation body, not a named object.
    fn type_node(&mut self) -> Result<Node, Error> {
        self.expression(0, false, false)
    }

    fn parameters(&mut self, end: u8) -> Result<Vec<Parameter>, Error> {
        let mut params = Vec::new();
        while self.peek() != Some(end) {
            let offset = self.offset;
            let property = self.peek() == Some(b'@');
            if property {
                self.offset += 1;
            }
            let name = self.name()?;
            self.expect(b':')?;
            // The model checks that this is a type, not an arbitrary value.
            let ty = self.value(0)?;
            params.push(Parameter {
                offset,
                name,
                property,
                ty,
            });
            if self.peek() != Some(end) {
                self.expect(b',')?;
            }
        }
        self.expect(end)?;
        Ok(params)
    }

    fn fields(&mut self, depth: u8) -> Result<BTreeMap<String, Node>, Error> {
        self.expect(b'{')?;
        let mut fields = BTreeMap::new();
        while self.peek() != Some(b'}') {
            let offset = self.offset;
            let field = self.name()?;
            self.expect(b':')?;
            let value = self.value(depth)?;
            if fields.insert(field.clone(), value).is_some() {
                return Err(self.error(offset, format!("duplicate field `{field}`")));
            }
            if self.peek() != Some(b'}') {
                self.expect(b',')?;
            }
        }
        self.expect(b'}')?;
        Ok(fields)
    }

    fn error(&self, offset: usize, message: impl Into<String>) -> Error {
        Error::at(self.source, offset, message)
    }

    fn peek(&mut self) -> Option<u8> {
        let bytes = self.source.as_bytes();
        loop {
            while bytes.get(self.offset).is_some_and(u8::is_ascii_whitespace) {
                self.offset += 1;
            }
            if bytes.get(self.offset..self.offset + 2) == Some(b"//") {
                while bytes.get(self.offset).is_some_and(|&b| b != b'\n') {
                    self.offset += 1;
                }
            } else {
                return bytes.get(self.offset).copied();
            }
        }
    }

    fn expect(&mut self, byte: u8) -> Result<(), Error> {
        if self.peek() != Some(byte) {
            return Err(self.error(self.offset, format!("expected `{}`", byte as char)));
        }
        self.offset += 1;
        Ok(())
    }

    fn name(&mut self) -> Result<String, Error> {
        if !self
            .peek()
            .is_some_and(|b| b.is_ascii_alphabetic() || b == b'_')
        {
            return Err(self.error(self.offset, "expected a name"));
        }
        let start = self.offset;
        self.offset += 1;
        while self
            .source
            .as_bytes()
            .get(self.offset)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'.')
        {
            self.offset += 1;
        }
        Ok(self.source[start..self.offset].to_owned())
    }

    fn value(&mut self, depth: u8) -> Result<Node, Error> {
        self.expression(depth, true, false)
    }

    // Intersection binds more tightly than union. N-ary nodes avoid recursive
    // ASTs for long flat expressions; only explicit nesting consumes depth.
    fn expression(&mut self, depth: u8, objects: bool, intersection: bool) -> Result<Node, Error> {
        let mut parts = Vec::new();
        loop {
            parts.push(if intersection {
                self.atom(depth, objects)?
            } else {
                self.expression(depth, objects, true)?
            });
            if self.peek() != Some(if intersection { b'&' } else { b'|' }) {
                break;
            }
            self.offset += 1;
        }
        if parts.len() == 1 {
            return Ok(parts.pop().unwrap());
        }
        Ok(Node {
            offset: parts[0].offset,
            kind: if intersection {
                Kind::Intersection(parts)
            } else {
                Kind::Union(parts)
            },
        })
    }

    fn atom(&mut self, depth: u8, objects: bool) -> Result<Node, Error> {
        if depth >= 64 {
            return Err(self.error(self.offset, "definition nesting exceeds 64 levels"));
        }
        let token = self.peek();
        let offset = self.offset;
        let kind = match token {
            Some(b'(') => {
                self.offset += 1;
                let node = self.expression(depth + 1, objects, false)?;
                self.expect(b')')?;
                return Ok(node);
            }
            Some(b'[') => {
                self.offset += 1;
                Kind::List(self.sequence(b']', depth + 1)?)
            }
            Some(b'"') => {
                self.offset += 1;
                let mut text = String::new();
                loop {
                    let Some(ch) = self.source[self.offset..].chars().next() else {
                        return Err(self.error(offset, "unterminated string"));
                    };
                    self.offset += ch.len_utf8();
                    match ch {
                        '"' => break,
                        '\\' => {
                            let escape = self.source.as_bytes().get(self.offset).copied();
                            let ch = match escape {
                                Some(b'"') => '"',
                                Some(b'\\') => '\\',
                                Some(b'n') => '\n',
                                Some(b'r') => '\r',
                                Some(b't') => '\t',
                                _ => return Err(self.error(self.offset, "invalid string escape")),
                            };
                            self.offset += 1;
                            text.push(ch);
                        }
                        '\n' | '\r' => return Err(self.error(offset, "newline in string")),
                        _ => text.push(ch),
                    }
                }
                Kind::Text(text)
            }
            Some(b'0'..=b'9') => {
                self.offset += 1;
                while self
                    .source
                    .as_bytes()
                    .get(self.offset)
                    .is_some_and(u8::is_ascii_digit)
                {
                    self.offset += 1;
                }
                let value = self.source[offset..self.offset]
                    .parse()
                    .map_err(|_| self.error(offset, "integer is out of range"))?;
                Kind::Number(value)
            }
            _ => {
                let name = self.name()?;
                if self.peek() == Some(b'(') {
                    self.offset += 1;
                    Kind::Call(name, self.sequence(b')', depth + 1)?)
                } else if objects && self.peek() == Some(b'{') {
                    Kind::Object(name, self.fields(depth + 1)?)
                } else {
                    Kind::Name(name)
                }
            }
        };
        Ok(Node { offset, kind })
    }

    fn sequence(&mut self, end: u8, depth: u8) -> Result<Vec<Node>, Error> {
        let mut values = Vec::new();
        while self.peek() != Some(end) {
            values.push(self.value(depth)?);
            if self.peek() != Some(end) {
                self.expect(b',')?;
            }
        }
        self.expect(end)?;
        Ok(values)
    }
}
