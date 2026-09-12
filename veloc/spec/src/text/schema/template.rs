//! A checked grammar for operand templates with unambiguous boundaries.

use super::*;

pub(super) fn compile(
    checker: &mut Checker<'_>,
    schema: &mut Schema,
    text: &str,
    offset: usize,
) -> Result<(), Error> {
    Parser {
        checker,
        rest: text,
        offset,
    }
    .parse(schema)
}

struct Parser<'a, 's> {
    checker: &'a mut Checker<'s>,
    rest: &'a str,
    // Diagnose at the definition string, including escaped templates.
    offset: usize,
}

impl Parser<'_, '_> {
    fn error(&self, message: &str) -> Error {
        self.checker.error(self.offset, message)
    }

    fn eat(&mut self, token: &str) -> bool {
        if let Some(rest) = self.rest.trim_start().strip_prefix(token) {
            self.rest = rest;
            true
        } else {
            false
        }
    }

    fn expect(&mut self, token: &str) -> Result<(), Error> {
        if self.eat(token) {
            Ok(())
        } else {
            Err(self.error(&format!("expected `{token}` in text template")))
        }
    }

    fn hole(&mut self) -> Result<String, Error> {
        self.expect("{")?;
        let end = self
            .rest
            .find('}')
            .ok_or_else(|| self.error("unclosed text placeholder"))?;
        let field = self.rest[..end].trim().to_owned();
        self.rest = &self.rest[end + 1..];
        Ok(field)
    }

    fn atom(&mut self) -> Result<Atom, Error> {
        let hole = self.hole()?;
        let (path, codec) = match hole.split_once(':') {
            Some((field, codec)) => (field.trim(), Some(codec.trim())),
            None => (hole.as_str(), None),
        };
        self.checker.atom(path, codec, self.offset)
    }

    fn parse(&mut self, schema: &mut Schema) -> Result<(), Error> {
        // A fixed binding consumes a logical field but emits no text. Unlike a
        // record default, it belongs to this projection and must match on print.
        while self.rest.trim_start().starts_with('{') {
            let Some(end) = self.rest.find('}') else {
                break;
            };
            if !self.rest[..end].contains('=') {
                break;
            }
            let hole = self.hole()?;
            let (path, literal) = hole.split_once('=').unwrap();
            let path = path.trim();
            let kind = self.checker.consume(path, self.offset)?;
            let value = match (kind, literal.trim().parse::<u32>()) {
                (AtomKind::Scalar(ty), Ok(n)) if crate::model::data::fits_number(&ty, n.into()) => {
                    Value::Number(n.into())
                }
                _ => {
                    return Err(
                        self.error("fixed text binding requires an in-range integer literal")
                    );
                }
            };
            schema.bindings.push((path.into(), value));
        }
        // Flags belong to the opcode suffix, before all operands.
        if self.rest.trim_start().starts_with("{.") {
            let field = self.hole()?;
            let path = &field[1..];
            if self.checker.consume(path, self.offset)? != AtomKind::Scalar("MemFlags".into()) {
                return Err(self.error("text flags must reference MemFlags"));
            }
            schema.flags = Some(path.into());
        }
        let mut first = true;
        let mut keys = BTreeSet::new();
        while !self.rest.trim().is_empty() {
            let optional = self.eat("[");
            if !first {
                self.expect(",")?;
            } else if optional {
                return Err(
                    self.error("optional named fields must follow an operand or required field")
                );
            }
            if self.rest.trim_start().starts_with('{') {
                if optional || !schema.named.is_empty() {
                    return Err(self.error("positional operands must precede named fields"));
                }
                schema.args.push(self.item()?);
            } else {
                let end = self
                    .rest
                    .find('=')
                    .ok_or_else(|| self.error("expected key={field} in text template"))?;
                let key = self.rest[..end].trim().to_owned();
                if key.is_empty() || !key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_') {
                    return Err(self.error("invalid named field key"));
                }
                self.rest = &self.rest[end + 1..];
                let atom = self.atom()?;
                let mode = if optional {
                    if atom.kind != AtomKind::OptionalValue {
                        return Err(self.error("optional named fields require optional(Value)"));
                    }
                    self.expect("]")?;
                    Mode::Optional
                } else {
                    if matches!(atom.kind, AtomKind::OptionalValue | AtomKind::Values) {
                        return Err(self.error(
                            "named fields require a bounded atom or explicit optional value",
                        ));
                    }
                    Mode::Required
                };
                if !keys.insert(key.clone()) {
                    return Err(self.error(&format!("duplicate named key `{key}`")));
                }
                schema.named.push(Named { atom, key, mode });
            }
            first = false;
        }
        Ok(())
    }

    fn item(&mut self) -> Result<Item, Error> {
        let atom = self.atom()?;
        if atom.kind == AtomKind::OptionalValue {
            return Err(self.error("optional values require an optional named field"));
        }
        if self.eat("(") {
            return self.call(atom);
        }
        if self.rest.trim_start().starts_with('{') {
            if !self.rest.starts_with(char::is_whitespace) {
                return Err(self.error("adjacent text atoms require whitespace"));
            }
            let rhs = self.atom()?;
            if !single_token(&atom.kind) || !single_token(&rhs.kind) {
                return Err(self.error("spaced operands require two single-token atoms"));
            }
            return Ok(Item::Space(
                Box::new(Item::Atom(atom)),
                Box::new(Item::Atom(rhs)),
            ));
        }
        Ok(Item::Atom(atom))
    }

    fn call(&mut self, callee: Atom) -> Result<Item, Error> {
        let args = self.atom()?;
        self.expect(")")?;
        if args.kind != AtomKind::Values {
            return Err(self.error("call arguments must be values"));
        }
        if !matches!(&callee.kind, AtomKind::Value)
            && !matches!(&callee.kind, AtomKind::Scalar(ty) if matches!(ty.as_str(), "FuncId" | "Intrinsic"))
        {
            return Err(self.error("call callee must be a value, FuncId or Intrinsic"));
        }
        let signature = if self.eat(":") {
            if self.rest.trim_start().starts_with("{function(") {
                let hole = self.hole()?;
                let source = hole
                    .strip_prefix("function(")
                    .and_then(|s| s.strip_suffix(')'));
                if source != Some(callee.path.as_str())
                    || callee.kind != AtomKind::Scalar("FuncId".into())
                {
                    return Err(self.error("function signature must reference the FuncId callee"));
                }
                CallSignature::Function
            } else {
                let atom = self.atom()?;
                if atom.kind != AtomKind::Scalar("SigId".into()) {
                    return Err(self.error("call signature must be a SigId"));
                }
                CallSignature::Field(atom)
            }
        } else {
            if callee.kind != AtomKind::Value {
                return Err(self.error("non-value calls require an explicit signature"));
            }
            CallSignature::Value
        };
        Ok(Item::Invoke {
            callee,
            args,
            signature,
        })
    }
}
