//! Candidate-local expressions lower directly to matching and deferred builds.
//! Pure field reads are interned; construction does not mutate the host IR.
use super::*;

#[derive(Default)]
struct Case {
    fields: BTreeMap<String, Pattern>,
    definitions: Vec<DefMatch>,
    aliases: BTreeMap<String, String>,
    names: BTreeSet<String>,
    temps: Vec<(String, String)>,
    builds: Vec<Constructor>,
    handles: BTreeMap<String, usize>,
    constructing: bool,
}

impl Case {
    fn field(&mut self, field: String) -> String {
        let name = format!("__field_{}", field.replace('.', "_"));
        self.fields
            .entry(field)
            .or_insert_with(|| Pattern::Variable(name.clone()));
        name
    }
}

impl Reader<'_> {
    pub(super) fn selection_case(
        &self,
        case: &Node,
        root: &str,
        opcode: &str,
        type_args: &[Vec<String>],
    ) -> Result<Def, Error> {
        let mut state = Case::default();
        state.names.insert(root.into());
        let mut committed = false;
        for statement in self.list(case)? {
            if committed {
                return Err(self.error(statement, "replace must be the last operation"));
            }
            if let Kind::Let(name, value) = &statement.kind {
                if name.starts_with("__") || !state.names.insert(name.clone()) {
                    return Err(
                        self.error(statement, "binding must be fresh and cannot start with __")
                    );
                }
                match &value.kind {
                    Kind::TypedCall(op, types, args)
                        if op == "def"
                            && types.len() == 1
                            && args.len() == 1
                            && !state.constructing =>
                    {
                        let (opcode, type_args) = self.selection_type(&types[0])?;
                        let input = self.selection_field(&args[0], root, &state)?;
                        state.field(input.clone());
                        state.definitions.push(DefMatch {
                            name: name.clone(),
                            input,
                            opcode,
                            type_args,
                            schema: String::new(),
                        });
                    }
                    Kind::Call(op, args) if op == "temp" && args.len() == 1 => {
                        let exemplar = self.selection_value(&args[0], root, &mut state)?;
                        state.constructing = true;
                        state.temps.push((name.clone(), exemplar));
                    }
                    Kind::Call(op, args) if op == "build" && args.len() == 1 => {
                        let index = self.selection_build(&args[0], root, &mut state)?;
                        state.handles.insert(name.clone(), index);
                    }
                    _ => {
                        let field = self.selection_field(value, root, &state)?;
                        state.field(field.clone());
                        state.aliases.insert(name.clone(), field);
                    }
                }
                continue;
            }
            let Kind::Call(op, args) = &statement.kind else {
                return Err(self.error(statement, "expected require or replace"));
            };
            match (op.as_str(), args.as_slice()) {
                ("require", [condition]) if !state.constructing => match &condition.kind {
                    Kind::TypedCall(op, types, values)
                        if op == "type_is" && types.len() == 1 && values.len() == 1 =>
                    {
                        let field = self.selection_field(&values[0], root, &state)?;
                        let name = state.field(field.clone());
                        if !matches!(state.fields[&field], Pattern::Variable(_)) {
                            return Err(self.error(condition, "duplicate field constraint"));
                        }
                        let pattern = Node {
                            offset: condition.offset,
                            kind: Kind::TypedCall(
                                "Value".into(),
                                types.clone(),
                                vec![Node {
                                    offset: values[0].offset,
                                    kind: Kind::Name(name),
                                }],
                            ),
                        };
                        state.fields.insert(field, self.pattern(&pattern)?);
                    }
                    Kind::Call(op, args) if op == "matches" && args.len() == 2 => {
                        let field = self.selection_field(&args[0], root, &state)?;
                        let pattern = self.pattern(&args[1])?;
                        if !matches!(pattern, Pattern::CondCode(_) | Pattern::IntConst(_)) {
                            return Err(
                                self.error(condition, "matches requires a literal condition")
                            );
                        }
                        let binding = Pattern::Variable(state.field(field.clone()));
                        if !matches!(state.fields[&field], Pattern::Variable(_)) {
                            return Err(self.error(condition, "duplicate field constraint"));
                        }
                        state
                            .fields
                            .insert(field, Pattern::And(vec![binding, pattern]));
                    }
                    _ => {
                        return Err(self.error(
                            condition,
                            "expected type_is<T>(value) or matches(value, literal)",
                        ));
                    }
                },
                ("replace", [node, replacement]) if self.name(node)? == root => {
                    let replacements = match &replacement.kind {
                        Kind::List(values) => values.as_slice(),
                        _ => std::slice::from_ref(replacement),
                    };
                    let mut selected = Vec::new();
                    for value in replacements {
                        let index = match &value.kind {
                            Kind::Call(op, args) if op == "build" && args.len() == 1 => {
                                self.selection_build(&args[0], root, &mut state)?
                            }
                            Kind::Name(name) => *state
                                .handles
                                .get(name)
                                .ok_or_else(|| self.error(value, "expected a build handle"))?,
                            _ => {
                                return Err(
                                    self.error(value, "expected build(...) or a build handle")
                                );
                            }
                        };
                        selected.push(index);
                    }
                    if selected.is_empty()
                        || selected != (0..state.builds.len()).collect::<Vec<_>>()
                    {
                        return Err(self.error(
                            replacement,
                            "replace must commit all builds once, in construction order",
                        ));
                    }
                    committed = true;
                }
                _ => {
                    return Err(self.error(
                        statement,
                        "unknown operation, invalid arguments or require after construction",
                    ));
                }
            }
        }
        if !committed {
            return Err(self.error(case, "candidate requires replace"));
        }
        Ok(Def::SelectRule(SelectRuleDef {
            opcode: opcode.into(),
            type_args: type_args.to_vec(),
            schema: String::new(),
            definitions: state.definitions,
            fields: state
                .fields
                .into_iter()
                .map(|(name, pattern)| PatternArg::Named {
                    name,
                    pattern: Box::new(pattern),
                })
                .collect(),
            temps: state.temps,
            builds: state.builds,
        }))
    }

    fn selection_field(&self, node: &Node, root: &str, state: &Case) -> Result<String, Error> {
        match &node.kind {
            Kind::Member(base, field) if matches!(&base.kind, Kind::Name(name) if name == root) => {
                Ok(field.clone())
            }
            Kind::Member(base, field) if matches!(&base.kind, Kind::Name(name) if state.definitions.iter().any(|p| &p.name == name)) =>
            {
                let Kind::Name(name) = &base.kind else {
                    unreachable!()
                };
                Ok(format!("{name}.{field}"))
            }
            Kind::Name(name) => state
                .aliases
                .get(name)
                .cloned()
                .ok_or_else(|| self.error(node, "expected a root field or field alias")),
            _ => Err(self.error(node, "expected a root field or field alias")),
        }
    }

    fn selection_value(&self, node: &Node, root: &str, state: &mut Case) -> Result<String, Error> {
        if let Kind::Name(name) = &node.kind {
            if state.temps.iter().any(|(temp, _)| temp == name) {
                return Ok(name.clone());
            }
        }
        let field = self.selection_field(node, root, state)?;
        Ok(state.field(field))
    }

    fn selection_build(&self, node: &Node, root: &str, state: &mut Case) -> Result<usize, Error> {
        let Kind::Call(opcode, args) = &node.kind else {
            return Err(self.error(node, "build requires an instruction constructor"));
        };
        let args = args
            .iter()
            .map(|arg| match &arg.kind {
                Kind::Member(..) | Kind::Name(_) => self
                    .selection_value(arg, root, state)
                    .map(Constructor::Variable),
                _ => {
                    let value = self.constructor(arg)?;
                    if matches!(value, Constructor::Inst { .. }) {
                        return Err(self.error(
                            arg,
                            "nested instruction constructors require separate build operations",
                        ));
                    }
                    Ok(value)
                }
            })
            .collect::<Result<Vec<_>, Error>>()?;
        state.constructing = true;
        let index = state.builds.len();
        state.builds.push(Constructor::Inst {
            opcode: opcode.clone(),
            args,
        });
        Ok(index)
    }
}
