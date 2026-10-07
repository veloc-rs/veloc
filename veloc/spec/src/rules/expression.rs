//! Checked expression rules for direct reductions and e-class queries.
//! Type sources refer to pattern slots; storage layout and application policy
//! belong to the runtime hosts.
use crate::{
    Definitions, Error,
    schema::Operation,
    syntax::{DeclKind, Kind, Node, Results},
    types::TypeSet,
};
use std::collections::{BTreeMap, BTreeSet};
mod attributes;
mod local;
mod predicate;
pub(super) use attributes::emit_attributes;
pub(super) use local::emit_local;
use predicate::{GuardType, Predicate};

#[derive(Clone)]
pub(super) struct TypeRef {
    pub domain: TypeSet,
    pub source: TypeSource,
}
#[derive(Clone, PartialEq, Eq)]
pub(super) enum TypeSource {
    Value(usize),
    Exact(String),
}
impl TypeRef {
    fn code(&self, types: &str, captures: &[usize]) -> String {
        match &self.source {
            TypeSource::Value(slot) => format!("cx.ty({})", capture_code(captures, *slot)),
            TypeSource::Exact(name) => name.replacen("Type", types, 1),
        }
    }
}

fn capture_code(captures: &[usize], slot: usize) -> String {
    let index = captures
        .iter()
        .position(|&captured| captured == slot)
        .expect("referenced pattern slot must be captured");
    format!("captures[{index}]")
}

pub(super) struct Pattern {
    pub ty: TypeRef,
    pub kind: PatternKind,
}
pub(super) enum PatternKind {
    Value(usize), // Slot of the first occurrence of this variable.
    Constant(u64),
    Operation {
        opcode: String,
        args: Vec<usize>,
        commutative: bool,
        attributes: Vec<Attribute>,
    },
}

pub(super) enum Recipe {
    Value(usize),
    Constant(TypeRef, u64),
    Build {
        opcode: String,
        ty: TypeRef,
        args: Vec<Recipe>,
        attributes: Vec<Attribute>,
    },
}

#[derive(Clone)]
pub(super) struct Attribute {
    pub name: String,
    pub ty: String,
    pub value: AttributeValue,
}
#[derive(Clone, PartialEq, Eq)]
pub(super) enum AttributeValue {
    Binding { node: usize, index: usize },
    Literal(String),
}
impl AttributeValue {
    fn code(&self) -> String {
        match self {
            Self::Binding { node, index } => format!("attr_{node}_{index}"),
            Self::Literal(code) => code.clone(),
        }
    }
}

pub(super) struct CheckedRule {
    pub pattern: Vec<Pattern>,
    pub replacement: Recipe,
    pub guard: Option<Predicate>,
    pub name: String,
}
pub(super) enum Filter {
    IsConstant(usize),
    Constant(usize, u64, bool),
}
impl CheckedRule {
    /// Operand permutations are shared by e-class queries and direct reductions.
    pub fn orders(&self, slot: usize) -> Vec<Vec<usize>> {
        let PatternKind::Operation {
            args, commutative, ..
        } = &self.pattern[slot].kind
        else {
            unreachable!()
        };
        let mut orders = vec![(0..args.len()).collect()];
        if *commutative && let [a, b] = args.as_slice() {
            let symmetric = match (&self.pattern[*a].kind, &self.pattern[*b].kind) {
                (PatternKind::Value(a), PatternKind::Value(b)) => a == b,
                (PatternKind::Constant(a), PatternKind::Constant(b)) => a == b,
                _ => false,
            };
            if !symmetric {
                orders.push(vec![1, 0]);
            }
        }
        orders
    }
    pub fn captures(&self) -> Vec<usize> {
        fn ty(ty: &TypeRef, out: &mut BTreeSet<usize>) {
            if let TypeSource::Value(slot) = ty.source {
                out.insert(slot);
            }
        }
        fn visit(recipe: &Recipe, out: &mut BTreeSet<usize>) {
            match recipe {
                Recipe::Value(slot) => {
                    out.insert(*slot);
                }
                Recipe::Constant(t, _) => ty(t, out),
                Recipe::Build { ty: t, args, .. } => {
                    ty(t, out);
                    for arg in args {
                        visit(arg, out);
                    }
                }
            }
        }
        let mut slots = BTreeSet::new();
        if let Some(guard) = &self.guard {
            guard.captures(&mut slots);
        }
        slots.insert(0);
        visit(&self.replacement, &mut slots);
        slots.into_iter().collect()
    }
    pub fn nodes(&self) -> Vec<usize> {
        self.pattern.iter().enumerate().filter_map(|(slot, p)| {
            matches!(&p.kind, PatternKind::Operation { attributes, .. } if !attributes.is_empty()).then_some(slot)
        }).collect()
    }
    pub fn filters(&self) -> Vec<Filter> {
        let mut out = Vec::new();
        if let Some(guard) = &self.guard {
            guard.filters(&mut out);
        }
        out
    }
    pub fn opcode(&self) -> &str {
        let PatternKind::Operation { opcode, .. } = &self.pattern[0].kind else {
            unreachable!()
        };
        opcode
    }
    /// Allocation-free reductions remain a specialized projection of the model.
    pub fn is_flat(&self) -> bool {
        self.guard.is_none()
            && self
                .pattern
                .iter()
                .skip(1)
                .all(|p| !matches!(p.kind, PatternKind::Operation { .. }))
            && !matches!(self.replacement, Recipe::Build { .. })
            && match &self.replacement {
                Recipe::Value(slot) => self.pattern[*slot].ty.source == self.pattern[0].ty.source,
                Recipe::Constant(..) => true,
                _ => false,
            }
    }
}

pub(super) fn compile(
    source: &crate::Source,
    defs: &Definitions,
    dialect: &str,
) -> Result<Vec<CheckedRule>, Error> {
    let aliases: BTreeMap<_, _> = source
        .declarations()
        .iter()
        .filter_map(|d| match &d.kind {
            DeclKind::TypeSet(n) => Some((d.name.clone(), n.clone())),
            _ => None,
        })
        .collect();
    let operations: BTreeMap<_, _> = defs.operations().map(|op| (op.name.clone(), op)).collect();
    let mut rules = Vec::new();
    // Primitive laws and authored rules enter exactly the same checker.
    for op in &defs.ops {
        let Some(sem) = &op.semantics else { continue };
        if sem.primitive().is_none() {
            continue;
        }
        let mut domains = BTreeMap::<_, TypeSet>::new();
        for instance in &sem.instances {
            let [a, b, result] = instance.kinds.as_slice() else {
                continue;
            };
            if !instance.scalar || instance.error.is_some() || a != b || a != result {
                continue;
            }
            let bits = match a {
                crate::types::Primitive::Int(n) if *n <= 64 => *n,
                crate::types::Primitive::Bool => 1,
                _ => continue,
            };
            let constant = |v: veloc_semantics::BvConst| {
                if v == veloc_semantics::BvConst::AllOnes {
                    u64::MAX
                } else {
                    v.eval(bits as u16).unwrap() as u64
                }
            };
            let scalar = defs.types.scalars.iter().find(|s| s.ty == *a).unwrap();
            let domain = defs
                .type_domain(&format!("Type::{}", scalar.exact()))
                .unwrap();
            domains
                .entry((op.identity.map(constant), op.absorbing.map(constant)))
                .or_default()
                .union(domain);
        }
        for ((identity, absorbing), domain) in domains {
            let node = |kind| Node { offset: 0, kind };
            let x = node(Kind::Name("x".into()));
            let mut laws = Vec::new();
            if let Some(c) = identity {
                laws.push((node(Kind::Integer(c.into())), x.clone()));
            }
            if let Some(c) = absorbing {
                laws.push((node(Kind::Integer(c.into())), node(Kind::Integer(c.into()))));
            }
            if op.traits.contains("IDEMPOTENT") {
                laws.push((x.clone(), x.clone()));
            }
            for (arg, rhs) in laws {
                let mut checker =
                    Checker::new(source.text(), defs, &operations, dialect, BTreeMap::new());
                let ty = TypeRef {
                    domain: domain.clone(),
                    source: TypeSource::Value(0),
                };
                let root = node(Kind::Call(
                    format!("{dialect}::{}", op.name),
                    vec![x.clone(), arg],
                ));
                rules.push(checker.rule(
                    &root,
                    ty,
                    &rhs,
                    None,
                    format!("{} primitive law", op.name),
                )?);
            }
        }
    }
    for decl in source.declarations() {
        let DeclKind::Rule(sig) = &decl.kind else {
            continue;
        };
        let fail = |message| Error::at(source.text(), decl.offset, message);
        if decl.fields.keys().any(|k| k != "cases") {
            return Err(fail("expression groups only contain cases"));
        }
        if !matches!(&sig.results, Results::Fixed(r) if r.is_empty()) {
            return Err(fail("expression rules do not declare return values"));
        }
        let [root] = sig.params.as_slice() else {
            return Err(fail("expected one root instruction"));
        };
        if root.moves {
            return Err(fail("expression rules cannot move their root"));
        }
        let Kind::Name(path) = &root.ty.kind else {
            return Err(fail(
                "declare only the root operation here; put its result type on each case",
            ));
        };
        let Some(Node {
            kind: Kind::List(cases),
            ..
        }) = decl.fields.get("cases")
        else {
            return Err(fail("expected cases"));
        };
        if cases.is_empty() {
            return Err(fail("expected nonempty cases"));
        }
        for case in cases {
            let fail = |message| Error::at(source.text(), case.offset, message);
            let Kind::Record(fields) = &case.kind else {
                return Err(fail("expected case"));
            };
            if fields
                .keys()
                .any(|k| !matches!(k.as_str(), "match" | "emit" | "when" | "generics"))
            {
                return Err(fail("unknown expression case field"));
            }
            let mut generics = BTreeMap::new();
            if let Some(params) = fields.get("generics") {
                let Kind::Record(params) = &params.kind else {
                    return Err(fail("expected case type parameters"));
                };
                for (name, bound) in params {
                    let domain = defs.types.set(
                        source.text(),
                        &expand_domain(source.text(), bound, &aliases, &mut BTreeSet::new())?,
                    )?;
                    if domain.is_empty()
                        || name == &root.name
                        || generics.insert(name.clone(), domain).is_some()
                    {
                        return Err(fail("invalid or shadowed case type parameter"));
                    }
                }
            }
            let pattern = fields
                .get("match")
                .ok_or_else(|| fail("expected root pattern"))?;
            let (args, ty) = match &pattern.kind {
                Kind::TypedCall(name, types, args) if name == &root.name && types.len() == 1 => {
                    (args, &types[0])
                }
                _ => {
                    return Err(fail(
                        "expected the declared root with one result type: root<T>(...)",
                    ));
                }
            };
            let rhs = fields
                .get("emit")
                .ok_or_else(|| fail("expected replacement"))?;
            let mut checker =
                Checker::new(source.text(), defs, &operations, dialect, generics.clone());
            let ty = checker.bind_type(ty, 0)?;
            let lhs = Node {
                offset: case.offset,
                kind: Kind::Call(path.clone(), args.clone()),
            };
            let rule = checker.rule(
                &lhs,
                ty,
                rhs,
                fields.get("when"),
                format!("{path} case at byte {}", case.offset),
            )?;
            if checker.attributes.contains_key(&root.name)
                || generics.keys().any(|n| checker.attributes.contains_key(n))
                || checker.variables.contains_key(&root.name)
                || generics.keys().any(|n| checker.variables.contains_key(n))
            {
                return Err(fail("pattern binding shadows root or type parameter"));
            }
            rules.push(rule);
        }
    }
    Ok(rules)
}

struct Checker<'a> {
    source: &'a str,
    defs: &'a Definitions,
    operations: &'a BTreeMap<String, Operation>,
    dialect: &'a str,
    generics: BTreeMap<String, TypeSet>,
    bound_types: BTreeMap<String, TypeRef>,
    variables: BTreeMap<String, usize>,
    attributes: BTreeMap<String, Attribute>,
    pattern: Vec<Pattern>,
}
impl<'a> Checker<'a> {
    fn new(
        source: &'a str,
        defs: &'a Definitions,
        operations: &'a BTreeMap<String, Operation>,
        dialect: &'a str,
        generics: BTreeMap<String, TypeSet>,
    ) -> Self {
        Self {
            source,
            defs,
            operations,
            dialect,
            generics,
            bound_types: BTreeMap::new(),
            variables: BTreeMap::new(),
            attributes: BTreeMap::new(),
            pattern: Vec::new(),
        }
    }
    fn fail(&self, at: usize, message: &str) -> Error {
        Error::at(self.source, at, message)
    }
    fn operation(&self, path: &str, arity: usize, at: usize) -> Result<&Operation, Error> {
        let name = path
            .strip_prefix(&format!("{}::", self.dialect))
            .ok_or_else(|| self.fail(at, "unknown expression dialect"))?;
        let op = self
            .operations
            .get(name)
            .ok_or_else(|| self.fail(at, "unknown expression operation"))?;
        let sig = op
            .expression
            .as_ref()
            .ok_or_else(|| self.fail(at, "expression requires fixed values and attributes"))?;
        let model = self.defs.ops.iter().find(|o| o.name == name).unwrap();
        if op.declaration.params.len() != arity
            || sig.results.len() != 1
            || model.traits.contains("MAY_TRAP")
            || model
                .constraints
                .iter()
                .any(|c| c.condition.context_type().is_some())
        {
            return Err(self.fail(at, "expression requires a non-trapping single-result operation with context-free constraints"));
        }
        Ok(op)
    }
    fn bind_type(&mut self, node: &Node, slot: usize) -> Result<TypeRef, Error> {
        let Kind::Name(name) = &node.kind else {
            return Err(self.fail(node.offset, "expected a type parameter or exact type"));
        };
        if let Some(ty) = self.bound_types.get(name) {
            return Ok(ty.clone());
        }
        if let Some(domain) = self.generics.get(name) {
            let ty = TypeRef {
                domain: domain.clone(),
                source: TypeSource::Value(slot),
            };
            self.bound_types.insert(name.clone(), ty.clone());
            return Ok(ty);
        }
        self.exact(node)
    }
    fn exact(&self, node: &Node) -> Result<TypeRef, Error> {
        if let Kind::Name(name) = &node.kind
            && let Some(domain) = self.defs.type_domain(name).filter(|s| s.is_singleton())
        {
            return Ok(TypeRef {
                domain: domain.clone(),
                source: TypeSource::Exact(name.clone()),
            });
        }
        Err(self.fail(node.offset, "unknown or unbound result type"))
    }
    fn output_type(&self, node: &Node) -> Result<TypeRef, Error> {
        if let Kind::Call(name, args) = &node.kind
            && name == "type_of"
            && let [
                Node {
                    kind: Kind::Name(value),
                    ..
                },
            ] = args.as_slice()
            && let Some(&slot) = self.variables.get(value)
        {
            return Ok(self.pattern[slot].ty.clone());
        }
        if let Kind::Name(name) = &node.kind
            && let Some(ty) = self.bound_types.get(name)
        {
            return Ok(ty.clone());
        }
        self.exact(node)
    }
    fn input_type(
        &self,
        input: &crate::schema::Term,
        result: &crate::schema::Term,
        ty: &TypeRef,
        slot: usize,
    ) -> TypeRef {
        if result.variable.is_some() && input.variable == result.variable {
            return ty.clone();
        }
        let source = self
            .defs
            .types
            .exact
            .iter()
            .find(|(n, d)| n.starts_with("Type::") && d.is_singleton() && **d == input.domain)
            .map(|(n, _)| TypeSource::Exact(n.clone()))
            .unwrap_or(TypeSource::Value(slot));
        TypeRef {
            domain: input.domain.clone(),
            source,
        }
    }
    fn rule(
        &mut self,
        lhs: &Node,
        ty: TypeRef,
        rhs: &Node,
        guard: Option<&Node>,
        name: String,
    ) -> Result<CheckedRule, Error> {
        self.pattern.push(Pattern {
            ty: ty.clone(),
            kind: PatternKind::Value(0),
        });
        self.match_node(lhs, ty.clone(), 0)?;
        let replacement = self.recipe(rhs, Some(ty))?;
        let guard = guard
            .map(|node| {
                let (guard, ty) = self.guard(node)?;
                if ty != GuardType::Bool {
                    return Err(self.fail(node.offset, "guard must be boolean"));
                }
                Ok(guard)
            })
            .transpose()?;
        Ok(CheckedRule {
            pattern: std::mem::take(&mut self.pattern),
            replacement,
            guard,
            name,
        })
    }
    fn match_node(&mut self, node: &Node, expected: TypeRef, slot: usize) -> Result<(), Error> {
        let mut ty = match &node.kind {
            Kind::TypedCall(_, ts, _) => {
                let [t] = ts.as_slice() else {
                    return Err(self.fail(node.offset, "expected one explicit result type"));
                };
                self.bind_type(t, slot)?
            }
            _ => expected,
        };
        let kind = if let Some(bits) = literal(node) {
            check_literal(self.source, node, &mut ty.domain)?;
            PatternKind::Constant(bits)
        } else {
            match &node.kind {
                Kind::Name(name) => {
                    if self.attributes.contains_key(name) {
                        return Err(self.fail(node.offset, "value shadows an attribute binding"));
                    }
                    if let Some(&previous) = self.variables.get(name) {
                        let mut common = self.pattern[previous].ty.domain.clone();
                        common.intersect(&ty.domain);
                        if common.is_empty() {
                            return Err(
                                self.fail(node.offset, "repeated value has incompatible types")
                            );
                        }
                    }
                    PatternKind::Value(*self.variables.entry(name.clone()).or_insert(slot))
                }
                Kind::Call(path, args) | Kind::TypedCall(path, _, args) => {
                    let op = self.operation(path, args.len(), node.offset)?;
                    let sig = op.expression.as_ref().unwrap().clone();
                    let params = op.declaration.params.clone();
                    let opcode = op.name.clone();
                    if !ty.domain.subset_of(&sig.results[0].domain) {
                        return Err(
                            self.fail(node.offset, "pattern result type exceeds operation domain")
                        );
                    }
                    let commutative = self
                        .defs
                        .ops
                        .iter()
                        .find(|o| o.name == opcode)
                        .unwrap()
                        .traits
                        .contains("COMMUTATIVE");
                    let mut attributes = Vec::new();
                    let mut values = Vec::new();
                    for (arg, param) in args.iter().zip(&params) {
                        if matches!(&param.ty.kind, Kind::Call(name, _) if name == "Value") {
                            values.push(arg);
                        } else {
                            let Kind::Name(ty) = &param.ty.kind else {
                                return Err(
                                    self.fail(arg.offset, "expected a named attribute type")
                                );
                            };
                            let value = self.match_attribute(arg, ty, slot, attributes.len())?;
                            attributes.push(Attribute {
                                name: param.name.clone(),
                                ty: ty.clone(),
                                value,
                            });
                        }
                    }
                    let first = self.pattern.len();
                    let slots: Vec<_> = (first..first + values.len()).collect();
                    for (&s, input) in slots.iter().zip(&sig.inputs) {
                        self.pattern.push(Pattern {
                            ty: self.input_type(input, &sig.results[0], &ty, s),
                            kind: PatternKind::Value(s),
                        });
                    }
                    let mut input_types = BTreeMap::new();
                    for ((arg, &s), input) in values.iter().zip(&slots).zip(&sig.inputs) {
                        let expected = input
                            .variable
                            .and_then(|v| input_types.get(&v))
                            .cloned()
                            .unwrap_or_else(|| self.pattern[s].ty.clone());
                        self.match_node(arg, expected, s)?;
                        if let Some(variable) = input.variable {
                            input_types
                                .entry(variable)
                                .or_insert_with(|| self.pattern[s].ty.clone());
                        }
                    }
                    PatternKind::Operation {
                        opcode,
                        args: slots,
                        commutative,
                        attributes,
                    }
                }
                _ => return Err(self.fail(node.offset, "expected a value, literal or operation")),
            }
        };
        self.pattern[slot] = Pattern { ty, kind };
        Ok(())
    }
    fn recipe(&self, node: &Node, expected: Option<TypeRef>) -> Result<Recipe, Error> {
        if let Some(bits) = literal(node) {
            let mut ty = expected
                .ok_or_else(|| self.fail(node.offset, "literal needs an unambiguous type"))?;
            check_literal(self.source, node, &mut ty.domain)?;
            return Ok(Recipe::Constant(ty, bits));
        }
        match &node.kind {
            Kind::Name(name) => {
                let slot = *self
                    .variables
                    .get(name)
                    .ok_or_else(|| self.fail(node.offset, "unmatched replacement value"))?;
                if let Some(expected) = expected {
                    let mut common = self.pattern[slot].ty.domain.clone();
                    common.intersect(&expected.domain);
                    if common.is_empty() {
                        return Err(
                            self.fail(node.offset, "replacement value has incompatible type")
                        );
                    }
                }
                Ok(Recipe::Value(slot))
            }
            Kind::Call(path, args) | Kind::TypedCall(path, _, args) => {
                let mut ty = if let Kind::TypedCall(_, ts, _) = &node.kind {
                    let [t] = ts.as_slice() else {
                        return Err(self.fail(node.offset, "expected one result type"));
                    };
                    self.output_type(t)?
                } else {
                    expected.ok_or_else(|| {
                        self.fail(node.offset, "construction requires an explicit result type")
                    })?
                };
                let op = self.operation(path, args.len(), node.offset)?;
                let sig = op.expression.as_ref().unwrap();
                ty.domain.intersect(&sig.results[0].domain);
                if ty.domain.is_empty() {
                    return Err(self.fail(
                        node.offset,
                        "replacement result type exceeds operation domain",
                    ));
                }
                let mut attributes = Vec::new();
                let mut values = Vec::new();
                let mut inputs = sig.inputs.iter();
                let mut input_types = BTreeMap::new();
                for (arg, param) in args.iter().zip(&op.declaration.params) {
                    if matches!(&param.ty.kind, Kind::Call(name, _) if name == "Value") {
                        let input = inputs.next().unwrap();
                        let expected = if (sig.results[0].variable.is_some()
                            && input.variable == sig.results[0].variable)
                            || input.domain.is_singleton()
                        {
                            Some(self.input_type(input, &sig.results[0], &ty, 0))
                        } else {
                            input.variable.and_then(|v| input_types.get(&v)).cloned()
                        };
                        let recipe = self.recipe(arg, expected)?;
                        if let Some(variable) = input.variable {
                            input_types
                                .entry(variable)
                                .or_insert_with(|| self.recipe_type(&recipe));
                        }
                        values.push(recipe);
                    } else {
                        let Kind::Name(ty) = &param.ty.kind else {
                            return Err(self.fail(arg.offset, "expected a named attribute type"));
                        };
                        attributes.push(Attribute {
                            name: param.name.clone(),
                            ty: ty.clone(),
                            value: self.attribute(arg, ty)?,
                        });
                    }
                }
                let recipe = Recipe::Build {
                    opcode: op.name.clone(),
                    ty,
                    args: values,
                    attributes,
                };
                // Reuse matched subexpressions instead of reconstructing them.
                // Their SSA witnesses are already available at the matched root.
                Ok((0..self.pattern.len())
                    .find(|&slot| self.same_recipe(&recipe, slot))
                    .map(Recipe::Value)
                    .unwrap_or(recipe))
            }
            _ => Err(self.fail(node.offset, "invalid replacement expression")),
        }
    }
    fn recipe_type(&self, recipe: &Recipe) -> TypeRef {
        match recipe {
            Recipe::Value(slot) => self.pattern[*slot].ty.clone(),
            Recipe::Constant(ty, _) | Recipe::Build { ty, .. } => ty.clone(),
        }
    }
    fn same_recipe(&self, recipe: &Recipe, slot: usize) -> bool {
        if let Recipe::Value(value) = recipe
            && *value == slot
        {
            return true;
        }
        match (recipe, &self.pattern[slot].kind) {
            (Recipe::Value(a), PatternKind::Value(b)) => a == b,
            (Recipe::Constant(ty, a), PatternKind::Constant(b)) => {
                a == b && ty.source == self.pattern[slot].ty.source
            }
            (
                Recipe::Build {
                    opcode,
                    ty,
                    args,
                    attributes,
                },
                PatternKind::Operation {
                    opcode: op,
                    args: inputs,
                    attributes: props,
                    ..
                },
            ) => {
                opcode == op
                    && ty.source == self.pattern[slot].ty.source
                    && args.len() == inputs.len()
                    && args
                        .iter()
                        .zip(inputs)
                        .all(|(arg, &input)| self.same_recipe(arg, input))
                    && attributes.len() == props.len()
                    && attributes
                        .iter()
                        .zip(props)
                        .all(|(a, b)| a.ty == b.ty && a.value == b.value)
            }
            _ => false,
        }
    }
    fn attribute(&self, node: &Node, ty: &str) -> Result<AttributeValue, Error> {
        if let Kind::Name(name) = &node.kind
            && let Some(attribute) = self.attributes.get(name)
        {
            if attribute.ty != ty {
                return Err(self.fail(node.offset, "attribute type mismatch"));
            }
            return Ok(attribute.value.clone());
        }
        let code = self.defs.expressions.clone().expression(
            self.source,
            &[],
            crate::model::Vocabulary {
                types: &self.defs.types,
                encodings: &self.defs.encodings,
                data: &self.defs.data,
            },
            node,
            ty,
            &BTreeMap::new(),
            "veloc_mir::inst::",
        )?;
        Ok(AttributeValue::Literal(code))
    }
    fn match_attribute(
        &mut self,
        node: &Node,
        ty: &str,
        slot: usize,
        index: usize,
    ) -> Result<AttributeValue, Error> {
        if let Kind::Name(name) = &node.kind
            && super::identifier(name)
            && !matches!(name.as_str(), "true" | "false")
        {
            if self.variables.contains_key(name) || self.bound_types.contains_key(name) {
                return Err(self.fail(node.offset, "attribute shadows a value or type binding"));
            }
            self.attributes
                .entry(name.clone())
                .or_insert_with(|| Attribute {
                    name: name.clone(),
                    ty: ty.into(),
                    value: AttributeValue::Binding { node: slot, index },
                });
        }
        self.attribute(node, ty)
    }
}
fn literal(node: &Node) -> Option<u64> {
    match &node.kind {
        Kind::Name(n) if n == "true" => Some(1),
        Kind::Name(n) if n == "false" => Some(0),
        Kind::Number(n) => Some(u64::from(*n)),
        Kind::Integer(n) => u64::try_from(*n)
            .ok()
            .or_else(|| i64::try_from(*n).ok().map(|n| n as u64)),
        Kind::Unary("-", n) => literal(n).map(u64::wrapping_neg),
        _ => None,
    }
}
fn check_literal(source: &str, node: &Node, domain: &mut TypeSet) -> Result<(), Error> {
    domain.0.retain(|kind, shapes| {
        *shapes &= 1;
        matches!(
            kind,
            crate::types::Primitive::Int(1..=64) | crate::types::Primitive::Bool
        ) && *shapes != 0
    });
    if domain.is_empty() {
        Err(Error::at(
            source,
            node.offset,
            "literal requires a scalar integer or boolean",
        ))
    } else {
        Ok(())
    }
}

fn expand_domain(
    source: &str,
    node: &Node,
    aliases: &BTreeMap<String, Node>,
    active: &mut BTreeSet<String>,
) -> Result<Node, Error> {
    if let Kind::Name(name) = &node.kind
        && let Some(alias) = aliases.get(name)
    {
        if !active.insert(name.clone()) {
            return Err(Error::at(source, node.offset, "cyclic type-set alias"));
        }
        let result = expand_domain(source, alias, aliases, active);
        active.remove(name);
        return result;
    }
    let mut result = node.clone();
    if let Kind::Union(parts) | Kind::Intersection(parts) | Kind::Call(_, parts) = &mut result.kind
    {
        for part in parts {
            *part = expand_domain(source, part, aliases, active)?;
        }
    }
    Ok(result)
}
