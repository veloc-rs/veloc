//! Checked expression rules for direct reductions and e-class queries.
//! Type sources refer to pattern slots; storage layout and application policy
//! belong to the runtime hosts.
use crate::{
    Definitions, Error,
    schema::Operation,
    syntax::{Kind, Node},
    types::TypeSet,
};
use std::collections::{BTreeMap, BTreeSet};
mod attributes;
mod fold;
mod plan;
mod predicate;
pub(super) use attributes::emit_attributes;
pub(super) use fold::emit_folds;
pub(super) use plan::emit_plans;
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
    local_types: bool,
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
    /// The fold interface can inspect only the root and return an operand or
    /// constant. Guards qualify when all their queries are available locally.
    pub fn can_fold_locally(&self) -> bool {
        self.local_types
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
            && self.guard.as_ref().is_none_or(|guard| {
                let PatternKind::Operation { args, .. } = &self.pattern[0].kind else {
                    unreachable!()
                };
                let order: Vec<_> = (0..args.len()).collect();
                guard.local_code(self, "Type", &order).is_some()
            })
    }
}

/// Unsupported host capabilities are distinct from malformed rules. Only the
/// former continue through the instruction adapter; type errors stay errors.
enum CheckError {
    Invalid(Error),
    Instruction,
}
impl From<Error> for CheckError {
    fn from(error: Error) -> Self {
        Self::Invalid(error)
    }
}
type CheckResult<T> = Result<T, CheckError>;

pub(super) fn compile<'a>(
    source: &crate::Source,
    defs: &Definitions,
    dialect: &str,
    cases: &'a [super::optimizer::Case<'a>],
) -> Result<(Vec<CheckedRule>, Vec<&'a super::optimizer::Case<'a>>), Error> {
    let operations: BTreeMap<_, _> = defs.operations().map(|op| (op.name.clone(), op)).collect();
    let functions = source
        .declarations()
        .iter()
        .filter_map(|decl| {
            matches!(&decl.kind, crate::syntax::DeclKind::Function { .. })
                .then_some(decl.name.clone())
        })
        .collect();
    let mut rules = Vec::new();
    let mut instructions = Vec::new();
    for case in cases {
        let mut checker = Checker::new(
            source.text(),
            defs,
            &operations,
            &functions,
            dialect,
            case.generics.clone(),
        );
        match checker.case(case) {
            Ok(rule) => rules.push(rule),
            Err(CheckError::Instruction) => instructions.push(case),
            Err(CheckError::Invalid(error)) => return Err(error),
        }
    }
    Ok((rules, instructions))
}

struct Checker<'a> {
    source: &'a str,
    defs: &'a Definitions,
    operations: &'a BTreeMap<String, Operation>,
    functions: &'a BTreeSet<String>,
    dialect: &'a str,
    generics: BTreeMap<String, TypeSet>,
    bound_types: BTreeMap<String, TypeRef>,
    type_links: BTreeMap<usize, TypeRef>,
    variables: BTreeMap<String, usize>,
    attributes: BTreeMap<String, Attribute>,
    pattern: Vec<Pattern>,
}
impl<'a> Checker<'a> {
    fn new(
        source: &'a str,
        defs: &'a Definitions,
        operations: &'a BTreeMap<String, Operation>,
        functions: &'a BTreeSet<String>,
        dialect: &'a str,
        generics: BTreeMap<String, TypeSet>,
    ) -> Self {
        Self {
            source,
            defs,
            operations,
            functions,
            dialect,
            generics,
            bound_types: BTreeMap::new(),
            type_links: BTreeMap::new(),
            variables: BTreeMap::new(),
            attributes: BTreeMap::new(),
            pattern: Vec::new(),
        }
    }
    fn case(&mut self, case: &super::optimizer::Case<'_>) -> CheckResult<CheckedRule> {
        let op = self.operation(case.operation, case.args.len(), case.offset)?;
        let mut ty = self.type_ref(op.expression.as_ref().unwrap().results[0].domain.clone(), 0);
        for results in &case.results {
            let [result] = *results else {
                return Err(self.fail(case.offset, "value rules require one result type"));
            };
            if matches!(&result.kind, Kind::Name(name) if name == "_") {
                continue;
            }
            let annotated = self.bind_type(result, 0)?;
            ty = self.unify(ty, annotated, result.offset)?;
        }
        let lhs = Node {
            offset: case.offset,
            kind: Kind::Call(case.operation.into(), case.args.to_vec()),
        };
        let rule = self.rule(
            &lhs,
            ty,
            case.replacement,
            case.guard,
            format!("{} case at byte {}", case.operation, case.offset),
        )?;
        if self
            .generics
            .keys()
            .any(|name| self.attributes.contains_key(name) || self.variables.contains_key(name))
        {
            return Err(self.fail(case.offset, "pattern binding shadows type parameter"));
        }
        Ok(rule)
    }
    fn fail(&self, at: usize, message: &str) -> CheckError {
        Error::at(self.source, at, message).into()
    }
    fn operation(&self, path: &str, arity: usize, at: usize) -> CheckResult<&Operation> {
        let model =
            super::optimizer::operation(self.source, self.defs, self.dialect, path, arity, at)?;
        let op = &self.operations[&model.name];
        if !op
            .expression
            .as_ref()
            .is_some_and(|sig| sig.results.len() == 1)
            || !model.meta.memory_free
            || model.traits.contains("MAY_TRAP")
            || model.declaration.params.iter().any(|param| param.moves)
            || model
                .constraints
                .iter()
                .any(|c| c.condition.context_type().is_some())
        {
            return Err(CheckError::Instruction);
        }
        Ok(op)
    }
    fn bind_type(&mut self, node: &Node, slot: usize) -> CheckResult<TypeRef> {
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
    fn exact(&self, node: &Node) -> CheckResult<TypeRef> {
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
    fn output_type(&self, node: &Node) -> CheckResult<TypeRef> {
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
    fn type_ref(&self, domain: TypeSet, slot: usize) -> TypeRef {
        let source = self
            .defs
            .types
            .exact
            .iter()
            .find(|(name, set)| name.starts_with("Type::") && set.is_singleton() && **set == domain)
            .map(|(name, _)| TypeSource::Exact(name.clone()))
            .unwrap_or(TypeSource::Value(slot));
        TypeRef { domain, source }
    }

    fn resolve(&self, mut ty: TypeRef) -> TypeRef {
        while let TypeSource::Value(slot) = ty.source {
            let Some(link) = self.type_links.get(&slot) else {
                break;
            };
            ty.domain.intersect(&link.domain);
            if ty.source == link.source {
                break;
            }
            ty.source = link.source.clone();
        }
        ty
    }

    /// Matching constrains existing types. Merge aliases so annotations and
    /// signature relationships are checked even when a type binds in a child.
    fn unify(&mut self, a: TypeRef, b: TypeRef, at: usize) -> CheckResult<TypeRef> {
        let a = self.resolve(a);
        let b = self.resolve(b);
        let mut domain = a.domain.clone();
        domain.intersect(&b.domain);
        if domain.is_empty() {
            return Err(self.fail(at, "incompatible pattern types"));
        }
        let source = match (&a.source, &b.source) {
            (TypeSource::Exact(_), _) => a.source.clone(),
            (_, TypeSource::Exact(_)) => b.source.clone(),
            (TypeSource::Value(a), TypeSource::Value(b)) => {
                self.type_ref(domain.clone(), (*a).min(*b)).source
            }
        };
        let ty = TypeRef { domain, source };
        for source in [a.source, b.source] {
            if let TypeSource::Value(slot) = source {
                self.type_links.insert(slot, ty.clone());
            }
        }
        Ok(ty)
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
        self.type_ref(input.domain.clone(), slot)
    }

    fn rule(
        &mut self,
        lhs: &Node,
        ty: TypeRef,
        rhs: &Node,
        guard: Option<&Node>,
        name: String,
    ) -> CheckResult<CheckedRule> {
        self.pattern.push(Pattern {
            ty: ty.clone(),
            kind: PatternKind::Value(0),
        });
        self.match_node(lhs, ty.clone(), 0)?;
        // Finish matching before resolving recipe types: an input annotation can
        // constrain the root result through the operation's signature.
        for slot in 0..self.pattern.len() {
            self.pattern[slot].ty = self.resolve(self.pattern[slot].ty.clone());
        }
        for (name, ty) in std::mem::take(&mut self.bound_types) {
            self.bound_types.insert(name, self.resolve(ty));
        }
        let replacement = self.recipe(rhs, Some(self.pattern[0].ty.clone()))?;
        let local_types = self.local_types();
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
            local_types,
        })
    }
    fn match_node(&mut self, node: &Node, expected: TypeRef, slot: usize) -> CheckResult<()> {
        let (node, annotation) = super::optimizer::annotation(node);
        if let Some(types) = annotation {
            let [t] = types else {
                return Err(self.fail(node.offset, "expected one value type"));
            };
            if matches!(&t.kind, Kind::Name(name) if name == "_") {
                return self.match_node(node, expected, slot);
            }
            let annotated = self.bind_type(t, slot)?;
            let ty = self.unify(expected, annotated, node.offset)?;
            return self.match_node(node, ty, slot);
        }
        let mut ty = self.resolve(expected);
        let kind = if let Some(bits) = literal(node) {
            let mut literal_ty = ty.clone();
            check_literal(self.source, node, &mut literal_ty.domain)?;
            ty = self.unify(ty, literal_ty, node.offset)?;
            PatternKind::Constant(bits)
        } else {
            match &node.kind {
                Kind::Name(name) => {
                    if self.attributes.contains_key(name) {
                        return Err(self.fail(node.offset, "value shadows an attribute binding"));
                    }
                    if let Some(&previous) = self.variables.get(name) {
                        ty = self.unify(ty, self.pattern[previous].ty.clone(), node.offset)?;
                    }
                    PatternKind::Value(*self.variables.entry(name.clone()).or_insert(slot))
                }
                Kind::Call(path, args) => {
                    let op = self.operation(path, args.len(), node.offset)?;
                    let sig = op.expression.as_ref().unwrap().clone();
                    let params = op.declaration.params.clone();
                    let opcode = op.name.clone();
                    ty = self.unify(
                        ty,
                        self.type_ref(sig.results[0].domain.clone(), slot),
                        node.offset,
                    )?;
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
        self.pattern[slot] = Pattern {
            ty: self.resolve(ty),
            kind,
        };
        Ok(())
    }
    // The folding host cannot query arbitrary operand types. Restrictions that
    // are not implied by the root signature belong to the graph matcher.
    fn local_types(&self) -> bool {
        let PatternKind::Operation { opcode, args, .. } = &self.pattern[0].kind else {
            unreachable!()
        };
        let sig = self.operations[opcode].expression.as_ref().unwrap();
        let root = &self.pattern[0].ty;
        let mut inputs = BTreeMap::new();
        for (&slot, input) in args.iter().zip(&sig.inputs) {
            let expected = input
                .variable
                .and_then(|v| inputs.get(&v))
                .cloned()
                .unwrap_or_else(|| self.input_type(input, &sig.results[0], root, slot));
            if let Some(var) = input.variable {
                inputs.insert(var, expected.clone());
            }
            let actual = &self.pattern[slot].ty;
            if actual.domain != expected.domain || actual.source != expected.source {
                return false;
            }
        }
        true
    }

    /// Construction may use a known operand type only when the signature ties
    /// that operand to the result. A widening conversion alone gives no width.
    fn infer_result(&self, node: &Node) -> CheckResult<Option<TypeRef>> {
        let (node, types) = super::optimizer::annotation(node);
        if let Some(types) = types {
            let [ty] = types else {
                return Err(self.fail(node.offset, "expected one result type"));
            };
            return self.output_type(ty).map(Some);
        }
        match &node.kind {
            Kind::Name(name) => Ok(self
                .variables
                .get(name)
                .map(|&slot| self.pattern[slot].ty.clone())),
            Kind::Call(path, args) => {
                let op = self.operation(path, args.len(), node.offset)?;
                let sig = op.expression.as_ref().unwrap();
                let result = &sig.results[0];
                if result.domain.is_singleton() {
                    return Ok(Some(self.type_ref(result.domain.clone(), 0)));
                }
                let mut inputs = sig.inputs.iter();
                for (arg, param) in args.iter().zip(&op.declaration.params) {
                    if matches!(&param.ty.kind, Kind::Call(name, _) if name == "Value") {
                        let input = inputs.next().unwrap();
                        if result.variable.is_some() && input.variable == result.variable {
                            if let Some(ty) = self.infer_result(arg)? {
                                return Ok(Some(ty));
                            }
                        }
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn recipe(&self, node: &Node, expected: Option<TypeRef>) -> CheckResult<Recipe> {
        let (node, annotation) = super::optimizer::annotation(node);
        let expected = if let Some(types) = annotation {
            let [ty] = types else {
                return Err(self.fail(node.offset, "expected one result type"));
            };
            let annotated = self.output_type(ty)?;
            if let Some(expected) = &expected {
                let mut common = annotated.domain.clone();
                common.intersect(&expected.domain);
                if common.is_empty() {
                    return Err(self.fail(ty.offset, "incompatible replacement type annotation"));
                }
            }
            Some(annotated)
        } else {
            expected
        };
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
                if annotation.is_some()
                    && expected
                        .as_ref()
                        .is_some_and(|ty| ty.source != self.pattern[slot].ty.source)
                {
                    return Err(self.fail(
                        node.offset,
                        "replacement annotation must match the captured value type",
                    ));
                }
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
            Kind::Call(path, args) => {
                let mut ty = match expected {
                    Some(ty) => Some(ty),
                    None => self.infer_result(node)?,
                }
                .ok_or_else(|| {
                    self.fail(
                        node.offset,
                        "cannot infer construction result type; add -> Type",
                    )
                })?;
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
                // A later operand can establish the type of an earlier one,
                // e.g. Icmp(kind, ExtendU(x), y) uses y's type for the extension.
                for (arg, param) in args.iter().zip(&op.declaration.params) {
                    if matches!(&param.ty.kind, Kind::Call(name, _) if name == "Value") {
                        let input = inputs.next().unwrap();
                        if let Some(variable) = input.variable
                            && !input_types.contains_key(&variable)
                            && let Some(known) = self.infer_result(arg)?
                        {
                            input_types.insert(variable, known);
                        }
                    }
                }
                let mut inputs = sig.inputs.iter();
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
    fn attribute(&self, node: &Node, ty: &str) -> CheckResult<AttributeValue> {
        if let Kind::Name(name) = &node.kind
            && let Some(attribute) = self.attributes.get(name)
        {
            if attribute.ty != ty {
                return Err(self.fail(node.offset, "attribute type mismatch"));
            }
            return Ok(attribute.value.clone());
        }
        if self.needs_instruction(node) {
            return Err(CheckError::Instruction);
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
    // Graph attributes are literals or captures. Computing new attributes from
    // captures, constant facts or user helpers needs the instruction host.
    fn needs_instruction(&self, node: &Node) -> bool {
        match &node.kind {
            Kind::Name(name) => {
                self.variables.contains_key(name)
                    || self.attributes.contains_key(name)
                    || self.bound_types.contains_key(name)
            }
            Kind::Call(name, args) => {
                name == "constant_bits"
                    || self.functions.contains(name)
                    || args.iter().any(|arg| self.needs_instruction(arg))
            }
            Kind::Member(value, _)
            | Kind::Unary(_, value)
            | Kind::Try(value)
            | Kind::Ref(value) => self.needs_instruction(value),
            Kind::Binary(_, a, b) => self.needs_instruction(a) || self.needs_instruction(b),
            Kind::Object(_, fields) | Kind::Record(fields) => {
                fields.values().any(|value| self.needs_instruction(value))
            }
            Kind::Number(_) | Kind::Integer(_) | Kind::Text(_) => false,
            _ => true,
        }
    }

    fn match_attribute(
        &mut self,
        node: &Node,
        ty: &str,
        slot: usize,
        index: usize,
    ) -> CheckResult<AttributeValue> {
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
fn check_literal(source: &str, node: &Node, domain: &mut TypeSet) -> CheckResult<()> {
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
        )
        .into())
    } else {
        Ok(())
    }
}
