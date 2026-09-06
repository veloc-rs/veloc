//! Runtime contracts for MIR operations.
//!
//! Layouts, type schemes and operation tables are compiled from `defs/*.ops`.
//! This module implements their shared type/effect machinery; it does not
//! contain individual operation definitions.

include!(concat!(env!("OUT_DIR"), "/formats.rs"));

/// A set of value types accepted by a type pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeClass {
    Any,
    Scalar,
    ScalarInteger,
    ScalarIntegerOrPointer,
    ScalarFloat,
    Integer,
    IntegerOrBool,
    Float,
    Number,
    Vector,
    IntegerVector,
}

/// One operand or result in a declarative type scheme.
///
/// `Bind` introduces a type variable. Later patterns can require the exact same
/// type, its scalar element, a vector of that element, or merely its vector
/// shape. This small vocabulary is intentionally independent of any opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypePattern {
    Class(TypeClass),
    Exact(crate::Type),
    Bind(u8, TypeClass),
    Same(u8),
    ElementOf(u8),
    VectorOf(u8),
    ShapeOf(u8, TypeClass),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeList {
    Fixed(&'static [TypePattern]),
    /// A checked prefix followed by values from a signature or CFG structure.
    Variadic(&'static [TypePattern]),
    /// Results are supplied by the instruction's function signature.
    Signature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSlot {
    Operand(u8),
    Result(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeRelation {
    /// The destination has more logical bits per lane. Shape constraints are
    /// specified separately in the operand and result patterns.
    Wider { from: TypeSlot, to: TypeSlot },
    /// The destination has fewer logical bits per lane.
    Narrower { from: TypeSlot, to: TypeSlot },
    /// Distinct types with equal whole-value logical sizes, including vscale.
    SameWidthDistinct { lhs: TypeSlot, rhs: TypeSlot },
}

/// Complete, opcode-independent description of value type constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeScheme {
    pub operands: TypeList,
    pub results: TypeList,
    pub relations: &'static [TypeRelation],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResultTypes {
    Inferred(smallvec::SmallVec<[crate::Type; 2]>),
    Explicit,
    Signature,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeSchemeError {
    Arity {
        results: bool,
        expected: usize,
        got: usize,
    },
    Pattern {
        results: bool,
        index: usize,
        expected: TypePattern,
        got: crate::Type,
    },
    Relation(TypeRelation),
}

type TypeBindings = smallvec::SmallVec<[Option<crate::Type>; 4]>;

impl TypeScheme {
    pub const fn fixed(operands: &'static [TypePattern], results: &'static [TypePattern]) -> Self {
        Self {
            operands: TypeList::Fixed(operands),
            results: TypeList::Fixed(results),
            relations: &[],
        }
    }

    pub const fn with_relations(mut self, relations: &'static [TypeRelation]) -> Self {
        self.relations = relations;
        self
    }

    /// Infer result types when the scheme determines them completely.
    /// Operations with caller-selected types report `Explicit`; calls report
    /// `Signature` so the consumer can consult the module.
    /// Operand constraints and inferred results use the same checks as
    /// [`Self::validate`]; invalid inputs are not incomplete type information.
    pub fn infer_results(
        self,
        operands: &[crate::Type],
    ) -> core::result::Result<ResultTypes, TypeSchemeError> {
        let bindings = self.bind_operands(operands)?;
        let TypeList::Fixed(results) = self.results else {
            return Ok(match self.results {
                TypeList::Signature => ResultTypes::Signature,
                TypeList::Variadic(_) => ResultTypes::Explicit,
                TypeList::Fixed(_) => unreachable!(),
            });
        };
        let mut inferred = smallvec::SmallVec::new();
        for pattern in results {
            let Some(ty) = infer_pattern(*pattern, &bindings) else {
                return Ok(ResultTypes::Explicit);
            };
            inferred.push(ty);
        }
        self.validate_results(operands, &inferred, bindings)?;
        Ok(ResultTypes::Inferred(inferred))
    }

    pub fn validate(
        self,
        operands: &[crate::Type],
        results: &[crate::Type],
    ) -> core::result::Result<(), TypeSchemeError> {
        let bindings = self.bind_operands(operands)?;
        self.validate_results(operands, results, bindings)
    }

    fn validate_results(
        self,
        operands: &[crate::Type],
        results: &[crate::Type],
        mut bindings: TypeBindings,
    ) -> core::result::Result<(), TypeSchemeError> {
        validate_list(self.results, results, true, &mut bindings)?;

        for &relation in self.relations {
            let resolve = |slot| match slot {
                TypeSlot::Operand(index) => operands.get(index as usize),
                TypeSlot::Result(index) => results.get(index as usize),
            };
            let resolve = |slot| {
                resolve(slot)
                    .copied()
                    .expect("OpSpec type relation refers to a missing type slot")
            };
            let valid = match relation {
                TypeRelation::Wider { from, to } => resolve(from)
                    .element_bits()
                    .zip(resolve(to).element_bits())
                    .is_some_and(|(from, to)| to > from),
                TypeRelation::Narrower { from, to } => resolve(from)
                    .element_bits()
                    .zip(resolve(to).element_bits())
                    .is_some_and(|(from, to)| to < from),
                TypeRelation::SameWidthDistinct { lhs, rhs } => {
                    let lhs = resolve(lhs);
                    let rhs = resolve(rhs);
                    lhs.bit_size()
                        .zip(rhs.bit_size())
                        .is_some_and(|(lhs_width, rhs_width)| lhs != rhs && lhs_width == rhs_width)
                }
            };
            if !valid {
                return Err(TypeSchemeError::Relation(relation));
            }
        }
        Ok(())
    }

    fn bind_operands(
        self,
        operands: &[crate::Type],
    ) -> core::result::Result<TypeBindings, TypeSchemeError> {
        let mut bindings = smallvec::smallvec![None; self.binding_count()];
        validate_list(self.operands, operands, false, &mut bindings)?;
        Ok(bindings)
    }

    fn binding_count(self) -> usize {
        [self.operands, self.results]
            .into_iter()
            .flat_map(|list| match list {
                TypeList::Fixed(patterns) | TypeList::Variadic(patterns) => patterns,
                TypeList::Signature => &[],
            })
            .filter_map(|pattern| match *pattern {
                TypePattern::Bind(slot, _)
                | TypePattern::Same(slot)
                | TypePattern::ElementOf(slot)
                | TypePattern::VectorOf(slot)
                | TypePattern::ShapeOf(slot, _) => Some(usize::from(slot) + 1),
                TypePattern::Class(_) | TypePattern::Exact(_) => None,
            })
            .max()
            .unwrap_or(0)
    }
}

fn validate_list(
    spec: TypeList,
    types: &[crate::Type],
    results: bool,
    bindings: &mut [Option<crate::Type>],
) -> core::result::Result<(), TypeSchemeError> {
    let (patterns, exact) = match spec {
        TypeList::Fixed(patterns) => (patterns, true),
        TypeList::Variadic(prefix) => (prefix, false),
        TypeList::Signature => return Ok(()),
    };
    if (exact && patterns.len() != types.len()) || (!exact && types.len() < patterns.len()) {
        return Err(TypeSchemeError::Arity {
            results,
            expected: patterns.len(),
            got: types.len(),
        });
    }
    for (index, (&pattern, &ty)) in patterns.iter().zip(types).enumerate() {
        if !matches_pattern(pattern, ty, bindings) {
            return Err(TypeSchemeError::Pattern {
                results,
                index,
                expected: pattern,
                got: ty,
            });
        }
    }
    Ok(())
}

fn matches_pattern(
    pattern: TypePattern,
    ty: crate::Type,
    bindings: &mut [Option<crate::Type>],
) -> bool {
    match pattern {
        TypePattern::Class(class) => class_accepts(class, ty),
        TypePattern::Exact(expected) => ty == expected,
        TypePattern::Bind(variable, class) => {
            if !class_accepts(class, ty) {
                return false;
            }
            let binding = &mut bindings[variable as usize];
            match *binding {
                Some(expected) => ty == expected,
                None => {
                    *binding = Some(ty);
                    true
                }
            }
        }
        TypePattern::Same(variable) => binding(bindings, variable) == Some(ty),
        TypePattern::ElementOf(variable) => binding(bindings, variable)
            .filter(|bound| bound.is_vector())
            .is_some_and(|bound| ty == bound.element_type()),
        TypePattern::VectorOf(variable) => binding(bindings, variable)
            .filter(|bound| bound.is_scalar() && !bound.is_ptr())
            .is_some_and(|bound| ty.is_vector() && ty.element_type() == bound),
        TypePattern::ShapeOf(variable, class) => binding(bindings, variable).is_some_and(|bound| {
            class_accepts(class, ty)
                && if bound.is_vector() {
                    ty.is_vector() && ty.vector_shape() == bound.vector_shape()
                } else {
                    ty.is_scalar()
                }
        }),
    }
}

fn infer_pattern(pattern: TypePattern, bindings: &[Option<crate::Type>]) -> Option<crate::Type> {
    match pattern {
        TypePattern::Exact(ty) => Some(ty),
        TypePattern::Same(variable) | TypePattern::Bind(variable, _) => binding(bindings, variable),
        TypePattern::ElementOf(variable) => binding(bindings, variable)
            .filter(|ty| ty.is_vector())
            .map(crate::Type::element_type),
        TypePattern::Class(_) | TypePattern::VectorOf(_) | TypePattern::ShapeOf(_, _) => None,
    }
}

fn binding(bindings: &[Option<crate::Type>], variable: u8) -> Option<crate::Type> {
    bindings[variable as usize]
}

fn class_accepts(class: TypeClass, ty: crate::Type) -> bool {
    match class {
        TypeClass::Any => ty.is_valid(),
        TypeClass::Scalar => ty.is_scalar() && !ty.is_ptr(),
        TypeClass::ScalarInteger => ty.is_scalar() && ty.is_integer(),
        TypeClass::ScalarIntegerOrPointer => ty.is_scalar() && (ty.is_integer() || ty.is_ptr()),
        TypeClass::ScalarFloat => ty.is_scalar() && ty.is_float(),
        TypeClass::Integer => ty.is_integer(),
        TypeClass::IntegerOrBool => ty.is_integer() || ty == crate::Type::BOOL || ty.is_predicate(),
        TypeClass::Float => ty.is_float(),
        TypeClass::Number => ty.is_integer() || ty.is_float(),
        TypeClass::Vector => ty.is_vector(),
        TypeClass::IntegerVector => ty.is_vector() && ty.is_integer(),
    }
}

/// Reusable type schemes. They are built from generic patterns rather than
/// opcode-specific validator code, so new operations can compose existing
/// constraints or add a new scheme without changing the validator.
pub mod type_schemes {
    include!(concat!(env!("OUT_DIR"), "/type_schemes.rs"));
}

/// Abstract memory regions used by generic scheduling, DCE, and alias queries.
/// The set is deliberately target-independent; a backend may refine it later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRegions(u8);

impl MemoryRegions {
    pub const NONE: Self = Self(0);
    pub const HEAP: Self = Self(1 << 0);
    pub const STACK: Self = Self(1 << 1);
    pub const GLOBAL: Self = Self(1 << 2);
    pub const TABLE: Self = Self(1 << 3);
    pub const EXTERNAL: Self = Self(1 << 4);
    pub const ALL: Self =
        Self(Self::HEAP.0 | Self::STACK.0 | Self::GLOBAL.0 | Self::TABLE.0 | Self::EXTERNAL.0);

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
}

impl core::fmt::Display for MemoryRegions {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_empty() {
            return f.write_str("none");
        }
        let mut separator = "";
        for (region, name) in [
            (Self::HEAP, "heap"),
            (Self::STACK, "stack"),
            (Self::GLOBAL, "global"),
            (Self::TABLE, "table"),
            (Self::EXTERNAL, "external"),
        ] {
            if self.intersects(region) {
                f.write_str(separator)?;
                f.write_str(name)?;
                separator = ",";
            }
        }
        Ok(())
    }
}

/// Memory behavior of an operation. Reads and writes are kept separately so
/// consumers do not have to collapse all memory operations into "impure".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryEffect {
    pub reads: MemoryRegions,
    pub writes: MemoryRegions,
    pub volatile: bool,
    pub atomic: bool,
}

impl MemoryEffect {
    pub const NONE: Self = Self::new(MemoryRegions::NONE, MemoryRegions::NONE);
    pub const HEAP_READ: Self = Self::new(MemoryRegions::HEAP, MemoryRegions::NONE);
    pub const HEAP_WRITE: Self = Self::new(MemoryRegions::NONE, MemoryRegions::HEAP);
    pub const STACK_READ: Self = Self::new(MemoryRegions::STACK, MemoryRegions::NONE);
    pub const STACK_WRITE: Self = Self::new(MemoryRegions::NONE, MemoryRegions::STACK);
    pub const UNKNOWN: Self = Self::new(MemoryRegions::ALL, MemoryRegions::ALL);

    pub const fn new(reads: MemoryRegions, writes: MemoryRegions) -> Self {
        Self {
            reads,
            writes,
            volatile: false,
            atomic: false,
        }
    }

    pub const fn with_volatile(mut self) -> Self {
        self.volatile = true;
        self
    }

    pub const fn with_atomic(mut self) -> Self {
        self.atomic = true;
        self
    }

    pub const fn is_none(self) -> bool {
        self.reads.is_empty() && self.writes.is_empty() && !self.volatile && !self.atomic
    }

    pub const fn may_read(self) -> bool {
        !self.reads.is_empty()
    }

    pub const fn may_write(self) -> bool {
        !self.writes.is_empty()
    }

    pub const fn has_side_effects(self) -> bool {
        self.may_write() || self.volatile || self.atomic
    }

    /// Conservative conflict query suitable for generic motion/scheduling.
    pub const fn conflicts_with(self, other: Self) -> bool {
        self.writes.intersects(other.reads.union(other.writes))
            || other.writes.intersects(self.reads)
            || (self.volatile && !other.is_none())
            || (other.volatile && !self.is_none())
            || (self.atomic && !other.is_none())
            || (other.atomic && !self.is_none())
    }
}

impl core::fmt::Display for MemoryEffect {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_none() {
            return f.write_str("none");
        }
        let mut separator = "";
        if self.may_read() {
            write!(f, "read({})", self.reads)?;
            separator = ", ";
        }
        if self.may_write() {
            write!(f, "{}write({})", separator, self.writes)?;
            separator = ", ";
        }
        if self.volatile {
            write!(f, "{}volatile", separator)?;
            separator = ", ";
        }
        if self.atomic {
            write!(f, "{}atomic", separator)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpTraits(u16);

impl OpTraits {
    pub const TERMINATOR: Self = Self(1 << 0);
    pub const COMMUTATIVE: Self = Self(1 << 1);
    pub const MAY_TRAP: Self = Self(1 << 2);
    pub const ASSOCIATIVE: Self = Self(1 << 3);
    pub const IDEMPOTENT: Self = Self(1 << 4);

    pub const fn empty() -> Self {
        Self(0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl core::fmt::Display for OpTraits {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut separator = "";
        for (property, name) in [
            (Self::TERMINATOR, "terminator"),
            (Self::COMMUTATIVE, "commutative"),
            (Self::ASSOCIATIVE, "associative"),
            (Self::IDEMPOTENT, "idempotent"),
            (Self::MAY_TRAP, "may-trap"),
        ] {
            if self.contains(property) {
                f.write_str(separator)?;
                f.write_str(name)?;
                separator = ", ";
            }
        }
        if separator.is_empty() {
            f.write_str("none")?;
        }
        Ok(())
    }
}

/// Type-independent constants used by canonicalization. Consumers materialize
/// the value according to the operation's concrete operand type.
pub use veloc_semantics::BvConst as AlgebraicConstant;

/// Structural constraints that are not expressible as relationships between
/// SSA value types. The validator interprets these uniformly from `OpSpec`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpConstraint {
    PointerComparison,
    NonZeroScale,
    VectorConstant,
    ShuffleMask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSpec {
    pub mnemonic: &'static str,
    pub format: OpFormat,
    pub type_scheme: &'static TypeScheme,
    pub traits: OpTraits,
    pub memory_effect: MemoryEffect,
    pub constraints: &'static [OpConstraint],
    pub identity: Option<AlgebraicConstant>,
    pub absorbing: Option<AlgebraicConstant>,
    /// Scalar bitvector program, applied independently to vector lanes.
    /// This does not describe predication, memory, or machine state effects.
    pub semantics: Option<crate::semantics::Program>,
}

impl OpSpec {
    pub const fn is_pure(self) -> bool {
        self.memory_effect.is_none() && !self.is_terminator() && !self.may_trap()
    }

    pub const fn is_terminator(self) -> bool {
        self.traits.contains(OpTraits::TERMINATOR)
    }

    pub const fn has_side_effects(self) -> bool {
        self.is_terminator() || self.may_trap() || self.memory_effect.has_side_effects()
    }

    pub const fn is_commutative(self) -> bool {
        self.traits.contains(OpTraits::COMMUTATIVE)
    }

    pub const fn is_associative(self) -> bool {
        self.traits.contains(OpTraits::ASSOCIATIVE)
    }

    pub const fn is_idempotent(self) -> bool {
        self.traits.contains(OpTraits::IDEMPOTENT)
    }

    pub const fn may_trap(self) -> bool {
        self.traits.contains(OpTraits::MAY_TRAP)
    }
}

/// Emit a reference table from exactly the metadata used by the IR itself.
/// This is intentionally descriptive documentation, not a stable wire format.
pub fn write_opcode_markdown(output: &mut dyn core::fmt::Write) -> core::fmt::Result {
    writeln!(
        output,
        "| Mnemonic | Format | Operands | Results | Memory | Properties | Constraints |"
    )?;
    writeln!(output, "|---|---|---|---|---|---|---|")?;
    for &opcode in crate::Opcode::ALL {
        let spec = opcode.spec();
        writeln!(
            output,
            "| `{}` | `{:?}` | `{:?}` | `{:?}` | {} | {} | `{:?}` |",
            spec.mnemonic,
            spec.format,
            spec.type_scheme.operands,
            spec.type_scheme.results,
            spec.memory_effect,
            spec.traits,
            spec.constraints,
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::ResultTypes;
    use super::{MemoryEffect, OpConstraint, OpFormat, TypeList};
    use crate::{FloatCC, IntCC, Opcode, Type};

    #[test]
    fn opcode_mnemonics_are_unique_and_round_trip() {
        for (index, &opcode) in Opcode::ALL.iter().enumerate() {
            let spec = opcode.spec();
            assert_eq!(Opcode::from_mnemonic(spec.mnemonic), Some(opcode));
            assert!(
                Opcode::ALL[..index]
                    .iter()
                    .all(|earlier| earlier.spec().mnemonic != spec.mnemonic),
                "duplicate opcode mnemonic: {}",
                spec.mnemonic
            );
        }
    }

    #[test]
    fn control_and_arithmetic_traits_are_declarative() {
        assert!(Opcode::Return.spec().is_terminator());
        assert!(Opcode::Store.spec().has_side_effects());
        assert!(Opcode::IAdd.spec().is_commutative());
        assert!(Opcode::IAdd.spec().is_pure());
        assert_eq!(
            Opcode::Icmp
                .spec()
                .type_scheme
                .infer_results(&[Type::I32, Type::I32])
                .unwrap(),
            ResultTypes::Inferred(smallvec::smallvec![Type::BOOL])
        );
    }

    #[test]
    fn variadic_operands_validate_the_known_prefix() {
        use super::{TypeClass, TypePattern, TypeScheme, TypeSchemeError};

        let scheme = TypeScheme {
            operands: TypeList::Variadic(&[TypePattern::Bind(0, TypeClass::Integer)]),
            results: TypeList::Fixed(&[TypePattern::Same(0)]),
            relations: &[],
        };
        assert_eq!(
            scheme.infer_results(&[Type::I32, Type::F64]).unwrap(),
            ResultTypes::Inferred(smallvec::smallvec![Type::I32])
        );
        assert!(scheme.validate(&[Type::I32], &[Type::I32]).is_ok());
        assert!(matches!(
            scheme.validate(&[], &[Type::I32]),
            Err(TypeSchemeError::Arity { .. })
        ));
        assert!(matches!(
            scheme.validate(&[Type::F64], &[Type::F64]),
            Err(TypeSchemeError::Pattern { .. })
        ));
        assert!(scheme.validate(&[Type::I32], &[Type::I64]).is_err());

        let branch = Opcode::Br.spec().type_scheme;
        assert!(branch.validate(&[Type::BOOL, Type::I64], &[]).is_ok());
        assert!(branch.validate(&[Type::I32, Type::I64], &[]).is_err());
        assert!(branch.validate(&[], &[]).is_err());
        let call = Opcode::CallIndirect.spec().type_scheme;
        assert!(call.validate(&[Type::PTR, Type::I32], &[Type::I64]).is_ok());
        assert!(
            call.validate(&[Type::I64, Type::I32], &[Type::I64])
                .is_err()
        );
        assert!(call.validate(&[], &[]).is_err());
    }

    #[test]
    fn composed_negation_contract_drives_execution_and_smt_export() {
        use crate::semantics::{BvOp, Expr, Function, Sort, Value, Width, equivalence_query};

        let program = Opcode::INeg.spec().semantics.unwrap();
        assert_eq!(program.primitive(), None);
        assert_eq!(program.arity(), 1);
        for bits in [8, 16, 32, 64, 128] {
            let width = Width::new(bits).unwrap();
            let function = program.instantiate(width).unwrap();
            let primitive = Function::new(
                alloc::vec![Sort::Bv(width)],
                Expr::apply(BvOp::Neg, &[Expr::input(0, Sort::Bv(width))]).unwrap(),
            )
            .unwrap();
            for input in [0, 1, width.mask(), 1u128 << (bits - 1)] {
                let expected = BvOp::Neg.eval(bits, &[input]).unwrap();
                assert_eq!(program.eval(width, &[input]).unwrap(), expected);
                assert_eq!(
                    function.eval(&[Value::Bv(input)]).unwrap(),
                    Value::Bv(expected)
                );
            }
            // Emission is tested here; proving the query unsatisfiable remains
            // the optional solver consumer's responsibility.
            let query = equivalence_query(&function, &primitive).unwrap();
            assert!(query.contains("bvsub"));
            assert!(query.contains("bvneg"));
        }
    }

    #[test]
    fn opcode_spec_invariants_hold() {
        for &opcode in Opcode::ALL {
            let spec = opcode.spec();
            if let (Some(format_arity), TypeList::Fixed(operands)) =
                (spec.format.fixed_value_arity(), spec.type_scheme.operands)
            {
                assert_eq!(
                    format_arity,
                    operands.len(),
                    "{} format and type scheme disagree",
                    spec.mnemonic
                );
            }

            for constraint in spec.constraints {
                let compatible = match constraint {
                    OpConstraint::PointerComparison => spec.format == OpFormat::IntCompare,
                    OpConstraint::NonZeroScale => spec.format == OpFormat::PtrIndex,
                    OpConstraint::VectorConstant => spec.format == OpFormat::Vconst,
                    OpConstraint::ShuffleMask => spec.format == OpFormat::Shuffle,
                };
                assert!(compatible, "{} has incompatible constraint", spec.mnemonic);
            }

            if spec.identity.is_some() || spec.absorbing.is_some() {
                assert!(
                    spec.is_commutative(),
                    "{} algebraic constant",
                    spec.mnemonic
                );
                assert!(
                    spec.is_associative(),
                    "{} algebraic constant",
                    spec.mnemonic
                );
            }
            if spec.is_idempotent() {
                assert!(spec.is_commutative());
                assert!(spec.is_associative());
            }
        }
    }

    #[test]
    fn memory_effects_preserve_region_information() {
        assert!(MemoryEffect::HEAP_READ.conflicts_with(MemoryEffect::HEAP_WRITE));
        assert!(!MemoryEffect::STACK_READ.conflicts_with(MemoryEffect::HEAP_WRITE));
        assert!(MemoryEffect::HEAP_READ.with_volatile().has_side_effects());
        assert!(MemoryEffect::UNKNOWN.conflicts_with(MemoryEffect::STACK_READ));
    }

    #[test]
    fn condition_code_transforms_are_involutions() {
        let integers = [
            IntCC::Eq,
            IntCC::Ne,
            IntCC::LtS,
            IntCC::LtU,
            IntCC::GtS,
            IntCC::GtU,
            IntCC::LeS,
            IntCC::LeU,
            IntCC::GeS,
            IntCC::GeU,
        ];
        for cc in integers {
            assert_eq!(cc.swap().swap(), cc);
            assert_eq!(cc.complement().complement(), cc);
        }

        let floats = [
            FloatCC::Eq,
            FloatCC::Ne,
            FloatCC::Lt,
            FloatCC::Gt,
            FloatCC::Le,
            FloatCC::Ge,
        ];
        for cc in floats {
            assert_eq!(cc.swap().swap(), cc);
            assert_eq!(cc.complement_ordered().complement_ordered(), cc);
        }
        assert_eq!(FloatCC::Lt.complement(), None);
    }

    #[test]
    fn generated_markdown_covers_every_opcode() {
        let mut output = alloc::string::String::new();
        super::write_opcode_markdown(&mut output).unwrap();
        for &opcode in Opcode::ALL {
            assert!(output.contains(opcode.spec().mnemonic));
        }
        assert!(output.contains("| Mnemonic | Format |"));
    }
}
