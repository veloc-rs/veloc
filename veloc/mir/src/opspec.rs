//! Runtime contracts for MIR operations.
//!
//! Layouts, type schemes and operation tables are compiled from `defs/*.ops`.
//! This module implements their shared type/effect machinery; it does not
//! contain individual operation definitions.

include!(concat!(env!("OUT_DIR"), "/formats.rs"));

include!(concat!(env!("OUT_DIR"), "/builtins.rs"));

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

/// Read-only descriptive metadata. Executable checks are generated on Opcode.
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

fn same_shape(bound: crate::Type, ty: crate::Type) -> bool {
    if let Some(bound) = bound.as_vector() {
        ty.as_vector()
            .is_some_and(|vector| vector.shape() == bound.shape())
    } else {
        ty.as_scalar().is_some()
    }
}

/// Shared metadata and executable rules generated from the definitions.
pub(crate) mod type_schemes {
    include!(concat!(env!("OUT_DIR"), "/type_schemes.rs"));
}

#[cfg(test)]
#[allow(dead_code, unused_imports)]
mod test_type_schemes {
    include!(concat!(env!("OUT_DIR"), "/test_type_schemes.rs"));
}

/// Abstract memory regions used by generic scheduling, DCE, and alias queries.
/// The set is deliberately target-independent; a backend may refine it later.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryRegions(u8);

impl MemoryRegions {
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
        for &(region, name) in Self::NAMES {
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
        for &(property, name) in Self::NAMES {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpSpec {
    pub mnemonic: &'static str,
    pub format: OpFormat,
    pub type_scheme: &'static TypeScheme,
    pub traits: OpTraits,
    pub memory_effect: MemoryEffect,
    /// Descriptive predicates/diagnostics; executable checks are generated from definitions.
    pub constraints: &'static [&'static str],
    pub identity: Option<AlgebraicConstant>,
    pub absorbing: Option<AlgebraicConstant>,
    /// Scalar bitvector program, applied independently to vector lanes.
    /// This does not describe predication, memory, or machine state effects.
    pub semantics: Option<crate::semantics::Program<'static>>,
}

impl OpSpec {
    pub const fn is_pure(&self) -> bool {
        self.memory_effect.is_none() && !self.is_terminator() && !self.may_trap()
    }

    pub const fn is_terminator(&self) -> bool {
        self.traits.contains(OpTraits::TERMINATOR)
    }

    pub const fn has_side_effects(&self) -> bool {
        self.is_terminator() || self.may_trap() || self.memory_effect.has_side_effects()
    }

    pub const fn is_commutative(&self) -> bool {
        self.traits.contains(OpTraits::COMMUTATIVE)
    }

    pub const fn is_associative(&self) -> bool {
        self.traits.contains(OpTraits::ASSOCIATIVE)
    }

    pub const fn is_idempotent(&self) -> bool {
        self.traits.contains(OpTraits::IDEMPOTENT)
    }

    pub const fn may_trap(&self) -> bool {
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
    use super::{MemoryEffect, TypeList};
    use crate::{FloatCC, IntCC, Opcode, Type};

    #[test]
    fn generated_contracts_cover_many_variables_and_result_only_bindings() {
        use super::test_type_schemes::{
            infer_0 as infer_many, infer_1 as infer_output, validate_0 as validate_many,
            validate_1 as validate_output,
        };
        let operands = [Type::I8, Type::I16, Type::I32, Type::I64, Type::F32];
        assert_eq!(
            infer_many(&operands),
            Ok(ResultTypes::Inferred(smallvec::smallvec![
                Type::F32,
                Type::I8
            ]))
        );
        assert!(validate_many(&operands, &[Type::F32, Type::I8]).is_ok());
        assert!(validate_many(&operands, &[Type::I8, Type::F32]).is_err());
        assert!(infer_many(&operands[..4]).is_err());
        assert_eq!(infer_output(&[]), Ok(ResultTypes::Explicit));
        assert!(validate_output(&[], &[Type::I32]).is_ok());
        assert!(validate_output(&[], &[Type::F32]).is_err());
    }

    #[test]
    fn generated_inference_checks_relations_on_known_results() {
        use super::test_type_schemes::{infer_2, validate_2};
        let expected = super::TypeSchemeError::Relation(super::TypeRelation::Wider {
            from: super::TypeSlot::Operand(0),
            to: super::TypeSlot::Result(0),
        });
        assert_eq!(infer_2(&[Type::I32]), Err(expected));
        assert_eq!(validate_2(&[Type::I32], &[Type::I8]), Err(expected));
        assert_eq!(
            infer_2(&[Type::BOOL]),
            Ok(ResultTypes::Inferred(smallvec::smallvec![Type::I8]))
        );
    }

    #[test]
    fn opcode_mnemonics_are_unique_and_round_trip() {
        for (index, &opcode) in Opcode::ALL.iter().enumerate() {
            let spec: &'static super::OpSpec = opcode.spec();
            assert!(core::ptr::eq(spec, opcode.spec()));
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
        const ADD: &super::OpSpec = Opcode::IAdd.spec();
        const _: () = assert!(ADD.is_pure() && ADD.is_commutative());
        assert!(Opcode::Return.spec().is_terminator());
        assert!(Opcode::Store.spec().has_side_effects());
        assert!(Opcode::IAdd.spec().is_commutative());
        assert!(Opcode::IAdd.spec().is_pure());
        assert_eq!(
            Opcode::Icmp
                .infer_result_types(&[Type::I32, Type::I32])
                .unwrap(),
            ResultTypes::Inferred(smallvec::smallvec![Type::BOOL])
        );
    }

    #[test]
    fn variadic_operands_validate_the_known_prefix() {
        let branch = Opcode::Br;
        assert!(branch.validate_types(&[Type::BOOL, Type::I64], &[]).is_ok());
        assert!(branch.validate_types(&[Type::I32, Type::I64], &[]).is_err());
        assert!(branch.validate_types(&[], &[]).is_err());
        let call = Opcode::CallIndirect;
        assert!(
            call.validate_types(&[Type::PTR, Type::I32], &[Type::I64])
                .is_ok()
        );
        assert!(
            call.validate_types(&[Type::I64, Type::I32], &[Type::I64])
                .is_err()
        );
        assert!(call.validate_types(&[], &[]).is_err());
    }

    #[test]
    fn composed_negation_contract_drives_execution_and_smt_export() {
        use crate::semantics::{BvOp, Expr, Function, Sort, Value, Width, equivalence_query};

        let program = Opcode::INeg.spec().semantics.unwrap();
        assert_eq!(program.primitive(), None);
        assert_eq!(program.arity(), 1);
        for bits in [8, 16, 32, 64, 128] {
            let width = Width::new(bits).unwrap();
            let function = program
                .instantiate(&[Sort::Bv(width)], &[Sort::Bv(width)], &[])
                .unwrap();
            let primitive = Function::new(
                alloc::vec![Sort::Bv(width)],
                Expr::apply(BvOp::Neg, &[Expr::input(0, Sort::Bv(width))]).unwrap(),
            )
            .unwrap();
            for input in [0, 1, width.mask(), 1u128 << (bits - 1)] {
                let expected = BvOp::Neg.eval(bits, &[input]).unwrap();
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
        assert!(core::ptr::eq(
            Opcode::IAdd.spec().type_scheme,
            Opcode::ISub.spec().type_scheme,
        ));
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
        assert!(MemoryEffect::GLOBAL_READ.conflicts_with(MemoryEffect::GLOBAL_WRITE));
        assert!(MemoryEffect::TABLE_READ.conflicts_with(MemoryEffect::TABLE_WRITE));
        assert!(!MemoryEffect::GLOBAL_READ.conflicts_with(MemoryEffect::TABLE_WRITE));
        assert!(MemoryEffect::UNKNOWN.conflicts_with(MemoryEffect::GLOBAL_READ));
        assert!(MemoryEffect::UNKNOWN.conflicts_with(MemoryEffect::TABLE_READ));
        assert_eq!(
            MemoryEffect::GLOBAL_READ.reads,
            super::MemoryRegions::GLOBAL
        );
        assert_eq!(
            MemoryEffect::TABLE_WRITE.writes,
            super::MemoryRegions::TABLE
        );
    }

    #[test]
    fn inline_type_sets_preserve_membership_and_shared_bindings() {
        use super::{TypeList, TypePattern};

        let class = |opcode: Opcode, index: usize| {
            let TypeList::Fixed(operands) = opcode.spec().type_scheme.operands else {
                panic!("expected fixed operands");
            };
            let TypePattern::Bind(_, class) = operands[index] else {
                panic!("expected a type binding");
            };
            class
        };
        let bits = class(Opcode::IAnd, 0);
        let comparable = class(Opcode::Icmp, 0);
        let indices = class(Opcode::Gather, 1);
        assert_eq!(bits, class(Opcode::IOr, 0));
        assert_eq!(bits, class(Opcode::IXor, 0));
        assert_eq!(bits, class(Opcode::ExtendU, 0));
        for raw in 0..=u16::MAX {
            let Some(ty) = Type::from_raw(raw) else {
                continue;
            };
            assert_eq!(
                bits.accepts(ty),
                ty.is_integer() || ty == Type::BOOL || ty.is_predicate(),
                "{ty}"
            );
            assert_eq!(
                comparable.accepts(ty),
                ty.is_scalar() && (ty.is_integer() || ty.is_ptr()),
                "{ty}"
            );
            assert_eq!(
                indices.accepts(ty),
                ty.is_integer() && ty.is_vector(),
                "{ty}"
            );
        }
        for class in [bits, comparable, indices] {
            assert!(!class.accepts(Type::INVALID));
        }
        let scheme = Opcode::IAnd;
        assert!(
            scheme
                .validate_types(&[Type::I32, Type::I32], &[Type::I32])
                .is_ok()
        );
        assert!(
            scheme
                .validate_types(&[Type::BOOL, Type::BOOL], &[Type::BOOL])
                .is_ok()
        );
        assert!(
            scheme
                .validate_types(&[Type::I32, Type::I64], &[Type::I32])
                .is_err()
        );
        assert!(
            scheme
                .validate_types(&[Type::I32, Type::I32], &[Type::BOOL])
                .is_err()
        );
    }

    #[test]
    fn generated_classes_preserve_membership_for_every_valid_encoding() {
        use super::TypeClass;

        for raw in 0..=u16::MAX {
            let Some(ty) = Type::from_raw(raw) else {
                continue;
            };
            for (class, expected) in [
                (TypeClass::Any, true),
                (TypeClass::Scalar, ty.is_scalar() && !ty.is_ptr()),
                (TypeClass::ScalarInteger, ty.is_scalar() && ty.is_integer()),
                (TypeClass::ScalarFloat, ty.is_scalar() && ty.is_float()),
                (TypeClass::Integer, ty.is_integer()),
                (TypeClass::Float, ty.is_float()),
                (TypeClass::Number, ty.is_integer() || ty.is_float()),
                (TypeClass::Vector, ty.is_vector()),
            ] {
                assert_eq!(class.accepts(ty), expected, "{class:?}: {ty}");
                assert!(!class.accepts(Type::INVALID), "{class:?}");
            }
        }
    }

    #[test]
    fn generated_flag_names_preserve_display_order() {
        use super::{MemoryRegions, OpTraits};
        use alloc::string::ToString;

        assert_eq!(
            MemoryRegions::ALL.to_string(),
            "heap,stack,global,table,external"
        );
        assert_eq!(MemoryRegions::NONE.to_string(), "none");
        let traits = OpTraits::TERMINATOR
            .union(OpTraits::COMMUTATIVE)
            .union(OpTraits::MAY_TRAP)
            .union(OpTraits::ASSOCIATIVE)
            .union(OpTraits::IDEMPOTENT);
        assert_eq!(
            traits.to_string(),
            "terminator, commutative, associative, idempotent, may-trap"
        );
        assert_eq!(OpTraits::empty().to_string(), "none");
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
            assert_eq!(IntCC::from_mnemonic(cc.mnemonic()), Some(cc));
            assert_eq!(cc.to_string(), cc.mnemonic());
            assert_eq!(
                cc.is_unsigned(),
                matches!(cc, IntCC::LtU | IntCC::GtU | IntCC::LeU | IntCC::GeU)
            );
            assert_eq!(cc.swap().is_unsigned(), cc.is_unsigned());
            assert_eq!(cc.complement().is_unsigned(), cc.is_unsigned());
        }
        assert_eq!(IntCC::from_mnemonic("unknown"), None);

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
            assert_eq!(FloatCC::from_mnemonic(cc.mnemonic()), Some(cc));
            assert_eq!(cc.to_string(), cc.mnemonic());
        }
        assert_eq!(FloatCC::from_mnemonic("unknown"), None);
        assert_eq!(FloatCC::Lt.complement(), None);
    }

    #[test]
    fn float_condition_transforms_preserve_ieee_comparisons() {
        fn eval(cc: FloatCC, lhs: f64, rhs: f64) -> bool {
            match cc {
                FloatCC::Eq => lhs == rhs,
                FloatCC::Ne => lhs != rhs,
                FloatCC::Lt => lhs < rhs,
                FloatCC::Gt => lhs > rhs,
                FloatCC::Le => lhs <= rhs,
                FloatCC::Ge => lhs >= rhs,
            }
        }
        let values = [
            f64::NEG_INFINITY,
            -1.0,
            -0.0,
            0.0,
            1.0,
            f64::INFINITY,
            f64::NAN,
        ];
        for cc in [
            FloatCC::Eq,
            FloatCC::Ne,
            FloatCC::Lt,
            FloatCC::Gt,
            FloatCC::Le,
            FloatCC::Ge,
        ] {
            assert_eq!(
                cc.complement().is_some(),
                matches!(cc, FloatCC::Eq | FloatCC::Ne)
            );
            for lhs in values {
                for rhs in values {
                    assert_eq!(eval(cc, lhs, rhs), eval(cc.swap(), rhs, lhs));
                    if let Some(inverse) = cc.complement() {
                        assert_ne!(eval(cc, lhs, rhs), eval(inverse, lhs, rhs));
                    }
                    if !lhs.is_nan() && !rhs.is_nan() {
                        assert_ne!(eval(cc, lhs, rhs), eval(cc.complement_ordered(), lhs, rhs));
                    }
                }
            }
        }
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
