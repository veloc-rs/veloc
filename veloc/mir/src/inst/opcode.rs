//! Instruction kinds: opcodes, metadata, memory flags and type contracts.
//!
//! Layouts, type checks and operation tables are compiled from `defs/*.ops`.
//! Their shared runtime support lives alongside the generated definitions.

use super::MemoryEffect;

include!(concat!(env!("OUT_DIR"), "/opcodes.rs"));

impl MemFlags {
    pub const fn new() -> Self {
        Self::empty()
    }

    pub fn with_alignment(self, align: u32) -> Self {
        assert!(align.is_power_of_two(), "Alignment must be a power of 2");
        let log2 = align.trailing_zeros().min(Self::ALIGNMENT_LOG2_MAX as u32) as u16;
        self.with_alignment_log2(log2)
    }

    pub fn alignment(&self) -> u32 {
        1 << self.alignment_log2()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeError {
    Arity {
        results: bool,
        expected: usize,
        got: usize,
    },
    Pattern {
        results: bool,
        index: usize,
        expected: &'static str,
        got: crate::Type,
    },
    /// Static diagnostic emitted with the failed type-relation check.
    Relation(&'static str),
}

/// Shared executable rules generated from the definitions.
mod type_rules {
    include!(concat!(env!("OUT_DIR"), "/type_rules.rs"));
}

impl MemoryEffect {
    pub const fn is_unknown(self) -> bool {
        matches!(self, Self::Unknown)
    }

    pub const fn is_none(self) -> bool {
        matches!(self, Self::Known(effects) if effects.is_empty())
    }

    const fn may(self, effect: MemoryEffects) -> bool {
        match self {
            Self::Known(effects) => effects.intersects(effect),
            Self::Unknown => true,
        }
    }

    pub const fn may_read(self) -> bool {
        self.may(MemoryEffects::READ)
    }
    pub const fn may_write(self) -> bool {
        self.may(MemoryEffects::WRITE)
    }
    pub const fn may_allocate(self) -> bool {
        self.may(MemoryEffects::ALLOCATE)
    }
    pub const fn may_free(self) -> bool {
        self.may(MemoryEffects::FREE)
    }

    /// Unused abstract objects can be removed; new or unmodeled effects cannot.
    pub const fn can_erase(self) -> bool {
        matches!(self, Self::Known(effects)
            if MemoryEffects::READ.union(MemoryEffects::ALLOCATE).contains(effects))
    }

    pub const fn has_side_effects(self) -> bool {
        !matches!(self, Self::Known(effects) if MemoryEffects::READ.contains(effects))
    }

    /// Only memory interference; movement also requires control/trap/ordering checks.
    pub const fn conflicts_with(self, other: Self) -> bool {
        if self.is_none() || other.is_none() {
            return false;
        }
        let supported = MemoryEffects::READ
            .union(MemoryEffects::WRITE)
            .union(MemoryEffects::ALLOCATE)
            .union(MemoryEffects::FREE);
        match (self, other) {
            (Self::Known(lhs), Self::Known(rhs))
                if supported.contains(lhs) && supported.contains(rhs) =>
            {
                self.may_free()
                    || other.may_free()
                    || (self.may_allocate() && other.may_allocate())
                    || (self.may_write() && (other.may_read() || other.may_write()))
                    || (other.may_write() && self.may_read())
            }
            _ => true,
        }
    }
}

impl core::fmt::Display for MemoryEffect {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Known(effects) => effects.fmt(f),
            Self::Unknown => f.write_str("unknown"),
        }
    }
}

impl OpSpec {
    pub const fn is_pure(&self) -> bool {
        self.memory_effect().is_none() && !self.is_terminator() && !self.may_trap()
    }

    pub const fn is_terminator(&self) -> bool {
        self.traits().contains(OpTraits::TERMINATOR)
    }

    pub const fn has_side_effects(&self) -> bool {
        self.is_terminator() || self.may_trap() || self.memory_effect().has_side_effects()
    }

    pub const fn is_commutative(&self) -> bool {
        self.traits().contains(OpTraits::COMMUTATIVE)
    }

    pub const fn is_associative(&self) -> bool {
        self.traits().contains(OpTraits::ASSOCIATIVE)
    }

    pub const fn is_idempotent(&self) -> bool {
        self.traits().contains(OpTraits::IDEMPOTENT)
    }

    pub const fn may_trap(&self) -> bool {
        self.traits().contains(OpTraits::MAY_TRAP)
    }
}

#[cfg(test)]
mod tests {
    use super::{MemoryEffect, MemoryEffects};
    use crate::{FloatCC, IntCC, Opcode, Type};

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
    fn opcode_spec_invariants_hold() {
        for &opcode in Opcode::ALL {
            let spec = opcode.spec();
            if spec.is_idempotent() {
                assert!(spec.is_commutative());
                assert!(spec.is_associative());
            }
        }
    }

    #[test]
    fn unknown_effects_remain_conservative() {
        assert!(
            MemoryEffect::Known(MemoryEffects::READ)
                .conflicts_with(MemoryEffect::Known(MemoryEffects::WRITE))
        );
        assert!(MemoryEffect::Unknown.has_side_effects());
        assert!(MemoryEffect::Unknown.conflicts_with(MemoryEffect::Known(MemoryEffects::READ)));
        assert!(MemoryEffect::Unknown.is_unknown());
        assert!(MemoryEffect::Unknown.may_allocate());
        assert!(MemoryEffect::Unknown.may_free());
        assert!(MemoryEffect::Unknown.may_read());
        assert!(MemoryEffect::Unknown.may_write());
        assert!(!MemoryEffect::Unknown.can_erase());
        assert!(!MemoryEffect::Unknown.is_none());
        assert!(
            !MemoryEffect::Known(MemoryEffects::READ)
                .conflicts_with(MemoryEffect::Known(MemoryEffects::READ))
        );
        assert_eq!(MemoryEffect::Unknown.to_string(), "unknown");
        assert_eq!(MemoryEffect::Known(MemoryEffects::READ).to_string(), "read");
        assert_eq!(
            MemoryEffect::Known(MemoryEffects::WRITE).to_string(),
            "write"
        );
    }

    #[test]
    fn inline_type_sets_preserve_membership_and_shared_bindings() {
        for raw in 0..=u16::MAX {
            let Some(ty) = Type::from_raw(raw) else {
                continue;
            };
            for op in [Opcode::IAnd, Opcode::IOr, Opcode::IXor] {
                assert_eq!(
                    op.validate_types(&[ty, ty], &[ty]).is_ok(),
                    ty.is_integer() || ty == Type::BOOL || ty.is_predicate(),
                    "{op:?}: {ty}"
                );
            }
            assert_eq!(
                Opcode::Icmp
                    .validate_types(&[ty, ty], &[Type::BOOL])
                    .is_ok(),
                ty.is_scalar() && (ty.is_integer() || ty.is_ptr()),
                "{ty}"
            );
            assert_eq!(
                Opcode::Gather
                    .validate_types(&[Type::PTR, ty], &[ty])
                    .is_ok(),
                ty.is_integer() && ty.is_vector(),
                "{ty}"
            );
        }
        for op in [Opcode::IAnd, Opcode::Icmp, Opcode::Gather] {
            assert!(
                op.validate_types(&[Type::INVALID, Type::INVALID], &[Type::INVALID])
                    .is_err()
            );
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
        use super::OpTraits;
        use alloc::string::ToString;

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
}
