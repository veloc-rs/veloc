//! Declarative semantic metadata for core IR operations.
//!
//! `OpSpec` is intentionally compile-time and closed. It centralizes facts used
//! by generic IR infrastructure without turning the core IR into a dynamic
//! dialect system.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpFormat {
    Unary,
    Binary,
    Ternary,
    Iconst,
    Fconst,
    Bconst,
    Vconst,
    Load,
    Store,
    StackLoad,
    StackStore,
    StackAddr,
    PtrOffset,
    PtrIndex,
    IntToPtr,
    PtrToInt,
    Call,
    CallIndirect,
    CallIntrinsic,
    Jump,
    Br,
    BrTable,
    Return,
    IntCompare,
    FloatCompare,
    VectorLoadStrided,
    VectorStoreStrided,
    VectorGather,
    VectorScatter,
    Shuffle,
    Unreachable,
    Nop,
}

impl OpFormat {
    /// Number of core SSA operands when it is determined by the physical
    /// format. `None` means the operands are carried by a signature, block
    /// destination, or another variadic container.
    pub const fn fixed_value_arity(self) -> Option<usize> {
        match self {
            Self::Iconst
            | Self::Fconst
            | Self::Bconst
            | Self::Vconst
            | Self::StackLoad
            | Self::StackAddr
            | Self::Unreachable
            | Self::Nop => Some(0),
            Self::Unary
            | Self::Load
            | Self::StackStore
            | Self::IntToPtr
            | Self::PtrToInt
            | Self::PtrOffset => Some(1),
            Self::Binary
            | Self::PtrIndex
            | Self::IntCompare
            | Self::FloatCompare
            | Self::VectorLoadStrided
            | Self::VectorGather
            | Self::Shuffle => Some(2),
            Self::Ternary | Self::VectorStoreStrided | Self::VectorScatter => Some(3),
            Self::Store => Some(2),
            Self::Call
            | Self::CallIndirect
            | Self::CallIntrinsic
            | Self::Jump
            | Self::Br
            | Self::BrTable
            | Self::Return => None,
        }
    }
}

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
    /// Types are supplied by external structure such as a signature or block.
    External,
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
    Wider { from: TypeSlot, to: TypeSlot },
    Narrower { from: TypeSlot, to: TypeSlot },
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
    pub fn infer_results(self, operands: &[crate::Type]) -> ResultTypes {
        let TypeList::Fixed(results) = self.results else {
            return match self.results {
                TypeList::Signature => ResultTypes::Signature,
                TypeList::External => ResultTypes::Explicit,
                TypeList::Fixed(_) => unreachable!(),
            };
        };
        let bindings = self.bindings(operands);
        let mut inferred = smallvec::SmallVec::new();
        for pattern in results {
            let Some(ty) = infer_pattern(*pattern, &bindings) else {
                return ResultTypes::Explicit;
            };
            inferred.push(ty);
        }
        ResultTypes::Inferred(inferred)
    }

    pub fn validate(
        self,
        operands: &[crate::Type],
        results: &[crate::Type],
    ) -> core::result::Result<(), TypeSchemeError> {
        let mut bindings = [None; 4];
        validate_list(self.operands, operands, false, &mut bindings)?;
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
                TypeRelation::Wider { from, to } => logical_width(resolve(from))
                    .zip(logical_width(resolve(to)))
                    .is_some_and(|(from, to)| to > from),
                TypeRelation::Narrower { from, to } => logical_width(resolve(from))
                    .zip(logical_width(resolve(to)))
                    .is_some_and(|(from, to)| to < from),
                TypeRelation::SameWidthDistinct { lhs, rhs } => {
                    let lhs = resolve(lhs);
                    let rhs = resolve(rhs);
                    logical_width(lhs)
                        .zip(logical_width(rhs))
                        .map(|widths| (lhs, rhs, widths))
                        .is_some_and(|(lhs, rhs, (lhs_width, rhs_width))| {
                            lhs != rhs && lhs_width == rhs_width
                        })
                }
            };
            if !valid {
                return Err(TypeSchemeError::Relation(relation));
            }
        }
        Ok(())
    }

    fn bindings(self, operands: &[crate::Type]) -> [Option<crate::Type>; 4] {
        let mut bindings = [None; 4];
        if let TypeList::Fixed(patterns) = self.operands {
            for (&pattern, &ty) in patterns.iter().zip(operands) {
                if let TypePattern::Bind(variable, _) = pattern {
                    *bindings
                        .get_mut(variable as usize)
                        .expect("OpSpec type variable exceeds binding capacity") = Some(ty);
                }
            }
        }
        bindings
    }
}

fn validate_list(
    spec: TypeList,
    types: &[crate::Type],
    results: bool,
    bindings: &mut [Option<crate::Type>; 4],
) -> core::result::Result<(), TypeSchemeError> {
    let TypeList::Fixed(patterns) = spec else {
        return Ok(());
    };
    if patterns.len() != types.len() {
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
    bindings: &mut [Option<crate::Type>; 4],
) -> bool {
    match pattern {
        TypePattern::Class(class) => class_accepts(class, ty),
        TypePattern::Exact(expected) => ty == expected,
        TypePattern::Bind(variable, class) => {
            if !class_accepts(class, ty) {
                return false;
            }
            let binding = bindings
                .get_mut(variable as usize)
                .expect("OpSpec type variable exceeds binding capacity");
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

fn infer_pattern(pattern: TypePattern, bindings: &[Option<crate::Type>; 4]) -> Option<crate::Type> {
    match pattern {
        TypePattern::Exact(ty) => Some(ty),
        TypePattern::Same(variable) => binding(bindings, variable),
        TypePattern::ElementOf(variable) => binding(bindings, variable)
            .filter(|ty| ty.is_vector())
            .map(crate::Type::element_type),
        TypePattern::Class(_)
        | TypePattern::Bind(_, _)
        | TypePattern::VectorOf(_)
        | TypePattern::ShapeOf(_, _) => None,
    }
}

fn binding(bindings: &[Option<crate::Type>; 4], variable: u8) -> Option<crate::Type> {
    *bindings
        .get(variable as usize)
        .expect("OpSpec type variable exceeds binding capacity")
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

fn logical_width(ty: crate::Type) -> Option<u32> {
    if ty == crate::Type::BOOL {
        Some(1)
    } else {
        ty.min_bit_width()
    }
}

/// Reusable type schemes. They are built from generic patterns rather than
/// opcode-specific validator code, so new operations can compose existing
/// constraints or add a new scheme without changing the validator.
pub mod type_schemes {
    use super::{TypeClass as C, TypeList as L, TypePattern as P, TypeRelation as R};
    use super::{TypeScheme as S, TypeSlot as Slot};
    use crate::Type;

    const EMPTY: &[P] = &[];
    const BOOL: P = P::Exact(Type::BOOL);
    const PTR: P = P::Exact(Type::PTR);
    const I32: P = P::Exact(Type::I32);

    pub const NONE: S = S::fixed(EMPTY, EMPTY);
    pub const EXTERNAL_OPERANDS: S = S {
        operands: L::External,
        results: L::Fixed(EMPTY),
        relations: &[],
    };
    pub const SIGNATURE: S = S {
        operands: L::External,
        results: L::Signature,
        relations: &[],
    };

    pub const INTEGER_RESULT: S = S::fixed(EMPTY, &[P::Class(C::ScalarInteger)]);
    pub const FLOAT_RESULT: S = S::fixed(EMPTY, &[P::Class(C::ScalarFloat)]);
    pub const BOOL_RESULT: S = S::fixed(EMPTY, &[BOOL]);
    pub const VECTOR_RESULT: S = S::fixed(EMPTY, &[P::Class(C::Vector)]);

    pub const INTEGER_UNARY: S = S::fixed(&[P::Bind(0, C::Integer)], &[P::Same(0)]);
    pub const INTEGER_BINARY: S = S::fixed(&[P::Bind(0, C::Integer), P::Same(0)], &[P::Same(0)]);
    pub const INTEGER_OVERFLOW: S = S::fixed(
        &[P::Bind(0, C::ScalarInteger), P::Same(0)],
        &[P::Same(0), BOOL],
    );
    pub const INTEGER_OR_BOOL_BINARY: S =
        S::fixed(&[P::Bind(0, C::IntegerOrBool), P::Same(0)], &[P::Same(0)]);
    pub const FLOAT_UNARY: S = S::fixed(&[P::Bind(0, C::Float)], &[P::Same(0)]);
    pub const FLOAT_BINARY: S = S::fixed(&[P::Bind(0, C::Float), P::Same(0)], &[P::Same(0)]);
    pub const INTEGER_TO_BOOL: S = S::fixed(&[P::Class(C::ScalarInteger)], &[BOOL]);
    pub const INTEGER_COMPARE: S = S::fixed(
        &[P::Bind(0, C::ScalarIntegerOrPointer), P::Same(0)],
        &[BOOL],
    );
    pub const FLOAT_COMPARE: S = S::fixed(&[P::Bind(0, C::ScalarFloat), P::Same(0)], &[BOOL]);

    pub const EXTEND_SIGNED: S = S::fixed(&[P::Bind(0, C::Integer)], &[P::ShapeOf(0, C::Integer)])
        .with_relations(&[R::Wider {
            from: Slot::Operand(0),
            to: Slot::Result(0),
        }]);
    pub const EXTEND_UNSIGNED: S = S::fixed(
        &[P::Bind(0, C::IntegerOrBool)],
        &[P::ShapeOf(0, C::Integer)],
    )
    .with_relations(&[R::Wider {
        from: Slot::Operand(0),
        to: Slot::Result(0),
    }]);
    pub const NARROW_INTEGER: S = S::fixed(&[P::Bind(0, C::Integer)], &[P::ShapeOf(0, C::Integer)])
        .with_relations(&[R::Narrower {
            from: Slot::Operand(0),
            to: Slot::Result(0),
        }]);
    pub const FLOAT_TO_INTEGER: S = S::fixed(&[P::Bind(0, C::Float)], &[P::ShapeOf(0, C::Integer)]);
    pub const INTEGER_TO_FLOAT: S = S::fixed(&[P::Bind(0, C::Integer)], &[P::ShapeOf(0, C::Float)]);
    pub const FLOAT_PROMOTE: S = S::fixed(&[P::Exact(Type::F32)], &[P::Exact(Type::F64)]);
    pub const FLOAT_DEMOTE: S = S::fixed(&[P::Exact(Type::F64)], &[P::Exact(Type::F32)]);
    pub const REINTERPRET: S = S::fixed(&[P::Class(C::Number)], &[P::Class(C::Number)])
        .with_relations(&[R::SameWidthDistinct {
            lhs: Slot::Operand(0),
            rhs: Slot::Result(0),
        }]);

    pub const INT_TO_PTR: S = S::fixed(&[P::Class(C::ScalarInteger)], &[PTR]);
    pub const PTR_TO_INT: S = S::fixed(&[PTR], &[P::Class(C::ScalarInteger)]);
    pub const LOAD: S = S::fixed(&[PTR], &[P::Class(C::Any)]);
    pub const STORE: S = S::fixed(&[PTR, P::Class(C::Any)], EMPTY);
    pub const STACK_LOAD: S = S::fixed(EMPTY, &[P::Class(C::Any)]);
    pub const STACK_STORE: S = S::fixed(&[P::Class(C::Any)], EMPTY);
    pub const STACK_ADDR: S = S::fixed(EMPTY, &[PTR]);
    pub const PTR_OFFSET: S = S::fixed(&[PTR], &[PTR]);
    pub const PTR_INDEX: S = S::fixed(&[PTR, P::Class(C::ScalarInteger)], &[PTR]);

    pub const SELECT: S = S::fixed(&[BOOL, P::Bind(0, C::Any), P::Same(0)], &[P::Same(0)]);
    pub const SPLAT: S = S::fixed(&[P::Bind(0, C::Scalar)], &[P::VectorOf(0)]);
    pub const SHUFFLE: S = S::fixed(&[P::Bind(0, C::Vector), P::Same(0)], &[P::Same(0)]);
    pub const INSERT_ELEMENT: S = S::fixed(
        &[
            P::Bind(0, C::Vector),
            P::ElementOf(0),
            P::Class(C::ScalarInteger),
        ],
        &[P::Same(0)],
    );
    pub const EXTRACT_ELEMENT: S = S::fixed(
        &[P::Bind(0, C::Vector), P::Class(C::ScalarInteger)],
        &[P::ElementOf(0)],
    );
    pub const REDUCTION: S = S::fixed(&[P::Bind(0, C::Vector)], &[P::ElementOf(0)]);
    pub const VECTOR_LOAD_STRIDED: S =
        S::fixed(&[PTR, P::Class(C::ScalarInteger)], &[P::Class(C::Vector)]);
    pub const VECTOR_STORE_STRIDED: S = S::fixed(
        &[PTR, P::Class(C::ScalarInteger), P::Class(C::Vector)],
        EMPTY,
    );
    pub const GATHER: S = S::fixed(
        &[PTR, P::Bind(0, C::IntegerVector)],
        &[P::ShapeOf(0, C::Vector)],
    );
    pub const SCATTER: S = S::fixed(
        &[PTR, P::Bind(0, C::IntegerVector), P::ShapeOf(0, C::Vector)],
        EMPTY,
    );
    pub const SET_VECTOR_LENGTH: S = S::fixed(&[P::Class(C::ScalarInteger)], &[I32]);
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgebraicConstant {
    Zero,
    One,
    AllOnes,
}

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

macro_rules! define_opcodes {
    ($(
        $name:ident {
            mnemonic: $mnemonic:literal,
            format: $format:ident,
            types: $types:ident,
            $(builder: $builder_kind:ident($builder_name:ident),)?
            traits: [$($trait:ident),* $(,)?],
            memory: $memory:ident
            $(, constraints: [$($constraint:ident),* $(,)?])?
            $(, identity: $identity:ident)?
            $(, absorbing: $absorbing:ident)?
            $(,)?
        }
    )*) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum Opcode {
            $($name,)*
        }

        impl Opcode {
            pub const ALL: &'static [Self] = &[$(Self::$name,)*];

            pub const fn spec(self) -> $crate::opspec::OpSpec {
                match self {
                    $(Self::$name => $crate::opspec::OpSpec {
                        mnemonic: $mnemonic,
                        format: $crate::opspec::OpFormat::$format,
                        type_scheme: &$crate::opspec::type_schemes::$types,
                        traits: $crate::opspec::OpTraits::empty()
                            $(.union($crate::opspec::OpTraits::$trait))*,
                        memory_effect: $crate::opspec::MemoryEffect::$memory,
                        constraints: &[$($($crate::opspec::OpConstraint::$constraint),*)?],
                        identity: define_opcodes!(@algebraic $($identity)?),
                        absorbing: define_opcodes!(@algebraic $($absorbing)?),
                    },)*
                }
            }

            pub fn from_mnemonic(mnemonic: &str) -> Option<Self> {
                match mnemonic {
                    $($mnemonic => Some(Self::$name),)*
                    _ => None,
                }
            }
        }


        impl<'b, 'a> $crate::builder::InstBuilder<'b, 'a> {
            $(define_opcodes!(@builder_method $name, $format $(, $builder_kind($builder_name))?);)*
        }
    };

    (@builder_method $opcode:ident, $format:ident) => {};
    (@builder_method $opcode:ident, Unary, unary($method:ident)) => {
        pub fn $method(&mut self, arg: $crate::Value) -> $crate::Value {
            self.push_unary($crate::Opcode::$opcode, arg)
        }
    };
    (@builder_method $opcode:ident, Binary, binary($method:ident)) => {
        pub fn $method(&mut self, lhs: $crate::Value, rhs: $crate::Value) -> $crate::Value {
            self.push_binary($crate::Opcode::$opcode, lhs, rhs)
        }
    };
    (@builder_method $opcode:ident, Binary, binary_pair($method:ident)) => {
        pub fn $method(
            &mut self,
            lhs: $crate::Value,
            rhs: $crate::Value,
        ) -> ($crate::Value, $crate::Value) {
            let inst = self.push_raw($crate::InstructionData::Binary {
                opcode: $crate::Opcode::$opcode,
                args: [lhs, rhs],
            });
            self.result_pair(inst)
        }
    };
    (@builder_method $opcode:ident, Unary, unary_typed($method:ident)) => {
        pub fn $method(&mut self, arg: $crate::Value, ty: $crate::Type) -> $crate::Value {
            self.push_with_type(
                $crate::InstructionData::Unary {
                    opcode: $crate::Opcode::$opcode,
                    arg,
                },
                ty,
            )
        }
    };
    (@builder_method IntToPtr, IntToPtr, unary_inst($method:ident)) => {
        pub fn $method(&mut self, arg: $crate::Value) -> $crate::Value {
            self.push($crate::InstructionData::IntToPtr { arg }).unwrap()
        }
    };
    (@builder_method PtrToInt, PtrToInt, unary_inst_typed($method:ident)) => {
        pub fn $method(&mut self, arg: $crate::Value, ty: $crate::Type) -> $crate::Value {
            self.push_with_type($crate::InstructionData::PtrToInt { arg }, ty)
        }
    };
    (@builder_method Unreachable, Unreachable, nullary($method:ident)) => {
        pub fn $method(&mut self) {
            self.push($crate::InstructionData::Unreachable);
        }
    };
    (@builder_method Nop, Nop, nullary($method:ident)) => {
        pub fn $method(&mut self) {
            self.push($crate::InstructionData::Nop);
        }
    };
    (@builder_method Icmp, IntCompare, int_compare($method:ident)) => {
        pub fn $method(
            &mut self,
            kind: $crate::IntCC,
            lhs: $crate::Value,
            rhs: $crate::Value,
        ) -> $crate::Value {
            self.push($crate::InstructionData::IntCompare {
                kind,
                args: [lhs, rhs],
            })
            .unwrap()
        }
    };
    (@builder_method Fcmp, FloatCompare, float_compare($method:ident)) => {
        pub fn $method(
            &mut self,
            kind: $crate::FloatCC,
            lhs: $crate::Value,
            rhs: $crate::Value,
        ) -> $crate::Value {
            self.push($crate::InstructionData::FloatCompare {
                kind,
                args: [lhs, rhs],
            })
            .unwrap()
        }
    };
    (@builder_method $opcode:ident, Ternary, ternary($method:ident)) => {
        pub fn $method(
            &mut self,
            first: $crate::Value,
            second: $crate::Value,
            third: $crate::Value,
        ) -> $crate::Value {
            self.push($crate::InstructionData::Ternary {
                opcode: $crate::Opcode::$opcode,
                args: [first, second, third],
            })
            .unwrap()
        }
    };

    (@algebraic) => {
        None
    };
    (@algebraic $constant:ident) => {
        Some($crate::opspec::AlgebraicConstant::$constant)
    };

}

pub(crate) use define_opcodes;

#[cfg(test)]
mod tests {
    use super::ResultTypes;
    use super::{
        MemoryEffect, OpConstraint, OpFormat, TypeClass, TypeList, TypePattern, TypeScheme,
    };
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
                .infer_results(&[Type::I32, Type::I32]),
            ResultTypes::Inferred(smallvec::smallvec![Type::BOOL])
        );
    }

    #[test]
    #[should_panic(expected = "OpSpec type variable exceeds binding capacity")]
    fn malformed_type_variable_is_an_internal_error() {
        const OPERANDS: &[TypePattern] = &[TypePattern::Bind(4, TypeClass::Integer)];
        const RESULTS: &[TypePattern] = &[TypePattern::Same(4)];
        TypeScheme::fixed(OPERANDS, RESULTS).infer_results(&[Type::I32]);
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
