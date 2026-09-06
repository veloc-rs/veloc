//! Executable, fixed-width semantics shared by IRs and verification tools.
//!
//! This crate does not depend on a program IR or an SMT solver. Generated
//! evaluator generators use its primitive/value vocabulary. Checked graphs,
//! reference execution and SMT export serve build-time generation and offline
//! checks; the normal folding path has no dependency on this crate.
//!
//! The model covers bitvectors, booleans and explicit operation-level traps: no memory,
//! floating point, undefined behavior, or scalable vectors. A solver result
//! concerns this model, not the correctness of an entire compiler.

mod comparison;
mod expr;
mod program;
mod smt;
mod value;

pub use comparison::IntPredicate;
pub use expr::{Expr, Function};
pub use program::{ComparisonRef, Conversion, Program, SemanticSpec, Step, TypeRef};
pub use smt::equivalence_query;
pub use value::{Outcome, Sort, Trap, Value};

/// Width-dependent constants used by trusted algebraic facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvConst {
    Zero,
    One,
    AllOnes,
    Width,
    SignedMin,
}

impl BvConst {
    pub const ALL: &'static [Self] = &[
        Self::Zero,
        Self::One,
        Self::AllOnes,
        Self::Width,
        Self::SignedMin,
    ];

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|value| value.name() == name)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Zero => "Zero",
            Self::One => "One",
            Self::AllOnes => "AllOnes",
            Self::Width => "Width",
            Self::SignedMin => "SignedMin",
        }
    }

    pub fn eval(self, width: u16) -> Result<u128, Error> {
        let width = Width::new(width)?;
        Ok(match self {
            Self::Zero => 0,
            Self::One => 1,
            Self::AllOnes => width.mask(),
            Self::Width => u128::from(width.bits()),
            Self::SignedMin => 1 << (width.bits() - 1),
        })
    }
}

/// Trusted facts about a primitive, valid at every supported bitvector width.
///
/// These are reviewed definitions, not automatically proved claims. Identity
/// and absorbing elements are two-sided; no width-specific facts or one-sided
/// identities (such as `x - 0 = x`) are implied. False/None means no fact is
/// provided, rather than claiming the property fails at every width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Algebra {
    pub commutative: bool,
    pub associative: bool,
    pub idempotent: bool,
    pub identity: Option<BvConst>,
    pub absorbing: Option<BvConst>,
}

impl Algebra {
    pub const NONE: Self = Self {
        commutative: false,
        associative: false,
        idempotent: false,
        identity: None,
        absorbing: None,
    };
}

/// A nonzero bitvector width supported by the executable model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Width(u16);

impl Width {
    pub fn new(bits: u16) -> Result<Self, Error> {
        if (1..=128).contains(&bits) {
            Ok(Self(bits))
        } else {
            Err(Error::InvalidWidth(bits))
        }
    }

    pub const fn bits(self) -> u16 {
        self.0
    }

    pub const fn mask(self) -> u128 {
        if self.0 == 128 {
            u128::MAX
        } else {
            (1u128 << self.0) - 1
        }
    }

    pub const fn normalize(self, value: u128) -> u128 {
        value & self.mask()
    }
}

/// Total operations on equally sized bitvectors. Division and shifts use
/// SMT-LIB semantics; language-specific traps and masked shifts are composed
/// explicitly by operation definitions, not hidden in these primitives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BvOp {
    Add,
    Sub,
    Mul,
    Neg,
    And,
    Or,
    Xor,
    Shl,
    LShr,
    AShr,
    UDiv,
    SDiv,
    URem,
    SRem,
    Clz,
    Ctz,
    Popcnt,
}

impl BvOp {
    pub const ALL: &'static [Self] = &[
        Self::Add,
        Self::Sub,
        Self::Mul,
        Self::Neg,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Shl,
        Self::LShr,
        Self::AShr,
        Self::UDiv,
        Self::SDiv,
        Self::URem,
        Self::SRem,
        Self::Clz,
        Self::Ctz,
        Self::Popcnt,
    ];

    pub const fn arity(self) -> usize {
        match self {
            Self::Neg | Self::Clz | Self::Ctz | Self::Popcnt => 1,
            _ => 2,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "bv.add",
            Self::Sub => "bv.sub",
            Self::Mul => "bv.mul",
            Self::Neg => "bv.neg",
            Self::And => "bv.and",
            Self::Or => "bv.or",
            Self::Xor => "bv.xor",
            Self::Shl => "bv.shl",
            Self::LShr => "bv.lshr",
            Self::AShr => "bv.ashr",
            Self::UDiv => "bv.udiv",
            Self::SDiv => "bv.sdiv",
            Self::URem => "bv.urem",
            Self::SRem => "bv.srem",
            Self::Clz => "bv.clz",
            Self::Ctz => "bv.ctz",
            Self::Popcnt => "bv.popcnt",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|op| op.name() == name)
    }

    pub const fn algebra(self) -> Algebra {
        use BvConst::{AllOnes, One, Zero};
        let ac = Algebra {
            commutative: true,
            associative: true,
            ..Algebra::NONE
        };
        match self {
            Self::Add | Self::Xor => Algebra {
                identity: Some(Zero),
                ..ac
            },
            Self::Mul => Algebra {
                identity: Some(One),
                absorbing: Some(Zero),
                ..ac
            },
            Self::And => Algebra {
                idempotent: true,
                identity: Some(AllOnes),
                absorbing: Some(Zero),
                ..ac
            },
            Self::Or => Algebra {
                idempotent: true,
                identity: Some(Zero),
                absorbing: Some(AllOnes),
                ..ac
            },
            _ => Algebra::NONE,
        }
    }

    /// Evaluate modulo `2^width`. Inputs are normalized to the same width.
    ///
    /// Widths outside 1..=128 and incorrect operand counts are errors.
    #[inline]
    pub fn eval(self, width: u16, args: &[u128]) -> Result<u128, Error> {
        let width = Width::new(width)?;
        if args.len() != self.arity() {
            return Err(Error::Arity {
                expected: self.arity(),
                actual: args.len(),
            });
        }
        let x = width.normalize(args[0]);
        let y = args.get(1).map_or(0, |&v| width.normalize(v));
        let signed = |v: u128| ((v << (128 - width.bits())) as i128) >> (128 - width.bits());
        let result = match self {
            Self::Add => x.wrapping_add(y),
            Self::Sub => x.wrapping_sub(y),
            Self::Mul => x.wrapping_mul(y),
            Self::And => x & y,
            Self::Or => x | y,
            Self::Xor => x ^ y,
            Self::Neg => x.wrapping_neg(),
            Self::Shl => {
                if y >= width.bits() as u128 {
                    0
                } else {
                    x << y
                }
            }
            Self::LShr => {
                if y >= width.bits() as u128 {
                    0
                } else {
                    x >> y
                }
            }
            Self::AShr => (signed(x) >> y.min((width.bits() - 1) as u128)) as u128,
            Self::UDiv => x.checked_div(y).unwrap_or_else(|| width.mask()),
            Self::URem => {
                if y == 0 {
                    x
                } else {
                    x % y
                }
            }
            Self::SDiv => {
                if y == 0 {
                    if signed(x) < 0 { 1 } else { width.mask() }
                } else {
                    signed(x).wrapping_div(signed(y)) as u128
                }
            }
            Self::SRem => {
                if y == 0 {
                    x
                } else {
                    signed(x).wrapping_rem(signed(y)) as u128
                }
            }
            Self::Clz => (x.leading_zeros() - (128 - u32::from(width.bits()))) as u128,
            Self::Ctz => x.trailing_zeros().min(u32::from(width.bits())) as u128,
            Self::Popcnt => x.count_ones() as u128,
        };
        Ok(width.normalize(result))
    }

    pub(crate) const fn smt_name(self) -> Option<&'static str> {
        Some(match self {
            Self::Add => "bvadd",
            Self::Sub => "bvsub",
            Self::Mul => "bvmul",
            Self::Neg => "bvneg",
            Self::And => "bvand",
            Self::Or => "bvor",
            Self::Xor => "bvxor",
            Self::Shl => "bvshl",
            Self::LShr => "bvlshr",
            Self::AShr => "bvashr",
            Self::UDiv => "bvudiv",
            Self::SDiv => "bvsdiv",
            Self::URem => "bvurem",
            Self::SRem => "bvsrem",
            Self::Clz | Self::Ctz | Self::Popcnt => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Trapped(Trap),
    InvalidWidth(u16),
    Arity {
        expected: usize,
        actual: usize,
    },
    ExpectedBv(Sort),
    SortMismatch {
        expected: Sort,
        actual: Sort,
    },
    InvalidExtract {
        width: u16,
        high: u16,
        low: u16,
    },
    InvalidExtension {
        from: u16,
        to: u16,
    },
    InputIndex {
        index: usize,
        count: usize,
    },
    StepIndex {
        index: usize,
        count: usize,
    },
    TooManySteps {
        count: usize,
        max: usize,
    },
    InputSort {
        index: usize,
        expected: Sort,
        actual: Sort,
    },
    ValueSort {
        index: usize,
        expected: Sort,
    },
    SignatureMismatch,
    ResultArity {
        expected: usize,
        actual: usize,
    },
    PropertyIndex {
        index: usize,
        count: usize,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Trapped(trap) => write!(f, "execution trapped: {trap:?}"),
            Self::InvalidWidth(width) => write!(f, "bitvector width {width} is outside 1..=128"),
            Self::Arity { expected, actual } => {
                write!(f, "expected {expected} operands, got {actual}")
            }
            Self::ExpectedBv(actual) => write!(f, "expected a bitvector, got {actual}"),
            Self::SortMismatch { expected, actual } => {
                write!(f, "expected {expected}, got {actual}")
            }
            Self::InvalidExtract { width, high, low } => {
                write!(f, "cannot extract bits {high}..={low} from bv{width}")
            }
            Self::InvalidExtension { from, to } => {
                write!(f, "cannot extend bv{from} to bv{to}")
            }
            Self::InputIndex { index, count } => {
                write!(
                    f,
                    "input {index} is outside a signature with {count} inputs"
                )
            }
            Self::StepIndex { index, count } => {
                write!(f, "step {index} is outside the {count} available steps")
            }
            Self::TooManySteps { count, max } => {
                write!(
                    f,
                    "program has {count} steps, exceeding the index capacity of {max}"
                )
            }
            Self::InputSort {
                index,
                expected,
                actual,
            } => {
                write!(f, "input {index} has sort {actual}, expected {expected}")
            }
            Self::ValueSort { index, expected } => {
                write!(f, "value for input {index} is not a {expected}")
            }
            Self::SignatureMismatch => write!(f, "equivalence requires identical input signatures"),
            Self::ResultArity { expected, actual } => {
                write!(f, "expected {expected} results, got {actual}")
            }
            Self::PropertyIndex { index, count } => write!(
                f,
                "property {index} is outside the {count} supplied properties"
            ),
        }
    }
}

impl std::error::Error for Error {}
