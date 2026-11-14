//! Shared expression evaluation over operand identities and constant facts.
//! Spec supplies concrete semantics and local identities; callers own analysis
//! state, control-flow reachability and publication of the returned results.
use veloc_mir::constant::ScalarConst;
use veloc_mir::{Opcode, Type, TypeInfo, Value};

/// A replacement for one instruction result. Operand indices refer to the
/// original argument order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fold {
    Operand(usize),
    Constant(ScalarConst),
}

/// Constant-propagation lattice. Unknown means no information has arrived yet,
/// not an IR undef value. Facts never serve as operand identities.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum Fact {
    #[default]
    Unknown,
    Constant(ScalarConst),
    Varying,
}

impl Fact {
    pub(crate) fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Unknown, x) | (x, Self::Unknown) => x,
            (a, b) if a == b => a,
            _ => Self::Varying,
        }
    }

    pub(crate) fn constant(self) -> Option<ScalarConst> {
        match self {
            Self::Constant(c) => Some(c),
            _ => None,
        }
    }

    /// Select between facts after established operand/constant folds have run.
    /// A varying condition joins both arms; Unknown contributes no information
    /// yet, so a resulting constant remains provisional until the fixed point.
    pub(crate) fn select(self, yes: Self, no: Self) -> Self {
        match self {
            Self::Constant(c) => {
                if c.to_bits() != 0 {
                    yes
                } else {
                    no
                }
            }
            Self::Unknown => Self::Unknown,
            Self::Varying => yes.join(no),
        }
    }
}

/// Executable successor knowledge; Unknown selectors must not activate edges.
pub(crate) enum Successors {
    Pending,
    All,
    One(usize),
}

mod local {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/local_folds.rs"));
}

pub(crate) use local::{accepts, plan, successors, validate_fields};
pub use local::{can_fold, fold};

/// Pattern literals are masked to the operand width. An unknown constant proves
/// neither equality nor inequality, in direct reductions, guards or queries.
pub(crate) fn matches_constant(value: Option<ScalarConst>, bits: u64, equal: bool) -> bool {
    value.is_some_and(|c| {
        c.ty()
            .element_bits()
            .and_then(|n| 64u32.checked_sub(n))
            .and_then(|shift| u64::MAX.checked_shr(shift))
            .is_some_and(|mask| (c.to_bits() == (bits & mask)) == equal)
    })
}

/// Repeated pattern variables accept identical values or identical established
/// constant facts. ScalarConst equality includes the type and exact bit pattern.
pub(crate) fn same_value<V: Copy + Eq>(
    lhs: V,
    rhs: V,
    mut constant: impl FnMut(V) -> Option<ScalarConst>,
) -> bool {
    lhs == rhs || constant(lhs).is_some_and(|value| constant(rhs) == Some(value))
}

/// Fold an existing instruction to constants or original operands.
pub fn fold_inst(
    dfg: &veloc_mir::dfg::DataFlowGraph,
    inst: veloc_mir::Inst,
    constant: impl FnMut(Value) -> Option<ScalarConst>,
) -> Option<smallvec::SmallVec<[Fold; 2]>> {
    let results: smallvec::SmallVec<[Type; 2]> = dfg
        .inst_results(inst)
        .iter()
        .map(|&v| dfg.value_type(v))
        .collect();
    fold(
        dfg.inst_fields(inst),
        dfg.operands(inst),
        &results,
        constant,
    )
}
