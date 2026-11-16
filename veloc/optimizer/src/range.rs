//! One half-open interval on the fixed-width integer circle, like LLVM's
//! ConstantRange. Disconnected results are conservatively covered by one arc.
use crate::rewrite::Fact;
use veloc_mir::{Inst, InstView, IntCC, Opcode, ScalarConst, Type, Value, dfg::DataFlowGraph};

fn integer_bits(ty: Type) -> Option<u32> {
    match ty {
        Type::BOOL => Some(1),
        Type::I8 => Some(8),
        Type::I16 => Some(16),
        Type::I32 => Some(32),
        Type::I64 => Some(64),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct IntRange {
    pub(crate) ty: Type,
    // lower > upper wraps through zero. Equal endpoints are reserved:
    // [0, 0) is empty; [MAX, MAX) is full.
    lower: u64,
    upper: u64,
}

impl IntRange {
    pub(crate) fn full(ty: Type) -> Option<Self> {
        let modulus = 1u128 << integer_bits(ty)?;
        Some(Self::from_span(ty, 0, modulus))
    }

    fn modulus(&self) -> u128 {
        1u128 << integer_bits(self.ty).unwrap()
    }

    fn empty(ty: Type) -> Self {
        Self {
            ty,
            lower: 0,
            upper: 0,
        }
    }

    /// u128 arithmetic can represent a full 64-bit span without overflow.
    fn from_span(ty: Type, start: u128, len: u128) -> Self {
        let modulus = 1u128 << integer_bits(ty).unwrap();
        if len == 0 {
            return Self::empty(ty);
        }
        if len >= modulus {
            let max = (modulus - 1) as u64;
            return Self {
                ty,
                lower: max,
                upper: max,
            };
        }
        let lower = start % modulus;
        Self {
            ty,
            lower: lower as u64,
            upper: ((lower + len) % modulus) as u64,
        }
    }

    fn len(&self) -> u128 {
        if self.is_full() {
            self.modulus()
        } else {
            (u128::from(self.upper) + self.modulus() - u128::from(self.lower)) % self.modulus()
        }
    }

    /// The last value along the arc, which need not be the unsigned maximum.
    fn last(&self) -> u128 {
        (u128::from(self.upper) + self.modulus() - 1) % self.modulus()
    }

    fn interval(ty: Type, lo: u128, hi: u128) -> Self {
        Self::from_span(ty, lo, hi.saturating_sub(lo))
    }

    /// Project an inclusive mathematical interval onto the fixed-width domain.
    fn modular(ty: Type, min: u128, max: u128) -> Self {
        let modulus = 1u128 << integer_bits(ty).unwrap();
        let len = if max - min >= modulus - 1 {
            modulus
        } else {
            max - min + 1
        };
        Self::from_span(ty, min, len)
    }

    pub(crate) fn constant(value: ScalarConst) -> Option<Self> {
        integer_bits(value.ty())?;
        Some(Self::from_span(value.ty(), value.to_bits().into(), 1))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.lower == 0 && self.upper == 0
    }

    fn is_full(&self) -> bool {
        self.lower == self.upper && !self.is_empty()
    }

    pub(crate) fn contains(&self, value: u64) -> bool {
        let distance =
            (u128::from(value) + self.modulus() - u128::from(self.lower)) % self.modulus();
        distance < self.len()
    }

    pub(crate) fn singleton(&self) -> Option<ScalarConst> {
        (self.len() == 1).then(|| ScalarConst::from_bits(self.ty, self.lower).unwrap())
    }

    pub(crate) fn fact(self) -> Fact {
        // Empty means an infeasible constraint, not a pending SCCP fact.
        assert!(!self.is_empty(), "empty range is not a value fact");
        if let Some(value) = self.singleton() {
            Fact::Constant(value)
        } else if self.is_full() {
            Fact::Varying
        } else {
            Fact::Range(self)
        }
    }

    /// Smallest covering arc. Ties prefer the smaller unsigned lower bound.
    /// The result contains both inputs, but may also fill a gap between them.
    pub(crate) fn union(&self, other: &Self) -> Self {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        let modulus = self.modulus();
        let (a, b) = (u128::from(self.lower), u128::from(other.lower));
        let (alen, blen) = (self.len(), other.len());
        let from_a = alen.max((b + modulus - a) % modulus + blen).min(modulus);
        let from_b = blen.max((a + modulus - b) % modulus + alen).min(modulus);
        let (len, start) = (from_a, a).min((from_b, b));
        Self::from_span(self.ty, start, len)
    }

    /// Cover the intersection with one arc. Empty intersections stay exact;
    /// a disconnected intersection may retain values excluded by one input.
    pub(crate) fn intersect(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_full() {
            return *self;
        }
        if other.is_empty() || self.is_full() {
            return *other;
        }
        // Cut the circle at our lower bound. The other arc has at most two
        // pieces in this coordinate system; intersect each with [0, len).
        let modulus = self.modulus();
        let start = u128::from(self.lower);
        let offset = (u128::from(other.lower) + modulus - start) % modulus;
        let end = offset + other.len();
        let first = Self::from_span(
            self.ty,
            start + offset,
            self.len().min(end).saturating_sub(offset),
        );
        let second = Self::from_span(self.ty, start, self.len().min(end.saturating_sub(modulus)));
        first.union(&second)
    }

    fn unsigned_bounds(&self) -> (u128, u128) {
        let max = self.modulus() - 1;
        (
            if self.contains(0) {
                0
            } else {
                self.lower.into()
            },
            if self.contains(max as u64) {
                max
            } else {
                self.last()
            },
        )
    }

    pub(crate) fn unsigned_min(&self) -> u64 {
        self.unsigned_bounds().0 as u64
    }

    fn bounds(&self, signed: bool) -> (i128, i128) {
        if !signed {
            let (min, max) = self.unsigned_bounds();
            return (min as i128, max as i128);
        }
        let modulus = self.modulus();
        let sign = modulus / 2;
        let interpret = |value: u128| {
            if value < sign {
                value as i128
            } else {
                value as i128 - modulus as i128
            }
        };
        (
            if self.contains(sign as u64) {
                -(sign as i128)
            } else {
                interpret(self.lower.into())
            },
            if self.contains((sign - 1) as u64) {
                (sign - 1) as i128
            } else {
                interpret(self.last())
            },
        )
    }

    /// Compare nonempty value ranges; callers handle infeasible paths separately.
    pub(crate) fn compare(&self, kind: IntCC, other: &Self) -> Option<bool> {
        debug_assert!(!self.is_empty() && !other.is_empty());
        if matches!(kind, IntCC::Eq | IntCC::Ne) {
            let equal = if self.singleton().is_some() && self == other {
                true
            } else if self.intersect(other).is_empty() {
                false
            } else {
                return None;
            };
            return Some(equal == (kind == IntCC::Eq));
        }
        let (a, b) = self.bounds(kind.is_signed());
        let (c, d) = other.bounds(kind.is_signed());
        let (yes, no) = match kind {
            IntCC::LtS | IntCC::LtU => (b < c, a >= d),
            IntCC::LeS | IntCC::LeU => (b <= c, a > d),
            IntCC::GtS | IntCC::GtU => (a > d, b <= c),
            IntCC::GeS | IntCC::GeU => (a >= d, b < c),
            _ => unreachable!(),
        };
        if yes {
            Some(true)
        } else if no {
            Some(false)
        } else {
            None
        }
    }

    fn truncate(&self, ty: Type) -> Self {
        Self::from_span(ty, self.lower.into(), self.len())
    }

    fn zero_extend(&self, ty: Type) -> Self {
        let (min, max) = self.unsigned_bounds();
        Self::modular(ty, min, max)
    }

    fn sign_extend(&self, ty: Type) -> Self {
        let (min, max) = self.bounds(true);
        let modulus = 1i128 << integer_bits(ty).unwrap();
        Self::modular(ty, (modulus + min) as u128, (modulus + max) as u128)
    }

    fn wrapping_neg(&self) -> Self {
        Self::from_span(self.ty, self.modulus() - self.last(), self.len())
    }

    fn wrapping_add(&self, rhs: &Self) -> Self {
        Self::from_span(
            self.ty,
            u128::from(self.lower) + u128::from(rhs.lower),
            self.len() + rhs.len() - 1,
        )
    }

    fn wrapping_sub(&self, rhs: &Self) -> Self {
        Self::from_span(
            self.ty,
            self.modulus() + u128::from(self.lower) - rhs.last(),
            self.len() + rhs.len() - 1,
        )
    }

    /// Bound operations in unsigned order. A wrapping input may lose precision,
    /// but evaluation always has constant cost, independent of the value count.
    fn map_unsigned(
        &self,
        rhs: &Self,
        map: impl Fn((u128, u128), (u128, u128)) -> (u128, u128),
    ) -> Self {
        let (min, max) = map(self.unsigned_bounds(), rhs.unsigned_bounds());
        Self::modular(self.ty, min, max)
    }

    fn wrapping_mul(&self, rhs: &Self) -> Self {
        self.map_unsigned(rhs, |(al, ah), (bl, bh)| (al * bl, ah * bh))
    }

    fn bitand(&self, rhs: &Self) -> Self {
        self.map_unsigned(rhs, |(al, ah), (bl, bh)| {
            if al == ah && bl == bh {
                (al & bl, al & bl)
            } else {
                (0, ah.min(bh))
            }
        })
    }

    fn bitor(&self, rhs: &Self) -> Self {
        self.map_unsigned(rhs, |(al, ah), (bl, bh)| {
            if al == ah && bl == bh {
                (al | bl, al | bl)
            } else {
                (al.max(bl), self.modulus() - 1)
            }
        })
    }

    fn bitxor(&self, rhs: &Self) -> Self {
        self.map_unsigned(rhs, |(al, ah), (bl, bh)| {
            if al == ah && bl == bh {
                (al ^ bl, al ^ bl)
            } else {
                (0, self.modulus() - 1)
            }
        })
    }

    fn div_unsigned(&self, rhs: &Self) -> Option<Self> {
        if rhs.contains(0) {
            return None;
        }
        Some(self.map_unsigned(rhs, |(al, ah), (bl, bh)| (al / bh, ah / bl)))
    }

    fn rem_unsigned(&self, rhs: &Self) -> Option<Self> {
        if rhs.contains(0) {
            return None;
        }
        Some(self.map_unsigned(rhs, |(_, ah), (_, bh)| (0, ah.min(bh - 1))))
    }

    /// Exact bit patterns satisfying `x kind constant`, including signed order.
    pub(crate) fn constraint(kind: IntCC, constant: ScalarConst) -> Option<Self> {
        let ty = constant.ty();
        let modulus = 1u128 << integer_bits(ty)?;
        let bias = if kind.is_signed() { modulus / 2 } else { 0 };
        let c = (u128::from(constant.to_bits()) + bias) % modulus;
        let allowed = match kind {
            IntCC::Eq => Self::from_span(ty, c, 1),
            IntCC::Ne => Self::from_span(ty, c + 1, modulus - 1),
            IntCC::LtS | IntCC::LtU => Self::interval(ty, 0, c),
            IntCC::LeS | IntCC::LeU => Self::interval(ty, 0, c + 1),
            IntCC::GtS | IntCC::GtU => Self::interval(ty, c + 1, modulus),
            IntCC::GeS | IntCC::GeU => Self::interval(ty, c, modulus),
        };
        Some(Self::from_span(
            ty,
            u128::from(allowed.lower) + bias,
            allowed.len(),
        ))
    }
}
/// Transfer a single integer result. Operations that can still trap, or whose
/// semantics are unsupported, return None so callers retain the instruction.
pub(crate) fn evaluate(
    dfg: &DataFlowGraph,
    inst: Inst,
    fact: impl Fn(Value) -> Fact,
) -> Option<Fact> {
    let &[result] = dfg.inst_results(inst) else {
        return None;
    };
    let ty = dfg.value_type(result);
    integer_bits(ty)?;
    let range = |value| fact(value).range(dfg.value_type(value));
    let output = match (dfg.opcode(inst), dfg.operands(inst)) {
        (Opcode::Select, &[condition, yes, no]) => {
            return Some(fact(condition).select(fact(yes), fact(no)));
        }
        (Opcode::IEqz, &[arg]) => {
            let a = range(arg)?;
            return Some(
                a.compare(IntCC::Eq, &IntRange::interval(a.ty, 0, 1))
                    .map_or(Fact::Varying, |v| Fact::Constant(v.into())),
            );
        }
        (Opcode::Wrap, &[arg]) => range(arg)?.truncate(ty),
        (Opcode::ExtendU, &[arg]) => range(arg)?.zero_extend(ty),
        (Opcode::ExtendS, &[arg]) => range(arg)?.sign_extend(ty),
        (Opcode::INeg, &[arg]) => range(arg)?.wrapping_neg(),
        (opcode, &[lhs, rhs]) => {
            let a = range(lhs)?;
            let b = range(rhs)?;
            if let InstView::IntCompare { kind, .. } = dfg.inst(inst) {
                return Some(
                    a.compare(kind, &b)
                        .map_or(Fact::Varying, |v| Fact::Constant(v.into())),
                );
            }
            match opcode {
                Opcode::IAdd => a.wrapping_add(&b),
                Opcode::ISub => a.wrapping_sub(&b),
                Opcode::IMul => a.wrapping_mul(&b),
                Opcode::IAnd => a.bitand(&b),
                Opcode::IOr => a.bitor(&b),
                Opcode::IXor => a.bitxor(&b),
                Opcode::IDivU => a.div_unsigned(&b)?,
                Opcode::IRemU => a.rem_unsigned(&b)?,
                _ => return None,
            }
        }
        _ => return None,
    };
    Some(output.fact())
}
