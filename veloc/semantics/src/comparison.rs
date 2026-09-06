use crate::{Error, Width};

/// An integer predicate defined by signedness and accepted ordered outcomes.
/// Bits 0/1/2 denote less/equal/greater. There is no unordered integer outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntPredicate {
    signed: bool,
    outcomes: u8,
}

impl IntPredicate {
    pub const fn new(signed: bool, outcomes: u8) -> Self {
        assert!(outcomes < 8, "invalid integer comparison outcomes");
        Self { signed, outcomes }
    }

    pub const fn signed(self) -> bool {
        self.signed
    }
    pub const fn outcomes(self) -> u8 {
        self.outcomes
    }

    pub fn eval(self, width: u16, lhs: u128, rhs: u128) -> Result<bool, Error> {
        let width = Width::new(width)?;
        let sign = if self.signed {
            1 << (width.bits() - 1)
        } else {
            0
        };
        let lhs = width.normalize(lhs) ^ sign;
        let rhs = width.normalize(rhs) ^ sign;
        let outcome = match lhs.cmp(&rhs) {
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Equal => 2,
            std::cmp::Ordering::Greater => 4,
        };
        Ok(self.outcomes & outcome != 0)
    }
}
