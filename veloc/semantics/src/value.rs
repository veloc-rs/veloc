use crate::{Error, Width};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sort {
    Bool,
    Bv(Width),
}

impl Sort {
    pub fn bv(width: u16) -> Result<Self, Error> {
        Width::new(width).map(Self::Bv)
    }

    pub fn width(self) -> Result<Width, Error> {
        match self {
            Self::Bv(width) => Ok(width),
            Self::Bool => Err(Error::ExpectedBv(self)),
        }
    }
}

impl std::fmt::Display for Sort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bool => f.write_str("bool"),
            Self::Bv(width) => write!(f, "bv{}", width.bits()),
        }
    }
}

/// Runtime values use the width declared in the function signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Value {
    Bv(u128),
    Bool(bool),
}

/// Observable operation-level failures, separate from malformed semantic graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Trap {
    DivisionByZero,
    IntegerOverflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Values(Vec<Value>),
    Trap(Trap),
}
