//! Stable identities for input and result positions, shared by IR and rule codecs.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperandRef {
    Input(usize),
    Result(usize),
}

impl OperandRef {
    pub fn encode(self) -> usize {
        let (index, result) = match self {
            Self::Input(index) => (index, 0),
            Self::Result(index) => (index, 1),
        };
        index.checked_mul(2).expect("operand index overflow") | result
    }

    pub fn decode(value: usize) -> Self {
        if value & 1 == 0 {
            Self::Input(value >> 1)
        } else {
            Self::Result(value >> 1)
        }
    }

    pub fn get<'a, T>(self, inputs: &'a [T], results: &'a [T]) -> Option<&'a T> {
        match self {
            Self::Input(index) => inputs.get(index),
            Self::Result(index) => results.get(index),
        }
    }
}
