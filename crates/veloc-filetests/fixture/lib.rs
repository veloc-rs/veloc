include!("../../../veloc/mir/src/lib.rs");

pub use inst::{FloatOrderCC, OrderCC};

pub mod tokens {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Stamp(pub u32);
}

impl<M> host::traits::Tokens for host::Context<'_, M> {
    fn stamp(&self, number: u32) -> tokens::Stamp {
        tokens::Stamp(number)
    }
}
