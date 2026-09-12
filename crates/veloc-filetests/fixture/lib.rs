#![feature(const_trait_impl, const_cmp)]

include!("../../../veloc/mir/src/root.rs");

pub use inst::{FloatOrderCC, OrderCC};

pub mod tokens {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Stamp(pub u32);

    impl crate::type_methods::Stamp for Stamp {
        fn number(self) -> u32 {
            self.0
        }
    }
}

impl<M> host::traits::Tokens for host::Context<'_, M> {
    fn stamp(&self, number: u32) -> tokens::Stamp {
        tokens::Stamp(number)
    }
}
