#![feature(const_trait_impl, const_cmp)]

include!("../../../veloc/mir/src/root.rs");

pub mod tokens {
    pub struct Tokens<'a>(pub &'a u32);
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Stamp(pub u32);

    impl crate::type_methods::Stamp for Stamp {
        fn number(self) -> u32 {
            self.0
        }
    }
}

impl crate::type_methods::Tokens for tokens::Tokens<'_> {
    fn stamp(&self, number: u32) -> tokens::Stamp {
        tokens::Stamp(number + *self.0)
    }
}
