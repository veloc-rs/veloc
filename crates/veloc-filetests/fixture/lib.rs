#![feature(const_trait_impl)]

include!("../../../veloc/mir/src/root.rs");

pub mod tokens {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct Stamp(pub u32);
}
