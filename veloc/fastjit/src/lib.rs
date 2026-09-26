//! A fast baseline compiler built from small, pre-encoded machine-code stencils.
//!
//! This crate owns code selection, stencil patching, and object production. It
//! does not own Wasm semantics or the runtime ABI: its input is validated MIR.

mod image;
mod stencil;
mod x86_64;

pub use stencil::{Assembler, Hole, Label, Patch, PatchKind, Stencil};

#[derive(Debug)]
pub enum Error {
    Unsupported(String),
    InvalidStencil(&'static str),
    UndefinedLabel,
    BranchOutOfRange,
    Object(object::write::Error),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unsupported(message) => write!(f, "unsupported by fast JIT: {message}"),
            Self::InvalidStencil(message) => write!(f, "invalid stencil: {message}"),
            Self::UndefinedLabel => write!(f, "undefined code label"),
            Self::BranchOutOfRange => write!(f, "branch displacement out of range"),
            Self::Object(error) => write!(f, "object generation: {error}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<object::write::Error> for Error {
    fn from(error: object::write::Error) -> Self {
        Self::Object(error)
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Compile a MIR module for the host's x86-64 System V ABI.
/// Unsupported operations are reported so another tier can be selected.
pub fn compile_object(module: &veloc_mir::Module) -> Result<Vec<u8>> {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    {
        image::compile::<x86_64::Target>(module)
    }
    #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
    {
        let _ = module;
        Err(Error::Unsupported("host architecture".into()))
    }
}
