//! A fast baseline compiler built from small, pre-encoded machine-code stencils.
//!
//! This crate owns code selection, stencil patching, and object production. It
//! does not own Wasm semantics or the runtime ABI: its input is validated MIR.

#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
mod image;
mod stencil;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
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
    compile_object_with_profile(module, &veloc_profile::Profile::default())
}

pub fn compile_object_with_profile(
    module: &veloc_mir::Module,
    profile: &veloc_profile::Profile,
) -> Result<Vec<u8>> {
    profile.measure("fastjit", 0, || {
        #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
        {
            image::compile::<x86_64::Target>(module, profile)
        }
        #[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
        {
            let _ = module;
            Err(Error::Unsupported("host architecture".into()))
        }
    })
}
