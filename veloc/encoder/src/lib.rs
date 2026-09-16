//! Instruction bytes without IR, allocation, symbols or code-layout policy.
#![no_std]
extern crate self as veloc_encoder;

pub mod x86_64;

/// A signed PC-relative field. Its target identity belongs to the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fixup {
    pub offset: u8,
    pub bytes: u8,
    pub base: u8,
    pub addend: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    Register,
    Address,
    Immediate,
    Encoding,
    TooLong,
}
impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Register => "register cannot be represented by this encoding",
            Self::Address => "invalid address or displacement out of range",
            Self::Immediate => "immediate out of range",
            Self::Encoding => "invalid encoding descriptor",
            Self::TooLong => "instruction exceeds architectural length limit",
        })
    }
}
impl core::error::Error for Error {}

/// Inline storage with an architecture-selected capacity, not a heap allocation.
#[derive(Debug, Clone)]
pub struct Encoded<const N: usize> {
    data: [u8; N],
    len: u8,
    pub fixup: Option<Fixup>,
}
impl<const N: usize> Default for Encoded<N> {
    fn default() -> Self {
        Self {
            data: [0; N],
            len: 0,
            fixup: None,
        }
    }
}
impl<const N: usize> Encoded<N> {
    pub fn bytes(&self) -> &[u8] {
        &self.data[..usize::from(self.len)]
    }
    pub(crate) fn push(&mut self, byte: u8) -> Result<(), Error> {
        if usize::from(self.len) == N || self.len == u8::MAX {
            return Err(Error::TooLong);
        }
        self.data[usize::from(self.len)] = byte;
        self.len += 1;
        Ok(())
    }
    pub(crate) fn extend(&mut self, bytes: &[u8]) -> Result<(), Error> {
        for &byte in bytes {
            self.push(byte)?;
        }
        Ok(())
    }
}
