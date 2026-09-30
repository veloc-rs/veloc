//! Shared inline type patterns and borrowed signature decoding.
//! Independent of the instruction set and the host type representation.

use crate::codec::{FieldCodec, List, RawWord, Values, WordCodec};
use crate::{Reader, Words};
use core::marker::PhantomData;

/// A signature visits results before inputs. A pattern starts with one tagged
/// u32 word. Set/Bind carry a count followed by that many inline type-code words;
/// Exact carries the code itself. Each Bind introduces the next local slot;
/// Same refers to a slot already bound in this signature match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatternHeader {
    Set(usize),
    Bind(usize),
    Same(usize),
    Exact(usize),
}

impl PatternHeader {
    pub const fn encode(self) -> usize {
        let (index, tag) = match self {
            Self::Set(count) => (count, 0),
            Self::Bind(count) => (count, 1),
            Self::Same(slot) => (slot, 2),
            Self::Exact(code) => (code, 3),
        };
        assert!(
            index <= (u32::MAX >> 2) as usize,
            "type pattern payload overflow"
        );
        (index << 2) | tag
    }

    pub fn decode(value: usize) -> Self {
        match value & 3 {
            0 => Self::Set(value >> 2),
            1 => Self::Bind(value >> 2),
            2 => Self::Same(value >> 2),
            3 => Self::Exact(value >> 2),
            _ => unreachable!(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum TypePattern<'a, C: WordCodec> {
    Exact(C::Value),
    Set(Values<'a, C>),
    Bind(Values<'a, C>),
    Same(usize),
}

/// A zero-allocation view shared by build-time encoding and runtime decoding.
#[derive(Clone, Copy)]
pub struct TypePatterns<'a, C: WordCodec> {
    words: Words<'a>,
    codec: PhantomData<C>,
}

impl<C: WordCodec> core::fmt::Debug for TypePatterns<'_, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, C: WordCodec> TypePatterns<'a, C> {
    pub fn from_words(words: Words<'a>) -> Self {
        Self {
            words,
            codec: PhantomData,
        }
    }

    pub fn iter(self) -> impl Iterator<Item = TypePattern<'a, C>> {
        let mut words = self.words;
        core::iter::from_fn(move || {
            if words.is_empty() {
                return None;
            }
            let (head, tail) = words.split_at(1);
            words = tail;
            let header = PatternHeader::decode(head.iter().next().unwrap());
            Some(match header {
                PatternHeader::Exact(code) => TypePattern::Exact(C::decode(code)),
                PatternHeader::Same(slot) => TypePattern::Same(slot),
                PatternHeader::Set(count) | PatternHeader::Bind(count) => {
                    let (domain, tail) = words.split_at(count);
                    words = tail;
                    let types = Values::from_words(domain);
                    match header {
                        PatternHeader::Set(_) => TypePattern::Set(types),
                        _ => TypePattern::Bind(types),
                    }
                }
            })
        })
    }
}

pub struct PatternsCodec<C>(PhantomData<C>);

impl<'a, C: WordCodec> FieldCodec<'a> for PatternsCodec<C> {
    type Value = TypePatterns<'a, C>;

    fn read(reader: &mut Reader<'a>) -> Self::Value {
        TypePatterns::from_words(reader.words())
    }
    fn write(value: Self::Value, out: &mut impl Extend<u8>) {
        List::<RawWord>::write(Values::from_words(value.words), out);
    }
    fn size(value: Self::Value) -> usize {
        List::<RawWord>::size(Values::from_words(value.words))
    }
}
