//! Field codecs separate semantic values from their byte representation.
use crate::{OperandRef, Reader, Words};
use core::{fmt::Debug, marker::PhantomData};

/// One instruction field, possibly borrowing part of the bytecode.
pub trait FieldCodec<'a> {
    type Value: Copy + Debug;

    fn read(reader: &mut Reader<'a>) -> Self::Value;
    fn write(value: Self::Value, out: &mut impl Extend<u8>);
    fn size(value: Self::Value) -> usize;
}

/// Semantic interpretation of one word, shared by scalar and list fields.
pub trait WordCodec: Copy + Debug {
    type Value: Copy + Debug;

    fn decode(word: usize) -> Self::Value;
    fn encode(value: Self::Value) -> usize;
}

/// Build-time binding when the host value is a symbolic constant.
#[derive(Clone, Copy, Debug)]
pub struct RawWord;

impl WordCodec for RawWord {
    type Value = usize;

    fn decode(word: usize) -> usize {
        word
    }
    fn encode(value: usize) -> usize {
        value
    }
}

pub struct Word<C>(PhantomData<C>);

impl<'a, C: WordCodec> FieldCodec<'a> for Word<C> {
    type Value = C::Value;

    fn read(reader: &mut Reader<'a>) -> Self::Value {
        C::decode(reader.u32())
    }
    fn write(value: Self::Value, out: &mut impl Extend<u8>) {
        out.extend(crate::encode_u32(C::encode(value)));
    }
    fn size(_: Self::Value) -> usize {
        4
    }
}

pub struct Operand;

impl<'a> FieldCodec<'a> for Operand {
    type Value = OperandRef;

    fn read(reader: &mut Reader<'a>) -> OperandRef {
        OperandRef::decode(reader.uleb())
    }
    fn write(value: OperandRef, out: &mut impl Extend<u8>) {
        out.extend(crate::encode_uleb(value.encode()));
    }
    fn size(value: OperandRef) -> usize {
        crate::encode_uleb(value.encode()).len()
    }
}

/// A borrowed word sequence with a semantic interpretation for each element.
#[derive(Clone, Copy)]
pub struct Values<'a, C: WordCodec> {
    words: Words<'a>,
    codec: PhantomData<C>,
}

impl<C: WordCodec> Debug for Values<'_, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, C: WordCodec> Values<'a, C> {
    pub fn from_words(words: Words<'a>) -> Self {
        Self {
            words,
            codec: PhantomData,
        }
    }
    pub fn len(self) -> usize {
        self.words.len()
    }
    pub fn is_empty(self) -> bool {
        self.words.is_empty()
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = C::Value> {
        self.words.iter().map(C::decode)
    }
}

/// A length-prefixed list using the same word codec as a scalar field.
pub struct List<C>(PhantomData<C>);

impl<'a, C: WordCodec> FieldCodec<'a> for List<C> {
    type Value = Values<'a, C>;

    fn read(reader: &mut Reader<'a>) -> Self::Value {
        Values::from_words(reader.words())
    }
    fn write(value: Self::Value, out: &mut impl Extend<u8>) {
        out.extend(crate::encode_u32(value.words.len()));
        for word in value.words.iter() {
            out.extend(crate::encode_u32(word));
        }
    }
    fn size(value: Self::Value) -> usize {
        4 + value.words.len() * 4
    }
}
