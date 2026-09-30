//! Shared rule bytecode formats and encoding primitives. No IR dependencies.
#![no_std]

mod operand;
pub use operand::OperandRef;
pub mod codec;

pub mod equivalence;
pub mod rewrite;
pub mod selection;
pub mod signature;

/// Build-time encoding contract shared by all bytecode dialects.
pub trait Encode {
    fn encode(&self, out: &mut impl Extend<u8>);
    fn field_offset(&self, name: &str) -> Option<usize>;
}

/// Describe a bytecode once for its compiler, interpreter and disassembler.
/// Fields choose fixed little-endian u32 or ULEB128 encoding. Lists use the
/// same encoding for their length prefix and elements. `(codec C)` binds a
/// semantic field type and its encoding through `FieldCodec`.
/// Optional generic parameters bind host codecs without duplicating the schema.
#[macro_export]
macro_rules! bytecode {
    ($vis:vis enum $inst:ident $(<$($param:ident : $bound:path),+>)?, $opcode:ident {
        $($name:ident { $($field:ident : $kind:tt),* $(,)? }),* $(,)?
    }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        #[repr(u8)]
        $vis enum $opcode { $($name),* }

        $vis enum $inst<'a $(, $($param: $bound),+)?> {
            $($name { $($field: $crate::bytecode!(@ty $kind, 'a)),* }),*
        }

        // Generic enums containing type macros cannot use built-in derives.
        // FieldCodec requires Copy + Debug, so all fields support these impls.
        impl<'a $(, $($param: $bound),+)?> Copy for $inst<'a $(, $($param),+)?> {}
        impl<'a $(, $($param: $bound),+)?> Clone for $inst<'a $(, $($param),+)?> {
            fn clone(&self) -> Self { *self }
        }
        impl<'a $(, $($param: $bound),+)?> core::fmt::Debug for $inst<'a $(, $($param),+)?> {
            #[allow(unused_mut)]
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self {
                    $(Self::$name { $($field),* } => {
                        let mut debug = f.debug_struct(stringify!($name));
                        $(debug.field(stringify!($field), $field);)*
                        debug.finish()
                    }),*
                }
            }
        }

        impl<'a $(, $($param: $bound),+)?> $crate::Encode for $inst<'a $(, $($param),+)?> {
            fn encode(&self, out: &mut impl Extend<u8>) { self.encode(out); }
            fn field_offset(&self, name: &str) -> Option<usize> { self.field_offset(name) }
        }

        impl<'a $(, $($param: $bound),+)?> $inst<'a $(, $($param),+)?> {
            #[inline]
            $vis fn read(reader: &mut $crate::Reader<'a>) -> Self {
                match reader.byte() {
                    $(x if x == $opcode::$name as u8 => Self::$name {
                        $($field: $crate::bytecode!(@read reader, $kind)),*
                    },)*
                    byte => panic!("invalid {} opcode {}", stringify!($opcode), byte),
                }
            }

            $vis fn opcode(&self) -> $opcode {
                match self { $(Self::$name { .. } => $opcode::$name),* }
            }

            $vis fn encode(&self, out: &mut impl Extend<u8>) {
                out.extend(core::iter::once(self.opcode() as u8));
                match self {
                    $(Self::$name { $($field),* } => {
                        $($crate::bytecode!(@encode out, $field, $kind);)*
                    }),*
                }
            }

            /// Byte offset from the opcode, for build-time relocations.
            #[allow(unused_variables, unused_mut, unused_assignments)]
            $vis fn field_offset(&self, name: &str) -> Option<usize> {
                let mut offset = 1;
                match self {
                    $(Self::$name { $($field),* } => {
                        $(if name == stringify!($field) { return Some(offset); }
                        offset += $crate::bytecode!(@size $field, $kind);)*
                    }),*
                }
                None
            }
        }
    };
    (@ty u32, $lt:lifetime) => { usize };
    (@ty uleb, $lt:lifetime) => { usize };
    (@ty [uleb], $lt:lifetime) => { $crate::Lebs<$lt> };
    (@ty [u32], $lt:lifetime) => { $crate::Words<$lt> };
    (@ty (codec $codec:ty), $lt:lifetime) => { <$codec as $crate::codec::FieldCodec<$lt>>::Value };
    (@read $r:ident, u32) => { $r.u32() };
    (@read $r:ident, uleb) => { $r.uleb() };
    (@read $r:ident, [uleb]) => { $r.lebs() };
    (@read $r:ident, [u32]) => { $r.words() };
    (@read $r:ident, (codec $codec:ty)) => { <$codec as $crate::codec::FieldCodec<'_>>::read($r) };
    (@encode $out:ident, $v:ident, u32) => { $out.extend($crate::encode_u32(*$v)) };
    (@encode $out:ident, $v:ident, uleb) => { $out.extend($crate::encode_uleb(*$v)) };
    (@encode $out:ident, $v:ident, (codec $codec:ty)) => {
        <$codec as $crate::codec::FieldCodec<'_>>::write(*$v, $out)
    };
    (@encode $out:ident, $v:ident, [uleb]) => {
        $out.extend($crate::encode_uleb($v.len()));
        for value in $v.iter() { $out.extend($crate::encode_uleb(value)); }
    };
    (@encode $out:ident, $v:ident, [u32]) => {
        $out.extend($crate::encode_u32($v.len()));
        for value in $v.iter() { $out.extend($crate::encode_u32(value)); }
    };
    (@size $v:ident, u32) => { 4 };
    (@size $v:ident, uleb) => { $crate::encode_uleb(*$v).len() };
    (@size $v:ident, [uleb]) => {
        $crate::encode_uleb($v.len()).len()
            + $v.iter().map(|v| $crate::encode_uleb(v).len()).sum::<usize>()
    };
    (@size $v:ident, [u32]) => { 4 + $v.len() * 4 };
    (@size $v:ident, (codec $codec:ty)) => { <$codec as $crate::codec::FieldCodec<'_>>::size(*$v) };
}

/// Borrowed list used both by the encoder and by zero-allocation decoding.
#[derive(Clone, Copy)]
pub enum Words<'a> {
    Values(&'a [usize]),
    Encoded(&'a [u8]),
}

impl Words<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::Values(values) => values.len(),
            Self::Encoded(bytes) => bytes.len() / 4,
        }
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    pub fn split_at(self, index: usize) -> (Self, Self) {
        assert!(index <= self.len(), "bytecode word range");
        match self {
            Self::Values(values) => {
                let (head, tail) = values.split_at(index);
                (Self::Values(head), Self::Values(tail))
            }
            Self::Encoded(bytes) => {
                let (head, tail) = bytes.split_at(index * 4);
                (Self::Encoded(head), Self::Encoded(tail))
            }
        }
    }

    pub fn iter(self) -> impl ExactSizeIterator<Item = usize> + DoubleEndedIterator {
        (0..self.len()).map(move |i| match self {
            Self::Values(values) => values[i],
            Self::Encoded(bytes) => {
                u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap()) as usize
            }
        })
    }
}

impl core::fmt::Debug for Words<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

/// Reader for trusted build output. Malformed programs are internal errors.
pub struct Reader<'a> {
    pub bytes: &'a [u8],
    pub pc: usize,
}

impl<'a> Reader<'a> {
    #[inline]
    pub fn lebs(&mut self) -> Lebs<'a> {
        let len = self.uleb();
        let start = self.pc;
        for _ in 0..len {
            self.uleb();
        }
        Lebs::Encoded {
            bytes: &self.bytes[start..self.pc],
            len,
        }
    }
    #[inline]
    pub fn words(&mut self) -> Words<'a> {
        let len = self.u32().checked_mul(4).expect("bytecode list overflow");
        let bytes = &self.bytes[self.pc..self.pc + len];
        self.pc += len;
        Words::Encoded(bytes)
    }
    #[inline]
    pub fn byte(&mut self) -> u8 {
        let byte = self.bytes[self.pc];
        self.pc += 1;
        byte
    }

    #[inline]
    pub fn u32(&mut self) -> usize {
        let bytes = self.bytes[self.pc..self.pc + 4].try_into().unwrap();
        self.pc += 4;
        u32::from_le_bytes(bytes) as usize
    }

    #[inline]
    pub fn uleb(&mut self) -> usize {
        let mut value = 0u32;
        for shift in (0..35).step_by(7) {
            let byte = self.byte();
            assert!(shift != 28 || byte & 0xf0 == 0, "bytecode index overflow");
            value |= u32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return value as usize;
            }
        }
        unreachable!()
    }
}

/// Borrowed variable-length operands, without allocating a decoded array.
#[derive(Clone, Copy)]
pub enum Lebs<'a> {
    Values(&'a [usize]),
    Encoded { bytes: &'a [u8], len: usize },
}

impl Lebs<'_> {
    pub fn len(self) -> usize {
        match self {
            Self::Values(v) => v.len(),
            Self::Encoded { len, .. } => len,
        }
    }
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    pub fn iter(self) -> impl ExactSizeIterator<Item = usize> {
        let mut pc = 0;
        (0..self.len()).map(move |i| match self {
            Self::Values(values) => values[i],
            Self::Encoded { bytes, .. } => {
                let mut reader = Reader { bytes, pc };
                let value = reader.uleb();
                pc = reader.pc;
                value
            }
        })
    }
}

impl core::fmt::Debug for Lebs<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

pub fn encode_u32(value: usize) -> [u8; 4] {
    u32::try_from(value)
        .expect("bytecode index overflow")
        .to_le_bytes()
}

/// Allocation-free encoding, usable both to measure and to emit an operand.
pub fn encode_uleb(value: usize) -> impl ExactSizeIterator<Item = u8> {
    let mut value = u32::try_from(value).expect("bytecode index overflow");
    let mut bytes = [0; 5];
    let mut len = 0;
    loop {
        let byte = (value & 0x7f) as u8;
        value >>= 7;
        bytes[len] = byte | if value == 0 { 0 } else { 0x80 };
        len += 1;
        if value == 0 {
            return bytes.into_iter().take(len);
        }
    }
}
