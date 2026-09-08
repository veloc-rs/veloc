//! Bidirectional, statically dispatched codecs for text atoms.
//!
//! Codec identity describes notation, not just the stored Rust type: both
//! IntegerBits and FloatBits store u64, but have different textual contracts.
use super::lexer::Cursor;
use super::parser::{self, OperandParser, ParseError};
use super::printer::InstPrinter;
use crate::{BlockCall, FloatCC, FuncId, IntCC, Intrinsic, SigId, StackSlot, Type, Value};
use alloc::vec::Vec;
use core::{fmt, marker::PhantomData, str::FromStr};

pub(super) trait AtomCodec {
    type Owned;
    type View<'a>: ?Sized;

    fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>)
    -> Result<Self::Owned, ParseError>;
    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Self::View<'_>,
        ty: Option<Type>,
    ) -> fmt::Result;
}

pub(super) struct Decimal<T>(PhantomData<T>);
pub(super) struct IntegerBits;
pub(super) struct FloatBits;
pub(super) struct Bytes;
pub(super) struct Values;
pub(super) struct Successors;
pub(super) struct FunctionName;

impl<T: FromStr + fmt::Display> AtomCodec for Decimal<T> {
    type Owned = T;
    type View<'a> = T;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<T, ParseError> {
        input.atom(|text| {
            text.parse()
                .map_err(|_| format!("invalid numeric value `{text}`"))
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &T,
        _: Option<Type>,
    ) -> fmt::Result {
        write!(out, "{value}")
    }
}

impl AtomCodec for IntegerBits {
    type Owned = u64;
    type View<'a> = u64;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<u64, ParseError> {
        input.atom(|text| {
            if let Some(hex) = text.strip_prefix("0x") {
                u64::from_str_radix(hex, 16)
                    .map_err(|_| format!("invalid integer constant `{text}`"))
            } else {
                text.parse::<i64>()
                    .map(|value| value as u64)
                    .map_err(|_| format!("invalid integer constant `{text}`"))
            }
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &u64,
        _: Option<Type>,
    ) -> fmt::Result {
        write!(out, "{}", *value as i64)
    }
}

impl AtomCodec for FloatBits {
    type Owned = u64;
    type View<'a> = u64;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<u64, ParseError> {
        input.atom(|text| {
            let hex = text
                .strip_prefix("0x")
                .ok_or("floating constants use an exact hexadecimal bit pattern")?;
            u64::from_str_radix(hex, 16)
                .map_err(|_| format!("invalid floating bit pattern `{text}`"))
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &u64,
        ty: Option<Type>,
    ) -> fmt::Result {
        match ty {
            Some(Type::F32) if u32::try_from(*value).is_ok() => write!(out, "0x{value:08x}"),
            Some(Type::F64) => write!(out, "0x{value:016x}"),
            _ => Err(fmt::Error),
        }
    }
}

impl AtomCodec for Bytes {
    type Owned = Vec<u8>;
    type View<'a> = [u8];

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<Vec<u8>, ParseError> {
        input.atom(|text| {
            let hex = text
                .strip_prefix("0x")
                .ok_or_else(|| format!("expected hexadecimal bytes, found `{text}`"))?;
            if !hex.is_ascii() {
                return Err(format!("invalid hexadecimal bytes `{text}`"));
            }
            if hex.len() % 2 != 0 {
                return Err("hex byte strings must contain an even number of digits".into());
            }
            (0..hex.len())
                .step_by(2)
                .map(|index| {
                    u8::from_str_radix(&hex[index..index + 2], 16)
                        .map_err(|_| format!("invalid hexadecimal bytes `{text}`"))
                })
                .collect()
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &[u8],
        _: Option<Type>,
    ) -> fmt::Result {
        out.write_str("0x")?;
        for byte in value {
            write!(out, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl AtomCodec for bool {
    type Owned = bool;
    type View<'a> = bool;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<bool, ParseError> {
        input.atom(|text| match text {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(format!("expected `true` or `false`, found `{other}`")),
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &bool,
        _: Option<Type>,
    ) -> fmt::Result {
        write!(out, "{value}")
    }
}

impl AtomCodec for IntCC {
    type Owned = IntCC;
    type View<'a> = IntCC;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<IntCC, ParseError> {
        input.atom(|cc| {
            IntCC::from_mnemonic(cc).ok_or_else(|| format!("unknown integer condition `{cc}`"))
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &IntCC,
        _: Option<Type>,
    ) -> fmt::Result {
        write!(out, "{value}")
    }
}

impl AtomCodec for FloatCC {
    type Owned = FloatCC;
    type View<'a> = FloatCC;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<FloatCC, ParseError> {
        input.atom(|cc| {
            FloatCC::from_mnemonic(cc).ok_or_else(|| format!("unknown float condition `{cc}`"))
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &FloatCC,
        _: Option<Type>,
    ) -> fmt::Result {
        write!(out, "{value}")
    }
}

impl AtomCodec for Intrinsic {
    type Owned = Intrinsic;
    type View<'a> = Intrinsic;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<Intrinsic, ParseError> {
        input.atom(|name| {
            Intrinsic::from_name(name).ok_or_else(|| format!("unknown intrinsic `{name}`"))
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Intrinsic,
        _: Option<Type>,
    ) -> fmt::Result {
        out.write_str(value.name())
    }
}

impl AtomCodec for StackSlot {
    type Owned = StackSlot;
    type View<'a> = StackSlot;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<StackSlot, ParseError> {
        input.atom(parser::parse_stack_slot_ref)
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &StackSlot,
        _: Option<Type>,
    ) -> fmt::Result {
        write!(out, "{value}")
    }
}

impl AtomCodec for Value {
    type Owned = Value;
    type View<'a> = Value;

    fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<Value, ParseError> {
        cx.value(input)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Value,
        _: Option<Type>,
    ) -> fmt::Result {
        write!(out, "{}", cx.vf(*value))
    }
}

impl AtomCodec for Values {
    type Owned = Vec<Value>;
    type View<'a> = [Value];

    fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<Vec<Value>, ParseError> {
        cx.values(input)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &[Value],
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_values(out, value)
    }
}

impl AtomCodec for BlockCall {
    type Owned = BlockCall;
    type View<'a> = crate::Successor<'a>;

    fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<BlockCall, ParseError> {
        cx.block_call(input)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &crate::Successor<'_>,
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_block_call(out, *value)
    }
}

impl AtomCodec for Successors {
    type Owned = Vec<BlockCall>;
    type View<'a> = crate::Successors<'a>;

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
    ) -> Result<Vec<BlockCall>, ParseError> {
        cx.block_calls(input)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &crate::Successors<'_>,
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_block_calls(out, *value)
    }
}

impl AtomCodec for FunctionName {
    type Owned = parser::FunctionName;
    type View<'a> = FuncId;

    fn parse(_: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<Self::Owned, ParseError> {
        parser::parse_function_name(input, true)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &FuncId,
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_func_ref(out, *value)
    }
}

impl AtomCodec for FuncId {
    type Owned = FuncId;
    type View<'a> = FuncId;

    fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<FuncId, ParseError> {
        cx.func_ref(input)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &FuncId,
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_func_ref(out, *value)?;
        out.write_str(" : ")?;
        cx.fmt_function_signature(out, *value)
    }
}

impl AtomCodec for SigId {
    type Owned = SigId;
    type View<'a> = SigId;

    fn parse(cx: &mut OperandParser<'_>, input: &mut Cursor<'_>) -> Result<SigId, ParseError> {
        cx.signature(input)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &SigId,
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_signature_ref(out, *value)
    }
}
