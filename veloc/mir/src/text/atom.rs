//! Bidirectional, statically dispatched codecs for text atoms.
//!
//! Codec identity describes notation, not just the stored Rust type: both
//! IntegerBits and FloatBits store u64, but have different textual contracts.
use super::parser::{self, OperandParser, ParseError};
use super::printer::InstPrinter;
use crate::{BlockCall, FloatCC, FuncId, IntCC, Intrinsic, SigId, StackSlot, Type, Value};
use alloc::vec::Vec;
use core::{borrow::Borrow, fmt, marker::PhantomData, str::FromStr};

pub(super) trait AtomCodec {
    type Owned: Borrow<Self::View>;
    type View: ?Sized;

    fn parse(
        cx: &mut OperandParser<'_>,
        text: &str,
        ty: Option<Type>,
    ) -> Result<Self::Owned, ParseError>;
    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Self::View,
        ty: Option<Type>,
    ) -> fmt::Result;
}

pub(super) struct Decimal<T>(PhantomData<T>);
pub(super) struct IntegerBits;
pub(super) struct FloatBits;
pub(super) struct Bytes;
pub(super) struct Values;
pub(super) struct Successors;

impl<T: FromStr + fmt::Display> AtomCodec for Decimal<T> {
    type Owned = T;
    type View = T;

    fn parse(_: &mut OperandParser<'_>, text: &str, _: Option<Type>) -> Result<T, ParseError> {
        text.trim()
            .parse()
            .map_err(|_| ParseError(format!("invalid numeric value `{text}`")))
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
    type View = u64;

    fn parse(_: &mut OperandParser<'_>, text: &str, _: Option<Type>) -> Result<u64, ParseError> {
        let text = text.trim();
        if let Some(hex) = text.strip_prefix("0x") {
            u64::from_str_radix(hex, 16)
                .map_err(|_| ParseError(format!("invalid integer constant `{text}`")))
        } else {
            text.parse::<i64>()
                .map(|value| value as u64)
                .map_err(|_| ParseError(format!("invalid integer constant `{text}`")))
        }
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
    type View = u64;

    fn parse(_: &mut OperandParser<'_>, text: &str, ty: Option<Type>) -> Result<u64, ParseError> {
        if !matches!(ty, Some(Type::F32 | Type::F64)) {
            return Err(ParseError(
                "floating constants require an `f32` or `f64` result suffix".into(),
            ));
        }
        let text = text.trim();
        let hex = text.strip_prefix("0x").ok_or_else(|| {
            ParseError("floating constants use an exact hexadecimal bit pattern".into())
        })?;
        let bits = u64::from_str_radix(hex, 16)
            .map_err(|_| ParseError(format!("invalid floating bit pattern `{text}`")))?;
        if ty == Some(Type::F32) && bits > u64::from(u32::MAX) {
            return Err(ParseError(format!(
                "f32 bit pattern does not fit in 32 bits: `{text}`"
            )));
        }
        Ok(bits)
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
    type View = [u8];

    fn parse(
        _: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<Vec<u8>, ParseError> {
        let text = text.trim();
        let hex = text
            .strip_prefix("0x")
            .ok_or_else(|| ParseError(format!("expected hexadecimal bytes, found `{text}`")))?;
        if !hex.is_ascii() {
            return Err(ParseError(format!("invalid hexadecimal bytes `{text}`")));
        }
        if hex.len() % 2 != 0 {
            return Err(ParseError(
                "hex byte strings must contain an even number of digits".into(),
            ));
        }
        (0..hex.len())
            .step_by(2)
            .map(|index| {
                u8::from_str_radix(&hex[index..index + 2], 16)
                    .map_err(|_| ParseError(format!("invalid hexadecimal bytes `{text}`")))
            })
            .collect()
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
    type View = bool;

    fn parse(_: &mut OperandParser<'_>, text: &str, _: Option<Type>) -> Result<bool, ParseError> {
        match text.trim() {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(ParseError(format!(
                "expected `true` or `false`, found `{other}`"
            ))),
        }
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
    type View = IntCC;

    fn parse(_: &mut OperandParser<'_>, text: &str, _: Option<Type>) -> Result<IntCC, ParseError> {
        let cc = text.trim();
        IntCC::from_mnemonic(cc)
            .ok_or_else(|| ParseError(format!("unknown integer condition `{cc}`")))
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
    type View = FloatCC;

    fn parse(
        _: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<FloatCC, ParseError> {
        let cc = text.trim();
        FloatCC::from_mnemonic(cc)
            .ok_or_else(|| ParseError(format!("unknown float condition `{cc}`")))
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
    type View = Intrinsic;

    fn parse(
        _: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<Intrinsic, ParseError> {
        let name = text.trim();
        Intrinsic::from_name(name).ok_or_else(|| ParseError(format!("unknown intrinsic `{name}`")))
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
    type View = StackSlot;

    fn parse(
        _: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<StackSlot, ParseError> {
        parser::parse_stack_slot_ref(text)
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
    type View = Value;

    fn parse(cx: &mut OperandParser<'_>, text: &str, _: Option<Type>) -> Result<Value, ParseError> {
        cx.value(text)
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
    type View = [Value];

    fn parse(
        cx: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<Vec<Value>, ParseError> {
        cx.values(text)
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
    type View = BlockCall;

    fn parse(
        cx: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<BlockCall, ParseError> {
        cx.block_call(text)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &BlockCall,
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_block_call(out, *value)
    }
}

impl AtomCodec for Successors {
    type Owned = Vec<BlockCall>;
    type View = [BlockCall];

    fn parse(
        cx: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<Vec<BlockCall>, ParseError> {
        cx.block_calls(text)
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &[BlockCall],
        _: Option<Type>,
    ) -> fmt::Result {
        cx.fmt_block_calls(out, value)
    }
}

impl AtomCodec for FuncId {
    type Owned = FuncId;
    type View = FuncId;

    fn parse(
        cx: &mut OperandParser<'_>,
        text: &str,
        _: Option<Type>,
    ) -> Result<FuncId, ParseError> {
        cx.func_ref(text)
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

impl AtomCodec for SigId {
    type Owned = SigId;
    type View = SigId;

    fn parse(cx: &mut OperandParser<'_>, text: &str, _: Option<Type>) -> Result<SigId, ParseError> {
        cx.signature(text)
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
