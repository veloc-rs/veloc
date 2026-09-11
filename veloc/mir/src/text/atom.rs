//! Bidirectional, statically dispatched codecs for text atoms.
//!
//! Typed properties select their own notation. IntegerBits is an explicit
//! projection for untyped integer payloads; Float preserves precision and bits.
use super::lexer::{Cursor, Kind};
use super::parser::{self, OperandParser, ParseError};
use super::printer::InstPrinter;
use crate::{
    BlockCall, Float, FloatCC, FuncId, Int, IntCC, Intrinsic, ScalarConst, ScalarType, SigId, Type,
    Value, VectorConst,
};
use alloc::vec::Vec;
use core::{fmt, marker::PhantomData, str::FromStr};

pub(super) trait AtomCodec {
    type Owned;
    type View<'a>: ?Sized;

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<Self::Owned, ParseError>;
    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Self::View<'_>,
        ty: Option<Type>,
    ) -> fmt::Result;
}

pub(super) struct Decimal<T>(PhantomData<T>);
pub(super) struct IntegerBits;
pub(super) struct Bytes;
pub(super) struct Values;
pub(super) struct Successors;
pub(super) struct FunctionName;

impl<T: FromStr + fmt::Display> AtomCodec for Decimal<T> {
    type Owned = T;
    type View<'a> = T;

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<T, ParseError> {
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

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<u64, ParseError> {
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

impl AtomCodec for Float {
    type Owned = Self;
    type View<'a> = Self;

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        ty: Option<Type>,
    ) -> Result<Self, ParseError> {
        // The explicit result annotation supplies precision, not a numeric
        // conversion. Reject unrepresentable payloads instead of truncating.
        let ty = ty
            .filter(|ty| matches!(*ty, Type::F32 | Type::F64))
            .ok_or_else(|| input.error("floating constant requires an f32 or f64 result type"))?;
        input.atom(|text| {
            let hex = text
                .strip_prefix("0x")
                .ok_or("floating constants use an exact hexadecimal bit pattern")?;
            let bits = u64::from_str_radix(hex, 16)
                .map_err(|_| format!("invalid floating bit pattern `{text}`"))?;
            if ty == Type::F32 {
                u32::try_from(bits)
                    .map(Self::from_f32_bits)
                    .map_err(|_| "f32 bit pattern does not fit in 32 bits".into())
            } else {
                Ok(Self::from_f64_bits(bits))
            }
        })
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Self,
        ty: Option<Type>,
    ) -> fmt::Result {
        if ty.is_some_and(|ty| ty != value.ty()) {
            return Err(fmt::Error);
        }
        let bits = value.to_bits();
        if value.ty() == Type::F32 {
            write!(out, "0x{bits:08x}")
        } else {
            write!(out, "0x{bits:016x}")
        }
    }
}

impl AtomCodec for Int {
    type Owned = Self;
    type View<'a> = Self;

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        ty: Option<Type>,
    ) -> Result<Self, ParseError> {
        let ty = ty
            .filter(|ty| ty.is_integer() && ty.as_scalar().is_some())
            .ok_or_else(|| input.error("integer constant requires a scalar integer result type"))?;
        let bits = IntegerBits::parse(cx, input, None)?;
        Ok(Int::from_bits(ty, bits).expect("checked integer type"))
    }

    fn print(
        _: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Self,
        ty: Option<Type>,
    ) -> fmt::Result {
        if ty.is_some_and(|ty| ty != value.ty()) {
            return Err(fmt::Error);
        }
        write!(out, "{}", value.signed())
    }
}

impl AtomCodec for VectorConst {
    type Owned = Self;
    type View<'a> = Self;

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        ty: Option<Type>,
    ) -> Result<Self, ParseError> {
        let vector = ty
            .and_then(Type::as_vector)
            .ok_or_else(|| input.error("vector constant requires a vector result type"))?;
        if !input.peek_is(0, "splat") {
            let bytes = Bytes::parse(cx, input, None)?;
            return Ok(cx.dense_constant(vector, bytes));
        }
        input.keyword("splat")?;
        input.expect(Kind::LParen)?;
        let element = vector.element_type();
        let ty = Some(element.as_type());
        let lane: ScalarConst = match element {
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 => {
                Int::parse(cx, input, ty)?.into()
            }
            ScalarType::F32 | ScalarType::F64 => Float::parse(cx, input, ty)?.into(),
            ScalarType::BOOL => bool::parse(cx, input, ty)?.into(),
            ScalarType::PTR => unreachable!("pointer vector types are not representable"),
        };
        input.expect(Kind::RParen)?;
        Ok(Self::splat(lane, vector.lane_count(), vector.is_scalable())
            .expect("parsed vector shape"))
    }

    fn print(
        cx: &InstPrinter<'_>,
        out: &mut dyn fmt::Write,
        value: &Self,
        ty: Option<Type>,
    ) -> fmt::Result {
        if ty.is_some_and(|ty| ty != value.ty()) {
            return Err(fmt::Error);
        }
        if value.is_dense() {
            return Bytes::print(cx, out, value.bytes(cx.dfg).ok_or(fmt::Error)?, None);
        }
        let scalar = value.splat_value().expect("splat constant");
        out.write_str("splat(")?;
        if let Some(value) = scalar.as_int() {
            Int::print(cx, out, &value, None)?;
        } else if let Some(value) = scalar.as_float() {
            Float::print(cx, out, &value, None)?;
        } else {
            bool::print(cx, out, &scalar.as_bool().expect("boolean constant"), None)?;
        }
        out.write_char(')')
    }
}

impl AtomCodec for Bytes {
    type Owned = Vec<u8>;
    type View<'a> = [u8];

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<Vec<u8>, ParseError> {
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

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<bool, ParseError> {
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

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<IntCC, ParseError> {
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

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<FloatCC, ParseError> {
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

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<Intrinsic, ParseError> {
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

impl AtomCodec for Value {
    type Owned = Value;
    type View<'a> = Value;

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<Value, ParseError> {
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

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<Vec<Value>, ParseError> {
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

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<BlockCall, ParseError> {
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
        _: Option<Type>,
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

    fn parse(
        _: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<Self::Owned, ParseError> {
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

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<FuncId, ParseError> {
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

    fn parse(
        cx: &mut OperandParser<'_>,
        input: &mut Cursor<'_>,
        _: Option<Type>,
    ) -> Result<SigId, ParseError> {
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
