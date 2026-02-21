use crate::{
    module::Linkage,
    opcode::{FloatCC, IntCC, Opcode},
    types::{ScalarType, Type},
};
use alloc::string::{String, ToString};

// ==================== Type Format ====================

pub fn type_to_string(ty: Type) -> String {
    ty.to_string()
}

pub fn parse_type(s: &str) -> Option<Type> {
    match s {
        "i8" => Some(Type::I8),
        "i16" => Some(Type::I16),
        "i32" => Some(Type::I32),
        "i64" => Some(Type::I64),
        "f32" => Some(Type::F32),
        "f64" => Some(Type::F64),
        "bool" => Some(Type::BOOL),
        "ptr" => Some(Type::PTR),
        "void" => Some(Type::VOID),
        _ if s.contains('<') => {
            let lt_pos = s.find('<')?;
            let base = &s[..lt_pos];
            let rest = &s[lt_pos + 1..s.len().checked_sub(1)?];

            let scalar = parse_scalar_type(base)?;
            let (is_scalable, lanes_str) = if rest.starts_with("scalable ") {
                (true, &rest[9..])
            } else {
                (false, rest)
            };

            let lanes: u16 = lanes_str.parse().ok()?;
            Some(Type::new_vector(scalar, lanes, is_scalable))
        }
        _ => None,
    }
}

pub fn parse_scalar_type(s: &str) -> Option<ScalarType> {
    match s {
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f32" => Some(ScalarType::F32),
        "f64" => Some(ScalarType::F64),
        "bool" => Some(ScalarType::Bool),
        "ptr" => Some(ScalarType::Ptr),
        _ => None,
    }
}

// ==================== Condition Code Format ====================

macro_rules! for_all_intcc {
    ($macro:ident) => {
        $macro! {
            Eq => "eq",
            Ne => "ne",
            LtS => "lts",
            LtU => "ltu",
            GtS => "gts",
            GtU => "gtu",
            LeS => "les",
            LeU => "leu",
            GeS => "ges",
            GeU => "geu",
        }
    };
}

macro_rules! for_all_floatcc {
    ($macro:ident) => {
        $macro! {
            Eq => "eq",
            Ne => "ne",
            Lt => "lt",
            Gt => "gt",
            Le => "le",
            Ge => "ge",
        }
    };
}

pub fn intcc_to_string(cc: IntCC) -> &'static str {
    macro_rules! as_str {
        ($($variant:ident => $mnemonic:expr),* $(,)?) => {
            match cc {
                $(IntCC::$variant => $mnemonic,)*
            }
        }
    }
    for_all_intcc!(as_str)
}

pub fn parse_intcc(s: &str) -> Option<IntCC> {
    macro_rules! parse {
        ($($variant:ident => $mnemonic:expr),* $(,)?) => {
            match s {
                $($mnemonic => Some(IntCC::$variant),)*
                _ => None,
            }
        }
    }
    for_all_intcc!(parse)
}

pub fn floatcc_to_string(cc: FloatCC) -> &'static str {
    macro_rules! as_str {
        ($($variant:ident => $mnemonic:expr),* $(,)?) => {
            match cc {
                $(FloatCC::$variant => $mnemonic,)*
            }
        }
    }
    for_all_floatcc!(as_str)
}

pub fn parse_floatcc(s: &str) -> Option<FloatCC> {
    macro_rules! parse {
        ($($variant:ident => $mnemonic:expr),* $(,)?) => {
            match s {
                $($mnemonic => Some(FloatCC::$variant),)*
                _ => None,
            }
        }
    }
    for_all_floatcc!(parse)
}

// ==================== Opcode Format ====================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InstFormat {
    Unary,
    Binary,
    Ternary,
    Iconst,
    Fconst,
    Bconst,
    Vconst,
    Load,
    Store,
    StackLoad,
    StackStore,
    StackAddr,
    PtrOffset,
    PtrIndex,
    IntToPtr,
    PtrToInt,
    Call,
    CallIndirect,
    CallIntrinsic,
    Jump,
    Br,
    BrTable,
    Return,
    IntCompare,
    FloatCompare,
    VectorOpWithExt,
    VectorLoadStrided,
    VectorStoreStrided,
    VectorGather,
    VectorScatter,
    Shuffle,
    Unreachable,
    Nop,
    SetVL,
}

macro_rules! for_all_opcodes {
    ($macro:ident) => {
        $macro! {
            Iconst => ("iconst", Iconst),
            Fconst => ("fconst", Fconst),
            Bconst => ("bconst", Bconst),
            Vconst => ("vconst", Vconst),
            IAdd => ("iadd", Binary),
            ISub => ("isub", Binary),
            IMul => ("imul", Binary),
            INeg => ("ineg", Unary),
            IAddSat => ("iadd-sat", Binary),
            ISubSat => ("isub-sat", Binary),
            IAddWithOverflow => ("iadd-with-overflow", Binary),
            ISubWithOverflow => ("isub-with-overflow", Binary),
            IMulWithOverflow => ("imul-with-overflow", Binary),
            IDivS => ("idiv-s", Binary),
            IDivU => ("idiv-u", Binary),
            IRemS => ("irem-s", Binary),
            IRemU => ("irem-u", Binary),
            FAdd => ("fadd", Binary),
            FSub => ("fsub", Binary),
            FMul => ("fmul", Binary),
            FNeg => ("fneg", Unary),
            FDiv => ("fdiv", Binary),
            FMin => ("fmin", Binary),
            FMax => ("fmax", Binary),
            FCopysign => ("fcopysign", Binary),
            FAbs => ("fabs", Unary),
            FSqrt => ("fsqrt", Unary),
            FCeil => ("fceil", Unary),
            FFloor => ("ffloor", Unary),
            FTrunc => ("ftrunc", Unary),
            FNearest => ("fnearest", Unary),
            IAnd => ("iand", Binary),
            IOr => ("ior", Binary),
            IXor => ("ixor", Binary),
            IShl => ("ishl", Binary),
            IShrS => ("ishr-s", Binary),
            IShrU => ("ishr-u", Binary),
            IRotl => ("irotl", Binary),
            IRotr => ("irotr", Binary),
            IClz => ("iclz", Unary),
            ICtz => ("ictz", Unary),
            IPopcnt => ("ipopcnt", Unary),
            IEqz => ("ieqz", Unary),
            Icmp => ("icmp", IntCompare),
            Fcmp => ("fcmp", FloatCompare),
            ExtendS => ("extends", Unary),
            ExtendU => ("extendu", Unary),
            Wrap => ("wrap", Unary),
            FloatToIntSatS => ("float-to-int-sat-s", Unary),
            FloatToIntSatU => ("float-to-int-sat-u", Unary),
            FloatToIntS => ("float-to-int-s", Unary),
            FloatToIntU => ("float-to-int-u", Unary),
            IntToFloatS => ("int-to-float-s", Unary),
            IntToFloatU => ("int-to-float-u", Unary),
            FloatPromote => ("float-promote", Unary),
            FloatDemote => ("float-demote", Unary),
            Reinterpret => ("reinterpret", Unary),
            IntToPtr => ("inttoptr", IntToPtr),
            PtrToInt => ("ptrtoint", PtrToInt),
            Load => ("load", Load),
            Store => ("store", Store),
            StackLoad => ("stack-load", StackLoad),
            StackStore => ("stack-store", StackStore),
            StackAddr => ("stack-addr", StackAddr),
            PtrOffset => ("ptr-offset", PtrOffset),
            PtrIndex => ("ptr-index", PtrIndex),
            Call => ("call", Call),
            CallIndirect => ("call-indirect", CallIndirect),
            CallIntrinsic => ("call-intrinsic", CallIntrinsic),
            Jump => ("jump", Jump),
            Br => ("br", Br),
            BrTable => ("br-table", BrTable),
            Return => ("return", Return),
            Select => ("select", Ternary),
            Unreachable => ("unreachable", Unreachable),
            Nop => ("nop", Nop),
            Splat => ("splat", VectorOpWithExt),
            Shuffle => ("shuffle", Shuffle),
            InsertElement => ("insertelement", Ternary),
            ExtractElement => ("extractelement", Binary),
            ReduceSum => ("reduce-sum", VectorOpWithExt),
            ReduceAdd => ("reduce-add", VectorOpWithExt),
            ReduceMin => ("reduce-min", VectorOpWithExt),
            ReduceMax => ("reduce-max", VectorOpWithExt),
            ReduceAnd => ("reduce-and", VectorOpWithExt),
            ReduceOr => ("reduce-or", VectorOpWithExt),
            ReduceXor => ("reduce-xor", VectorOpWithExt),
            LoadStride => ("load-stride", VectorLoadStrided),
            StoreStride => ("store-stride", VectorStoreStrided),
            Gather => ("gather", VectorGather),
            Scatter => ("scatter", VectorScatter),
            SetVL => ("setvl", SetVL),
        }
    };
}

pub fn opcode_to_string(op: Opcode) -> &'static str {
    macro_rules! as_str {
        ($($variant:ident => ($mnemonic:expr, $format:ident)),* $(,)?) => {
            match op {
                $(Opcode::$variant => $mnemonic,)*
            }
        }
    }
    for_all_opcodes!(as_str)
}

pub fn parse_opcode(s: &str) -> Option<Opcode> {
    macro_rules! parse {
        ($($variant:ident => ($mnemonic:expr, $format:ident)),* $(,)?) => {
            match s {
                $($mnemonic => Some(Opcode::$variant),)*
                _ => None,
            }
        }
    }
    for_all_opcodes!(parse)
}

pub fn opcode_to_format(op: Opcode) -> InstFormat {
    macro_rules! as_format {
        ($($variant:ident => ($mnemonic:expr, $format:ident)),* $(,)?) => {
            match op {
                $(Opcode::$variant => InstFormat::$format,)*
            }
        }
    }
    for_all_opcodes!(as_format)
}

// ==================== Linkage Format ====================

pub fn linkage_to_string(linkage: Linkage) -> &'static str {
    match linkage {
        Linkage::Local => "local",
        Linkage::Export => "export",
        Linkage::Import => "import",
    }
}

pub fn parse_linkage(s: &str) -> Option<Linkage> {
    match s {
        "local" => Some(Linkage::Local),
        "export" => Some(Linkage::Export),
        "import" => Some(Linkage::Import),
        _ => None,
    }
}

// ==================== Vector Memory Extension Keys ====================

pub const STRIDE: &str = "stride";
pub const INDEX: &str = "index";
pub const SCALE: &str = "scale";
pub const MASK: &str = "mask";
pub const EVL: &str = "evl";
