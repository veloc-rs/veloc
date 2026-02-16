use ::alloc::vec::Vec;
use veloc_ir::Type;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(transparent)]
pub struct InterpreterValue(pub u64);

impl InterpreterValue {
    #[inline(always)]
    pub fn i32(v: i32) -> Self {
        Self(v as u32 as u64)
    }

    #[inline(always)]
    pub fn i64(v: i64) -> Self {
        Self(v as u64)
    }

    #[inline(always)]
    pub fn f32(v: f32) -> Self {
        Self(v.to_bits() as u64)
    }

    #[inline(always)]
    pub fn f64(v: f64) -> Self {
        Self(v.to_bits())
    }

    #[inline(always)]
    pub fn bool(v: bool) -> Self {
        Self(v as u64)
    }

    #[inline(always)]
    pub fn none() -> Self {
        Self(0)
    }

    #[inline(always)]
    pub fn unwarp_i32(self) -> i32 {
        self.0 as i32
    }

    #[inline(always)]
    pub fn unwarp_i64(self) -> i64 {
        self.0 as i64
    }

    #[inline(always)]
    pub fn unwarp_f32(self) -> f32 {
        f32::from_bits(self.0 as u32)
    }

    #[inline(always)]
    pub fn unwarp_f64(self) -> f64 {
        f64::from_bits(self.0)
    }

    #[inline(always)]
    pub fn unwarp_bool(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub fn to_i64_bits(self) -> i64 {
        self.0 as i64
    }

    pub fn from_i64(v: i64, res_ty: Type) -> Self {
        match res_ty {
            Type::I8 => InterpreterValue::i32((v as i8) as i32),
            Type::I16 => InterpreterValue::i32((v as i16) as i32),
            Type::I32 => InterpreterValue::i32(v as i32),
            Type::I64 | Type::Ptr => InterpreterValue::i64(v),
            Type::F32 => InterpreterValue::f32(f32::from_bits(v as u32)),
            Type::F64 => InterpreterValue::f64(f64::from_bits(v as u64)),
            Type::Bool => InterpreterValue::bool(v != 0),
            _ => InterpreterValue::none(),
        }
    }
}

pub trait HostFuncArg: Copy {
    fn from_val(v: InterpreterValue) -> Self;
}

pub trait HostFuncRet {
    fn into_val(self) -> InterpreterValue;
}

macro_rules! impl_host_func_types {
    ($($t:ty, $meth:ident, $unwarp:ident);*) => {
        $(
            impl HostFuncArg for $t {
                fn from_val(v: InterpreterValue) -> Self { v.$unwarp() }
            }
            impl HostFuncRet for $t {
                fn into_val(self) -> InterpreterValue { InterpreterValue::$meth(self) }
            }
        )*
    };
}

impl_host_func_types! {
    i32, i32, unwarp_i32;
    i64, i64, unwarp_i64;
    f32, f32, unwarp_f32;
    f64, f64, unwarp_f64;
    bool, bool, unwarp_bool
}

impl HostFuncArg for InterpreterValue {
    fn from_val(v: InterpreterValue) -> Self {
        v
    }
}

impl HostFuncRet for InterpreterValue {
    fn into_val(self) -> InterpreterValue {
        self
    }
}

pub trait HostFuncArgs {
    fn arity() -> usize;
    fn decode(args: &[InterpreterValue]) -> Self;
}

pub trait HostFuncRets {
    fn encode(self, results: &mut [InterpreterValue]);
}

macro_rules! impl_args_rets {
    ($n:expr => ($($t:ident),*)) => {
        impl<$($t: HostFuncArg),*> HostFuncArgs for ($($t,)*) {
            fn arity() -> usize { $n }
            fn decode(args: &[InterpreterValue]) -> Self {
                #[allow(unused_mut)]
                let mut _iter = args.iter();
                ( $( $t::from_val(*_iter.next().expect("missing arg")), )* )
            }
        }

        impl<$($t: HostFuncRet),*> HostFuncRets for ($($t,)*) {
            fn encode(self, _results: &mut [InterpreterValue]) {
                #[allow(non_snake_case)]
                let ($($t,)*) = self;
                #[allow(unused_mut)]
                let mut i = 0;
                $(
                    _results[i] = $t.into_val();
                    i += 1;
                )*
                let _ = i;
            }
        }
    };
}

impl_args_rets!(0 => ());
impl_args_rets!(1 => (A));
impl_args_rets!(2 => (A, B));
impl_args_rets!(3 => (A, B, C));
impl_args_rets!(4 => (A, B, C, D));
impl_args_rets!(5 => (A, B, C, D, E));
impl_args_rets!(6 => (A, B, C, D, E, F));
impl_args_rets!(7 => (A, B, C, D, E, F, G));
impl_args_rets!(8 => (A, B, C, D, E, F, G, H));

impl<T: HostFuncRet> HostFuncRets for T {
    fn encode(self, results: &mut [InterpreterValue]) {
        results[0] = self.into_val();
    }
}

impl HostFuncArgs for Vec<InterpreterValue> {
    fn arity() -> usize {
        0
    }
    fn decode(args: &[InterpreterValue]) -> Self {
        args.to_vec()
    }
}

impl HostFuncRets for Vec<InterpreterValue> {
    fn encode(self, results: &mut [InterpreterValue]) {
        for (i, v) in self.into_iter().enumerate() {
            if i < results.len() {
                results[i] = v;
            }
        }
    }
}
