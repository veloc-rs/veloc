use crate::vm::VMContext;
use alloc::sync::Arc;
use alloc::vec::Vec;
use veloc::interpreter::{HostFunction, InterpreterValue};
use wasmparser::ValType;

pub struct Caller<'a> {
    pub(crate) vmctx: *mut VMContext,
    _marker: core::marker::PhantomData<&'a mut crate::Store>,
}

impl<'a> Caller<'a> {
    pub unsafe fn from_vmctx(vmctx: *mut VMContext) -> Self {
        Self {
            vmctx,
            _marker: core::marker::PhantomData,
        }
    }

    pub fn vmctx(&self) -> *mut VMContext {
        self.vmctx
    }
}

pub trait WasmTy {
    fn val_type() -> ValType;
    fn from_val(val: InterpreterValue) -> Self;
    fn to_val(self) -> InterpreterValue;
}

impl WasmTy for i32 {
    fn val_type() -> ValType {
        ValType::I32
    }
    fn from_val(val: InterpreterValue) -> Self {
        val.unwarp_i32()
    }
    fn to_val(self) -> InterpreterValue {
        InterpreterValue::I32(self)
    }
}

impl WasmTy for i64 {
    fn val_type() -> ValType {
        ValType::I64
    }
    fn from_val(val: InterpreterValue) -> Self {
        val.unwarp_i64()
    }
    fn to_val(self) -> InterpreterValue {
        InterpreterValue::I64(self)
    }
}

impl WasmTy for f32 {
    fn val_type() -> ValType {
        ValType::F32
    }
    fn from_val(val: InterpreterValue) -> Self {
        val.unwarp_f32()
    }
    fn to_val(self) -> InterpreterValue {
        InterpreterValue::F32(self)
    }
}

impl WasmTy for f64 {
    fn val_type() -> ValType {
        ValType::F64
    }
    fn from_val(val: InterpreterValue) -> Self {
        val.unwarp_f64()
    }
    fn to_val(self) -> InterpreterValue {
        InterpreterValue::F64(self)
    }
}

pub trait WasmRet {
    fn val_types() -> Vec<ValType>;
    fn to_vals(self) -> InterpreterValue;
}

impl WasmRet for () {
    fn val_types() -> Vec<ValType> {
        vec![]
    }
    fn to_vals(self) -> InterpreterValue {
        InterpreterValue::None
    }
}

impl<T: WasmTy> WasmRet for T {
    fn val_types() -> Vec<ValType> {
        vec![T::val_type()]
    }
    fn to_vals(self) -> InterpreterValue {
        self.to_val()
    }
}

pub trait IntoFunc<Params, Results> {
    fn into_func(self) -> (Vec<ValType>, Vec<ValType>, HostFunction);
}

pub trait HostParam {
    fn is_caller() -> bool {
        false
    }
    fn val_type() -> Option<ValType>;
    fn from_args(vmctx: *mut VMContext, val: Option<InterpreterValue>) -> Self;
}

impl<'a> HostParam for Caller<'a> {
    fn is_caller() -> bool {
        true
    }
    fn val_type() -> Option<ValType> {
        None
    }
    fn from_args(vmctx: *mut VMContext, _val: Option<InterpreterValue>) -> Self {
        unsafe { Caller::from_vmctx(vmctx) }
    }
}

impl<T: WasmTy> HostParam for T {
    fn val_type() -> Option<ValType> {
        Some(T::val_type())
    }
    fn from_args(_vmctx: *mut VMContext, val: Option<InterpreterValue>) -> Self {
        T::from_val(val.expect("missing argument"))
    }
}

macro_rules! impl_into_func {
    ($($p:ident),*) => {
        impl<F, $($p,)* R> IntoFunc<($($p,)*), R> for F
        where
            F: Fn($($p),*) -> R + Send + Sync + 'static,
            $($p: HostParam,)*
            R: WasmRet,
        {
            #[allow(non_snake_case)]
            fn into_func(self) -> (Vec<ValType>, Vec<ValType>, HostFunction) {
                #[allow(unused_mut)]
                let mut params = Vec::new();
                $(if let Some(ty) = $p::val_type() {
                    params.push(ty);
                })*

                let results = R::val_types();
                let host_fn = Arc::new(move |args: &[InterpreterValue]| {
                    #[allow(unused_variables)]
                    let vmctx = args[0].to_i64_bits() as *const VMContext as *mut VMContext;
                    #[allow(unused_mut, unused_variables)]
                    let mut iter = args.iter().skip(1);

                    $(let $p = $p::from_args(vmctx, if $p::is_caller() { None } else { iter.next().copied() });)*
                    let res = self($($p),*);
                    res.to_vals()
                });
                (params, results, host_fn)
            }
        }
    }
}

impl_into_func!();
impl_into_func!(A);
impl_into_func!(A, B);
impl_into_func!(A, B, C);
impl_into_func!(A, B, C, D);
impl_into_func!(A, B, C, D, E);
impl_into_func!(A, B, C, D, E, G);
impl_into_func!(A, B, C, D, E, G, H);
impl_into_func!(A, B, C, D, E, G, H, I);
impl_into_func!(A, B, C, D, E, G, H, I, J);
impl_into_func!(A, B, C, D, E, G, H, I, J, K);
impl_into_func!(A, B, C, D, E, G, H, I, J, K, L);
impl_into_func!(A, B, C, D, E, G, H, I, J, K, L, M);
impl_into_func!(A, B, C, D, E, G, H, I, J, K, L, M, N);
impl_into_func!(A, B, C, D, E, G, H, I, J, K, L, M, N, O);
impl_into_func!(A, B, C, D, E, G, H, I, J, K, L, M, N, O, P);
