extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use veloc_ir::{Block, FloatCC, FuncId, Function, InstructionData, IntCC, Module, Opcode, Type};
pub mod error;
use ::alloc::boxed::Box;
use ::alloc::string::String;
use ::alloc::sync::Arc;
use ::alloc::vec;
use ::alloc::vec::Vec;
pub use error::{Error, Result};
use hashbrown::HashMap;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub usize);

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

pub type HostFunction = Arc<dyn Fn(&[InterpreterValue]) -> InterpreterValue + Send + Sync>;

pub type TrampolineFn =
    unsafe extern "C" fn(env: *mut u8, args_results: *mut InterpreterValue, arity: usize);

pub struct HostFunctionInner {
    handler: TrampolineFn,
    env: *mut u8,
    drop_fn: fn(*mut u8),
}

unsafe impl Send for HostFunctionInner {}
unsafe impl Sync for HostFunctionInner {}

impl Drop for HostFunctionInner {
    fn drop(&mut self) {
        (self.drop_fn)(self.env);
    }
}

#[derive(Clone)]
pub struct HostFunc(pub Arc<HostFunctionInner>);

impl core::fmt::Debug for HostFunc {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HostFunc").finish()
    }
}

impl HostFunc {
    pub fn call(&self, args: &mut [InterpreterValue]) -> InterpreterValue {
        let arity = args.len();
        unsafe {
            (self.0.handler)(self.0.env, args.as_mut_ptr(), arity);
        }
        if arity > 0 {
            args[0]
        } else {
            InterpreterValue::none()
        }
    }
}

#[derive(Clone)]
pub struct Program {
    pub host_functions: HashMap<String, HostFunc>,
    pub host_functions_list: Vec<HostFunc>,
    pub modules: Vec<Module>,
}

impl Program {
    pub fn new() -> Self {
        Self {
            host_functions: HashMap::new(),
            host_functions_list: Vec::new(),
            modules: Vec::new(),
        }
    }

    pub fn register_module(&mut self, module: Module) -> ModuleId {
        let id = ModuleId(self.modules.len());
        self.modules.push(module);
        id
    }

    fn register_handler<F>(&mut self, name: String, handler: F, trampoline: TrampolineFn) -> usize
    where
        F: Send + Sync + 'static,
    {
        let env = Box::into_raw(Box::new(handler)) as *mut u8;
        let drop_fn = |ptr: *mut u8| unsafe {
            let _ = Box::from_raw(ptr as *mut F);
        };

        let host_func = HostFunc(Arc::new(HostFunctionInner {
            handler: trampoline,
            env,
            drop_fn,
        }));

        self.host_functions.insert(name, host_func.clone());
        let id = self.host_functions_list.len();
        self.host_functions_list.push(host_func);
        id
    }

    pub fn register_raw(&mut self, name: String, f: HostFunction) -> usize {
        unsafe extern "C" fn trampoline(
            env: *mut u8,
            args_results: *mut InterpreterValue,
            arity: usize,
        ) {
            unsafe {
                let func = &*(env as *const HostFunction);
                let args_slice = core::slice::from_raw_parts(args_results, arity);
                let res = func(args_slice);
                *args_results = res;
            }
        }

        self.register_handler(name, f, trampoline)
    }

    pub fn register_func<F, Args, Rets>(&mut self, name: String, func: F) -> usize
    where
        F: Fn(Args) -> Rets + Send + Sync + 'static,
        Args: HostFuncArgs,
        Rets: HostFuncRets,
    {
        unsafe extern "C" fn trampoline<F, Args, Rets>(
            env: *mut u8,
            args_results: *mut InterpreterValue,
            arity: usize,
        ) where
            F: Fn(Args) -> Rets + Send + Sync + 'static,
            Args: HostFuncArgs,
            Rets: HostFuncRets,
        {
            unsafe {
                let func = &*(env as *const F);
                let args_slice = core::slice::from_raw_parts(args_results, arity);
                let args = Args::decode(args_slice);
                let rets = func(args);
                let results_slice = core::slice::from_raw_parts_mut(args_results, 8.max(arity));
                rets.encode(results_slice);
            }
        }

        self.register_handler(name, func, trampoline::<F, Args, Rets>)
    }

    pub fn get_host_func_ptr(&self, id: usize) -> *const u8 {
        // [ID (62 bits) | Tag (2 bits = 10)]
        let val = ((id as u64) << 2) | 2;
        val as *const u8
    }

    pub fn get_interpreter_func_ptr(&self, module_id: ModuleId, func_id: FuncId) -> *const u8 {
        // [ModuleId (31 bits) | FuncId (32 bits) | Tag (1 bit = 1)]
        let mid = module_id.0 as u64;
        let fid = func_id.0 as u64;
        let val = (mid << 33) | (fid << 1) | 1;
        val as *const u8
    }

    pub fn decode_interpreter_ptr(&self, ptr_val: usize) -> Option<(ModuleId, FuncId)> {
        if ptr_val & 1 == 1 {
            let mid = (ptr_val >> 33) as usize;
            let fid = ((ptr_val >> 1) & 0xFFFFFFFF) as u32;
            Some((ModuleId(mid), FuncId(fid)))
        } else {
            None
        }
    }

    fn decode_host_ptr(&self, ptr_val: usize) -> Option<usize> {
        if ptr_val & 3 == 2 {
            Some(ptr_val >> 2)
        } else {
            None
        }
    }
}

pub trait VirtualMemory {
    fn translate_addr(&self, logical_addr: usize, size: usize) -> Option<*mut u8>;
}

pub struct Interpreter {
    pub module_id: ModuleId,
    // 统一的大栈，减少分配
    pub(crate) value_stack: Vec<InterpreterValue>,
    pub(crate) frames: Vec<StackFrame>,
}

struct StackFrame {
    module_id: ModuleId,
    func_id: FuncId,
    // 基础偏移，用于索引 value_stack
    base: usize,
    stack_slots: Vec<Vec<u8>>,
    current_block: Block,
    inst_idx: usize,
    block_args: Vec<InterpreterValue>,
    // 记录返回值应该存放在之前帧的哪个 Value 中
    pending_result_target: Option<veloc_ir::Value>,
}

impl Interpreter {
    pub fn new(module_id: ModuleId) -> Self {
        Self {
            module_id,
            value_stack: Vec::with_capacity(1024),
            frames: Vec::with_capacity(64),
        }
    }

    pub fn run_function<M: VirtualMemory>(
        &mut self,
        program: &Program,
        vm: &M,
        func_id: FuncId,
        args: &[InterpreterValue],
    ) -> InterpreterValue {
        let initial_frame_depth = self.frames.len();
        let mut last_result = InterpreterValue::none();

        // 初始帧
        let module = &program.modules[self.module_id.0];
        let func = &module.functions[func_id];

        // 如果是外部函数，直接执行并返回
        if !func.is_defined() {
            if let Some(host_fn) = program.host_functions.get(&func.name).cloned() {
                let mut args_vec = args.to_vec();
                return host_fn.call(&mut args_vec);
            }
            panic!("External function {} not registered", func.name);
        }

        let first_frame = self.create_frame(program, self.module_id, func_id, args.to_vec());
        self.frames.push(first_frame);

        while self.frames.len() > initial_frame_depth {
            let mut frame = self.frames.pop().unwrap();
            self.module_id = frame.module_id;

            // 如果有待处理的返回值，将其存入之前记录的结果目标中
            if let Some(target) = frame.pending_result_target.take() {
                self.value_stack[frame.base + target.0 as usize] = last_result;
                last_result = InterpreterValue::none();
            }

            let func_ptr =
                &program.modules[frame.module_id.0].functions[frame.func_id] as *const Function;

            // 执行当前帧
            'frame_loop: loop {
                let func = unsafe { &*func_ptr };
                let block_data = &func.layout.blocks[frame.current_block];

                // 如果是新进入一个块（inst_idx == 0），设置块参数
                if frame.inst_idx == 0 {
                    for (param, arg) in block_data.params.iter().zip(frame.block_args.iter()) {
                        self.value_stack[frame.base + param.0 as usize] = *arg;
                    }
                }

                while frame.inst_idx < block_data.insts.len() {
                    let inst = block_data.insts[frame.inst_idx];
                    let idata = &func.dfg.instructions[inst];
                    let res_val = func.dfg.inst_results(inst);

                    frame.inst_idx += 1;

                    // 优化：内联常见指令执行，减少 ControlFlow 开销
                    let flow = self.execute_inst(
                        program,
                        vm,
                        idata,
                        frame.base,
                        &mut frame.stack_slots,
                        func_ptr,
                    );

                    match flow {
                        ControlFlow::Continue(res) => {
                            if let Some(rv) = res_val {
                                self.value_stack[frame.base + rv.0 as usize] = res;
                            }
                        }
                        ControlFlow::Call(m_id, f_id, c_args) => {
                            let callee_module = &program.modules[m_id.0];
                            let callee_func = &callee_module.functions[f_id];

                            if !callee_func.is_defined() {
                                if let Some(host_fn) =
                                    program.host_functions.get(&callee_func.name).cloned()
                                {
                                    let mut c_args = c_args;
                                    let h_res = host_fn.call(&mut c_args);
                                    if let Some(rv) = res_val {
                                        self.value_stack[frame.base + rv.0 as usize] = h_res;
                                    }
                                } else {
                                    panic!("External function {} not registered", callee_func.name);
                                }
                            } else {
                                frame.pending_result_target = res_val;
                                self.frames.push(frame);
                                let new_frame = self.create_frame(program, m_id, f_id, c_args);
                                self.frames.push(new_frame);
                                break 'frame_loop;
                            }
                        }
                        ControlFlow::Return(res) => {
                            last_result = res;
                            // 函数结束，清理当前函数的栈空间
                            self.value_stack.truncate(frame.base);
                            break 'frame_loop;
                        }
                        ControlFlow::Jump(next_block, next_args) => {
                            frame.current_block = next_block;
                            frame.block_args = next_args;
                            frame.inst_idx = 0;
                            continue 'frame_loop;
                        }
                    }
                }
            }
        }

        last_result
    }

    fn create_frame(
        &mut self,
        program: &Program,
        module_id: ModuleId,
        func_id: FuncId,
        args: Vec<InterpreterValue>,
    ) -> StackFrame {
        let module = &program.modules[module_id.0];
        let func = &module.functions[func_id];

        // 分配栈空间
        let base = self.value_stack.len();
        self.value_stack
            .resize(base + func.dfg.values.len(), InterpreterValue::none());

        let stack_slots = func
            .stack_slots
            .iter()
            .map(|(_, data)| vec![0; data.size as usize])
            .collect();
        let current_block = func.entry_block.expect("Function has no entry block");

        StackFrame {
            module_id,
            func_id,
            base,
            stack_slots,
            current_block,
            inst_idx: 0,
            block_args: args,
            pending_result_target: None,
        }
    }

    fn execute_inst<M: VirtualMemory>(
        &mut self,
        program: &Program,
        vm: &M,
        idata: &InstructionData,
        base: usize,
        stack_slots: &mut [Vec<u8>],
        func_ptr: *const Function,
    ) -> ControlFlow {
        let func = unsafe { &*func_ptr };
        match idata {
            InstructionData::Iconst { value, ty } => match ty {
                Type::I8 | Type::I16 | Type::I32 => {
                    ControlFlow::Continue(InterpreterValue::i32(*value as i32))
                }
                Type::I64 | Type::Ptr => ControlFlow::Continue(InterpreterValue::i64(*value)),
                Type::Bool => ControlFlow::Continue(InterpreterValue::bool(*value != 0)),
                _ => panic!("Invalid type for iconst: {:?}", ty),
            },
            InstructionData::Fconst { value, ty } => {
                if *ty == Type::F32 {
                    ControlFlow::Continue(InterpreterValue::f32(f32::from_bits(*value as u32)))
                } else {
                    ControlFlow::Continue(InterpreterValue::f64(f64::from_bits(*value)))
                }
            }
            InstructionData::Bconst { value } => {
                ControlFlow::Continue(InterpreterValue::bool(*value))
            }
            InstructionData::Binary { opcode, args, ty } => {
                let lhs = self.value_stack[base + args[0].0 as usize];
                let rhs = self.value_stack[base + args[1].0 as usize];
                ControlFlow::Continue(self.exec_binary(*opcode, lhs, rhs, *ty))
            }
            InstructionData::Unary { opcode, arg, ty } => {
                let val = self.value_stack[base + arg.0 as usize];
                let arg_ty = func.dfg.values[*arg].ty;
                ControlFlow::Continue(self.exec_unary(*opcode, val, arg_ty, *ty))
            }
            InstructionData::IntCompare { kind, args, .. } => {
                let lhs = self.value_stack[base + args[0].0 as usize];
                let rhs = self.value_stack[base + args[1].0 as usize];
                let ty = func.dfg.values[args[0]].ty;
                ControlFlow::Continue(self.exec_icmp(*kind, lhs, rhs, ty))
            }
            InstructionData::FloatCompare { kind, args, .. } => {
                let lhs = self.value_stack[base + args[0].0 as usize];
                let rhs = self.value_stack[base + args[1].0 as usize];
                let ty = func.dfg.values[args[0]].ty;
                ControlFlow::Continue(self.exec_fcmp(*kind, lhs, rhs, ty))
            }
            InstructionData::Select {
                condition,
                then_val,
                else_val,
                ..
            } => {
                let cond = self.value_stack[base + condition.0 as usize].unwarp_bool();
                let val = if cond {
                    self.value_stack[base + then_val.0 as usize]
                } else {
                    self.value_stack[base + else_val.0 as usize]
                };
                ControlFlow::Continue(val)
            }
            InstructionData::Load {
                ptr, offset, ty, ..
            } => {
                let addr = self.value_stack[base + ptr.0 as usize].unwarp_i64() as usize
                    + *offset as usize;
                let res = self.read_memory(vm, addr, *ty);
                ControlFlow::Continue(res)
            }
            InstructionData::Store {
                ptr, value, offset, ..
            } => {
                let addr = self.value_stack[base + ptr.0 as usize].unwarp_i64() as usize
                    + *offset as usize;
                let val = self.value_stack[base + value.0 as usize];
                let ty = func.dfg.values[*value].ty;
                self.write_memory(vm, addr, val, ty);
                ControlFlow::Continue(InterpreterValue::none())
            }
            InstructionData::StackAddr { slot, offset } => {
                let addr = stack_slots[(*slot).0 as usize].as_mut_ptr() as i64 + (*offset as i64);
                ControlFlow::Continue(InterpreterValue::i64(addr))
            }
            InstructionData::StackStore {
                slot,
                value,
                offset,
            } => {
                let val = self.value_stack[base + value.0 as usize];
                let ty = func.dfg.values[*value].ty;
                let slot_data = &mut stack_slots[slot.0 as usize];
                let bytes = match (val, ty) {
                    (v, Type::I8) => (v.unwarp_i32() as u8).to_le_bytes().to_vec(),
                    (v, Type::I16) => (v.unwarp_i32() as u16).to_le_bytes().to_vec(),
                    (v, Type::I32) => v.unwarp_i32().to_le_bytes().to_vec(),
                    (v, Type::I64) => v.unwarp_i64().to_le_bytes().to_vec(),
                    (v, Type::Ptr) => v.unwarp_i64().to_le_bytes().to_vec(),
                    (v, Type::F32) => v.unwarp_f32().to_bits().to_le_bytes().to_vec(),
                    (v, Type::F64) => v.unwarp_f64().to_bits().to_le_bytes().to_vec(),
                    _ => panic!(
                        "Unsupported stack store value: {:?} with type {:?}",
                        val, ty
                    ),
                };
                let off = *offset as usize;
                slot_data[off..off + bytes.len()].copy_from_slice(&bytes);
                ControlFlow::Continue(InterpreterValue::none())
            }
            InstructionData::StackLoad { slot, offset, ty } => {
                let slot_data = &stack_slots[slot.0 as usize];
                let off = *offset as usize;
                let res = match ty {
                    Type::I8 => InterpreterValue::i32(slot_data[off] as i32),
                    Type::I16 => {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(&slot_data[off..off + 2]);
                        InterpreterValue::i32(u16::from_le_bytes(b) as i32)
                    }
                    Type::I32 => {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&slot_data[off..off + 4]);
                        InterpreterValue::i32(i32::from_le_bytes(b))
                    }
                    Type::I64 => {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slot_data[off..off + 8]);
                        InterpreterValue::i64(i64::from_le_bytes(b))
                    }
                    Type::F32 => {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&slot_data[off..off + 4]);
                        InterpreterValue::f32(f32::from_bits(u32::from_le_bytes(b)))
                    }
                    Type::F64 => {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slot_data[off..off + 8]);
                        InterpreterValue::f64(f64::from_bits(u64::from_le_bytes(b)))
                    }
                    _ => panic!("Unsupported stack load type"),
                };
                ControlFlow::Continue(res)
            }
            InstructionData::Jump { dest } => {
                let dest_data = func.dfg.block_calls[*dest];
                let j_args = func
                    .dfg
                    .get_value_list(dest_data.args)
                    .iter()
                    .map(|&v| self.value_stack[base + v.0 as usize])
                    .collect();
                ControlFlow::Jump(dest_data.block, j_args)
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                let cond = self.value_stack[base + condition.0 as usize].unwarp_bool();
                let dest = if cond { then_dest } else { else_dest };
                let dest_data = func.dfg.block_calls[*dest];
                let j_args = func
                    .dfg
                    .get_value_list(dest_data.args)
                    .iter()
                    .map(|&v| self.value_stack[base + v.0 as usize])
                    .collect();
                ControlFlow::Jump(dest_data.block, j_args)
            }
            InstructionData::Return { value } => {
                let res = if let Some(v) = value {
                    self.value_stack[base + v.0 as usize]
                } else {
                    InterpreterValue::none()
                };
                ControlFlow::Return(res)
            }
            InstructionData::BrTable { index, table } => {
                let idx = self.value_stack[base + index.0 as usize].unwarp_i32() as usize;
                let table_data = &func.dfg.jump_tables[*table];
                let target_call = if idx < table_data.targets.len() - 1 {
                    table_data.targets[idx + 1]
                } else {
                    table_data.targets[0]
                };
                let target_data = func.dfg.block_calls[target_call];

                let j_args = func
                    .dfg
                    .get_value_list(target_data.args)
                    .iter()
                    .map(|&v| self.value_stack[base + v.0 as usize])
                    .collect();
                ControlFlow::Jump(target_data.block, j_args)
            }
            InstructionData::Call { func_id, args, .. } => {
                let call_args: Vec<_> = func
                    .dfg
                    .get_value_list(*args)
                    .iter()
                    .map(|&v| self.value_stack[base + v.0 as usize])
                    .collect();
                ControlFlow::Call(self.module_id, *func_id, call_args)
            }
            InstructionData::CallIndirect {
                ptr,
                args,
                sig_id: _,
                ..
            } => {
                let ptr_val = self.value_stack[base + ptr.0 as usize].unwarp_i64() as usize;
                let call_args: Vec<InterpreterValue> = func
                    .dfg
                    .get_value_list(*args)
                    .iter()
                    .map(|&v| self.value_stack[base + v.0 as usize])
                    .collect();

                // 检查是否为虚拟指针 (Tagged Pointer)
                if let Some((module_id, func_id)) = program.decode_interpreter_ptr(ptr_val) {
                    ControlFlow::Call(module_id, func_id, call_args)
                } else if let Some(host_id) = program.decode_host_ptr(ptr_val) {
                    let host_fn = &program.host_functions_list[host_id];
                    let mut call_args = call_args;
                    let res = host_fn.call(&mut call_args);
                    ControlFlow::Continue(res)
                } else {
                    panic!(
                        "Indirect call to raw pointer {:#x} is not supported. All host functions must be registered via register_func.",
                        ptr_val
                    );
                }
            }
            InstructionData::IntToPtr { arg } => {
                let v = self.value_stack[base + arg.0 as usize].unwarp_i64();
                ControlFlow::Continue(InterpreterValue::i64(v))
            }
            InstructionData::PtrToInt { arg, ty } => {
                let v = self.value_stack[base + arg.0 as usize].unwarp_i64();
                match ty {
                    Type::I32 => ControlFlow::Continue(InterpreterValue::i32(v as i32)),
                    _ => ControlFlow::Continue(InterpreterValue::i64(v)),
                }
            }
            InstructionData::PtrOffset { ptr, offset } => {
                let p = self.value_stack[base + ptr.0 as usize].unwarp_i64();
                ControlFlow::Continue(InterpreterValue::i64(p + *offset as i64))
            }
            InstructionData::PtrIndex {
                ptr,
                index,
                scale,
                offset,
            } => {
                let p = self.value_stack[base + ptr.0 as usize].unwarp_i64();
                let idx = self.value_stack[base + index.0 as usize].unwarp_i64();
                ControlFlow::Continue(InterpreterValue::i64(
                    p + idx * (*scale as i64) + (*offset as i64),
                ))
            }
            InstructionData::Unreachable => panic!("Unreachable code hit"),
        }
    }

    fn exec_binary(
        &self,
        opcode: Opcode,
        lhs: InterpreterValue,
        rhs: InterpreterValue,
        ty: Type,
    ) -> InterpreterValue {
        match opcode {
            Opcode::Iadd => {
                if ty == Type::I32 {
                    InterpreterValue::i32(lhs.unwarp_i32().wrapping_add(rhs.unwarp_i32()))
                } else {
                    InterpreterValue::i64(lhs.unwarp_i64().wrapping_add(rhs.unwarp_i64()))
                }
            }
            Opcode::Isub => {
                if ty == Type::I32 {
                    InterpreterValue::i32(lhs.unwarp_i32().wrapping_sub(rhs.unwarp_i32()))
                } else {
                    InterpreterValue::i64(lhs.unwarp_i64().wrapping_sub(rhs.unwarp_i64()))
                }
            }
            Opcode::Imul => {
                if ty == Type::I32 {
                    InterpreterValue::i32(lhs.unwarp_i32().wrapping_mul(rhs.unwarp_i32()))
                } else {
                    InterpreterValue::i64(lhs.unwarp_i64().wrapping_mul(rhs.unwarp_i64()))
                }
            }
            Opcode::Idiv => {
                if ty == Type::I32 {
                    let a = lhs.unwarp_i32();
                    let b = rhs.unwarp_i32();
                    InterpreterValue::i32(a.wrapping_div(b))
                } else {
                    let a = lhs.unwarp_i64();
                    let b = rhs.unwarp_i64();
                    InterpreterValue::i64(a.wrapping_div(b))
                }
            }
            Opcode::Udiv => {
                if ty == Type::I32 {
                    let a = lhs.unwarp_i32() as u32;
                    let b = rhs.unwarp_i32() as u32;
                    InterpreterValue::i32((a / b) as i32)
                } else {
                    let a = lhs.0;
                    let b = rhs.0;
                    InterpreterValue::i64((a / b) as i64)
                }
            }
            Opcode::Irem => {
                if ty == Type::I32 {
                    let a = lhs.unwarp_i32();
                    let b = rhs.unwarp_i32();
                    InterpreterValue::i32(a.wrapping_rem(b))
                } else {
                    let a = lhs.unwarp_i64();
                    let b = rhs.unwarp_i64();
                    InterpreterValue::i64(a.wrapping_rem(b))
                }
            }
            Opcode::Urem => {
                if ty == Type::I32 {
                    let a = lhs.unwarp_i32() as u32;
                    let b = rhs.unwarp_i32() as u32;
                    InterpreterValue::i32((a % b) as i32)
                } else {
                    let a = lhs.0;
                    let b = rhs.0;
                    InterpreterValue::i64((a % b) as i64)
                }
            }
            Opcode::And => InterpreterValue(lhs.0 & rhs.0),
            Opcode::Or => InterpreterValue(lhs.0 | rhs.0),
            Opcode::Xor => InterpreterValue(lhs.0 ^ rhs.0),
            Opcode::Shl => {
                if ty == Type::I32 {
                    InterpreterValue::i32(lhs.unwarp_i32().wrapping_shl(rhs.0 as u32 % 32))
                } else {
                    InterpreterValue::i64(lhs.unwarp_i64().wrapping_shl(rhs.0 as u32 % 64))
                }
            }
            Opcode::ShrS => {
                if ty == Type::I32 {
                    InterpreterValue::i32(lhs.unwarp_i32().wrapping_shr(rhs.0 as u32 % 32))
                } else {
                    InterpreterValue::i64(lhs.unwarp_i64().wrapping_shr(rhs.0 as u32 % 64))
                }
            }
            Opcode::ShrU => {
                if ty == Type::I32 {
                    InterpreterValue::i32((lhs.0 as u32).wrapping_shr(rhs.0 as u32 % 32) as i32)
                } else {
                    InterpreterValue::i64(lhs.0.wrapping_shr(rhs.0 as u32 % 64) as i64)
                }
            }
            Opcode::Rotl => {
                if ty == Type::I32 {
                    InterpreterValue::i32((lhs.0 as u32).rotate_left(rhs.0 as u32 % 32) as i32)
                } else {
                    InterpreterValue::i64(lhs.0.rotate_left(rhs.0 as u32 % 64) as i64)
                }
            }
            Opcode::Rotr => {
                if ty == Type::I32 {
                    InterpreterValue::i32((lhs.0 as u32).rotate_right(rhs.0 as u32 % 32) as i32)
                } else {
                    InterpreterValue::i64(lhs.0.rotate_right(rhs.0 as u32 % 64) as i64)
                }
            }
            Opcode::Fadd => {
                if ty == Type::F32 {
                    InterpreterValue::f32(lhs.unwarp_f32() + rhs.unwarp_f32())
                } else {
                    InterpreterValue::f64(lhs.unwarp_f64() + rhs.unwarp_f64())
                }
            }
            Opcode::Fsub => {
                if ty == Type::F32 {
                    InterpreterValue::f32(lhs.unwarp_f32() - rhs.unwarp_f32())
                } else {
                    InterpreterValue::f64(lhs.unwarp_f64() - rhs.unwarp_f64())
                }
            }
            Opcode::Fmul => {
                if ty == Type::F32 {
                    InterpreterValue::f32(lhs.unwarp_f32() * rhs.unwarp_f32())
                } else {
                    InterpreterValue::f64(lhs.unwarp_f64() * rhs.unwarp_f64())
                }
            }
            Opcode::Fdiv => {
                if ty == Type::F32 {
                    InterpreterValue::f32(lhs.unwarp_f32() / rhs.unwarp_f32())
                } else {
                    InterpreterValue::f64(lhs.unwarp_f64() / rhs.unwarp_f64())
                }
            }
            Opcode::Min => {
                if ty == Type::F32 {
                    InterpreterValue::f32(lhs.unwarp_f32().min(rhs.unwarp_f32()))
                } else {
                    InterpreterValue::f64(lhs.unwarp_f64().min(rhs.unwarp_f64()))
                }
            }
            Opcode::Max => {
                if ty == Type::F32 {
                    InterpreterValue::f32(lhs.unwarp_f32().max(rhs.unwarp_f32()))
                } else {
                    InterpreterValue::f64(lhs.unwarp_f64().max(rhs.unwarp_f64()))
                }
            }
            Opcode::Copysign => {
                if ty == Type::F32 {
                    InterpreterValue::f32(lhs.unwarp_f32().copysign(rhs.unwarp_f32()))
                } else {
                    InterpreterValue::f64(lhs.unwarp_f64().copysign(rhs.unwarp_f64()))
                }
            }
            _ => todo!("Implement more binary ops: {:?}", opcode),
        }
    }

    fn exec_icmp(
        &self,
        kind: IntCC,
        lhs: InterpreterValue,
        rhs: InterpreterValue,
        ty: Type,
    ) -> InterpreterValue {
        let res = match kind {
            IntCC::Eq => lhs.0 == rhs.0,
            IntCC::Ne => lhs.0 != rhs.0,
            IntCC::LtS => {
                if ty == Type::I32 {
                    lhs.unwarp_i32() < rhs.unwarp_i32()
                } else {
                    lhs.unwarp_i64() < rhs.unwarp_i64()
                }
            }
            IntCC::LeS => {
                if ty == Type::I32 {
                    lhs.unwarp_i32() <= rhs.unwarp_i32()
                } else {
                    lhs.unwarp_i64() <= rhs.unwarp_i64()
                }
            }
            IntCC::GtS => {
                if ty == Type::I32 {
                    lhs.unwarp_i32() > rhs.unwarp_i32()
                } else {
                    lhs.unwarp_i64() > rhs.unwarp_i64()
                }
            }
            IntCC::GeS => {
                if ty == Type::I32 {
                    lhs.unwarp_i32() >= rhs.unwarp_i32()
                } else {
                    lhs.unwarp_i64() >= rhs.unwarp_i64()
                }
            }
            IntCC::LtU => lhs.0 < rhs.0,
            IntCC::LeU => lhs.0 <= rhs.0,
            IntCC::GtU => lhs.0 > rhs.0,
            IntCC::GeU => lhs.0 >= rhs.0,
        };
        InterpreterValue::bool(res)
    }

    fn exec_fcmp(
        &self,
        kind: FloatCC,
        lhs: InterpreterValue,
        rhs: InterpreterValue,
        ty: Type,
    ) -> InterpreterValue {
        let res = if ty == Type::F32 {
            let a = lhs.unwarp_f32();
            let b = rhs.unwarp_f32();
            match kind {
                FloatCC::Eq => a == b,
                FloatCC::Ne => a != b,
                FloatCC::Lt => a < b,
                FloatCC::Le => a <= b,
                FloatCC::Gt => a > b,
                FloatCC::Ge => a >= b,
            }
        } else {
            let a = lhs.unwarp_f64();
            let b = rhs.unwarp_f64();
            match kind {
                FloatCC::Eq => a == b,
                FloatCC::Ne => a != b,
                FloatCC::Lt => a < b,
                FloatCC::Le => a <= b,
                FloatCC::Gt => a > b,
                FloatCC::Ge => a >= b,
            }
        };
        InterpreterValue::bool(res)
    }

    fn exec_unary(
        &self,
        opcode: Opcode,
        val: InterpreterValue,
        arg_ty: Type,
        res_ty: Type,
    ) -> InterpreterValue {
        match opcode {
            Opcode::Ineg => {
                if res_ty == Type::I32 {
                    InterpreterValue::i32(val.unwarp_i32().wrapping_neg())
                } else {
                    InterpreterValue::i64(val.unwarp_i64().wrapping_neg())
                }
            }
            Opcode::Eqz => InterpreterValue::bool(val.0 == 0),
            Opcode::Clz => {
                if res_ty == Type::I32 {
                    InterpreterValue::i32((val.0 as u32).leading_zeros() as i32)
                } else {
                    InterpreterValue::i64(val.0.leading_zeros() as i64)
                }
            }
            Opcode::Ctz => {
                if res_ty == Type::I32 {
                    InterpreterValue::i32((val.0 as u32).trailing_zeros() as i32)
                } else {
                    InterpreterValue::i64(val.0.trailing_zeros() as i64)
                }
            }
            Opcode::Popcnt => {
                if res_ty == Type::I32 {
                    InterpreterValue::i32((val.0 as u32).count_ones() as i32)
                } else {
                    InterpreterValue::i64(val.0.count_ones() as i64)
                }
            }
            Opcode::Abs => {
                if res_ty == Type::F32 {
                    InterpreterValue::f32(val.unwarp_f32().abs())
                } else {
                    InterpreterValue::f64(val.unwarp_f64().abs())
                }
            }
            Opcode::Fneg => {
                if res_ty == Type::F32 {
                    InterpreterValue::f32(-val.unwarp_f32())
                } else {
                    InterpreterValue::f64(-val.unwarp_f64())
                }
            }
            Opcode::Sqrt => {
                if res_ty == Type::F32 {
                    InterpreterValue::f32(val.unwarp_f32().sqrt())
                } else {
                    InterpreterValue::f64(val.unwarp_f64().sqrt())
                }
            }
            Opcode::Ceil => {
                if res_ty == Type::F32 {
                    InterpreterValue::f32(val.unwarp_f32().ceil())
                } else {
                    InterpreterValue::f64(val.unwarp_f64().ceil())
                }
            }
            Opcode::Floor => {
                if res_ty == Type::F32 {
                    InterpreterValue::f32(val.unwarp_f32().floor())
                } else {
                    InterpreterValue::f64(val.unwarp_f64().floor())
                }
            }
            Opcode::Trunc => {
                if res_ty == Type::F32 {
                    InterpreterValue::f32(val.unwarp_f32().trunc())
                } else {
                    InterpreterValue::f64(val.unwarp_f64().trunc())
                }
            }
            Opcode::Nearest => {
                if res_ty == Type::F32 {
                    InterpreterValue::f32(val.unwarp_f32().round_ties_even())
                } else {
                    InterpreterValue::f64(val.unwarp_f64().round_ties_even())
                }
            }
            Opcode::ExtendS => {
                let v = val.0;
                match arg_ty {
                    Type::I8 => InterpreterValue::i64(v as i8 as i64),
                    Type::I16 => InterpreterValue::i64(v as i16 as i64),
                    Type::I32 => InterpreterValue::i64(v as i32 as i64),
                    _ => val,
                }
            }
            Opcode::ExtendU => {
                let v = val.0;
                match arg_ty {
                    Type::I8 => InterpreterValue::i64(v as u8 as i64),
                    Type::I16 => InterpreterValue::i64(v as u16 as i64),
                    Type::I32 => InterpreterValue::i64(v as u32 as i64),
                    _ => val,
                }
            }
            Opcode::Wrap => InterpreterValue::i32(val.0 as i32),
            Opcode::Promote => InterpreterValue::f64(val.unwarp_f32() as f64),
            Opcode::Demote => InterpreterValue::f32(val.unwarp_f64() as f32),
            Opcode::ConvertS => {
                let v = if arg_ty == Type::I32 {
                    val.unwarp_i32() as i64
                } else {
                    val.unwarp_i64()
                };
                if res_ty == Type::F32 {
                    InterpreterValue::f32(v as f32)
                } else {
                    InterpreterValue::f64(v as f64)
                }
            }
            Opcode::ConvertU => {
                let v = if arg_ty == Type::I32 {
                    val.unwarp_i32() as u32 as u64
                } else {
                    val.unwarp_i64() as u64
                };
                if res_ty == Type::F32 {
                    InterpreterValue::f32(v as f32)
                } else {
                    InterpreterValue::f64(v as f64)
                }
            }
            Opcode::TruncS => {
                if res_ty == Type::I32 {
                    if arg_ty == Type::F32 {
                        InterpreterValue::i32(val.unwarp_f32() as i32)
                    } else {
                        InterpreterValue::i32(val.unwarp_f64() as i32)
                    }
                } else {
                    if arg_ty == Type::F32 {
                        InterpreterValue::i64(val.unwarp_f32() as i64)
                    } else {
                        InterpreterValue::i64(val.unwarp_f64() as i64)
                    }
                }
            }
            Opcode::TruncU => {
                if res_ty == Type::I32 {
                    if arg_ty == Type::F32 {
                        InterpreterValue::i32(val.unwarp_f32() as u32 as i32)
                    } else {
                        InterpreterValue::i32(val.unwarp_f64() as u32 as i32)
                    }
                } else {
                    if arg_ty == Type::F32 {
                        InterpreterValue::i64(val.unwarp_f32() as u64 as i64)
                    } else {
                        InterpreterValue::i64(val.unwarp_f64() as u64 as i64)
                    }
                }
            }
            Opcode::Reinterpret => val,
            _ => todo!("Implement more unary ops: {:?}", opcode),
        }
    }

    fn read_memory<M: VirtualMemory>(&self, vm: &M, addr: usize, res_ty: Type) -> InterpreterValue {
        let size = self.type_size(res_ty);
        let host_ptr = vm.translate_addr(addr, size).expect("Memory out of bounds");

        unsafe {
            match res_ty {
                Type::I8 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i8);
                    InterpreterValue::i32(val as i32)
                }
                Type::I16 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i16);
                    InterpreterValue::i32(val as i32)
                }
                Type::I32 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i32);
                    InterpreterValue::i32(val)
                }
                Type::I64 | Type::Ptr => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i64);
                    InterpreterValue::i64(val)
                }
                Type::F32 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const f32);
                    InterpreterValue::f32(val)
                }
                Type::F64 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const f64);
                    InterpreterValue::f64(val)
                }
                _ => todo!("Implement more memory read types: {:?}", res_ty),
            }
        }
    }

    fn write_memory<M: VirtualMemory>(
        &mut self,
        vm: &M,
        addr: usize,
        val: InterpreterValue,
        res_ty: Type,
    ) {
        let size = self.type_size(res_ty);
        let host_ptr = vm.translate_addr(addr, size).expect("Memory out of bounds");

        unsafe {
            match res_ty {
                Type::I8 => {
                    core::ptr::write_unaligned(host_ptr as *mut i8, val.0 as i8);
                }
                Type::I16 => {
                    core::ptr::write_unaligned(host_ptr as *mut i16, val.0 as i16);
                }
                Type::I32 => {
                    core::ptr::write_unaligned(host_ptr as *mut i32, val.0 as i32);
                }
                Type::I64 | Type::Ptr => {
                    core::ptr::write_unaligned(host_ptr as *mut i64, val.0 as i64);
                }
                Type::F32 => {
                    core::ptr::write_unaligned(host_ptr as *mut f32, val.unwarp_f32());
                }
                Type::F64 => {
                    core::ptr::write_unaligned(host_ptr as *mut f64, val.unwarp_f64());
                }
                _ => todo!("Implement more memory write types: {:?}", res_ty),
            }
        }
    }

    fn type_size(&self, ty: Type) -> usize {
        match ty {
            Type::I8 | Type::Bool => 1,
            Type::I16 => 2,
            Type::I32 | Type::F32 => 4,
            Type::I64 | Type::F64 | Type::Ptr => 8,
            Type::Void => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum ControlFlow {
    Continue(InterpreterValue),
    Jump(Block, Vec<InterpreterValue>),
    Return(InterpreterValue),
    Call(ModuleId, FuncId, Vec<InterpreterValue>),
}
