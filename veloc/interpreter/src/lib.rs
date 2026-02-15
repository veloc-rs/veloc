extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

use veloc_ir::{
    Block, FloatCC, FuncId, Function, InstructionData, IntCC, Module, Opcode, Type,
};
pub mod error;
use ::alloc::string::String;
use ::alloc::sync::Arc;
use ::alloc::vec;
use ::alloc::vec::Vec;
use ::alloc::boxed::Box;
pub use error::{Error, Result};
use hashbrown::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpreterValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    None,
}

impl InterpreterValue {
    pub fn unwarp_i32(self) -> i32 {
        match self {
            InterpreterValue::I32(v) => v,
            _ => panic!("Expected I32 value, got {:?}", self),
        }
    }

    pub fn unwarp_i64(self) -> i64 {
        match self {
            InterpreterValue::I64(v) => v,
            _ => panic!("Expected I64 value, got {:?}", self),
        }
    }

    pub fn unwarp_f32(self) -> f32 {
        match self {
            InterpreterValue::F32(v) => v,
            _ => panic!("Expected F32 value, got {:?}", self),
        }
    }

    pub fn unwarp_f64(self) -> f64 {
        match self {
            InterpreterValue::F64(v) => v,
            _ => panic!("Expected F64 value, got {:?}", self),
        }
    }

    pub fn unwarp_bool(self) -> bool {
        match self {
            InterpreterValue::Bool(v) => v,
            _ => panic!("Expected Bool value, got {:?}", self),
        }
    }

    pub fn to_i64_bits(self) -> i64 {
        match self {
            InterpreterValue::I32(v) => v as i64,
            InterpreterValue::I64(v) => v,
            InterpreterValue::F32(v) => v.to_bits() as i64,
            InterpreterValue::F64(v) => v.to_bits() as i64,
            InterpreterValue::Bool(v) => {
                if v {
                    1
                } else {
                    0
                }
            }
            InterpreterValue::None => 0,
        }
    }

    pub fn from_i64(v: i64, res_ty: Type) -> Self {
        match res_ty {
            Type::I8 => InterpreterValue::I32((v as i8) as i32),
            Type::I16 => InterpreterValue::I32((v as i16) as i32),
            Type::I32 => InterpreterValue::I32(v as i32),
            Type::I64 | Type::Ptr => InterpreterValue::I64(v),
            Type::F32 => InterpreterValue::F32(f32::from_bits(v as u32)),
            Type::F64 => InterpreterValue::F64(f64::from_bits(v as u64)),
            Type::Bool => InterpreterValue::Bool(v != 0),
            Type::Void => InterpreterValue::None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModuleId(pub usize);

#[derive(Debug, Clone)]
pub struct MemoryRegion {
    pub logical_base: usize,
    pub host_base: *mut u8,
    pub size: usize,
    pub name: String,
}

impl MemoryRegion {
    pub fn new(logical_base: u64, host_base: u64, size: usize, name: String) -> Self {
        Self {
            logical_base: logical_base as usize,
            host_base: host_base as *mut u8,
            size,
            name,
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
    ($($t:ty, $variant:ident, $unwarp:ident);*) => {
        $(
            impl HostFuncArg for $t {
                fn from_val(v: InterpreterValue) -> Self { v.$unwarp() }
            }
            impl HostFuncRet for $t {
                fn into_val(self) -> InterpreterValue { InterpreterValue::$variant(self) }
            }
        )*
    };
}

impl_host_func_types! {
    i32, I32, unwarp_i32;
    i64, I64, unwarp_i64;
    f32, F32, unwarp_f32;
    f64, F64, unwarp_f64;
    bool, Bool, unwarp_bool
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

pub type HostFunction = Arc<dyn Fn(&[InterpreterValue]) -> InterpreterValue + Send + Sync>;

pub type TrampolineFn = unsafe extern "C" fn(env: *mut u8, args_results: *mut InterpreterValue, arity: usize);

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
            InterpreterValue::None
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

    pub fn register_raw(
        &mut self,
        name: String,
        f: HostFunction,
    ) -> usize {
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

        let env = Box::into_raw(Box::new(f)) as *mut u8;
        let drop_fn = |ptr: *mut u8| {
            unsafe {
                let _ = Box::from_raw(ptr as *mut HostFunction);
            }
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

        let env = Box::into_raw(Box::new(func)) as *mut u8;
        let drop_fn = |ptr: *mut u8| {
            unsafe {
                let _ = Box::from_raw(ptr as *mut F);
            }
        };

        let host_func = HostFunc(Arc::new(HostFunctionInner {
            handler: trampoline::<F, Args, Rets>,
            env,
            drop_fn,
        }));

        self.host_functions.insert(name, host_func.clone());
        let id = self.host_functions_list.len();
        self.host_functions_list.push(host_func);
        id
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

pub struct VM {
    pub memories: Vec<MemoryRegion>,
}

impl VM {
    pub fn new() -> Self {
        Self {
            memories: Vec::new(),
        }
    }

    pub fn register_region(&mut self, region: MemoryRegion) {
        self.memories.push(region);
    }

    pub fn clear_regions(&mut self) {
        self.memories.clear();
    }

    #[inline(always)]
    fn translate_addr(&self, logical_addr: usize, size: usize) -> Option<*mut u8> {
        // 性能优化：对于单内存（如标准 Wasm）进行快速处理
        if self.memories.len() == 1 {
            let region = &self.memories[0];
            if logical_addr >= region.logical_base
                && logical_addr + size <= region.logical_base + region.size
            {
                return unsafe { Some(region.host_base.add(logical_addr - region.logical_base)) };
            }
            return None;
        }

        // 通用多内存搜索
        for region in &self.memories {
            if logical_addr >= region.logical_base
                && logical_addr + size <= region.logical_base + region.size
            {
                return unsafe { Some(region.host_base.add(logical_addr - region.logical_base)) };
            }
        }
        None
    }
}

pub struct Interpreter {
    pub module_id: ModuleId,
    pub(crate) stack: Vec<StackFrame>,
}

struct StackFrame {
    module_id: ModuleId,
    func_id: FuncId,
    values: Vec<InterpreterValue>,
    stack_slots: Vec<Vec<u8>>,
    current_block: Block,
    inst_idx: usize,
    block_args: Vec<InterpreterValue>,
    pending_result_target: Option<veloc_ir::Value>,
}

impl Interpreter {
    pub fn new(module_id: ModuleId) -> Self {
        Self {
            module_id,
            stack: Vec::new(),
        }
    }

    pub fn run_function(
        &mut self,
        program: &Program,
        vm: &mut VM,
        func_id: FuncId,
        args: &[InterpreterValue],
    ) -> InterpreterValue {
        let initial_stack_depth = self.stack.len();
        let mut last_result = InterpreterValue::None;

        // 初始帧
        let module = &program.modules[self.module_id.0];
        let func = &module.functions[func_id];

        // 如果是外部函数，直接执行并返回
        if func.entry_block.is_none() {
            if let Some(host_fn) = program.host_functions.get(&func.name).cloned() {
                let mut args_vec = args.to_vec();
                return host_fn.call(&mut args_vec);
            }
            panic!("External function {} not registered", func.name);
        }

        let first_frame = self.create_frame(program, self.module_id, func_id, args.to_vec());
        self.stack.push(first_frame);

        while self.stack.len() > initial_stack_depth {
            let mut frame = self.stack.pop().unwrap();
            self.module_id = frame.module_id;
            // 如果有待处理的返回值，将其存入之前记录的结果目标中
            if let Some(target) = frame.pending_result_target.take() {
                frame.values[target.0 as usize] = last_result;
                last_result = InterpreterValue::None;
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
                        frame.values[param.0 as usize] = *arg;
                    }
                }

                while frame.inst_idx < block_data.insts.len() {
                    let inst = block_data.insts[frame.inst_idx];
                    let idata = &func.dfg.instructions[inst];
                    let res_val = func.dfg.inst_results(inst);

                    frame.inst_idx += 1;

                    let flow = self.execute_inst(
                        program,
                        vm,
                        idata,
                        &mut frame.values,
                        &mut frame.stack_slots,
                        func_ptr,
                    );
                    match flow {
                        ControlFlow::Continue(res) => {
                            if let Some(rv) = res_val {
                                frame.values[rv.0 as usize] = res;
                            }
                        }
                        ControlFlow::Call(m_id, f_id, c_args) => {
                            let callee_module = &program.modules[m_id.0];
                            let callee_func = &callee_module.functions[f_id];

                            if callee_func.entry_block.is_none() {
                                if let Some(host_fn) =
                                    program.host_functions.get(&callee_func.name).cloned()
                                {
                                    let mut c_args = c_args;
                                    let h_res = host_fn.call(&mut c_args);
                                    if let Some(rv) = res_val {
                                        frame.values[rv.0 as usize] = h_res;
                                    }
                                } else {
                                    panic!("External function {} not registered", callee_func.name);
                                }
                            } else {
                                frame.pending_result_target = res_val;
                                self.stack.push(frame);
                                let new_frame = self.create_frame(program, m_id, f_id, c_args);
                                self.stack.push(new_frame);
                                break 'frame_loop;
                            }
                        }
                        ControlFlow::Return(res) => {
                            last_result = res;
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
        &self,
        program: &Program,
        module_id: ModuleId,
        func_id: FuncId,
        args: Vec<InterpreterValue>,
    ) -> StackFrame {
        let module = &program.modules[module_id.0];
        let func = &module.functions[func_id];
        let values = vec![InterpreterValue::None; func.dfg.values.len()];
        let stack_slots = func
            .stack_slots
            .iter()
            .map(|(_, data)| vec![0; data.size as usize])
            .collect();
        let current_block = func.entry_block.expect("Function has no entry block");

        StackFrame {
            module_id,
            func_id,
            values,
            stack_slots,
            current_block,
            inst_idx: 0,
            block_args: args,
            pending_result_target: None,
        }
    }

    fn execute_inst(
        &mut self,
        program: &Program,
        vm: &mut VM,
        idata: &InstructionData,
        values: &mut [InterpreterValue],
        stack_slots: &mut [Vec<u8>],
        func_ptr: *const Function,
    ) -> ControlFlow {
        let func = unsafe { &*func_ptr };
        match idata {
            InstructionData::Iconst { value, ty } => match ty {
                Type::I8 | Type::I16 | Type::I32 => {
                    ControlFlow::Continue(InterpreterValue::I32(*value as i32))
                }
                Type::I64 | Type::Ptr => ControlFlow::Continue(InterpreterValue::I64(*value)),
                Type::Bool => ControlFlow::Continue(InterpreterValue::Bool(*value != 0)),
                _ => panic!("Invalid type for iconst: {:?}", ty),
            },
            InstructionData::Fconst { value, ty } => {
                if *ty == Type::F32 {
                    ControlFlow::Continue(InterpreterValue::F32(f32::from_bits(*value as u32)))
                } else {
                    ControlFlow::Continue(InterpreterValue::F64(f64::from_bits(*value)))
                }
            }
            InstructionData::Bconst { value } => {
                ControlFlow::Continue(InterpreterValue::Bool(*value))
            }
            InstructionData::Binary { opcode, args, .. } => {
                let lhs = values[args[0].0 as usize];
                let rhs = values[args[1].0 as usize];
                ControlFlow::Continue(self.exec_binary(*opcode, lhs, rhs))
            }
            InstructionData::Unary { opcode, arg, ty } => {
                let val = values[arg.0 as usize];
                let arg_ty = func.dfg.values[*arg].ty;
                ControlFlow::Continue(self.exec_unary(*opcode, val, arg_ty, *ty))
            }
            InstructionData::IntCompare { kind, args, .. } => {
                let lhs = values[args[0].0 as usize];
                let rhs = values[args[1].0 as usize];
                ControlFlow::Continue(self.exec_icmp(*kind, lhs, rhs))
            }
            InstructionData::FloatCompare { kind, args, .. } => {
                let lhs = values[args[0].0 as usize];
                let rhs = values[args[1].0 as usize];
                ControlFlow::Continue(self.exec_fcmp(*kind, lhs, rhs))
            }
            InstructionData::Select {
                condition,
                then_val,
                else_val,
                ..
            } => {
                let cond = match values[condition.0 as usize] {
                    InterpreterValue::Bool(b) => b,
                    InterpreterValue::I32(v) => v != 0,
                    InterpreterValue::I64(v) => v != 0,
                    _ => panic!("Invalid condition type for select"),
                };
                let val = if cond {
                    values[then_val.0 as usize]
                } else {
                    values[else_val.0 as usize]
                };
                ControlFlow::Continue(val)
            }
            InstructionData::Load { ptr, offset, ty } => {
                let addr = values[ptr.0 as usize].unwarp_i64() as usize + *offset as usize;
                let res = self.read_memory(vm, addr, *ty);
                ControlFlow::Continue(res)
            }
            InstructionData::Store { ptr, value, offset } => {
                let addr = values[ptr.0 as usize].unwarp_i64() as usize + *offset as usize;
                let val = values[value.0 as usize];
                let ty = func.dfg.values[*value].ty;
                self.write_memory(vm, addr, val, ty);
                ControlFlow::Continue(InterpreterValue::None)
            }
            InstructionData::StackAddr { slot, offset } => {
                let addr = stack_slots[(*slot).0 as usize].as_mut_ptr() as i64 + (*offset as i64);
                ControlFlow::Continue(InterpreterValue::I64(addr))
            }
            InstructionData::StackStore {
                slot,
                value,
                offset,
            } => {
                let val = values[value.0 as usize];
                let ty = func.dfg.values[*value].ty;
                let slot_data = &mut stack_slots[slot.0 as usize];
                if let InterpreterValue::None = val {
                    panic!(
                        "Interpreter error: attempting to store None to stack slot {:?} at offset {} from value {:?}. Instruction: {:?}",
                        slot, offset, value, idata
                    );
                }
                let bytes = match (val, ty) {
                    (InterpreterValue::I32(v), Type::I8) => (v as u8).to_le_bytes().to_vec(),
                    (InterpreterValue::I32(v), Type::I16) => (v as u16).to_le_bytes().to_vec(),
                    (InterpreterValue::I32(v), _) => v.to_le_bytes().to_vec(),
                    (InterpreterValue::I64(v), Type::I8) => (v as u8).to_le_bytes().to_vec(),
                    (InterpreterValue::I64(v), Type::I16) => (v as u16).to_le_bytes().to_vec(),
                    (InterpreterValue::I64(v), Type::I32) => (v as u32).to_le_bytes().to_vec(),
                    (InterpreterValue::I64(v), _) => v.to_le_bytes().to_vec(),
                    (InterpreterValue::F32(v), _) => v.to_bits().to_le_bytes().to_vec(),
                    (InterpreterValue::F64(v), _) => v.to_bits().to_le_bytes().to_vec(),
                    _ => panic!(
                        "Unsupported stack store value: {:?} with type {:?}",
                        val, ty
                    ),
                };
                let off = *offset as usize;
                slot_data[off..off + bytes.len()].copy_from_slice(&bytes);
                ControlFlow::Continue(InterpreterValue::None)
            }
            InstructionData::StackLoad { slot, offset, ty } => {
                let slot_data = &stack_slots[slot.0 as usize];
                let off = *offset as usize;
                let res = match ty {
                    Type::I8 => InterpreterValue::I32(slot_data[off] as i32),
                    Type::I16 => {
                        let mut b = [0u8; 2];
                        b.copy_from_slice(&slot_data[off..off + 2]);
                        InterpreterValue::I32(u16::from_le_bytes(b) as i32)
                    }
                    Type::I32 => {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&slot_data[off..off + 4]);
                        InterpreterValue::I32(i32::from_le_bytes(b))
                    }
                    Type::I64 => {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slot_data[off..off + 8]);
                        InterpreterValue::I64(i64::from_le_bytes(b))
                    }
                    Type::F32 => {
                        let mut b = [0u8; 4];
                        b.copy_from_slice(&slot_data[off..off + 4]);
                        InterpreterValue::F32(f32::from_bits(u32::from_le_bytes(b)))
                    }
                    Type::F64 => {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(&slot_data[off..off + 8]);
                        InterpreterValue::F64(f64::from_bits(u64::from_le_bytes(b)))
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
                    .map(|&v| values[v.0 as usize])
                    .collect();
                ControlFlow::Jump(dest_data.block, j_args)
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                let cond = match values[condition.0 as usize] {
                    InterpreterValue::Bool(b) => b,
                    InterpreterValue::I32(v) => v != 0,
                    InterpreterValue::I64(v) => v != 0,
                    _ => panic!("Invalid condition type"),
                };
                let dest = if cond { then_dest } else { else_dest };
                let dest_data = func.dfg.block_calls[*dest];
                let j_args = func
                    .dfg
                    .get_value_list(dest_data.args)
                    .iter()
                    .map(|&v| values[v.0 as usize])
                    .collect();
                ControlFlow::Jump(dest_data.block, j_args)
            }
            InstructionData::Return { value } => {
                let res = if let Some(v) = value {
                    values[v.0 as usize]
                } else {
                    InterpreterValue::None
                };
                ControlFlow::Return(res)
            }
            InstructionData::BrTable { index, table } => {
                let idx = match values[index.0 as usize] {
                    InterpreterValue::I32(v) => v as usize,
                    InterpreterValue::I64(v) => v as usize,
                    _ => panic!("Invalid index type for br_table"),
                };
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
                    .map(|&v| values[v.0 as usize])
                    .collect();
                ControlFlow::Jump(target_data.block, j_args)
            }
            InstructionData::Call { func_id, args, .. } => {
                let call_args: Vec<_> = func
                    .dfg
                    .get_value_list(*args)
                    .iter()
                    .map(|&v| values[v.0 as usize])
                    .collect();
                ControlFlow::Call(self.module_id, *func_id, call_args)
            }
            InstructionData::CallIndirect {
                ptr, args, sig_id: _, ..
            } => {
                let ptr_val = values[ptr.0 as usize].unwarp_i64() as usize;
                let call_args: Vec<InterpreterValue> = func
                    .dfg
                    .get_value_list(*args)
                    .iter()
                    .map(|&v| values[v.0 as usize])
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
                    panic!("Indirect call to raw pointer {:#x} is not supported. All host functions must be registered via register_func.", ptr_val);
                }
            }
            InstructionData::IntToPtr { arg } => {
                let v = values[arg.0 as usize].unwarp_i64();
                ControlFlow::Continue(InterpreterValue::I64(v))
            }
            InstructionData::PtrToInt { arg, ty } => {
                let v = values[arg.0 as usize].unwarp_i64();
                match ty {
                    Type::I32 => ControlFlow::Continue(InterpreterValue::I32(v as i32)),
                    _ => ControlFlow::Continue(InterpreterValue::I64(v)),
                }
            }
            InstructionData::PtrOffset { ptr, offset } => {
                let p = values[ptr.0 as usize].unwarp_i64();
                ControlFlow::Continue(InterpreterValue::I64(p + *offset as i64))
            }
            InstructionData::PtrIndex {
                ptr,
                index,
                scale,
                offset,
            } => {
                let p = values[ptr.0 as usize].unwarp_i64();
                let idx = values[index.0 as usize].unwarp_i64();
                ControlFlow::Continue(InterpreterValue::I64(
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
    ) -> InterpreterValue {
        match lhs {
            InterpreterValue::I32(a) => {
                let b = rhs.unwarp_i32();
                let res = match opcode {
                    Opcode::Iadd => a.wrapping_add(b),
                    Opcode::Isub => a.wrapping_sub(b),
                    Opcode::Imul => a.wrapping_mul(b),
                    Opcode::Idiv => a.wrapping_div(b),
                    Opcode::Udiv => (a as u32 / b as u32) as i32,
                    Opcode::Irem => a.wrapping_rem(b),
                    Opcode::Urem => (a as u32 % b as u32) as i32,
                    Opcode::And => a & b,
                    Opcode::Or => a | b,
                    Opcode::Xor => a ^ b,
                    Opcode::Shl => a.wrapping_shl(b as u32),
                    Opcode::ShrS => a.wrapping_shr(b as u32),
                    Opcode::ShrU => ((a as u32).wrapping_shr(b as u32 & 31)) as i32,
                    Opcode::Rotl => a.rotate_left(b as u32),
                    Opcode::Rotr => a.rotate_right(b as u32),
                    _ => panic!("Unsupported I32 binary op: {:?}", opcode),
                };
                InterpreterValue::I32(res)
            }
            InterpreterValue::I64(a) => {
                let b = rhs.unwarp_i64();
                match opcode {
                    Opcode::Iadd => InterpreterValue::I64(a.wrapping_add(b)),
                    Opcode::Isub => InterpreterValue::I64(a.wrapping_sub(b)),
                    Opcode::Imul => InterpreterValue::I64(a.wrapping_mul(b)),
                    Opcode::Idiv => InterpreterValue::I64(a.wrapping_div(b)),
                    Opcode::Udiv => InterpreterValue::I64((a as u64 / b as u64) as i64),
                    Opcode::Irem => InterpreterValue::I64(a.wrapping_rem(b)),
                    Opcode::Urem => InterpreterValue::I64((a as u64 % b as u64) as i64),
                    Opcode::And => InterpreterValue::I64(a & b),
                    Opcode::Or => InterpreterValue::I64(a | b),
                    Opcode::Xor => InterpreterValue::I64(a ^ b),
                    Opcode::Shl => InterpreterValue::I64(a.wrapping_shl(b as u32)),
                    Opcode::ShrS => InterpreterValue::I64(a.wrapping_shr(b as u32)),
                    Opcode::ShrU => {
                        InterpreterValue::I64(((a as u64).wrapping_shr(b as u32 & 63)) as i64)
                    }
                    Opcode::Rotl => InterpreterValue::I64(a.rotate_left(b as u32)),
                    Opcode::Rotr => InterpreterValue::I64(a.rotate_right(b as u32)),
                    Opcode::Copysign => {
                        let fa = f64::from_bits(a as u64);
                        let fb = f64::from_bits(b as u64);
                        InterpreterValue::F64(fa.copysign(fb))
                    }
                    _ => panic!("Unsupported I64 binary op: {:?}", opcode),
                }
            }
            InterpreterValue::F32(a) => {
                let b = rhs.unwarp_f32();
                let res = match opcode {
                    Opcode::Fadd => a + b,
                    Opcode::Fsub => a - b,
                    Opcode::Fmul => a * b,
                    Opcode::Fdiv => a / b,
                    Opcode::Min => a.min(b),
                    Opcode::Max => a.max(b),
                    Opcode::Copysign => a.copysign(b),
                    _ => panic!("Unsupported F32 binary op: {:?}", opcode),
                };
                InterpreterValue::F32(res)
            }
            InterpreterValue::F64(a) => {
                let b = rhs.unwarp_f64();
                let res = match opcode {
                    Opcode::Fadd => a + b,
                    Opcode::Fsub => a - b,
                    Opcode::Fmul => a * b,
                    Opcode::Fdiv => a / b,
                    Opcode::Min => a.min(b),
                    Opcode::Max => a.max(b),
                    Opcode::Copysign => a.copysign(b),
                    _ => panic!("Unsupported F64 binary op: {:?}", opcode),
                };
                InterpreterValue::F64(res)
            }
            InterpreterValue::Bool(a) => {
                let b = rhs.unwarp_bool();
                let res = match opcode {
                    Opcode::And => a & b,
                    Opcode::Or => a | b,
                    Opcode::Xor => a ^ b,
                    _ => panic!("Unsupported Bool binary op: {:?}", opcode),
                };
                InterpreterValue::Bool(res)
            }
            _ => todo!("Binary op {:?} for {:?}", opcode, lhs),
        }
    }

    fn exec_icmp(
        &self,
        kind: IntCC,
        lhs: InterpreterValue,
        rhs: InterpreterValue,
    ) -> InterpreterValue {
        let res = match lhs {
            InterpreterValue::I32(a) => {
                let b = rhs.unwarp_i32();
                match kind {
                    IntCC::Eq => a == b,
                    IntCC::Ne => a != b,
                    IntCC::LtS => a < b,
                    IntCC::LeS => a <= b,
                    IntCC::GtS => a > b,
                    IntCC::GeS => a >= b,
                    IntCC::LtU => (a as u32) < (b as u32),
                    IntCC::LeU => (a as u32) <= (b as u32),
                    IntCC::GtU => (a as u32) > (b as u32),
                    IntCC::GeU => (a as u32) >= (b as u32),
                }
            }
            InterpreterValue::I64(_) | InterpreterValue::None => {
                let b = rhs.unwarp_i64();
                let a = lhs.unwarp_i64();
                match kind {
                    IntCC::Eq => a == b,
                    IntCC::Ne => a != b,
                    IntCC::LtS => a < b,
                    IntCC::LeS => a <= b,
                    IntCC::GtS => a > b,
                    IntCC::GeS => a >= b,
                    IntCC::LtU => (a as u64) < (b as u64),
                    IntCC::LeU => (a as u64) <= (b as u64),
                    IntCC::GtU => (a as u64) > (b as u64),
                    IntCC::GeU => (a as u64) >= (b as u64),
                }
            }
            InterpreterValue::Bool(a) => {
                let b = rhs.unwarp_bool();
                match kind {
                    IntCC::Eq => a == b,
                    IntCC::Ne => a != b,
                    _ => panic!("Unsupported icmp kind for Bool: {:?}", kind),
                }
            }
            _ => {
                let a = lhs.unwarp_i64();
                let b = rhs.unwarp_i64();
                match kind {
                    IntCC::Eq => a == b,
                    IntCC::Ne => a != b,
                    _ => panic!("Unsupported icmp types: {:?}, {:?}", lhs, rhs),
                }
            }
        };
        InterpreterValue::Bool(res)
    }

    fn exec_fcmp(
        &self,
        kind: FloatCC,
        lhs: InterpreterValue,
        rhs: InterpreterValue,
    ) -> InterpreterValue {
        let res = match lhs {
            InterpreterValue::F32(a) => {
                let b = rhs.unwarp_f32();
                match kind {
                    FloatCC::Eq => a == b,
                    FloatCC::Ne => a != b,
                    FloatCC::Lt => a < b,
                    FloatCC::Le => a <= b,
                    FloatCC::Gt => a > b,
                    FloatCC::Ge => a >= b,
                }
            }
            InterpreterValue::F64(a) => {
                let b = rhs.unwarp_f64();
                match kind {
                    FloatCC::Eq => a == b,
                    FloatCC::Ne => a != b,
                    FloatCC::Lt => a < b,
                    FloatCC::Le => a <= b,
                    FloatCC::Gt => a > b,
                    FloatCC::Ge => a >= b,
                }
            }
            _ => panic!("Invalid types for fcmp: {:?}", lhs),
        };
        InterpreterValue::Bool(res)
    }

    fn exec_unary(
        &self,
        opcode: Opcode,
        val: InterpreterValue,
        arg_ty: Type,
        res_ty: Type,
    ) -> InterpreterValue {
        match val {
            InterpreterValue::Bool(v) => match opcode {
                Opcode::ExtendU => {
                    let bits = if v { 1i64 } else { 0i64 };
                    if res_ty == Type::I32 {
                        InterpreterValue::I32(bits as i32)
                    } else {
                        InterpreterValue::I64(bits)
                    }
                }
                _ => panic!("Unary op {:?} for Bool", opcode),
            },
            InterpreterValue::I32(v) => match opcode {
                Opcode::Ineg => InterpreterValue::I32(v.wrapping_neg()),
                Opcode::Eqz => InterpreterValue::Bool(v == 0),
                Opcode::Clz => InterpreterValue::I32(v.leading_zeros() as i32),
                Opcode::Ctz => InterpreterValue::I32(v.trailing_zeros() as i32),
                Opcode::Popcnt => InterpreterValue::I32(v.count_ones() as i32),
                Opcode::ExtendS => {
                    let bits = v as i64;
                    match (arg_ty, res_ty) {
                        (Type::I8, Type::I32) => InterpreterValue::I32((bits as i8) as i32),
                        (Type::I8, Type::I64) | (Type::I8, Type::Ptr) => {
                            InterpreterValue::I64((bits as i8) as i64)
                        }
                        (Type::I16, Type::I32) => InterpreterValue::I32((bits as i16) as i32),
                        (Type::I16, Type::I64) | (Type::I16, Type::Ptr) => {
                            InterpreterValue::I64((bits as i16) as i64)
                        }
                        (Type::I32, Type::I64) | (Type::I32, Type::Ptr) => {
                            InterpreterValue::I64(v as i32 as i64)
                        }
                        _ => panic!("Invalid ExtendS: {:?} -> {:?}", arg_ty, res_ty),
                    }
                }
                Opcode::ExtendU => {
                    let bits = v as u64;
                    match (arg_ty, res_ty) {
                        (Type::I8, Type::I32) => InterpreterValue::I32((bits as u8) as i32),
                        (Type::I8, Type::I64) | (Type::I8, Type::Ptr) => {
                            InterpreterValue::I64((bits as u8) as i64)
                        }
                        (Type::I16, Type::I32) => InterpreterValue::I32((bits as u16) as i32),
                        (Type::I16, Type::I64) | (Type::I16, Type::Ptr) => {
                            InterpreterValue::I64((bits as u16) as i64)
                        }
                        (Type::I32, Type::I64) | (Type::I32, Type::Ptr) => {
                            InterpreterValue::I64(v as u32 as i64)
                        }
                        _ => panic!("Invalid ExtendU: {:?} -> {:?}", arg_ty, res_ty),
                    }
                }
                Opcode::Wrap => InterpreterValue::I32(v), // Already I32
                Opcode::Reinterpret => {
                    if res_ty == Type::F32 {
                        InterpreterValue::F32(f32::from_bits(v as u32))
                    } else {
                        val
                    }
                }
                Opcode::ConvertS | Opcode::ConvertU => {
                    if res_ty == Type::F32 {
                        if opcode == Opcode::ConvertS {
                            InterpreterValue::F32(v as f32)
                        } else {
                            InterpreterValue::F32(v as u32 as f32)
                        }
                    } else if res_ty == Type::F64 {
                        if opcode == Opcode::ConvertS {
                            InterpreterValue::F64(v as f64)
                        } else {
                            InterpreterValue::F64(v as u32 as f64)
                        }
                    } else {
                        panic!("Invalid Convert target: {:?}", res_ty)
                    }
                }
                _ => todo!("Unary op {:?} for I32", opcode),
            },
            InterpreterValue::I64(v) => match opcode {
                Opcode::Ineg => InterpreterValue::I64(v.wrapping_neg()),
                Opcode::Eqz => InterpreterValue::Bool(v == 0),
                Opcode::Clz => InterpreterValue::I64(v.leading_zeros() as i64),
                Opcode::Ctz => InterpreterValue::I64(v.trailing_zeros() as i64),
                Opcode::Popcnt => InterpreterValue::I64(v.count_ones() as i64),
                Opcode::Wrap => InterpreterValue::I32(v as i32),
                Opcode::ExtendS => {
                    if arg_ty == Type::I32 {
                        InterpreterValue::I64(v as i32 as i64)
                    } else {
                        val
                    }
                }
                Opcode::ExtendU => {
                    if arg_ty == Type::I32 {
                        InterpreterValue::I64(v as u32 as i64)
                    } else {
                        val
                    }
                }
                Opcode::Reinterpret => {
                    if res_ty == Type::F64 {
                        InterpreterValue::F64(f64::from_bits(v as u64))
                    } else {
                        val
                    }
                }
                Opcode::ConvertS | Opcode::ConvertU => {
                    if res_ty == Type::F32 {
                        if opcode == Opcode::ConvertS {
                            InterpreterValue::F32(v as f32)
                        } else {
                            InterpreterValue::F32(v as u64 as f32)
                        }
                    } else if res_ty == Type::F64 {
                        if opcode == Opcode::ConvertS {
                            InterpreterValue::F64(v as f64)
                        } else {
                            InterpreterValue::F64(v as u64 as f64)
                        }
                    } else {
                        panic!("Invalid Convert target: {:?}", res_ty)
                    }
                }
                _ => todo!("Unary op {:?} for I64", opcode),
            },
            InterpreterValue::F32(v) => match opcode {
                Opcode::Fneg => InterpreterValue::F32(-v),
                Opcode::Abs => InterpreterValue::F32(v.abs()),
                Opcode::Sqrt => InterpreterValue::F32(v.sqrt()),
                Opcode::Ceil => InterpreterValue::F32(v.ceil()),
                Opcode::Floor => InterpreterValue::F32(v.floor()),
                Opcode::Trunc => InterpreterValue::F32(v.trunc()),
                Opcode::Nearest => InterpreterValue::F32(v.round_ties_even()),
                Opcode::Promote => InterpreterValue::F64(v as f64),
                Opcode::Reinterpret => InterpreterValue::I32(v.to_bits() as i32),
                Opcode::TruncS | Opcode::TruncU => {
                    if res_ty == Type::I32 {
                        if opcode == Opcode::TruncS {
                            InterpreterValue::I32(v.trunc() as i32)
                        } else {
                            InterpreterValue::I32(v.trunc() as u32 as i32)
                        }
                    } else if res_ty == Type::I64 {
                        if opcode == Opcode::TruncS {
                            InterpreterValue::I64(v.trunc() as i64)
                        } else {
                            InterpreterValue::I64(v.trunc() as u64 as i64)
                        }
                    } else {
                        panic!("Invalid Trunc target: {:?}", res_ty)
                    }
                }
                _ => todo!("Unary op {:?} for F32", opcode),
            },
            InterpreterValue::F64(v) => match opcode {
                Opcode::Fneg => InterpreterValue::F64(-v),
                Opcode::Abs => InterpreterValue::F64(v.abs()),
                Opcode::Sqrt => InterpreterValue::F64(v.sqrt()),
                Opcode::Ceil => InterpreterValue::F64(v.ceil()),
                Opcode::Floor => InterpreterValue::F64(v.floor()),
                Opcode::Trunc => InterpreterValue::F64(v.trunc()),
                Opcode::Nearest => InterpreterValue::F64(v.round_ties_even()),
                Opcode::Demote => InterpreterValue::F32(v as f32),
                Opcode::Reinterpret => InterpreterValue::I64(v.to_bits() as i64),
                Opcode::TruncS | Opcode::TruncU => {
                    if res_ty == Type::I32 {
                        if opcode == Opcode::TruncS {
                            InterpreterValue::I32(v.trunc() as i32)
                        } else {
                            InterpreterValue::I32(v.trunc() as u32 as i32)
                        }
                    } else if res_ty == Type::I64 {
                        if opcode == Opcode::TruncS {
                            InterpreterValue::I64(v.trunc() as i64)
                        } else {
                            InterpreterValue::I64(v.trunc() as u64 as i64)
                        }
                    } else {
                        panic!("Invalid Trunc target: {:?}", res_ty)
                    }
                }
                _ => todo!("Unary op {:?} for F64", opcode),
            },
            _ => panic!(
                "not yet implemented: Unary op {:?} for {:?} (arg_ty: {:?}, res_ty: {:?})",
                opcode, val, arg_ty, res_ty
            ),
        }
    }

    fn read_memory(&self, vm: &VM, addr: usize, res_ty: Type) -> InterpreterValue {
        let size = self.type_size(res_ty);
        let host_ptr = vm.translate_addr(addr, size).expect("Memory out of bounds");

        unsafe {
            match res_ty {
                Type::I8 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i8);
                    InterpreterValue::I32(val as i32)
                }
                Type::I16 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i16);
                    InterpreterValue::I32(val as i32)
                }
                Type::I32 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i32);
                    InterpreterValue::I32(val)
                }
                Type::I64 | Type::Ptr => {
                    let val = core::ptr::read_unaligned(host_ptr as *const i64);
                    InterpreterValue::I64(val)
                }
                Type::F32 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const f32);
                    InterpreterValue::F32(val)
                }
                Type::F64 => {
                    let val = core::ptr::read_unaligned(host_ptr as *const f64);
                    InterpreterValue::F64(val)
                }
                _ => todo!("Implement more memory read types: {:?}", res_ty),
            }
        }
    }

    fn write_memory(&mut self, vm: &mut VM, addr: usize, val: InterpreterValue, res_ty: Type) {
        let size = self.type_size(res_ty);
        let host_ptr = vm.translate_addr(addr, size).expect("Memory out of bounds");

        unsafe {
            match (val, res_ty) {
                (InterpreterValue::I32(v), Type::I8) => {
                    core::ptr::write_unaligned(host_ptr as *mut i8, v as i8);
                }
                (InterpreterValue::I32(v), Type::I16) => {
                    core::ptr::write_unaligned(host_ptr as *mut i16, v as i16);
                }
                (InterpreterValue::I32(v), Type::I32) => {
                    core::ptr::write_unaligned(host_ptr as *mut i32, v);
                }
                (InterpreterValue::I64(v), Type::I8) => {
                    core::ptr::write_unaligned(host_ptr as *mut i8, v as i8);
                }
                (InterpreterValue::I64(v), Type::I16) => {
                    core::ptr::write_unaligned(host_ptr as *mut i16, v as i16);
                }
                (InterpreterValue::I64(v), Type::I32) => {
                    core::ptr::write_unaligned(host_ptr as *mut i32, v as i32);
                }
                (InterpreterValue::I64(v), Type::I64) | (InterpreterValue::I64(v), Type::Ptr) => {
                    core::ptr::write_unaligned(host_ptr as *mut i64, v);
                }
                (InterpreterValue::F32(v), _) => {
                    core::ptr::write_unaligned(host_ptr as *mut f32, v);
                }
                (InterpreterValue::F64(v), _) => {
                    core::ptr::write_unaligned(host_ptr as *mut f64, v);
                }
                _ => todo!(
                    "Implement more memory write types: {:?} to {:?}",
                    val,
                    res_ty
                ),
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
