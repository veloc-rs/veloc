use crate::bytecode::{CodeWord, CompiledFunction, OpcodeHandlers, Reg, decode};
use crate::error::Result;
use crate::runtime::{CallTarget, Program};
use crate::value::InterpreterValue;
use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use veloc_mir::{ModuleId, Type};

pub trait VirtualMemory {
    fn translate_addr(&self, logical_addr: usize, size: usize) -> Option<*mut u8>;
}

pub struct Interpreter {
    callables: callable::Callables,
    next_scope: u64,
    value_stack: Vec<InterpreterValue>,
    stack_memory: Box<[u8]>,
    stack_top: usize,
    frames: Vec<StackFrame>,
    args_buffer: Vec<InterpreterValue>,
    dst_regs_buffer: Vec<Reg>,
    results_buffer: Vec<InterpreterValue>,
}

pub(crate) struct StackFrame {
    scope: u64,
    stack_mark: usize,
    roots_pc: usize,
    module: ModuleId,
    func: ::alloc::sync::Arc<CompiledFunction>,
    pc: usize,
    base: usize,
    stack_base: usize,
    dst_regs_start: usize,
    dst_regs_count: usize,
}

/// Cold interpreter state shared by every opcode handler. Keeping it behind
/// one pointer leaves registers available for the instruction and value-stack
/// pointers that change at every dispatch.
pub(crate) struct DispatchContext<M> {
    interpreter: *mut Interpreter,
    program: *const Program,
    memory: *const M,
    frame: *mut StackFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DispatchExit {
    InvalidCallSignature,
    InvalidCallable,
    Returned,
    OutOfBounds,
    StackOverflow,
    Unreachable,
    InvalidFunction(ModuleId, veloc_mir::FuncId),
    InvalidFunctionReference,
    InvalidHostCall,
}

// Opcode boundaries are tail jumps, so preserving callee-saved registers at
// every boundary is pure overhead. This ABI lets the whole handler chain keep
// interpreter state in registers without a push/pop pair per instruction. The
// table pointer is threaded through the chain so handlers do not rematerialize
// its address before every dispatch.
pub(crate) type OpcodeHandler<M> = unsafe extern "rust-preserve-none" fn(
    *mut DispatchContext<M>,
    *const CodeWord,
    *mut InterpreterValue,
    *const OpcodeHandlers<M>,
) -> DispatchExit;

/// Low-level generator shared by the semantic handler macros below.
macro_rules! define_handlers {
    (
        $context_ptr:ident,
        $interpreter:ident,
        $program:ident,
        $mem:ident,
        $frame:ident,
        $ip:ident,
        $next_ip:ident,
        $values_ptr:ident,
        $handlers:ident;
        $($rest:tt)*
    ) => {
        define_handlers!(@handlers
            [$context_ptr, $interpreter, $program, $mem, $frame, $ip, $next_ip, $values_ptr,
             $handlers]
            $($rest)*
        );
    };

    (@handlers
        [$context_ptr:ident, $interpreter:ident, $program:ident, $mem:ident, $frame:ident,
         $ip:ident, $next_ip:ident, $values_ptr:ident, $handlers:ident]
    $($(#[$meta:meta])*
        $name:ident {
            $($arg:ident $( : $binding:ident )?),* $(,)?
        } => $body:block
    )*) => {
        $(
            $(#[$meta])*
            #[allow(
                non_snake_case,
                unreachable_code,
                unused_assignments,
                unused_macros,
                unused_mut,
                unused_variables
            )]
            pub(crate) unsafe extern "rust-preserve-none" fn $name<M>(
                $context_ptr: *mut DispatchContext<M>,
                $ip: *const CodeWord,
                mut $values_ptr: *mut InterpreterValue,
                $handlers: *const OpcodeHandlers<M>,
            ) -> DispatchExit
            where
                M: VirtualMemory,
            {
                unsafe {
                    let context = &mut *$context_ptr;
                    let $interpreter = &mut *context.interpreter;
                    let $program = &*context.program;
                    let $mem = &*context.memory;
                    let $frame = &mut *context.frame;

                    macro_rules! reg {
                        ($$r:expr) => {
                            &mut *$values_ptr.add(($$r).index() as usize)
                        };
                    }
                    macro_rules! get {
                        ($$r:expr) => {
                            *$values_ptr.add(($$r).index() as usize)
                        };
                    }
                    macro_rules! set {
                        ($$d:expr, $$v:expr) => {
                            *reg!($$d) = $$v
                        };
                    }
                    macro_rules! dispatch_next {
                        ($$next:expr, $$values:expr) => {{
                            let opcode = (*$$next).opcode();
                            let handler = (*$handlers).get(opcode);
                            become handler($context_ptr, $$next, $$values, $handlers)
                        }};
                    }
                    let header = *$ip;
                    let mut $next_ip = $ip.add(crate::bytecode::Opcode::$name.words());
                    debug_assert_eq!(header.opcode(), crate::bytecode::Opcode::$name);
                    let ($(define_handlers!(@binding $arg $(, $binding)?),)*) =
                        decode::$name($ip);

                    $body
                    dispatch_next!($next_ip, $values_ptr);
                }
            }
        )*
    };

    (@binding $arg:ident, $binding:ident) => { $binding };
    (@binding $arg:ident) => { $arg };
}

/// Opcodes whose semantics only read and write virtual registers.
macro_rules! define_register_handlers {
    ($($rest:tt)*) => {
        define_handlers! {
            __context_ptr, __interpreter, __program, __mem, __frame,
            __ip, __next_ip, __values_ptr, __handlers;
            $($rest)*
        }
    };
}

/// Opcodes that additionally access linear or interpreter stack memory.
macro_rules! define_memory_handlers {
    (
        $interpreter:ident, $mem:ident, $frame:ident;
        $($rest:tt)*
    ) => {
        define_handlers! {
            __context_ptr, $interpreter, __program, $mem, $frame,
            __ip, __next_ip, __values_ptr, __handlers;
            $($rest)*
        }
    };
}

/// Opcodes allowed to redirect dispatch or switch call frames.
macro_rules! define_control_handlers {
    (
        $interpreter:ident, $program:ident, $frame:ident,
        $ip:ident, $next_ip:ident, $values_ptr:ident;
        $($rest:tt)*
    ) => {
        define_handlers! {
            __context_ptr, $interpreter, $program, __mem, $frame,
            $ip, $next_ip, $values_ptr, __handlers;
            $($rest)*
        }
    };
}

mod callable;
pub(crate) mod handlers;

impl Interpreter {
    const DEFAULT_STACK_LIMIT: usize = 1024 * 1024;

    pub fn new() -> Self {
        Self::with_stack_limit(Self::DEFAULT_STACK_LIMIT)
    }

    pub fn with_stack_limit(stack_limit: usize) -> Self {
        Self {
            callables: callable::Callables::default(),
            next_scope: 0,
            value_stack: Vec::with_capacity(4096),
            stack_memory: vec![0; stack_limit].into_boxed_slice(),
            stack_top: 0,
            frames: Vec::with_capacity(128),
            args_buffer: Vec::with_capacity(32),
            dst_regs_buffer: Vec::with_capacity(128),
            results_buffer: Vec::with_capacity(8),
        }
    }

    #[inline]
    fn alloc_stack_frame(&mut self, size: usize, align: usize) -> Option<usize> {
        let start = self.stack_memory.as_ptr() as usize;
        let address = start.checked_add(self.stack_top)?.checked_add(align - 1)? & !(align - 1);
        let base = address.checked_sub(start)?;
        let end = base.checked_add(size)?;
        if end > self.stack_memory.len() {
            return None;
        }
        self.stack_memory[base..end].fill(0);
        self.stack_top = end;
        Some(base)
    }

    fn memory_address<M: VirtualMemory>(
        &self,
        mem: &M,
        addr: usize,
        bytes: usize,
    ) -> Option<*mut u8> {
        let base = self.stack_memory.as_ptr() as usize;
        if let Some(offset) = addr.checked_sub(base) {
            if offset <= self.stack_memory.len() {
                return offset
                    .checked_add(bytes)
                    .filter(|end| *end <= self.stack_top)
                    .map(|_| addr as *mut u8);
            }
        }
        mem.translate_addr(addr, bytes)
    }

    #[inline(always)]
    unsafe fn load_memory<M, T>(&self, mem: &M, addr: usize) -> Option<T>
    where
        M: VirtualMemory,
    {
        let ptr = self.memory_address(mem, addr, core::mem::size_of::<T>())?;
        Some(unsafe { (ptr as *const T).read_unaligned() })
    }

    #[inline(always)]
    unsafe fn store_memory<M, T>(&self, mem: &M, addr: usize, value: T) -> bool
    where
        M: VirtualMemory,
    {
        let Some(ptr) = self.memory_address(mem, addr, core::mem::size_of::<T>()) else {
            return false;
        };
        unsafe { (ptr as *mut T).write_unaligned(value) };
        true
    }

    #[inline(always)]
    unsafe fn execute_moves(
        frame: &StackFrame,
        values_ptr: *mut InterpreterValue,
        target: &crate::bytecode::JumpTarget,
    ) {
        for i in 0..target.num_moves as usize {
            let (dst, src) = frame.func.data_section.jump_move_pair(target, i);
            let value = unsafe { *values_ptr.add(src.index() as usize) };
            unsafe { *values_ptr.add(dst.index() as usize) = value };
        }
    }

    #[inline(always)]
    unsafe fn read_call_data(
        &mut self,
        data_sec: &crate::bytecode::DataSection,
        values_ptr: *const InterpreterValue,
        off: usize,
        num_rets: u16,
        num_args: u16,
    ) -> usize {
        let dst_start = self.dst_regs_buffer.len();
        for i in 0..num_rets {
            self.dst_regs_buffer
                .push(data_sec.call_ret_reg(off, i as usize));
        }

        self.args_buffer.clear();
        self.args_buffer.resize(
            usize::from(num_args.max(num_rets)).max(1),
            InterpreterValue::none(),
        );
        for i in 0..num_args {
            let reg = data_sec.call_arg_reg(off, num_rets as usize, i as usize);
            self.args_buffer[i as usize] = unsafe { *values_ptr.add(reg.index() as usize) };
        }
        dst_start
    }

    pub fn run_function<M>(
        &mut self,
        program: &Program,
        mem: &M,
        module: ModuleId,
        func: veloc_mir::FuncId,
        args: &[InterpreterValue],
    ) -> Result<&[InterpreterValue]>
    where
        M: VirtualMemory,
    {
        let compiled = program.compiled_func(module, func)?;
        let frame_checkpoint = self.frames.len();
        let dst_checkpoint = self.dst_regs_buffer.len();
        let base = self.value_stack.len();
        let stack_checkpoint = self.stack_top;
        let total_stack_size: usize = compiled.stack_size;
        let Some(stack_base) = self.alloc_stack_frame(total_stack_size, compiled.stack_align)
        else {
            return Err(crate::error::Error::StackOverflow);
        };
        let result_types = match self.begin_callables(program, module, func, args) {
            Ok(types) => types,
            Err(error) => {
                self.stack_top = stack_checkpoint;
                return Err(error);
            }
        };
        let func = compiled;
        self.value_stack
            .resize(base + func.register_count, InterpreterValue::none());

        // Initialize parameters
        for (i, &new_idx) in func.param_indices.iter().enumerate() {
            self.value_stack[base + new_idx.0 as usize] = args[i];
        }

        self.frames.push(StackFrame {
            scope: self.next_scope,
            stack_mark: stack_base,
            roots_pc: 0,
            module,
            func,
            pc: 0,
            base,
            stack_base,
            dst_regs_start: 0,
            dst_regs_count: 0,
        });

        let result = self.execute(program, mem);

        // Restore the invocation boundary on both normal returns and traps so
        // an Interpreter can be safely reused after a failed guest call.
        self.value_stack.truncate(base);
        self.stack_top = stack_checkpoint;
        self.frames.truncate(frame_checkpoint);
        self.dst_regs_buffer.truncate(dst_checkpoint);
        self.args_buffer.clear();

        self.end_callables(result_types, result.is_ok());

        result?;
        Ok(&self.results_buffer)
    }

    #[inline(always)]
    fn do_call(
        &mut self,
        program: &Program,
        target_module: ModuleId,
        target_func: veloc_mir::FuncId,
        dst_regs_start: usize,
        dst_regs_count: usize,
        return_pc: usize,
        frame: &mut StackFrame,
    ) -> core::result::Result<(), DispatchExit> {
        let next_func = program
            .compiled_func(target_module, target_func)
            .map_err(|_| DispatchExit::InvalidFunction(target_module, target_func))?;
        let total_size: usize = next_func.stack_size;
        let Some(next_stack_base) = self.alloc_stack_frame(total_size, next_func.stack_align)
        else {
            return Err(DispatchExit::StackOverflow);
        };
        self.frames.push(StackFrame {
            scope: frame.scope,
            stack_mark: frame.stack_mark,
            roots_pc: frame.roots_pc,
            module: frame.module,
            func: frame.func.clone(),
            pc: return_pc,
            base: frame.base,
            stack_base: frame.stack_base,
            dst_regs_start,
            dst_regs_count,
        });
        frame.module = target_module;
        frame.func = next_func;
        frame.pc = 0;
        frame.base = self.value_stack.len();
        self.value_stack.resize(
            frame.base + frame.func.register_count,
            InterpreterValue::none(),
        );
        frame.stack_base = next_stack_base;
        frame.stack_mark = next_stack_base;
        self.next_scope += 1;
        frame.scope = self.next_scope;
        frame.roots_pc = 0;
        for (i, &new_idx) in frame.func.param_indices.iter().enumerate() {
            let val = self.args_buffer[i];
            self.value_stack[frame.base + new_idx.0 as usize] = val;
        }
        Ok(())
    }

    fn execute<M>(&mut self, program: &Program, mem: &M) -> Result<()>
    where
        M: VirtualMemory,
    {
        let mut frame = self.frames.pop().unwrap();
        self.results_buffer.clear();
        let exit = unsafe {
            let ip = frame.func.code.as_ptr().add(frame.pc);
            let values_ptr = self.value_stack.as_mut_ptr().add(frame.base);
            let mut context = DispatchContext {
                interpreter: self,
                program,
                memory: mem,
                frame: &mut frame,
            };
            let opcode = (*ip).opcode();
            let handlers = &OpcodeHandlers::<M>::TABLE;
            let handler = handlers.get(opcode);
            handler(&mut context, ip, values_ptr, handlers)
        };
        match exit {
            DispatchExit::InvalidCallSignature => Err(crate::error::Error::Message(
                "indirect call signature mismatch".into(),
            )),
            DispatchExit::InvalidCallable => Err(crate::error::Error::Message(
                "invalid, expired or already consumed callable".into(),
            )),
            DispatchExit::Returned => Ok(()),
            DispatchExit::OutOfBounds => Err(crate::error::Error::OutOfBounds),
            DispatchExit::StackOverflow => Err(crate::error::Error::StackOverflow),
            DispatchExit::Unreachable => Err(crate::error::Error::Unreachable),
            DispatchExit::InvalidFunction(module, func) => {
                Err(crate::error::Error::InvalidFunction { module, func })
            }
            DispatchExit::InvalidFunctionReference => {
                Err(crate::error::Error::InvalidFunctionReference)
            }
            DispatchExit::InvalidHostCall => Err(crate::error::Error::InvalidHostCall),
        }
    }
}
