//! Environments for typed callable values. Handles are never reused; tracing uses
//! compiler-provided typed safepoint maps, not guesses about integer bit patterns.
use super::*;
use crate::bytecode::ControlSite;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering};
use hashbrown::{HashMap, HashSet};
use veloc_mir::{CallableKind, FuncId, TypeInfo};

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1);

pub(super) enum Action {
    Continue,
    Enter(usize),
    Returned,
}

struct Environment {
    module: ModuleId,
    ty: Type,
    function: FuncId,
    cleanup: Option<FuncId>,
    captures: Vec<InterpreterValue>,
    references: Arc<[usize]>,
    scope: Option<u64>,
}

#[derive(Default)]
pub(super) struct Callables {
    program: Option<Arc<()>>,
    entries: HashMap<u64, Environment>,
    external: HashSet<u64>,
    threshold: usize,
}

impl Callables {
    fn check(
        &self,
        program: &Program,
        handle: InterpreterValue,
        module: ModuleId,
        ty: Type,
    ) -> bool {
        self.entries.get(&handle.0).is_some_and(|env| {
            (env.module == module && env.ty == ty)
                || program.type_eq(env.module, env.ty, module, ty)
        })
    }

    fn collect(&mut self, roots: impl IntoIterator<Item = u64>) {
        let mut marked = HashSet::new();
        let mut work: Vec<_> = roots
            .into_iter()
            .chain(self.external.iter().copied())
            .collect();
        while let Some(id) = work.pop() {
            if !marked.insert(id) {
                continue;
            }
            if let Some(env) = self.entries.get(&id) {
                work.extend(env.references.iter().map(|&i| env.captures[i].0));
            }
        }
        self.entries.retain(|id, _| marked.contains(id));
        self.threshold = self.entries.len().saturating_mul(2).max(256);
    }
}

impl Interpreter {
    pub(super) fn begin_callables<'p>(
        &mut self,
        program: &'p Program,
        module: ModuleId,
        function: FuncId,
        args: &[InterpreterValue],
    ) -> Result<&'p [Type]> {
        let sig = program.signature(module, function)?;
        if sig.params().len() != args.len() {
            return Err(crate::Error::Message("argument count mismatch".into()));
        }
        if let Some(identity) = &self.callables.program
            && !Arc::ptr_eq(identity, &program.identity)
            && !self.callables.entries.is_empty()
        {
            return Err(crate::Error::Message(
                "live callables belong to another program".into(),
            ));
        }
        self.callables.program = Some(program.identity.clone());
        for (&arg, &ty) in args.iter().zip(sig.params()) {
            if ty.is_callable()
                && (!self.callables.check(program, arg, module, ty)
                    || !self.callables.external.contains(&arg.0)
                    || matches!(ty.as_callable(), Some((_, CallableKind::Local))))
            {
                return Err(crate::Error::Message("invalid callable argument".into()));
            }
        }
        let mut moved = HashSet::new();
        for (&arg, &ty) in args.iter().zip(sig.params()) {
            if ty.is_owned() && !moved.insert(arg.0) {
                return Err(crate::Error::Message(
                    "one-shot argument passed more than once".into(),
                ));
            }
        }
        for id in moved {
            self.callables.external.remove(&id);
        }
        self.next_scope = self
            .next_scope
            .checked_add(1)
            .expect("scope identity exhausted");
        Ok(sig.returns())
    }

    pub(super) fn end_callables(&mut self, results: &[Type], success: bool) {
        if success {
            for (&value, ty) in self.results_buffer.iter().zip(results) {
                if ty.is_callable() {
                    self.callables.external.insert(value.0);
                }
            }
        }
        self.callables.collect([]);
    }

    /// Release a shared callable returned to the host. Owned values must be
    /// consumed by call-value, tail-call-value, or closure-drop, so cleanup is
    /// never implicit.
    pub fn release_shared(&mut self, value: InterpreterValue) -> Result<()> {
        let env = self
            .callables
            .entries
            .get(&value.0)
            .ok_or_else(|| crate::Error::Message("unknown callable".into()))?;
        if !matches!(env.ty.as_callable(), Some((_, CallableKind::Shared))) {
            return Err(crate::Error::Message(
                "only shared callables can be released by the host".into(),
            ));
        }
        self.callables.external.remove(&value.0);
        self.callables.collect([]);
        Ok(())
    }

    pub fn live_callables(&self) -> usize {
        self.callables.entries.len()
    }

    fn collect_callables(&mut self, frame: &StackFrame) {
        if self.callables.entries.len() < self.callables.threshold {
            return;
        }
        let mut roots = Vec::new();
        for frame in self.frames.iter().chain(core::iter::once(frame)) {
            if let Some(regs) = frame.func.roots.get(&frame.roots_pc) {
                roots.extend(
                    regs.iter()
                        .map(|r| self.value_stack[frame.base + r.index() as usize].0),
                );
            }
        }
        self.callables.collect(roots);
    }

    pub(super) fn execute_control(
        &mut self,
        program: &Program,
        frame: &mut StackFrame,
        site: usize,
        return_pc: usize,
    ) -> core::result::Result<Action, DispatchExit> {
        let compiled = frame.func.clone();
        match &compiled.data_section.controls[site] {
            ControlSite::Create {
                dst,
                function,
                cleanup,
                ty,
                captures,
                references,
            } => {
                self.collect_callables(frame);
                let captures = captures
                    .iter()
                    .map(|r| self.value_stack[frame.base + r.index() as usize])
                    .collect();
                let id = NEXT_HANDLE
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |v| v.checked_add(1))
                    .expect("callable identities exhausted");
                let scope = matches!(ty.as_callable(), Some((_, CallableKind::Local)))
                    .then_some(frame.scope);
                self.callables.entries.insert(
                    id,
                    Environment {
                        module: frame.module,
                        ty: *ty,
                        function: *function,
                        cleanup: *cleanup,
                        captures,
                        references: references.clone(),
                        scope,
                    },
                );
                self.value_stack[frame.base + dst.index() as usize] = InterpreterValue(id);
                Ok(Action::Continue)
            }
            ControlSite::TailCall { function, args } => {
                self.args_buffer.clear();
                self.args_buffer.extend(
                    args.iter()
                        .map(|r| self.value_stack[frame.base + r.index() as usize]),
                );
                self.tail_transfer(program, frame.module, *function, frame)
            }
            ControlSite::TailCallValue { callee, ty, args } => {
                let (module, function) =
                    self.prepare_callable_args(program, frame, *callee, *ty, args)?;
                self.tail_transfer(program, module, function, frame)
            }
            ControlSite::Call {
                callee,
                ty,
                args,
                results,
            } => {
                let (module, function) =
                    self.prepare_callable_args(program, frame, *callee, *ty, args)?;
                self.call_transfer(program, module, function, results, return_pc, frame)
            }
            ControlSite::Drop { callee } => {
                let handle = self.value_stack[frame.base + callee.index() as usize];
                let env = self
                    .callables
                    .entries
                    .get(&handle.0)
                    .ok_or(DispatchExit::InvalidCallable)?;
                if !env.ty.is_owned() {
                    return Err(DispatchExit::InvalidCallable);
                }
                let env = self.callables.entries.remove(&handle.0).unwrap();
                self.callables.external.remove(&handle.0);
                self.args_buffer.clear();
                self.args_buffer.extend_from_slice(&env.captures);
                let function = env.cleanup.ok_or(DispatchExit::InvalidCallable)?;
                self.call_transfer(program, env.module, function, &[], return_pc, frame)
            }
        }
    }

    fn prepare_callable_args(
        &mut self,
        program: &Program,
        frame: &StackFrame,
        callee: Reg,
        ty: Type,
        args: &[Reg],
    ) -> core::result::Result<(ModuleId, FuncId), DispatchExit> {
        let handle = self.value_stack[frame.base + callee.index() as usize];
        if !self.callables.check(program, handle, frame.module, ty) {
            return Err(DispatchExit::InvalidCallable);
        }
        let env = &self.callables.entries[&handle.0];
        if env
            .scope
            .is_some_and(|s| frame.scope != s && !self.frames.iter().any(|f| f.scope == s))
        {
            return Err(DispatchExit::InvalidCallable);
        }
        let target = (env.module, env.function);
        self.args_buffer.clear();
        self.args_buffer.extend(
            env.captures.iter().copied().chain(
                args.iter()
                    .map(|r| self.value_stack[frame.base + r.index() as usize]),
            ),
        );
        if ty.is_owned() {
            self.callables.entries.remove(&handle.0);
            self.callables.external.remove(&handle.0);
        }
        Ok(target)
    }

    fn call_transfer(
        &mut self,
        program: &Program,
        module: ModuleId,
        function: FuncId,
        results: &[Reg],
        return_pc: usize,
        frame: &mut StackFrame,
    ) -> core::result::Result<Action, DispatchExit> {
        let dst_start = self.dst_regs_buffer.len();
        self.dst_regs_buffer.extend_from_slice(results);
        match program.call_target(module, function) {
            CallTarget::Bytecode(module, function) => {
                self.do_call(
                    program,
                    module,
                    function,
                    dst_start,
                    results.len(),
                    return_pc,
                    frame,
                )?;
                Ok(Action::Enter(0))
            }
            CallTarget::Host(host) => {
                let count = self.args_buffer.len();
                self.args_buffer
                    .resize(count.max(results.len()).max(1), InterpreterValue::none());
                program
                    .call_host(host, &mut self.args_buffer, count, results.len())
                    .map_err(|_| DispatchExit::InvalidHostCall)?;
                for (i, &dst) in results.iter().enumerate() {
                    if dst != Reg::NULL {
                        self.value_stack[frame.base + dst.index() as usize] = self.args_buffer[i];
                    }
                }
                self.dst_regs_buffer.truncate(dst_start);
                Ok(Action::Continue)
            }
        }
    }

    fn tail_transfer(
        &mut self,
        program: &Program,
        module: ModuleId,
        function: FuncId,
        frame: &mut StackFrame,
    ) -> core::result::Result<Action, DispatchExit> {
        let (module, function) = match program.call_target(module, function) {
            CallTarget::Bytecode(module, function) => (module, function),
            CallTarget::Host(host) => {
                let sig = program
                    .signature(module, function)
                    .map_err(|_| DispatchExit::InvalidHostCall)?;
                let args = self.args_buffer.len();
                let results = sig.returns().len();
                self.args_buffer
                    .resize(args.max(results).max(1), InterpreterValue::none());
                program
                    .call_host(host, &mut self.args_buffer, args, results)
                    .map_err(|_| DispatchExit::InvalidHostCall)?;
                self.args_buffer.truncate(results);
                return Ok(self
                    .return_frame(frame)
                    .map_or(Action::Returned, Action::Enter));
            }
        };
        let next = program
            .compiled_func(module, function)
            .map_err(|_| DispatchExit::InvalidFunction(module, function))?;
        if self.args_buffer.len() != next.param_indices.len() {
            return Err(DispatchExit::InvalidCallable);
        }
        let size = next.stack_size;
        // Borrowed stack addresses remain valid through a tail transfer. The
        // caller's original mark is restored only when the answer returns.
        let base = self
            .alloc_stack_frame(size, next.stack_align)
            .ok_or(DispatchExit::StackOverflow)?;
        self.value_stack.truncate(frame.base);
        self.value_stack
            .resize(frame.base + next.register_count, InterpreterValue::none());
        for (value, reg) in self.args_buffer.iter().zip(&next.param_indices) {
            self.value_stack[frame.base + reg.index() as usize] = *value;
        }
        frame.func = next;
        frame.module = module;
        frame.stack_base = base;
        frame.pc = 0;
        frame.roots_pc = 0;
        Ok(Action::Enter(0))
    }

    #[inline(always)]
    pub(super) fn return_frame(&mut self, frame: &mut StackFrame) -> Option<usize> {
        self.value_stack.truncate(frame.base);
        self.stack_top = frame.stack_mark;
        let Some(previous) = self.frames.pop() else {
            self.results_buffer.clear();
            self.results_buffer.extend_from_slice(&self.args_buffer);
            return None;
        };
        let start = previous.dst_regs_start;
        let count = previous.dst_regs_count;
        debug_assert_eq!(count, self.args_buffer.len());
        for i in 0..count {
            let dst = self.dst_regs_buffer[start + i];
            if dst != Reg::NULL {
                self.value_stack[previous.base + dst.index() as usize] = self.args_buffer[i];
            }
        }
        self.dst_regs_buffer.truncate(start);
        let pc = previous.pc;
        *frame = previous;
        Some(pc)
    }
}
