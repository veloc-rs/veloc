//! Program runtime management
//!
//! This module provides the main Program structure that manages all loaded
//! modules, host functions, and their interactions.

use crate::bytecode::{CompiledFunction, compile_function};
use crate::error::{Error, Result};
use crate::host::{HostFuncId, HostFunction};
use crate::runtime::{CallTarget, FunctionRef, RuntimeModule};
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;
use cranelift_entity::PrimaryMap;
use hashbrown::HashMap;
use veloc_ir::{FuncId, Module, ModuleId};

/// Main program structure managing all modules and host functions
pub struct Program {
    /// Map from host function name to ID
    hosts_by_name: HashMap<String, HostFuncId>,
    /// Storage for host function implementations
    hosts: PrimaryMap<HostFuncId, HostFunction>,
    /// Loaded modules
    modules: PrimaryMap<ModuleId, RuntimeModule>,
    /// Stable opaque references used by indirect calls.
    func_refs: Vec<CallTarget>,
    /// Reference handle assigned to each host function.
    host_refs: PrimaryMap<HostFuncId, FunctionRef>,
}

/// Stages one module and commits it only after every import is linked.
pub struct ProgramBuilder<'a> {
    program: &'a mut Program,
    module: Module,
    id: ModuleId,
    targets: PrimaryMap<FuncId, Option<CallTarget>>,
}

impl Program {
    /// Start building a module without exposing partial state through `Program`.
    pub fn builder(&mut self, module: Module) -> ProgramBuilder<'_> {
        ProgramBuilder::new(self, module)
    }

    /// Get a host function ID by name
    pub fn find_host(&self, name: &str) -> Option<HostFuncId> {
        self.hosts_by_name.get(name).copied()
    }

    /// Iterate over compiled functions without allocating or cloning their `Arc`s.
    pub fn compiled_funcs(&self) -> impl Iterator<Item = (ModuleId, FuncId, &CompiledFunction)> {
        self.modules.iter().flat_map(|(module, runtime_module)| {
            runtime_module
                .compiled
                .iter()
                .filter_map(move |(func, compiled)| {
                    compiled.as_deref().map(|compiled| (module, func, compiled))
                })
        })
    }

    /// Create a new empty program
    pub fn new() -> Self {
        Self {
            hosts_by_name: HashMap::new(),
            hosts: PrimaryMap::new(),
            modules: PrimaryMap::new(),
            func_refs: Vec::new(),
            host_refs: PrimaryMap::new(),
        }
    }

    /// Get a compiled function from a module
    pub(crate) fn compiled_func(
        &self,
        module: ModuleId,
        func: FuncId,
    ) -> Result<Arc<CompiledFunction>> {
        let loaded = self
            .modules
            .get(module)
            .ok_or(Error::InvalidModule(module))?;
        let compiled = loaded
            .compiled
            .get(func)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidFunction { module, func })?;
        Ok(Arc::clone(compiled))
    }

    fn push_func_ref(&mut self, target: CallTarget) -> FunctionRef {
        let index = self.func_refs.len();
        let reference = FunctionRef::from_index(index)
            .expect("function-reference table exceeded the address space");
        self.func_refs.push(target);
        reference
    }

    /// Get the stable reference assigned to a host function.
    pub fn host_ref(&self, host: HostFuncId) -> Option<FunctionRef> {
        self.host_refs.get(host).copied()
    }

    /// Get the stable reference assigned to a compiled bytecode function.
    pub fn func_ref(&self, module: ModuleId, func: FuncId) -> Option<FunctionRef> {
        self.modules
            .get(module)?
            .func_refs
            .get(func)
            .and_then(|reference| *reference)
    }

    /// Resolve the pointer-sized function reference used by the current VM ABI.
    pub fn resolve_ref(&self, address: usize) -> Option<CallTarget> {
        let index = FunctionRef::index_from_address(address)?;
        self.func_refs.get(index).copied()
    }

    #[inline(always)]
    pub(crate) fn call_target(&self, module: ModuleId, func: FuncId) -> CallTarget {
        self.modules[module].call_targets[func]
    }

    #[inline(always)]
    pub(crate) fn call_host(
        &self,
        host: HostFuncId,
        values: &mut [crate::value::InterpreterValue],
        args: usize,
        results: usize,
    ) -> Result<()> {
        self.hosts[host].call(values, args, results)
    }

    /// Register a host function. Its signature is used to validate links and calls.
    pub fn register_host(&mut self, name: String, host: HostFunction) -> HostFuncId {
        let id = self.hosts.push(host);
        let reference = self.push_func_ref(CallTarget::Host(id));
        let ref_id = self.host_refs.push(reference);
        debug_assert_eq!(id, ref_id);
        self.hosts_by_name.insert(name, id);
        id
    }
}

impl<'a> ProgramBuilder<'a> {
    fn new(program: &'a mut Program, module: Module) -> Self {
        let id = program.modules.next_key();
        let mut targets = PrimaryMap::new();
        for (func, function) in module.functions.iter() {
            let target = function
                .is_defined()
                .then_some(CallTarget::Bytecode(id, func));
            let actual = targets.push(target);
            debug_assert_eq!(func, actual);
        }
        Self {
            program,
            module,
            id,
            targets,
        }
    }

    /// ID the module will have after a successful `finish`.
    pub fn id(&self) -> ModuleId {
        self.id
    }

    /// Find a host function registered in the target program.
    pub fn find_host(&self, name: &str) -> Option<HostFuncId> {
        self.program.find_host(name)
    }

    /// Link one import to a defined bytecode function.
    pub fn link_import(
        &mut self,
        import: FuncId,
        target_module: ModuleId,
        target_func: FuncId,
    ) -> Result<&mut Self> {
        self.validate_import(import)?;

        let target = if target_module == self.id {
            &self.module
        } else {
            &self
                .program
                .modules
                .get(target_module)
                .ok_or(Error::InvalidModule(target_module))?
                .ir
        };
        let target_data = target
            .functions
            .get(target_func)
            .ok_or(Error::InvalidFunction {
                module: target_module,
                func: target_func,
            })?;
        if !target_data.is_defined() {
            return Err(Error::InvalidFunction {
                module: target_module,
                func: target_func,
            });
        }

        let source = &self.module.functions[import];
        let source_sig = self.module.get_signature(source.signature);
        let target_sig = target.get_signature(target_data.signature);
        if source_sig != target_sig {
            return Err(Error::SignatureMismatch {
                module: self.id,
                func: import,
                target_module,
                target_func,
            });
        }

        self.targets[import] = Some(CallTarget::Bytecode(target_module, target_func));
        Ok(self)
    }

    /// Link one import to a host function.
    pub fn link_host(&mut self, import: FuncId, host: HostFuncId) -> Result<&mut Self> {
        self.validate_import(import)?;
        let host_func = self
            .program
            .hosts
            .get(host)
            .ok_or(Error::InvalidHostFunction(host))?;
        let source = &self.module.functions[import];
        let signature = self.module.get_signature(source.signature);
        if host_func.signature() != signature {
            return Err(Error::HostSignatureMismatch {
                module: self.id,
                func: import,
                host,
            });
        }
        self.targets[import] = Some(CallTarget::Host(host));
        Ok(self)
    }

    /// Resolve and link an opaque function reference owned by this program.
    pub fn link_ref(&mut self, import: FuncId, address: usize) -> Result<&mut Self> {
        match self
            .program
            .resolve_ref(address)
            .ok_or(Error::InvalidFunctionReference)?
        {
            CallTarget::Bytecode(module, func) => self.link_import(import, module, func),
            CallTarget::Host(host) => self.link_host(import, host),
        }
    }

    /// Validate, compile, and atomically add the module to the program.
    pub fn finish(self) -> Result<ModuleId> {
        for (func, function) in self.module.functions.iter() {
            if !function.is_defined() && self.targets[func].is_none() {
                return Err(Error::UnresolvedImport {
                    module: self.id,
                    func,
                });
            }
        }

        let Self {
            program,
            module,
            id,
            targets,
        } = self;
        debug_assert_eq!(id, program.modules.next_key());

        let mut compiled = PrimaryMap::new();
        let mut call_targets = PrimaryMap::new();
        let mut func_refs = PrimaryMap::new();
        for (func, function) in module.functions.iter() {
            let target = targets[func].expect("imports were validated above");
            let compiled_func = function
                .is_defined()
                .then(|| Arc::new(compile_function(id, func, function)));
            let reference = function.is_defined().then(|| program.push_func_ref(target));

            let compiled_id = compiled.push(compiled_func);
            let target_id = call_targets.push(target);
            let ref_id = func_refs.push(reference);
            debug_assert_eq!(func, compiled_id);
            debug_assert_eq!(func, target_id);
            debug_assert_eq!(func, ref_id);
        }

        let actual = program.modules.push(RuntimeModule {
            ir: module,
            compiled,
            call_targets,
            func_refs,
        });
        debug_assert_eq!(id, actual);
        Ok(actual)
    }

    fn validate_import(&self, import: FuncId) -> Result<()> {
        let func = self
            .module
            .functions
            .get(import)
            .ok_or(Error::InvalidFunction {
                module: self.id,
                func: import,
            })?;
        if func.linkage != veloc_ir::Linkage::Import {
            return Err(Error::ExpectedImport {
                module: self.id,
                func: import,
            });
        }
        Ok(())
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::InterpreterValue;
    use veloc_ir::{CallConv, Linkage, ModuleBuilder, Signature, Type};

    fn module_with_func(
        name: &str,
        linkage: Linkage,
        params: Vec<Type>,
        returns: Vec<Type>,
    ) -> (Module, FuncId) {
        let mut module = ModuleBuilder::new();
        let signature = module.make_signature(params, returns, CallConv::SystemV);
        let function = module.declare_function(name.into(), signature, linkage);
        if linkage != Linkage::Import {
            let mut builder = module.builder(function);
            builder.init_entry_block();
            builder.ins().ret(&[]);
        }
        (module.build(), function)
    }

    #[test]
    fn builder_commits_only_resolved_modules() {
        let mut program = Program::new();
        let (local_module, local_function) =
            module_with_func("local", Linkage::Export, Vec::new(), Vec::new());
        let local_module = program.builder(local_module).finish().unwrap();
        assert_eq!(
            program.modules[local_module].call_targets[local_function],
            CallTarget::Bytecode(local_module, local_function)
        );

        let reference = program.func_ref(local_module, local_function).unwrap();
        assert_eq!(
            program.resolve_ref(reference.address()),
            Some(CallTarget::Bytecode(local_module, local_function))
        );

        let (import_module, import_function) =
            module_with_func("missing", Linkage::Import, Vec::new(), Vec::new());
        assert!(matches!(
            program.builder(import_module).finish(),
            Err(Error::UnresolvedImport { func, .. }) if func == import_function
        ));
        assert_eq!(program.modules.len(), 1);
    }

    #[test]
    fn linking_checks_import_kind_and_signature() {
        let mut program = Program::new();
        let (target, target_function) =
            module_with_func("target", Linkage::Export, Vec::new(), Vec::new());
        let target = program.builder(target).finish().unwrap();

        let (source, import) =
            module_with_func("target", Linkage::Import, vec![Type::I32], Vec::new());
        let mut builder = program.builder(source);
        assert!(matches!(
            builder.link_import(import, target, target_function),
            Err(Error::SignatureMismatch { .. })
        ));
        drop(builder);

        let (source, import) = module_with_func("target", Linkage::Import, Vec::new(), Vec::new());
        let mut builder = program.builder(source);
        builder
            .link_import(import, target, target_function)
            .unwrap();
        let source = builder.finish().unwrap();
        assert_eq!(
            program.modules[source].call_targets[import],
            CallTarget::Bytecode(target, target_function)
        );

        let (defined, function) =
            module_with_func("defined", Linkage::Export, Vec::new(), Vec::new());
        let mut builder = program.builder(defined);
        assert!(matches!(
            builder.link_import(function, target, target_function),
            Err(Error::ExpectedImport { .. })
        ));
    }

    #[test]
    fn host_links_require_the_exact_signature() {
        let mut program = Program::new();
        let (module, import) =
            module_with_func("host", Linkage::Import, vec![Type::I32], vec![Type::I32]);

        let wrong = program.register_host(
            "wrong".into(),
            HostFunction::new(
                Signature::new(vec![Type::F32], vec![Type::I32], CallConv::SystemV),
                |_| {},
            ),
        );
        let host = program.register_host(
            "host".into(),
            HostFunction::new(
                Signature::new(vec![Type::I32], vec![Type::I32], CallConv::SystemV),
                |values| values[0] = InterpreterValue::i32(values[0].unwrap_i32() + 1),
            ),
        );
        let mut builder = program.builder(module);
        assert!(matches!(
            builder.link_host(import, wrong),
            Err(Error::HostSignatureMismatch { .. })
        ));
        builder.link_host(import, host).unwrap();
        let module = builder.finish().unwrap();
        assert_eq!(
            program.modules[module].call_targets[import],
            CallTarget::Host(host)
        );
    }

    #[test]
    fn reregistering_a_name_keeps_existing_hosts_stable() {
        let mut program = Program::new();
        let first = program.register_host(
            "host".into(),
            HostFunction::new(
                Signature::new(Vec::new(), vec![Type::I32], CallConv::SystemV),
                |values| values[0] = InterpreterValue::i32(1),
            ),
        );
        let reference = program.host_ref(first).unwrap();
        let second = program.register_host(
            "host".into(),
            HostFunction::new(
                Signature::new(Vec::new(), vec![Type::I32], CallConv::SystemV),
                |values| values[0] = InterpreterValue::i32(2),
            ),
        );

        assert_ne!(first, second);
        assert_eq!(program.find_host("host"), Some(second));
        assert_eq!(
            program.resolve_ref(reference.address()),
            Some(CallTarget::Host(first))
        );
        let mut buffer = [InterpreterValue::none()];
        program.hosts[first].call(&mut buffer, 0, 1).unwrap();
        assert_eq!(buffer[0].unwrap_i32(), 1);
        program.hosts[second].call(&mut buffer, 0, 1).unwrap();
        assert_eq!(buffer[0].unwrap_i32(), 2);
    }

    #[test]
    fn arbitrary_addresses_do_not_decode_as_function_references() {
        let program = Program::new();
        assert_eq!(program.resolve_ref(0), None);
        assert_eq!(program.resolve_ref(usize::MAX), None);
        assert_eq!(program.resolve_ref(0x1000), None);
    }
}
