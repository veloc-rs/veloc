//! Canonical textual IR printer.
//!
//! Pool indices and derived CFG metadata are deliberately not serialized.
//! Constants, signatures, branch destinations, and vector extensions are
//! printed by semantic value so the output can be parsed into a fresh module.

use crate::{
    BlockCall, FuncId, Function, Inst, MemFlags, Module, SigId, Signature, Type, Value,
    dfg::DataFlowGraph,
};
use core::fmt::{Display, Formatter, Result, Write};

/// A stable SSA spelling: optional human hint followed by the entity number.
pub struct ValueFmt<'a>(&'a DataFlowGraph, Value);

impl Display for ValueFmt<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let name = &self.0.value_names[self.1];
        if name.is_empty() {
            write!(f, "v{}", self.1.0)
        } else {
            write!(f, "{}.v{}", name, self.1.0)
        }
    }
}

pub struct InstPrinter<'a> {
    dfg: &'a DataFlowGraph,
    module: Option<&'a Module>,
}

impl<'a> InstPrinter<'a> {
    pub fn new(dfg: &'a DataFlowGraph, module: Option<&'a Module>) -> Self {
        Self { dfg, module }
    }

    pub fn fmt_inst(&self, f: &mut dyn Write, inst: Inst) -> Result {
        let data = &self.dfg.instructions[inst];
        let ty = self
            .dfg
            .inst_results(inst)
            .first()
            .map(|&value| self.dfg.value_type(value));
        self.fmt_instruction_data(f, data, ty)
    }

    pub fn fmt_inst_with_results(&self, f: &mut dyn Write, inst: Inst) -> Result {
        let results = self.dfg.inst_results(inst);
        match results {
            [] => {}
            [result] => write!(f, "{} = ", self.vf(*result))?,
            results => {
                f.write_char('(')?;
                self.fmt_values(f, results)?;
                f.write_str(") = ")?;
            }
        }
        self.fmt_inst(f, inst)
    }

    fn fmt_head(&self, f: &mut dyn Write, name: &str, ty: Option<Type>, flags: MemFlags) -> Result {
        f.write_str(name)?;
        if let Some(ty) = ty {
            write!(f, ".{ty}")?;
        }
        if flags.is_volatile() {
            f.write_str(".volatile")?;
        }
        if flags.alignment() != 1 {
            write!(f, ".align{}", flags.alignment())?;
        }
        Ok(())
    }

    pub(super) fn fmt_signature_ref(&self, f: &mut dyn Write, sig_id: SigId) -> Result {
        if let Some(module) = self.module {
            self.fmt_signature(f, &module.signatures[sig_id])
        } else {
            write!(f, "sig{}", sig_id.0)
        }
    }

    fn fmt_signature(&self, f: &mut dyn Write, sig: &Signature) -> Result {
        f.write_char('(')?;
        self.fmt_types(f, &sig.params)?;
        f.write_str(") -> ")?;
        self.fmt_ret_types(f, &sig.returns)
    }

    pub(super) fn fmt_func_ref(&self, f: &mut dyn Write, id: FuncId) -> Result {
        match self.module.and_then(|module| module.functions.get(id)) {
            Some(function) => f.write_str(&function.name),
            None => write!(f, "func{}", id.0),
        }
    }

    pub(super) fn fmt_values(&self, f: &mut dyn Write, values: &[Value]) -> Result {
        for (index, &value) in values.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}", self.vf(value))?;
        }
        Ok(())
    }

    fn fmt_types(&self, f: &mut dyn Write, types: &[Type]) -> Result {
        for (index, ty) in types.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            write!(f, "{ty}")?;
        }
        Ok(())
    }

    pub fn fmt_ret_types(&self, f: &mut dyn Write, types: &[Type]) -> Result {
        match types {
            [] => f.write_str("void"),
            [ty] => write!(f, "{ty}"),
            types => {
                f.write_char('(')?;
                self.fmt_types(f, types)?;
                f.write_char(')')
            }
        }
    }

    pub(super) fn fmt_block_call(&self, f: &mut dyn Write, call: crate::BlockCall) -> Result {
        let data = &self.dfg.block_calls[call];
        write!(f, "{}(", data.block)?;
        self.fmt_values(f, self.dfg.get_value_list(data.args))?;
        f.write_char(')')
    }

    pub(super) fn fmt_block_calls(&self, f: &mut dyn Write, calls: &[BlockCall]) -> Result {
        f.write_char('[')?;
        for (index, &call) in calls.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            self.fmt_block_call(f, call)?;
        }
        f.write_char(']')
    }

    pub(super) fn vf(&self, value: Value) -> ValueFmt<'a> {
        ValueFmt(self.dfg, value)
    }
}

pub struct FuncPrinter<'a> {
    pub func: &'a Function,
    pub module: &'a Module,
    inst_printer: InstPrinter<'a>,
}

impl<'a> FuncPrinter<'a> {
    pub fn new(func: &'a Function, module: &'a Module) -> Self {
        Self {
            func,
            module,
            inst_printer: InstPrinter::new(&func.dfg, Some(module)),
        }
    }

    pub fn print(&self, f: &mut dyn Write) -> Result {
        self.fmt_signature(f)?;
        writeln!(f)?;
        for (slot, data) in self.func.stack_slots.iter() {
            writeln!(f, "  {slot}: size {}", data.size)?;
        }
        for &block in &self.func.layout.block_order {
            self.fmt_block(f, block)?;
        }
        Ok(())
    }

    fn fmt_signature(&self, f: &mut dyn Write) -> Result {
        write!(f, "{} function {}(", self.func.linkage, self.func.name)?;
        let sig = &self.module.signatures[self.func.signature];
        self.inst_printer.fmt_types(f, &sig.params)?;
        f.write_str(") -> ")?;
        self.inst_printer.fmt_ret_types(f, &sig.returns)
    }

    fn fmt_block(&self, f: &mut dyn Write, block: crate::Block) -> Result {
        self.fmt_block_header(f, block)?;
        for &inst in &self.func.layout.blocks[block].insts {
            f.write_str("  ")?;
            self.inst_printer.fmt_inst_with_results(f, inst)?;
            writeln!(f)?;
        }
        Ok(())
    }

    fn fmt_block_header(&self, f: &mut dyn Write, block: crate::Block) -> Result {
        write!(f, "{block}(")?;
        for (index, &param) in self.func.layout.blocks[block].params.iter().enumerate() {
            if index != 0 {
                f.write_str(", ")?;
            }
            write!(
                f,
                "{}: {}",
                ValueFmt(&self.func.dfg, param),
                self.func.dfg.value_type(param)
            )?;
        }
        writeln!(f, "):")
    }
}

pub struct ModulePrinter<'a> {
    pub module: &'a Module,
}

impl<'a> ModulePrinter<'a> {
    pub const fn new(module: &'a Module) -> Self {
        Self { module }
    }

    pub fn print(&self, f: &mut dyn Write) -> Result {
        for global in &self.module.globals {
            writeln!(
                f,
                "global {}: {} ({})",
                global.name, global.ty, global.linkage
            )?;
        }
        for (index, (_, func)) in self.module.functions.iter().enumerate() {
            if index != 0 || !self.module.globals.is_empty() {
                writeln!(f)?;
            }
            FuncPrinter::new(func, self.module).print(f)?;
        }
        Ok(())
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        ModulePrinter::new(self).print(f)
    }
}

include!(concat!(env!("OUT_DIR"), "/text_printer.rs"));
