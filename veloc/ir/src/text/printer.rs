//! Canonical textual IR printer.
//!
//! Pool indices and derived CFG metadata are deliberately not serialized.
//! Constants, signatures, branch destinations, and vector extensions are
//! printed by semantic value so the output can be parsed into a fresh module.

use super::TextCodec;
use crate::{
    Function, Inst, InstructionData, MemFlags, Module, SigId, Signature, Type, Value,
    dfg::DataFlowGraph,
    inst::{ConstantPoolData, ConstantPoolId},
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

    fn fmt_instruction_data(
        &self,
        f: &mut dyn Write,
        data: &InstructionData,
        ty: Option<Type>,
    ) -> Result {
        let opcode = data.opcode();
        let name = opcode.spec().mnemonic;
        match TextCodec::for_opcode(opcode) {
            TextCodec::Values { arity } => {
                let mut values = smallvec::SmallVec::<[crate::Value; 3]>::new();
                data.visit_type_operands(self.dfg, |value| values.push(value));
                let ext = match data {
                    InstructionData::VectorOpWithExt { ext, .. } => {
                        Some(self.dfg.vector_ext(*ext).ok_or(core::fmt::Error)?)
                    }
                    _ => None,
                };
                debug_assert_eq!(values.len(), arity as usize);
                self.fmt_head(f, name, ty, MemFlags::new())?;
                f.write_char(' ')?;
                self.fmt_values(f, &values)?;
                if let Some(ext) = ext {
                    self.fmt_named_value(f, "mask", ext.mask)?;
                    if let Some(evl) = ext.evl {
                        self.fmt_named_value(f, "evl", evl)?;
                    }
                }
                Ok(())
            }
            TextCodec::Nullary => f.write_str(name),
            TextCodec::IntegerConstant => {
                let InstructionData::Iconst { value } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {}", *value as i64)
            }
            TextCodec::FloatConstant => {
                let InstructionData::Fconst { value } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                if ty == Some(Type::F32) {
                    write!(f, " 0x{:08x}", *value as u32)
                } else {
                    write!(f, " 0x{value:016x}")
                }
            }
            TextCodec::BoolConstant => {
                let InstructionData::Bconst { value } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {value}")
            }
            TextCodec::VectorConstant => {
                let InstructionData::Vconst { pool_id } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                f.write_str(" 0x")?;
                self.fmt_pool_bytes(f, *pool_id)
            }
            TextCodec::Load => {
                let InstructionData::Load { ptr, offset, flags } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, *flags)?;
                write!(f, " {}", self.vf(*ptr))?;
                self.fmt_named_nonzero(f, "offset", *offset)
            }
            TextCodec::Store => {
                let InstructionData::Store {
                    ptr,
                    value,
                    offset,
                    flags,
                } = data
                else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, None, *flags)?;
                write!(f, " {}, {}", self.vf(*value), self.vf(*ptr))?;
                self.fmt_named_nonzero(f, "offset", *offset)
            }
            TextCodec::StackLoad => {
                let InstructionData::StackLoad { slot, offset } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {slot}")?;
                self.fmt_named_nonzero(f, "offset", *offset)
            }
            TextCodec::StackStore => {
                let InstructionData::StackStore {
                    slot,
                    value,
                    offset,
                } = data
                else {
                    return Err(core::fmt::Error);
                };
                write!(f, "{name} {}, {slot}", self.vf(*value))?;
                self.fmt_named_nonzero(f, "offset", *offset)
            }
            TextCodec::StackAddr => {
                let InstructionData::StackAddr { slot, offset } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {slot}")?;
                self.fmt_named_nonzero(f, "offset", *offset)
            }
            TextCodec::PtrOffset => {
                let InstructionData::PtrOffset { ptr, offset } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {}, {offset}", self.vf(*ptr))
            }
            TextCodec::PtrIndex => {
                let InstructionData::PtrIndex { ptr, index, imm_id } = data else {
                    return Err(core::fmt::Error);
                };
                let imm = self.dfg.ptr_imm(*imm_id).ok_or(core::fmt::Error)?;
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {}, {}", self.vf(*ptr), self.vf(*index))?;
                if imm.scale != 1 {
                    write!(f, ", scale={}", imm.scale)?;
                }
                if imm.offset != 0 {
                    write!(f, ", offset={}", imm.offset)?;
                }
                Ok(())
            }
            TextCodec::DirectCall => {
                let InstructionData::Call { func_id, args } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                let target = self
                    .module
                    .and_then(|module| module.functions.get(*func_id))
                    .map_or_else(
                        || alloc::format!("func{}", func_id.0),
                        |func| func.name.clone(),
                    );
                write!(f, " {target}(")?;
                self.fmt_values(f, self.dfg.get_value_list(*args))?;
                f.write_char(')')
            }
            TextCodec::IndirectCall => {
                let InstructionData::CallIndirect { ptr, args, sig_id } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {}(", self.vf(*ptr))?;
                self.fmt_values(f, self.dfg.get_value_list(*args))?;
                f.write_str(") : ")?;
                self.fmt_signature_ref(f, *sig_id)
            }
            TextCodec::IntrinsicCall => {
                let InstructionData::CallIntrinsic {
                    intrinsic,
                    args,
                    sig_id,
                } = data
                else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {intrinsic}(")?;
                self.fmt_values(f, self.dfg.get_value_list(*args))?;
                f.write_str(") : ")?;
                self.fmt_signature_ref(f, *sig_id)
            }
            TextCodec::Jump => {
                let InstructionData::Jump { dest } = data else {
                    return Err(core::fmt::Error);
                };
                write!(f, "{name} ")?;
                self.fmt_block_call(f, *dest)
            }
            TextCodec::Branch => {
                let InstructionData::Br {
                    condition,
                    then_dest,
                    else_dest,
                } = data
                else {
                    return Err(core::fmt::Error);
                };
                write!(f, "{name} {}, ", self.vf(*condition))?;
                self.fmt_block_call(f, *then_dest)?;
                f.write_str(", ")?;
                self.fmt_block_call(f, *else_dest)
            }
            TextCodec::BranchTable => {
                let InstructionData::BrTable { index, table } = data else {
                    return Err(core::fmt::Error);
                };
                let (default, cases) = self
                    .dfg
                    .jump_table_targets(*table)
                    .split_last()
                    .ok_or(core::fmt::Error)?;
                write!(f, "{name} {}, [", self.vf(*index))?;
                for (position, &target) in cases.iter().enumerate() {
                    if position != 0 {
                        f.write_str(", ")?;
                    }
                    self.fmt_block_call(f, target)?;
                }
                f.write_str("], ")?;
                self.fmt_block_call(f, *default)
            }
            TextCodec::Return => {
                let InstructionData::Return { values } = data else {
                    return Err(core::fmt::Error);
                };
                f.write_str(name)?;
                let values = self.dfg.get_value_list(*values);
                if !values.is_empty() {
                    f.write_char(' ')?;
                    self.fmt_values(f, values)?;
                }
                Ok(())
            }
            TextCodec::IntCompare => {
                let InstructionData::IntCompare { kind, args } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {} {}, {}", kind, self.vf(args[0]), self.vf(args[1]))
            }
            TextCodec::FloatCompare => {
                let InstructionData::FloatCompare { kind, args } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {} {}, {}", kind, self.vf(args[0]), self.vf(args[1]))
            }
            codec @ (TextCodec::VectorLoadStrided
            | TextCodec::VectorStoreStrided
            | TextCodec::VectorGather
            | TextCodec::VectorScatter) => self.fmt_vector_memory(f, name, ty, codec, data),
            TextCodec::Shuffle => {
                let InstructionData::Shuffle { args, mask } = data else {
                    return Err(core::fmt::Error);
                };
                self.fmt_head(f, name, ty, MemFlags::new())?;
                write!(f, " {}, {}, mask=0x", self.vf(args[0]), self.vf(args[1]))?;
                self.fmt_pool_bytes(f, *mask)
            }
        }
    }

    fn fmt_vector_memory(
        &self,
        f: &mut dyn Write,
        name: &str,
        ty: Option<Type>,
        codec: TextCodec,
        data: &InstructionData,
    ) -> Result {
        let (core, stride, index, ext_id) = match (codec, data) {
            (
                TextCodec::VectorLoadStrided,
                InstructionData::VectorLoadStrided { ptr, stride, ext },
            ) => (core::slice::from_ref(ptr), Some(*stride), None, *ext),
            (TextCodec::VectorStoreStrided, InstructionData::VectorStoreStrided { args, ext }) => {
                let args = self.dfg.get_value_list(*args);
                (&[args[2], args[0]][..], Some(args[1]), None, *ext)
            }
            (TextCodec::VectorGather, InstructionData::VectorGather { ptr, index, ext }) => {
                (core::slice::from_ref(ptr), None, Some(*index), *ext)
            }
            (TextCodec::VectorScatter, InstructionData::VectorScatter { args, ext }) => {
                let args = self.dfg.get_value_list(*args);
                (&[args[2], args[0]][..], None, Some(args[1]), *ext)
            }
            _ => return Err(core::fmt::Error),
        };
        let ext = self.dfg.vector_mem_ext(ext_id).ok_or(core::fmt::Error)?;
        let result_ty = match codec {
            TextCodec::VectorLoadStrided | TextCodec::VectorGather => ty,
            _ => None,
        };
        self.fmt_head(f, name, result_ty, ext.flags)?;
        f.write_char(' ')?;
        self.fmt_values(f, core)?;
        if let Some(stride) = stride {
            self.fmt_named_value(f, "stride", stride)?;
        }
        if let Some(index) = index {
            self.fmt_named_value(f, "index", index)?;
            if ext.scale != 1 {
                write!(f, ", scale={}", ext.scale)?;
            }
        }
        if ext.offset != 0 {
            write!(f, ", offset={}", ext.offset)?;
        }
        if let Some(mask) = ext.mask {
            self.fmt_named_value(f, "mask", mask)?;
        }
        if let Some(evl) = ext.evl {
            self.fmt_named_value(f, "evl", evl)?;
        }
        Ok(())
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

    fn fmt_signature_ref(&self, f: &mut dyn Write, sig_id: SigId) -> Result {
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

    fn fmt_pool_bytes(&self, f: &mut dyn Write, id: ConstantPoolId) -> Result {
        let ConstantPoolData::Bytes(bytes) =
            self.dfg.constant_pool_data(id).ok_or(core::fmt::Error)?;
        for byte in bytes {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }

    fn fmt_named_value(&self, f: &mut dyn Write, key: &str, value: Value) -> Result {
        write!(f, ", {key}={}", self.vf(value))
    }

    fn fmt_named_nonzero(&self, f: &mut dyn Write, key: &str, value: u32) -> Result {
        if value == 0 {
            Ok(())
        } else {
            write!(f, ", {key}={value}")
        }
    }

    fn fmt_values(&self, f: &mut dyn Write, values: &[Value]) -> Result {
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

    fn fmt_block_call(&self, f: &mut dyn Write, call: crate::BlockCall) -> Result {
        let data = &self.dfg.block_calls[call];
        write!(f, "{}(", data.block)?;
        self.fmt_values(f, self.dfg.get_value_list(data.args))?;
        f.write_char(')')
    }

    fn vf(&self, value: Value) -> ValueFmt<'a> {
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
