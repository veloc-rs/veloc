use super::Linkage;
use super::function::Function;
use super::module::Module;
use crate::inst::InstructionData;
use crate::{DataFlowGraph, Inst, Type, Value};
use core::fmt::{Display, Formatter, Result, Write};

/// 格式化值引用，优先使用命名，否则使用 vN 格式
pub struct ValueFmt<'a>(&'a DataFlowGraph, Value);

impl<'a> Display for ValueFmt<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let name = &self.0.value_names[self.1];
        if name.is_empty() {
            write!(f, "v{}", self.1.0)
        } else {
            write!(f, "{}", name)
        }
    }
}

/// 指令格式化器，用于将指令格式化为可读字符串
pub struct InstPrinter<'a> {
    dfg: &'a DataFlowGraph,
    module: Option<&'a Module>,
}

impl<'a> InstPrinter<'a> {
    /// 创建新的指令格式化器
    pub fn new(dfg: &'a DataFlowGraph, module: Option<&'a Module>) -> Self {
        Self { dfg, module }
    }

    /// 格式化单个指令（不包含结果值部分）
    pub fn fmt_inst(&self, f: &mut dyn Write, inst: Inst) -> Result {
        let idata = &self.dfg.instructions[inst];
        let ty = self
            .dfg
            .inst_results(inst)
            .first()
            .map(|&r| self.dfg.value_type(r));
        self.fmt_instruction_data(f, idata, ty)
    }

    /// 格式化单个指令的完整形式（包含结果值）
    pub fn fmt_inst_with_results(&self, f: &mut dyn Write, inst: Inst) -> Result {
        let results = self.dfg.inst_results(inst);
        if !results.is_empty() {
            if results.len() == 1 {
                write!(f, "{} = ", ValueFmt(self.dfg, results[0]))?;
            } else {
                write!(f, "(")?;
                for (i, &r) in results.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ValueFmt(self.dfg, r))?;
                }
                write!(f, ") = ")?;
            }
        }
        self.fmt_inst(f, inst)
    }

    fn vf(&self, v: Value) -> ValueFmt<'a> {
        ValueFmt(self.dfg, v)
    }

    fn fmt_value_list(&self, f: &mut dyn Write, values: &[Value]) -> Result {
        for (i, &val) in values.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.vf(val))?;
        }
        Ok(())
    }

    fn fmt_block_call(&self, f: &mut dyn Write, block_call: crate::BlockCall) -> Result {
        let data = &self.dfg.block_calls[block_call];
        write!(f, "{}(", data.block)?;
        let args = self.dfg.get_value_list(data.args);
        self.fmt_value_list(f, args)?;
        write!(f, ")")
    }

    fn fmt_ret_types(&self, f: &mut dyn Write, ret: &[Type]) -> Result {
        match ret.len() {
            0 => write!(f, "void"),
            1 => write!(f, "{}", ret[0]),
            _ => {
                write!(f, "(")?;
                for (i, ty) in ret.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
        }
    }

    fn fmt_mem_flags(&self, f: &mut dyn Write, flags: &crate::MemFlags) -> Result {
        if flags.is_trusted() {
            write!(f, ".trusted")?;
        }
        if flags.alignment() != 1 {
            write!(f, ".align{}", flags.alignment())?;
        }
        Ok(())
    }

    fn fmt_instruction_data(
        &self,
        f: &mut dyn Write,
        idata: &InstructionData,
        ty: Option<Type>,
    ) -> Result {
        match idata {
            InstructionData::Unary { opcode, arg } => {
                write!(f, "{}.{:?} {}", opcode, ty.unwrap(), self.vf(*arg))
            }
            InstructionData::Binary { opcode, args } => {
                write!(
                    f,
                    "{}.{:?} {}, {}",
                    opcode,
                    ty.unwrap(),
                    self.vf(args[0]),
                    self.vf(args[1])
                )
            }
            InstructionData::Ternary { opcode, args } => {
                write!(
                    f,
                    "{}.{:?} {}, {}, {}",
                    opcode,
                    ty.unwrap(),
                    self.vf(args[0]),
                    self.vf(args[1]),
                    self.vf(args[2])
                )
            }
            InstructionData::Load { ptr, offset, flags } => {
                write!(f, "load.{}", ty.unwrap())?;
                self.fmt_mem_flags(f, flags)?;
                write!(f, " {} + {}", self.vf(*ptr), offset)
            }
            InstructionData::Store {
                ptr,
                value,
                offset,
                flags,
            } => {
                write!(f, "store")?;
                self.fmt_mem_flags(f, flags)?;
                write!(f, " {}, {} + {}", self.vf(*value), self.vf(*ptr), offset)
            }
            InstructionData::StackLoad { slot, offset } => {
                write!(f, "stack_load.{} {} + {}", ty.unwrap(), slot, offset)
            }
            InstructionData::StackStore {
                slot,
                value,
                offset,
            } => {
                write!(
                    f,
                    "stack_store {} -> {} + {}",
                    self.vf(*value),
                    slot,
                    offset
                )
            }
            InstructionData::StackAddr { slot, offset } => {
                write!(f, "stack_addr {} + {}", slot, offset)
            }
            InstructionData::Iconst { value } => {
                write!(f, "iconst.{} {}", ty.unwrap(), *value as i64)
            }
            InstructionData::Fconst { value } => {
                let t = ty.unwrap();
                let val = *value;
                if t == Type::F32 {
                    let float = f32::from_bits(val as u32);
                    write!(f, "fconst.{} {:?} (0x{:08x})", t, float, val as u32)
                } else {
                    let float = f64::from_bits(val);
                    write!(f, "fconst.{} {:?} (0x{:016x})", t, float, val)
                }
            }
            InstructionData::Bconst { value } => {
                write!(f, "bconst {}", value)
            }
            InstructionData::Vconst { pool_id } => {
                write!(f, "vconst.{} {}", ty.unwrap(), pool_id)
            }
            InstructionData::IntToPtr { arg } => {
                write!(f, "inttoptr {}", self.vf(*arg))
            }
            InstructionData::PtrToInt { arg } => {
                write!(f, "ptrtoint.{} {}", ty.unwrap(), self.vf(*arg))
            }
            InstructionData::PtrOffset { ptr, offset } => {
                write!(f, "ptr_offset {}, {}", self.vf(*ptr), offset)
            }
            InstructionData::PtrIndex { ptr, index, imm_id } => {
                let imm = self.dfg.get_ptr_imm(*imm_id);
                write!(
                    f,
                    "ptr_index {}, {}, {}, {}",
                    self.vf(*ptr),
                    self.vf(*index),
                    imm.scale,
                    imm.offset
                )
            }
            InstructionData::Jump { dest } => {
                write!(f, "jump ")?;
                self.fmt_block_call(f, *dest)
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                write!(f, "br {}, ", self.vf(*condition))?;
                self.fmt_block_call(f, *then_dest)?;
                write!(f, ", ")?;
                self.fmt_block_call(f, *else_dest)
            }
            InstructionData::BrTable { index, table } => {
                let table_data = &self.dfg.jump_tables[*table];
                let targets = &table_data.targets[1..];
                write!(f, "br_table {}, ", self.vf(*index))?;
                self.fmt_block_call(f, table_data.targets[0])?;
                write!(f, ", [")?;
                for (i, target_call) in targets.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    self.fmt_block_call(f, *target_call)?;
                }
                write!(f, "]")
            }
            InstructionData::Return { values } => {
                let ret_vals = self.dfg.get_value_list(*values);
                write!(f, "return")?;
                if !ret_vals.is_empty() {
                    write!(f, " ")?;
                    self.fmt_value_list(f, ret_vals)?;
                }
                Ok(())
            }
            InstructionData::IntCompare { kind, args } => {
                write!(
                    f,
                    "icmp.{:?} {} {}, {}",
                    ty.unwrap(),
                    kind,
                    self.vf(args[0]),
                    self.vf(args[1])
                )
            }
            InstructionData::FloatCompare { kind, args } => {
                write!(
                    f,
                    "fcmp.{:?} {} {}, {}",
                    ty.unwrap(),
                    kind,
                    self.vf(args[0]),
                    self.vf(args[1])
                )
            }
            InstructionData::Unreachable => {
                write!(f, "unreachable")
            }
            InstructionData::Nop => {
                write!(f, "nop")
            }
            InstructionData::Call { func_id, args } => {
                let name = self
                    .module
                    .and_then(|m| m.functions.get(*func_id))
                    .map(|f| f.name.as_str());
                let ret_tys: Option<&[Type]> = self.module.map(|m| {
                    let sig_id = m.functions[*func_id].signature;
                    m.signatures[sig_id].returns.as_ref()
                });
                if let Some(name) = name {
                    write!(f, "call {}(", name)?;
                } else {
                    write!(f, "call {}(", func_id)?;
                }
                let arg_list = self.dfg.get_value_list(*args);
                self.fmt_value_list(f, arg_list)?;
                write!(f, ")")?;
                if let Some(ret) = ret_tys {
                    write!(f, " -> ")?;
                    self.fmt_ret_types(f, ret)?;
                }
                Ok(())
            }
            InstructionData::CallIndirect { ptr, args, sig_id } => {
                let ret_tys: Option<&[Type]> =
                    self.module.map(|m| m.signatures[*sig_id].returns.as_ref());
                write!(f, "call_indirect")?;
                if let Some(ret) = ret_tys {
                    write!(f, ".")?;
                    self.fmt_ret_types(f, ret)?;
                }
                write!(f, " {}(", self.vf(*ptr))?;
                let arg_list = self.dfg.get_value_list(*args);
                self.fmt_value_list(f, arg_list)?;
                write!(f, ")")
            }
            InstructionData::CallIntrinsic {
                intrinsic,
                args,
                sig_id,
            } => {
                let ret_tys: Option<&[Type]> =
                    self.module.map(|m| m.signatures[*sig_id].returns.as_ref());
                write!(f, "call_intrinsic")?;
                if let Some(ret) = ret_tys {
                    write!(f, ".")?;
                    self.fmt_ret_types(f, ret)?;
                }
                write!(f, " {}(", intrinsic)?;
                let arg_list = self.dfg.get_value_list(*args);
                self.fmt_value_list(f, arg_list)?;
                write!(f, ")")
            }
            InstructionData::VectorOpWithExt { opcode, args, ext } => {
                let ext_data = &self.dfg.vector_ext_pool[*ext];
                write!(f, "{}.{:?} ", opcode, ty.unwrap())?;
                let args_slice = self.dfg.get_value_list(*args);
                self.fmt_value_list(f, args_slice)?;
                write!(f, ", mask={}", self.vf(ext_data.mask))?;
                if let Some(evl) = ext_data.evl {
                    write!(f, ", evl={}", self.vf(evl))?;
                }
                Ok(())
            }
            InstructionData::VectorLoadStrided { ptr, stride, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                let result_ty = ty.unwrap_or(Type::VOID);
                write!(f, "load_stride.{}", result_ty)?;
                if ext_data.flags.is_trusted() {
                    write!(f, ".trusted")?;
                }
                write!(f, " {}, stride={}", self.vf(*ptr), self.vf(*stride))?;
                if let Some(mask) = ext_data.mask {
                    write!(f, ", mask={}", self.vf(mask))?;
                }
                if let Some(evl) = ext_data.evl {
                    write!(f, ", evl={}", self.vf(evl))?;
                }
                Ok(())
            }
            InstructionData::VectorStoreStrided { args, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                let args = self.dfg.get_value_list(*args);
                let ptr = args[0];
                let stride = args[1];
                let value = args[2];
                write!(f, "store_stride")?;
                if ext_data.flags.is_trusted() {
                    write!(f, ".trusted")?;
                }
                write!(
                    f,
                    " {}, {}, stride={}",
                    self.vf(value),
                    self.vf(ptr),
                    self.vf(stride)
                )?;
                if let Some(mask) = ext_data.mask {
                    write!(f, ", mask={}", self.vf(mask))?;
                }
                if let Some(evl) = ext_data.evl {
                    write!(f, ", evl={}", self.vf(evl))?;
                }
                Ok(())
            }
            InstructionData::VectorGather { ptr, index, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                let result_ty = ty.unwrap_or(Type::VOID);
                write!(f, "gather.{}", result_ty)?;
                if ext_data.flags.is_trusted() {
                    write!(f, ".trusted")?;
                }
                write!(f, " {}, index={}", self.vf(*ptr), self.vf(*index))?;
                if ext_data.scale != 1 {
                    write!(f, " * {}", ext_data.scale)?;
                }
                if let Some(mask) = ext_data.mask {
                    write!(f, ", mask={}", self.vf(mask))?;
                }
                if let Some(evl) = ext_data.evl {
                    write!(f, ", evl={}", self.vf(evl))?;
                }
                Ok(())
            }
            InstructionData::VectorScatter { args, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                let vals = self.dfg.get_value_list(*args);
                let ptr = vals[0];
                let index = vals[1];
                let value = vals[2];
                write!(f, "scatter")?;
                if ext_data.flags.is_trusted() {
                    write!(f, ".trusted")?;
                }
                write!(
                    f,
                    " {}, {}, index={}",
                    self.vf(value),
                    self.vf(ptr),
                    self.vf(index)
                )?;
                if ext_data.scale != 1 {
                    write!(f, " * {}", ext_data.scale)?;
                }
                if let Some(mask) = ext_data.mask {
                    write!(f, ", mask={}", self.vf(mask))?;
                }
                if let Some(evl) = ext_data.evl {
                    write!(f, ", evl={}", self.vf(evl))?;
                }
                Ok(())
            }
            InstructionData::Shuffle { args, mask } => {
                write!(
                    f,
                    "shuffle.{} {}, {}, mask={:?}",
                    ty.unwrap(),
                    self.vf(args[0]),
                    self.vf(args[1]),
                    mask
                )
            }
        }
    }
}

/// 格式化函数签名
fn fmt_signature(f: &mut dyn Write, module: Option<&Module>, func: &Function) -> Result {
    let linkage = match func.linkage {
        Linkage::Local => "local",
        Linkage::Export => "export",
        Linkage::Import => "import",
    };

    write!(f, "{} function {}(", linkage, func.name)?;

    if let Some(m) = module {
        let sig = &m.signatures[func.signature];
        for (i, param) in sig.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", param)?;
        }
        write!(f, ") -> ")?;
        match sig.returns.len() {
            0 => write!(f, "void")?,
            1 => write!(f, "{}", sig.returns[0])?,
            _ => {
                write!(f, "(")?;
                for (i, ty) in sig.returns.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")?;
            }
        }
    } else {
        write!(f, "...)")?;
    }

    Ok(())
}

/// 格式化基本块头部（参数和前驱/后继信息）
fn fmt_block_header(f: &mut dyn Write, func: &Function, block: crate::Block) -> Result {
    let block_data = &func.layout.blocks[block];

    write!(f, "{}(", block)?;
    for (i, &param) in block_data.params.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        let ty = func.dfg.values[param].ty;
        write!(f, "{}: {}", ValueFmt(&func.dfg, param), ty)?;
    }
    write!(f, ")")?;

    let has_preds = !block_data.preds.is_empty();
    let has_succs = !block_data.succs.is_empty();

    if has_preds || has_succs {
        write!(f, " [")?;
        if has_preds {
            write!(f, "preds: ")?;
            for (i, p) in block_data.preds.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", p)?;
            }
        }
        if has_preds && has_succs {
            write!(f, "; ")?;
        }
        if has_succs {
            write!(f, "succs: ")?;
            for (i, s) in block_data.succs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", s)?;
            }
        }
        write!(f, "]")?;
    }

    writeln!(f, ":")
}

/// 写入模块的完整 IR 表示
pub(crate) fn write_module(f: &mut dyn Write, module: &Module) -> Result {
    for global in module.globals.iter() {
        writeln!(
            f,
            "global {}: {} ({})",
            global.name, global.ty, global.linkage
        )?;
    }

    for (i, (_func_id, func)) in module.functions.iter().enumerate() {
        if i > 0 || !module.globals.is_empty() {
            writeln!(f)?;
        }
        write_function(f, func, Some(module))?;
    }
    Ok(())
}

/// 写入函数的完整 IR 表示
pub(crate) fn write_function(
    f: &mut dyn Write,
    func: &Function,
    module: Option<&Module>,
) -> Result {
    fmt_signature(f, module, func)?;
    writeln!(f)?;

    for (ss, data) in func.stack_slots.iter() {
        writeln!(f, "  {}: size {}", ss, data.size)?;
    }

    let printer = InstPrinter::new(&func.dfg, module);

    for &block in &func.layout.block_order {
        fmt_block_header(f, func, block)?;

        for &inst in &func.layout.blocks[block].insts {
            write!(f, "  ")?;
            printer.fmt_inst_with_results(f, inst)?;
            writeln!(f)?;
        }
    }
    Ok(())
}

impl Display for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write_function(f, self, None)
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write_module(f, self)
    }
}
