use super::format;
use crate::{
    DataFlowGraph, Inst, Type, Value, function::Function, inst::InstructionData, module::Module,
    opcode::MemFlags,
};
use core::fmt::{Display, Formatter, Result, Write};

/// 格式化值引用，优先使用命名，否则使用 vN 格式
pub struct ValueFmt<'a>(&'a DataFlowGraph, Value);

impl<'a> Display for ValueFmt<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let name = &self.0.value_names[self.1];
        if name.is_empty() {
            write!(f, "v{}", self.1.0)
        } else {
            // Format as "name.vID" to ensure uniqueness while keeping the name hint.
            // This also allows the parser to extract the original ID if needed.
            write!(f, "{}.v{}", name, self.1.0)
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

    // ==================== Private Formatting Helpers ====================

    fn fmt_mem_flags(&self, f: &mut dyn Write, flags: MemFlags) -> Result {
        if flags.is_trusted() {
            f.write_str(".trusted")?;
        }
        if flags.alignment() != 1 {
            write!(f, ".align{}", flags.alignment())?;
        }
        Ok(())
    }

    /// 格式化内存或向量内存操作的头部 (e.g., "load.i32.align4")
    fn fmt_vmem_op(
        &self,
        f: &mut dyn Write,
        name: &str,
        ty: Option<Type>,
        flags: MemFlags,
    ) -> Result {
        f.write_str(name)?;
        if let Some(t) = ty {
            write!(f, ".{}", t)?;
        }
        self.fmt_mem_flags(f, flags)
    }

    fn fmt_v_off(&self, f: &mut dyn Write, val: Value, offset: u32) -> Result {
        if offset == 0 {
            write!(f, "{}", self.vf(val))
        } else {
            write!(f, "{} + {}", self.vf(val), offset)
        }
    }

    fn fmt_ss_off(&self, f: &mut dyn Write, slot: u32, offset: u32) -> Result {
        if offset == 0 {
            write!(f, "ss{}", slot)
        } else {
            write!(f, "ss{} + {}", slot, offset)
        }
    }

    fn fmt_list<F, T>(
        &self,
        f: &mut dyn Write,
        items: &[T],
        start: &str,
        end: &str,
        sep: &str,
        mut fmt_item: F,
    ) -> Result
    where
        F: FnMut(&mut dyn Write, &T) -> Result,
    {
        f.write_str(start)?;
        for (i, item) in items.iter().enumerate() {
            if i > 0 {
                f.write_str(sep)?;
            }
            fmt_item(f, item)?;
        }
        f.write_str(end)
    }

    pub fn fmt_ret_types(&self, f: &mut dyn Write, ret: &[Type]) -> Result {
        match ret.len() {
            0 => f.write_str("void"),
            1 => write!(f, "{}", ret[0]),
            _ => self.fmt_list(f, ret, "(", ")", ", ", |f, ty| write!(f, "{}", ty)),
        }
    }

    fn fmt_named_arg<D: core::fmt::Display>(&self, f: &mut dyn Write, key: &str, val: D) -> Result {
        write!(f, ", {}={}", key, val)
    }

    fn fmt_vmem_ext(
        &self,
        f: &mut dyn Write,
        stride: Option<Value>,
        index: Option<Value>,
        scale: u32,
        mask: Option<Value>,
        evl: Option<Value>,
    ) -> Result {
        if let Some(s) = stride {
            self.fmt_named_arg(f, format::STRIDE, self.vf(s))?;
        }
        if let Some(i) = index {
            if scale != 1 {
                write!(f, ", {}={} * {}", format::INDEX, self.vf(i), scale)?;
            } else {
                self.fmt_named_arg(f, format::INDEX, self.vf(i))?;
            }
        }
        if let Some(m) = mask {
            self.fmt_named_arg(f, format::MASK, self.vf(m))?;
        }
        if let Some(e) = evl {
            self.fmt_named_arg(f, format::EVL, self.vf(e))?;
        }
        Ok(())
    }

    // ==================== Public Interfaces ====================

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
        self.fmt_list(f, values, "", "", ", ", |f, &v| write!(f, "{}", self.vf(v)))
    }

    fn fmt_block_call(&self, f: &mut dyn Write, block_call: crate::BlockCall) -> Result {
        let data = &self.dfg.block_calls[block_call];
        write!(f, "{}(", data.block)?;
        let args = self.dfg.get_value_list(data.args);
        self.fmt_value_list(f, args)?;
        write!(f, ")")
    }

    fn fmt_instruction_data(
        &self,
        f: &mut dyn Write,
        idata: &InstructionData,
        ty: Option<Type>,
    ) -> Result {
        let op = idata.opcode();
        let name = format::opcode_to_string(op);

        match idata {
            InstructionData::Unary { arg, .. }
            | InstructionData::IntToPtr { arg }
            | InstructionData::PtrToInt { arg } => {
                write!(f, "{}.{} {}", name, ty.unwrap_or(Type::VOID), self.vf(*arg))
            }
            InstructionData::Binary { args, .. } => {
                write!(
                    f,
                    "{}.{} {}, {}",
                    name,
                    ty.unwrap_or(Type::VOID),
                    self.vf(args[0]),
                    self.vf(args[1])
                )
            }
            InstructionData::Ternary { args, .. } => {
                write!(
                    f,
                    "{}.{} {}, {}, {}",
                    name,
                    ty.unwrap_or(Type::VOID),
                    self.vf(args[0]),
                    self.vf(args[1]),
                    self.vf(args[2])
                )
            }
            InstructionData::Load { ptr, offset, flags } => {
                self.fmt_vmem_op(f, name, ty, *flags)?;
                write!(f, " ")?;
                self.fmt_v_off(f, *ptr, *offset)
            }
            InstructionData::Store {
                ptr,
                value,
                offset,
                flags,
            } => {
                self.fmt_vmem_op(f, name, None, *flags)?;
                write!(f, " {}, ", self.vf(*value))?;
                self.fmt_v_off(f, *ptr, *offset)
            }
            InstructionData::StackLoad { slot, offset } => {
                write!(f, "{}.{} ", name, ty.unwrap_or(Type::VOID))?;
                self.fmt_ss_off(f, slot.0, *offset)
            }
            InstructionData::StackStore {
                slot,
                value,
                offset,
            } => {
                write!(f, "{} {} -> ", name, self.vf(*value))?;
                self.fmt_ss_off(f, slot.0, *offset)
            }
            InstructionData::StackAddr { slot, offset } => {
                write!(f, "{} ", name)?;
                self.fmt_ss_off(f, slot.0, *offset)
            }
            InstructionData::Iconst { value } => {
                write!(f, "{}.{} {}", name, ty.unwrap_or(Type::VOID), *value as i64)
            }
            InstructionData::Fconst { value } => {
                let t = ty.unwrap_or(Type::F64);
                let val = *value;
                if t == Type::F32 {
                    let float = f32::from_bits(val as u32);
                    write!(f, "{}.{} {:?} (0x{:08x})", name, t, float, val as u32)
                } else {
                    let float = f64::from_bits(val);
                    write!(f, "{}.{} {:?} (0x{:016x})", name, t, float, val)
                }
            }
            InstructionData::Bconst { value } => {
                write!(f, "{} {}", name, value)
            }
            InstructionData::Vconst { pool_id } => {
                write!(f, "{}.{} {}", name, ty.unwrap_or(Type::VOID), pool_id.0)
            }
            InstructionData::PtrOffset { ptr, offset } => {
                write!(f, "{} {}, {}", name, self.vf(*ptr), offset)
            }
            InstructionData::PtrIndex { ptr, index, imm_id } => {
                let imm = self.dfg.get_ptr_imm(*imm_id);
                write!(
                    f,
                    "{} {}, {}, {}, {}",
                    name,
                    self.vf(*ptr),
                    self.vf(*index),
                    imm.scale,
                    imm.offset
                )
            }
            InstructionData::Jump { dest } => {
                write!(f, "{} ", name)?;
                self.fmt_block_call(f, *dest)
            }
            InstructionData::Br {
                condition,
                then_dest,
                else_dest,
            } => {
                write!(f, "{} {}, ", name, self.vf(*condition))?;
                self.fmt_block_call(f, *then_dest)?;
                write!(f, ", ")?;
                self.fmt_block_call(f, *else_dest)
            }
            InstructionData::BrTable { index, table } => {
                let table_data = &self.dfg.jump_tables[*table];
                let targets = &table_data.targets[1..];
                write!(f, "{} {}, ", name, self.vf(*index))?;
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
                write!(f, "{}", name)?;
                if !ret_vals.is_empty() {
                    write!(f, " ")?;
                    self.fmt_value_list(f, ret_vals)?;
                }
                Ok(())
            }
            InstructionData::IntCompare { kind, args } => {
                write!(
                    f,
                    "{}.{} {} {}, {}",
                    name,
                    ty.unwrap_or(Type::BOOL),
                    format::intcc_to_string(*kind),
                    self.vf(args[0]),
                    self.vf(args[1])
                )
            }
            InstructionData::FloatCompare { kind, args } => {
                write!(
                    f,
                    "{}.{} {} {}, {}",
                    name,
                    ty.unwrap_or(Type::BOOL),
                    format::floatcc_to_string(*kind),
                    self.vf(args[0]),
                    self.vf(args[1])
                )
            }
            InstructionData::Unreachable | InstructionData::Nop => f.write_str(name),
            InstructionData::Call { func_id, args } => {
                let name_str = self
                    .module
                    .and_then(|m| m.functions.get(*func_id))
                    .map(|f| f.name.as_str());
                let ret_tys: Option<&[Type]> = self.module.map(|m| {
                    let sig_id = m.functions[*func_id].signature;
                    m.signatures[sig_id].returns.as_ref()
                });
                if let Some(n) = name_str {
                    write!(f, "{} {}(", name, n)?;
                } else {
                    write!(f, "{} FuncId({})(", name, func_id.0)?;
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
                write!(f, "{}", name)?;
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
                write!(f, "{}", name)?;
                if let Some(ret) = ret_tys {
                    write!(f, ".")?;
                    self.fmt_ret_types(f, ret)?;
                }
                write!(f, " {}(", intrinsic)?;
                let arg_list = self.dfg.get_value_list(*args);
                self.fmt_value_list(f, arg_list)?;
                write!(f, ")")
            }
            InstructionData::VectorOpWithExt { args, ext, .. } => {
                let ext_data = &self.dfg.vector_ext_pool[*ext];
                write!(f, "{}.{} ", name, ty.unwrap_or(Type::VOID))?;
                self.fmt_value_list(f, self.dfg.get_value_list(*args))?;
                self.fmt_vmem_ext(f, None, None, 1, Some(ext_data.mask), ext_data.evl)
            }
            InstructionData::VectorLoadStrided { ptr, stride, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                self.fmt_vmem_op(f, name, ty, ext_data.flags)?;
                write!(f, " ")?;
                self.fmt_v_off(f, *ptr, ext_data.offset as u32)?;
                self.fmt_vmem_ext(f, Some(*stride), None, 1, ext_data.mask, ext_data.evl)
            }
            InstructionData::VectorStoreStrided { args, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                let args_list = self.dfg.get_value_list(*args);
                self.fmt_vmem_op(f, name, None, ext_data.flags)?;
                write!(f, " {}, ", self.vf(args_list[2]))?;
                self.fmt_v_off(f, args_list[0], ext_data.offset as u32)?;
                self.fmt_vmem_ext(f, Some(args_list[1]), None, 1, ext_data.mask, ext_data.evl)
            }
            InstructionData::VectorGather { ptr, index, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                self.fmt_vmem_op(f, name, ty, ext_data.flags)?;
                write!(f, " ")?;
                self.fmt_v_off(f, *ptr, ext_data.offset as u32)?;
                self.fmt_vmem_ext(
                    f,
                    None,
                    Some(*index),
                    ext_data.scale as u32,
                    ext_data.mask,
                    ext_data.evl,
                )
            }
            InstructionData::VectorScatter { args, ext } => {
                let ext_data = &self.dfg.vector_mem_ext_pool[*ext];
                let vals = self.dfg.get_value_list(*args);
                self.fmt_vmem_op(f, name, None, ext_data.flags)?;
                write!(f, " {}, ", self.vf(vals[2]))?;
                self.fmt_v_off(f, vals[0], ext_data.offset as u32)?;
                self.fmt_vmem_ext(
                    f,
                    None,
                    Some(vals[1]),
                    ext_data.scale as u32,
                    ext_data.mask,
                    ext_data.evl,
                )
            }
            InstructionData::Shuffle { args, mask } => {
                write!(
                    f,
                    "{}.{} {}, {}, mask={}",
                    name,
                    ty.unwrap_or(Type::VOID),
                    self.vf(args[0]),
                    self.vf(args[1]),
                    mask.0
                )
            }
        }
    }
}

/// 格式化基本块头部（参数和前驱/后继信息）
pub struct FuncPrinter<'a> {
    pub func: &'a Function,
    pub module: Option<&'a Module>,
    inst_printer: InstPrinter<'a>,
}

impl<'a> FuncPrinter<'a> {
    pub fn new(func: &'a Function, module: Option<&'a Module>) -> Self {
        Self {
            func,
            module,
            inst_printer: InstPrinter::new(&func.dfg, module),
        }
    }

    pub fn print(&self, f: &mut dyn Write) -> Result {
        self.fmt_signature(f)?;
        writeln!(f)?;

        for (ss, data) in self.func.stack_slots.iter() {
            writeln!(f, "  {}: size {}", ss, data.size)?;
        }

        for &block in &self.func.layout.block_order {
            self.fmt_block(f, block)?;
        }
        Ok(())
    }

    fn fmt_signature(&self, f: &mut dyn Write) -> Result {
        let linkage = format::linkage_to_string(self.func.linkage);
        write!(f, "{} function {}(", linkage, self.func.name)?;

        if let Some(m) = self.module {
            let sig = &m.signatures[self.func.signature];
            for (i, param) in sig.params.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", param)?;
            }
            write!(f, ") -> ")?;
            self.inst_printer.fmt_ret_types(f, &sig.returns)?;
        } else {
            write!(f, "...)")?;
        }
        Ok(())
    }

    fn fmt_block(&self, f: &mut dyn Write, block: crate::Block) -> Result {
        self.fmt_block_header(f, block)?;

        for &inst in &self.func.layout.blocks[block].insts {
            write!(f, "  ")?;
            self.inst_printer.fmt_inst_with_results(f, inst)?;
            writeln!(f)?;
        }
        Ok(())
    }

    fn fmt_block_header(&self, f: &mut dyn Write, block: crate::Block) -> Result {
        let block_data = &self.func.layout.blocks[block];

        write!(f, "{}(", block)?;
        for (i, &param) in block_data.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let ty = self.func.dfg.values[param].ty;
            write!(f, "{}: {}", ValueFmt(&self.func.dfg, param), ty)?;
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
}

pub struct ModulePrinter<'a> {
    pub module: &'a Module,
}

impl<'a> ModulePrinter<'a> {
    pub fn new(module: &'a Module) -> Self {
        Self { module }
    }

    pub fn print(&self, f: &mut dyn Write) -> Result {
        for global in self.module.globals.iter() {
            writeln!(
                f,
                "global {}: {} ({})",
                global.name, global.ty, global.linkage
            )?;
        }

        for (i, (_func_id, func)) in self.module.functions.iter().enumerate() {
            if i > 0 || !self.module.globals.is_empty() {
                writeln!(f)?;
            }
            let fp = FuncPrinter::new(func, Some(self.module));
            fp.print(f)?;
        }
        Ok(())
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        FuncPrinter::new(self, None).print(f)
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        ModulePrinter::new(self).print(f)
    }
}
