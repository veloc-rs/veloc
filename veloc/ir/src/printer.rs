use super::Linkage;
use super::function::Function;
use super::module::Module;
use crate::inst::InstructionData;
use crate::{DataFlowGraph, Type, Value};
use core::fmt::{Display, Formatter, Result, Write};

/// 辅助函数：打印返回类型列表（支持多返回值）
fn fmt_ret_types(f: &mut dyn Write, ret: &[Type]) -> Result {
    if ret.len() == 1 {
        write!(f, "{}", ret[0])
    } else {
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

struct V<'a>(&'a DataFlowGraph, Value);

impl<'a> Display for V<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        let name = &self.0.value_names[self.1];
        if !name.is_empty() {
            write!(f, "{}", name)
        } else {
            write!(f, "v{}", self.1.0)
        }
    }
}

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
        write_function_template(f, func, Some(module))?;
    }
    Ok(())
}

pub(crate) fn write_function(f: &mut dyn Write, func: &Function) -> Result {
    write_function_template(f, func, None)
}

fn write_function_template(f: &mut dyn Write, func: &Function, module: Option<&Module>) -> Result {
    let linkage = match func.linkage {
        Linkage::Local => "local",
        Linkage::Export => "export",
        Linkage::Import => "import",
    };

    if let Some(m) = module {
        let sig = &m.signatures[func.signature];
        write!(f, "{} function {}(", linkage, func.name)?;
        for (i, param) in sig.params.iter().cloned().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", param)?;
        }
        write!(f, ") -> ")?;
        fmt_ret_types(f, &sig.returns)?;
    } else {
        write!(
            f,
            "{} function {}({:?})",
            linkage, func.name, func.signature
        )?;
    }
    writeln!(f)?;

    for (ss, data) in func.stack_slots.iter() {
        writeln!(f, "  {}: size {}", ss, data.size)?;
    }

    for &block in &func.layout.block_order {
        let block_data = &func.layout.blocks[block];
        write!(f, "{}(", block)?;
        for (i, &param) in block_data.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: {}", V(&func.dfg, param), func.dfg.values[param].ty)?;
        }
        write!(f, ")")?;

        if !block_data.preds.is_empty() || !block_data.succs.is_empty() {
            write!(f, " [")?;
            if !block_data.preds.is_empty() {
                write!(f, "preds: ")?;
                for (i, p) in block_data.preds.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
            }
            if !block_data.preds.is_empty() && !block_data.succs.is_empty() {
                write!(f, "; ")?;
            }
            if !block_data.succs.is_empty() {
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
        writeln!(f, ":")?;

        for &inst in &func.layout.blocks[block].insts {
            let idata = &func.dfg.instructions[inst];
            let results = func.dfg.inst_results(inst);
            let dfg = &func.dfg;
            let v = |val| V(dfg, val);

            write!(f, "  ")?;
            if !results.is_empty() {
                if results.len() == 1 {
                    write!(f, "{} = ", v(results[0]))?;
                } else {
                    // 多返回值打印：(v1, v2, ...) =
                    write!(f, "(")?;
                    for (i, &r) in results.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(r))?;
                    }
                    write!(f, ") = ")?;
                }
            }

            // Get the result type from the instruction's first result value (if any)
            let ty = results.first().map(|&r| func.dfg.value_type(r));

            match idata {
                InstructionData::Unary { opcode, arg } => {
                    write!(f, "{}.{:?} {}", opcode, ty.unwrap(), v(*arg))?;
                }
                InstructionData::Binary { opcode, args } => {
                    write!(
                        f,
                        "{}.{:?} {}, {}",
                        opcode,
                        ty.unwrap(),
                        v(args[0]),
                        v(args[1])
                    )?;
                }
                InstructionData::Load { ptr, offset, flags } => {
                    write!(f, "load.{:?}", ty.unwrap())?;
                    if flags.is_trusted() {
                        write!(f, ".trusted")?;
                    }
                    if flags.alignment() != 1 {
                        write!(f, ".align{}", flags.alignment())?;
                    }
                    write!(f, " {} + {}", v(*ptr), offset)?;
                }
                InstructionData::Store {
                    ptr,
                    value,
                    offset,
                    flags,
                } => {
                    write!(f, "store")?;
                    if flags.is_trusted() {
                        write!(f, ".trusted")?;
                    }
                    if flags.alignment() != 1 {
                        write!(f, ".align{}", flags.alignment())?;
                    }
                    write!(f, " {}, {} + {}", v(*value), v(*ptr), offset)?;
                }
                InstructionData::StackLoad { slot, offset } => {
                    write!(f, "stack_load.{:?} {} + {}", ty.unwrap(), slot, offset)?;
                }
                InstructionData::StackStore {
                    slot,
                    value,
                    offset,
                } => {
                    write!(f, "stack_store {} -> {} + {}", v(*value), slot, offset)?;
                }
                InstructionData::StackAddr { slot, offset } => {
                    write!(f, "stack_addr {} + {}", slot, offset)?;
                }
                InstructionData::Iconst { value } => {
                    let val = *value as i64;
                    write!(f, "iconst.{:?} {}", ty.unwrap(), val)?;
                }
                InstructionData::Fconst { value } => {
                    let ty = ty.unwrap();
                    let val = *value;
                    if ty == Type::F32 {
                        let float = f32::from_bits(val as u32);
                        write!(f, "fconst.{:?} {:?} (0x{:08x})", ty, float, val as u32)?;
                    } else {
                        let float = f64::from_bits(val);
                        write!(f, "fconst.{:?} {:?} (0x{:016x})", ty, float, val)?;
                    }
                }
                InstructionData::IntToPtr { arg } => {
                    write!(f, "inttoptr {}", v(*arg))?;
                }
                InstructionData::PtrToInt { arg } => {
                    write!(f, "ptrtoint.{:?} {}", ty.unwrap(), v(*arg))?;
                }
                InstructionData::PtrOffset { ptr, offset } => {
                    write!(f, "ptr_offset {}, {}", v(*ptr), offset)?;
                }
                InstructionData::PtrIndex { ptr, index, imm_id } => {
                    let imm = dfg.get_ptr_imm(*imm_id);
                    write!(
                        f,
                        "ptr_index {}, {}, {}, {}",
                        v(*ptr),
                        v(*index),
                        imm.scale,
                        imm.offset
                    )?;
                }
                InstructionData::Bconst { value } => {
                    write!(f, "bconst {}", value)?;
                }
                InstructionData::Vconst { pool_id } => {
                    write!(f, "vconst.{:?} {}", ty.unwrap(), pool_id)?;
                }
                InstructionData::Call { func_id, args } => {
                    let name = module
                        .and_then(|m| m.functions.get(*func_id))
                        .map(|f| f.name.as_str());
                    let ret_tys: Option<&[Type]> = module.map(|m| {
                        let sig_id = m.functions[*func_id].signature;
                        m.signatures[sig_id].returns.as_ref()
                    });

                    if let Some(name) = name {
                        write!(f, "call {} (", name)?;
                    } else {
                        write!(f, "call {} (", func_id)?;
                    }

                    let args = dfg.get_value_list(*args);
                    for (i, &arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }

                    if let Some(ret) = ret_tys {
                        write!(f, ") -> ")?;
                        fmt_ret_types(f, ret)?;
                    } else {
                        write!(f, ")")?;
                    }
                }
                InstructionData::Jump { dest } => {
                    let dest_data = dfg.block_calls[*dest];
                    write!(f, "jump {}(", dest_data.block)?;
                    let args = dfg.get_value_list(dest_data.args);
                    for (i, &arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }
                    write!(f, ")")?;
                }
                InstructionData::Br {
                    condition,
                    then_dest,
                    else_dest,
                } => {
                    let then_data = dfg.block_calls[*then_dest];
                    let else_data = dfg.block_calls[*else_dest];

                    write!(f, "br {}, {}(", v(*condition), then_data.block)?;
                    let t_args = dfg.get_value_list(then_data.args);
                    for (i, &arg) in t_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }
                    write!(f, "), {}(", else_data.block)?;
                    let e_args = dfg.get_value_list(else_data.args);
                    for (i, &arg) in e_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }
                    write!(f, ")")?;
                }
                InstructionData::BrTable { index, table } => {
                    let table_data = &dfg.jump_tables[*table];
                    let default_call = dfg.block_calls[table_data.targets[0]];
                    let targets = &table_data.targets[1..];

                    write!(f, "br_table {}, {}(", v(*index), default_call.block)?;
                    let d_args = dfg.get_value_list(default_call.args);
                    for (i, &arg) in d_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }
                    write!(f, "), [")?;
                    for (i, target_call) in targets.iter().enumerate() {
                        let target_data = dfg.block_calls[*target_call];
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}(", target_data.block)?;
                        let t_args = dfg.get_value_list(target_data.args);
                        for (j, &arg) in t_args.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}", v(arg))?;
                        }
                        write!(f, ")")?;
                    }
                    write!(f, "]")?;
                }
                InstructionData::Return { values } => {
                    let ret_vals = dfg.get_value_list(*values);
                    if ret_vals.is_empty() {
                        write!(f, "return")?;
                    } else if ret_vals.len() == 1 {
                        write!(f, "return {}", v(ret_vals[0]))?;
                    } else {
                        write!(f, "return (")?;
                        for (i, &val) in ret_vals.iter().enumerate() {
                            if i > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}", v(val))?;
                        }
                        write!(f, ")")?;
                    }
                }
                InstructionData::IntCompare { kind, args } => {
                    write!(
                        f,
                        "icmp.{:?} {} {}, {}",
                        ty.unwrap(),
                        kind,
                        v(args[0]),
                        v(args[1])
                    )?;
                }
                InstructionData::FloatCompare { kind, args } => {
                    write!(
                        f,
                        "fcmp.{:?} {} {}, {}",
                        ty.unwrap(),
                        kind,
                        v(args[0]),
                        v(args[1])
                    )?;
                }
                InstructionData::Unreachable => {
                    write!(f, "unreachable")?;
                }
                InstructionData::CallIndirect { ptr, args, sig_id } => {
                    let ret_tys: Option<&[Type]> =
                        module.map(|m| m.signatures[*sig_id].returns.as_ref());
                    if let Some(ret) = ret_tys {
                        write!(f, "call_indirect.{:?}.", sig_id)?;
                        fmt_ret_types(f, ret)?;
                        write!(f, " {} (", v(*ptr))?;
                    } else {
                        write!(f, "call_indirect.{:?} {} (", sig_id, v(*ptr))?;
                    }
                    let args = dfg.get_value_list(*args);
                    for (i, &arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }
                    write!(f, ")")?;
                }
                InstructionData::CallIntrinsic {
                    intrinsic,
                    args,
                    sig_id,
                } => {
                    let ret_tys: Option<&[Type]> =
                        module.map(|m| m.signatures[*sig_id].returns.as_ref());
                    if let Some(ret) = ret_tys {
                        write!(f, "call_intrinsic.")?;
                        fmt_ret_types(f, ret)?;
                        write!(f, " {}(", intrinsic)?;
                    } else {
                        write!(f, "call_intrinsic {}(", intrinsic)?;
                    }
                    for (i, &arg) in dfg.get_value_list(*args).iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }
                    write!(f, ")")?;
                }
                InstructionData::Nop => {
                    write!(f, "nop")?;
                }
                // Vector operations
                InstructionData::Ternary { opcode, args } => {
                    write!(
                        f,
                        "{}.{:?} {}, {}, {}",
                        opcode,
                        ty.unwrap(),
                        v(args[0]),
                        v(args[1]),
                        v(args[2])
                    )?;
                }
                InstructionData::VectorOpWithExt { opcode, args, ext } => {
                    let ext_data = &dfg.vector_ext_pool[*ext];
                    write!(f, "{}.{:?} ", opcode, ty.unwrap())?;
                    let args_slice = dfg.get_value_list(*args);
                    for (i, &arg) in args_slice.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", v(arg))?;
                    }
                    write!(f, ", mask={}", v(ext_data.mask))?;
                    if let Some(evl) = ext_data.evl {
                        write!(f, ", evl={}", v(evl))?;
                    }
                }
                // Strided 操作
                InstructionData::VectorLoadStrided { ptr, stride, ext } => {
                    let ext_data = &dfg.vector_mem_ext_pool[*ext];
                    let result_ty = ty.unwrap_or(Type::VOID);
                    write!(f, "load_stride.{:?} ", result_ty)?;
                    if ext_data.flags.is_trusted() {
                        write!(f, "trusted ")?;
                    }
                    write!(f, "{}, stride={}", v(*ptr), v(*stride))?;
                    if let Some(mask) = ext_data.mask {
                        write!(f, ", mask={}", v(mask))?;
                    }
                    if let Some(evl) = ext_data.evl {
                        write!(f, ", evl={}", v(evl))?;
                    }
                }
                InstructionData::VectorStoreStrided { args, ext } => {
                    let ext_data = &dfg.vector_mem_ext_pool[*ext];
                    let args = dfg.get_value_list(*args);
                    let ptr = args[0];
                    let stride = args[1];
                    let value = args[2];
                    write!(f, "store_stride ")?;
                    if ext_data.flags.is_trusted() {
                        write!(f, "trusted ")?;
                    }
                    write!(f, "{}, {}, stride={}", v(value), v(ptr), v(stride))?;
                    if let Some(mask) = ext_data.mask {
                        write!(f, ", mask={}", v(mask))?;
                    }
                    if let Some(evl) = ext_data.evl {
                        write!(f, ", evl={}", v(evl))?;
                    }
                }
                // Gather/Scatter 操作
                InstructionData::VectorGather { ptr, index, ext } => {
                    let ext_data = &dfg.vector_mem_ext_pool[*ext];
                    let result_ty = ty.unwrap_or(Type::VOID);
                    write!(f, "gather.{:?} ", result_ty)?;
                    if ext_data.flags.is_trusted() {
                        write!(f, "trusted ")?;
                    }
                    write!(f, "{}, index={}", v(*ptr), v(*index))?;
                    if ext_data.scale != 1 {
                        write!(f, "* {}", ext_data.scale)?;
                    }
                    if let Some(mask) = ext_data.mask {
                        write!(f, ", mask={}", v(mask))?;
                    }
                    if let Some(evl) = ext_data.evl {
                        write!(f, ", evl={}", v(evl))?;
                    }
                }
                InstructionData::VectorScatter { args, ext } => {
                    let ext_data = &dfg.vector_mem_ext_pool[*ext];
                    let vals = dfg.get_value_list(*args);
                    let ptr = vals[0];
                    let index = vals[1];
                    let value = vals[2];
                    write!(f, "scatter ")?;
                    if ext_data.flags.is_trusted() {
                        write!(f, "trusted ")?;
                    }
                    write!(f, "{}, {}, index={}", v(value), v(ptr), v(index))?;
                    if ext_data.scale != 1 {
                        write!(f, "* {}", ext_data.scale)?;
                    }
                    if let Some(mask) = ext_data.mask {
                        write!(f, ", mask={}", v(mask))?;
                    }
                    if let Some(evl) = ext_data.evl {
                        write!(f, ", evl={}", v(evl))?;
                    }
                }
                InstructionData::Shuffle { args, mask } => {
                    write!(
                        f,
                        "shuffle.{:?} {}, {}, mask={:?}",
                        ty.unwrap(),
                        v(args[0]),
                        v(args[1]),
                        mask
                    )?;
                }
            }
            writeln!(f)?;
        }
    }
    Ok(())
}

impl Display for Function {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write_function(f, self)
    }
}

impl Display for Module {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write_module(f, self)
    }
}
