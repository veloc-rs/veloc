use super::Linkage;
use super::function::Function;
use super::module::Module;
use core::fmt::{Display, Formatter, Result, Write};

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
    let cc_info = if let Some(m) = module {
        alloc::format!("{:?}", m.signatures[func.signature].call_conv)
    } else {
        alloc::format!("{:?}", func.signature)
    };
    writeln!(f, "{} function {}({})", linkage, func.name, cc_info)?;

    for (ss, data) in func.stack_slots.iter() {
        writeln!(f, "  {}: size {}", ss, data.size)?;
    }

    for &block in &func.layout.block_order {
        let block_data = &func.layout.blocks[block];
        write!(f, "{}(", block)?;
        for (i, param) in block_data.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let val_data = &func.dfg.values[*param];
            write!(f, "{}: {}", param, val_data.ty)?;
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
            let res = func.dfg.inst_results(inst);
            let dfg = &func.dfg;

            match res {
                Some(r) => write!(f, "  {:<5} = ", r)?,
                None => write!(f, "  ")?,
            }

            match idata {
                crate::inst::InstructionData::Unary { opcode, arg, ty } => {
                    write!(f, "{}.{} {}", opcode, ty, arg)?;
                }
                crate::inst::InstructionData::Binary { opcode, args, ty } => {
                    write!(f, "{}.{} {}, {}", opcode, ty, args[0], args[1])?;
                }
                crate::inst::InstructionData::Load { ptr, offset, ty } => {
                    write!(f, "load.{} {} + {}", ty, ptr, offset)?;
                }
                crate::inst::InstructionData::Store { ptr, value, offset } => {
                    write!(f, "store {} -> {} + {}", value, ptr, offset)?;
                }
                crate::inst::InstructionData::StackLoad { slot, offset, ty } => {
                    write!(f, "stack_load.{} {} + {}", ty, slot, offset)?;
                }
                crate::inst::InstructionData::StackStore {
                    slot,
                    value,
                    offset,
                } => {
                    write!(f, "stack_store {} -> {} + {}", value, slot, offset)?;
                }
                crate::inst::InstructionData::StackAddr { slot, offset } => {
                    write!(f, "stack_addr {} + {}", slot, offset)?;
                }
                crate::inst::InstructionData::Iconst { value, ty } => {
                    write!(f, "iconst.{} {}", ty, value)?;
                }
                crate::inst::InstructionData::Fconst { value, ty } => {
                    if *ty == crate::types::Type::F32 {
                        let float = f32::from_bits(*value as u32);
                        write!(f, "fconst.{} {:?} (0x{:08x})", ty, float, *value as u32)?;
                    } else {
                        let float = f64::from_bits(*value);
                        write!(f, "fconst.{} {:?} (0x{:016x})", ty, float, value)?;
                    }
                }
                crate::inst::InstructionData::IntToPtr { arg } => {
                    write!(f, "inttoptr {}", arg)?;
                }
                crate::inst::InstructionData::PtrToInt { arg, ty } => {
                    write!(f, "ptrtoint.{} {}", ty, arg)?;
                }
                crate::inst::InstructionData::Gep { ptr, offset } => {
                    write!(f, "gep {}, {}", ptr, offset)?;
                }
                crate::inst::InstructionData::Bconst { value } => {
                    write!(f, "bconst {}", value)?;
                }
                crate::inst::InstructionData::Call { func_id, args, ty } => {
                    let name = module
                        .and_then(|m| m.functions.get(*func_id))
                        .map(|f| f.name.as_str());

                    if let Some(name) = name {
                        write!(f, "call {} (", name)?;
                    } else {
                        write!(f, "call {} (", func_id)?;
                    }
                    let args = dfg.get_value_list(*args);
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ") -> {}", ty)?;
                }
                crate::inst::InstructionData::Jump { dest } => {
                    let dest_data = dfg.block_calls[*dest];
                    write!(f, "jump {} (", dest_data.block)?;
                    let args = dfg.get_value_list(dest_data.args);
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ")")?;
                }
                crate::inst::InstructionData::Br {
                    condition,
                    then_dest,
                    else_dest,
                } => {
                    let then_data = dfg.block_calls[*then_dest];
                    let else_data = dfg.block_calls[*else_dest];

                    write!(f, "br {}, {} (", condition, then_data.block)?;
                    let t_args = dfg.get_value_list(then_data.args);
                    for (i, arg) in t_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, "), {} (", else_data.block)?;
                    let e_args = dfg.get_value_list(else_data.args);
                    for (i, arg) in e_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ")")?;
                }
                crate::inst::InstructionData::BrTable { index, table } => {
                    let table_data = &dfg.jump_tables[*table];
                    let default_call = dfg.block_calls[table_data.targets[0]];
                    let targets = &table_data.targets[1..];

                    write!(f, "br_table {}, {} (", index, default_call.block)?;
                    let d_args = dfg.get_value_list(default_call.args);
                    for (i, arg) in d_args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, "), [")?;
                    for (i, target_call) in targets.iter().enumerate() {
                        let target_data = dfg.block_calls[*target_call];
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{} (", target_data.block)?;
                        let t_args = dfg.get_value_list(target_data.args);
                        for (j, arg) in t_args.iter().enumerate() {
                            if j > 0 {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}", arg)?;
                        }
                        write!(f, ")")?;
                    }
                    write!(f, "]")?;
                }
                crate::inst::InstructionData::Return { value } => {
                    write!(f, "return")?;
                    if let Some(v) = value {
                        write!(f, " {}", v)?;
                    }
                }
                crate::inst::InstructionData::Select {
                    condition,
                    then_val,
                    else_val,
                    ..
                } => {
                    write!(f, "select {}, {}, {}", condition, then_val, else_val)?;
                }
                crate::inst::InstructionData::IntCompare { kind, args, ty } => {
                    write!(f, "icmp.{} {} {}, {}", ty, kind, args[0], args[1])?;
                }
                crate::inst::InstructionData::FloatCompare { kind, args, ty } => {
                    write!(f, "fcmp.{} {} {}, {}", ty, kind, args[0], args[1])?;
                }
                crate::inst::InstructionData::Unreachable => {
                    write!(f, "unreachable")?;
                }
                crate::inst::InstructionData::CallIndirect {
                    ptr,
                    args,
                    sig_id,
                    ty,
                } => {
                    write!(f, "call_indirect.{:?}.{} {} (", sig_id, ty, ptr)?;
                    let args = dfg.get_value_list(*args);
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ")")?;
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
