//! Bytecode printer for debugging and visualization
//!
//! This module provides functionality to print interpreted bytecode
//! in a human-readable format similar to assembly.

use crate::bytecode::{
    compile::{DataSection, JumpTarget},
    inst::{DecodedInstruction, Instruction},
    CompiledFunction,
};
use core::fmt::{Display, Formatter, Result, Write};
use cranelift_entity::EntityRef;

/// Format a scalar type (encoded as u8) to human-readable name
fn scalar_ty_name(ty: u8) -> &'static str {
    match ty {
        0 => "i8",
        1 => "i16",
        2 => "i32",
        3 => "i64",
        4 => "f32",
        5 => "f64",
        8 => "bool",
        9 => "ptr",
        10 => "void",
        11 => "evl",
        _ => "?",
    }
}

/// Formats a register list from data section
fn fmt_reg_list(
    f: &mut dyn Write,
    data_section: &DataSection,
    offset: usize,
    count: usize,
) -> Result {
    write!(f, "[")?;
    for i in 0..count {
        if i > 0 {
            write!(f, ", ")?;
        }
        let reg = data_section.regs[offset + i];
        write!(f, "{}", reg)?;
    }
    write!(f, "]")
}

/// Formats register moves from data section
fn fmt_moves(f: &mut dyn Write, data_section: &DataSection, target: &JumpTarget) -> Result {
    if target.num_moves == 0 {
        return Ok(());
    }
    write!(f, " moves=[")?;
    let offset = target.moves_offset as usize;
    for i in 0..target.num_moves as usize {
        if i > 0 {
            write!(f, ", ")?;
        }
        let dst = data_section.regs[offset + i * 2];
        let src = data_section.regs[offset + i * 2 + 1];
        write!(f, "r{}<-r{}", dst, src)?;
    }
    write!(f, "]")
}

/// Bytecode instruction printer
struct InstPrinter<'a> {
    data_section: &'a DataSection,
}

impl<'a> InstPrinter<'a> {
    /// Create a new instruction printer
    pub fn new(data_section: &'a DataSection) -> Self {
        Self { data_section }
    }

    /// Format a single instruction
    pub fn fmt_inst(&self, f: &mut dyn Write, pc: usize, inst: &Instruction) -> Result {
        let inst = inst.decode();
        // Write PC and opcode name
        write!(
            f,
            "  {:4}  {:20}",
            pc,
            format!("{:?}", inst)
                .to_lowercase()
                .split('(')
                .next()
                .unwrap_or("")
        )?;

        match inst {
            // Constants
            DecodedInstruction::Iconst { dst, imm64 } => {
                write!(f, " {}, 0x{:016x}", dst, imm64)
            }
            DecodedInstruction::Fconst { dst, imm64 } => {
                write!(f, " {}, 0x{:016x}", dst, imm64)
            }
            DecodedInstruction::Bconst { dst, val } => {
                write!(f, " {}, {}", dst, val)
            }
            DecodedInstruction::Vconst { dst, pool_id } => {
                write!(f, " {}, pool[{}]", dst, pool_id)
            }

            // Binary operations (3 registers)
            DecodedInstruction::I32Add { dst, src1, src2 }
            | DecodedInstruction::I32Sub { dst, src1, src2 }
            | DecodedInstruction::I32Mul { dst, src1, src2 }
            | DecodedInstruction::I32DivS { dst, src1, src2 }
            | DecodedInstruction::I32DivU { dst, src1, src2 }
            | DecodedInstruction::I32RemS { dst, src1, src2 }
            | DecodedInstruction::I32RemU { dst, src1, src2 }
            | DecodedInstruction::I32And { dst, src1, src2 }
            | DecodedInstruction::I32Or { dst, src1, src2 }
            | DecodedInstruction::I32Xor { dst, src1, src2 }
            | DecodedInstruction::I32Shl { dst, src1, src2 }
            | DecodedInstruction::I32ShrS { dst, src1, src2 }
            | DecodedInstruction::I32ShrU { dst, src1, src2 }
            | DecodedInstruction::I32RotL { dst, src1, src2 }
            | DecodedInstruction::I32RotR { dst, src1, src2 }
            | DecodedInstruction::I64Add { dst, src1, src2 }
            | DecodedInstruction::I64Sub { dst, src1, src2 }
            | DecodedInstruction::I64Mul { dst, src1, src2 }
            | DecodedInstruction::I64DivS { dst, src1, src2 }
            | DecodedInstruction::I64DivU { dst, src1, src2 }
            | DecodedInstruction::I64RemS { dst, src1, src2 }
            | DecodedInstruction::I64RemU { dst, src1, src2 }
            | DecodedInstruction::I64And { dst, src1, src2 }
            | DecodedInstruction::I64Or { dst, src1, src2 }
            | DecodedInstruction::I64Xor { dst, src1, src2 }
            | DecodedInstruction::I64Shl { dst, src1, src2 }
            | DecodedInstruction::I64ShrS { dst, src1, src2 }
            | DecodedInstruction::I64ShrU { dst, src1, src2 }
            | DecodedInstruction::I64RotL { dst, src1, src2 }
            | DecodedInstruction::I64RotR { dst, src1, src2 }
            | DecodedInstruction::F32Add { dst, src1, src2 }
            | DecodedInstruction::F32Sub { dst, src1, src2 }
            | DecodedInstruction::F32Mul { dst, src1, src2 }
            | DecodedInstruction::F32Div { dst, src1, src2 }
            | DecodedInstruction::F32Min { dst, src1, src2 }
            | DecodedInstruction::F32Max { dst, src1, src2 }
            | DecodedInstruction::F32CopySign { dst, src1, src2 }
            | DecodedInstruction::F64Add { dst, src1, src2 }
            | DecodedInstruction::F64Sub { dst, src1, src2 }
            | DecodedInstruction::F64Mul { dst, src1, src2 }
            | DecodedInstruction::F64Div { dst, src1, src2 }
            | DecodedInstruction::F64Min { dst, src1, src2 }
            | DecodedInstruction::F64Max { dst, src1, src2 }
            | DecodedInstruction::F64CopySign { dst, src1, src2 }
            | DecodedInstruction::I32Eq { dst, src1, src2 }
            | DecodedInstruction::I32Ne { dst, src1, src2 }
            | DecodedInstruction::I32LtS { dst, src1, src2 }
            | DecodedInstruction::I32LtU { dst, src1, src2 }
            | DecodedInstruction::I32LeS { dst, src1, src2 }
            | DecodedInstruction::I32LeU { dst, src1, src2 }
            | DecodedInstruction::I32GtS { dst, src1, src2 }
            | DecodedInstruction::I32GtU { dst, src1, src2 }
            | DecodedInstruction::I32GeS { dst, src1, src2 }
            | DecodedInstruction::I32GeU { dst, src1, src2 }
            | DecodedInstruction::I64Eq { dst, src1, src2 }
            | DecodedInstruction::I64Ne { dst, src1, src2 }
            | DecodedInstruction::I64LtS { dst, src1, src2 }
            | DecodedInstruction::I64LtU { dst, src1, src2 }
            | DecodedInstruction::I64LeS { dst, src1, src2 }
            | DecodedInstruction::I64LeU { dst, src1, src2 }
            | DecodedInstruction::I64GtS { dst, src1, src2 }
            | DecodedInstruction::I64GtU { dst, src1, src2 }
            | DecodedInstruction::I64GeS { dst, src1, src2 }
            | DecodedInstruction::I64GeU { dst, src1, src2 }
            | DecodedInstruction::F32Eq { dst, src1, src2 }
            | DecodedInstruction::F32Ne { dst, src1, src2 }
            | DecodedInstruction::F32Lt { dst, src1, src2 }
            | DecodedInstruction::F32Le { dst, src1, src2 }
            | DecodedInstruction::F32Gt { dst, src1, src2 }
            | DecodedInstruction::F32Ge { dst, src1, src2 }
            | DecodedInstruction::F64Eq { dst, src1, src2 }
            | DecodedInstruction::F64Ne { dst, src1, src2 }
            | DecodedInstruction::F64Lt { dst, src1, src2 }
            | DecodedInstruction::F64Le { dst, src1, src2 }
            | DecodedInstruction::F64Gt { dst, src1, src2 }
            | DecodedInstruction::F64Ge { dst, src1, src2 } => {
                write!(f, " {}, {}, {}", dst, src1, src2)
            }

            // Immediate binary operations
            DecodedInstruction::I32AddImm { dst, src1, imm }
            | DecodedInstruction::I32SubImm { dst, src1, imm }
            | DecodedInstruction::I32AndImm { dst, src1, imm }
            | DecodedInstruction::I32OrImm { dst, src1, imm }
            | DecodedInstruction::I32XorImm { dst, src1, imm }
            | DecodedInstruction::I32ShlImm { dst, src1, imm }
            | DecodedInstruction::I32ShrSImm { dst, src1, imm }
            | DecodedInstruction::I32ShrUImm { dst, src1, imm } => {
                write!(f, " {}, {}, {}", dst, src1, imm as i32)
            }

            DecodedInstruction::I64AddImm { dst, src1, imm64 }
            | DecodedInstruction::I64SubImm { dst, src1, imm64 }
            | DecodedInstruction::I64AndImm { dst, src1, imm64 }
            | DecodedInstruction::I64OrImm { dst, src1, imm64 }
            | DecodedInstruction::I64XorImm { dst, src1, imm64 }
            | DecodedInstruction::I64ShlImm { dst, src1, imm64 }
            | DecodedInstruction::I64ShrSImm { dst, src1, imm64 }
            | DecodedInstruction::I64ShrUImm { dst, src1, imm64 } => {
                write!(f, " {}, {}, 0x{:016x}", dst, src1, imm64)
            }

            // Unary operations - Float
            DecodedInstruction::F32Neg { dst, src1 }
            | DecodedInstruction::F32Abs { dst, src1 }
            | DecodedInstruction::F32Sqrt { dst, src1 }
            | DecodedInstruction::F32Ceil { dst, src1 }
            | DecodedInstruction::F32Floor { dst, src1 }
            | DecodedInstruction::F32Trunc { dst, src1 }
            | DecodedInstruction::F32Nearest { dst, src1 }
            | DecodedInstruction::F64Neg { dst, src1 }
            | DecodedInstruction::F64Abs { dst, src1 }
            | DecodedInstruction::F64Sqrt { dst, src1 }
            | DecodedInstruction::F64Ceil { dst, src1 }
            | DecodedInstruction::F64Floor { dst, src1 }
            | DecodedInstruction::F64Trunc { dst, src1 }
            | DecodedInstruction::F64Nearest { dst, src1 } => {
                write!(f, " {}, {}", dst, src1)
            }
            // Unary operations - Bitwise
            DecodedInstruction::I32Clz { dst, src }
            | DecodedInstruction::I32Ctz { dst, src }
            | DecodedInstruction::I32Popcnt { dst, src }
            | DecodedInstruction::I64Clz { dst, src }
            | DecodedInstruction::I64Ctz { dst, src }
            | DecodedInstruction::I64Popcnt { dst, src } => {
                write!(f, " {}, {}", dst, src)
            }
            // Unary operations - Eqz
            DecodedInstruction::I32Eqz { dst, src_val }
            | DecodedInstruction::I64Eqz { dst, src_val } => {
                write!(f, " {}, {}", dst, src_val)
            }
            // Unary operations - Trunc
            DecodedInstruction::I32TruncF32S { dst, src }
            | DecodedInstruction::I32TruncF32U { dst, src }
            | DecodedInstruction::I32TruncF64S { dst, src }
            | DecodedInstruction::I32TruncF64U { dst, src }
            | DecodedInstruction::I64TruncF32S { dst, src }
            | DecodedInstruction::I64TruncF32U { dst, src }
            | DecodedInstruction::I64TruncF64S { dst, src }
            | DecodedInstruction::I64TruncF64U { dst, src }
            | DecodedInstruction::I32TruncSatF32S { dst, src }
            | DecodedInstruction::I32TruncSatF32U { dst, src }
            | DecodedInstruction::I32TruncSatF64S { dst, src }
            | DecodedInstruction::I32TruncSatF64U { dst, src }
            | DecodedInstruction::I64TruncSatF32S { dst, src }
            | DecodedInstruction::I64TruncSatF32U { dst, src }
            | DecodedInstruction::I64TruncSatF64S { dst, src }
            | DecodedInstruction::I64TruncSatF64U { dst, src } => {
                write!(f, " {}, {}", dst, src)
            }
            // Unary operations - Convert
            DecodedInstruction::F32ConvertI32S { dst, src }
            | DecodedInstruction::F32ConvertI32U { dst, src }
            | DecodedInstruction::F32ConvertI64S { dst, src }
            | DecodedInstruction::F32ConvertI64U { dst, src }
            | DecodedInstruction::F64ConvertI32S { dst, src }
            | DecodedInstruction::F64ConvertI32U { dst, src }
            | DecodedInstruction::F64ConvertI64S { dst, src }
            | DecodedInstruction::F64ConvertI64U { dst, src }
            | DecodedInstruction::F32DemoteF64 { dst, src }
            | DecodedInstruction::F64PromoteF32 { dst, src } => {
                write!(f, " {}, {}", dst, src)
            }

            // Extend operations
            DecodedInstruction::ExtendS { dst, src, ty }
            | DecodedInstruction::ExtendU { dst, src, ty } => {
                write!(
                    f,
                    " {}, {}, {}->{}",
                    dst,
                    src,
                    scalar_ty_name(ty.from as u8),
                    scalar_ty_name(ty.to as u8)
                )
            }

            // Wrap operation
            DecodedInstruction::Wrap { dst, src, ty } => {
                write!(
                    f,
                    " {}, {}, {}->{}",
                    dst,
                    src,
                    scalar_ty_name(ty.from as u8),
                    scalar_ty_name(ty.to as u8)
                )
            }

            // Memory operations
            DecodedInstruction::I32Load { dst, ptr, offset }
            | DecodedInstruction::I64Load { dst, ptr, offset }
            | DecodedInstruction::F32Load { dst, ptr, offset }
            | DecodedInstruction::F64Load { dst, ptr, offset }
            | DecodedInstruction::I8Load { dst, ptr, offset }
            | DecodedInstruction::I16Load { dst, ptr, offset } => {
                if offset != 0 {
                    write!(f, " {}, [{} + {}]", dst, ptr, offset)
                } else {
                    write!(f, " {}, [{}]", dst, ptr)
                }
            }

            DecodedInstruction::I32Store { val, ptr, offset }
            | DecodedInstruction::I64Store { val, ptr, offset }
            | DecodedInstruction::F32Store { val, ptr, offset }
            | DecodedInstruction::F64Store { val, ptr, offset }
            | DecodedInstruction::I8Store { val, ptr, offset }
            | DecodedInstruction::I16Store { val, ptr, offset } => {
                if offset != 0 {
                    write!(f, " {}, [{} + {}]", val, ptr, offset)
                } else {
                    write!(f, " {}, [{}]", val, ptr)
                }
            }

            // Stack operations
            DecodedInstruction::StackAddr { dst, offset } => {
                write!(f, " {}, offset={}", dst, offset)
            }

            DecodedInstruction::StackLoad { dst, ty, offset } => {
                write!(
                    f,
                    " {}, ty={}, offset={}",
                    dst,
                    scalar_ty_name(ty as u8),
                    offset
                )
            }

            DecodedInstruction::StackStore { val, ty, offset } => {
                write!(
                    f,
                    " {}, ty={}, offset={}",
                    val,
                    scalar_ty_name(ty as u8),
                    offset
                )
            }

            // Pointer indexing
            DecodedInstruction::PtrIndex {
                dst,
                ptr,
                index,
                scale,
                offset,
            } => {
                write!(
                    f,
                    " {}, [{} + {} * {} + {}]",
                    dst, ptr, index, scale as i32, offset as i32
                )
            }

            // Control flow
            DecodedInstruction::Jump { offset } => {
                let target_pc = pc as i64 + offset / core::mem::size_of::<Instruction>() as i64;
                write!(f, " pc={}", target_pc)
            }

            DecodedInstruction::JumpWithMoves { data_offset } => {
                if (data_offset as usize) < self.data_section.jump_targets.len() {
                    let target = &self.data_section.jump_targets[data_offset as usize];
                    let target_pc = pc as i64
                        + i64::from(target.offset) / core::mem::size_of::<Instruction>() as i64;
                    write!(f, " pc={}", target_pc)?;
                    fmt_moves(f, self.data_section, target)?;
                }
                Ok(())
            }

            DecodedInstruction::Br {
                cond,
                then_offset,
                else_offset,
            } => {
                let instruction_size = core::mem::size_of::<Instruction>() as i64;
                let then_pc = pc as i64 + i64::from(then_offset) / instruction_size;
                let else_pc = pc as i64 + i64::from(else_offset) / instruction_size;
                write!(f, " {} then={} else={}", cond, then_pc, else_pc)
            }

            DecodedInstruction::BrWithMoves {
                cond,
                then_idx,
                else_idx,
            } => {
                write!(f, " {} then={} else={}", cond, then_idx, else_idx)
            }

            DecodedInstruction::BrTable {
                idx_reg,
                data_offset,
                num_targets,
            } => {
                write!(f, " {}", idx_reg)?;
                let idx = data_offset as usize;
                let num = num_targets as usize;
                for i in 0..num {
                    if idx + i < self.data_section.jump_targets.len() {
                        let target = &self.data_section.jump_targets[idx + i];
                        let target_pc = pc as i64
                            + i64::from(target.offset) / core::mem::size_of::<Instruction>() as i64;
                        write!(
                            f,
                            "\n        [{}{}] pc={}",
                            idx + i,
                            if i == 0 { " (default)" } else { "" },
                            target_pc
                        )?;
                        fmt_moves(f, self.data_section, target)?;
                    }
                }
                Ok(())
            }

            DecodedInstruction::Select {
                dst,
                cond,
                then_reg,
                else_reg,
            } => {
                write!(f, " {}, {}, {}, r{}", dst, cond, then_reg, else_reg)
            }

            DecodedInstruction::Return {
                data_offset,
                num_vals,
            } => {
                write!(f, " regs=")?;
                fmt_reg_list(
                    f,
                    self.data_section,
                    data_offset as usize,
                    num_vals as usize,
                )
            }

            // Call operations
            DecodedInstruction::Call {
                func_id,
                data_offset,
                num_rets,
                num_args,
            } => {
                write!(f, " func={}", func_id)?;
                write!(f, " rets=")?;
                fmt_reg_list(
                    f,
                    self.data_section,
                    data_offset as usize,
                    num_rets as usize,
                )?;
                write!(f, " args=")?;
                fmt_reg_list(
                    f,
                    self.data_section,
                    data_offset as usize + num_rets as usize,
                    num_args as usize,
                )
            }

            DecodedInstruction::CallIndirect {
                ptr,
                data_offset,
                num_rets,
                num_args,
            } => {
                write!(f, " ptr=")?;
                write!(f, "{}", ptr)?;
                write!(f, " rets=")?;
                fmt_reg_list(
                    f,
                    self.data_section,
                    data_offset as usize,
                    num_rets as usize,
                )?;
                write!(f, " args=")?;
                fmt_reg_list(
                    f,
                    self.data_section,
                    data_offset as usize + num_rets as usize,
                    num_args as usize,
                )
            }

            DecodedInstruction::CallIntrinsic {
                intrinsic,
                data_offset,
                num_rets,
                num_args,
            } => {
                write!(f, " intrinsic={}", intrinsic)?;
                write!(f, " rets=")?;
                fmt_reg_list(
                    f,
                    self.data_section,
                    data_offset as usize,
                    num_rets as usize,
                )?;
                write!(f, " args=")?;
                fmt_reg_list(
                    f,
                    self.data_section,
                    data_offset as usize + num_rets as usize,
                    num_args as usize,
                )
            }

            DecodedInstruction::RegMove { dst, src } => {
                write!(f, " {}, {}", dst, src)
            }

            DecodedInstruction::Unreachable {} => Ok(()),
        }
    }
}

/// Compiled function printer
pub struct FuncPrinter<'a> {
    func: &'a CompiledFunction,
}

impl<'a> FuncPrinter<'a> {
    /// Create a new function printer
    pub fn new(func: &'a CompiledFunction) -> Self {
        Self { func }
    }

    /// Print the function to the writer
    pub fn print(&self, f: &mut dyn Write) -> Result {
        // Print function header
        writeln!(
            f,
            "function @{} in module{} (registers: {})",
            self.func.func_id.index(),
            self.func.module_id.index(),
            self.func.register_count
        )?;

        // Print data section summary
        if !self.func.data_section.regs.is_empty() {
            writeln!(
                f,
                "  data_section: {} registers",
                self.func.data_section.regs.len()
            )?;
        }
        if !self.func.data_section.jump_targets.is_empty() {
            writeln!(
                f,
                "  jump_targets: {} targets",
                self.func.data_section.jump_targets.len()
            )?;
        }

        // Print stack slots
        if !self.func.stack_slots_sizes.is_empty() {
            write!(f, "  stack_slots: [")?;
            for (i, size) in self.func.stack_slots_sizes.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "ss{}: {}", i, size)?;
            }
            writeln!(f, "]")?;
        }

        // Print params
        if !self.func.param_indices.is_empty() {
            write!(f, "  params: [")?;
            for (i, &reg) in self.func.param_indices.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", reg)?;
            }
            writeln!(f, "]")?;
        }

        // Print instructions
        let inst_printer = InstPrinter::new(&self.func.data_section);
        for (pc, inst) in self.func.code.iter().enumerate() {
            inst_printer.fmt_inst(f, pc, inst)?;
            writeln!(f)?;
        }

        Ok(())
    }
}

impl Display for CompiledFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        FuncPrinter::new(self).print(f)
    }
}
