//! Bytecode printer for debugging and visualization
//!
//! This module provides functionality to print interpreted bytecode
//! in a human-readable format similar to assembly.

use crate::bytecode::{
    CompiledFunction,
    compile::{DataSection, JumpTarget},
    inst::{Instruction, Opcode},
};
use core::fmt::{Display, Formatter, Result, Write};
use cranelift_entity::EntityRef;
use veloc_ir::ScalarType;

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

/// Format Extend type (encoded as (to_ty << 8) | from_ty)
fn fmt_extend_ty(f: &mut dyn Write, ty: u16) -> Result {
    let from_ty = (ty & 0xFF) as u8;
    let to_ty = (ty >> 8) as u8;
    write!(f, "{}->{}", scalar_ty_name(from_ty), scalar_ty_name(to_ty))
}

/// Formats a register reference
fn fmt_reg(f: &mut dyn Write, reg: u16) -> Result {
    if reg == 0 {
        write!(f, "_")
    } else {
        write!(f, "r{}", reg)
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
        let reg = data_section.u16_data[offset + i];
        fmt_reg(f, reg)?;
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
        let dst = data_section.u16_data[offset + i * 2];
        let src = data_section.u16_data[offset + i * 2 + 1];
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
        // Write PC and opcode name
        write!(f, "  {:4}  {:20}", pc, format!("{:?}", inst.opcode).to_lowercase())?;

        match inst.opcode {
            // Constants
            Opcode::Iconst | Opcode::Fconst => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", 0x{:016x}", inst.imm64)
            }
            Opcode::Bconst => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", {}", if inst.src2 != 0 { "true" } else { "false" })
            }
            Opcode::Vconst => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", pool[{}]", inst.imm32())
            }

            // Binary operations (3 registers)
            Opcode::I32Add
            | Opcode::I32Sub
            | Opcode::I32Mul
            | Opcode::I32DivS
            | Opcode::I32DivU
            | Opcode::I32RemS
            | Opcode::I32RemU
            | Opcode::I32And
            | Opcode::I32Or
            | Opcode::I32Xor
            | Opcode::I32Shl
            | Opcode::I32ShrS
            | Opcode::I32ShrU
            | Opcode::I32RotL
            | Opcode::I32RotR
            | Opcode::I64Add
            | Opcode::I64Sub
            | Opcode::I64Mul
            | Opcode::I64DivS
            | Opcode::I64DivU
            | Opcode::I64RemS
            | Opcode::I64RemU
            | Opcode::I64And
            | Opcode::I64Or
            | Opcode::I64Xor
            | Opcode::I64Shl
            | Opcode::I64ShrS
            | Opcode::I64ShrU
            | Opcode::I64RotL
            | Opcode::I64RotR
            | Opcode::F32Add
            | Opcode::F32Sub
            | Opcode::F32Mul
            | Opcode::F32Div
            | Opcode::F32Min
            | Opcode::F32Max
            | Opcode::F32CopySign
            | Opcode::F64Add
            | Opcode::F64Sub
            | Opcode::F64Mul
            | Opcode::F64Div
            | Opcode::F64Min
            | Opcode::F64Max
            | Opcode::F64CopySign => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src2)
            }

            // Immediate binary operations
            Opcode::I32AddImm
            | Opcode::I32SubImm
            | Opcode::I32AndImm
            | Opcode::I32OrImm
            | Opcode::I32XorImm
            | Opcode::I32ShlImm
            | Opcode::I32ShrSImm
            | Opcode::I32ShrUImm => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", {}", inst.imm32() as i32)
            }

            Opcode::I64AddImm
            | Opcode::I64SubImm
            | Opcode::I64AndImm
            | Opcode::I64OrImm
            | Opcode::I64XorImm
            | Opcode::I64ShlImm
            | Opcode::I64ShrSImm
            | Opcode::I64ShrUImm => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", 0x{:016x}", inst.imm64)
            }

            // Comparisons
            Opcode::I32Eq
            | Opcode::I32Ne
            | Opcode::I32LtS
            | Opcode::I32LtU
            | Opcode::I32LeS
            | Opcode::I32LeU
            | Opcode::I32GtS
            | Opcode::I32GtU
            | Opcode::I32GeS
            | Opcode::I32GeU
            | Opcode::I64Eq
            | Opcode::I64Ne
            | Opcode::I64LtS
            | Opcode::I64LtU
            | Opcode::I64LeS
            | Opcode::I64LeU
            | Opcode::I64GtS
            | Opcode::I64GtU
            | Opcode::I64GeS
            | Opcode::I64GeU
            | Opcode::F32Eq
            | Opcode::F32Ne
            | Opcode::F32Lt
            | Opcode::F32Le
            | Opcode::F32Gt
            | Opcode::F32Ge
            | Opcode::F64Eq
            | Opcode::F64Ne
            | Opcode::F64Lt
            | Opcode::F64Le
            | Opcode::F64Gt
            | Opcode::F64Ge => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src2)
            }

            // Unary operations
            Opcode::F32Neg
            | Opcode::F32Abs
            | Opcode::F32Sqrt
            | Opcode::F32Ceil
            | Opcode::F32Floor
            | Opcode::F32Trunc
            | Opcode::F32Nearest
            | Opcode::F64Neg
            | Opcode::F64Abs
            | Opcode::F64Sqrt
            | Opcode::F64Ceil
            | Opcode::F64Floor
            | Opcode::F64Trunc
            | Opcode::F64Nearest
            | Opcode::I32Clz
            | Opcode::I32Ctz
            | Opcode::I32Popcnt
            | Opcode::I64Clz
            | Opcode::I64Ctz
            | Opcode::I64Popcnt
            | Opcode::I32Eqz
            | Opcode::I64Eqz
            | Opcode::I32TruncF32S
            | Opcode::I32TruncF32U
            | Opcode::I32TruncF64S
            | Opcode::I32TruncF64U
            | Opcode::I64TruncF32S
            | Opcode::I64TruncF32U
            | Opcode::I64TruncF64S
            | Opcode::I64TruncF64U
            | Opcode::I32TruncSatF32S
            | Opcode::I32TruncSatF32U
            | Opcode::I32TruncSatF64S
            | Opcode::I32TruncSatF64U
            | Opcode::I64TruncSatF32S
            | Opcode::I64TruncSatF32U
            | Opcode::I64TruncSatF64S
            | Opcode::I64TruncSatF64U
            | Opcode::F32ConvertI32S
            | Opcode::F32ConvertI32U
            | Opcode::F32ConvertI64S
            | Opcode::F32ConvertI64U
            | Opcode::F64ConvertI32S
            | Opcode::F64ConvertI32U
            | Opcode::F64ConvertI64S
            | Opcode::F64ConvertI64U
            | Opcode::F32DemoteF64
            | Opcode::F64PromoteF32
            | Opcode::Bitcast => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)
            }

            // Extend operations
            Opcode::ExtendS | Opcode::ExtendU => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", ty=")?;
                fmt_extend_ty(f, inst.src2)
            }

            // Wrap operation
            Opcode::Wrap => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", ty=")?;
                fmt_extend_ty(f, inst.src2)
            }

            // Memory operations
            Opcode::I32Load
            | Opcode::I64Load
            | Opcode::F32Load
            | Opcode::F64Load
            | Opcode::I8Load
            | Opcode::I16Load => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", [")?;
                fmt_reg(f, inst.src1)?;
                if inst.imm32() != 0 {
                    write!(f, " + {}", inst.imm32())?;
                }
                write!(f, "]")
            }

            Opcode::I32Store
            | Opcode::I64Store
            | Opcode::F32Store
            | Opcode::F64Store
            | Opcode::I8Store
            | Opcode::I16Store => {
                write!(f, " ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", [")?;
                fmt_reg(f, inst.src2)?;
                if inst.imm32() != 0 {
                    write!(f, " + {}", inst.imm32())?;
                }
                write!(f, "]")
            }

            // Stack operations
            Opcode::StackAddr => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", offset={}", inst.imm32())
            }

            Opcode::StackLoad => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ty={}", scalar_ty_name(inst.src2 as u8))?;
                write!(f, ", offset={}", inst.imm32())
            }

            Opcode::StackStore => {
                write!(f, " ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", ty={}", scalar_ty_name(inst.src2 as u8))?;
                write!(f, ", offset={}", inst.imm32())
            }

            // Pointer indexing
            Opcode::PtrIndex => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", [")?;
                fmt_reg(f, inst.src1)?;
                write!(f, " + ")?;
                fmt_reg(f, inst.src2)?;
                write!(
                    f,
                    " * {} + {}]",
                    inst.ptr_index_scale(),
                    inst.ptr_index_offset()
                )
            }

            // Control flow
            Opcode::Jump => {
                write!(f, " pc={}", inst.imm32())
            }

            Opcode::JumpWithMoves => {
                let idx = inst.imm32() as usize;
                if idx < self.data_section.jump_targets.len() {
                    let target = &self.data_section.jump_targets[idx];
                    write!(f, " pc={}", target.pc)?;
                    fmt_moves(f, self.data_section, target)?;
                }
                Ok(())
            }

            Opcode::Br => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, " then={} else={}", inst.imm32(), inst.aux())
            }

            Opcode::BrTable => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                let idx = inst.imm32() as usize;
                let num = inst.aux() as usize;
                // Print all targets
                for i in 0..num {
                    if idx + i < self.data_section.jump_targets.len() {
                        let target = &self.data_section.jump_targets[idx + i];
                        write!(
                            f,
                            "\n        [{}{}] pc={}",
                            idx + i,
                            if i == 0 { " (default)" } else { "" },
                            target.pc
                        )?;
                        fmt_moves(f, self.data_section, target)?;
                    }
                }
                Ok(())
            }

            Opcode::Select => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src2)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.select_false_reg())
            }

            Opcode::Return => {
                let offset = inst.imm32() as usize;
                let num = inst.aux() as usize;
                write!(f, " regs=")?;
                fmt_reg_list(f, self.data_section, offset + 1, num)
            }

            // Call operations
            Opcode::Call => {
                let func_id = inst.imm32();
                let data_offset = inst.aux() as usize;
                let num_rets = self.data_section.u16_data[data_offset] as usize;
                let num_args = self.data_section.u16_data[data_offset + 1] as usize;
                write!(f, " func={}", func_id)?;
                write!(f, " rets=")?;
                fmt_reg_list(f, self.data_section, data_offset + 2, num_rets)?;
                write!(f, " args=")?;
                fmt_reg_list(f, self.data_section, data_offset + 2 + num_rets, num_args)
            }

            Opcode::CallIndirect => {
                let data_offset = inst.imm32() as usize;
                let counts = inst.aux();
                let num_rets = (counts >> 16) as usize;
                let num_args = (counts & 0xFFFF) as usize;
                write!(f, " ptr=")?;
                fmt_reg(f, inst.src1)?;
                write!(f, " rets=")?;
                fmt_reg_list(f, self.data_section, data_offset, num_rets)?;
                write!(f, " args=")?;
                fmt_reg_list(f, self.data_section, data_offset + num_rets, num_args)
            }

            Opcode::CallIntrinsic => {
                let data_offset = inst.imm32() as usize;
                let counts = inst.aux();
                let num_rets = (counts >> 16) as usize;
                let num_args = (counts & 0xFFFF) as usize;
                write!(f, " intrinsic={}", inst.src1)?;
                write!(f, " rets=")?;
                fmt_reg_list(f, self.data_section, data_offset, num_rets)?;
                write!(f, " args=")?;
                fmt_reg_list(f, self.data_section, data_offset + num_rets, num_args)
            }

            Opcode::GlobalAddr => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", global[{}]", inst.imm32())
            }

            Opcode::RegMove => {
                write!(f, " ")?;
                fmt_reg(f, inst.dst)?;
                write!(f, ", ")?;
                fmt_reg(f, inst.src1)
            }

            Opcode::Unreachable => Ok(()),
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
        if !self.func.data_section.u16_data.is_empty() {
            writeln!(
                f,
                "  data_section: {} u16 words",
                self.func.data_section.u16_data.len()
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
                fmt_reg(f, reg)?;
            }
            writeln!(f, "]")?;
        }

        // Print returns
        if !self.func.ret_indices.is_empty() {
            write!(f, "  returns: [")?;
            for (i, &reg) in self.func.ret_indices.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                fmt_reg(f, reg)?;
            }
            writeln!(f, "]")?;
        }

        writeln!(f)?;

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
