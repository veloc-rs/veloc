//! Intel assembly rendering. Instruction spelling and operand order are
//! generated; this host only renders registers, addresses and external names.
use super::inst;
use crate::target::AssemblyWriter;
use core::fmt::{self, Write};
use veloc_lir::BlockId as Block;
use veloc_lir::SymbolId;
use veloc_lir::{InstRef, MachineOpcode, Reg, StackFrame, StackSlot};

pub fn write(
    inst: &InstRef<'_>,
    frame: &StackFrame,
    out: &mut dyn Write,
    symbol: impl FnMut(SymbolId, &mut dyn Write) -> fmt::Result,
) -> fmt::Result {
    let MachineOpcode::Target(opcode) = inst.opcode() else {
        return Err(fmt::Error);
    };
    inst::TargetInst::from_u32(opcode).write_assembly(inst, &mut Intel { out, frame, symbol })
}

struct Intel<'a, F> {
    out: &'a mut dyn Write,
    frame: &'a StackFrame,
    symbol: F,
}

impl<F> Write for Intel<'_, F> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.out.write_str(text)
    }
}

impl<F: FnMut(SymbolId, &mut dyn Write) -> fmt::Result> AssemblyWriter for Intel<'_, F> {
    fn register(&mut self, reg: Reg, bits: u32) -> fmt::Result {
        self.out
            .write_str(inst::register_name(reg, bits).ok_or(fmt::Error)?)
    }
    fn immediate(&mut self, value: i64) -> fmt::Result {
        write!(self.out, "{value}")
    }
    fn block(&mut self, block: Block) -> fmt::Result {
        write!(self.out, ".Lblock{}", block.as_u32())
    }
    fn symbol(&mut self, symbol: SymbolId) -> fmt::Result {
        (self.symbol)(symbol, self.out)
    }
    fn memory(&mut self, base: Reg, index: Option<Reg>, offset: i64, bits: u32) -> fmt::Result {
        let size = match bits {
            8 => "byte",
            16 => "word",
            32 => "dword",
            64 => "qword",
            128 => "xmmword",
            256 => "ymmword",
            512 => "zmmword",
            _ => return Err(fmt::Error),
        };
        write!(self.out, "{size} ptr [")?;
        self.register(base, 64)?;
        if let Some(index) = index {
            self.out.write_str(" + ")?;
            self.register(index, 64)?;
        }
        if offset > 0 {
            write!(self.out, " + {offset}")?;
        }
        if offset < 0 {
            write!(self.out, " - {}", offset.unsigned_abs())?;
        }
        self.out.write_str("]")
    }
    fn stack_slot(&mut self, slot: StackSlot, bits: u32) -> fmt::Result {
        let slot = &self.frame.slots[slot];
        self.memory(
            slot.base.resolve(inst::SPECIAL_REG_FRAME_POINTER),
            None,
            i64::from(slot.offset),
            bits,
        )
    }
}
