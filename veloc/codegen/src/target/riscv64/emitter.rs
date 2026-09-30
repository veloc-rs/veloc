//! Adapts generated RV64 instructions to the encoder and resolves symbolic targets.
use super::*;
use veloc_encoder::riscv64::{self as rv, Instruction, Reg as R};
pub(crate) mod host {
    use veloc_encoder::riscv64::{Instruction, Reg};
    include!(concat!(env!("OUT_DIR"), "/encoding_host_riscv64.rs"));
}
pub enum Emission {
    Instructions(Vec<Instruction>),
    Jump(veloc_lir::BlockId),
    Branch(u32, R, R, veloc_lir::BlockId),
    Call(veloc_lir::SymbolId),
}
impl host::Emission for Emission {
    fn instructions(code: &[Instruction]) -> Self {
        Self::Instructions(code.to_vec())
    }
    fn jump(target: veloc_lir::BlockId) -> Self {
        Self::Jump(target)
    }
    fn branch(funct3: u32, lhs: R, rhs: R, target: veloc_lir::BlockId) -> Self {
        Self::Branch(funct3, lhs, rhs, target)
    }
    fn call(target: veloc_lir::SymbolId) -> Self {
        Self::Call(target)
    }
}
pub(crate) fn register(reg: veloc_lir::Reg) -> crate::Result<R> {
    rv::register(reg.0).map_err(|e| crate::Error::codegen(format!("{e:?}")))
}
pub(crate) fn stack_address(
    frame: &veloc_lir::StackFrame,
    slot: veloc_lir::StackSlot,
) -> crate::Result<rv::Address> {
    let addr = frame.address(slot);
    Ok(rv::Address {
        base: register(addr.base)?,
        offset: addr.offset as i64,
    })
}
fn instruction(e: &mut crate::Emitter, op: Instruction) -> crate::Result<()> {
    let encoded = rv::encode(op).map_err(|e| crate::Error::codegen(format!("{e:?}")))?;
    e.instruction(&encoded, None)
}
fn jump(e: &mut crate::Emitter, block: veloc_lir::BlockId) {
    let mut bytes = [0; 8];
    bytes[..4].copy_from_slice(&(0x17u32 | 31 << 7).to_le_bytes());
    bytes[4..].copy_from_slice(&rv::i(0x67, 0, 0, 31, 0).to_le_bytes());
    e.local_fixup(block, &bytes, rv::patch_jump);
}
pub(crate) fn encode_instruction(e: &mut crate::Emitter, emission: Emission) -> crate::Result<()> {
    match emission {
        Emission::Instructions(code) => {
            for op in code {
                instruction(e, op)?;
            }
        }
        Emission::Jump(block) => jump(e, block),
        Emission::Branch(funct3, lhs, rhs, target) => {
            // Invert the condition to skip the long-range jump.
            instruction(e, Instruction::B(funct3 ^ 1, lhs, rhs, 12))?;
            jump(e, target);
        }
        Emission::Call(symbol) => {
            // An aligned absolute pointer permits host calls across arbitrary mappings.
            if e.position() % 8 != 0 {
                instruction(e, Instruction::I(0x13, R::X0, 0, R::X0, 0))?;
            }
            e.bytes(&(0x17u32 | 31 << 7).to_le_bytes());
            instruction(e, Instruction::I(0x03, R::X31, 3, R::X31, 16))?;
            instruction(e, Instruction::I(0x67, R::X1, 0, R::X31, 0))?;
            instruction(e, Instruction::J(R::X0, 12))?;
            e.absolute64(symbol);
        }
    }
    Ok(())
}
pub(super) struct Emit;
impl TargetEmitter for Emit {
    fn begin_block(
        &self,
        e: &mut crate::Emitter,
        b: veloc_lir::BlockId,
        _: &MachineFunction,
    ) -> crate::Result<()> {
        e.mark_block(b);
        Ok(())
    }
    fn emit_instruction(
        &self,
        e: &mut crate::Emitter,
        i: &veloc_lir::InstRef<'_>,
        f: &MachineFunction,
    ) -> crate::Result<()> {
        let MachineOpcode::Target(op) = i.opcode() else {
            return Err(crate::Error::codegen("unselected RV64 instruction"));
        };
        inst::TargetInst::from_u32(op).emit(e, i, f)
    }
}
