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
fn jump_pair(link: R) -> Vec<u8> {
    let mut bytes = (0x17u32 | 31 << 7).to_le_bytes().to_vec();
    bytes.extend_from_slice(&rv::i(0x67, link.hardware(), 0, 31, 0).to_le_bytes());
    bytes
}

pub(crate) fn encode_instruction(e: &mut crate::Emitter, emission: Emission) -> crate::Result<()> {
    use crate::emitter::{CodeForm, ExternalRelocation, RelocationKind, Target};
    match emission {
        Emission::Instructions(code) => {
            for op in code {
                instruction(e, op)?;
            }
        }
        Emission::Jump(block) => e.alternatives(
            Target::Block(block),
            vec![
                CodeForm::relative(&rv::j(0, 0).to_le_bytes(), rv::patch_jal),
                CodeForm::relative(&jump_pair(R::X0), rv::patch_jump),
            ],
        ),
        Emission::Branch(funct3, lhs, rhs, target) => {
            let branch = |condition, distance| {
                rv::b(condition, lhs.hardware(), rhs.hardware(), distance).to_le_bytes()
            };
            let mut medium = branch(funct3 ^ 1, 8).to_vec();
            medium.extend_from_slice(&rv::j(0, 0).to_le_bytes());
            let mut far = branch(funct3 ^ 1, 12).to_vec();
            far.extend_from_slice(&jump_pair(R::X0));
            e.alternatives(
                Target::Block(target),
                vec![
                    CodeForm::relative(&branch(funct3, 0), rv::patch_branch),
                    CodeForm::relative(&medium, rv::patch_far_branch),
                    CodeForm::relative(&far, rv::patch_far_branch),
                ],
            );
        }
        Emission::Call(symbol) => {
            // Unknown external addresses use an aligned inline pointer. Alignment
            // belongs to this fallback form, not to direct calls or emission time.
            let mut far = (0x17u32 | 31 << 7).to_le_bytes().to_vec();
            far.extend_from_slice(&rv::i(0x03, 31, 3, 31, 16).to_le_bytes());
            far.extend_from_slice(&rv::i(0x67, 1, 0, 31, 0).to_le_bytes());
            far.extend_from_slice(&rv::j(0, 12).to_le_bytes());
            far.extend_from_slice(&[0; 8]);
            let absolute = CodeForm::relocated(
                &far,
                ExternalRelocation {
                    kind: RelocationKind::Absolute64,
                    offset: 16,
                    symbol,
                    addend: 0,
                },
            )
            .aligned(8, &rv::i(0x13, 0, 0, 0, 0).to_le_bytes());
            e.alternatives(
                Target::Symbol(symbol),
                vec![
                    CodeForm::relative(&rv::j(1, 0).to_le_bytes(), rv::patch_jal),
                    CodeForm::relative(&jump_pair(R::X1), rv::patch_jump),
                    absolute,
                ],
            );
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
