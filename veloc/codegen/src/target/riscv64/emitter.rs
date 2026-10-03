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
    Address(R, veloc_lir::SymbolId),
    Table(R, Vec<veloc_lir::BlockId>),
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
    fn table(index: R, targets: &[veloc_lir::BlockId]) -> Self {
        Self::Table(index, targets.to_vec())
    }
    fn address(dst: R, target: veloc_lir::SymbolId) -> Self {
        Self::Address(dst, target)
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
fn instruction(e: &mut crate::Emitter, op: Instruction, compressed: bool) -> crate::Result<()> {
    let encoded = rv::encode(op).map_err(|e| crate::Error::codegen(format!("{e:?}")))?;
    if compressed
        && let Ok(word) = <[u8; 4]>::try_from(encoded.bytes())
        && let Some(short) = rv::compressed::compress(u32::from_le_bytes(word))
    {
        e.bytes(&short.to_le_bytes());
        return Ok(());
    }
    e.instruction(&encoded, None)
}
fn jump_pair(link: R) -> Vec<u8> {
    let mut bytes = (0x17u32 | 31 << 7).to_le_bytes().to_vec();
    bytes.extend_from_slice(&rv::i(0x67, link.hardware(), 0, 31, 0).to_le_bytes());
    bytes
}

fn encode_instruction(
    e: &mut crate::Emitter,
    emission: Emission,
    compressed: bool,
    external_calls: ExternalCalls,
) -> crate::Result<()> {
    use crate::emitter::{CodeForm, ExternalRelocation, RelocationKind, Target};
    match emission {
        Emission::Instructions(code) => {
            // Only omit a complete copy. Embedded copies may be part of a
            // sequence with fixed branch offsets and must retain their size.
            // fsgnj.s can also normalize a malformed NaN-boxed source.
            if let [Instruction::Copy(dst, src, bits)] = code.as_slice() {
                if dst == src && (!dst.is_float() || *bits == 64) {
                    return Ok(());
                }
            }
            // Multi-instruction recipes may contain fixed internal offsets.
            let compress = compressed && code.len() == 1;
            for op in code {
                instruction(e, op, compress)?;
            }
        }
        Emission::Jump(block) => {
            let mut forms = vec![
                CodeForm::relative(&rv::j(0, 0).to_le_bytes(), rv::patch_jal),
                CodeForm::relative(&jump_pair(R::X0), rv::patch_jump),
            ];
            if compressed {
                forms.insert(
                    0,
                    CodeForm::relative(&0xa001u16.to_le_bytes(), rv::compressed::patch_jump),
                );
            }
            e.jump(block, forms);
        }
        Emission::Branch(funct3, lhs, rhs, target) => {
            let forms = |funct3| {
                let branch = |condition, distance| {
                    rv::b(condition, lhs.hardware(), rhs.hardware(), distance).to_le_bytes()
                };
                let mut medium = branch(funct3 ^ 1, 8).to_vec();
                medium.extend_from_slice(&rv::j(0, 0).to_le_bytes());
                let mut far = branch(funct3 ^ 1, 12).to_vec();
                far.extend_from_slice(&jump_pair(R::X0));
                let mut forms = vec![
                    CodeForm::relative(&branch(funct3, 0), rv::patch_branch),
                    CodeForm::relative(&medium, rv::patch_far_branch),
                    CodeForm::relative(&far, rv::patch_far_branch),
                ];
                let reg = if rhs == R::X0 {
                    lhs
                } else if lhs == R::X0 {
                    rhs
                } else {
                    R::X0
                };
                if compressed
                    && matches!(funct3, 0 | 1)
                    && let Ok(short) = rv::compressed::branch(funct3 == 1, reg.hardware(), 0)
                {
                    forms.insert(
                        0,
                        CodeForm::relative(&short.to_le_bytes(), rv::compressed::patch_branch),
                    );
                }
                forms
            };
            e.conditional_branch(target, forms(funct3), forms(funct3 ^ 1));
        }
        Emission::Address(dst, symbol) => {
            // The ELF writer binds the low half to this AUIPC, as required by
            // the psABI. No inline literal or branch is needed in the code.
            let reg = dst.hardware();
            let mut code = (0x17u32 | reg << 7).to_le_bytes().to_vec();
            code.extend_from_slice(&rv::i(0x13, reg, 0, reg, 0).to_le_bytes());
            e.alternatives(
                Target::Symbol(symbol),
                vec![CodeForm::relocated(
                    &code,
                    ExternalRelocation {
                        kind: RelocationKind::RiscvPcRelativeAddress,
                        offset: 0,
                        symbol,
                        addend: 0,
                    },
                )],
            );
        }
        Emission::Table(index, targets) => {
            let (&default, cases) = targets
                .split_last()
                .ok_or_else(|| crate::Error::codegen("empty branch table"))?;
            if cases.is_empty() {
                return encode_instruction(e, Emission::Jump(default), compressed, external_calls);
            }
            // Copy first: spill materialization is allowed to use x5 for index.
            instruction(e, Instruction::Copy(R::X6, index, 32), compressed)?;
            instruction(
                e,
                Instruction::Constant(R::X5, cases.len() as i64, 64),
                compressed,
            )?;
            encode_instruction(
                e,
                Emission::Branch(7, R::X6, R::X5, default),
                compressed,
                external_calls,
            )?;
            // Entry-relative offsets avoid ELF relocations and remain valid
            // when layout relaxes any preceding branch. The inline table is
            // reached only by loads; dispatch always transfers control.
            let words = [
                0x17 | 31 << 7, // auipc x31, 0
                rv::i(0x13, 6, 1, 6, 2),
                rv::r(0x33, 31, 0, 31, 6, 0),
                rv::i(0x03, 5, 2, 31, 24),
                rv::r(0x33, 31, 0, 31, 5, 0),
                rv::i(0x67, 0, 0, 31, 24),
            ];
            let code: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
            let padding = if compressed {
                0x0001u16.to_le_bytes().to_vec()
            } else {
                0x00000013u32.to_le_bytes().to_vec()
            };
            e.aligned_bytes(&code, 4, &padding);
            for &target in cases {
                e.block_offset(target);
            }
        }
        Emission::Call(symbol) => {
            if external_calls == ExternalCalls::Linker {
                // CALL_PLT resolves the callee directly or through the PLT.
                // Do not request relaxation: intra-function branches have
                // already been fixed up by our layout pass.
                let mut pair = (0x17u32 | 1 << 7).to_le_bytes().to_vec();
                pair.extend_from_slice(&rv::i(0x67, 1, 0, 1, 0).to_le_bytes());
                e.alternatives(
                    Target::Symbol(symbol),
                    vec![CodeForm::relocated(
                        &pair,
                        ExternalRelocation {
                            kind: RelocationKind::RiscvCall,
                            offset: 0,
                            symbol,
                            addend: 0,
                        },
                    )],
                );
                return Ok(());
            }
            // Unknown external addresses use an aligned inline pointer. Alignment
            // belongs to this fallback form, not to direct calls or emission time.
            let mut far = (0x17u32 | 31 << 7).to_le_bytes().to_vec();
            far.extend_from_slice(&rv::i(0x03, 31, 3, 31, 16).to_le_bytes());
            far.extend_from_slice(&rv::i(0x67, 1, 0, 31, 0).to_le_bytes());
            far.extend_from_slice(&rv::j(0, 12).to_le_bytes());
            far.extend_from_slice(&[0; 8]);
            let padding = if compressed {
                0x0001u16.to_le_bytes().to_vec()
            } else {
                rv::i(0x13, 0, 0, 0, 0).to_le_bytes().to_vec()
            };
            let absolute = CodeForm::relocated(
                &far,
                ExternalRelocation {
                    kind: RelocationKind::Absolute64,
                    offset: 16,
                    symbol,
                    addend: 0,
                },
            )
            .aligned(8, &padding);
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
pub(super) struct Emit {
    pub compressed: bool,
    pub external_calls: ExternalCalls,
}
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
        let emission = inst::TargetInst::from_u32(op).emission(i, f)?;
        encode_instruction(e, emission, self.compressed, self.external_calls)
    }
}
