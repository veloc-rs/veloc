//! x86_64 Machine Code Emitter
//!
//! x86_64 后端只保留发射时机和 fixup 收尾逻辑。
//! 具体编码字节序列由 ISLE DSL 生成的 `TargetInst::emit()` 负责。

use crate::mir::{MachineBlock, MachineFunction, MachineInst, MachineOpcode};
use crate::target::arch::TargetEmitter;

/// x86_64 机器码发射器实现
pub struct X86_64CodeEmitter;

impl X86_64CodeEmitter {
    pub fn new() -> Self {
        Self
    }
}

impl TargetEmitter for X86_64CodeEmitter {
    fn begin_block(
        &self,
        emitter: &mut crate::Emitter,
        block: &MachineBlock,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        emitter.mark_block(block.id);
        Ok(())
    }

    fn emit_instruction(
        &self,
        emitter: &mut crate::Emitter,
        inst: &MachineInst,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        match &inst.opcode {
            MachineOpcode::Invalid => Err(crate::error::Error::Emit(
                inst.opcode.clone(),
                alloc::string::String::from("Invalid opcode cannot be emitted"),
            )),
            MachineOpcode::Generic(_) => Err(crate::error::Error::Emit(
                inst.opcode.clone(),
                alloc::string::String::from(
                    "Generic opcode should be translated to Target opcode before emission",
                ),
            )),
            MachineOpcode::Target(target_inst_code) => {
                let target = crate::target::x86_64::isle::TargetInst::from_u32(*target_inst_code);
                target.emit::<Self>(emitter, inst).map_err(|err| match err {
                    crate::error::Error::Emit(opcode, message) => crate::error::Error::Emit(
                        opcode,
                        alloc::format!("{} | inst={:?}", message, inst),
                    ),
                    other => other,
                })
            }
        }
    }

    fn finish_function(
        &self,
        emitter: &mut crate::Emitter,
        _mfunc: &MachineFunction,
    ) -> Result<(), crate::error::Error> {
        emitter.apply_fixups()
    }
}
