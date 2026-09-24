use super::Reg;
use super::abi::{AbiAssignment, AbiDescriptor, AbiLocation, AbiPlan, AbiState};
use super::types::TargetArch;
use std::format;
use std::vec::Vec;
use veloc_mir::Type;
use veloc_types::DataLayout;

/// 调用约定
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallConv {
    /// System V AMD64 ABI (Linux, macOS, etc.)
    SystemV,
    /// Windows x64 ABI
    WindowsFastcall,
    /// AArch64 AAPCS
    AAPCS,
    /// RISC-V ABI
    RiscvABI,
    /// WebAssembly
    WasmC,
}

impl From<veloc_mir::CallConv> for CallConv {
    fn from(value: veloc_mir::CallConv) -> Self {
        match value {
            veloc_mir::CallConv::SystemV => CallConv::SystemV,
        }
    }
}

impl CallConv {
    /// Compute the same protocol for caller and callee.
    pub fn plan(
        &self,
        arch: TargetArch,
        layout: &DataLayout,
        args: &[Type],
        returns: &[Type],
    ) -> Result<AbiPlan, crate::error::Error> {
        self.plan_with_descriptor(self.descriptor(arch)?, layout, args, returns)
    }

    /// 获取该调用约定下需要由被调用者保留的寄存器集合。
    pub fn preserved_regs(&self, arch: TargetArch) -> &'static [Reg] {
        self.descriptor(arch)
            .expect("unsupported calling convention")
            .preserved
    }

    fn descriptor(&self, arch: TargetArch) -> Result<&'static AbiDescriptor, crate::error::Error> {
        match (self, arch) {
            (CallConv::SystemV, TargetArch::X86_64) => Ok(x86_64_systemv_descriptor()),
            (CallConv::WindowsFastcall, TargetArch::X86_64) => Ok(x86_64_win64_descriptor()),
            _ => Err(crate::error::Error::codegen(format!(
                "unsupported calling convention {:?} for architecture {:?}",
                self, arch
            ))),
        }
    }

    fn plan_with_descriptor(
        &self,
        descriptor: &'static AbiDescriptor,
        layout: &DataLayout,
        arg_types: &[Type],
        ret_types: &[Type],
    ) -> Result<AbiPlan, crate::error::Error> {
        let mut args_state = AbiState::new(descriptor.stack);
        let mut ret_state = AbiState::new(super::abi::StackArea { size: 0, align: 1 });
        let plan = |types: &[Type], assign: super::abi::AbiAssignFn, state: &mut AbiState| {
            types
                .iter()
                .map(|&ty| {
                    let loc = assign(ty, state)?;
                    if let AbiLocation::Stack { size, .. } = loc {
                        let bytes = layout
                            .layout_of(ty)
                            .and_then(|layout| layout.store_size.fixed_bytes())
                            .ok_or_else(|| {
                                crate::Error::codegen(format!(
                                    "ABI stack transfer requires a fixed storage layout for {ty:?}"
                                ))
                            })?;
                        // A supported type must fit the slot chosen by the ABI rules.
                        assert!(bytes <= size, "ABI stack slot cannot hold {ty:?}");
                    }
                    Ok(AbiAssignment { ty, loc })
                })
                .collect::<Result<Vec<_>, crate::error::Error>>()
        };
        let args = plan(arg_types, descriptor.args, &mut args_state)?;
        let returns = plan(ret_types, descriptor.returns, &mut ret_state)?;
        Ok(AbiPlan {
            abi: descriptor,
            args,
            returns,
            stack: args_state.stack,
        })
    }
}

fn x86_64_systemv_descriptor() -> &'static AbiDescriptor {
    &crate::target::x86_64::inst::ABI_X86_64SYSTEMV
}

fn x86_64_win64_descriptor() -> &'static AbiDescriptor {
    &crate::target::x86_64::inst::ABI_X86_64WINDOWSFASTCALL
}

#[cfg(test)]
mod tests {
    use super::{CallConv, Reg, TargetArch};
    use crate::target::AbiLocation;
    use crate::target::x86_64::inst::{REG_RAX, REG_RDX};
    use veloc_mir::Type;

    #[test]
    fn test_x86_64_systemv_callee_plan_uses_registers_then_stack() {
        let plan = CallConv::SystemV
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                ],
                &[],
            )
            .unwrap();

        assert_eq!(
            plan.args[0].loc,
            AbiLocation::Reg(crate::target::x86_64::inst::REG_RDI)
        );
        assert_eq!(
            plan.args[5].loc,
            AbiLocation::Reg(crate::target::x86_64::inst::REG_R9)
        );
        assert_eq!(
            plan.args[6].loc,
            AbiLocation::Stack {
                offset: 0,
                size: 8,
                align: 8
            }
        );
        assert_eq!(plan.stack.size, 8);
    }

    #[test]
    fn test_x86_64_systemv_return_plan_uses_rax_rdx() {
        let plan = CallConv::SystemV
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[],
                &[Type::I64, Type::I32],
            )
            .unwrap();

        assert_eq!(plan.returns[0].loc, AbiLocation::Reg(REG_RAX));
        assert_eq!(plan.returns[1].loc, AbiLocation::Reg(REG_RDX));
        assert!(matches!(plan.returns[0].loc, AbiLocation::Reg(_)));
    }

    #[test]
    fn test_x86_64_win64_plan_uses_fastcall_registers() {
        let plan = CallConv::WindowsFastcall
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[Type::I64, Type::I64, Type::I64, Type::I64, Type::I64],
                &[Type::I64],
            )
            .unwrap();

        assert_eq!(plan.args[0].loc, AbiLocation::Reg(Reg(1)));
        assert_eq!(plan.args[3].loc, AbiLocation::Reg(Reg(9)));
        assert_eq!(
            plan.args[4].loc,
            AbiLocation::Stack {
                offset: 32,
                size: 8,
                align: 8
            }
        );
        assert_eq!(plan.returns[0].loc, AbiLocation::Reg(Reg(0)));
    }

    #[test]
    fn test_x86_64_systemv_float_plan_uses_xmm_registers() {
        let plan = CallConv::SystemV
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[
                    Type::F32,
                    Type::F64,
                    Type::F32,
                    Type::F64,
                    Type::F32,
                    Type::F64,
                    Type::F32,
                    Type::F64,
                    Type::F32,
                ],
                &[Type::F64],
            )
            .unwrap();

        assert_eq!(plan.args[0].loc, AbiLocation::Reg(Reg(16)));
        assert_eq!(plan.args[7].loc, AbiLocation::Reg(Reg(23)));
        assert_eq!(
            plan.args[8].loc,
            AbiLocation::Stack {
                offset: 0,
                size: 8,
                align: 8
            }
        );
        assert_eq!(plan.returns[0].loc, AbiLocation::Reg(Reg(16)));
    }

    #[test]
    fn test_x86_64_systemv_vector_plan_uses_xmm_registers() {
        let plan = CallConv::SystemV
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[veloc_mir::Type::F32X4],
                &[veloc_mir::Type::F64X2],
            )
            .unwrap();

        assert_eq!(plan.args[0].loc, AbiLocation::Reg(Reg(16)));
        assert_eq!(plan.returns[0].loc, AbiLocation::Reg(Reg(16)));
    }
    #[test]
    fn mixed_domains_share_occupancy_and_stack_alignment() {
        use crate::target::x86_64::inst::*;
        let mut types = std::vec![Type::F64; 8];
        types.extend([Type::F64, Type::F32X4]);
        let plan = CallConv::SystemV
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &types,
                &[Type::F64, Type::F32X4],
            )
            .unwrap();
        assert_eq!(plan.returns[0].loc, AbiLocation::Reg(REG_XMM0));
        assert_eq!(plan.returns[1].loc, AbiLocation::Reg(REG_XMM1));
        assert_eq!(
            plan.args[8].loc,
            AbiLocation::Stack {
                offset: 0,
                size: 8,
                align: 8
            }
        );
        assert_eq!(
            plan.args[9].loc,
            AbiLocation::Stack {
                offset: 16,
                size: 16,
                align: 16
            }
        );
        assert_eq!(plan.stack.size, 32);

        let plan = CallConv::SystemV
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[Type::F64, Type::F32X4, Type::F32],
                &[],
            )
            .unwrap();
        assert_eq!(plan.args[0].loc, AbiLocation::Reg(REG_XMM0));
        assert_eq!(plan.args[1].loc, AbiLocation::Reg(REG_XMM1));
        assert_eq!(plan.args[2].loc, AbiLocation::Reg(REG_XMM2));

        let plan = CallConv::WindowsFastcall
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[Type::I64, Type::F64, Type::I64, Type::F32, Type::I64],
                &[],
            )
            .unwrap();
        assert_eq!(plan.args[0].loc, AbiLocation::Reg(REG_RCX));
        assert_eq!(plan.args[1].loc, AbiLocation::Reg(REG_XMM1));
        assert_eq!(plan.args[2].loc, AbiLocation::Reg(REG_R8));
        assert_eq!(plan.args[3].loc, AbiLocation::Reg(REG_XMM3));
        assert_eq!(
            plan.args[4].loc,
            AbiLocation::Stack {
                offset: 32,
                size: 8,
                align: 8
            }
        );
        assert_eq!(plan.stack.size, 40);
        let empty = CallConv::WindowsFastcall
            .plan(
                TargetArch::X86_64,
                &crate::target::x86_64::DATA_LAYOUT,
                &[],
                &[],
            )
            .unwrap();
        assert_eq!(empty.stack.size, 32);
        // Indirect vector arguments require a separate conversion plan:
        // never silently pass them directly using the return-value rule.
        assert!(
            CallConv::WindowsFastcall
                .plan(
                    TargetArch::X86_64,
                    &crate::target::x86_64::DATA_LAYOUT,
                    &[Type::F32X4],
                    &[],
                )
                .is_err()
        );
    }
}
