use super::abi::{
    AbiAssignment, AbiClassifierEntry, AbiDescriptor, AbiLocation, AbiPart, AbiStackBase,
    AbiValueClass, CallConvPlan,
};
use super::types::TargetArch;
use super::Reg;
use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use veloc_ir::{Signature, Type};

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

impl From<veloc_ir::CallConv> for CallConv {
    fn from(value: veloc_ir::CallConv) -> Self {
        match value {
            veloc_ir::CallConv::SystemV => CallConv::SystemV,
        }
    }
}

impl CallConv {
    /// 为被调用者入口构建 ABI 计划。
    pub fn plan_signature(
        &self,
        arch: TargetArch,
        sig: &Signature,
    ) -> Result<CallConvPlan, crate::error::Error> {
        self.plan_types(arch, &sig.params, &sig.returns, AbiStackBase::IncomingArgs)
    }

    /// 为调用点构建 ABI 计划。
    pub fn plan_callsite(
        &self,
        arch: TargetArch,
        arg_types: &[Type],
        ret_types: &[Type],
    ) -> Result<CallConvPlan, crate::error::Error> {
        self.plan_types(arch, arg_types, ret_types, AbiStackBase::OutgoingArgs)
    }

    fn plan_types(
        &self,
        arch: TargetArch,
        arg_types: &[Type],
        ret_types: &[Type],
        stack_base: AbiStackBase,
    ) -> Result<CallConvPlan, crate::error::Error> {
        let descriptor = self.descriptor(arch)?;
        self.plan_with_descriptor(descriptor, arg_types, ret_types, stack_base)
    }

    /// 获取参数寄存器列表
    pub fn arg_regs(&self, arch: TargetArch) -> Vec<Reg> {
        self.descriptor(arch)
            .map(|descriptor| {
                descriptor
                    .regs_for_class(AbiValueClass::Integer, false)
                    .to_vec()
            })
            .unwrap_or_default()
    }

    /// 获取返回值寄存器
    pub fn return_reg(&self, arch: TargetArch) -> Reg {
        self.descriptor(arch)
            .expect("unsupported architecture for return register lookup")
            .regs_for_class(AbiValueClass::Integer, true)
            .first()
            .copied()
            .expect("ABI descriptor must define at least one integer return register")
    }

    /// 获取该调用约定下需要由被调用者保留的寄存器集合。
    pub fn preserved_regs(&self, arch: TargetArch) -> Vec<Reg> {
        let Ok(descriptor) = self.descriptor(arch) else {
            return Vec::new();
        };

        let mut regs = Vec::new();
        for set in descriptor.preserved {
            for &reg in set.regs {
                if !regs.contains(&reg) {
                    regs.push(reg);
                }
            }
        }
        regs
    }

    fn descriptor(&self, arch: TargetArch) -> Result<&'static AbiDescriptor, crate::error::Error> {
        match (self, arch) {
            (CallConv::SystemV, TargetArch::X86_64) => Ok(x86_64_systemv_descriptor()),
            (CallConv::WindowsFastcall, TargetArch::X86_64) => Ok(x86_64_win64_descriptor()),
            (CallConv::AAPCS, TargetArch::AArch64) => Ok(aarch64_aapcs_descriptor()),
            _ => Err(crate::error::Error::codegen(format!(
                "unsupported calling convention {:?} for architecture {:?}",
                self, arch
            ))),
        }
    }

    fn plan_with_descriptor(
        &self,
        descriptor: &'static AbiDescriptor,
        arg_types: &[Type],
        ret_types: &[Type],
        stack_base: AbiStackBase,
    ) -> Result<CallConvPlan, crate::error::Error> {
        let mut context = AbiAllocationContext::new(descriptor, stack_base);

        let args = arg_types
            .iter()
            .enumerate()
            .map(|(index, &ty)| {
                let class = classify_with_descriptor(descriptor, ty)?;
                let loc = context.allocate(class, false);
                Ok::<AbiAssignment, crate::error::Error>(AbiAssignment {
                    index,
                    ty,
                    parts: vec![AbiPart { ty, class, loc }],
                })
            })
            .collect::<Result<Vec<_>, crate::error::Error>>()?;

        context.reset_for_returns();

        let returns = ret_types
            .iter()
            .enumerate()
            .map(|(index, &ty)| {
                let class = classify_with_descriptor(descriptor, ty)?;
                let loc = context.allocate_return(class)?;
                Ok::<AbiAssignment, crate::error::Error>(AbiAssignment {
                    index,
                    ty,
                    parts: vec![AbiPart { ty, class, loc }],
                })
            })
            .collect::<Result<Vec<_>, crate::error::Error>>()?;

        Ok(CallConvPlan {
            call_conv: *self,
            arch: descriptor.arch,
            args,
            returns,
            stack_alignment: descriptor.stack.align,
            stack_arg_bytes: context.stack_arg_bytes,
        })
    }
}

struct AbiAllocationContext {
    descriptor: &'static AbiDescriptor,
    stack_base: AbiStackBase,
    next_int_reg: usize,
    next_float_reg: usize,
    next_vector_reg: usize,
    stack_offset: i32,
    stack_arg_bytes: u32,
}

impl AbiAllocationContext {
    fn new(descriptor: &'static AbiDescriptor, stack_base: AbiStackBase) -> Self {
        Self {
            descriptor,
            stack_base,
            next_int_reg: 0,
            next_float_reg: 0,
            next_vector_reg: 0,
            stack_offset: match stack_base {
                AbiStackBase::IncomingArgs => descriptor.stack.incoming_base_offset,
                AbiStackBase::OutgoingArgs => 0,
            },
            stack_arg_bytes: 0,
        }
    }

    fn reset_for_returns(&mut self) {
        self.next_int_reg = 0;
        self.next_float_reg = 0;
        self.next_vector_reg = 0;
    }

    fn allocate(&mut self, class: AbiValueClass, returns: bool) -> AbiLocation {
        if class == AbiValueClass::Memory {
            return self.stack_location();
        }

        let regs = self.descriptor.regs_for_class(class, returns);
        let cursor = match class {
            AbiValueClass::Integer => &mut self.next_int_reg,
            AbiValueClass::Float => &mut self.next_float_reg,
            AbiValueClass::Vector => &mut self.next_vector_reg,
            _ => unreachable!(),
        };

        if let Some(&reg) = regs.get(*cursor) {
            *cursor += 1;
            AbiLocation::Reg(reg)
        } else {
            self.stack_location()
        }
    }

    fn allocate_return(
        &mut self,
        class: AbiValueClass,
    ) -> Result<AbiLocation, crate::error::Error> {
        if class == AbiValueClass::Memory {
            return Err(crate::error::Error::codegen(format!(
                "ABI `{}` return type classified as memory is not supported yet",
                self.descriptor.name
            )));
        }

        // 返回值通常不回退到堆栈（取决于具体 ABI，但此处逻辑保持原样）
        let regs = self.descriptor.regs_for_class(class, true);
        let cursor = match class {
            AbiValueClass::Integer => &mut self.next_int_reg,
            AbiValueClass::Float => &mut self.next_float_reg,
            AbiValueClass::Vector => &mut self.next_vector_reg,
            _ => unreachable!(),
        };

        let Some(&reg) = regs.get(*cursor) else {
            return Err(crate::error::Error::codegen(format!(
                "ABI `{}` ran out of return registers for class {:?}",
                self.descriptor.name, class
            )));
        };
        *cursor += 1;
        Ok(AbiLocation::Reg(reg))
    }

    fn stack_location(&mut self) -> AbiLocation {
        self.stack_arg_bytes += self.descriptor.stack.outgoing_slot_size;
        let loc = AbiLocation::Stack {
            base: self.stack_base,
            base_reg: match self.stack_base {
                AbiStackBase::IncomingArgs => self.descriptor.stack.incoming_base_reg,
                AbiStackBase::OutgoingArgs => None,
            },
            offset: self.stack_offset,
            size: self.descriptor.stack.outgoing_slot_size,
            align: self.descriptor.stack.outgoing_slot_align,
        };
        self.stack_offset += self.descriptor.stack.outgoing_slot_size as i32;
        loc
    }
}

#[inline]
fn classify_with_descriptor(
    descriptor: &AbiDescriptor,
    ty: Type,
) -> Result<AbiValueClass, crate::error::Error> {
    let classifier_name = descriptor.classifier.unwrap_or("scalar");
    let Some(entry) = ABI_CLASSIFIER_REGISTRY
        .iter()
        .find(|entry| entry.name == classifier_name)
    else {
        return Err(crate::error::Error::codegen(format!(
            "unknown ABI classifier `{}` for descriptor `{}`",
            classifier_name, descriptor.name
        )));
    };
    (entry.func)(ty)
}

#[inline]
fn classify_scalar_abi_type(ty: Type) -> Result<AbiValueClass, crate::error::Error> {
    if ty.is_integer() || ty.is_ptr() {
        Ok(AbiValueClass::Integer)
    } else if ty.is_float() {
        Ok(AbiValueClass::Float)
    } else if ty.is_vector() || ty.is_predicate() {
        Ok(AbiValueClass::Vector)
    } else {
        Err(crate::error::Error::codegen(format!(
            "unsupported ABI value type {:?}",
            ty
        )))
    }
}

fn x86_64_sysv_classifier(ty: Type) -> Result<AbiValueClass, crate::error::Error> {
    classify_scalar_abi_type(ty)
}

fn x86_64_win64_classifier(ty: Type) -> Result<AbiValueClass, crate::error::Error> {
    classify_scalar_abi_type(ty)
}

fn aarch64_aapcs_classifier(ty: Type) -> Result<AbiValueClass, crate::error::Error> {
    classify_scalar_abi_type(ty)
}

static ABI_CLASSIFIER_REGISTRY: &[AbiClassifierEntry] = &[
    AbiClassifierEntry {
        name: "scalar",
        func: classify_scalar_abi_type,
    },
    AbiClassifierEntry {
        name: "x86_64_sysv_classifier",
        func: x86_64_sysv_classifier,
    },
    AbiClassifierEntry {
        name: "x86_64_win64_classifier",
        func: x86_64_win64_classifier,
    },
    AbiClassifierEntry {
        name: "aarch64_aapcs_classifier",
        func: aarch64_aapcs_classifier,
    },
];

#[inline]
pub(crate) fn x86_64_rax() -> Reg {
    crate::target::x86_64::isle::REG_RAX
}

#[inline]
pub(crate) fn x86_64_rdx() -> Reg {
    crate::target::x86_64::isle::REG_RDX
}

fn x86_64_systemv_arg_regs() -> &'static [Reg] {
    x86_64_systemv_descriptor().regs_for_class(AbiValueClass::Integer, false)
}

fn x86_64_systemv_descriptor() -> &'static AbiDescriptor {
    &crate::target::x86_64::isle::ABI_X86_64SystemV
}

fn x86_64_win64_descriptor() -> &'static AbiDescriptor {
    &crate::target::x86_64::isle::ABI_X86_64WindowsFastcall
}

fn aarch64_aapcs_descriptor() -> &'static AbiDescriptor {
    static ARG_GPRS: [Reg; 8] = [
        Reg(0),
        Reg(1),
        Reg(2),
        Reg(3),
        Reg(4),
        Reg(5),
        Reg(6),
        Reg(7),
    ];
    static RET_GPRS: [Reg; 2] = [Reg(0), Reg(1)];
    static ARG_POOLS: [super::abi::AbiRegisterPool; 1] = [super::abi::AbiRegisterPool {
        class: AbiValueClass::Integer,
        regs: &ARG_GPRS,
    }];
    static RET_POOLS: [super::abi::AbiRegisterPool; 1] = [super::abi::AbiRegisterPool {
        class: AbiValueClass::Integer,
        regs: &RET_GPRS,
    }];
    static PRESERVED: [super::abi::AbiPreservedSet; 0] = [];
    static DESCRIPTOR: AbiDescriptor = AbiDescriptor {
        name: "AArch64AAPCS",
        arch: TargetArch::AArch64,
        classifier: None,
        stack: super::abi::AbiStackDescriptor {
            align: 16,
            incoming_base_reg: None,
            incoming_base_offset: 0,
            outgoing_slot_size: 8,
            outgoing_slot_align: 8,
        },
        args: &ARG_POOLS,
        returns: &RET_POOLS,
        preserved: &PRESERVED,
    };
    &DESCRIPTOR
}

#[cfg(test)]
mod tests {
    use super::{x86_64_rax, x86_64_rdx, CallConv, Reg, TargetArch};
    use crate::target::arch::{AbiLocation, AbiStackBase};
    use alloc::vec;
    use veloc_ir::Type;

    #[test]
    fn test_x86_64_systemv_callee_plan_uses_registers_then_stack() {
        let plan = CallConv::SystemV
            .plan_callsite(
                TargetArch::X86_64,
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
            plan.args[0].single_reg(),
            Some(super::x86_64_systemv_arg_regs()[0])
        );
        assert_eq!(
            plan.args[5].single_reg(),
            Some(super::x86_64_systemv_arg_regs()[5])
        );
        assert_eq!(
            plan.args[6].single_stack_slot(),
            Some((AbiStackBase::OutgoingArgs, None, 0, 8, 8))
        );
        assert_eq!(plan.stack_arg_bytes, 8);
    }

    #[test]
    fn test_x86_64_systemv_return_plan_uses_rax_rdx() {
        let plan = CallConv::SystemV
            .plan_callsite(TargetArch::X86_64, &[], &[Type::I64, Type::I32])
            .unwrap();

        assert_eq!(plan.returns[0].single_reg(), Some(x86_64_rax()));
        assert_eq!(plan.returns[1].single_reg(), Some(x86_64_rdx()));
        assert!(matches!(plan.returns[0].parts[0].loc, AbiLocation::Reg(_)));
    }

    #[test]
    fn test_x86_64_win64_plan_uses_fastcall_registers() {
        let regs = CallConv::WindowsFastcall.arg_regs(TargetArch::X86_64);
        assert_eq!(regs, vec![Reg(1), Reg(2), Reg(8), Reg(9)]);

        let plan = CallConv::WindowsFastcall
            .plan_callsite(
                TargetArch::X86_64,
                &[Type::I64, Type::I64, Type::I64, Type::I64, Type::I64],
                &[Type::I64],
            )
            .unwrap();

        assert_eq!(plan.args[0].single_reg(), Some(Reg(1)));
        assert_eq!(plan.args[3].single_reg(), Some(Reg(9)));
        assert_eq!(
            plan.args[4].single_stack_slot(),
            Some((AbiStackBase::OutgoingArgs, None, 0, 8, 8))
        );
        assert_eq!(plan.returns[0].single_reg(), Some(Reg(0)));
    }

    #[test]
    fn test_aarch64_aapcs_plan_uses_descriptor_engine() {
        let plan = CallConv::AAPCS
            .plan_callsite(
                TargetArch::AArch64,
                &[
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                    Type::I64,
                ],
                &[Type::I64, Type::I32],
            )
            .unwrap();

        assert_eq!(plan.args[0].single_reg(), Some(Reg(0)));
        assert_eq!(plan.args[7].single_reg(), Some(Reg(7)));
        assert_eq!(
            plan.args[8].single_stack_slot(),
            Some((AbiStackBase::OutgoingArgs, None, 0, 8, 8))
        );
        assert_eq!(plan.returns[0].single_reg(), Some(Reg(0)));
        assert_eq!(plan.returns[1].single_reg(), Some(Reg(1)));
    }

    #[test]
    fn test_x86_64_systemv_float_plan_uses_xmm_registers() {
        let plan = CallConv::SystemV
            .plan_callsite(
                TargetArch::X86_64,
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

        assert_eq!(plan.args[0].single_reg(), Some(Reg(16)));
        assert_eq!(plan.args[7].single_reg(), Some(Reg(23)));
        assert_eq!(
            plan.args[8].single_stack_slot(),
            Some((AbiStackBase::OutgoingArgs, None, 0, 8, 8))
        );
        assert_eq!(plan.returns[0].single_reg(), Some(Reg(16)));
    }

    #[test]
    fn test_x86_64_systemv_vector_plan_uses_xmm_registers() {
        let plan = CallConv::SystemV
            .plan_callsite(TargetArch::X86_64, &[Type::F32X4], &[Type::F64X2])
            .unwrap();

        assert_eq!(plan.args[0].single_reg(), Some(Reg(16)));
        assert_eq!(plan.returns[0].single_reg(), Some(Reg(16)));
    }
}
