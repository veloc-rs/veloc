use super::callconv::CallConv;
use super::types::TargetArch;
use super::Reg;
use alloc::vec::Vec;
use veloc_ir::Type;

/// ABI 值分类
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiValueClass {
    Integer,
    Float,
    Vector,
    Memory,
}

/// ABI 栈位置的基准
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiStackBase {
    /// 被调用者视角的传入参数区域，例如 x86_64 System V 下的 `[rbp + 16]`
    IncomingArgs,
    /// 调用点视角的传出参数区域
    OutgoingArgs,
}

/// ABI 位置
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiLocation {
    Reg(Reg),
    Stack {
        base: AbiStackBase,
        base_reg: Option<Reg>,
        offset: i32,
        size: u32,
        align: u32,
    },
}

/// 单个值的一部分（为未来的多寄存器/聚合拆分预留）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiPart {
    pub ty: Type,
    pub class: AbiValueClass,
    pub loc: AbiLocation,
}

/// 参数或返回值的 ABI 分配结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AbiAssignment {
    pub index: usize,
    pub ty: Type,
    pub parts: Vec<AbiPart>,
}

impl AbiAssignment {
    pub fn single_reg(&self) -> Option<Reg> {
        match self.parts.as_slice() {
            [AbiPart {
                loc: AbiLocation::Reg(reg),
                ..
            }] => Some(*reg),
            _ => None,
        }
    }

    pub fn single_stack_slot(&self) -> Option<(AbiStackBase, Option<Reg>, i32, u32, u32)> {
        match self.parts.as_slice() {
            [AbiPart {
                loc:
                    AbiLocation::Stack {
                        base,
                        base_reg,
                        offset,
                        size,
                        align,
                    },
                ..
            }] => Some((*base, *base_reg, *offset, *size, *align)),
            _ => None,
        }
    }
}

/// 调用约定计划
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallConvPlan {
    pub call_conv: CallConv,
    pub arch: TargetArch,
    pub args: Vec<AbiAssignment>,
    pub returns: Vec<AbiAssignment>,
    pub stack_alignment: u32,
    /// 仅统计真实的参数区大小，不包含调用者/被调用者额外的帧头。
    pub stack_arg_bytes: u32,
}

/// ABI 描述中的寄存器池
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiRegisterPool {
    pub class: AbiValueClass,
    pub regs: &'static [Reg],
}

/// ABI 描述中的 preserved 集合
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiPreservedSet {
    pub bank: &'static str,
    pub regs: &'static [Reg],
}

/// ABI 描述中的栈规则
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiStackDescriptor {
    pub align: u32,
    pub incoming_base_reg: Option<Reg>,
    pub incoming_base_offset: i32,
    pub outgoing_slot_size: u32,
    pub outgoing_slot_align: u32,
}

/// 由 DSL 生成或手工定义的 ABI 描述
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AbiDescriptor {
    pub name: &'static str,
    pub arch: TargetArch,
    pub classifier: Option<&'static str>,
    pub stack: AbiStackDescriptor,
    pub args: &'static [AbiRegisterPool],
    pub returns: &'static [AbiRegisterPool],
    pub preserved: &'static [AbiPreservedSet],
}

impl AbiDescriptor {
    pub fn regs_for_class(&self, class: AbiValueClass, returns: bool) -> &'static [Reg] {
        let pools = if returns { self.returns } else { self.args };
        pools
            .iter()
            .find(|pool| pool.class == class)
            .map(|pool| pool.regs)
            .unwrap_or(&[])
    }
}

/// ABI 分类器函数
pub type AbiClassifierFn = fn(Type) -> Result<AbiValueClass, crate::error::Error>;

/// ABI 分类器注册项
#[derive(Clone, Copy)]
pub struct AbiClassifierEntry {
    pub name: &'static str,
    pub func: AbiClassifierFn,
}
