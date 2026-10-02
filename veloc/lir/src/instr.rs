//! Low-level IR (LIR) 指令和操作数定义

use crate::FieldView;
use cranelift_entity::entity_impl;
use veloc_mir::Type;
pub use veloc_types::{MemFlags, MemoryEffect, MemoryEffects, OpTraits};

/// Abstract register bank; target-specific selection belongs to codegen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterBank {
    GPR,
    FPR,
    VR,
    PR,
    Special,
}

/// Function-local block identity, independent of MIR and physical order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);
entity_impl!(BlockId, "block");

/// 机器指令索引
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InstId(u32);
entity_impl!(InstId, "inst");

/// 虚拟寄存器索引
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VReg(u32);
entity_impl!(VReg, "vreg");

/// A physical register location. Unlike `Reg`, it cannot contain a virtual value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PReg(u32);

impl PReg {
    pub const fn new(index: u32) -> Self {
        assert!(
            index < Reg::VREG_MARK,
            "physical register index out of range"
        );
        Self(index)
    }

    pub const fn index(self) -> u32 {
        self.0
    }
}

impl From<PReg> for Reg {
    fn from(reg: PReg) -> Self {
        Self(reg.0)
    }
}

/// 寄存器标识符 (虚拟或物理)
///
/// 最高位为 1 表示虚拟寄存器 (VReg)，为 0 表示物理寄存器 (PReg)
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Reg(pub u32);

impl Reg {
    const VREG_MARK: u32 = 1 << 31;

    /// 创建一个虚拟寄存器
    pub fn new_vreg(index: u32) -> Self {
        debug_assert!(index < Self::VREG_MARK);
        Self(index | Self::VREG_MARK)
    }

    /// 创建一个物理寄存器
    pub fn new_preg(index: u32) -> Self {
        debug_assert!(index < Self::VREG_MARK);
        Self(index)
    }

    /// 检查是否为虚拟寄存器
    pub fn is_vreg(&self) -> bool {
        (self.0 & Self::VREG_MARK) != 0
    }

    /// 检查是否为物理寄存器
    pub fn is_preg(&self) -> bool {
        (self.0 & Self::VREG_MARK) == 0
    }

    pub fn as_preg(self) -> Option<PReg> {
        self.is_preg().then_some(PReg(self.0))
    }

    pub fn as_vreg(self) -> Option<VReg> {
        self.is_vreg().then(|| VReg::from_u32(self.index()))
    }

    /// 获取原始索引 (去掉 VReg 标记)
    pub fn index(&self) -> u32 {
        self.0 & !Self::VREG_MARK
    }
}

impl core::fmt::Debug for Reg {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.is_vreg() {
            write!(f, "v{}", self.index())
        } else {
            write!(f, "p{}", self.index())
        }
    }
}

/// 栈槽标识符
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StackSlot(pub u32);
entity_impl!(StackSlot, "stackslot");

/// A state category declared by an instruction's operand signature.
/// Operands absent from this signature use ordinary register values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StateOperand {
    pub operand: crate::OperandRef,
    pub unit: PReg,
}

/// Ordinary register placement and non-renamable state are mutually exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VRegKind {
    /// None leaves bank selection to the target and operand constraints.
    Register(Option<RegisterBank>),
    /// State category, identified by its target hardware unit. The logical
    /// value can outlive its residency in this unit; ordinary RA cannot spill it.
    State(PReg),
}

/// Typed identity; state categories share storage with ordinary bank metadata.
#[derive(Debug, Clone)]
pub struct VRegData {
    pub ty: Type,
    pub kind: VRegKind,
}

impl VRegData {
    pub fn state_unit(&self) -> Option<PReg> {
        match self.kind {
            VRegKind::State(unit) => Some(unit),
            VRegKind::Register(_) => None,
        }
    }

    /// Bank selection is meaningful only for ordinary register values.
    pub fn bank(&self) -> Option<RegisterBank> {
        match self.kind {
            VRegKind::Register(bank) => bank,
            VRegKind::State(_) => panic!("state values have no register bank"),
        }
    }
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

/// 机器指令操作码
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MachineOpcode {
    /// 无效指令（占位符，用于指令融合或删除）
    Invalid,
    /// 通用操作码（需要指令选择）
    Generic(GenericOpcode),
    /// 目标架构特定操作码（指令选择后）
    Target(u32),
}

/// Borrowed access to an instruction in its function's store.
#[derive(Clone, Copy)]
pub struct InstRef<'a> {
    pub(crate) store: &'a crate::store::InstStore,
    pub(crate) id: InstId,
}

impl core::fmt::Debug for InstRef<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("InstRef")
            .field("opcode", &self.opcode())
            .field("results", &self.results())
            .field("inputs", &self.inputs())
            .field("fields", &self.fields())
            .field(
                "clobbers",
                &self.clobbers().collect::<smallvec::SmallVec<[Reg; 4]>>(),
            )
            .finish()
    }
}

impl<'a> crate::InstRead<'a> for crate::InstRef<'a> {
    type Error = crate::ValidationError;
    fn opcode(self) -> Option<crate::GenericOpcode> {
        self.generic_opcode()
    }
    fn results(self) -> &'a [crate::Reg] {
        self.results()
    }
    fn inputs(self) -> &'a [crate::Reg] {
        self.inputs()
    }
    fn fields(self) -> crate::FieldView<'a> {
        self.fields()
    }
    fn error(self, message: &str) -> crate::ValidationError {
        crate::ValidationError {
            opcode: self.opcode(),
            reason: message.into(),
        }
    }
}

impl<'a> InstRef<'a> {
    /// Symbolic stack boundaries survive instruction selection and allocation.
    pub fn is_call_frame(&self) -> bool {
        matches!(
            self.opcode(),
            MachineOpcode::Generic(GenericOpcode::CallFrameSetup | GenericOpcode::CallFrameDestroy)
        )
    }
    /// Successor identities in instruction operand order, including repeated targets.
    pub fn edge_ids(self) -> impl Iterator<Item = crate::EdgeId> + 'a {
        self.store.edge_ids(self.id)
    }

    pub fn edge(self, id: crate::EdgeId) -> crate::Successor<&'a [Reg]> {
        assert!(
            self.store.edge_ids(self.id).any(|edge| edge == id),
            "edge belongs to another instruction"
        );
        self.store.edge(id)
    }
    /// Conservative, nontrapping value computation. This deliberately excludes
    /// loads, allocation, physical-register dependencies and auxiliary effects.
    pub fn is_pure_value(self) -> bool {
        let Some(opcode) = self.generic_opcode() else {
            return false;
        };
        let meta = opcode.meta();
        opcode.control() == crate::ControlFlow::Next
            && meta.memory.is_none()
            && !meta
                .traits
                .intersects(OpTraits::MAY_TRAP | OpTraits::ABORT | OpTraits::TERMINATOR)
            && self.mem_flags().is_none()
            && self.clobbers().next().is_none()
            && !self.results().is_empty()
            && self
                .results()
                .iter()
                .chain(self.inputs())
                .all(|reg| reg.is_vreg())
            && self.store.call_info(self.id).is_none()
    }

    pub fn opcode(self) -> MachineOpcode {
        self.store.opcode(self.id)
    }
    pub fn results(self) -> &'a [Reg] {
        self.store.results(self.id)
    }
    pub fn inputs(self) -> &'a [Reg] {
        self.store.inputs(self.id)
    }
    pub fn fields(self) -> crate::FieldView<'a> {
        self.store.fields(self.id)
    }
    pub fn constraints(self) -> &'a [crate::OperandConstraint] {
        self.store.constraints(self.id)
    }
    pub fn mem_flags(self) -> Option<crate::MemFlags> {
        self.fields().mem_flags()
    }

    /// Distinct register access for dependency, liveness and pressure analyses.
    /// Operand APIs retain their positions and multiplicity.
    pub fn register_access(self) -> crate::RegisterAccess<'a> {
        crate::RegisterAccess::new(self)
    }

    /// 检查指令是否有效
    pub fn is_invalid(&self) -> bool {
        matches!(self.opcode(), MachineOpcode::Invalid)
    }

    /// Instruction and ABI destruction effects have no use-def occurrences.
    pub fn clobbers(self) -> impl Iterator<Item = Reg> + 'a {
        self.store.clobbers(self.id).iter().copied().chain(
            self.store
                .call_info(self.id)
                .into_iter()
                .flat_map(|info| info.clobbers.iter()),
        )
    }

    /// Result definitions; clobbers are separate destruction effects.
    pub fn defs(self) -> impl Iterator<Item = Reg> + 'a {
        self.results().iter().copied()
    }

    /// Register inputs and edge arguments, including explicit physical inputs.
    /// Repeated operands remain separate occurrences.
    pub fn uses(self) -> impl Iterator<Item = Reg> + 'a {
        let operands = self.inputs().iter().copied();
        operands.chain(self.store.edge_args(self.id))
    }

    /// 检查是否是通用操作码（尚未指令选择）
    pub fn is_generic(&self) -> bool {
        matches!(self.opcode(), MachineOpcode::Generic(_))
    }

    /// 检查是否是目标特定操作码
    pub fn is_target(&self) -> bool {
        matches!(self.opcode(), MachineOpcode::Target(_))
    }

    /// 如果该指令是通用 LIR 指令，返回其 GenericOpcode。
    pub fn generic_opcode(&self) -> Option<GenericOpcode> {
        match self.opcode() {
            MachineOpcode::Generic(opcode) => Some(opcode),
            _ => None,
        }
    }
}
