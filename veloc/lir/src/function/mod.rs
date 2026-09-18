//! LIR 机器函数与基本块定义

use super::{CallInfo, InstExtra, InstId, InstRef, Reg, StackSlot, VReg, VRegData};
use crate::BlockId as Block;
use crate::InstWriter;
use crate::RegisterBank;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use cranelift_entity::PrimaryMap;
use veloc_mir::Type;

#[derive(Debug, Clone, Default)]
struct BlockData {
    params: Vec<Reg>,
}

/// All function-local identities and their editable body.
#[derive(Debug, Clone, Default)]
pub struct FuncBody {
    blocks: PrimaryMap<Block, BlockData>,
    entry: Option<Block>,
    layout: crate::layout::Layout,
    store: crate::store::InstStore,
    vregs: PrimaryMap<VReg, VRegData>,
    changed_blocks: Option<Vec<Block>>,
}

impl FuncBody {
    pub fn entry_block(&self) -> Option<Block> {
        self.entry
    }
    pub fn layout(&self) -> &crate::layout::Layout {
        &self.layout
    }
    pub fn inst(&self, id: InstId) -> InstRef<'_> {
        self.store.get(id)
    }
    pub fn vregs(&self) -> &PrimaryMap<VReg, VRegData> {
        &self.vregs
    }
    pub fn block_params(&self, block: Block) -> Option<&[Reg]> {
        self.layout
            .contains_block(block)
            .then(|| self.blocks[block].params.as_slice())
    }
}

/// Append-only access for selection rules that need fresh virtual registers.
pub struct VRegBuilder<'a>(pub(crate) &'a mut PrimaryMap<VReg, VRegData>);
impl VRegBuilder<'_> {
    pub fn alloc(&mut self, data: VRegData) -> Reg {
        Reg::new_vreg(self.0.push(data).as_u32())
    }
    pub fn get(&self, reg: VReg) -> &VRegData {
        &self.0[reg]
    }
}

/// 栈槽数据
#[derive(Debug, Clone, Copy)]
pub enum StackBase {
    /// Symbolic local frame base, resolved by the target at emission.
    Frame,
    Reg(Reg),
}

impl StackBase {
    pub fn resolve(self, frame: Reg) -> Reg {
        match self {
            Self::Frame => frame,
            Self::Reg(reg) => reg,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StackSlotData {
    pub base: StackBase,
    pub size: u32,
    pub align: u32,
    pub offset: i32,
}

/// 栈帧信息
#[derive(Debug, Clone)]
pub struct StackFrame {
    /// 局部变量占用的栈空间
    pub local_size: u32,
    /// 调用其他函数所需的最大传出参数区。
    pub arg_size: u32,
    /// 被调用者保存寄存器占用的空间
    pub callee_saved_size: u32,
    /// 当前函数实际使用到、需要保存恢复的 callee-saved 物理寄存器
    pub used_callee_saved: Vec<Reg>,
    /// 对齐后的总栈大小
    pub total_size: u32,
    /// 已分配的栈槽
    pub slots: cranelift_entity::PrimaryMap<StackSlot, StackSlotData>,
}

impl StackFrame {
    /// Allocate frame-relative storage without modifying instructions.
    pub fn alloc_slot(&mut self, size: u32, align: u32) -> StackSlot {
        assert!(align.is_power_of_two(), "invalid stack alignment");
        let end = self
            .local_size
            .checked_add(size)
            .expect("stack frame overflow");
        let end = end.checked_add(align - 1).expect("stack frame overflow") & !(align - 1);
        let offset = -i32::try_from(end).expect("stack frame exceeds signed offsets");
        self.local_size = end;
        self.slots.push(StackSlotData {
            base: StackBase::Frame,
            size,
            align,
            offset,
        })
    }
}

/// 机器函数主体数据。
#[derive(Debug, Clone)]
pub struct MachineFunction {
    pub name: String,
    body: FuncBody,
    pub stack_frame: StackFrame,
    /// 函数参数对应的虚拟寄存器
    pub params: Vec<Reg>,
}

mod edit;
pub use edit::{EditChanges, FuncEditor};

impl MachineFunction {
    pub fn new(name: String) -> Self {
        Self {
            name,
            body: FuncBody::default(),
            stack_frame: StackFrame {
                local_size: 0,
                arg_size: 0,
                callee_saved_size: 0,
                used_callee_saved: Vec::new(),
                total_size: 0,
                slots: PrimaryMap::new(),
            },
            params: Vec::new(),
        }
    }

    pub fn body(&self) -> &FuncBody {
        &self.body
    }
    pub fn layout(&self) -> &crate::layout::Layout {
        self.body.layout()
    }
    pub fn vregs(&self) -> &PrimaryMap<VReg, VRegData> {
        self.body.vregs()
    }
    pub fn block_insts(&self, block: Block) -> impl DoubleEndedIterator<Item = InstId> + '_ {
        self.body.layout.block_insts(block)
    }
    pub fn num_blocks(&self) -> usize {
        self.body.layout.len()
    }
    pub fn block_params(&self, block: Block) -> Option<&[Reg]> {
        self.body.block_params(block)
    }
    pub fn blocks(&self) -> impl DoubleEndedIterator<Item = Block> + '_ {
        self.body.layout.block_order()
    }
    pub fn entry_block(&self) -> Option<Block> {
        self.body.entry_block()
    }
    pub fn inst_block(&self, inst: InstId) -> Option<Block> {
        self.body.layout.inst_block(inst)
    }

    pub fn inst(&self, id: InstId) -> InstRef<'_> {
        self.body.inst(id)
    }

    pub fn operand(&self, id: crate::OperandId) -> Reg {
        self.body.store.operand(id)
    }
    pub fn input_id(&self, inst: InstId, index: usize) -> crate::OperandId {
        self.body.store.input_id(inst, index)
    }
    pub fn result_id(&self, inst: InstId, index: usize) -> crate::OperandId {
        self.body.store.result_id(inst, index)
    }

    pub fn inst_count(&self) -> usize {
        self.body.store.len()
    }

    pub fn uses(&self, reg: Reg) -> crate::RegRefs<'_> {
        self.body.store.uses(reg)
    }
    pub fn defs(&self, reg: Reg) -> crate::RegRefs<'_> {
        self.body.store.defs(reg)
    }
    pub fn check_refs(&self) -> Result<(), &'static str> {
        self.body.store.check_refs()
    }

    /// 获取虚拟寄存器数据
    pub fn vreg_data(&self, reg: Reg) -> &VRegData {
        debug_assert!(reg.is_vreg());
        &self.body.vregs[VReg::from_u32(reg.index())]
    }

    /// 获取指令的额外 payload。
    pub fn inst_extra(&self, inst_id: InstId) -> Option<crate::InstExtraRef<'_>> {
        self.body.store.extra(inst_id)
    }

    /// 获取调用指令的签名信息。
    pub fn call_info(&self, inst_id: InstId) -> &CallInfo {
        match self.inst_extra(inst_id) {
            Some(crate::InstExtraRef::Call(info)) => info,
            Some(_) => panic!(
                "instruction {:?} in `{}` does not carry call info payload",
                inst_id, self.name
            ),
            None => panic!(
                "call instruction {:?} in `{}` is missing call info payload",
                inst_id, self.name
            ),
        }
    }

    /// 生成便于调试的文本格式 LIR。
    pub fn format_for_dump(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "function {}", self.name);

        if !self.params.is_empty() {
            let params = self
                .params
                .iter()
                .map(|reg| format!("{:?}:{}", reg, self.vreg_data(*reg).ty))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "  params: {}", params);
        }

        for block in self.blocks() {
            let _ = writeln!(out, "  block {:?}:", block);
            if !self.block_params(block).unwrap().is_empty() {
                let params = self
                    .block_params(block)
                    .unwrap()
                    .iter()
                    .map(|reg| format!("{:?}:{}", reg, self.vreg_data(*reg).ty))
                    .collect::<Vec<_>>()
                    .join(", ");
                let _ = writeln!(out, "    params: {}", params);
            }
            for inst_id in self.block_insts(block) {
                let inst = self.inst(inst_id);
                let _ = write!(out, "    {:?}: {:?}", inst_id, inst);
                if let Some(extra) = self.inst_extra(inst_id) {
                    let _ = write!(out, " extra={:?}", extra);
                }
                let _ = writeln!(out);
            }
        }

        out
    }
}
