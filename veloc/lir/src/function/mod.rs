//! Function storage, layout, register references and structural editing.

mod cursor;
mod edit;
mod frame;
mod layout;
mod store;
mod use_def;

pub use cursor::InstCursor;
pub use edit::{EditChanges, FuncEditor, InstInserter, InstWriter};
pub use frame::*;
pub use layout::{InstOrder, Layout};
pub use store::InstStore;
pub use use_def::{OperandId, RefRole, RegRef, RegRefs};

use self::BlockId as Block;
use crate::{CallInfo, InstId, InstRef, Reg, RegisterBank, VReg, VRegData};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use cranelift_entity::{PrimaryMap, entity_impl};
use veloc_mir::Type;

/// Function-local block identity, independent of MIR and physical order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(u32);
entity_impl!(BlockId, "block");

#[derive(Debug, Clone, Default)]
struct BlockData {
    params: Vec<Reg>,
}

/// A fixed ABI location for a value defined by an entry block parameter.
/// This constrains the incoming definition, not the value's whole live range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntryBinding {
    pub value: Reg,
    pub location: crate::PReg,
}

/// All function-local identities and their editable body.
#[derive(Debug, Clone)]
pub struct FuncBody {
    blocks: PrimaryMap<Block, BlockData>,
    /// Dedicated call entry, retained for the lifetime of the function.
    entry: Block,
    layout: Layout,
    store: InstStore,
    vregs: PrimaryMap<VReg, VRegData>,
    entry_bindings: Vec<EntryBinding>,
}

impl FuncBody {
    fn with_capacity(blocks: usize, insts: usize, vregs: usize) -> Self {
        let mut data = PrimaryMap::with_capacity(blocks.max(1));
        let entry = data.push(BlockData::default());
        let mut layout = Layout::with_capacity(blocks.max(1), insts);
        layout.append_block(entry);
        Self {
            blocks: data,
            entry,
            layout,
            store: InstStore::with_capacity(insts),
            vregs: PrimaryMap::with_capacity(vregs),
            entry_bindings: Vec::new(),
        }
    }

    pub fn entry_block(&self) -> Block {
        self.entry
    }
    pub fn layout(&self) -> &Layout {
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

/// 机器函数主体数据。
#[derive(Debug, Clone)]
pub struct MachineFunction {
    pub name: String,
    body: FuncBody,
    pub stack_frame: StackFrame,
}

impl MachineFunction {
    pub fn new(name: String) -> Self {
        Self::with_capacity(name, 0, 0, 0)
    }

    /// Create a function with an entry block, reserving its known source-level shape.
    /// Later legalization and selection may still append stable identities.
    pub fn with_capacity(name: String, blocks: usize, insts: usize, vregs: usize) -> Self {
        Self {
            name,
            body: FuncBody::with_capacity(blocks, insts, vregs),
            stack_frame: StackFrame::default(),
        }
    }

    pub fn body(&self) -> &FuncBody {
        &self.body
    }
    pub fn entry_bindings(&self) -> &[EntryBinding] {
        &self.body.entry_bindings
    }
    /// Entry block parameters: all inputs before ABI lowering, register inputs
    /// afterward. The source signature remains the authority for call arguments.
    pub fn params(&self) -> &[Reg] {
        &self.body.blocks[self.entry_block()].params
    }
    pub fn layout(&self) -> &Layout {
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
    pub fn entry_block(&self) -> Block {
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
    /// Instruction definitions only. Block parameters (including entry inputs)
    /// have no defining operand slot; they are exposed by `block_params`.
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

    pub fn try_call_info(&self, inst_id: InstId) -> Option<&CallInfo> {
        self.body.store.call_info(inst_id)
    }
    pub fn successors(&self, inst: InstId) -> impl Iterator<Item = crate::Successor<&[Reg]>> {
        self.body.store.successors(inst)
    }

    /// 获取调用指令的签名信息。
    pub fn call_info(&self, inst_id: InstId) -> &CallInfo {
        self.try_call_info(inst_id)
            .expect("instruction has no call information")
    }

    /// 生成便于调试的文本格式 LIR。
    pub fn format_for_dump(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "function {}", self.name);

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
            if block == self.entry_block() && !self.entry_bindings().is_empty() {
                let _ = writeln!(out, "    incoming: {:?}", self.entry_bindings());
            }
            for inst_id in self.block_insts(block) {
                let inst = self.inst(inst_id);
                let _ = write!(out, "    {:?}: {:?}", inst_id, inst);
                if let Some(info) = self.try_call_info(inst_id) {
                    let _ = write!(out, " call={:?}", info);
                }
                if !inst.constraints().is_empty() {
                    let _ = write!(out, " constraints={:?}", inst.constraints());
                }
                let _ = writeln!(out);
            }
        }

        out
    }
}
