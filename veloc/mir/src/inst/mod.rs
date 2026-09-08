//! Instruction kinds, metadata, drafts and borrowed storage views.

use crate::dfg::DataFlowGraph;
use crate::types::{FuncId, StackSlot, Value};
use crate::{Intrinsic, SigId};
use core::fmt;
use cranelift_entity::entity_impl;

mod opcode;
pub use opcode::*;

/// The declared source of a call's argument/result signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureRef {
    Function(FuncId),
    Signature(SigId),
}

impl SignatureRef {
    pub fn resolve(self, module: &crate::ModuleData) -> Option<SigId> {
        let sig = match self {
            Self::Function(func) => module.functions.get(func)?.signature,
            Self::Signature(sig) => sig,
        };
        module.signatures.get(sig)?;
        Some(sig)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CallInfo<'a> {
    pub signature: SignatureRef,
    pub args: &'a [Value],
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantPoolId(pub u32);
entity_impl!(ConstantPoolId, "const");

mod storage;
pub(crate) use storage::StoredInst;
pub use storage::{Arguments, Successor, Successors};

/// Owned instruction draft using the same fields and operand order as the DFG.
/// Constructors guarantee storage shape, not the instruction's type contract.
#[derive(Debug, Clone)]
pub struct InstDraft {
    pub(crate) fields: InstFields,
    pub(crate) operands: Arguments,
}

impl InstDraft {
    pub fn as_view(&self) -> InstructionView<'_> {
        self.fields.view(&self.operands)
    }

    pub fn opcode(&self) -> Opcode {
        self.fields.opcode()
    }

    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    pub fn result_types(
        &self,
        dfg: &DataFlowGraph,
        module: &crate::ModuleData,
        explicit: &[crate::Type],
    ) -> Result<smallvec::SmallVec<[crate::Type; 2]>, &'static str> {
        self.as_view().result_types(dfg, module, explicit)
    }

    pub fn operands(&self) -> &[Value] {
        &self.operands
    }

    pub fn set_operand(&mut self, index: usize, value: Value) {
        self.operands[index] = value;
    }
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

impl InstructionView<'_> {
    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    pub fn memory_effect(&self) -> MemoryEffect {
        let effect = self.opcode().spec().memory_effect;
        let flags = self.memory_flags();
        if flags.is_some_and(|flags| flags.is_volatile()) {
            effect.with_volatile()
        } else {
            effect
        }
    }

    pub fn has_side_effects(&self) -> bool {
        let spec = self.opcode().spec();
        spec.is_terminator() || spec.may_trap() || self.memory_effect().has_side_effects()
    }
}

impl fmt::Display for InstructionView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, BlockCall, CallConv, Linkage, ModuleBuilder, Type};

    #[test]
    fn call_metadata_preserves_signature_and_borrows_installed_arguments() {
        let mut module = ModuleBuilder::new();
        let signature = module.make_signature(
            alloc::vec![Type::I32],
            alloc::vec![Type::I32],
            CallConv::SystemV,
        );
        let func = module.declare_function("callee".into(), signature, Linkage::Import);
        let module = module.build_data();
        let mut dfg = DataFlowGraph::new();
        for (data, expected) in [
            (
                InstDraft::call(func, &[Value(7)]),
                SignatureRef::Function(func),
            ),
            (
                InstDraft::call_indirect(Value(9), &[Value(7)], signature),
                SignatureRef::Signature(signature),
            ),
            (
                InstDraft::call_intrinsic(crate::intrinsic_ids::SIN_F32, &[Value(7)], signature),
                SignatureRef::Signature(signature),
            ),
        ] {
            let inst = dfg.create_inst(data);
            let call = dfg.inst(inst).call_info().unwrap();
            assert_eq!(call.signature, expected);
            assert_eq!(call.signature.resolve(&module), Some(signature));
            assert_eq!(call.args, &[Value(7)]);
            assert_eq!(
                call.args.as_ptr(),
                dfg.operands(inst)[dfg.operands(inst).len() - 1..].as_ptr()
            );
        }
        assert!(InstDraft::ret(&[]).as_view().call_info().is_none());
        assert_eq!(
            SignatureRef::Function(FuncId(u32::MAX)).resolve(&module),
            None
        );
        assert_eq!(
            SignatureRef::Signature(SigId(u32::MAX)).resolve(&module),
            None
        );
    }

    #[test]
    fn successor_views_preserve_occurrences_and_default_order() {
        let mut dfg = DataFlowGraph::new();
        let first = BlockCall::new(Block(1), &[Value(1)]);
        let second = BlockCall::new(Block(2), &[Value(2), Value(3)]);
        let default = BlockCall::new(Block(3), &[]);
        let inst = dfg.create_inst(InstDraft::br_table(
            Value(0),
            [first.clone(), second.clone(), first, default]
                .iter()
                .map(BlockCall::as_view),
        ));
        let InstructionView::BrTable { table, .. } = dfg.inst(inst) else {
            unreachable!()
        };
        assert_eq!(
            table
                .iter()
                .map(|c| c.block)
                .collect::<alloc::vec::Vec<_>>(),
            [Block(1), Block(2), Block(1), Block(3)]
        );
        assert_eq!(table.iter().nth(1).unwrap().args, &[Value(2), Value(3)]);
        let (default, cases) = table.split_last().unwrap();
        assert_eq!(default.block, Block(3));
        assert_eq!(cases.len(), 3);
        assert_eq!(
            dfg.operands(inst),
            &[Value(0), Value(1), Value(2), Value(3), Value(1)]
        );
        dfg.check_uses().unwrap();
    }
}
