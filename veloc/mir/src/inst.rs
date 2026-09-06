use super::dfg::DataFlowGraph;
use crate::opspec::{MemoryEffect, OpFormat};
use crate::types::{BlockCall, FuncId, JumpTable, StackSlot, Value, ValueList};
use crate::{FloatCC, IntCC, Intrinsic, MemFlags, Opcode, SigId};
use core::fmt;
use cranelift_entity::entity_impl;

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
pub struct CallInfo {
    pub signature: SignatureRef,
    pub args: ValueList,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PtrIndexImmId(pub u32);
entity_impl!(PtrIndexImmId, "ptr_index_imm");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

// =============================================================================
// Vector Extension IDs (用于指向辅助数据池)
// =============================================================================

/// 向量操作扩展信息 ID (指向 DFG.vector_ext_pool)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorExtId(pub u32);
entity_impl!(VectorExtId, "vext");

/// 常量池 ID (用于 Shuffle 掩码等)
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantPoolId(pub u32);
entity_impl!(ConstantPoolId, "const");

// =============================================================================
// 向量操作辅助数据结构 (存储在 DFG 的 Arena 中)
// =============================================================================

/// 扩展配置 ID
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorMemExtId(pub u32);
entity_impl!(VectorMemExtId, "vmem_ext");

impl Inst {
    pub fn visit_operands<F>(self, dfg: &DataFlowGraph, f: F)
    where
        F: FnMut(Value),
    {
        dfg.instructions[self].visit_operands(dfg, f)
    }
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

impl InstructionData {
    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    pub fn memory_effect(&self, dfg: &DataFlowGraph) -> MemoryEffect {
        let effect = self.opcode().spec().memory_effect;
        let flags = self.memory_flags(dfg);
        if flags.is_some_and(|flags| flags.is_volatile()) {
            effect.with_volatile()
        } else {
            effect
        }
    }

    pub fn has_side_effects(&self, dfg: &DataFlowGraph) -> bool {
        let spec = self.opcode().spec();
        spec.is_terminator() || spec.may_trap() || self.memory_effect(dfg).has_side_effects()
    }
}

impl fmt::Display for InstructionData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, CallConv, Linkage, ModuleBuilder, Type};

    #[test]
    fn call_metadata_preserves_the_signature_source_and_argument_list() {
        let mut module = ModuleBuilder::new();
        let signature = module.make_signature(
            alloc::vec![Type::I32],
            alloc::vec![Type::I32],
            CallConv::SystemV,
        );
        let func = module.declare_function("callee".into(), signature, Linkage::Import);
        let module = module.build_data();
        let mut dfg = DataFlowGraph::new();
        let args = dfg.make_value_list(&[Value(7)]);
        for (data, expected) in [
            (
                InstructionData::Call {
                    func_id: func,
                    args,
                },
                SignatureRef::Function(func),
            ),
            (
                InstructionData::CallIndirect {
                    ptr: Value(9),
                    args,
                    sig_id: signature,
                },
                SignatureRef::Signature(signature),
            ),
            (
                InstructionData::CallIntrinsic {
                    intrinsic: crate::intrinsic_ids::SIN_F32,
                    args,
                    sig_id: signature,
                },
                SignatureRef::Signature(signature),
            ),
        ] {
            let call = data.call_info().unwrap();
            assert_eq!(call.signature, expected);
            assert_eq!(call.signature.resolve(&module), Some(signature));
            assert_eq!(dfg.get_value_list(call.args), &[Value(7)]);
        }
        assert!(
            InstructionData::Return { values: args }
                .call_info()
                .is_none()
        );
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
    fn successor_visitation_preserves_duplicates_arguments_and_default_order() {
        let mut dfg = DataFlowGraph::new();
        let first = dfg.make_block_call(Block(1), &[Value(1)]);
        let second = dfg.make_block_call(Block(2), &[Value(2), Value(3)]);
        let default = dfg.make_block_call(Block(3), &[]);
        let table = dfg.make_jump_table(&[first, second, first], default);
        let data = InstructionData::BrTable {
            index: Value(0),
            table,
        };
        let mut calls = alloc::vec::Vec::new();
        data.visit_successors(&dfg, |call| calls.push(call));
        assert_eq!(calls, [first, second, first, default]);
        assert_eq!(dfg.block_call_args(calls[1]), &[Value(2), Value(3)]);
        let table = dfg.make_jump_table(&[], default);
        assert_eq!(dfg.jump_table_targets(table), &[default]);

        calls.clear();
        InstructionData::Br {
            condition: Value(0),
            then_dest: first,
            else_dest: second,
        }
        .visit_successors(&dfg, |call| calls.push(call));
        assert_eq!(calls, [first, second]);
        calls.clear();
        InstructionData::Iconst { value: 0 }.visit_successors(&dfg, |call| calls.push(call));
        assert!(calls.is_empty());
    }

    #[test]
    fn raw_builder_uses_declared_successors_to_update_the_cfg() {
        let mut module = ModuleBuilder::new();
        let sig = module.make_signature(alloc::vec![], alloc::vec![], CallConv::SystemV);
        let func = module.declare_function("edges".into(), sig, Linkage::Local);
        let mut builder = module.builder(func);
        let entry = builder.init_entry_block();
        let target = builder.create_block();
        let dest = builder.make_block_call(target, &[]);
        builder.ins().push_raw(InstructionData::Jump { dest });
        assert_eq!(builder.func().layout.blocks[entry].succs, [target]);
        assert_eq!(builder.func().layout.blocks[target].preds, [entry]);
    }
}
