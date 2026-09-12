//! Instruction kinds, metadata, direct writers and borrowed storage views.

use crate::dfg::DataFlowGraph;
use crate::types::Value;
use core::fmt;
use cranelift_entity::entity_impl;

mod opcode;
pub use opcode::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Inst(pub u32);
entity_impl!(Inst, "inst");

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConstantPoolId(pub u32);
entity_impl!(ConstantPoolId, "const");

mod storage;
pub use storage::{Arguments, Successor, SuccessorMut, Successors};
pub(crate) use storage::{FieldPool, StoredInst};

/// A single-use write into one DFG. Generated methods encode directly into
/// persistent fields; no owning instruction draft is materialized.
pub struct InstWriter<'a> {
    pub(crate) dfg: &'a mut DataFlowGraph,
    pub(crate) target: Option<Inst>,
}
impl InstWriter<'_> {
    /// Copy an instruction in this DFG, retaining its logical inputs but owning new pooled data.
    pub fn copy(self, inst: Inst) -> Inst {
        let fields = self.dfg.instructions[inst].fields.clone();
        let fields = fields.clone_in(&mut self.dfg.fields);
        let values = self.dfg.inst(inst).operands_owned();
        self.write(fields, &values)
    }

    fn write(self, fields: InstFields, values: &[Value]) -> Inst {
        self.dfg.write_inst(self.target, fields, values)
    }
}

include!(concat!(env!("OUT_DIR"), "/instructions.rs"));

impl InstView<'_> {
    pub(crate) fn operands_owned(&self) -> Arguments {
        let mut values = Arguments::new();
        self.visit_operands(|value| values.push(value));
        values
    }

    pub fn is_terminator(&self) -> bool {
        self.opcode().spec().is_terminator()
    }

    /// Coarse behavior only; per-access volatility is checked separately.
    pub fn memory_effect(&self) -> MemoryEffect {
        self.opcode().spec().memory_effect()
    }

    pub fn has_volatile_access(&self) -> bool {
        self.memory_flags().is_some_and(|flags| flags.is_volatile())
    }

    /// Deletion, speculation and commoning have different preconditions.
    pub fn can_erase(&self) -> bool {
        let spec = self.opcode().spec();
        !spec.is_terminator()
            && !spec.may_trap()
            && !self.opcode().transfers_ownership()
            && !self.has_volatile_access()
            && self.memory_effect().can_erase()
    }

    /// Conservative, context-free speculation. Analyses may prove more.
    pub fn can_speculate(&self) -> bool {
        self.opcode().spec().is_pure()
            && !self.opcode().transfers_ownership()
            && !self.has_volatile_access()
            && self.memory_effect().is_none()
    }

    /// Context-free commoning excludes mutable reads and fresh identities.
    pub fn can_cse(&self) -> bool {
        self.can_speculate()
    }

    pub fn has_side_effects(&self) -> bool {
        let spec = self.opcode().spec();
        spec.is_terminator()
            || spec.may_trap()
            || self.has_volatile_access()
            || self.memory_effect().has_side_effects()
    }
}

impl fmt::Display for InstView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.opcode())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Block, BlockCall, CallConv, Linkage, ModuleBuilder, Type};
    use crate::{FuncId, SigId};

    #[test]
    fn call_results_resolve_the_declared_source_without_validating_arguments() {
        let mut module = ModuleBuilder::new();
        let signature = module.make_signature(
            alloc::vec![Type::I32, Type::I64],
            alloc::vec![Type::I32, Type::I64],
            CallConv::SystemV,
        );
        let func = module.declare_function("callee".into(), signature, Linkage::Import);
        let module = module.build_data();
        let mut dfg = DataFlowGraph::new();
        let callee = dfg.values.push(crate::types::ValueData {
            ty: Type::callable(signature, crate::CallableKind::Shared),
            def: crate::ValueDef::Param(Block(0)),
        });
        // The nonexistent argument values are deliberately not validated here.
        for inst in [
            dfg.writer().call(func, &[Value(7)]),
            dfg.writer().call_indirect(Value(9), &[Value(7)], signature),
            dfg.writer()
                .call_intrinsic(crate::intrinsic_ids::SIN_F32, &[Value(7)], signature),
            dfg.writer()
                .call_value(Opcode::CallValue, callee, &[Value(7)]),
        ] {
            assert!(dfg.opcode(inst).has_signature());
            assert_eq!(
                dfg.inst(inst)
                    .result_types(&dfg, &module, &[Type::F32])
                    .unwrap()
                    .as_slice(),
                module.signatures[signature].returns()
            );
        }
        assert!(!Opcode::Return.has_signature());
        assert!(!Opcode::TailCall.has_signature());
        let scalar = dfg.values.push(crate::types::ValueData {
            ty: Type::I32,
            def: crate::ValueDef::Param(Block(0)),
        });
        let unknown = dfg.values.push(crate::types::ValueData {
            ty: Type::callable(SigId(u32::MAX), crate::CallableKind::Shared),
            def: crate::ValueDef::Param(Block(0)),
        });
        for inst in [
            dfg.writer().call(FuncId(u32::MAX), &[]),
            dfg.writer().call_indirect(Value(9), &[], SigId(u32::MAX)),
            dfg.writer().call_value(Opcode::CallValue, Value(99), &[]),
            dfg.writer().call_value(Opcode::CallValue, scalar, &[]),
            dfg.writer().call_value(Opcode::CallValue, unknown, &[]),
        ] {
            assert!(dfg.inst(inst).result_types(&dfg, &module, &[]).is_err());
        }
    }

    #[test]
    fn successor_views_preserve_occurrences_and_default_order() {
        let mut dfg = DataFlowGraph::new();
        let first = BlockCall::new(Block(1), &[Value(1)]);
        let second = BlockCall::new(Block(2), &[Value(2), Value(3)]);
        let default = BlockCall::new(Block(3), &[]);
        let inst = dfg.create_inst(|writer| {
            writer.br_table(
                Value(0),
                [first.clone(), second.clone(), first, default]
                    .iter()
                    .map(BlockCall::as_view),
            )
        });
        let InstView::BrTable { table, .. } = dfg.inst(inst) else {
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
