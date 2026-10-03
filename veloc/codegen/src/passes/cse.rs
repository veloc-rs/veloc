//! Common pure generic values introduced by legalization, before selection.
use crate::pipeline::{FunctionPass, FunctionSession, FunctionStage};
use std::collections::HashMap;
use veloc_lir::{FieldValueRef, GenericOpcode, InstId, Reg, Type};
use veloc_types::{FloatCC, IntCC};

#[derive(PartialEq, Eq, Hash)]
enum Attribute {
    Integer(i64),
    FloatBits(u64),
    IntCondition(IntCC),
    FloatCondition(FloatCC),
}

#[derive(PartialEq, Eq, Hash)]
struct Key {
    opcode: GenericOpcode,
    inputs: Vec<Reg>,
    types: Vec<Type>,
    fields: Vec<Attribute>,
}

pub struct CommonValues;

impl FunctionPass for CommonValues {
    fn name(&self) -> &'static str {
        "common-legalized-values"
    }
    fn input_stage(&self) -> FunctionStage {
        FunctionStage::Legal
    }
    fn run(&self, cx: &mut FunctionSession<'_>) -> crate::Result<()> {
        let entry = cx.function().entry_block();
        let blocks = cx.cfg().compute_post_order(entry);
        let dom = cx.dominators().clone();
        let mut table = HashMap::<Key, Vec<InstId>>::new();
        let mut f = cx.edit();
        for block in blocks.into_iter().rev() {
            let insts: Vec<_> = f.block_insts(block).collect();
            for id in insts {
                let inst = f.inst(id);
                if !inst.is_pure_value() || !inst.constraints().is_empty() {
                    continue;
                }
                // Only scalar attributes participate. Stack ownership, edges,
                // calls and memory payloads need their own equivalence rules.
                let fields: Option<Vec<_>> = (0..inst.fields().len())
                    .map(|i| match inst.fields().read(i) {
                        FieldValueRef::Imm(v) => Some(Attribute::Integer(*v)),
                        FieldValueRef::FImm(v) => Some(Attribute::FloatBits(v.to_bits())),
                        FieldValueRef::IntCC(v) => Some(Attribute::IntCondition(*v)),
                        FieldValueRef::FloatCC(v) => Some(Attribute::FloatCondition(*v)),
                        _ => None,
                    })
                    .collect();
                let Some(fields) = fields else { continue };
                let key = Key {
                    opcode: inst.generic_opcode().unwrap(),
                    inputs: inst.inputs().to_vec(),
                    types: inst.results().iter().map(|&r| f.vreg_data(r).ty).collect(),
                    fields,
                };
                let candidates = table.entry(key).or_default();
                if let Some(&old) = candidates
                    .iter()
                    .find(|&&i| dom.dominates(f.inst_block(i).unwrap(), block))
                {
                    let replacements: Vec<_> = inst
                        .results()
                        .iter()
                        .copied()
                        .zip(f.inst(old).results().iter().copied())
                        .collect();
                    for (old, new) in replacements {
                        f.editor()
                            .replace_uses(old.as_vreg().unwrap(), new.as_vreg().unwrap());
                    }
                    f.editor().invalidate_inst(id);
                } else {
                    candidates.push(id);
                }
            }
        }
        Ok(())
    }
}
