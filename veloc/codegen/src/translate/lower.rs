//! MIR/LIR storage adapter for generated, storage-independent value rules.
use super::*;
use veloc_lir::InstBuild;
use veloc_lir::Writable;

include!(concat!(env!("OUT_DIR"), "/mir_lowering.rs"));

struct Lower<'a, 'm> {
    ctx: &'a mut TranslationContext<'m>,
    block: &'a mut MachineBlock,
    args: &'m [Value],
    results: &'m [Value],
}

impl Context for Lower<'_, '_> {
    type Value = Reg;
    type Type = veloc_mir::Type;

    fn input(&self, index: usize) -> Reg {
        self.ctx.value_map[self.args[index]]
    }
    fn result(&self, index: usize) -> Reg {
        self.ctx.value_map[self.results[index]]
    }
    fn value_type(&self, value: Reg) -> Self::Type {
        self.ctx.mfunc.vreg_data(value).ty
    }
    fn temp(&mut self, ty: Self::Type) -> Reg {
        self.ctx.mfunc.alloc_vreg(ty)
    }
    fn emit(&mut self, opcode: GenericOpcode, results: &[Reg], inputs: &[Reg]) {
        let id = build(self.ctx.mfunc.writer(), opcode, results, inputs);
        self.block.append_inst_id(id);
    }
    fn bind(&mut self, result: usize, value: Reg) {
        let dst = self.result(result);
        if dst != value {
            let id = self.ctx.mfunc.writer().copy(Writable(dst), value);
            self.block.append_inst_id(id);
        }
    }
}

pub(super) fn instruction(
    inst: veloc_mir::Inst,
    ctx: &mut TranslationContext<'_>,
    block: &mut MachineBlock,
) -> Result<bool> {
    let view = ctx.func.dfg().inst(inst);
    let args = ctx.func.dfg().operands(inst);
    let results = ctx.func.dfg().inst_results(inst);
    // Preserve the translator's checked entry for arithmetic, independently of
    // whether the selected rule was explicit or inferred from semantics.
    if matches!(
        view,
        InstView::Unary { .. } | InstView::Binary { .. } | InstView::Ternary { .. }
    ) {
        let input_types: smallvec::SmallVec<[_; 3]> =
            args.iter().map(|&v| ctx.func.dfg().value_type(v)).collect();
        let result_types: smallvec::SmallVec<[_; 2]> = results
            .iter()
            .map(|&v| ctx.func.dfg().value_type(v))
            .collect();
        view.opcode()
            .validate_types(&input_types, &result_types)
            .map_err(|error| {
                Error::translate(format!(
                    "invalid types for {} lowering: {error:?}",
                    view.opcode().spec().mnemonic
                ))
            })?;
    }
    Ok(lower(
        view.opcode(),
        &mut Lower {
            ctx,
            block,
            args,
            results,
        },
    ))
}
