//! Scoped, append-only candidate expressions in the function's own DFG.
use super::{FuncBody, FuncDecl};
use crate::{Inst, InstWriter, Module, Type, Value};
use alloc::vec::Vec;
use smallvec::SmallVec;

/// Owns temporary expressions and the exclusive borrow of their function.
/// Existing definitions remain immutable while candidates are being explored.
/// Candidates use ordinary MIR values and use-def links, but have no layout.
/// Dropping the session releases every candidate, including its operand uses.
pub struct Expressions<'a> {
    body: &'a mut FuncBody,
    candidates: Vec<Inst>,
    first_inst: usize,
    first_value: usize,
}

impl FuncBody {
    pub fn expressions(&mut self) -> Expressions<'_> {
        Expressions {
            first_inst: self.dfg.inst_count(),
            first_value: self.dfg.values().len(),
            body: self,
            candidates: Vec::new(),
        }
    }
}

impl<'a> Expressions<'a> {
    pub fn body(&self) -> &FuncBody {
        self.body
    }

    /// Construct a floating expression with the same writers as scheduled MIR.
    /// Speculation safety is a required structural property of candidates;
    /// instruction type/semantic contracts are checked by explicit validation.
    pub fn create(&mut self, build: impl FnOnce(InstWriter<'_>) -> Inst, types: &[Type]) -> Inst {
        assert!(!types.is_empty(), "candidate must produce a value");
        let inst = self.body.dfg.create_inst(build);
        self.candidates.push(inst);
        assert!(
            self.body.dfg.inst(inst).can_speculate(),
            "candidate must be safe to speculate"
        );
        self.body.dfg.append_results(inst, types);
        inst
    }

    /// Validate both the executable body and the candidate definitions.
    pub fn validate(&self, decl: &FuncDecl, module: &Module) -> crate::Result<()> {
        crate::FunctionRef {
            decl,
            body: Some(self.body),
        }
        .validate_expressions(module, &self.candidates)
    }

    /// Freeze the candidate set before choosing executable occurrences.
    pub fn freeze(self) -> FrozenExpressions<'a> {
        FrozenExpressions { expressions: self }
    }
}

/// The candidate definitions are immutable while scheduled copies are emitted.
/// A candidate may have several occurrences in distinct dominance scopes.
pub struct FrozenExpressions<'a> {
    expressions: Expressions<'a>,
}

impl FrozenExpressions<'_> {
    pub fn body(&self) -> &FuncBody {
        self.expressions.body
    }

    /// Materialize a proven constant directly at an executable use.
    pub fn constant(&mut self, before: Inst, value: crate::ScalarConst) -> Value {
        let body = &mut self.expressions.body;
        let inst = body
            .edit()
            .insert_before(before, |w| w.scalar_const(value), &[value.ty()]);
        body.dfg.first_result(inst).expect("constant result")
    }

    pub fn place(&mut self, before: Inst, source: Inst, args: &[Value]) -> Inst {
        let body = &mut self.expressions.body;
        for &value in args {
            if let crate::ValueDef::Inst(def) = body.dfg.value_def(value) {
                assert!(
                    body.layout.inst_block(def).is_some(),
                    "place candidate operands before their consumer"
                );
            }
        }
        assert!(
            body.dfg.inst(source).can_speculate(),
            "cannot duplicate an effectful occurrence"
        );
        let types: SmallVec<[Type; 2]> = body
            .dfg
            .inst_results(source)
            .iter()
            .map(|&v| body.dfg.value_type(v))
            .collect();
        body.edit()
            .insert_before(before, |w| w.copy_with_operands(source, args), &types)
    }

    /// Commit one use to a placed value; never rewrite candidate definitions.
    pub fn replace_input(&mut self, inst: Inst, index: u32, value: Value) {
        let body = &mut self.expressions.body;
        assert!(body.layout.inst_block(inst).is_some(), "use must be placed");
        if let crate::ValueDef::Inst(def) = body.dfg.value_def(value) {
            assert!(
                body.layout.inst_block(def).is_some(),
                "candidate escaped into executable MIR"
            );
        }
        body.edit().set_operand(inst, index, value);
    }
}

impl Drop for Expressions<'_> {
    fn drop(&mut self) {
        let dfg = &mut self.body.dfg;
        dfg.remove_insts(&self.candidates);
        // An abandoned search has no scheduled copies after its candidates.
        // Reclaim that entire suffix rather than leaving empty arena records.
        if dfg.inst_count() == self.first_inst + self.candidates.len() {
            let mut instructions: Vec<_> = core::mem::take(&mut dfg.instructions).into();
            instructions.truncate(self.first_inst);
            dfg.instructions = instructions.into();
            let mut values: Vec<_> = core::mem::take(&mut dfg.values).into();
            values.truncate(self.first_value);
            dfg.values = values.into();
        }
    }
}
