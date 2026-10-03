use crate::target::FeatureSetRef;
use smallvec::SmallVec;
use veloc_lir::{FieldValue, GenericOpcode, InstId, MachineFunction, Reg};
use veloc_mir::Type;

/// Immutable target policy. The shared runtime owns matching and execution.
#[derive(Clone, Copy)]
pub struct LegalizePolicy<'a> {
    pub program: &'static super::vm::Program,
    pub features: FeatureSetRef<'a>,
}

/// A rewrite can read the function and edit through its invariant-preserving
/// editor. It cannot replace the function or bypass scoped change tracking.
/// Edits commit immediately; this is not a rollback transaction.
pub struct RewriteContext<'a> {
    root: InstId,
    function: veloc_lir::FuncEditor<'a>,
}
impl core::ops::Deref for RewriteContext<'_> {
    type Target = MachineFunction;
    fn deref(&self) -> &Self::Target {
        &self.function
    }
}
impl<'a> RewriteContext<'a> {
    pub(super) fn new(root: InstId, function: veloc_lir::FuncEditor<'a>) -> Self {
        Self { root, function }
    }

    pub(super) fn finish_value(&mut self, value: Reg) {
        let results = self.function.inst(self.root).results();
        assert_eq!(results.len(), 1, "value rewrite requires one result");
        let destination = results[0];
        if value != destination {
            assert_eq!(
                self.function.vreg_data(destination).ty,
                self.function.vreg_data(value).ty,
                "replacement result type"
            );
            replace_uses(&mut self.function, destination, value);
        }
        self.function.invalidate_inst(self.root);
    }

    /// Rebuild in place: result identities, access attributes and physical
    /// effects belong to the original instruction, not to address temporaries.
    pub(super) fn update(&mut self, changes: &[(usize, Reg)], attributes: &[(usize, FieldValue)]) {
        let inst = self.function.inst(self.root);
        let opcode = inst.opcode();
        let results: SmallVec<[Reg; 2]> = SmallVec::from_slice(inst.results());
        let mut inputs: SmallVec<[Reg; 4]> = SmallVec::from_slice(inst.inputs());
        let mut fields: SmallVec<[FieldValue; 2]> = (0..inst.fields().len())
            .map(|i| inst.fields().at(i))
            .collect();
        let clobbers: SmallVec<[Reg; 4]> = inst.clobbers().collect();
        // The recipe compiler checks the complete updated signature, including
        // coordinated operand widening and the unchanged result types.
        for &(index, value) in changes {
            inputs[index] = value;
        }
        for (index, value) in attributes {
            fields[*index] = value.clone();
        }
        let mut writer = self.function.replace(self.root).with_clobbers(clobbers);
        let veloc_lir::MachineOpcode::Generic(generic) = opcode else {
            unreachable!("legalization updates generic instructions");
        };
        let fields = generic.build_fields(&mut writer, fields);
        writer.write(opcode, &results, &inputs, fields);
    }

    /// Replace results with independent, same-typed values and erase the root.
    /// Values must not depend on the root; this is not a wrapping transformation.
    pub fn replace_results(&mut self, values: &[veloc_lir::Reg]) {
        let results: SmallVec<[veloc_lir::Reg; 2]> = self
            .function
            .inst(self.root)
            .results()
            .iter()
            .copied()
            .collect();
        assert_eq!(results.len(), values.len(), "replacement result arity");
        // Check the complete mapping before editing any uses. In particular,
        // mappings among old results would not survive erasing their definition.
        for (&old, &new) in results.iter().zip(values) {
            assert!(
                !results.contains(&new),
                "replacement refers to an erased result"
            );
            assert_eq!(
                self.function.vreg_data(old).ty,
                self.function.vreg_data(new).ty,
                "replacement result type"
            );
        }
        for (&old, &new) in results.iter().zip(values) {
            replace_uses(&mut self.function, old, new);
        }
        self.function.invalidate_inst(self.root);
    }

    pub fn root(&self) -> InstId {
        self.root
    }
    pub fn editor(&mut self) -> veloc_lir::function::FuncEditor<'_> {
        self.function.editor()
    }
}

fn replace_uses(editor: &mut veloc_lir::FuncEditor<'_>, old: veloc_lir::Reg, new: veloc_lir::Reg) {
    editor.replace_uses(
        old.as_vreg().expect("SSA result must be virtual"),
        new.as_vreg().expect("SSA replacement must be virtual"),
    );
}

impl RewriteContext<'_> {
    pub(super) fn emit(
        &mut self,
        opcode: GenericOpcode,
        ty: Type,
        inputs: &[veloc_lir::Reg],
        fields: &[FieldValue],
        result: Option<veloc_lir::Reg>,
    ) -> veloc_lir::Reg {
        let dst = match result {
            Some(value) => value,
            None => self.function.alloc_vreg(ty),
        };
        let mut cursor = self.function.before(self.root);
        let mut writer = cursor.writer();
        let fields = opcode.build_fields(&mut writer, fields.iter().cloned());
        writer.write(
            veloc_lir::MachineOpcode::Generic(opcode),
            &[dst],
            inputs,
            fields,
        );
        dst
    }
}
