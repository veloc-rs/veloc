use crate::error::{Error, Result};
use smallvec::SmallVec;
use veloc_lir::{GenericOpcode, InstField, InstId, InstRef, MachineFunction, MemoryAccess};
use veloc_mir::Type;

/// Instruction-local facts, also constructible for prospective instructions.
/// No graph access or mutation is available to the legality query.
#[derive(Debug, Clone)]
pub struct Query {
    pub opcode: GenericOpcode,
    pub results: SmallVec<[Type; 2]>,
    pub inputs: SmallVec<[Type; 3]>,
    pub fields: SmallVec<[InstField; 2]>,
    pub memory: Option<MemoryAccess>,
}

impl Query {
    pub fn from_inst(inst: &InstRef<'_>, f: &MachineFunction) -> Result<Self> {
        let ty = |r: &veloc_lir::Reg| {
            r.as_vreg()
                .map(|v| f.vregs()[v].ty)
                .ok_or_else(|| Error::codegen("legalization requires typed virtual operands"))
        };
        Ok(Self {
            opcode: inst
                .generic_opcode()
                .ok_or_else(|| Error::codegen("expected generic instruction"))?,
            results: inst.results().iter().map(ty).collect::<Result<_>>()?,
            inputs: inst.inputs().iter().map(ty).collect::<Result<_>>()?,
            fields: inst.fields().iter().cloned().collect(),
            memory: inst.memory(),
        })
    }
}

pub mod contracts {
    include!(concat!(env!("OUT_DIR"), "/legalize_contract.rs"));
}

impl contracts::Query for Query {
    fn value_type(&self, result: bool, index: u32) -> Type {
        if result {
            self.results[index as usize]
        } else {
            self.inputs[index as usize]
        }
    }

    fn signature(&self, results: &[&[Type]], inputs: &[&[Type]]) -> bool {
        fn matches(actual: &[Type], sets: &[&[Type]]) -> bool {
            actual.len() == sets.len() && actual.iter().zip(sets).all(|(ty, set)| set.contains(ty))
        }
        matches(&self.results, results) && matches(&self.inputs, inputs)
    }

    fn same(&self, indices: &[u32]) -> bool {
        let ty = |i: u32| self.results.iter().chain(&self.inputs).nth(i as usize);
        indices.split_first().is_none_or(|(&first, rest)| {
            ty(first).is_some_and(|first| rest.iter().all(|&i| ty(i) == Some(first)))
        })
    }

    fn input_is(&self, index: u32, ty: Type) -> bool {
        self.inputs.get(index as usize) == Some(&ty)
    }

    fn signed_offset(&self, bits: u32) -> bool {
        self.fields
            .iter()
            .find_map(|field| match field {
                InstField::Imm(offset) => Some(*offset),
                _ => None,
            })
            .is_some_and(|offset| {
                bits != 0
                    && bits <= 64
                    && (bits == 64
                        || (offset >= -(1i64 << (bits - 1)) && offset < (1i64 << (bits - 1))))
            })
    }
}

/// The selected implementation, not a second opcode dispatch. Function pointers
/// avoid a per-rewrite closure allocation and keep target selection out of the driver.
#[derive(Clone, Copy)]
pub struct Rewrite {
    pub name: &'static str,
    pub apply: fn(&mut RewriteContext<'_>) -> Result<()>,
}
impl core::fmt::Debug for Rewrite {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LegalizeAction {
    Legal,
    Rewrite(Rewrite),
}

impl LegalizeAction {
    pub fn rewrite(name: &'static str, apply: fn(&mut RewriteContext<'_>) -> Result<()>) -> Self {
        Self::Rewrite(Rewrite { name, apply })
    }
}

impl Rewrite {
    pub(super) fn apply(self, id: InstId, f: &mut veloc_lir::FuncEditor<'_>) -> Result<()> {
        (self.apply)(&mut RewriteContext {
            root: id,
            function: f.editor(),
        })
    }
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
impl RewriteContext<'_> {
    /// Snapshot the matched values/types, then build with explicit arguments.
    /// Construction may reuse the destination; existing-value results use RAUW.
    /// Edits are immediate and are not rolled back on failure.
    pub fn replace_values(
        &mut self,
        build: impl FnOnce(&mut Self, &[veloc_lir::Reg], &[Type], veloc_lir::Reg) -> veloc_lir::Reg,
    ) -> Result<()> {
        let root = self.function.inst(self.root);
        assert_eq!(root.results().len(), 1, "value rewrite requires one result");
        let destination = root.results()[0];
        let inputs: SmallVec<[veloc_lir::Reg; 3]> = root.inputs().iter().copied().collect();
        let types: SmallVec<[Type; 4]> = core::iter::once(destination)
            .chain(inputs.iter().copied())
            .map(|reg| self.function.vreg_data(reg).ty)
            .collect();
        let value = build(self, &inputs, &types, destination);
        if value != destination {
            assert_eq!(
                self.function.vreg_data(destination).ty,
                self.function.vreg_data(value).ty,
                "replacement result type"
            );
            replace_uses(&mut self.function, destination, value);
        }
        self.function.invalidate_inst(self.root);
        Ok(())
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
    pub fn replace(&mut self, output: &[InstId]) {
        self.function.editor().replace_with(self.root, output);
    }
}

fn replace_uses(editor: &mut veloc_lir::FuncEditor<'_>, old: veloc_lir::Reg, new: veloc_lir::Reg) {
    editor.replace_uses(
        old.as_vreg().expect("SSA result must be virtual"),
        new.as_vreg().expect("SSA replacement must be virtual"),
    );
}

pub use contracts::ValueRewrite;
impl ValueRewrite for RewriteContext<'_> {
    fn emit(
        &mut self,
        opcode: GenericOpcode,
        ty: Type,
        inputs: &[veloc_lir::Reg],
        fields: &[InstField],
        result: Option<veloc_lir::Reg>,
    ) -> veloc_lir::Reg {
        let dst = match result {
            Some(value) => value,
            None => self.function.alloc_vreg(ty),
        };
        let inst = self.function.writer().write(
            veloc_lir::MachineOpcode::Generic(opcode),
            &[dst],
            inputs,
            fields,
        );
        self.function.insert_before(self.root, inst);
        dst
    }
}
