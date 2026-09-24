use crate::error::{Error, Result};
use cranelift_entity::PrimaryMap;
use smallvec::SmallVec;
use veloc_lir::InstRead;
use veloc_lir::{FieldValue, GenericOpcode, InstId, InstRef, MachineFunction, Reg, VReg, VRegData};
use veloc_mir::Type;

/// Borrowed instruction-local facts. Types are read on demand; queries have
/// no access to CFG, users, or mutation. The borrow ends before rewriting.
#[derive(Debug, Clone, Copy)]
pub struct Query<'a> {
    inst: InstRef<'a>,
    vregs: &'a PrimaryMap<VReg, VRegData>,
}

impl<'a> Query<'a> {
    pub fn from_inst(inst: InstRef<'a>, vregs: &'a PrimaryMap<VReg, VRegData>) -> Result<Self> {
        if inst.generic_opcode().is_none() {
            return Err(Error::codegen("expected generic instruction"));
        }
        let regs = || inst.results().iter().chain(inst.inputs());
        if regs().any(|reg| reg.as_vreg().is_some_and(|reg| vregs.get(reg).is_none())) {
            return Err(Error::codegen("unknown virtual operand in legalization"));
        }
        if regs().any(|reg| reg.is_preg()) {
            // Physical locations are permitted only at explicit ABI boundaries.
            // In particular, a register name never supplies a semantic type.
            let valid = match inst.view() {
                veloc_lir::InstView::UnaryReg(copy)
                    if copy.opcode == veloc_lir::UnaryRegOpcode::Copy =>
                {
                    copy.dst.is_vreg() != copy.src.is_vreg()
                }
                veloc_lir::InstView::Call(call) => call
                    .args
                    .iter()
                    .chain(call.results)
                    .all(|reg| reg.is_preg()),
                veloc_lir::InstView::CallIndirect(call) => {
                    call.callee.is_vreg()
                        && call
                            .args
                            .iter()
                            .chain(call.results)
                            .all(|reg| reg.is_preg())
                }
                veloc_lir::InstView::Return(ret) => ret.values.iter().all(|reg| reg.is_preg()),
                _ => false,
            };
            if !valid {
                return Err(Error::codegen(
                    "physical operands require a typed copy or ABI call/return boundary",
                ));
            }
        }
        Ok(Self { inst, vregs })
    }

    pub fn opcode(&self) -> GenericOpcode {
        self.inst
            .generic_opcode()
            .expect("query requires a generic instruction")
    }

    fn ty(&self, reg: Reg) -> Type {
        if let Some(reg) = reg.as_vreg() {
            return self.vregs[reg].ty;
        }
        // A boundary copy's transfer type comes from its SSA endpoint, not
        // from the physical register. Both endpoints therefore match the same
        // ordinary Copy legality rule.
        assert_eq!(
            self.opcode(),
            GenericOpcode::Copy,
            "ABI locations have no standalone value type"
        );
        let value = self
            .inst
            .results()
            .iter()
            .chain(self.inst.inputs())
            .find_map(|reg| reg.as_vreg())
            .expect("typed boundary copy");
        self.vregs[value].ty
    }
}

pub mod contracts {
    include!(concat!(env!("OUT_DIR"), "/legalize_contract.rs"));
}

impl contracts::Query for Query<'_> {
    fn value_type(&self, result: bool, index: u32) -> Type {
        let regs = if result {
            self.inst.results()
        } else {
            self.inst.inputs()
        };
        self.ty(regs[index as usize])
    }

    fn signature(&self, results: &[&[Type]], inputs: &[&[Type]]) -> bool {
        let matches = |regs: &[Reg], sets: &[&[Type]]| {
            regs.len() == sets.len()
                && regs
                    .iter()
                    .zip(sets)
                    .all(|(&reg, set)| set.contains(&self.ty(reg)))
        };
        matches(self.inst.results(), results) && matches(self.inst.inputs(), inputs)
    }

    fn same(&self, indices: &[u32]) -> bool {
        let ty = |i: u32| {
            self.inst
                .results()
                .iter()
                .chain(self.inst.inputs())
                .nth(i as usize)
                .map(|&reg| self.ty(reg))
        };
        indices.split_first().is_none_or(|(&first, rest)| {
            ty(first).is_some_and(|first| rest.iter().all(|&i| ty(i) == Some(first)))
        })
    }

    fn input_is(&self, index: u32, ty: Type) -> bool {
        self.inst
            .inputs()
            .get(index as usize)
            .is_some_and(|&reg| self.ty(reg) == ty)
    }

    fn signed_offset(&self, bits: u32) -> bool {
        let fields = self.inst.fields();
        if fields.is_empty() {
            return false;
        }
        let veloc_lir::FieldValueRef::Imm(&offset) = fields.read(0) else {
            return false;
        };
        bits != 0
            && bits <= 64
            && (bits == 64 || (offset >= -(1i64 << (bits - 1)) && offset < (1i64 << (bits - 1))))
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
        fields: &[FieldValue],
        result: Option<veloc_lir::Reg>,
    ) -> veloc_lir::Reg {
        let dst = match result {
            Some(value) => value,
            None => self.function.alloc_vreg(ty),
        };
        self.function.before(self.root).write(
            veloc_lir::MachineOpcode::Generic(opcode),
            &[dst],
            inputs,
            fields.iter().cloned(),
        );
        dst
    }
}
