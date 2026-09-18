use crate::error::{Error, Result};
use alloc::vec::Vec;
use smallvec::SmallVec;
use veloc_lir::{
    GenericOpcode, InstBuild, InstField, InstId, InstRef, MachineFunction, MemoryAccess,
};
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

    pub fn signature(&self, results: &[&[Type]], inputs: &[&[Type]]) -> bool {
        fn matches(actual: &[Type], sets: &[&[Type]]) -> bool {
            actual.len() == sets.len() && actual.iter().zip(sets).all(|(ty, set)| set.contains(ty))
        }
        matches(&self.results, results) && matches(&self.inputs, inputs)
    }

    pub fn same(&self, indices: &[u32]) -> bool {
        let ty = |i: u32| self.results.iter().chain(&self.inputs).nth(i as usize);
        indices
            .first()
            .is_none_or(|&first| ty(first).is_some() && indices.iter().all(|&i| ty(i) == ty(first)))
    }

    pub fn input_is(&self, index: u32, ty: Type) -> bool {
        self.inputs.get(index as usize) == Some(&ty)
    }

    pub fn signed_offset(&self, bits: u32) -> bool {
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
    pub apply: fn(InstId, &mut MachineFunction) -> Result<LegalizeResult>,
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
    Values {
        name: &'static str,
        apply: fn(&mut ValueRewriter<'_>),
    },
}

impl LegalizeAction {
    pub fn values(name: &'static str, apply: fn(&mut ValueRewriter<'_>)) -> Self {
        Self::Values { name, apply }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Legal => "legal",
            Self::Rewrite(r) => r.name,
            Self::Values { name, .. } => name,
        }
    }
    pub fn apply(self, id: InstId, f: &mut MachineFunction) -> Result<LegalizeResult> {
        match self {
            Self::Legal => unreachable!("legal instructions do not have rewrite bodies"),
            Self::Rewrite(r) => (r.apply)(id, f),
            Self::Values { apply, .. } => {
                let mut rewriter = ValueRewriter {
                    root: id,
                    function: f,
                    output: Vec::new(),
                };
                apply(&mut rewriter);
                Ok(LegalizeResult::Replace(rewriter.output))
            }
        }
    }
}

/// Value-only rewrite adapter. Graph edits remain in the instruction store;
/// generated rules cannot mutate unrelated values or silently change types.
pub struct ValueRewriter<'a> {
    root: InstId,
    function: &'a mut MachineFunction,
    output: Vec<InstId>,
}
impl ValueRewriter<'_> {
    pub fn input(&self, index: usize) -> veloc_lir::Reg {
        self.function.inst(self.root).inputs()[index]
    }
    pub fn value_type(&self, result: bool, index: usize) -> Type {
        let inst = self.function.inst(self.root);
        self.function
            .vreg_data(if result {
                inst.results()[index]
            } else {
                inst.inputs()[index]
            })
            .ty
    }
    pub fn emit(
        &mut self,
        opcode: GenericOpcode,
        ty: Type,
        inputs: &[veloc_lir::Reg],
        result: Option<usize>,
    ) -> veloc_lir::Reg {
        let dst = match result {
            Some(i) => self.function.inst(self.root).results()[i],
            None => self.function.editor().alloc_vreg(ty),
        };
        self.output.push(self.function.editor().writer().write(
            veloc_lir::MachineOpcode::Generic(opcode),
            &[dst],
            inputs,
            &[],
        ));
        dst
    }
    pub fn bind(&mut self, result: usize, value: veloc_lir::Reg) {
        let dst = self.function.inst(self.root).results()[result];
        if dst != value {
            self.output.push(
                self.function
                    .editor()
                    .writer()
                    .copy(veloc_lir::Writable(dst), value),
            );
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LegalizeResult {
    Replace(Vec<InstId>),
}
