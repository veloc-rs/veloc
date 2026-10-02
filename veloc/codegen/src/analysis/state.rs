//! Symbolic contents of non-renamable units, shared by scheduling and lowering.
use hashbrown::HashMap;
use veloc_lir::{InstRef, MachineFunction, Reg};

#[derive(Clone, Default, PartialEq, Eq)]
pub(crate) struct StateContents {
    values: HashMap<Reg, Reg>,
}

impl StateContents {
    pub fn get(&self, unit: Reg) -> Option<&Reg> {
        self.values.get(&unit)
    }

    pub fn intersect(&mut self, other: &Self) {
        self.values
            .retain(|unit, value| other.get(*unit) == Some(value));
    }

    /// Read inputs before applying this transition. Every write invalidates the
    /// old contents, including dead results and clobbers. A state result then
    /// installs its identity. Unwritten units retain their previous contents.
    pub fn apply(&mut self, f: &MachineFunction, inst: InstRef<'_>) {
        for value in inst.register_access().writes() {
            self.values.remove(&f.register_unit(value));
        }
        for &value in inst.results() {
            if let Some(unit) = f.state_unit(value) {
                self.values.insert(unit, value);
            }
        }
    }
}
