//! Schedule symbolic state lifetimes without fixing their order in advance.
//! A candidate may move a whole lifetime across another writer, but never
//! destroy a still-needed resident value. If no such order is found, the caller
//! retains the ordinary schedule and state lowering's existing recovery plan.
use super::Region;
use crate::analysis::{RegSet, state::StateContents};
use hashbrown::HashMap;
use veloc_lir::Reg;

pub(super) struct StateOrder<'a> {
    region: Region<'a>,
    contents: StateContents,
    remaining: HashMap<Reg, usize>,
    exit: StateContents,
    live_out: &'a RegSet,
    pub flexible: RegSet,
}

impl<'a> StateOrder<'a> {
    pub fn new(region: Region<'a>, live_out: &'a RegSet) -> Self {
        let f = region.function;
        let mut remaining = HashMap::new();
        let mut flexible = RegSet::default();
        let mut physical_reads = RegSet::default();
        let mut exit = StateContents::default();
        for i in 0..region.insts.len() {
            let inst = region.inst(i);
            for value in inst.register_access().reads() {
                if let Some(unit) = f.state_unit(value) {
                    *remaining.entry(value).or_default() += 1;
                    flexible.insert(unit);
                } else if value.is_preg() {
                    physical_reads.insert(value);
                }
            }
            for &value in inst.results() {
                if let Some(unit) = f.state_unit(value) {
                    flexible.insert(unit);
                }
            }
            exit.apply(f, inst);
        }
        for value in live_out.iter() {
            if let Some(unit) = f.state_unit(value) {
                // Preserve the source order's outgoing resident version. Other
                // live versions remain the responsibility of state lowering.
                if exit.get(unit) == Some(&value) {
                    *remaining.entry(value).or_default() += 1;
                }
            } else if value.is_preg() {
                physical_reads.insert(value);
            }
        }
        for unit in physical_reads.iter() {
            flexible.remove(&unit);
        }
        Self {
            region,
            contents: StateContents::default(),
            remaining,
            exit,
            live_out,
            flexible,
        }
    }

    pub fn ready(&self, node: usize) -> bool {
        let f = self.region.function;
        let access = self.region.inst(node).register_access();
        if access.reads().any(|value| {
            f.state_unit(value)
                .is_some_and(|unit| self.contents.get(unit) != Some(&value))
        }) {
            return false;
        }
        !access.writes().any(|value| {
            let unit = f.register_unit(value);
            self.contents.get(unit).is_some_and(|resident| {
                *resident != value
                    && self.remaining.get(resident).copied().unwrap_or(0)
                        > usize::from(access.is_read(*resident))
            })
        })
    }

    pub fn advance(&mut self, node: usize) {
        let inst = self.region.inst(node);
        for value in inst.register_access().reads() {
            if let Some(count) = self.remaining.get_mut(&value) {
                *count -= 1;
            }
        }
        self.contents.apply(self.region.function, inst);
    }

    pub fn preserves_exit(&self) -> bool {
        self.live_out.iter().all(|value| {
            let unit = self.region.function.register_unit(value);
            self.contents.get(unit) == self.exit.get(unit)
        })
    }
}
