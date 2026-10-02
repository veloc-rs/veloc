//! Approximate live-register pressure using the allocator's register classes.
use super::{Region, before};
use crate::{
    analysis::RegSet,
    target::{RegClassInfo, TargetDescription},
};
use hashbrown::HashMap;
use veloc_lir::{Reg, RegisterAccess};

pub(super) struct PressureTracker<'a> {
    region: Region<'a>,
    target: &'a TargetDescription,
    sets: &'a [RegClassInfo],
    live: RegSet,
    live_out: &'a RegSet,
    remaining: HashMap<Reg, usize>,
    pressure: Vec<isize>,
    capacity: Vec<isize>,
}

impl<'a> PressureTracker<'a> {
    pub fn new(region: Region<'a>, target: &'a TargetDescription, live_out: &'a RegSet) -> Self {
        let mut live = live_out.clone();
        let mut remaining = HashMap::new();
        for i in (0..region.insts.len()).rev() {
            let inst = region.inst(i);
            before(inst, &mut live);
            for r in inst.register_access().reads() {
                *remaining.entry(r).or_default() += 1;
            }
        }
        let sets = target.registers.reg_classes;
        let capacity = sets
            .iter()
            .map(|set| {
                set.allocatable
                    .iter()
                    .filter(|r| !target.registers.reserved_regs.contains(r))
                    .count() as isize
            })
            .collect();
        let mut result = Self {
            region,
            target,
            sets,
            live,
            live_out,
            remaining,
            pressure: vec![0; sets.len()],
            capacity,
        };
        for (i, set) in sets.iter().enumerate() {
            result.pressure[i] = result
                .live
                .iter()
                .filter(|&r| result.in_set(r, set))
                .count() as isize;
        }
        result
    }

    fn in_set(&self, r: Reg, set: &RegClassInfo) -> bool {
        if self.region.function.state_unit(r).is_some() {
            return false;
        }
        if r.is_preg() {
            return set.allocatable.contains(&r);
        }
        let data = self.region.function.vreg_data(r);
        self.target.reg_class_for_vreg(&data.ty, data.bank()) == set.kind
    }

    fn needed_after(&self, access: RegisterAccess<'_>, r: Reg) -> bool {
        self.remaining.get(&r).copied().unwrap_or(0) > usize::from(access.is_read(r))
            || self.live_out.contains(&r)
    }

    fn delta(&self, access: RegisterAccess<'_>, set: &RegClassInfo) -> isize {
        access
            .all()
            .filter(|&r| self.in_set(r, set))
            .map(|r| {
                isize::from(self.needed_after(access, r)) - isize::from(self.live.contains(&r))
            })
            .sum()
    }

    pub fn score(&self, node: usize) -> (isize, isize) {
        let access = self.region.inst(node).register_access();
        self.sets
            .iter()
            .enumerate()
            .fold((0, 0), |(excess, total), (i, set)| {
                let next = self.pressure[i] + self.delta(access, set);
                (excess + (next - self.capacity[i]).max(0), total + next)
            })
    }

    pub fn advance(&mut self, node: usize) {
        let access = self.region.inst(node).register_access();
        for (i, set) in self.sets.iter().enumerate() {
            self.pressure[i] += self.delta(access, set);
        }
        // Remaining uses count instructions, independent of operand multiplicity.
        for r in access.all() {
            if self.needed_after(access, r) {
                self.live.insert(r);
            } else {
                self.live.remove(&r);
            }
        }
        for r in access.reads() {
            *self.remaining.get_mut(&r).unwrap() -= 1;
        }
    }
}
