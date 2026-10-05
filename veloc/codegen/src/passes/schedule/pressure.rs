//! Approximate live-register pressure using the allocator's register classes.
use super::{NodeId, Region, before};
use crate::{
    analysis::RegSet,
    target::{RegClassInfo, TargetDescription},
};
use cranelift_entity::{PrimaryMap, SecondaryMap, entity_impl};
use hashbrown::HashMap;
use veloc_lir::{Reg, RegisterAccess};

/// A local pressure-set index, independent of the target's register-class enum.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct PressureSetId(u32);
entity_impl!(PressureSetId, "pressure_set");

#[derive(Default)]
pub(super) struct PressureScore {
    pub excess: isize,
    pub total: isize,
}

pub(super) struct PressureTracker<'a> {
    region: Region<'a>,
    target: &'a TargetDescription,
    sets: PrimaryMap<PressureSetId, &'a RegClassInfo>,
    live: RegSet,
    live_out: &'a RegSet,
    remaining: HashMap<Reg, usize>,
    pressure: SecondaryMap<PressureSetId, isize>,
    capacity: SecondaryMap<PressureSetId, isize>,
}

impl<'a> PressureTracker<'a> {
    pub fn new(region: Region<'a>, target: &'a TargetDescription, live_out: &'a RegSet) -> Self {
        let mut live = live_out.clone();
        let mut remaining = HashMap::new();
        for node in region.nodes().rev() {
            let inst = region.inst(node);
            before(inst, &mut live);
            for r in inst.register_access().reads() {
                *remaining.entry(r).or_default() += 1;
            }
        }
        let sets: PrimaryMap<PressureSetId, _> = target.registers.reg_classes.iter().collect();
        let capacity = sets
            .iter()
            .map(|(id, set)| {
                let capacity = set
                    .allocatable
                    .iter()
                    .filter(|r| !target.registers.reserved_regs.contains(r))
                    .count() as isize;
                (id, capacity)
            })
            .collect();
        let mut result = Self {
            region,
            target,
            sets,
            live,
            live_out,
            remaining,
            pressure: SecondaryMap::new(),
            capacity,
        };
        for (set_id, set) in result.sets.iter() {
            result.pressure[set_id] = result
                .live
                .iter()
                .filter(|&r| result.in_set(r, set))
                .count() as isize;
        }
        result
    }

    fn in_set(&self, r: Reg, set: &RegClassInfo) -> bool {
        if r.is_preg() {
            return set.allocatable.contains(&r);
        }
        let data = self.region.function.vreg_data(r);
        self.target.reg_class_for_vreg(&data.ty, data.bank) == set.kind
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

    pub fn score(&self, node: NodeId) -> PressureScore {
        let access = self.region.inst(node).register_access();
        self.sets
            .iter()
            .fold(PressureScore::default(), |mut score, (set_id, set)| {
                let next = self.pressure[set_id] + self.delta(access, set);
                score.excess += (next - self.capacity[set_id]).max(0);
                score.total += next;
                score
            })
    }

    /// Pressure relative to each allocation class's capacity, without CPU IDs.
    pub fn max_relative_pressure(&self) -> f32 {
        self.sets
            .keys()
            .map(|id| self.pressure[id].max(0) as f32 / self.capacity[id].max(1) as f32)
            .fold(0.0, f32::max)
    }

    pub fn advance(&mut self, node: NodeId) {
        let access = self.region.inst(node).register_access();
        for (set_id, set) in self.sets.iter() {
            self.pressure[set_id] += self.delta(access, set);
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
