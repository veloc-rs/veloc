//! Approximate live-register pressure using the allocator's register classes.
use super::{
    before,
    graph::{DependencyGraph, Node},
};
use crate::{
    analysis::RegSet,
    target::{RegClassInfo, TargetDescription},
};
use hashbrown::HashMap;
use veloc_lir::{MachineFunction, Reg};

pub(super) struct PressureTracker<'a> {
    function: &'a MachineFunction,
    target: &'a TargetDescription,
    sets: &'a [RegClassInfo],
    live: RegSet,
    live_out: &'a RegSet,
    remaining: HashMap<Reg, usize>,
    pressure: Vec<isize>,
    capacity: Vec<isize>,
}

impl<'a> PressureTracker<'a> {
    pub fn new(
        function: &'a MachineFunction,
        graph: &DependencyGraph,
        target: &'a TargetDescription,
        live_out: &'a RegSet,
    ) -> Self {
        let mut live = live_out.clone();
        let mut remaining = HashMap::new();
        for node in graph.nodes.iter().rev() {
            before(function, node.inst, &mut live);
            for &r in &node.uses {
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
            function,
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
        if r.is_preg() {
            return set.allocatable.contains(&r);
        }
        let data = self.function.vreg_data(r);
        self.target.reg_class_for_vreg(&data.ty, data.bank) == set.kind
    }

    fn needed_after(&self, node: &Node, r: Reg) -> bool {
        self.remaining.get(&r).copied().unwrap_or(0) > usize::from(node.uses.contains(&r))
            || self.live_out.contains(&r)
    }

    fn delta(&self, node: &Node, set: &RegClassInfo) -> isize {
        node.uses
            .iter()
            .chain(node.defs.iter().filter(|r| !node.uses.contains(r)))
            .filter(|&&r| self.in_set(r, set))
            .map(|&r| isize::from(self.needed_after(node, r)) - isize::from(self.live.contains(&r)))
            .sum()
    }

    pub fn score(&self, node: &Node) -> (isize, isize) {
        self.sets
            .iter()
            .enumerate()
            .fold((0, 0), |(excess, total), (i, set)| {
                let next = self.pressure[i] + self.delta(node, set);
                (excess + (next - self.capacity[i]).max(0), total + next)
            })
    }

    pub fn advance(&mut self, node: &Node) {
        for (i, set) in self.sets.iter().enumerate() {
            self.pressure[i] += self.delta(node, set);
        }
        // Compute before decrementing remaining uses, including repeated operands.
        for &r in node.uses.iter().chain(&node.defs) {
            if self.needed_after(node, r) {
                self.live.insert(r);
            } else {
                self.live.remove(&r);
            }
        }
        for r in &node.uses {
            *self.remaining.get_mut(r).unwrap() -= 1;
        }
    }
}
