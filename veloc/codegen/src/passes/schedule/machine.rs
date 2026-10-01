//! Resource reservations affect priorities, never semantic legality.
use crate::target::{ScheduleCost, ScheduleModel};
use hashbrown::HashMap;

pub(super) struct MachineState {
    pub cycle: u32,
    width: u32,
    issued: u32,
    resources: HashMap<&'static str, Vec<u32>>,
}

impl MachineState {
    pub fn new(model: &ScheduleModel) -> Self {
        assert!(model.issue_width > 0);
        Self {
            cycle: 0,
            width: model.issue_width,
            issued: 0,
            resources: model
                .resources
                .iter()
                .map(|r| (r.name, vec![0; r.units as usize]))
                .collect(),
        }
    }

    pub fn earliest(&self, available: u32, cost: ScheduleCost) -> u32 {
        let resource = self
            .resources
            .get(cost.resource)
            .expect("unknown CPU resource")
            .iter()
            .min()
            .copied()
            .expect("CPU resource has no units");
        available.max(self.cycle).max(resource)
    }

    pub fn advance_to(&mut self, cycle: u32) {
        if cycle > self.cycle {
            self.cycle = cycle;
            self.issued = 0;
        }
    }

    pub fn issue(&mut self, cost: ScheduleCost) {
        assert!(cost.occupancy > 0);
        let units = self
            .resources
            .get_mut(cost.resource)
            .expect("unknown CPU resource");
        let next = units.iter_mut().min().unwrap();
        assert!(*next <= self.cycle);
        *next = self.cycle.saturating_add(cost.occupancy);
        self.issued += 1;
        if self.issued == self.width {
            self.cycle = self.cycle.saturating_add(1);
            self.issued = 0;
        }
    }
}
