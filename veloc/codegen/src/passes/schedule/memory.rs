//! Alias proofs relax only ordinary nontrapping accesses. Unknown addresses,
//! traps and volatile accesses retain their observable order.
use super::NodeId;
use crate::target::TargetSchedule;
use hashbrown::HashMap;
use smallvec::SmallVec;
use veloc_lir::{InstRef, MachineOpcode, Reg};
use veloc_types::{MemoryEffects, MemoryLocation};

type Location = MemoryLocation<(Reg, Option<NodeId>)>;
struct Access {
    node: NodeId,
    read: bool,
    location: Option<Location>,
}

#[derive(Default)]
pub(super) struct MemoryDependencies {
    barrier: Option<NodeId>,
    pending: Vec<Access>,
    versions: HashMap<Reg, NodeId>,
}

impl MemoryDependencies {
    pub fn write(&mut self, reg: Reg, node: NodeId) {
        self.versions.insert(reg, node);
    }

    pub fn predecessors(
        &mut self,
        node: NodeId,
        inst: InstRef<'_>,
        target: &dyn TargetSchedule,
    ) -> SmallVec<[NodeId; 4]> {
        let mut edges = SmallVec::new();
        let Some(flags) = inst.mem_flags() else {
            return edges;
        };
        if let Some(barrier) = self.barrier {
            edges.push(barrier);
        }
        let metadata = match inst.opcode() {
            MachineOpcode::Target(op) => target.instruction_metadata(op).memory,
            _ => None,
        };
        let Some(metadata) = metadata.filter(|_| flags.is_notrap() && !flags.is_volatile()) else {
            edges.extend(self.pending.drain(..).map(|access| access.node));
            self.barrier = Some(node);
            return edges;
        };
        let read = metadata.effect == MemoryEffects::READ;
        let location = metadata.location(inst).map(|address| MemoryLocation {
            base: (address.base, self.versions.get(&address.base).copied()),
            offset: address.offset,
            bytes: address.bytes,
        });
        let bits = u32::from(target.desc().data_layout.pointer_size) * 8;
        for previous in &self.pending {
            if read && previous.read {
                continue;
            }
            let aliases = match (&location, &previous.location) {
                (Some(a), Some(b)) => a.may_overlap(b, bits),
                _ => true,
            };
            if aliases {
                edges.push(previous.node);
            }
        }
        if !read && location.is_none() {
            // An unknown write orders every pending access. Future accesses can
            // depend on this node instead of retaining all its predecessors.
            self.pending.clear();
            self.barrier = Some(node);
        } else {
            if !read {
                // This write supersedes dependencies with the same footprint:
                // every future conflict with them also conflicts with this node.
                self.pending
                    .retain(|previous| previous.location != location);
            }
            self.pending.push(Access {
                node,
                read,
                location,
            });
        }
        edges
    }
}
