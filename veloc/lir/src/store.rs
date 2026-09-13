//! Function-owned instruction storage. IDs are stable; operand ranges and cold
//! payload slots are recycled on replacement, never exposed as owning fields.
use crate::use_def::{RefRange, References, Site};
use crate::{InstExtra, InstId, InstRef, MachineOpcode, MachineOperand, MemoryAccess};
use crate::{RefLocation, RefRole, Reg, RegRefs, VReg};
use alloc::vec::Vec;
use cranelift_entity::PrimaryMap;

#[derive(Debug, Clone, Copy, Default)]
struct Range {
    start: u32,
    len: u32,
}

impl Range {
    fn indices(self) -> core::ops::Range<usize> {
        self.start as usize..self.start as usize + self.len as usize
    }
}

#[derive(Debug, Clone, Default)]
struct Operands {
    data: Vec<MachineOperand>,
    free: Vec<Vec<u32>>,
}

impl Operands {
    fn insert(&mut self, values: &[MachineOperand]) -> Range {
        if values.is_empty() {
            return Range::default();
        }
        let capacity = values
            .len()
            .checked_next_power_of_two()
            .expect("operand count overflow");
        let class = capacity.trailing_zeros() as usize;
        self.free
            .resize_with(self.free.len().max(class + 1), Vec::new);
        let start = self.free[class].pop().unwrap_or_else(|| {
            let start = u32::try_from(self.data.len()).expect("operand store overflow");
            let end = self
                .data
                .len()
                .checked_add(capacity)
                .expect("operand store overflow");
            assert!(end <= u32::MAX as usize, "operand store overflow");
            self.data.resize(end, MachineOperand::Imm(0));
            start
        });
        let range = Range {
            start,
            len: values.len().try_into().expect("operand count overflow"),
        };
        self.data[range.indices()].clone_from_slice(values);
        range
    }

    fn release(&mut self, range: Range) {
        if range.len == 0 {
            return;
        }
        let class = range.len.next_power_of_two().trailing_zeros() as usize;
        self.free[class].push(range.start);
    }
}

const NONE: u32 = u32::MAX;

#[derive(Debug, Clone)]
enum Slot<T> {
    Live(T),
    Free(u32),
}

#[derive(Debug, Clone)]
struct Pool<T> {
    slots: Vec<Slot<T>>,
    free: u32,
}

impl<T> Default for Pool<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: NONE,
        }
    }
}

impl<T> Pool<T> {
    fn insert(&mut self, value: T) -> u32 {
        if self.free != NONE {
            let id = self.free;
            let Slot::Free(next) = self.slots[id as usize] else {
                unreachable!("free slot")
            };
            self.free = next;
            self.slots[id as usize] = Slot::Live(value);
            id
        } else {
            let id = u32::try_from(self.slots.len()).expect("payload store overflow");
            assert_ne!(id, NONE);
            self.slots.push(Slot::Live(value));
            id
        }
    }
    fn get(&self, id: u32) -> Option<&T> {
        if id == NONE {
            return None;
        }
        let Slot::Live(value) = &self.slots[id as usize] else {
            unreachable!("live slot")
        };
        Some(value)
    }
    fn remove(&mut self, id: u32) {
        if id == NONE {
            return;
        }
        assert!(matches!(self.slots[id as usize], Slot::Live(_)));
        self.slots[id as usize] = Slot::Free(self.free);
        self.free = id;
    }
}

#[derive(Debug, Clone)]
struct StoredInst {
    opcode: MachineOpcode,
    operands: Range,
    memory: u32,
    extra: u32,
    refs: RefRange,
}

#[derive(Debug, Clone, Default)]
pub struct InstStore {
    instructions: PrimaryMap<InstId, StoredInst>,
    operands: Operands,
    memory: Pool<MemoryAccess>,
    extras: Pool<InstExtra>,
    pub(crate) references: References,
}

/// A single committed write. Generated methods encode directly from their typed
/// arguments; no owning instruction or temporary operand vector is required.
pub struct InstWriter<'a> {
    store: &'a mut InstStore,
    target: Option<InstId>,
    memory: Option<crate::MemoryAccess>,
}

impl InstWriter<'_> {
    pub fn with_memory(mut self, access: crate::MemoryAccess) -> Self {
        self.memory = Some(access);
        self
    }

    pub fn write(self, opcode: crate::MachineOpcode, operands: &[crate::MachineOperand]) -> InstId {
        match self.target {
            Some(id) => {
                self.store.write_at(id, opcode, operands, self.memory);
                id
            }
            None => self.store.write(opcode, operands, self.memory),
        }
    }
}

impl InstStore {
    pub fn writer(&mut self) -> crate::InstWriter<'_> {
        crate::InstWriter {
            store: self,
            target: None,
            memory: None,
        }
    }

    pub fn rewriter(&mut self, id: InstId) -> InstWriter<'_> {
        InstWriter {
            store: self,
            target: Some(id),
            memory: None,
        }
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }
    pub fn get(&self, id: InstId) -> InstRef<'_> {
        // Validate the ID even if the caller does not access any fields.
        let _ = &self.instructions[id];
        InstRef { store: self, id }
    }
    pub fn opcode(&self, id: InstId) -> MachineOpcode {
        self.instructions[id].opcode
    }
    pub fn operands(&self, id: InstId) -> &[MachineOperand] {
        &self.operands.data[self.instructions[id].operands.indices()]
    }
    pub fn memory(&self, id: InstId) -> Option<MemoryAccess> {
        self.memory.get(self.instructions[id].memory).copied()
    }
    pub fn write(
        &mut self,
        opcode: MachineOpcode,
        operands: &[MachineOperand],
        memory: Option<MemoryAccess>,
    ) -> InstId {
        let operands = self.operands.insert(operands);
        let memory = memory.map_or(NONE, |access| self.memory.insert(access));
        let id = self.instructions.push(StoredInst {
            opcode,
            operands,
            memory,
            extra: NONE,
            refs: RefRange::default(),
        });
        self.index_refs(id);
        id
    }
    /// Transfer a detached instruction into a stable destination ID.
    pub fn replace(&mut self, id: InstId, source: InstId) {
        if id == source {
            return;
        }
        self.write_at(id, MachineOpcode::Invalid, &[], None);
        let empty = StoredInst {
            opcode: MachineOpcode::Invalid,
            operands: Range { start: 0, len: 0 },
            memory: NONE,
            extra: NONE,
            refs: RefRange::default(),
        };
        self.instructions[id] = core::mem::replace(&mut self.instructions[source], empty);
        for link in self.instructions[id].refs.ids() {
            let mut site = self.references.links.owner(link);
            site.inst = id;
            self.references.links.set_owner(link, site);
        }
    }
    pub fn write_at(
        &mut self,
        id: InstId,
        opcode: MachineOpcode,
        operands: &[MachineOperand],
        memory: Option<MemoryAccess>,
    ) {
        self.clear_refs(id);
        self.extras.remove(self.instructions[id].extra);
        self.instructions[id].extra = NONE;
        self.set_operand_data(id, operands);
        self.set_memory(id, memory);
        self.instructions[id].opcode = opcode;
        self.index_refs(id);
    }
    pub fn set_operands(&mut self, id: InstId, operands: &[MachineOperand]) {
        self.clear_refs(id);
        self.set_operand_data(id, operands);
        self.index_refs(id);
    }
    fn set_operand_data(&mut self, id: InstId, operands: &[MachineOperand]) {
        let old = self.instructions[id].operands;
        if operands.len() == old.len as usize {
            self.operands.data[old.indices()].clone_from_slice(operands);
        } else {
            self.operands.release(old);
            self.instructions[id].operands = self.operands.insert(operands);
        }
    }
    pub fn set_operand(&mut self, id: InstId, index: usize, operand: MachineOperand) {
        assert!(
            index < self.operands(id).len(),
            "operand index out of bounds"
        );
        let start = self.instructions[id].operands.start as usize;
        let old = &self.operands.data[start + index];
        if old.is_use() == operand.is_use() && old.is_def() == operand.is_def() {
            if let (Some(old), Some(new)) = (old.as_reg(), operand.as_reg()) {
                if old != new {
                    for link in self.instructions[id].refs.ids() {
                        let site = self.references.links.owner(link);
                        if site.location() == RefLocation::Operand(index as u32) {
                            self.references.detach(link, old);
                            self.references.attach(link, new, site);
                        }
                    }
                }
            }
            self.operands.data[start + index] = operand;
        } else {
            self.clear_refs(id);
            self.operands.data[start + index] = operand;
            self.index_refs(id);
        }
    }
    pub fn set_memory(&mut self, id: InstId, access: Option<MemoryAccess>) {
        self.memory.remove(self.instructions[id].memory);
        self.instructions[id].memory = access.map_or(NONE, |a| self.memory.insert(a));
    }
    pub fn extra(&self, id: InstId) -> Option<&InstExtra> {
        self.extras.get(self.instructions[id].extra)
    }
    pub fn set_extra(&mut self, id: InstId, extra: InstExtra) {
        self.clear_refs(id);
        self.extras.remove(self.instructions[id].extra);
        self.instructions[id].extra = self.extras.insert(extra);
        self.index_refs(id);
    }
    pub fn clear_extra(&mut self, id: InstId) {
        self.clear_refs(id);
        self.extras.remove(self.instructions[id].extra);
        self.instructions[id].extra = NONE;
        self.index_refs(id);
    }
    pub fn uses(&self, reg: Reg) -> RegRefs<'_> {
        RegRefs {
            store: self,
            next: self.references.head(reg, RefRole::Use),
        }
    }
    pub fn defs(&self, reg: Reg) -> RegRefs<'_> {
        RegRefs {
            store: self,
            next: self.references.head(reg, RefRole::Def),
        }
    }
    pub(crate) fn reg_at(&self, site: Site) -> Reg {
        match site.location() {
            RefLocation::Operand(index) => self.operands(site.inst)[index as usize]
                .as_reg()
                .expect("reference must point to a register"),
            RefLocation::EdgeArg(index) => self
                .extra(site.inst)
                .expect("edge payload")
                .edge_args()
                .nth(index as usize)
                .expect("edge reference"),
        }
    }
    fn clear_refs(&mut self, id: InstId) {
        let range = self.instructions[id].refs;
        for link in range.ids() {
            let reg = self.reg_at(self.references.links.owner(link));
            self.references.detach(link, reg);
        }
        self.references.release(range);
        self.instructions[id].refs = RefRange::default();
    }
    fn reference_sites(&self, id: InstId) -> impl Iterator<Item = (Reg, Site)> + '_ {
        let operands = self
            .operands(id)
            .iter()
            .enumerate()
            .flat_map(move |(index, operand)| {
                [RefRole::Use, RefRole::Def]
                    .into_iter()
                    .filter_map(move |role| {
                        let present = match role {
                            RefRole::Use => operand.is_use(),
                            RefRole::Def => operand.is_def(),
                        };
                        present.then(|| {
                            (
                                operand.as_reg().unwrap(),
                                Site::new(id, RefLocation::Operand(index as u32), role),
                            )
                        })
                    })
            });
        let edges = self
            .extra(id)
            .into_iter()
            .flat_map(InstExtra::edge_args)
            .enumerate()
            .map(move |(index, reg)| {
                (
                    reg,
                    Site::new(id, RefLocation::EdgeArg(index as u32), RefRole::Use),
                )
            });
        operands.chain(edges)
    }
    fn index_refs(&mut self, id: InstId) {
        let sites: smallvec::SmallVec<[_; 8]> = self.reference_sites(id).collect();
        let owner = Site::new(id, RefLocation::Operand(0), RefRole::Use);
        let range = self.references.alloc(sites.len(), owner);
        for (link, (reg, site)) in range.ids().zip(sites) {
            self.references.attach(link, reg, site);
        }
        self.instructions[id].refs = range;
    }

    /// Mechanically replace virtual uses, including edge arguments. The caller
    /// remains responsible for dominance and semantic/type correctness.
    pub fn replace_uses(&mut self, old: VReg, new: VReg) {
        if old == new {
            return;
        }
        let old = Reg::new_vreg(old.as_u32());
        let new = Reg::new_vreg(new.as_u32());
        while let Some(link) = self.references.head(old, RefRole::Use).expand() {
            let site = self.references.links.owner(link);
            self.references.detach(link, old);
            match site.location() {
                RefLocation::Operand(index) => {
                    let start = self.instructions[site.inst].operands.start as usize;
                    self.operands.data[start + index as usize] = MachineOperand::Use(new);
                }
                RefLocation::EdgeArg(index) => {
                    let Slot::Live(extra) =
                        &mut self.extras.slots[self.instructions[site.inst].extra as usize]
                    else {
                        unreachable!("live extra")
                    };
                    *extra.edge_arg_mut(index as usize) = new;
                }
            }
            self.references.attach(link, new, site);
        }
    }

    /// Independently audit every live occurrence and both directions of its list.
    pub fn check_refs(&self) -> Result<(), &'static str> {
        self.references
            .check_ranges(self.instructions.iter().map(|(id, inst)| (id, inst.refs)))?;
        let mut expected = hashbrown::HashMap::new();
        for (id, _) in self.instructions.iter() {
            for (reg, site) in self.reference_sites(id) {
                if !expected
                    .entry((reg, site.role() as u8))
                    .or_insert_with(hashbrown::HashSet::new)
                    .insert(site)
                {
                    return Err("duplicate logical reference");
                }
            }
        }
        for (reg, role, _) in self.references.heads() {
            if !expected.contains_key(&(reg, role as u8)) {
                return Err("head for an unreferenced register");
            }
        }
        let mut seen = alloc::vec![false; self.references.links.len()];
        for ((reg, role), mut sites) in expected {
            let role = if role == RefRole::Use as u8 {
                RefRole::Use
            } else {
                RefRole::Def
            };
            let mut next = self.references.head(reg, role);
            let mut prev = None.into();
            while let Some(link) = next.expand() {
                use cranelift_entity::EntityRef;
                let mark = seen
                    .get_mut(link.index())
                    .ok_or("reference link out of bounds")?;
                if core::mem::replace(mark, true) {
                    return Err("cyclic or duplicate reference");
                }
                let site = self.references.links.owner(link);
                if self.references.links.prev(link) != prev {
                    return Err("incorrect previous link");
                }
                if !sites.remove(&site) {
                    return Err("unexpected reference");
                }
                if self.reg_at(site) != reg {
                    return Err("reference register mismatch");
                }
                prev = Some(link).into();
                next = self.references.links.next(link);
            }
            if !sites.is_empty() {
                return Err("missing reference");
            }
        }
        for (id, inst) in self.instructions.iter() {
            for link in inst.refs.ids() {
                if self.references.links.owner(link).inst != id {
                    return Err("incorrect reference owner");
                }
                use cranelift_entity::EntityRef;
                if !seen[link.index()] {
                    return Err("unlinked live reference");
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BranchInfo, MachineFunction, MemoryKind, Reg, Writable, stages::RawLir};

    #[test]
    fn pooled_storage_recycles_payloads_and_preserves_borrowed_views() {
        assert!(core::mem::size_of::<StoredInst>() <= 32);
        let mut f = MachineFunction::<RawLir>::new("store".into());
        let block = f.create_synthetic_block();
        let reg = f.alloc_vreg(crate::Type::I64);
        let id = f.writer().constant(Writable(reg), 42);
        f.append_inst_id_to_block(0, id);
        let view = f.inst(id);
        assert_eq!(view.operands().as_ptr(), f.inst(id).operands().as_ptr());
        for _ in 0..100 {
            let access = MemoryAccess::new(MemoryKind::Read, 8);
            assert_eq!(
                f.rewriter(id)
                    .with_memory(access)
                    .offset_load(Writable(reg), reg, 16),
                id
            );
            f.set_inst_extra(
                id,
                InstExtra::Branch(BranchInfo {
                    args: Default::default(),
                }),
            );
            assert_eq!(f.inst(id).memory(), Some(access));
            assert!(f.inst_extra(id).is_some());
            assert_eq!(f.rewriter(id).constant(Writable(reg), 42), id);
            assert!(f.inst(id).memory().is_none());
            assert!(f.inst_extra(id).is_none());
        }
        // The generic and target namespaces use the exact same store.
        let target = f
            .writer()
            .write(MachineOpcode::Target(7), &[MachineOperand::Use(reg)]);
        f.append_inst_id_to_block(0, target);
        assert!(f.inst(id).is_generic());
        assert!(f.inst(target).is_target());
        let access = MemoryAccess::new(MemoryKind::Read, 8);
        let replacement = f
            .writer()
            .with_memory(access)
            .offset_load(Writable(reg), reg, 0);
        let extra = InstExtra::Branch(BranchInfo {
            args: Default::default(),
        });
        f.set_inst_extra(replacement, extra.clone());
        let operands = f.inst(replacement).operands().as_ptr();
        f.replace_inst(id, replacement);
        assert_eq!(f.inst(id).operands().as_ptr(), operands);
        assert_eq!(f.inst(id).memory(), Some(access));
        assert_eq!(f.inst_extra(id), Some(&extra));
        assert!(f.inst(replacement).is_invalid());
        assert!(f.inst_extra(replacement).is_none());
        // Worklist rewrites detach and reinsert IDs without erasing their data.
        f.rewrite_block(0, |cursor| {
            let id = cursor.current_inst_id();
            cursor.detach_current();
            cursor.emit(id);
            Ok::<(), ()>(())
        })
        .unwrap();
        assert_eq!(f.block_insts(0), &[id, target]);
        assert_eq!(f.inst(id).memory(), Some(access));
        let mut cloned = f.clone();
        cloned.set_inst_operands(target, [MachineOperand::Use(Reg::new_preg(1))]);
        assert_eq!(f.inst(target).uses().collect::<Vec<_>>(), [reg]);
        assert_eq!(
            cloned.inst(target).uses().collect::<Vec<_>>(),
            [Reg::new_preg(1)]
        );
        f.rewrite_block(0, |cursor| {
            cursor.remove_current();
            Ok::<(), ()>(())
        })
        .unwrap();
        assert!(f.inst(id).is_invalid());
        assert!(f.inst(target).is_invalid());
        assert!(f.block_insts(0).is_empty());
        assert_eq!(f.blocks[0].id, block);

        let mut store = InstStore::default();
        let id = store.write(MachineOpcode::Target(1), &[], None);
        for n in 0..100 {
            store.set_operands(id, &[]);
            store.set_operands(id, &[MachineOperand::Imm(n), MachineOperand::Use(reg)]);
            store.set_memory(id, Some(MemoryAccess::new(MemoryKind::Read, 8)));
            store.set_extra(
                id,
                InstExtra::Branch(BranchInfo {
                    args: Default::default(),
                }),
            );
            store.write_at(id, MachineOpcode::Invalid, &[], None);
        }
        assert_eq!(store.operands.data.len(), 2);
        assert_eq!(store.memory.slots.len(), 1);
        assert_eq!(store.extras.slots.len(), 1);
    }
}
