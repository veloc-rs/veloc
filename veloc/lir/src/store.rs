//! Function-owned instruction storage. IDs are stable; operand ranges and cold
//! payload slots are recycled on replacement, never exposed as owning fields.
use crate::InstField;
use crate::use_def::{RefRange, References, Site};
use crate::{InstExtra, InstId, InstRef, MachineOpcode, MemoryAccess};
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

#[derive(Debug, Clone)]
struct Operands<T> {
    data: Vec<T>,
    free: Vec<Vec<u32>>,
}

impl<T> Default for Operands<T> {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            free: Vec::new(),
        }
    }
}

impl<T: Clone> Operands<T> {
    fn insert(&mut self, values: &[T]) -> Range {
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
            self.data.resize(end, values[0].clone());
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

#[derive(Debug, Clone, Default)]
pub struct RegEffects {
    pub uses: Vec<Reg>,
    pub defs: Vec<Reg>,
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
    inputs: Range,
    fields: Range,
    results: Range,
    memory: u32,
    extra: u32,
    refs: RefRange,
    effects: u32,
}

#[derive(Debug, Clone, Default)]
pub struct InstStore {
    instructions: PrimaryMap<InstId, StoredInst>,
    inputs: Operands<Reg>,
    fields: Operands<InstField>,
    results: Operands<Reg>,
    memory: Pool<MemoryAccess>,
    extras: Pool<InstExtra>,
    effects: Pool<RegEffects>,
    pub(crate) references: References,
}

/// A single committed write. Generated methods encode directly from their typed
/// arguments; no owning instruction or temporary operand vector is required.
pub struct InstWriter<'a> {
    store: &'a mut InstStore,
    target: Option<InstId>,
    memory: Option<crate::MemoryAccess>,
    effects: RegEffects,
}

impl InstWriter<'_> {
    pub fn with_effects(mut self, uses: &[Reg], defs: &[Reg]) -> Self {
        assert!(
            uses.iter().chain(defs).all(Reg::is_preg),
            "implicit effects require physical registers"
        );
        self.effects = RegEffects {
            uses: uses.to_vec(),
            defs: defs.to_vec(),
        };
        self
    }
    pub fn with_memory(mut self, access: crate::MemoryAccess) -> Self {
        self.memory = Some(access);
        self
    }

    pub fn write(
        self,
        opcode: crate::MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: &[InstField],
    ) -> InstId {
        let id = match self.target {
            Some(id) => {
                self.store
                    .write_at(id, opcode, results, inputs, fields, self.memory);
                id
            }
            None => self
                .store
                .write(opcode, results, inputs, fields, self.memory),
        };
        if !self.effects.uses.is_empty() || !self.effects.defs.is_empty() {
            self.store.set_effects(id, self.effects);
        }
        id
    }
}

// The generated contract owns generic builders; this adapter owns storage and
// the conversion from generic to machine opcodes.
impl crate::InstBuild for InstWriter<'_> {
    type Inst = InstId;
    type Def = crate::Writable<Reg>;
    fn reg(value: Self::Def) -> Reg {
        value.to_reg()
    }
    fn write(
        self,
        opcode: crate::GenericOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: &[InstField],
    ) -> InstId {
        self.write(MachineOpcode::Generic(opcode), results, inputs, fields)
    }
}

impl InstStore {
    pub fn writer(&mut self) -> crate::InstWriter<'_> {
        crate::InstWriter {
            store: self,
            target: None,
            memory: None,
            effects: RegEffects::default(),
        }
    }

    pub fn rewriter(&mut self, id: InstId) -> InstWriter<'_> {
        InstWriter {
            store: self,
            target: Some(id),
            memory: None,
            effects: RegEffects::default(),
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
    pub fn results(&self, id: InstId) -> &[Reg] {
        &self.results.data[self.instructions[id].results.indices()]
    }
    pub fn set_results(&mut self, id: InstId, results: &[Reg]) {
        self.clear_refs(id);
        self.results.release(self.instructions[id].results);
        self.instructions[id].results = self.results.insert(results);
        self.index_refs(id);
    }
    pub fn set_result(&mut self, id: InstId, index: usize, reg: Reg) {
        assert!(index < self.results(id).len());
        self.clear_refs(id);
        self.results.data[self.instructions[id].results.start as usize + index] = reg;
        self.index_refs(id);
    }
    pub fn inputs(&self, id: InstId) -> &[Reg] {
        &self.inputs.data[self.instructions[id].inputs.indices()]
    }
    pub fn fields(&self, id: InstId) -> &[InstField] {
        &self.fields.data[self.instructions[id].fields.indices()]
    }
    pub fn set_input(&mut self, id: InstId, index: usize, reg: Reg) {
        let old = self.inputs(id)[index];
        if old == reg {
            return;
        }
        for link in self.instructions[id].refs.ids() {
            let site = self.references.links.owner(link);
            if site.location() == RefLocation::Input(index as u32) {
                self.references.detach(link, old);
                self.references.attach(link, reg, site);
            }
        }
        self.inputs.data[self.instructions[id].inputs.start as usize + index] = reg;
    }
    pub fn set_inputs(&mut self, id: InstId, inputs: &[Reg]) {
        assert_eq!(
            inputs.len(),
            self.inputs(id).len(),
            "input shape must not change"
        );
        for (i, &reg) in inputs.iter().enumerate() {
            self.set_input(id, i, reg);
        }
    }
    pub fn memory(&self, id: InstId) -> Option<MemoryAccess> {
        self.memory.get(self.instructions[id].memory).copied()
    }
    pub fn write(
        &mut self,
        opcode: MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: &[InstField],
        memory: Option<MemoryAccess>,
    ) -> InstId {
        let inputs = self.inputs.insert(inputs);
        let fields = self.fields.insert(fields);
        let results = self.results.insert(results);
        let memory = memory.map_or(NONE, |access| self.memory.insert(access));
        let id = self.instructions.push(StoredInst {
            opcode,
            inputs,
            fields,
            results,
            memory,
            extra: NONE,
            refs: RefRange::default(),
            effects: NONE,
        });
        self.index_refs(id);
        id
    }
    /// Transfer a detached instruction into a stable destination ID.
    pub fn replace(&mut self, id: InstId, source: InstId) {
        if id == source {
            return;
        }
        self.write_at(id, MachineOpcode::Invalid, &[], &[], &[], None);
        let empty = StoredInst {
            opcode: MachineOpcode::Invalid,
            inputs: Range::default(),
            fields: Range::default(),
            results: Range::default(),
            memory: NONE,
            extra: NONE,
            refs: RefRange::default(),
            effects: NONE,
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
        results: &[Reg],
        inputs: &[Reg],
        fields: &[InstField],
        memory: Option<MemoryAccess>,
    ) {
        self.clear_refs(id);
        self.effects.remove(self.instructions[id].effects);
        self.instructions[id].effects = NONE;
        self.extras.remove(self.instructions[id].extra);
        self.instructions[id].extra = NONE;
        self.inputs.release(self.instructions[id].inputs);
        self.fields.release(self.instructions[id].fields);
        self.instructions[id].inputs = self.inputs.insert(inputs);
        self.instructions[id].fields = self.fields.insert(fields);
        self.results.release(self.instructions[id].results);
        self.instructions[id].results = self.results.insert(results);
        self.set_memory(id, memory);
        self.instructions[id].opcode = opcode;
        self.index_refs(id);
    }
    pub fn set_fields(&mut self, id: InstId, fields: &[InstField]) {
        self.fields.release(self.instructions[id].fields);
        self.instructions[id].fields = self.fields.insert(fields);
    }
    pub fn set_field(&mut self, id: InstId, index: usize, field: InstField) {
        assert!(index < self.fields(id).len(), "field index out of bounds");
        self.fields.data[self.instructions[id].fields.start as usize + index] = field;
    }
    pub fn set_memory(&mut self, id: InstId, access: Option<MemoryAccess>) {
        self.memory.remove(self.instructions[id].memory);
        self.instructions[id].memory = access.map_or(NONE, |a| self.memory.insert(a));
    }
    pub fn effects(&self, id: InstId) -> Option<&RegEffects> {
        self.effects.get(self.instructions[id].effects)
    }
    pub fn set_effects(&mut self, id: InstId, effects: RegEffects) {
        assert!(effects.uses.iter().chain(&effects.defs).all(Reg::is_preg));
        self.clear_refs(id);
        self.effects.remove(self.instructions[id].effects);
        self.instructions[id].effects = if effects.uses.is_empty() && effects.defs.is_empty() {
            NONE
        } else {
            self.effects.insert(effects)
        };
        self.index_refs(id);
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
            RefLocation::ImplicitUse(index) => {
                self.effects(site.inst).unwrap().uses[index as usize]
            }
            RefLocation::ImplicitDef(index) => {
                self.effects(site.inst).unwrap().defs[index as usize]
            }
            RefLocation::Result(index) => self.results(site.inst)[index as usize],
            RefLocation::Input(index) => self.inputs(site.inst)[index as usize],
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
        let results = self
            .results(id)
            .iter()
            .copied()
            .enumerate()
            .map(move |(index, reg)| {
                (
                    reg,
                    Site::new(id, RefLocation::Result(index as u32), RefRole::Def),
                )
            });
        let operands = self
            .inputs(id)
            .iter()
            .copied()
            .enumerate()
            .map(move |(index, reg)| {
                (
                    reg,
                    Site::new(id, RefLocation::Input(index as u32), RefRole::Use),
                )
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
        let implicit = self.effects(id).into_iter().flat_map(move |effects| {
            effects
                .uses
                .iter()
                .copied()
                .enumerate()
                .map(move |(i, r)| {
                    (
                        r,
                        Site::new(id, RefLocation::ImplicitUse(i as u32), RefRole::Use),
                    )
                })
                .chain(effects.defs.iter().copied().enumerate().map(move |(i, r)| {
                    (
                        r,
                        Site::new(id, RefLocation::ImplicitDef(i as u32), RefRole::Def),
                    )
                }))
        });
        results.chain(operands).chain(edges).chain(implicit)
    }
    fn index_refs(&mut self, id: InstId) {
        let sites: smallvec::SmallVec<[_; 8]> = self.reference_sites(id).collect();
        let owner = Site::new(id, RefLocation::Input(0), RefRole::Use);
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
                RefLocation::Result(_)
                | RefLocation::ImplicitUse(_)
                | RefLocation::ImplicitDef(_) => {
                    unreachable!("virtual use cannot reference a result or physical effect")
                }
                RefLocation::Input(index) => {
                    let start = self.instructions[site.inst].inputs.start as usize;
                    self.inputs.data[start + index as usize] = new;
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
    use crate::InstBuild;
    use crate::{BranchInfo, MachineFunction, MemoryKind, Reg, Writable};

    #[test]
    fn pooled_storage_recycles_payloads_and_preserves_borrowed_views() {
        assert!(core::mem::size_of::<StoredInst>() <= 56);
        let mut f = MachineFunction::new("store".into());
        let block = f.create_synthetic_block();
        let reg = f.alloc_vreg(crate::Type::I64);
        let id = f.writer().constant(Writable(reg), 42);
        f.append_inst_id_to_block(0, id);
        let view = f.inst(id);
        assert_eq!(view.inputs().as_ptr(), f.inst(id).inputs().as_ptr());
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
        let target = f.writer().write(MachineOpcode::Target(7), &[], &[reg], &[]);
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
        let operands = f.inst(replacement).inputs().as_ptr();
        f.replace_inst(id, replacement);
        assert_eq!(f.inst(id).inputs().as_ptr(), operands);
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
        cloned.set_inst_inputs(target, &[Reg::new_preg(1)]);
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
        let id = store.write(MachineOpcode::Target(1), &[], &[], &[], None);
        for n in 0..100 {
            store.set_fields(id, &[]);
            store.write_at(
                id,
                MachineOpcode::Target(1),
                &[],
                &[reg],
                &[InstField::Imm(n)],
                None,
            );
            store.set_memory(id, Some(MemoryAccess::new(MemoryKind::Read, 8)));
            store.set_extra(
                id,
                InstExtra::Branch(BranchInfo {
                    args: Default::default(),
                }),
            );
            store.write_at(id, MachineOpcode::Invalid, &[], &[], &[], None);
        }
        assert_eq!(store.fields.data.len(), 1);
        assert_eq!(store.memory.slots.len(), 1);
        assert_eq!(store.extras.slots.len(), 1);
    }
}
