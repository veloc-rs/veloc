//! Function-owned instruction storage. IDs are stable; operand ranges and cold
//! payloads belong directly to InstId. Operand ranges are recycled on replacement.
use crate::FieldValue;
use crate::use_def::{Owner, References};
use crate::{InstId, InstRef, MachineOpcode, MemoryAccess};
use crate::{OperandId, RefRole, Reg, RegRefs, VReg};
use alloc::vec::Vec;
use cranelift_entity::{PrimaryMap, SecondaryMap};
use smallvec::SmallVec;
use veloc_collections::LinkId;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Range {
    start: u32,
    len: u32,
}

impl Range {
    fn ids(self) -> impl Iterator<Item = LinkId> {
        (self.start..self.start + self.len).map(LinkId::from_u32)
    }
    fn at(self, index: usize) -> OperandId {
        assert!(index < self.len as usize, "operand index out of bounds");
        OperandId::from_u32(self.start + index as u32)
    }
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
        self.insert_pair(values, &[])
    }
    fn insert_pair(&mut self, values: &[T], tail: &[T]) -> Range {
        let len = values
            .len()
            .checked_add(tail.len())
            .expect("operand count overflow");
        if len == 0 {
            return Range::default();
        }
        let capacity = len
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
            self.data.resize(
                end,
                values.first().or_else(|| tail.first()).unwrap().clone(),
            );
            start
        });
        let range = Range {
            start,
            len: len.try_into().expect("operand count overflow"),
        };
        let (first, second) = self.data[range.indices()].split_at_mut(values.len());
        first.clone_from_slice(values);
        second.clone_from_slice(tail);
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
pub struct RegEffects<R = Vec<Reg>> {
    pub uses: R,
    pub defs: R,
}

/// One allocation: explicit operands followed by implicit physical registers.
/// Store the boundary locally so storage need not depend on target descriptors.
#[derive(Debug, Clone, Copy, Default)]
struct RegRange {
    all: Range,
    explicit: u32,
}
impl RegRange {
    fn explicit(self) -> Range {
        Range {
            start: self.all.start,
            len: self.explicit,
        }
    }
    fn implicit(self) -> Range {
        Range {
            start: self.all.start + self.explicit,
            len: self.all.len - self.explicit,
        }
    }
}

#[derive(Debug, Clone)]
struct StoredInst {
    opcode: MachineOpcode,
    inputs: RegRange,
    fields: crate::Fields,
    results: RegRange,
}

#[derive(Debug, Clone)]
struct StoredEdge {
    owner: Option<InstId>,
    block: crate::BlockId,
    args: Range,
}

#[derive(Debug, Clone, Default)]
pub struct InstStore {
    instructions: PrimaryMap<InstId, StoredInst>,
    registers: Operands<Reg>,
    fields: crate::FieldPools,
    // Access facts are directly indexed; call contracts live in common fields.
    memory: SecondaryMap<InstId, Option<MemoryAccess>>,
    edges: PrimaryMap<crate::EdgeId, Option<StoredEdge>>,
    pub(crate) references: References,
}

/// Instruction construction for selection rules. Replacement/erasure must go
/// through the function editor. Replacement edges are copied during construction;
/// the selector transfers their identities only when committing the replacement.
pub struct InstBuilder<'a> {
    pub(crate) edge_transfers: Vec<(crate::EdgeId, crate::EdgeId)>,
    pub(crate) store: &'a mut InstStore,
    pub(crate) changes: Option<&'a mut crate::EditChanges>,
}
impl InstBuilder<'_> {
    pub fn into_edge_transfers(self) -> Vec<(crate::EdgeId, crate::EdgeId)> {
        self.edge_transfers
    }
    /// Prepare an independent edge for a replacement; ownership of the source
    /// is unchanged until the selector commits its explicit transfer list.
    pub fn replacement_edge(&mut self, source: InstId, id: crate::EdgeId) -> crate::EdgeId {
        assert_eq!(
            self.store.edges[id].as_ref().expect("deleted edge").owner,
            Some(source)
        );
        assert!(
            !self.edge_transfers.iter().any(|&(old, _)| old == id),
            "edge transferred twice"
        );
        let copy = self.store.clone_edge(id);
        self.edge_transfers.push((id, copy));
        copy
    }
    /// Find a virtual value's unique defining instruction. This is a read-only
    /// SSA query, not permission to move, fold or erase the definition.
    pub fn def(&self, reg: Reg) -> Option<InstId> {
        if !reg.is_vreg() {
            return None;
        }
        Some(self.store.defs(reg).single()?.inst())
    }

    pub fn get(&self, id: InstId) -> InstRef<'_> {
        self.store.get(id)
    }
    pub fn writer(&mut self) -> InstWriter<'_> {
        self.store.writer().tracking(self.changes.as_deref_mut())
    }
}

/// A single committed write. Generated methods encode directly from their typed
/// arguments; no owning instruction or temporary operand vector is required.
pub struct InstWriter<'a> {
    changes: Option<&'a mut crate::EditChanges>,
    store: &'a mut InstStore,
    target: Option<InstId>,
    memory: Option<crate::MemoryAccess>,
    effects: RegEffects<&'a [Reg]>,
}

impl<'a> InstWriter<'a> {
    pub fn edge(&mut self, block: crate::BlockId, args: &[Reg]) -> crate::EdgeId {
        self.store.create_edge(block, args)
    }
    pub(crate) fn tracking(mut self, changes: Option<&'a mut crate::EditChanges>) -> Self {
        self.changes = changes;
        self
    }

    pub fn with_effects(mut self, uses: &'a [Reg], defs: &'a [Reg]) -> Self {
        assert!(
            uses.iter().chain(defs).all(Reg::is_preg),
            "implicit effects require physical registers"
        );
        self.effects = RegEffects { uses, defs };
        self
    }
    pub fn with_memory(mut self, access: crate::MemoryAccess) -> Self {
        self.memory = Some(access);
        self
    }

    /// Convert transient positional fields at a low-level adapter boundary.
    pub fn write(
        self,
        opcode: crate::MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: impl IntoIterator<Item = FieldValue>,
    ) -> InstId {
        let fields = self.store.fields.pack(fields);
        let implicit = RegEffects {
            uses: self.effects.uses,
            defs: self.effects.defs,
        };
        let id = match self.target {
            Some(id) => {
                self.store.write_full_at(
                    id,
                    opcode,
                    results,
                    inputs,
                    fields,
                    self.memory,
                    implicit,
                );
                id
            }
            None => self
                .store
                .write_full(opcode, results, inputs, fields, self.memory, implicit),
        };
        if let Some(changes) = self.changes {
            changes.insts.push(id);
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
        fields: impl IntoIterator<Item = FieldValue>,
    ) -> InstId {
        self.write(MachineOpcode::Generic(opcode), results, inputs, fields)
    }
}

impl InstStore {
    pub(crate) fn with_capacity(insts: usize) -> Self {
        Self {
            instructions: PrimaryMap::with_capacity(insts),
            registers: Operands::default(),
            fields: crate::FieldPools::default(),
            memory: SecondaryMap::with_capacity(insts),
            edges: PrimaryMap::new(),
            references: References::default(),
        }
    }

    pub fn writer(&mut self) -> crate::InstWriter<'_> {
        crate::InstWriter {
            store: self,
            changes: None,
            target: None,
            memory: None,
            effects: RegEffects::default(),
        }
    }

    pub fn rewriter(&mut self, id: InstId) -> InstWriter<'_> {
        InstWriter {
            store: self,
            changes: None,
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
    pub(crate) fn registers(&self, range: Range) -> &[Reg] {
        &self.registers.data[range.indices()]
    }
    pub fn operand(&self, id: OperandId) -> Reg {
        self.registers.data[id.as_u32() as usize]
    }
    pub fn input_id(&self, id: InstId, index: usize) -> OperandId {
        self.instructions[id].inputs.explicit().at(index)
    }
    pub fn result_id(&self, id: InstId, index: usize) -> OperandId {
        self.instructions[id].results.explicit().at(index)
    }
    pub fn results(&self, id: InstId) -> &[Reg] {
        self.registers(self.instructions[id].results.explicit())
    }
    pub fn inputs(&self, id: InstId) -> &[Reg] {
        self.registers(self.instructions[id].inputs.explicit())
    }
    pub fn fields(&self, id: InstId) -> crate::FieldView<'_> {
        self.fields.view(&self.instructions[id].fields)
    }
    fn set_operand(&mut self, id: OperandId, reg: Reg) {
        let link = id.link();
        let owner = self.references.links.owner(link);
        let old = self.operand(id);
        if old == reg {
            return;
        }
        self.references.detach(link, old);
        self.registers.data[id.as_u32() as usize] = reg;
        self.references.attach(link, reg, owner);
    }
    pub fn set_results(&mut self, id: InstId, results: &[Reg]) {
        if self.results(id).len() == results.len() {
            for (index, &reg) in results.iter().enumerate() {
                self.set_result(id, index, reg);
            }
            return;
        }
        let implicit = self.implicit_defs(id).to_vec();
        self.release_registers(self.instructions[id].results.all);
        self.instructions[id].results = self.alloc_group(id, RefRole::Def, results, &implicit);
    }
    pub fn set_result(&mut self, id: InstId, index: usize, reg: Reg) {
        self.set_operand(self.result_id(id, index), reg);
    }
    pub fn set_input(&mut self, id: InstId, index: usize, reg: Reg) {
        self.set_operand(self.input_id(id, index), reg);
    }
    pub fn set_inputs(&mut self, id: InstId, inputs: &[Reg]) {
        if self.inputs(id).len() == inputs.len() {
            for (index, &reg) in inputs.iter().enumerate() {
                self.set_input(id, index, reg);
            }
            return;
        }
        let implicit = self.implicit_uses(id).to_vec();
        self.release_registers(self.instructions[id].inputs.all);
        self.instructions[id].inputs = self.alloc_group(id, RefRole::Use, inputs, &implicit);
    }
    pub fn memory(&self, id: InstId) -> Option<MemoryAccess> {
        self.memory[id]
    }
    fn write_full(
        &mut self,
        opcode: MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
        memory: Option<MemoryAccess>,
        implicit: RegEffects<&[Reg]>,
    ) -> InstId {
        let id = self.instructions.next_key();
        self.check_edges(id, &fields);
        let inputs = self.alloc_group(id, RefRole::Use, inputs, implicit.uses);
        self.attach_edges(id, &fields);
        let results = self.alloc_group(id, RefRole::Def, results, implicit.defs);
        let id = self.instructions.push(StoredInst {
            opcode,
            inputs,
            fields,
            results,
        });
        if memory.is_some() {
            self.memory[id] = memory;
        }
        id
    }
    /// Transfer a detached instruction into a stable destination ID.
    pub fn replace(&mut self, id: InstId, source: InstId) {
        if id == source {
            return;
        }
        self.clear(id);
        let empty = StoredInst {
            opcode: MachineOpcode::Invalid,
            inputs: RegRange::default(),
            fields: crate::Fields::None,
            results: RegRange::default(),
        };
        self.instructions[id] = core::mem::replace(&mut self.instructions[source], empty);
        if let Some(access) = self.memory[source] {
            self.memory[source] = None;
            self.memory[id] = Some(access);
        }
        let edge_ids: Vec<_> = self.edge_ids(id).collect();
        for edge in edge_ids {
            self.edges[edge].as_mut().unwrap().owner = Some(id);
        }
        let links: Vec<_> = self
            .operand_ranges(id)
            .flat_map(|(range, _)| range.ids())
            .collect();
        for link in links {
            let mut site = self.references.links.owner(link);
            site.inst = id;
            self.references.links.set_owner(link, site);
        }
    }
    pub(crate) fn clear(&mut self, id: InstId) {
        self.write_full_at(
            id,
            MachineOpcode::Invalid,
            &[],
            &[],
            crate::Fields::None,
            None,
            RegEffects {
                uses: &[],
                defs: &[],
            },
        );
    }
    fn write_full_at(
        &mut self,
        id: InstId,
        opcode: MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
        memory: Option<MemoryAccess>,
        implicit: RegEffects<&[Reg]>,
    ) {
        self.check_edges(id, &fields);
        let removed: Vec<_> = self
            .edge_ids(id)
            .filter(|edge| !self.fields.view(&fields).successors().contains(edge))
            .collect();
        for edge in removed {
            self.delete_edge(edge);
        }
        self.attach_edges(id, &fields);
        self.release_registers(self.instructions[id].inputs.all);
        self.release_registers(self.instructions[id].results.all);
        self.fields
            .remove(core::mem::take(&mut self.instructions[id].fields));
        self.instructions[id].inputs = self.alloc_group(id, RefRole::Use, inputs, implicit.uses);
        self.instructions[id].results = self.alloc_group(id, RefRole::Def, results, implicit.defs);
        self.instructions[id].fields = fields;
        self.set_memory(id, memory);
        self.instructions[id].opcode = opcode;
    }
    pub fn set_memory(&mut self, id: InstId, access: Option<MemoryAccess>) {
        if access.is_some() || self.memory[id].is_some() {
            self.memory[id] = access;
        }
    }
    pub fn implicit_uses(&self, id: InstId) -> &[Reg] {
        self.registers(self.instructions[id].inputs.implicit())
    }
    pub fn implicit_defs(&self, id: InstId) -> &[Reg] {
        self.registers(self.instructions[id].results.implicit())
    }
    pub fn effects(&self, id: InstId) -> Option<RegEffects<&[Reg]>> {
        let uses = self.implicit_uses(id);
        let defs = self.implicit_defs(id);
        (!uses.is_empty() || !defs.is_empty()).then_some(RegEffects { uses, defs })
    }
    pub fn set_effects(&mut self, id: InstId, effects: RegEffects) {
        assert!(effects.uses.iter().chain(&effects.defs).all(Reg::is_preg));
        let inputs = self.inputs(id).to_vec();
        let results = self.results(id).to_vec();
        self.release_registers(self.instructions[id].inputs.all);
        self.release_registers(self.instructions[id].results.all);
        self.instructions[id].inputs = self.alloc_group(id, RefRole::Use, &inputs, &effects.uses);
        self.instructions[id].results = self.alloc_group(id, RefRole::Def, &results, &effects.defs);
    }
    pub fn call_info(&self, id: InstId) -> Option<&crate::CallInfo> {
        self.fields(id).call_info()
    }
    pub fn edge_ids(&self, id: InstId) -> impl Iterator<Item = crate::EdgeId> + '_ {
        self.fields(id).successors().iter().copied()
    }
    pub fn edge(&self, id: crate::EdgeId) -> crate::Successor<&[Reg]> {
        let edge = self.edges[id].as_ref().expect("deleted edge");
        crate::Successor {
            block: edge.block,
            args: self.registers(edge.args),
        }
    }
    pub fn create_edge(&mut self, block: crate::BlockId, args: &[Reg]) -> crate::EdgeId {
        let args = self.registers.insert(args);
        self.edges.push(Some(StoredEdge {
            owner: None,
            block,
            args,
        }))
    }
    /// Explicit duplication: a copy has a fresh identity and independent arguments.
    pub fn clone_edge(&mut self, id: crate::EdgeId) -> crate::EdgeId {
        let edge = self.edge(id);
        let (block, args) = (edge.block, edge.args.to_vec());
        self.create_edge(block, &args)
    }

    fn check_edges(&self, owner: InstId, fields: &crate::Fields) {
        // Ordinary branches need no allocation; tables use a set to avoid
        // quadratic duplicate detection.
        let mut seen = hashbrown::HashSet::new();
        let successors = self.fields.view(fields).successors();
        for (index, &id) in successors.iter().enumerate() {
            let unique = if successors.len() > 2 {
                seen.insert(id)
            } else {
                !successors[..index].contains(&id)
            };
            assert!(unique, "duplicate edge");
            let edge = self.edges[id].as_ref().expect("deleted edge");
            assert!(
                edge.owner.is_none() || edge.owner == Some(owner),
                "edge already belongs to another instruction; clone it explicitly"
            );
        }
    }

    fn attach_edges(&mut self, owner: InstId, fields: &crate::Fields) {
        for index in 0..self.fields.view(fields).successors().len() {
            let id = self.fields.view(fields).successors()[index];
            let edge = self.edges[id].as_mut().unwrap();
            if edge.owner == Some(owner) {
                continue;
            }
            edge.owner = Some(owner);
            let args = edge.args;
            self.attach_range(owner, RefRole::Use, args);
        }
    }

    /// Exchange identities at replacement commit. Both instructions remain
    /// internally consistent; deleting the old instruction deletes the copy.
    pub(crate) fn transfer_edge(
        &mut self,
        original: crate::EdgeId,
        replacement: crate::EdgeId,
    ) -> (InstId, InstId) {
        assert_ne!(original, replacement);
        let from = self.edges[original]
            .as_ref()
            .expect("deleted edge")
            .owner
            .expect("unattached edge");
        let to = self.edges[replacement]
            .as_ref()
            .expect("deleted edge")
            .owner
            .expect("unattached edge");
        assert_ne!(from, to);
        let a = self.edge_ids(from).position(|id| id == original).unwrap();
        let b = self.edge_ids(to).position(|id| id == replacement).unwrap();
        self.fields
            .successors_mut(&mut self.instructions[from].fields)[a] = replacement;
        self.fields
            .successors_mut(&mut self.instructions[to].fields)[b] = original;
        let old = self.edges[original].take();
        self.edges[original] = self.edges[replacement].take();
        self.edges[replacement] = old;
        (from, to)
    }

    pub fn successors(&self, id: InstId) -> impl Iterator<Item = crate::Successor<&[Reg]>> {
        self.edge_ids(id).map(|edge| self.edge(edge))
    }
    pub fn edge_args(&self, id: InstId) -> impl Iterator<Item = Reg> + '_ {
        self.successors(id)
            .flat_map(|edge| edge.args.iter().copied())
    }
    pub(crate) fn redirect_edge(&mut self, id: crate::EdgeId, target: crate::BlockId) -> InstId {
        let edge = self.edges[id].as_mut().expect("deleted edge");
        let owner = edge.owner.expect("edge is not attached to an instruction");
        edge.block = target;
        owner
    }

    pub(crate) fn set_edge_args(&mut self, id: crate::EdgeId, args: &[Reg]) -> InstId {
        let edge = self.edges[id].as_ref().expect("deleted edge");
        let owner = edge.owner.expect("edge is not attached to an instruction");
        if self.registers(edge.args) == args {
            return owner;
        }
        let old = edge.args;
        self.release_registers(old);
        let range = self.alloc_registers(owner, RefRole::Use, args);
        self.edges[id].as_mut().unwrap().args = range;
        owner
    }
    pub fn clear_successor_args(&mut self, id: InstId) {
        let ids: SmallVec<[_; 2]> = self.edge_ids(id).collect();
        for edge_id in ids {
            let edge = self.edges[edge_id].as_mut().unwrap();
            let args = core::mem::take(&mut edge.args);
            self.release_registers(args);
        }
    }
    fn delete_edge(&mut self, id: crate::EdgeId) {
        let edge = self.edges[id].take().expect("deleted edge");
        self.release_registers(edge.args);
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
    fn alloc_group(
        &mut self,
        inst: InstId,
        role: RefRole,
        explicit: &[Reg],
        implicit: &[Reg],
    ) -> RegRange {
        let range = self.registers.insert_pair(explicit, implicit);
        self.attach_range(inst, role, range);
        RegRange {
            all: range,
            explicit: explicit.len().try_into().expect("operand count overflow"),
        }
    }
    fn alloc_registers(&mut self, inst: InstId, role: RefRole, regs: &[Reg]) -> Range {
        let range = self.registers.insert(regs);
        self.attach_range(inst, role, range);
        range
    }
    fn attach_range(&mut self, inst: InstId, role: RefRole, range: Range) {
        let owner = Owner { inst, role };
        self.references
            .links
            .resize(self.registers.data.len(), owner);
        for link in range.ids() {
            let reg = self.registers.data[link.as_u32() as usize];
            self.references.attach(link, reg, owner);
        }
    }
    fn release_registers(&mut self, range: Range) {
        for link in range.ids() {
            self.references
                .detach(link, self.registers.data[link.as_u32() as usize]);
        }
        self.registers.release(range);
    }
    fn operand_ranges(&self, id: InstId) -> impl Iterator<Item = (Range, RefRole)> + '_ {
        let inst = &self.instructions[id];
        [
            (inst.inputs.all, RefRole::Use),
            (inst.results.all, RefRole::Def),
        ]
        .into_iter()
        .chain(
            self.edge_ids(id)
                .map(|edge| (self.edges[edge].as_ref().unwrap().args, RefRole::Use)),
        )
    }

    /// Replace virtual uses directly by slot identity, including edge arguments.
    pub fn replace_uses(&mut self, old: VReg, new: VReg) {
        if old == new {
            return;
        }
        let old = Reg::new_vreg(old.as_u32());
        let new = Reg::new_vreg(new.as_u32());
        while let Some(link) = self.references.head(old, RefRole::Use).expand() {
            self.set_operand(OperandId::from_u32(link.as_u32()), new);
        }
    }
    /// Explicit audit of slot ownership, range allocation and register chains.
    pub fn check_refs(&self) -> Result<(), &'static str> {
        let mut allocated = alloc::vec![false; self.registers.data.len()];
        let mut expected = hashbrown::HashMap::new();
        let mut seen_edges = hashbrown::HashSet::new();
        let mut mark = |start: usize, len: usize| -> Result<(), &'static str> {
            let end = start.checked_add(len).ok_or("operand range overflow")?;
            for slot in allocated
                .get_mut(start..end)
                .ok_or("operand range out of bounds")?
            {
                if core::mem::replace(slot, true) {
                    return Err("overlapping operand ranges");
                }
            }
            Ok(())
        };
        for (inst, data) in self.instructions.iter() {
            for edge in self.edge_ids(inst) {
                if !seen_edges.insert(edge) {
                    return Err("successor referenced more than once");
                }
                if self
                    .edges
                    .get(edge)
                    .and_then(Option::as_ref)
                    .is_none_or(|edge| edge.owner != Some(inst))
                {
                    return Err("incorrect successor owner or deleted edge");
                }
            }
            for group in [data.inputs, data.results] {
                if group.explicit > group.all.len {
                    return Err("invalid explicit operand boundary");
                }
            }
            if !self
                .implicit_uses(inst)
                .iter()
                .chain(self.implicit_defs(inst))
                .all(Reg::is_preg)
            {
                return Err("implicit operand is not a physical register");
            }
            for (range, role) in self.operand_ranges(inst) {
                if range.len == 0 {
                    continue;
                }
                mark(range.start as usize, range.len.next_power_of_two() as usize)?;
                for link in range.ids() {
                    if self.references.links.owner(link) != (Owner { inst, role }) {
                        return Err("incorrect operand owner or role");
                    }
                    expected.insert(
                        link.as_u32(),
                        (self.registers.data[link.as_u32() as usize], role),
                    );
                }
            }
        }
        for (id, edge) in self.edges.iter() {
            if edge.is_some() && !seen_edges.contains(&id) {
                return Err("successor is not attached to an instruction");
            }
        }
        for (class, ranges) in self.registers.free.iter().enumerate() {
            for &start in ranges {
                mark(start as usize, 1 << class)?;
            }
        }
        if allocated.iter().any(|&a| !a) {
            return Err("lost operand range");
        }
        for (reg, role, mut next) in self.references.heads() {
            let mut prev = None.into();
            while let Some(link) = next.expand() {
                if expected.remove(&link.as_u32()) != Some((reg, role)) {
                    return Err("unexpected, duplicate or cyclic operand reference");
                }
                if self.references.links.prev(link) != prev {
                    return Err("incorrect previous link");
                }
                prev = Some(link).into();
                next = self.references.links.next(link);
            }
        }
        if !expected.is_empty() {
            return Err("unlinked live operand");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InstBuild;
    use crate::{MachineFunction, MemoryKind, Reg, Writable};

    #[test]
    fn storage_preserves_views_and_transfers_instruction_properties() {
        assert!(core::mem::size_of::<StoredInst>() <= 44);
        let mut f = MachineFunction::new("store".into());
        let block = f.editor().create_block();
        let reg = f.editor().alloc_vreg(crate::Type::I64);
        let id = f.editor().writer().constant(Writable(reg), 42);
        f.editor().append_inst(crate::BlockId::from_u32(0), id);
        let view = f.inst(id);
        assert_eq!(view.inputs().as_ptr(), f.inst(id).inputs().as_ptr());
        for _ in 0..100 {
            let access = MemoryAccess::new(MemoryKind::Read, 8);
            assert_eq!(
                f.editor()
                    .rewriter(id)
                    .with_memory(access)
                    .load(Writable(reg), reg, 16),
                id
            );
            assert_eq!(f.inst(id).memory(), Some(access));
            assert_eq!(f.editor().rewriter(id).constant(Writable(reg), 42), id);
            assert!(f.inst(id).memory().is_none());
            assert!(f.try_call_info(id).is_none());
        }
        // The generic and target namespaces use the exact same store.
        let target = f
            .editor()
            .writer()
            .write(MachineOpcode::Target(7), &[], &[reg], []);
        f.editor().append_inst(crate::BlockId::from_u32(0), target);
        assert!(f.inst(id).is_generic());
        assert!(f.inst(target).is_target());
        let access = MemoryAccess::new(MemoryKind::Read, 8);
        let replacement = f
            .editor()
            .writer()
            .with_memory(access)
            .load(Writable(reg), reg, 0);
        let operands = f.inst(replacement).inputs().as_ptr();
        f.editor().replace_inst(id, replacement);
        assert_eq!(f.inst(id).inputs().as_ptr(), operands);
        assert_eq!(f.inst(id).memory(), Some(access));
        assert!(f.inst(replacement).is_invalid());
        assert!(f.inst(replacement).memory().is_none());
        assert!(f.try_call_info(replacement).is_none());
        // Worklist rewrites detach and reinsert IDs without erasing their data.
        for id in f.block_insts(block).collect::<Vec<_>>() {
            let mut edit = f.editor();
            edit.detach_inst(id);
            edit.append_inst(block, id);
        }
        assert_eq!(
            f.block_insts(crate::BlockId::from_u32(0))
                .collect::<Vec<_>>(),
            &[id, target]
        );
        assert_eq!(f.inst(id).memory(), Some(access));
        let mut cloned = f.clone();
        cloned.editor().set_inst_inputs(target, &[Reg::new_preg(1)]);
        assert_eq!(f.inst(target).uses().collect::<Vec<_>>(), [reg]);
        assert_eq!(
            cloned.inst(target).uses().collect::<Vec<_>>(),
            [Reg::new_preg(1)]
        );
        for id in f.block_insts(block).collect::<Vec<_>>() {
            f.editor().invalidate_inst(id);
        }
        assert!(f.inst(id).is_invalid());
        assert!(f.inst(target).is_invalid());
        assert!(
            f.block_insts(crate::BlockId::from_u32(0))
                .collect::<Vec<_>>()
                .is_empty()
        );
        assert_eq!(f.blocks().nth(0).unwrap(), block);

        let mut store = InstStore::default();
        let id = store.writer().write(MachineOpcode::Target(1), &[], &[], []);
        for n in 0..100 {
            store
                .rewriter(id)
                .write(MachineOpcode::Target(1), &[], &[reg], [FieldValue::Imm(n)]);
            store.set_memory(id, Some(MemoryAccess::new(MemoryKind::Read, 8)));
            store.clear(id);
        }
        assert!(store.fields(id).is_empty());
        assert!(store.memory(id).is_none());
        assert!(store.effects(id).is_none());
        let source = store.writer().write(MachineOpcode::Target(2), &[], &[], []);
        store.set_effects(
            source,
            RegEffects {
                uses: alloc::vec![Reg::new_preg(1)],
                defs: alloc::vec![Reg::new_preg(2)],
            },
        );
        store.replace(id, source);
        assert_eq!(store.effects(id).unwrap().uses, [Reg::new_preg(1)]);
        assert_eq!(store.effects(id).unwrap().defs, [Reg::new_preg(2)]);
        assert!(store.effects(source).is_none());
        store.check_refs().unwrap();
        store.clear(id);
        assert!(store.effects(id).is_none());
        store.check_refs().unwrap();

        // One range per direction, with explicit APIs excluding the suffix.
        let input = Reg::new_vreg(7);
        let output = Reg::new_vreg(8);
        let physical = Reg::new_preg(1);
        let combined = store.writer().with_effects(&[physical], &[physical]).write(
            MachineOpcode::Target(3),
            &[output],
            &[input],
            [],
        );
        assert_eq!(store.inputs(combined), &[input]);
        assert_eq!(store.results(combined), &[output]);
        assert_eq!(store.implicit_uses(combined), &[physical]);
        assert_eq!(store.implicit_defs(combined), &[physical]);
        assert_eq!(
            store.inputs(combined).as_ptr().wrapping_add(1),
            store.implicit_uses(combined).as_ptr()
        );
        assert_eq!(
            store.results(combined).as_ptr().wrapping_add(1),
            store.implicit_defs(combined).as_ptr()
        );
        assert_eq!(
            store.get(combined).uses().collect::<Vec<_>>(),
            [input, physical]
        );
        // Reshaping explicit results must preserve the implicit suffix.
        store.set_results(combined, &[output, input]);
        assert_eq!(store.results(combined), &[output, input]);
        assert_eq!(store.implicit_defs(combined), &[physical]);
        store.check_refs().unwrap();
        store.set_effects(combined, RegEffects::default());
        assert!(store.implicit_uses(combined).is_empty());
        assert!(store.implicit_defs(combined).is_empty());
        assert_eq!(store.results(combined), &[output, input]);
        assert_eq!(store.inputs(combined), &[input]);
        store.check_refs().unwrap();
        store
            .rewriter(combined)
            .with_effects(&[physical], &[physical])
            .write(MachineOpcode::Target(4), &[], &[], []);
        assert!(store.inputs(combined).is_empty());
        assert!(store.results(combined).is_empty());
        assert_eq!(store.get(combined).uses().collect::<Vec<_>>(), [physical]);
        store.clear(combined);
        assert_eq!(store.uses(physical).count(), 0);
        assert_eq!(store.defs(physical).count(), 0);
        store.check_refs().unwrap();
    }
}
