//! Function-owned instruction storage. IDs are stable; operand ranges and cold
//! payloads belong directly to InstId. Operand ranges are recycled on replacement.
use super::use_def::{Owner, References};
use crate::{InstId, InstRef, MachineOpcode};
use crate::{OperandId, RefRole, Reg, RegRefs, VReg};
use alloc::vec::Vec;
use cranelift_entity::PrimaryMap;
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

#[derive(Debug, Clone, Default)]
struct OperandStorage {
    data: Vec<Reg>,
    free: Vec<Vec<u32>>,
}

impl OperandStorage {
    fn insert(&mut self, values: &[Reg]) -> Range {
        let len = values.len();
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
            self.data.resize(end, values[0]);
            start
        });
        let range = Range {
            start,
            len: len.try_into().expect("operand count overflow"),
        };
        self.data[range.indices()].copy_from_slice(values);
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

#[derive(Debug)]
struct StoredInst {
    opcode: MachineOpcode,
    inputs: Range,
    fields: crate::Fields,
    results: Range,
}

// Pool handles can be duplicated only as part of a full InstStore clone.
impl Clone for StoredInst {
    fn clone(&self) -> Self {
        Self {
            opcode: self.opcode,
            inputs: self.inputs,
            fields: self.fields.clone_handle(),
            results: self.results,
        }
    }
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
    registers: OperandStorage,
    fields: crate::FieldPools,
    constraints: hashbrown::HashMap<InstId, Vec<crate::OperandConstraint>>,
    clobbers: hashbrown::HashMap<InstId, SmallVec<[Reg; 4]>>,
    edges: PrimaryMap<crate::EdgeId, Option<StoredEdge>>,
    pub(crate) references: References,
}

impl InstStore {
    pub(crate) fn with_capacity(insts: usize) -> Self {
        Self {
            instructions: PrimaryMap::with_capacity(insts),
            registers: OperandStorage::default(),
            fields: crate::FieldPools::default(),
            constraints: hashbrown::HashMap::new(),
            clobbers: hashbrown::HashMap::new(),
            edges: PrimaryMap::new(),
            references: References::default(),
        }
    }

    pub(crate) fn call_fields(
        &mut self,
        target: Option<crate::SymbolId>,
        info: crate::CallInfo,
    ) -> crate::Fields {
        self.fields.call(target, info)
    }

    pub(crate) fn switch_fields(&mut self, edges: &[crate::EdgeId]) -> crate::Fields {
        self.fields.switch(edges)
    }

    pub(crate) fn copy_fields(&mut self, id: InstId) -> crate::Fields {
        let mut fields = self.fields.copy(&self.instructions[id].fields);
        let edges = self.fields.view(&fields).successors().to_vec();
        for (index, edge) in edges.into_iter().enumerate() {
            let copy = self.clone_edge(edge);
            self.fields.successors_mut(&mut fields)[index] = copy;
        }
        fields
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
        self.instructions[id].inputs.at(index)
    }
    pub fn result_id(&self, id: InstId, index: usize) -> OperandId {
        self.instructions[id].results.at(index)
    }
    pub fn results(&self, id: InstId) -> &[Reg] {
        self.registers(self.instructions[id].results)
    }
    pub fn inputs(&self, id: InstId) -> &[Reg] {
        self.registers(self.instructions[id].inputs)
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
        self.release_registers(self.instructions[id].results);
        self.instructions[id].results = self.alloc_registers(id, RefRole::Def, results);
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
        self.release_registers(self.instructions[id].inputs);
        self.instructions[id].inputs = self.alloc_registers(id, RefRole::Use, inputs);
    }
    pub(crate) fn write_full(
        &mut self,
        opcode: MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
        clobbers: &[Reg],
    ) -> InstId {
        let id = self.instructions.next_key();
        self.check_edges(id, &fields);
        let inputs = self.alloc_registers(id, RefRole::Use, inputs);
        self.attach_edges(id, &fields);
        let results = self.alloc_registers(id, RefRole::Def, results);
        let id = self.instructions.push(StoredInst {
            opcode,
            inputs,
            fields,
            results,
        });
        self.set_clobbers(id, clobbers);
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
            inputs: Range::default(),
            fields: crate::Fields::None,
            results: Range::default(),
        };
        self.instructions[id] = core::mem::replace(&mut self.instructions[source], empty);
        if let Some(constraints) = self.constraints.remove(&source) {
            self.constraints.insert(id, constraints);
        }
        if let Some(clobbers) = self.clobbers.remove(&source) {
            self.clobbers.insert(id, clobbers);
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
            &[],
        );
    }
    pub(crate) fn write_full_at(
        &mut self,
        id: InstId,
        opcode: MachineOpcode,
        results: &[Reg],
        inputs: &[Reg],
        fields: crate::Fields,
        clobbers: &[Reg],
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
        self.release_registers(self.instructions[id].inputs);
        self.release_registers(self.instructions[id].results);
        self.fields
            .remove(core::mem::take(&mut self.instructions[id].fields));
        self.instructions[id].inputs = self.alloc_registers(id, RefRole::Use, inputs);
        self.instructions[id].results = self.alloc_registers(id, RefRole::Def, results);
        self.instructions[id].fields = fields;
        self.set_clobbers(id, clobbers);
        self.instructions[id].opcode = opcode;
        self.constraints.remove(&id);
    }
    pub fn constraints(&self, id: InstId) -> &[crate::OperandConstraint] {
        self.constraints.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }
    pub fn set_constraints(&mut self, id: InstId, constraints: Vec<crate::OperandConstraint>) {
        if constraints.is_empty() {
            self.constraints.remove(&id);
        } else {
            self.constraints.insert(id, constraints);
        }
    }
    pub fn clobbers(&self, id: InstId) -> &[Reg] {
        self.clobbers
            .get(&id)
            .map(|regs| regs.as_slice())
            .unwrap_or(&[])
    }
    pub fn set_clobbers(&mut self, id: InstId, regs: &[Reg]) {
        assert!(
            regs.iter().all(Reg::is_preg),
            "clobbers require physical registers"
        );
        let abi = self.call_info(id).map(|info| info.clobbers);
        let mut clobbers = SmallVec::<[Reg; 4]>::new();
        for &reg in regs {
            // Keep ABI destruction in the shared mask when copying through
            // the unified clobbers() view.
            if !abi.is_some_and(|mask| mask.contains(reg.as_preg().unwrap()))
                && !clobbers.contains(&reg)
            {
                clobbers.push(reg);
            }
        }
        if clobbers.is_empty() {
            self.clobbers.remove(&id);
        } else {
            self.clobbers.insert(id, clobbers);
        }
    }
    pub fn call_info(&self, id: InstId) -> Option<&crate::CallInfo> {
        self.fields(id).call_info()
    }
    pub(crate) fn set_call_abi(
        &mut self,
        id: InstId,
        inputs: &[Reg],
        frame: crate::CallFrameId,
        clobbers: crate::RegMask,
        stack_args: smallvec::SmallVec<[crate::StackSlot; 2]>,
    ) {
        assert!(
            self.call_info(id).expect("call fields").frame.is_none(),
            "call already lowered"
        );
        // These updates preserve instruction clobbers and memory attributes.
        self.set_inputs(id, inputs);
        let info = self.fields.call_info_mut(&self.instructions[id].fields);
        info.frame = Some(frame);
        info.clobbers = clobbers;
        info.stack_args = stack_args;
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
        [(inst.inputs, RefRole::Use), (inst.results, RefRole::Def)]
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
        for inst in self.instructions.keys() {
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
    use crate::{MachineFunction, MemFlags, Reg};

    #[test]
    fn storage_preserves_views_and_transfers_instruction_properties() {
        assert!(core::mem::size_of::<StoredInst>() <= 44);
        let mut f = MachineFunction::new("store".into());
        let entry = f.entry_block();
        let block = f.editor().create_block();
        let reg = f.editor().alloc_vreg(crate::Type::I64);
        let id = f
            .editor()
            .at_end(crate::BlockId::from_u32(0))
            .writer()
            .constant(reg, 42);

        let view = f.inst(id);
        assert_eq!(view.inputs().as_ptr(), f.inst(id).inputs().as_ptr());
        for _ in 0..100 {
            let flags = MemFlags::new().with_alignment(8);
            assert_eq!(f.editor().replace(id).load(reg, reg, 16, flags), id);
            assert_eq!(f.inst(id).mem_flags(), Some(flags));
            assert_eq!(f.editor().replace(id).constant(reg, 42), id);
            assert!(f.inst(id).mem_flags().is_none());
            assert!(f.try_call_info(id).is_none());
        }
        // The generic and target namespaces use the exact same store.
        let target = f
            .editor()
            .at_end(crate::BlockId::from_u32(0))
            .writer()
            .write(MachineOpcode::Target(7), &[], &[reg], crate::Fields::None);

        assert!(f.inst(id).is_generic());
        assert!(f.inst(target).is_target());
        let flags = MemFlags::new().with_alignment(8);
        let replacement = f.editor().at_end(entry).writer().load(reg, reg, 0, flags);
        let operands = f.inst(replacement).inputs().as_ptr();
        f.editor().replace_inst(id, replacement);
        assert_eq!(f.inst(id).inputs().as_ptr(), operands);
        assert_eq!(f.inst(id).mem_flags(), Some(flags));
        assert!(f.inst(replacement).is_invalid());
        assert!(f.inst(replacement).mem_flags().is_none());
        assert!(f.try_call_info(replacement).is_none());
        // Moving instructions preserves their IDs and data.
        for id in f.block_insts(block).collect::<Vec<_>>() {
            let mut edit = f.editor();
            edit.at_end(block).move_here(id);
        }
        assert_eq!(
            f.block_insts(crate::BlockId::from_u32(0))
                .collect::<Vec<_>>(),
            &[id, target]
        );
        assert_eq!(f.inst(id).mem_flags(), Some(flags));
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
        let id = store.write_full(
            MachineOpcode::Target(1),
            &[],
            &[],
            crate::Fields::default(),
            &[],
        );
        for n in 0..100 {
            let fields = crate::Fields::Memory {
                offset: n,
                flags: MemFlags::new(),
            };
            store.write_full_at(id, MachineOpcode::Target(1), &[], &[reg], fields, &[]);
            store.clear(id);
        }
        assert!(store.fields(id).is_empty());
        assert!(store.fields(id).mem_flags().is_none());
        assert!(store.clobbers(id).is_empty());
        let source = store.write_full(
            MachineOpcode::Target(2),
            &[],
            &[],
            crate::Fields::default(),
            &[],
        );
        store.set_inputs(source, &[Reg::new_preg(1)]);
        store.set_clobbers(source, &[Reg::new_preg(2)]);
        store.replace(id, source);
        assert_eq!(store.inputs(id), [Reg::new_preg(1)]);
        assert_eq!(store.clobbers(id), [Reg::new_preg(2)]);
        assert!(store.clobbers(source).is_empty());
        store.check_refs().unwrap();
        store.clear(id);
        assert!(store.clobbers(id).is_empty());
        store.check_refs().unwrap();

        // Clobbers are independent of operand ranges and never define values.
        let input = Reg::new_vreg(7);
        let output = Reg::new_vreg(8);
        let physical = Reg::new_preg(1);
        let combined = store.write_full(
            MachineOpcode::Target(3),
            &[output],
            &[input, physical],
            crate::Fields::default(),
            &[physical],
        );
        assert_eq!(store.inputs(combined), &[input, physical]);
        assert_eq!(store.results(combined), &[output]);
        assert_eq!(store.clobbers(combined), &[physical]);
        assert_eq!(store.defs(physical).count(), 0);
        assert_eq!(
            store.get(combined).uses().collect::<Vec<_>>(),
            [input, physical]
        );
        store.set_results(combined, &[output, input]);
        assert_eq!(store.results(combined), &[output, input]);
        assert_eq!(store.clobbers(combined), &[physical]);
        store.check_refs().unwrap();
        store.set_clobbers(combined, &[]);
        assert!(store.clobbers(combined).is_empty());
        assert_eq!(store.results(combined), &[output, input]);
        assert_eq!(store.inputs(combined), &[input, physical]);
        store.check_refs().unwrap();
        store.write_full_at(
            combined,
            MachineOpcode::Target(4),
            &[],
            &[physical],
            crate::Fields::None,
            &[physical],
        );
        assert!(store.results(combined).is_empty());
        assert_eq!(store.get(combined).uses().collect::<Vec<_>>(), [physical]);
        store.clear(combined);
        assert_eq!(store.uses(physical).count(), 0);
        assert_eq!(store.defs(physical).count(), 0);
        store.check_refs().unwrap();
    }
}
