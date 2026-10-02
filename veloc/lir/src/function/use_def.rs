//! Register occurrences share the identity of their operand storage slot.
use crate::{InstId, InstStore, Reg};
use alloc::vec::Vec;
use cranelift_entity::packed_option::PackedOption;
use veloc_collections::{LinkId, Links};

type Head = PackedOption<LinkId>;

/// A function-local operand slot. Erasing or reshaping its range invalidates it.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperandId(u32);
cranelift_entity::entity_impl!(OperandId, "operand");
impl OperandId {
    pub(crate) fn link(self) -> LinkId {
        LinkId::from_u32(self.as_u32())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefRole {
    Use,
    Def,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Owner {
    pub inst: InstId,
    pub role: RefRole,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct References {
    pub links: Links<Owner>,
    virtual_heads: Vec<[Head; 2]>,
    physical_heads: Vec<[Head; 2]>,
}
impl References {
    pub fn heads(&self) -> impl Iterator<Item = (Reg, RefRole, Head)> + '_ {
        [(true, &self.virtual_heads), (false, &self.physical_heads)]
            .into_iter()
            .flat_map(|(virtual_, heads)| {
                heads.iter().enumerate().flat_map(move |(index, heads)| {
                    let reg = if virtual_ {
                        Reg::new_vreg(index as u32)
                    } else {
                        Reg::new_preg(index as u32)
                    };
                    [RefRole::Use, RefRole::Def]
                        .into_iter()
                        .filter_map(move |role| {
                            let head = heads[role as usize];
                            head.is_some().then_some((reg, role, head))
                        })
                })
            })
    }
    pub fn head(&self, reg: Reg, role: RefRole) -> Head {
        let heads = if reg.is_vreg() {
            &self.virtual_heads
        } else {
            &self.physical_heads
        };
        heads
            .get(reg.index() as usize)
            .map_or(None.into(), |h| h[role as usize])
    }
    pub fn attach(&mut self, id: LinkId, reg: Reg, site: Owner) {
        let heads = if reg.is_vreg() {
            &mut self.virtual_heads
        } else {
            &mut self.physical_heads
        };
        heads.resize(heads.len().max(reg.index() as usize + 1), [None.into(); 2]);
        self.links.set_owner(id, site);
        self.links
            .attach(id, &mut heads[reg.index() as usize][site.role as usize]);
    }
    pub fn detach(&mut self, id: LinkId, reg: Reg) {
        let role = self.links.owner(id).role;
        let heads = if reg.is_vreg() {
            &mut self.virtual_heads
        } else {
            &mut self.physical_heads
        };
        self.links
            .detach(id, &mut heads[reg.index() as usize][role as usize]);
    }
}

/// One borrowed occurrence. Physical references are not SSA def-use edges.
#[derive(Clone, Copy)]
pub struct RegRef<'a> {
    pub(crate) store: &'a InstStore,
    pub(crate) id: LinkId,
}
impl RegRef<'_> {
    pub fn operand(self) -> OperandId {
        OperandId::from_u32(self.id.as_u32())
    }
    pub fn inst(self) -> InstId {
        self.store.references.links.owner(self.id).inst
    }
    pub fn role(self) -> RefRole {
        self.store.references.links.owner(self.id).role
    }
    pub fn reg(self) -> Reg {
        self.store.operand(self.operand())
    }
}
#[derive(Clone)]
pub struct RegRefs<'a> {
    pub(crate) store: &'a InstStore,
    pub(crate) next: Head,
}
impl<'a> RegRefs<'a> {
    /// Exactly one occurrence, not merely one distinct owner instruction.
    pub fn single(mut self) -> Option<RegRef<'a>> {
        let first = self.next()?;
        self.next().is_none().then_some(first)
    }
}
impl<'a> Iterator for RegRefs<'a> {
    type Item = RegRef<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let id = self.next.expand()?;
        self.next = self.store.references.links.next(id);
        Some(RegRef {
            store: self.store,
            id,
        })
    }
}
