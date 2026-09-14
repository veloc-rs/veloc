//! Store-maintained register occurrences, not a separately rebuilt analysis.
use crate::{InstId, InstStore, Reg};
use alloc::vec::Vec;
use cranelift_entity::packed_option::PackedOption;
use veloc_collections::{LinkId, Links};

type Head = PackedOption<LinkId>;
const TAG_SHIFT: u32 = 29;
const INDEX: u32 = (1 << TAG_SHIFT) - 1;

/// Logical locations are invalidated by reshaping an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefLocation {
    Result(u32),
    ImplicitUse(u32),
    ImplicitDef(u32),
    Input(u32),
    EdgeArg(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefRole {
    Use,
    Def,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Site {
    pub inst: InstId,
    slot: u32,
}
impl Site {
    pub fn new(inst: InstId, location: RefLocation, role: RefRole) -> Self {
        let (index, tag, expected) = match location {
            RefLocation::Input(index) => (index, 0, RefRole::Use),
            RefLocation::Result(index) => (index, 1, RefRole::Def),
            RefLocation::EdgeArg(index) => (index, 2, RefRole::Use),
            RefLocation::ImplicitUse(index) => (index, 3, RefRole::Use),
            RefLocation::ImplicitDef(index) => (index, 4, RefRole::Def),
        };
        assert_eq!(role, expected);
        assert!(index <= INDEX, "reference index overflow");
        Self {
            inst,
            slot: index | tag << TAG_SHIFT,
        }
    }
    pub fn role(self) -> RefRole {
        match self.slot >> TAG_SHIFT {
            1 | 4 => RefRole::Def,
            _ => RefRole::Use,
        }
    }
    pub fn location(self) -> RefLocation {
        let index = self.slot & INDEX;
        match self.slot >> TAG_SHIFT {
            0 => RefLocation::Input(index),
            1 => RefLocation::Result(index),
            2 => RefLocation::EdgeArg(index),
            3 => RefLocation::ImplicitUse(index),
            4 => RefLocation::ImplicitDef(index),
            _ => unreachable!("invalid reference location"),
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RefRange {
    start: u32,
    len: u32,
}
impl RefRange {
    pub fn ids(self) -> impl Iterator<Item = LinkId> {
        (self.start..self.start + self.len).map(LinkId::from_u32)
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct References {
    pub links: Links<Site>,
    // Dense indices are separate: physical IDs cannot inflate the value index.
    virtual_heads: Vec<[Head; 2]>,
    physical_heads: Vec<[Head; 2]>,
    free: Vec<Vec<u32>>,
}
impl References {
    pub fn check_ranges(
        &self,
        ranges: impl Iterator<Item = (InstId, RefRange)>,
    ) -> Result<(), &'static str> {
        let mut allocated = alloc::vec![false; self.links.len()];
        let mut mark = |start: usize, capacity: usize| -> Result<(), &'static str> {
            let end = start
                .checked_add(capacity)
                .ok_or("reference range overflow")?;
            let slots = allocated
                .get_mut(start..end)
                .ok_or("reference range out of bounds")?;
            for slot in slots {
                if core::mem::replace(slot, true) {
                    return Err("overlapping reference ranges");
                }
            }
            Ok(())
        };
        for (owner, range) in ranges {
            if range.len == 0 {
                continue;
            }
            mark(range.start as usize, range.len.next_power_of_two() as usize)?;
            for id in range.ids() {
                if self.links.owner(id).inst != owner {
                    return Err("incorrect reference range owner");
                }
            }
        }
        for (class, ranges) in self.free.iter().enumerate() {
            for &start in ranges {
                mark(start as usize, 1 << class)?;
            }
        }
        if allocated.iter().any(|&slot| !slot) {
            return Err("lost reference range");
        }
        Ok(())
    }

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
    pub fn alloc(&mut self, len: usize, owner: Site) -> RefRange {
        if len == 0 {
            return RefRange::default();
        }
        let capacity = len
            .checked_next_power_of_two()
            .expect("too many references");
        let class = capacity.trailing_zeros() as usize;
        self.free
            .resize_with(self.free.len().max(class + 1), Vec::new);
        let start = self.free[class].pop().unwrap_or_else(|| {
            let start = self.links.len();
            let end = start.checked_add(capacity).expect("too many references");
            assert!(end < u32::MAX as usize, "reference store overflow");
            self.links.resize(end, owner);
            start as u32
        });
        RefRange {
            start,
            len: len.try_into().expect("too many references"),
        }
    }
    pub fn release(&mut self, range: RefRange) {
        if range.len != 0 {
            self.free[range.len.next_power_of_two().trailing_zeros() as usize].push(range.start);
        }
    }
    pub fn attach(&mut self, id: LinkId, reg: Reg, site: Site) {
        let heads = if reg.is_vreg() {
            &mut self.virtual_heads
        } else {
            &mut self.physical_heads
        };
        heads.resize(heads.len().max(reg.index() as usize + 1), [None.into(); 2]);
        self.links.set_owner(id, site);
        self.links
            .attach(id, &mut heads[reg.index() as usize][site.role() as usize]);
    }
    pub fn detach(&mut self, id: LinkId, reg: Reg) {
        let role = self.links.owner(id).role();
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
    pub fn inst(self) -> InstId {
        self.store.references.links.owner(self.id).inst
    }
    pub fn location(self) -> RefLocation {
        self.store.references.links.owner(self.id).location()
    }
    pub fn role(self) -> RefRole {
        self.store.references.links.owner(self.id).role()
    }
    pub fn reg(self) -> Reg {
        self.store
            .reg_at(self.store.references.links.owner(self.id))
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
