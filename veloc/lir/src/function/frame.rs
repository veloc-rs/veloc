use super::{Reg, StackSlot};
use alloc::vec::Vec;
use cranelift_entity::PrimaryMap;

/// Symbolic placement constraints; no physical register is chosen here.
#[derive(Debug, Clone, Copy)]
pub enum StackObject {
    Local,
    Incoming {
        offset: u32,
    },
    Outgoing {
        frame: crate::CallFrameId,
        offset: u32,
    },
}

#[derive(Debug, Clone)]
pub struct StackSlotData {
    pub object: StackObject,
    pub size: u32,
    pub align: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct StackAddress {
    pub base: Reg,
    pub offset: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct SavedReg {
    pub reg: Reg,
    pub slot: StackSlot,
}

/// Final layout for the fixed-SP frame strategy. A future variable-SP strategy
/// must resolve addresses using program-point state instead of this table.
#[derive(Debug, Clone)]
pub struct FrameLayout {
    pub addresses: PrimaryMap<StackSlot, StackAddress>,
    pub saves: Vec<SavedReg>,
    pub local_size: u32,
    pub callee_saved_size: u32,
    pub total_size: u32,
}

#[derive(Debug, Clone, Default)]
pub struct StackFrame {
    calls: PrimaryMap<crate::CallFrameId, crate::StackArea>,
    slots: PrimaryMap<StackSlot, StackSlotData>,
    layout: Option<FrameLayout>,
}

/// Append-only allocation plan. IDs are valid for the originating frame once
/// committed; callers must not publish them into live IR before that commit.
pub struct StackBatch {
    base: usize,
    slots: Vec<StackSlotData>,
}

impl StackBatch {
    pub fn slots(&self) -> &[StackSlotData] {
        &self.slots
    }

    pub fn alloc_object(&mut self, object: StackObject, size: u32, align: u32) -> StackSlot {
        assert!(size != 0 && align.is_power_of_two(), "invalid stack object");
        let index = self
            .base
            .checked_add(self.slots.len())
            .expect("stack slot overflow");
        let id = StackSlot::from_u32(u32::try_from(index).expect("stack slot overflow"));
        self.slots.push(StackSlotData {
            object,
            size,
            align,
        });
        id
    }
}

impl StackFrame {
    pub fn alloc_call(&mut self, area: crate::StackArea) -> crate::CallFrameId {
        assert!(self.layout.is_none(), "frame already laid out");
        assert!(area.align.is_power_of_two(), "invalid call alignment");
        self.calls.push(area)
    }

    pub fn call(&self, id: crate::CallFrameId) -> Option<&crate::StackArea> {
        self.calls.get(id)
    }

    pub fn batch(&self) -> StackBatch {
        assert!(self.layout.is_none(), "frame already laid out");
        StackBatch {
            base: self.slots.len(),
            slots: Vec::new(),
        }
    }

    pub fn append(&mut self, batch: StackBatch) {
        assert!(self.layout.is_none(), "frame already laid out");
        assert_eq!(batch.base, self.slots.len(), "stale stack allocation batch");
        self.slots.reserve(batch.slots.len());
        for slot in batch.slots {
            self.slots.push(slot);
        }
    }

    pub fn slots(&self) -> &PrimaryMap<StackSlot, StackSlotData> {
        &self.slots
    }
    pub fn layout(&self) -> Option<&FrameLayout> {
        self.layout.as_ref()
    }

    pub fn address(&self, slot: StackSlot) -> StackAddress {
        self.layout
            .as_ref()
            .expect("frame has not been laid out")
            .addresses[slot]
    }

    pub fn finish(&mut self, layout: FrameLayout) {
        assert!(self.layout.is_none(), "frame already laid out");
        assert_eq!(layout.addresses.len(), self.slots.len());
        self.layout = Some(layout);
    }

    pub fn alloc_object(&mut self, object: StackObject, size: u32, align: u32) -> StackSlot {
        assert!(self.layout.is_none(), "cannot allocate after frame layout");
        assert!(size != 0 && align.is_power_of_two(), "invalid stack object");
        self.slots.push(StackSlotData {
            object,
            size,
            align,
        })
    }
}
