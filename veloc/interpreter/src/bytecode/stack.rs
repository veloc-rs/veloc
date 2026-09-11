//! Fixed frame layout for entry allocations. Dynamic allocations are rejected
//! before installing a module; bytecode emission uses this checked layout.
use alloc::string::String;
use cranelift_entity::SecondaryMap;
use veloc_mir::{Function, Inst, InstructionView};

pub(crate) struct StackLayout {
    pub offsets: SecondaryMap<Inst, u32>,
    pub size: usize,
    pub align: usize,
}

pub(crate) fn stack_layout(func: &Function) -> Result<StackLayout, String> {
    let mut layout = StackLayout {
        offsets: SecondaryMap::new(),
        size: 0,
        align: 1,
    };
    for &block in func.layout().block_order() {
        for &inst in &func.layout().blocks()[block].insts {
            if let InstructionView::Alloca { size, align } = func.dfg().inst(inst) {
                if Some(block) != func.entry_block {
                    return Err("non-entry alloca requires dynamic stack support".into());
                }
                if size == 0 || !align.is_power_of_two() {
                    return Err("invalid alloca size or alignment".into());
                }
                let offset = layout
                    .size
                    .checked_add(align as usize - 1)
                    .map(|n| n & !(align as usize - 1))
                    .ok_or("alloca frame size overflow")?;
                layout.offsets[inst] = u32::try_from(offset)
                    .map_err(|_| "alloca frame exceeds bytecode offset range")?;
                layout.size = offset
                    .checked_add(size as usize)
                    .filter(|n| *n <= u32::MAX as usize)
                    .ok_or("alloca frame size overflow")?;
                layout.align = layout.align.max(align as usize);
            }
        }
    }
    Ok(layout)
}
