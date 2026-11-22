//! Fixed frame layout for function-local stack slots.
use alloc::string::String;
use cranelift_entity::SecondaryMap;
use veloc_mir::{FuncBody, Value};

pub(crate) struct StackLayout {
    pub offsets: SecondaryMap<Value, u32>,
    pub size: usize,
    pub align: usize,
}

pub(crate) fn stack_layout(func: &FuncBody) -> Result<StackLayout, String> {
    let mut layout = StackLayout {
        offsets: SecondaryMap::new(),
        size: 0,
        align: 1,
    };
    for (value, size, align) in func.used_stack_slots() {
        if size == 0 || !align.is_power_of_two() {
            return Err("invalid stack slot size or alignment".into());
        }
        let offset = layout
            .size
            .checked_add(align as usize - 1)
            .map(|n| n & !(align as usize - 1))
            .ok_or("stack slot frame size overflow")?;
        layout.offsets[value] =
            u32::try_from(offset).map_err(|_| "stack slot frame exceeds bytecode offset range")?;
        layout.size = offset
            .checked_add(size as usize)
            .filter(|n| *n <= u32::MAX as usize)
            .ok_or("stack slot frame size overflow")?;
        layout.align = layout.align.max(align as usize);
    }
    Ok(layout)
}
