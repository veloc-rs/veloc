use crate::instance::VMInstance;
use crate::vm::VMContext;

unsafe extern "C" {
    fn siglongjmp(env: *mut u8, val: i32) -> !;
}

pub extern "C" fn wasm_trap_handler(_vmctx: *mut VMContext, code: u32) -> ! {
    unsafe {
        let vmctx = &*_vmctx;
        let instance = VMInstance::from_vmctx(_vmctx);
        let offsets = &instance.module.inner.offsets;
        let jmp_buf = vmctx.jmp_buf_ptr(offsets);
        siglongjmp(jmp_buf, (code as i32) + 1);
    }
}

pub extern "C" fn wasm_memory_size(vmctx: *mut VMContext, mem_idx: u32) -> u32 {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    instance.memory_size(mem_idx)
}

pub extern "C" fn wasm_memory_grow(vmctx: *mut VMContext, mem_idx: u32, delta: u32) -> i32 {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    match instance.memory_grow(mem_idx, delta) {
        Ok(old_pages) => old_pages as i32,
        Err(_) => -1,
    }
}

pub extern "C" fn wasm_table_size(vmctx: *mut VMContext, table_idx: u32) -> u32 {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    instance.table_size(table_idx)
}

pub extern "C" fn wasm_table_grow(
    vmctx: *mut VMContext,
    table_idx: u32,
    init_val: *mut crate::vm::VMFuncRef,
    delta: u32,
) -> i32 {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    match instance.table_grow(table_idx, init_val, delta) {
        Ok(old_size) => old_size,
        Err(_) => -1,
    }
}

pub extern "C" fn wasm_table_fill(
    vmctx: *mut VMContext,
    table_idx: u32,
    dst: u32,
    val: *mut crate::vm::VMFuncRef,
    len: u32,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = instance.table_fill(table_idx, dst, val, len) {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::TableOutOfBounds as u32);
    }
}

pub extern "C" fn wasm_table_copy(
    vmctx: *mut VMContext,
    dst_idx: u32,
    src_idx: u32,
    dst: u32,
    src: u32,
    len: u32,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = instance.table_copy(dst_idx, src_idx, dst, src, len) {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::TableOutOfBounds as u32);
    }
}

pub extern "C" fn wasm_table_init(
    vmctx: *mut VMContext,
    table_idx: u32,
    elem_idx: u32,
    dst: u32,
    src: u32,
    len: u32,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = instance.table_init(table_idx, elem_idx, dst, src, len) {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::TableOutOfBounds as u32);
    }
}

pub extern "C" fn wasm_elem_drop(vmctx: *mut VMContext, elem_idx: u32) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    instance.elem_drop(elem_idx);
}

pub extern "C" fn wasm_memory_init(
    vmctx: *mut VMContext,
    mem_idx: u32,
    data_idx: u32,
    dst: u32,
    src: u32,
    len: u32,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = instance.memory_init(mem_idx, data_idx, dst, src, len) {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::MemoryOutOfBounds as u32);
    }
}

pub extern "C" fn wasm_data_drop(vmctx: *mut VMContext, data_idx: u32) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    instance.data_drop(data_idx);
}

pub extern "C" fn wasm_memory_copy(
    vmctx: *mut VMContext,
    dst_idx: u32,
    src_idx: u32,
    dst: u32,
    src: u32,
    len: u32,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = instance.memory_copy(dst_idx, src_idx, dst, src, len) {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::MemoryOutOfBounds as u32);
    }
}

pub extern "C" fn wasm_memory_fill(
    vmctx: *mut VMContext,
    mem_idx: u32,
    dst: u32,
    val: u32,
    len: u32,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = instance.memory_fill(mem_idx, dst, val, len) {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::MemoryOutOfBounds as u32);
    }
}

pub unsafe extern "C" fn wasm_init_table_element(
    vmctx: *mut VMContext,
    element_idx: u32,
    offset: u32,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = unsafe { instance.init_table_segment(element_idx, offset) } {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::TableOutOfBounds as u32);
    }
}

pub unsafe extern "C" fn wasm_init_memory_data(vmctx: *mut VMContext, data_idx: u32, offset: u32) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    if let Err(_e) = unsafe { instance.init_memory_segment(data_idx, offset) } {
        wasm_trap_handler(vmctx, crate::vm::TrapCode::MemoryOutOfBounds as u32);
    }
}

pub unsafe extern "C" fn wasm_init_table(
    vmctx: *mut VMContext,
    table_idx: u32,
    val: *mut crate::vm::VMFuncRef,
) {
    let instance = unsafe { VMInstance::from_vmctx(vmctx) };
    let offsets = &instance.module.inner.offsets;
    let tables = unsafe { (*vmctx).tables_mut(offsets, instance.module.metadata().tables.len()) };
    let table = unsafe { &mut *tables[table_idx as usize] };
    for i in 0..table.current_elements {
        unsafe {
            *table.base.add(i) = val;
        }
    }
}
