//! Native ABI to shared host-callback ABI bridges, specialized by import signature.
use super::*;
use crate::instance::VMInstance;
use crate::vm::VMContext;
use veloc::interpreter::InterpreterValue;

pub(super) fn generate(ir: &mut veloc::mir::ModuleBuilder, meta: &WasmMetadata) {
    let sig = ir.make_signature(
        vec![VelocType::PTR, VelocType::I32, VelocType::PTR],
        vec![],
        CallConv::SystemV,
    );
    let dispatch = ir.declare_function("wasm_host_call".into(), sig, Linkage::Import);
    for i in 0..meta.num_imported_funcs {
        let sig = &meta.signatures[meta.functions[i].type_index as usize];
        let native_sig = sig.intern_veloc_sig(ir);
        let id = ir.declare_function(format!("__veloc_host_{i}"), native_sig, Linkage::Export);
        let mut builder = ir.builder(id);
        builder.init_entry_block();
        let params = builder.func_params().to_vec();
        let slots = (sig.params.len() + 1).max(sig.results.len());
        let buffer = builder.entry_alloca((slots * 8) as u32, 8);
        let mut ins = builder.ins();
        // Every slot is a complete InterpreterValue, including zeroed high bits
        // for narrow arguments. Slot zero carries the current instance context.
        for (j, &param) in params.iter().take(sig.params.len() + 1).enumerate() {
            let ty = ins.builder().value_type(param);
            let bits = match ty {
                VelocType::PTR => ins.ptrtoint(param, VelocType::I64),
                VelocType::F64 => ins.reinterpret(param, VelocType::I64),
                VelocType::F32 => {
                    let bits = ins.reinterpret(param, VelocType::I32);
                    ins.extendu(bits, VelocType::I64)
                }
                VelocType::I32 => ins.extendu(param, VelocType::I64),
                _ => param,
            };
            ins.store(buffer, bits, (j * 8) as u32, MemFlags::default());
        }
        let index = ins.i32const(i as i32);
        ins.call(dispatch, &[params[0], index, buffer]);
        let mut returns = Vec::new();
        for (j, &ty) in sig.results.iter().enumerate() {
            let ty = valtype_to_veloc(ty);
            let result = ins.load(buffer, (j * 8) as u32, MemFlags::default(), ty);
            if sig.results.len() > 1 {
                ins.store(
                    *params.last().unwrap(),
                    result,
                    (j * 8) as u32,
                    MemFlags::default(),
                );
            } else {
                returns.push(result);
            }
        }
        ins.ret(&returns);
        builder.seal_all_blocks();
    }
}

pub(super) unsafe extern "C" fn wasm_host_call(
    vmctx: *mut VMContext,
    index: u32,
    values: *mut InterpreterValue,
) {
    // The instance owns a clone of each callback, independent of Store moves.
    // Do not retain an instance borrow while the callback accesses its VMContext.
    let host = unsafe { VMInstance::from_vmctx(vmctx) }.host_imports[index as usize]
        .as_ref()
        .expect("host bridge must have a registered callback")
        .clone();
    let sig = host.signature();
    let slots = sig.params().len().max(sig.returns().len()).max(1);
    let values = unsafe { core::slice::from_raw_parts_mut(values, slots) };
    host.invoke(values)
        .expect("generated host bridge must match the callback signature");
}
