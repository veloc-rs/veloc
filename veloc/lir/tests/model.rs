use veloc_lir::stages::{LegalizedLir, RawLir};
use veloc_lir::{MachineFunction, MachineInst, MachineModule, RegisterBank, SymbolTable, Writable};
use veloc_mir::{Linkage, Type};

#[test]
fn standalone_module_supports_instruction_and_stage_apis() {
    let mut function = MachineFunction::<RawLir>::new("example".into());
    let block = function.create_synthetic_block();
    let reg = function.alloc_vreg(Type::I64);
    let inst = function.alloc_inst(MachineInst::build_constant(Writable(reg), 42));
    function.append_inst_id_to_block(function.find_block_index(block).unwrap(), inst);
    assert_eq!(function.dfg[inst].as_constant().unwrap().imm, 42);

    let mut module = MachineModule::new("standalone".into());
    let id = module.add_function(function);
    assert_eq!(module.find_function_by_name("example"), Some(id));
    assert_eq!(module.functions[id].block_insts(0), &[inst]);
    let mut function = module.functions[id].clone().into_stage::<LegalizedLir>();
    let banked = function.alloc_vreg_in_bank(Type::I64, RegisterBank::GPR);
    assert!(banked.is_vreg());
}

#[test]
fn symbol_interning_does_not_require_a_source_module() {
    let mut symbols = SymbolTable::new();
    let first = symbols.get_or_create_function("callee", Linkage::Import);
    assert_eq!(
        symbols.get_or_create_function("callee", Linkage::Import),
        first
    );
    assert_eq!(symbols.get(first).linkage, Linkage::Import);
    assert_ne!(
        symbols.get_or_create_function("other", Linkage::Export),
        first
    );
}

#[test]
fn decode_errors_are_owned_by_lir() {
    let inst = MachineInst::build_constant(Writable(veloc_lir::Reg::new_vreg(0)), 42);
    let error: veloc_lir::DecodeError = inst.as_binary_reg().unwrap_err();
    assert!(matches!(
        error.opcode,
        veloc_lir::MachineOpcode::Generic(veloc_lir::GenericOpcode::G_CONSTANT)
    ));
    assert!(!error.reason.is_empty());
    assert!(error.to_string().contains("invalid LIR instruction"));
}
