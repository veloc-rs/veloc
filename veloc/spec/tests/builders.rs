mod common;
use common::compile_mir;

#[test]
fn property_order_follows_the_logical_signature_not_storage() {
    let source = r#"
        format Load {
            fields: [ptr(Value), offset(u32), flags(MemFlags)], opcode: fixed(Load)
        }
        format Store {
            fields: [ptr(Value), value(Value), offset(u32), flags(MemFlags)], opcode: fixed(Store)
        }
        op Load(@flags: MemFlags, address: PTR, @displacement: u32) -> (result: Any) {
            mnemonic: "load", storage: Load { ptr: address, offset: displacement, flags: flags },
            text: Text { args: [address], named: [default(displacement, 0)], flags: flags },
            traits: [MAY_TRAP], memory: HEAP_READ
        }
        op Store(ptr: PTR, @flags: MemFlags, value: Any, @offset: u32) -> () {
            mnemonic: "store", storage: Store { ptr: ptr, value: value, offset: offset, flags: flags },
            text: Text { args: [value, ptr], named: [default(offset, 0)], flags: flags },
            traits: [MAY_TRAP], memory: HEAP_WRITE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(output.opcodes.contains(
        "pub fn load(&mut self, flags: crate::MemFlags, address: crate::Value, displacement: u32, ty: crate::Type) -> crate::Value"
    ));
    assert!(
        output
            .opcodes
            .contains("InstructionData::Load { ptr: address, offset: displacement, flags }")
    );
    assert!(output.opcodes.contains(
        "pub fn store(&mut self, ptr: crate::Value, flags: crate::MemFlags, value: crate::Value, offset: u32)"
    ));
}

#[test]
fn contextual_selection_does_not_depend_on_the_method_name() {
    let source = r#"
        format Jump { fields: [dest(BlockCall)], opcode: fixed(Jump) }
        op Jump(dest: successor) -> () {
            mnemonic: "connect-edge", storage: Jump { dest: dest }, traits: [TERMINATOR], memory: NONE
        }
    "#;
    let output = compile_mir(source).unwrap();
    assert!(!output.opcodes.contains("pub fn connect_edge("));
    assert!(!output.opcodes.contains("self.push("));
}
