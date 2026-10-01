import "../../../defs/prelude.spec";
import "../../../encoder/defs/x86_64.spec";
import "registers.spec";
import "cpu/features.spec";
import "schedule.spec";

type CallInfo = rust("veloc_lir::CallInfo") { view = borrowed; }

type Block = rust("veloc_lir::BlockId");
type Successor = rust("veloc_lir::EdgeId");
type Global = rust("veloc_lir::SymbolId");
type StackSlot = rust("veloc_lir::StackSlot");
type Emission = rust("crate::target::x86_64::emitter::Emission") {
    trait = rust("crate::target::x86_64::emitter::host::Emission");
    fn legacy(descriptor: Legacy, form: Form, immediate: Immediate) -> Self;
    fn branch(target: Block, form: Branch) -> Self;
    fn relative(target: Global, descriptor: Legacy, form: Form, addend: i64) -> Self;
}

// Arguments name ModRM fields, not semantic source/destination roles.
fn legacy_rr(opcode: u8, wide: bool, reg: Reg, rm: Reg) -> Emission {
    value = Emission::legacy(
        Legacy { prefix: Prefix::None, map: OpcodeMap::Primary, opcode: opcode, wide: wide },
        Form::ModRm(RegField::Register(reg), Rm::Register(rm)),
        Immediate::None,
    );
}

// Representation domains for selected virtual values, not pointer provenance.
// GPR operations may consume low bits or define a wider zero-extended value.
typeset GprValue = ScalarInteger | Type::BOOL | Type::PTR;
typeset AddressValue = Type::I64 | Type::PTR;
typeset SmallInt = Type::I8 | Type::I16 | Type::I32;
typeset WordOrPtr = Type::I64 | Type::PTR;
