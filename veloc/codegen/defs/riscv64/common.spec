import "../../../defs/prelude.spec";
import "../../../encoder/defs/riscv64.spec";
import "registers.spec";
import "features.spec";
import "schedule.spec";

type CallInfo = rust("veloc_lir::CallInfo") { view = borrowed; }

type Block = rust("veloc_lir::BlockId");
type Successor = rust("veloc_lir::EdgeId");
type Global = rust("veloc_lir::SymbolId");
type StackSlot = rust("veloc_lir::StackSlot");
type Emission = rust("crate::target::riscv64::emitter::Emission") {
    trait = rust("crate::target::riscv64::emitter::host::Emission");
    fn instructions(code: sequence(Instruction)) -> Self;
    fn jump(target: Block) -> Self;
    fn branch(funct3: u32, lhs: Reg, rhs: Reg, target: Block) -> Self;
    fn call(target: Global) -> Self;
    fn address(dst: Reg, target: Global) -> Self;
    fn table(index: Reg, targets: sequence(Block)) -> Self;
}

typeset GprValue = Type::BOOL | ScalarInteger | Type::PTR;
typeset ScalarValue = GprValue | ScalarFloat;

// All arithmetic families describe actual encoding fields here.
template Binary(Name: ident, Domain: expr, Class: ident, Major: expr, F3: expr, F7: expr, Mnemonic: expr, Extension: ident, Scheduling: ident, Movable: expr) {
    op Name(lhs: Value<Domain>, rhs: Value<Domain>) -> (dst: Value<Domain>) {
        encoding = Emission::instructions([Instruction::R(Major,dst,F3,lhs,rhs,F7)]);
        registers = { dst: Class, lhs: Class, rhs: Class };
        requires = [Extension];
        schedule = Scheduling;
        movable = Movable;
        assembly = {
            lines: [{ mnemonic: Mnemonic, operands: [reg(dst,64),reg(lhs,64),reg(rhs,64)] }]
        };
    }
}
