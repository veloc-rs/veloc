// Optimizer rules are grouped by instruction family, independent of the
// generated consumer. Every case uses `rule`; the compiler chooses folding,
// equality search or instruction edits from the complete rule's capabilities.
// Type annotations constrain matches without selecting an execution host.
// Equalities are reviewed, not proven by the pattern/type checker.
// Integer arithmetic is modular; -1 is all bits set at the matched width.
// Commutative value patterns automatically try both operand orders.
import "arithmetic.spec";
import "bitwise.spec";
import "compare.spec";
import "convert.spec";
import "memory.spec";
import "control.spec";
