// Optimizer rules are grouped by instruction family, independent of the
// generated consumer. root<T>(...) describes a value equality; root(...)
// replaces an instruction while retaining its existing SSA results.
// Equalities are reviewed, not proven by the pattern/type checker.
// Integer arithmetic is modular; -1 is all bits set at the matched width.
// Commutative value patterns automatically try both operand orders.
import "arithmetic.spec";
import "bitwise.spec";
import "compare.spec";
import "convert.spec";
import "memory.spec";
import "control.spec";
