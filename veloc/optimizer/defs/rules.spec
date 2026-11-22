// Optimizer rules are grouped by instruction family, independent of the
// generated consumer. Scalar evaluation uses explicit `fold` declarations.
// Pattern rules can supply local folds, equality search or instruction edits.
// Type annotations constrain matches without selecting an execution host.
// Equalities are reviewed, not proven by the pattern/type checker.
// Integer arithmetic is modular; -1 is all bits set at the matched width.
// Commutative value patterns automatically try both operand orders.
import "folds.spec";
import "arithmetic.spec";
import "bitwise.spec";
import "compare.spec";
import "convert.spec";
import "memory.spec";
import "control.spec";
