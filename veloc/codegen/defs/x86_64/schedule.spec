// Instruction categories; each CPU supplies its own resources and costs.
schedule_class Copy { doc = "Register moves and constant materialization"; }
schedule_class IntAlu { doc = "Integer arithmetic, logic and comparisons"; }
schedule_class IntMul { doc = "Integer multiplication"; }
schedule_class IntPopcnt { doc = "Integer population count"; }
schedule_class IntToFloat { doc = "Integer to floating-point conversion"; }
schedule_class FloatToInt { doc = "Floating-point to integer conversion"; }
schedule_class FloatConvert { doc = "Floating-point precision conversion"; }
schedule_class FloatSqrt { doc = "Floating-point square root"; }

schedule_class Address { doc = "Address formation"; }
schedule_class Branch { doc = "Conditional and unconditional branches"; }
schedule_class Call { doc = "Call transfer only, excluding the callee"; }
schedule_class FloatAdd { doc = "Floating-point addition and subtraction"; }
schedule_class FloatCompare { doc = "Floating-point comparisons"; }
schedule_class FloatDiv { doc = "Floating-point division"; }
schedule_class FloatMul { doc = "Floating-point multiplication"; }
schedule_class IntDiv32 { doc = "32-bit integer division"; }
schedule_class IntDiv64 { doc = "64-bit integer division"; }
schedule_class IntShift { doc = "Variable-count shifts and rotates"; }
schedule_class Load { doc = "Memory loads including address formation"; }
schedule_class Push { doc = "Stack push"; }
schedule_class Pop { doc = "Stack pop"; }
schedule_class Return { doc = "Return transfer"; }
schedule_class Store { doc = "Memory stores including address formation"; }
schedule_class Trap { doc = "Trap instruction issue, excluding exception handling"; }
