// Instruction categories; each CPU supplies its own resources and costs.
schedule_class Copy { doc = "Register moves and constant materialization"; }
schedule_class IntAlu { doc = "Integer arithmetic, logic and comparisons"; }
schedule_class IntMul { doc = "Integer multiplication"; }
schedule_class IntPopcnt { doc = "Integer population count"; }
schedule_class IntToFloat { doc = "Integer to floating-point conversion"; }
schedule_class FloatToInt { doc = "Floating-point to integer conversion"; }
schedule_class FloatConvert { doc = "Floating-point precision conversion"; }
schedule_class FloatSqrt { doc = "Floating-point square root"; }
