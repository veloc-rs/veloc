// Instruction categories; each CPU supplies its own resources and costs.
schedule_class IntAlu { doc = "Integer arithmetic, logic and shifts"; }
schedule_class IntMul32 { doc = "32-bit integer multiplication"; }
schedule_class IntMul64 { doc = "64-bit integer multiplication"; }
schedule_class IntDiv32 { doc = "32-bit integer division and remainder"; }
schedule_class IntDiv64 { doc = "64-bit integer division and remainder"; }
