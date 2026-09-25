//! Generated scalar evaluation consumed by the e-graph's constant analysis.
use veloc_mir::constant::ScalarConst;
use veloc_mir::{IntCC, Opcode, Type, Value};

include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));
