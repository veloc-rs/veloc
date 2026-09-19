//! Generated scalar evaluation and algebraic facts consumed by the e-graph.
use alloc::vec::Vec;
use veloc_mir::constant::ScalarConst;
use veloc_mir::{IntCC, Opcode, Type, Value};

pub(crate) enum Replacement<V = Value> {
    Constants(Vec<ScalarConst>),
    Value(V),
}

include!(concat!(env!("OUT_DIR"), "/evaluation.rs"));
