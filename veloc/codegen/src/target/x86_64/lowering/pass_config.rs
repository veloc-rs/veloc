use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64PostIsel;

#[derive(Debug, Clone, Copy)]
pub struct X86_64PassConfig;

impl TargetPostIsel for X86_64PostIsel {}

impl TargetPassConfig for X86_64PassConfig {}
