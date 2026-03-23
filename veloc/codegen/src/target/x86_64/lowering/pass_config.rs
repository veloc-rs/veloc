use super::*;

#[derive(Debug, Clone, Copy)]
pub struct X86_64PostIsel;

impl X86_64PostIsel {
    pub fn new(_cpu: CpuDescription) -> Self {
        Self
    }
}

#[derive(Debug, Clone, Copy)]
pub struct X86_64PassConfig;

impl X86_64PassConfig {
    pub fn new(_cpu: CpuDescription) -> Self {
        Self
    }
}

impl TargetPostIsel for X86_64PostIsel {}

impl TargetPassConfig for X86_64PassConfig {}
