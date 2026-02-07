use crate::Function;
use alloc::boxed::Box;
use alloc::vec::Vec;

pub trait Pass {
    fn name(&self) -> &str;
    fn run_on_function(&mut self, func: &mut Function);
}

pub struct PassManager {
    passes: Vec<Box<dyn Pass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: Vec::new() }
    }

    pub fn add_pass(&mut self, pass: Box<dyn Pass>) {
        self.passes.push(pass);
    }

    pub fn run(&mut self, func: &mut Function) {
        for pass in &mut self.passes {
            pass.run_on_function(func);
        }
    }
}
