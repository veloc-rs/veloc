#![allow(dead_code)]

pub const TYPES: &str = include_str!("../../../mir/defs/types.ops");
pub const BUILTINS: &str = concat!(
    include_str!("../../../mir/defs/types.ops"),
    "\n",
    include_str!("../../../mir/defs/builtins.ops"),
    "\n",
    include_str!("../../../mir/defs/comparisons.ops")
);

pub fn source(ops: &str) -> String {
    format!("{BUILTINS}\n{ops}")
}

pub fn parse(ops: &str) -> Result<veloc_opgen::Definitions, veloc_opgen::Error> {
    veloc_opgen::parse(&source(ops))
}

pub fn compile_mir(ops: &str) -> Result<veloc_opgen::Generated, veloc_opgen::Error> {
    veloc_opgen::compile_mir(&source(ops))
}
