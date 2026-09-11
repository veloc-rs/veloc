#![allow(dead_code)]

pub const TYPES: &str = include_str!("../../../defs/types.ops");
pub const BUILTINS: &str = concat!(
    include_str!("../../../defs/types.ops"),
    "\n",
    include_str!("../../../defs/builtins.ops"),
    "\n",
    include_str!("../../../defs/comparisons.ops")
);

pub fn source(ops: &str) -> String {
    format!("{BUILTINS}\n{ops}")
}

pub fn parse(ops: &str) -> Result<veloc_opgen::Definitions, veloc_opgen::Error> {
    veloc_opgen::parse(&source(ops))
}

pub fn compile(ops: &str) -> Result<veloc_opgen::Generated, veloc_opgen::Error> {
    veloc_opgen::compile(&source(ops))
}
