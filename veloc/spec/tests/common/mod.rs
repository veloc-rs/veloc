#![allow(dead_code)]

pub const TYPES: &str = include_str!("../../../defs/types.ops");
pub static BUILTINS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    concat!(
        include_str!("../../../defs/types.ops"),
        "\n",
        include_str!("../../../defs/builtins.ops"),
        "\n",
        include_str!("../../../defs/comparisons.ops")
    )
    .lines()
    .filter(|line| !line.trim_start().starts_with("import "))
    .collect::<Vec<_>>()
    .join("\n")
});

pub fn source(ops: &str) -> String {
    format!("{}\n{ops}", *BUILTINS)
}

pub fn parse(ops: &str) -> Result<veloc_opgen::Definitions, veloc_opgen::Error> {
    veloc_opgen::parse(&source(ops))
}

pub fn compile(ops: &str) -> Result<veloc_opgen::Generated, veloc_opgen::Error> {
    veloc_opgen::compile(&source(ops))
}
