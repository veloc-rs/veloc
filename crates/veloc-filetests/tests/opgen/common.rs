#![allow(dead_code)]

pub const TYPES: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../veloc/defs/types.spec"
));
pub static BUILTINS: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    concat!(
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../veloc/types/defs/types.spec"
        )),
        "\n",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../veloc/defs/types.spec"
        )),
        "\n",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../veloc/mir/defs/types.spec"
        ))
    )
    .lines()
    .filter(|line| !line.trim_start().starts_with("import "))
    .collect::<Vec<_>>()
    .join("\n")
});

pub fn source(ops: &str) -> String {
    format!("{}\n{ops}", *BUILTINS)
}

pub fn parse(ops: &str) -> Result<veloc_spec::Definitions, veloc_spec::Error> {
    raw_parse(&source(ops))
}

pub fn raw_parse(source: &str) -> Result<veloc_spec::Definitions, veloc_spec::Error> {
    compiler::source(source)?
        .parse()
        .map_err(|error| error.diagnostic)
}

pub fn raw_plan(source: &str) -> Result<veloc_spec::Plan, veloc_spec::Error> {
    compiler::source(source)?
        .plan()
        .map_err(|error| error.diagnostic)
}

pub fn compile(ops: &str) -> Result<veloc_spec::Artifacts, veloc_spec::Error> {
    raw_plan(&source(ops)).map(|plan| plan.generate())
}

pub fn load(
    path: impl AsRef<std::path::Path>,
) -> Result<veloc_spec::Source, veloc_spec::SourceError> {
    veloc_spec::Source::load(path)
}

#[path = "../../compiler.rs"]
pub mod compiler;
pub fn const_rejected(source: &str, expected: &str) {
    let generated = compile(source).expect("definition syntax and types");
    let result = compiler::check(&generated).expect("run rustc");
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        !result.status.success() && stderr.contains("E0080") && stderr.contains(expected),
        "{stderr}"
    );
}

#[track_caller]
pub fn rejected(source: &str, expected: &str) {
    raw_rejected(&self::source(source), expected);
}

#[track_caller]
pub fn raw_rejected(source: &str, expected: &str) {
    let error = match raw_plan(source) {
        Ok(_) => panic!("invalid definition was accepted:\n{source}"),
        Err(error) => error,
    };
    assert!(
        error.message.contains(expected),
        "{error}\nsource:\n{source}"
    );
    assert!(error.line > 0 && error.column > 0, "{error}");
}
