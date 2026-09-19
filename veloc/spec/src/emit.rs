//! Artifact selection is shared by the CLI and build-script API.
use crate::{Source, SourceError};
use std::{
    collections::BTreeMap,
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Emit {
    Types,
    TypeRules,
    Opcodes,
    Instructions,
    Builders,
    Validator,
    Checks,
    Evaluation,
    Semantics,
    TextParser,
    TextPrinter,
    DataTypes,
    Interfaces,
    Target,
    Selector,
    Encoder,
    Assembly,
    Rules,
    Decisions,
}
impl Emit {
    pub const ALL: &'static [Self] = &[
        Self::Types,
        Self::TypeRules,
        Self::Opcodes,
        Self::Instructions,
        Self::Builders,
        Self::Validator,
        Self::Checks,
        Self::Evaluation,
        Self::Semantics,
        Self::TextParser,
        Self::TextPrinter,
        Self::DataTypes,
        Self::Interfaces,
        Self::Target,
        Self::Selector,
        Self::Encoder,
        Self::Assembly,
        Self::Rules,
        Self::Decisions,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Types => "types",
            Self::TypeRules => "type-rules",
            Self::Opcodes => "opcodes",
            Self::Instructions => "instructions",
            Self::Builders => "builders",
            Self::Validator => "validator",
            Self::Checks => "checks",
            Self::Evaluation => "evaluation",
            Self::Semantics => "semantics",
            Self::TextParser => "text-parser",
            Self::TextPrinter => "text-printer",
            Self::DataTypes => "data-types",
            Self::Interfaces => "interfaces",
            Self::Target => "target",
            Self::Selector => "selector",
            Self::Encoder => "encoder",
            Self::Assembly => "assembly",
            Self::Rules => "rules",
            Self::Decisions => "decisions",
        }
    }
    pub fn filename(self) -> String {
        if self == Self::Validator {
            "validation.rs".into()
        } else {
            format!("{}.rs", self.name().replace('-', "_"))
        }
    }
    fn target(self) -> bool {
        matches!(
            self,
            Self::Target | Self::Selector | Self::Encoder | Self::Assembly
        )
    }
}
impl fmt::Display for Emit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}
impl FromStr for Emit {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::ALL
            .iter()
            .copied()
            .find(|e| e.name() == s)
            .ok_or_else(|| format!("unknown artifact {s}"))
    }
}

/// Explicit Rust/target bindings, independent of source file names.
#[derive(Default)]
pub struct Options<'a> {
    pub target: Option<Target<'a>>,
    pub interfaces: Option<&'a str>,
    pub rules: Option<ValueRules<'a>>,
    pub decisions: Option<Decisions<'a>>,
}

/// Bind a checked rule module to two explicitly named IR dialects.
pub struct ValueRules<'a> {
    pub source: &'a Source,
    pub target: &'a Source,
    pub rust: crate::rules::Rust<'a>,
    pub infer_primitives: bool,
}
pub struct Decisions<'a> {
    pub definitions: &'a Source,
    pub rust: crate::rules::DecisionRust<'a>,
}
pub struct Target<'a> {
    /// Logical input dialect and its operation definitions for selection.
    pub input: Option<(&'a str, &'a Source)>,
    pub arch: &'a str,
    /// Runtime trait implementing the declared selector predicates.
    pub context: &'a str,
    pub definitions: &'a Source,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Artifacts(BTreeMap<Emit, String>);
impl FromIterator<(Emit, String)> for Artifacts {
    fn from_iter<T: IntoIterator<Item = (Emit, String)>>(iter: T) -> Self {
        Self(iter.into_iter().collect())
    }
}
impl std::ops::Index<Emit> for Artifacts {
    type Output = String;
    fn index(&self, kind: Emit) -> &String {
        self.0.get(&kind).expect("artifact was not selected")
    }
}
impl Artifacts {
    pub fn get(&self, artifact: Emit) -> Option<&str> {
        self.0.get(&artifact).map(String::as_str)
    }
    pub fn iter(&self) -> impl Iterator<Item = (Emit, &str)> {
        self.0.iter().map(|(&kind, text)| (kind, text.as_str()))
    }
    /// Generation finishes before any file is written. Unselected files are untouched.
    pub fn write(&self, dir: &Path) -> std::io::Result<Vec<PathBuf>> {
        std::fs::create_dir_all(dir)?;
        self.iter()
            .map(|(kind, text)| {
                let path = dir.join(kind.filename());
                std::fs::write(&path, text)?;
                Ok(path)
            })
            .collect()
    }
}
impl Source {
    pub fn generate(
        &self,
        selected: &[Emit],
        options: Options<'_>,
    ) -> Result<Artifacts, SourceError> {
        let fail = |message: &str| self.locate(crate::Error::at(self.text(), 0, message));
        if selected.is_empty() {
            return Err(fail("select at least one output artifact"));
        }
        self.check_imports()?;
        let mut output = BTreeMap::new();
        let mut ir = None;
        let mut target = None;
        for &kind in selected {
            if output.contains_key(&kind) {
                continue;
            }
            let text = match kind {
                Emit::Rules => {
                    let config = options.rules.as_ref().ok_or_else(|| {
                        fail("rules require source/target definitions and Rust bindings")
                    })?;
                    let mut dialects = crate::rules::Dialects::default();
                    dialects
                        .insert(config.rust.source.0, &config.source.parse()?)
                        .map_err(|e| self.locate(e))?;
                    dialects
                        .insert(config.rust.target.0, &config.target.parse()?)
                        .map_err(|e| self.locate(e))?;
                    let mut program = crate::rules::Program::from_declarations(
                        self.text(),
                        self.declarations(),
                        &dialects,
                    )
                    .map_err(|e| self.locate(e))?;
                    if config.infer_primitives {
                        program
                            .infer_primitives(&dialects, config.rust.source.0, config.rust.target.0)
                            .map_err(|e| self.locate(e))?;
                    }
                    program.rust(config.rust).map_err(|e| self.locate(e))?
                }
                Emit::Decisions => {
                    let config = options.decisions.as_ref().ok_or_else(|| {
                        fail("decisions require operation definitions and Rust bindings")
                    })?;
                    self.decisions(&config.definitions.parse()?, config.rust)?
                }
                Emit::Interfaces => self.interfaces(
                    options
                        .interfaces
                        .ok_or_else(|| fail("interfaces require an explicit Rust namespace"))?,
                )?,
                Emit::DataTypes => self.data_types()?,
                kind if kind.target() => {
                    if target.is_none() {
                        let config = options.target.as_ref().ok_or_else(|| {
                            fail("target output requires target architecture and definitions")
                        })?;
                        target = Some(crate::target::Plan::prepare(
                            self,
                            config.arch,
                            config.context,
                            config.definitions,
                            config.input,
                        )?);
                    }
                    target.as_ref().unwrap().emit(kind)
                }
                _ => {
                    if ir.is_none() {
                        ir = Some(self.plan()?);
                    }
                    let plan = ir.as_ref().unwrap();
                    if !plan.supports(kind) {
                        return Err(fail(&format!(
                            "artifact `{kind}` is not supported by this instruction storage"
                        )));
                    }
                    plan.emit(kind)
                }
            };
            output.insert(kind, text);
        }
        Ok(Artifacts(output))
    }
}
