//! Definition-file imports, dependency tracking and original-file diagnostics.
//! Files are parsed independently; imports cannot complete another file's syntax.

mod scopes;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::{Definitions, Error, Generated, Plan, model, syntax};

#[derive(Debug)]
pub struct SourceError {
    pub path: PathBuf,
    pub diagnostic: Error,
}

impl std::fmt::Display for SourceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.path.display(), self.diagnostic)
    }
}

impl std::error::Error for SourceError {}

pub(crate) struct File {
    pub(crate) path: PathBuf,
    first_line: usize,
    pub(crate) declarations: std::ops::Range<usize>,
    pub(crate) visible: BTreeSet<usize>,
}

pub struct Source {
    text: String,
    declarations: Vec<syntax::Decl>,
    files: Vec<File>,
    dependencies: BTreeSet<PathBuf>,
}

impl Source {
    /// Load imports relative to the physical importing file. Canonical identity
    /// deduplicates diamond imports and detects aliases in import cycles.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, SourceError> {
        let mut source = Self {
            text: String::new(),
            declarations: Vec::new(),
            files: Vec::new(),
            dependencies: BTreeSet::new(),
        };
        Loader {
            source: &mut source,
            active: Vec::new(),
            loaded: BTreeMap::new(),
            next_line: 1,
        }
        .visit(path.as_ref())?;
        Ok(source)
    }

    /// Includes requested paths and canonical targets so changing an import
    /// symlink also invalidates Cargo's build-script cache.
    pub fn dependencies(&self) -> impl Iterator<Item = &Path> {
        self.dependencies.iter().map(PathBuf::as_path)
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }
    pub(crate) fn declarations(&self) -> &[syntax::Decl] {
        &self.declarations
    }
    pub(crate) fn files(&self) -> &[File] {
        &self.files
    }
    pub fn interfaces(&self, namespace: &str) -> Result<String, SourceError> {
        crate::interfaces::generate(self, namespace).map_err(|e| self.locate(e))
    }

    pub fn parse(&self) -> Result<Definitions, SourceError> {
        let defs = model::from_declarations(&self.text, self.declarations.clone())
            .map_err(|e| self.locate(e))?;
        scopes::check(&self.text, &self.declarations, &self.files, &defs)
            .map_err(|e| self.locate(e))?;
        Ok(defs)
    }

    /// Prepare checked output projections while retaining original-file diagnostics.
    pub fn plan(&self) -> Result<Plan, SourceError> {
        Plan::prepare(self.parse()?, &self.text).map_err(|e| self.locate(e))
    }

    pub fn compile(&self) -> Result<Generated, SourceError> {
        Ok(self.plan()?.generate())
    }

    fn locate(&self, mut diagnostic: Error) -> SourceError {
        let index = self
            .files
            .partition_point(|file| file.first_line <= diagnostic.line)
            .saturating_sub(1);
        let file = &self.files[index];
        diagnostic.line = diagnostic.line.saturating_sub(file.first_line) + 1;
        SourceError {
            path: file.path.clone(),
            diagnostic,
        }
    }
}

struct Loader<'a> {
    source: &'a mut Source,
    active: Vec<PathBuf>,
    loaded: BTreeMap<PathBuf, usize>,
    next_line: usize,
}

impl Loader<'_> {
    fn visit(&mut self, path: &Path) -> Result<usize, SourceError> {
        self.source.dependencies.insert(path.to_owned());
        let canonical = path
            .canonicalize()
            .map_err(|e| file_error(path, e.to_string()))?;
        self.source.dependencies.insert(canonical.clone());
        if self.active.contains(&canonical) {
            let mut chain = self
                .active
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>();
            chain.push(canonical.display().to_string());
            return Err(file_error(
                &canonical,
                format!("import cycle: {}", chain.join(" -> ")),
            ));
        }
        if let Some(&file) = self.loaded.get(&canonical) {
            return Ok(file);
        }
        if self.active.len() >= 128 {
            return Err(file_error(
                &canonical,
                "import nesting exceeds 128 files".into(),
            ));
        }
        let text = std::fs::read_to_string(&canonical)
            .map_err(|e| file_error(&canonical, e.to_string()))?;
        let located = |diagnostic| SourceError {
            path: canonical.clone(),
            diagnostic,
        };
        let syntax::File {
            imports,
            mut declarations,
        } = syntax::parse_file(&text).map_err(located)?;
        for import in &imports {
            if import.path.is_empty() || Path::new(&import.path).is_absolute() {
                return Err(located(Error::at(
                    &text,
                    import.offset,
                    "import requires a nonempty relative path",
                )));
            }
        }
        self.active.push(canonical.clone());
        let mut visible = BTreeSet::new();
        for import in imports {
            let target = canonical
                .parent()
                .expect("canonical file has a parent")
                .join(&import.path);
            let child = self.visit(&target).map_err(|mut error| {
                let site = Error::at(&text, import.offset, "");
                error.diagnostic.message.push_str(&format!(
                    "\n  imported from {}:{}:{}",
                    canonical.display(),
                    site.line,
                    site.column,
                ));
                error
            })?;
            visible.extend(&self.source.files[child].visible);
        }
        self.active.pop();
        let base = self.source.text.len();
        for record in &mut declarations {
            record.relocate(base);
        }
        let file = self.source.files.len();
        visible.insert(file);
        let start = self.source.declarations.len();
        self.source.files.push(File {
            path: canonical.clone(),
            first_line: self.next_line,
            declarations: start..start + declarations.len(),
            visible,
        });
        self.source.declarations.extend(declarations);
        self.source.text.push_str(&text);
        self.next_line += text.bytes().filter(|&b| b == b'\n').count();
        if !text.ends_with('\n') {
            self.source.text.push('\n');
            self.next_line += 1;
        }
        self.loaded.insert(canonical, file);
        Ok(file)
    }
}

fn file_error(path: &Path, message: String) -> SourceError {
    SourceError {
        path: path.to_owned(),
        diagnostic: Error {
            line: 1,
            column: 1,
            message,
        },
    }
}
