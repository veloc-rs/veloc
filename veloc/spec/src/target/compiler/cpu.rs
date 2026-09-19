//! Resolve feature dependencies once and generate target-local typed capabilities.
use crate::target::ast::{Def, FeatureDef, Module};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;

fn dependencies(
    index: usize,
    features: &[FeatureDef],
    names: &BTreeMap<&str, usize>,
    active: &mut BTreeSet<usize>,
) -> Result<BTreeSet<usize>, String> {
    if !active.insert(index) {
        return Err(format!(
            "cyclic feature dependency at {}",
            features[index].name
        ));
    }
    let mut result = BTreeSet::new();
    for name in &features[index].requires {
        let &required = names
            .get(name.as_str())
            .ok_or_else(|| format!("unknown feature dependency {name}"))?;
        result.insert(required);
        result.extend(dependencies(required, features, names, active)?);
    }
    active.remove(&index);
    Ok(result)
}

fn set(indices: impl IntoIterator<Item = usize>, features: &[FeatureDef]) -> String {
    indices
        .into_iter()
        .fold("FeatureSet::empty()".into(), |set, i| {
            format!("{set}.with(Feature::{})", features[i].name)
        })
}

pub(super) struct Plan {
    features: Vec<FeatureDef>,
    closures: Vec<BTreeSet<usize>>,
    cpus: Vec<(String, BTreeSet<usize>)>,
}
impl Plan {
    pub(super) fn prepare(module: &Module) -> Result<Self, String> {
        let features: Vec<_> = module
            .defs
            .iter()
            .filter_map(|d| match d {
                Def::Feature(f) => Some(f.clone()),
                _ => None,
            })
            .collect();
        let mut names = BTreeMap::new();
        let mut spellings = BTreeSet::new();
        for (i, f) in features.iter().enumerate() {
            if !spellings.insert(f.name.to_ascii_lowercase()) {
                return Err(format!("duplicate feature {}", f.name));
            }
            names.insert(f.name.as_str(), i);
        }
        let closures = (0..features.len())
            .map(|i| dependencies(i, &features, &names, &mut BTreeSet::new()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut names_seen = BTreeSet::new();
        let mut cpus = Vec::new();
        for def in &module.defs {
            let Def::Cpu(cpu) = def else {
                continue;
            };
            if !names_seen.insert(&cpu.name) {
                return Err(format!("duplicate CPU {}", cpu.name));
            }
            let mut enabled = BTreeSet::new();
            for name in &cpu.features {
                let &index = names
                    .get(name.as_str())
                    .ok_or_else(|| format!("CPU {} references unknown feature {name}", cpu.name))?;
                enabled.insert(index);
                enabled.extend(closures[index].iter().copied());
            }
            cpus.push((cpu.name.clone(), enabled));
        }
        Ok(Self {
            features,
            closures,
            cpus,
        })
    }
    pub(super) fn generate(&self, out: &mut String) {
        let features = &self.features;
        let closures = &self.closures;
        out.push_str("#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub enum Feature {\n");
        for f in features {
            writeln!(out, "#[doc = {:?}] {},", f.doc, f.name).unwrap();
        }
        out.push_str("}\nconst ALL_FEATURES: &[Feature] = &[\n");
        for f in features {
            writeln!(out, "Feature::{},", f.name).unwrap();
        }
        out.push_str("];\nimpl Feature { pub const fn name(self) -> &'static str { match self {\n");
        for f in features {
            writeln!(out, "Self::{} => {:?},", f.name, f.name).unwrap();
        }
        out.push_str("} }\npub fn parse(name: &str) -> Option<Self> { ALL_FEATURES.iter().copied().find(|f| f.name().eq_ignore_ascii_case(name)) }\n");
        out.push_str("pub const fn requires(self) -> FeatureSet { match self {\n");
        for (i, f) in features.iter().enumerate() {
            writeln!(
                out,
                "Self::{} => {},",
                f.name,
                set(closures[i].iter().copied(), &features)
            )
            .unwrap();
        }
        out.push_str("} } }\n");
        let words = features.len().div_ceil(64).max(1);
        writeln!(
            out,
            "#[derive(Debug, Clone, Copy, PartialEq, Eq)]\npub struct FeatureSet([u64; {words}]);"
        )
        .unwrap();
        writeln!(
            out,
            "impl FeatureSet {{ pub const fn empty() -> Self {{ Self([0; {words}]) }}"
        )
        .unwrap();
        out.push_str(r#"
    pub const fn as_words(&self) -> &[u64] { &self.0 }
    pub const fn contains(self, feature: Feature) -> bool {
        let index = feature as usize;
        self.0[index / 64] & (1u64 << (index % 64)) != 0
    }
    pub const fn with(mut self, feature: Feature) -> Self {
        let index = feature as usize;
        self.0[index / 64] |= 1u64 << (index % 64);
        self
    }
    pub const fn without(mut self, feature: Feature) -> Self {
        let index = feature as usize;
        self.0[index / 64] &= !(1u64 << (index % 64));
        self
    }
    pub const fn union(mut self, other: Self) -> Self {
        let mut i = 0;
        while i < self.0.len() { self.0[i] |= other.0[i]; i += 1; }
        self
    }
    pub const fn contains_all(self, required: Self) -> bool {
        let mut i = 0;
        while i < self.0.len() {
            if self.0[i] & required.0[i] != required.0[i] { return false; }
            i += 1;
        }
        true
    }
    pub fn iter(self) -> impl Iterator<Item = Feature> {
        ALL_FEATURES.iter().copied().filter(move |&f| self.contains(f))
    }
    /// Resolve user overrides once. The last setting for a feature wins;
    /// dependencies are enabled unless explicitly disabled, which is an error.
    pub fn resolve(mut self, overrides: &[alloc::string::String]) -> Result<Self, alloc::string::String> {
        let mut disabled = Self::empty();
        for option in overrides {
            let (enabled, name) = if let Some(name) = option.strip_prefix('-') {
                (false, name)
            } else { (true, option.strip_prefix('+').unwrap_or(option)) };
            let feature = Feature::parse(name)
                .ok_or_else(|| alloc::format!("unknown target feature: {name}"))?;
            if enabled {
                self = self.with(feature);
                disabled = disabled.without(feature);
            } else {
                self = self.without(feature);
                disabled = disabled.with(feature);
            }
        }
        let mut resolved = self;
        for feature in self.iter() {
            let required = feature.requires();
            for dependency in required.iter() {
                if disabled.contains(dependency) {
                    return Err(alloc::format!("{} requires explicitly disabled feature {}", feature.name(), dependency.name()));
                }
            }
            resolved = resolved.union(required);
        }
        Ok(resolved)
    }
}
#[derive(Debug, Clone, Copy)]
pub struct CpuModel {
    pub name: &'static str,
    pub features: FeatureSet,
}
pub const SUPPORTED_CPUS: &[CpuModel] = &[
"#);
        for (name, enabled) in &self.cpus {
            writeln!(
                out,
                "CpuModel {{ name: {:?}, features: {} }},",
                name,
                set(enabled.iter().copied(), features)
            )
            .unwrap();
        }
        out.push_str("];\n");
    }
}
