/// Borrowed feature bits. Both sets in a comparison must use the same target's
/// feature numbering; absent words represent disabled features.
#[derive(Debug, Clone, Copy)]
pub struct FeatureSetRef<'a> {
    words: &'a [u64],
}

impl<'a> FeatureSetRef<'a> {
    pub const fn new(words: &'a [u64]) -> Self {
        Self { words }
    }

    pub fn contains_all(self, required: FeatureSetRef<'_>) -> bool {
        required
            .words
            .iter()
            .enumerate()
            .all(|(index, mask)| self.words.get(index).copied().unwrap_or(0) & mask == *mask)
    }
}

/// Target names taken from the same schemas used by code generation.
pub struct TargetCapabilities {
    pub cpus: std::vec::Vec<&'static str>,
    pub features: std::vec::Vec<&'static str>,
}

pub fn capabilities(arch: super::TargetArch) -> crate::Result<TargetCapabilities> {
    match arch {
        super::TargetArch::X86_64 => {
            use super::x86_64::inst;
            Ok(TargetCapabilities {
                cpus: inst::SUPPORTED_CPUS.iter().map(|cpu| cpu.name).collect(),
                features: inst::ALL_FEATURES
                    .iter()
                    .map(|feature| feature.name())
                    .collect(),
            })
        }
        super::TargetArch::Riscv64 => {
            use super::riscv64::inst;
            Ok(TargetCapabilities {
                cpus: inst::SUPPORTED_CPUS.iter().map(|cpu| cpu.name).collect(),
                features: inst::ALL_FEATURES
                    .iter()
                    .map(|feature| feature.name())
                    .collect(),
            })
        }
        _ => Err(crate::Error::target_machine_unavailable(arch)),
    }
}
