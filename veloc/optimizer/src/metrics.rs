use alloc::string::String;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// 全局或 Session 级别的统计指标。
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub counters: HashMap<String, u64>,
}

impl Metrics {
    pub fn add(&mut self, name: &str, delta: u64) {
        *self.counters.entry(name.to_string()).or_insert(0) += delta;
    }
}

impl core::fmt::Display for Metrics {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        if self.counters.is_empty() {
            return Ok(());
        }
        writeln!(f, "\n===== Optimization Metric Counters =====")?;
        let mut keys: Vec<_> = self.counters.keys().collect();
        keys.sort();
        for key in keys {
            writeln!(f, "{:<30} : {:>10}", key, self.counters[key])?;
        }
        writeln!(f, "========================================\n")?;
        Ok(())
    }
}
