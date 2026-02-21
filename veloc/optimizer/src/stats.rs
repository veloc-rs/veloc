use crate::metrics::Metrics;
use alloc::string::String;
use alloc::vec::Vec;
use std::time::{Duration, Instant};

/// 优化 Pass 执行的统计信息。
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    pub name: String,
    pub duration: Duration,
}

/// 汇总所有 Pass 的统计结果。
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub passes: Vec<PassStats>,
    pub total_duration: Duration,
    pub metrics: Metrics,
    pub session_start: Option<Instant>,
}

impl PipelineStats {
    pub fn start_session(&mut self) {
        self.session_start = Some(Instant::now());
        self.passes.clear();
        self.metrics.counters.clear();
    }
}

impl core::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "\n===== Optimization Pipeline Statistics =====")?;
        writeln!(f, "{:<25} | {:>12}", "Pass Name", "Duration")?;
        writeln!(f, "{:-<40}", "")?;
        for stats in &self.passes {
            writeln!(f, "{:<25} | {:>12?}", stats.name, stats.duration)?;
        }
        writeln!(f, "{:-<40}", "")?;
        writeln!(f, "{:<25} | {:>12?}", "Total", self.total_duration)?;
        writeln!(f, "===========================================\n")?;
        write!(f, "{}", self.metrics)?;
        Ok(())
    }
}

impl PipelineStats {
    /// 导出为 Chrome Trace 事件格式 (JSON)。
    /// 可以在 Chrome 浏览器中访问 chrome://tracing 进行查看。
    pub fn dump_chrome_trace(&self) -> String {
        let mut json = String::from("[\n");
        let mut first = true;
        let mut current_ts_offset_us = 0u64;

        for stats in &self.passes {
            if !first {
                json.push_str(",\n");
            }
            first = false;

            let dur_us = stats.duration.as_micros() as u64;
            // 构造 Chrome Trace Event Format (Complete Event 'X')
            json.push_str(&format!(
                r#"  {{"name": "{}", "ph": "X", "ts": {}, "dur": {}, "pid": 1, "tid": 1}}"#,
                stats.name, current_ts_offset_us, dur_us
            ));

            current_ts_offset_us += dur_us;
        }

        json.push_str("\n]");
        json
    }
}

/// 一个任务记录器，用于记录 Pass 的执行时间。
pub struct TimingGuard {
    name: String,
    start_time: Instant,
}

impl TimingGuard {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            start_time: Instant::now(),
        }
    }

    pub fn finish(self, stats: &mut PipelineStats) {
        let duration = self.start_time.elapsed();
        stats.passes.push(PassStats {
            name: self.name.clone(),
            duration,
        });
    }
}
