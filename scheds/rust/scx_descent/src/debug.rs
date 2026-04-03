// Phase 5: Thompson Sampling debug output
// --debug-gradients now shows Thompson sampler state

use std::io::Write;
use std::time::{Duration, Instant};

use crate::optimizer_thompson::ThompsonSampler;

pub struct ThompsonDebugger {
    interval: Duration,
    last_output: Instant,
    format: DebugFormat,
}

#[derive(Clone, Copy)]
pub enum DebugFormat {
    Text,
    Json,
}

impl ThompsonDebugger {
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval: Duration::from_millis(interval_ms),
            last_output: Instant::now(),
            format: DebugFormat::Text,
        }
    }

    pub fn maybe_output(&mut self, thompson: &ThompsonSampler) {
        if self.last_output.elapsed() >= self.interval {
            self.output(thompson);
            self.last_output = Instant::now();
        }
    }

    fn output(&self, thompson: &ThompsonSampler) {
        match self.format {
            DebugFormat::Text => self.output_text(thompson),
            DebugFormat::Json => self.output_json(thompson),
        }
    }

    fn output_text(&self, thompson: &ThompsonSampler) {
        let (total_posteriors, total_obs, avg_uncertainty) = thompson.get_stats();

        eprintln!(
            "[scx_descent Thompson Sampling] State at {:?}",
            Instant::now()
        );
        eprintln!(
            "  Total posteriors: {} | Total observations: {:.0} | Avg uncertainty: {:.2}",
            total_posteriors, total_obs, avg_uncertainty
        );

        // Show stats for CPU 0 as representative sample
        let cpu = 0;
        for class in 0..4u32 {
            let class_name = match class {
                0 => "LATENCY_CRITICAL",
                1 => "NORMAL",
                2 => "HOG",
                3 => "BACKGROUND",
                _ => "UNKNOWN",
            };

            for param_idx in 0..5usize {
                if let Some(posterior) = thompson.get_posterior(cpu, class, param_idx) {
                    eprintln!(
                        "  CPU {} class {} param {}: mean={:.0} std={:.0} n={:.0} [{}, {}]",
                        cpu,
                        class_name,
                        param_idx,
                        posterior.mean,
                        posterior.std,
                        posterior.n_observations,
                        posterior.min as u64,
                        posterior.max as u64
                    );
                }
            }

            if let Some(baseline) = thompson.get_baseline(cpu, class) {
                eprintln!(
                    "  CPU {} class {} baseline loss: {:.2}",
                    cpu, class_name, baseline
                );
            }
        }
    }

    fn output_json(&self, _thompson: &ThompsonSampler) {
        // JSON output for external analysis tools
        // TODO: Implement if needed
    }
}
