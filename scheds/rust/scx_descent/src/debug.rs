// Phase 2: Gradient debugging output
// --debug-gradients support

use std::io::Write;
use std::time::{Duration, Instant};

use crate::optimizer::DescentOptimizer;

pub struct GradientDebugger {
    interval: Duration,
    last_output: Instant,
    format: DebugFormat,
}

#[derive(Clone, Copy)]
pub enum DebugFormat {
    Text,
    Json,
}

impl GradientDebugger {
    pub fn new(interval_ms: u64) -> Self {
        Self {
            interval: Duration::from_millis(interval_ms),
            last_output: Instant::now(),
            format: DebugFormat::Text,
        }
    }

    pub fn maybe_output(&mut self, optimizer: &DescentOptimizer) {
        if self.last_output.elapsed() >= self.interval {
            self.output(optimizer);
            self.last_output = Instant::now();
        }
    }

    fn output(&self, optimizer: &DescentOptimizer) {
        match self.format {
            DebugFormat::Text => self.output_text(optimizer),
            DebugFormat::Json => self.output_json(optimizer),
        }
    }

    fn output_text(&self, optimizer: &DescentOptimizer) {
        eprintln!("[scx_descent debug] Gradient state at {:?}", Instant::now());
        for ((cpu, class), state) in optimizer.get_states() {
            let params = state.get_params_fixed();
            let avg_loss = state.loss_history.back().copied().unwrap_or(0.0);
            eprintln!(
                "  CPU {} class {}: params=[{}, {}, {}, {}, {}], loss={:.2}",
                cpu, class, params[0], params[1], params[2], params[3], params[4], avg_loss
            );
        }
    }

    fn output_json(&self, _optimizer: &DescentOptimizer) {
        // JSON output for external analysis tools
        // TODO: Implement if needed
    }
}
