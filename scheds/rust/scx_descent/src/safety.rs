// Phase 2: Safety mechanisms
// - Parameter bounds checking
// - Oscillation detection
// - Checkpoint/rollback

use std::collections::VecDeque;
use std::time::{Duration, Instant};

pub struct SafetyMonitor {
    max_loss_increase_pct: f64,
    checkpoint_interval: Duration,
    last_checkpoint: Instant,
    loss_history: VecDeque<f64>,
}

impl SafetyMonitor {
    pub fn new() -> Self {
        Self {
            max_loss_increase_pct: 0.2, // 20%
            checkpoint_interval: Duration::from_secs(10),
            last_checkpoint: Instant::now(),
            loss_history: VecDeque::with_capacity(100),
        }
    }

    pub fn should_checkpoint(&self) -> bool {
        self.last_checkpoint.elapsed() >= self.checkpoint_interval
    }

    pub fn record_checkpoint(&mut self) {
        self.last_checkpoint = Instant::now();
    }

    pub fn is_degradation(&self, current_loss: f64, checkpoint_loss: f64) -> bool {
        if checkpoint_loss == 0.0 {
            return false;
        }
        let increase = (current_loss - checkpoint_loss) / checkpoint_loss;
        increase > self.max_loss_increase_pct
    }

    pub fn record_loss(&mut self, loss: f64) {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 100 {
            self.loss_history.pop_front();
        }
    }

    pub fn get_loss_trend(&self) -> f64 {
        if self.loss_history.len() < 10 {
            return 0.0;
        }

        let recent: Vec<_> = self.loss_history.iter().rev().take(10).collect();
        let older: Vec<_> = self.loss_history.iter().rev().skip(10).take(10).collect();

        if older.is_empty() {
            return 0.0;
        }

        let recent_avg: f64 = recent.iter().copied().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().copied().sum::<f64>() / older.len() as f64;

        if older_avg == 0.0 {
            return 0.0;
        }

        (recent_avg - older_avg) / older_avg
    }
}
