//! Safety Monitoring for scx_descent
//!
//! Provides protection against catastrophic parameter configurations.
//! Monitors loss values and triggers rollback to checkpointed parameters
//! when loss increases beyond threshold (default: 500%).
//!
//! Features:
//! - Automatic checkpointing every N updates
//! - Loss spike detection using EWMA baseline
//! - Warm-up period before safety checks activate
//! - Minimum baseline threshold to prevent over-sensitivity
//! - Parameter restoration with posterior reset
//! - Restoration tracking for monitoring

// Phase 2: Safety mechanisms
// - Parameter bounds checking
// - Oscillation detection
// - Checkpoint/rollback

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

/// Default maximum loss increase percentage (500% = 6x baseline)
const DEFAULT_MAX_LOSS_INCREASE_PCT: f64 = 500.0;
/// Minimum number of observations before safety checks activate (warm-up period)
const MIN_OBSERVATIONS_FOR_SAFETY: usize = 20;
/// Minimum baseline loss value for safety checks to activate
/// Prevents triggering on tiny loss values during initial warm-up
const MIN_BASELINE_FOR_SAFETY: f64 = 50.0;
/// EWMA alpha for loss history smoothing
const LOSS_EWMA_ALPHA: f64 = 0.3;

/// Checkpoint storing parameter state for rollback
struct Checkpoint {
    params: [u64; 5],
    #[allow(dead_code)] // Stored for debugging/analytics purposes
    loss: f64,
    #[allow(dead_code)] // Stored for debugging/analytics purposes
    timestamp: Instant,
}

/// Statistics about safety monitoring activity
pub struct SafetyStats {
    pub checkpoints_stored: usize,
    pub loss_history_size: usize,
    pub restorations: usize,
}

/// Safety monitor that detects catastrophic loss spikes and enables rollback
///
/// Tracks loss history using EWMA and creates periodic checkpoints for each
/// (CPU, class) combination. When a loss spike exceeds the threshold,
/// restores the last known good checkpoint.
pub struct SafetyMonitor {
    max_loss_increase_pct: f64,
    loss_history: VecDeque<f64>,
    last_checkpoint: HashMap<(u32, u32), Checkpoint>,
    checkpoint_interval: usize,
    update_counter: HashMap<(u32, u32), usize>,
    /// Per-(cpu, class) observation count for warm-up period
    observation_count: HashMap<(u32, u32), usize>,
    restoration_count: usize,
}

impl SafetyMonitor {
    /// Create a new safety monitor with the specified threshold
    ///
    /// # Arguments
    /// * `max_loss_increase_pct` - Percentage increase that triggers rollback (e.g., 500.0 = 500%)
    pub fn new_with_threshold(max_loss_increase_pct: f64) -> Self {
        Self {
            max_loss_increase_pct,
            loss_history: VecDeque::with_capacity(100),
            last_checkpoint: HashMap::new(),
            checkpoint_interval: 10,
            update_counter: HashMap::new(),
            observation_count: HashMap::new(),
            restoration_count: 0,
        }
    }

    /// Create a new safety monitor with default 500% threshold
    pub fn new() -> Self {
        Self::new_with_threshold(DEFAULT_MAX_LOSS_INCREASE_PCT)
    }

    /// Check if current loss is safe. Returns checkpoint params if unsafe.
    ///
    /// This method:
    /// 1. Tracks observation count for warm-up period
    /// 2. Computes baseline loss from EWMA history
    /// 3. Skips safety checks during warm-up or with small baselines
    /// 4. Checks for catastrophic spike against threshold
    /// 5. Creates periodic checkpoints every N updates
    /// 6. Updates loss history with new observation
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class
    /// * `current_loss` - Current observed loss value
    /// * `current_params` - Current parameter values being tested
    ///
    /// Returns Some(checkpoint_params) if loss spike detected, None otherwise
    pub fn check_and_protect(
        &mut self,
        cpu: u32,
        class: u32,
        current_loss: f64,
        current_params: &[u64; 5],
    ) -> Option<[u64; 5]> {
        // Track observations for this (cpu, class)
        let obs_count = self.observation_count.entry((cpu, class)).or_insert(0);
        *obs_count += 1;

        // Skip safety check during warm-up period
        if *obs_count >= MIN_OBSERVATIONS_FOR_SAFETY {
            // Get baseline (EWMA of recent losses)
            let baseline = self.loss_history.back().copied().unwrap_or(current_loss);

            // Only check for spikes if baseline is established and large enough
            if baseline >= MIN_BASELINE_FOR_SAFETY {
                // Calculate threshold: baseline * (1 + percentage/100)
                let threshold = baseline * (1.0 + self.max_loss_increase_pct / 100.0);

                if current_loss > threshold {
                    // Loss spike detected - try to restore checkpoint
                    if let Some(checkpoint) = self.last_checkpoint.get(&(cpu, class)) {
                        /*eprintln!(
                            "Safety: Loss spike on CPU {} class {} - restoring checkpoint (loss={:.0}, threshold={:.0}, baseline={:.0}, +{:.0}%)",
                            cpu, class, current_loss, threshold, baseline, self.max_loss_increase_pct
                        );*/
                        self.restoration_count += 1;
                        return Some(checkpoint.params);
                    }
                }
            }
        }

        // Record checkpoint periodically
        let counter = self.update_counter.entry((cpu, class)).or_insert(0);
        *counter += 1;

        if *counter % self.checkpoint_interval == 0 {
            self.last_checkpoint.insert(
                (cpu, class),
                Checkpoint {
                    params: *current_params,
                    loss: current_loss,
                    timestamp: Instant::now(),
                },
            );
        }

        // Update loss history (EWMA)
        let new_baseline = if self.loss_history.is_empty() {
            current_loss
        } else {
            let baseline = self.loss_history.back().copied().unwrap_or(current_loss);
            LOSS_EWMA_ALPHA * current_loss + (1.0 - LOSS_EWMA_ALPHA) * baseline
        };
        self.loss_history.push_back(new_baseline);
        if self.loss_history.len() > 100 {
            self.loss_history.pop_front();
        }

        None
    }

    /// Get current safety statistics
    pub fn get_stats(&self) -> SafetyStats {
        SafetyStats {
            checkpoints_stored: self.last_checkpoint.len(),
            loss_history_size: self.loss_history.len(),
            restorations: self.restoration_count,
        }
    }

    /// Legacy: Check if degradation threshold exceeded
    #[allow(dead_code)] // Deprecated but kept for compatibility
    #[deprecated(note = "Use check_and_protect instead")]
    pub fn is_degradation(&self, current_loss: f64, checkpoint_loss: f64) -> bool {
        if checkpoint_loss == 0.0 {
            return false;
        }
        let increase = (current_loss - checkpoint_loss) / checkpoint_loss;
        increase > (self.max_loss_increase_pct / 100.0)
    }

    /// Legacy: Record a loss value
    #[allow(dead_code)] // Deprecated but kept for compatibility
    #[deprecated(note = "Use check_and_protect instead")]
    pub fn record_loss(&mut self, loss: f64) {
        self.loss_history.push_back(loss);
        if self.loss_history.len() > 100 {
            self.loss_history.pop_front();
        }
    }

    /// Legacy: Get loss trend
    #[allow(dead_code)] // Deprecated but kept for compatibility
    #[deprecated(note = "Use get_stats instead")]
    pub fn get_loss_trend(&self) -> f64 {
        if self.loss_history.len() < 10 {
            return 0.0;
        }

        let recent: Vec<_> = self.loss_history.iter().rev().take(10).copied().collect();
        let older: Vec<_> = self
            .loss_history
            .iter()
            .rev()
            .skip(10)
            .take(10)
            .copied()
            .collect();

        if older.is_empty() {
            return 0.0;
        }

        let recent_avg: f64 = recent.iter().sum::<f64>() / recent.len() as f64;
        let older_avg: f64 = older.iter().sum::<f64>() / older.len() as f64;

        if older_avg == 0.0 {
            return 0.0;
        }

        (recent_avg - older_avg) / older_avg
    }
}

impl Default for SafetyMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_monitor_creation() {
        let monitor = SafetyMonitor::new();
        let stats = monitor.get_stats();
        assert_eq!(stats.checkpoints_stored, 0);
        assert_eq!(stats.loss_history_size, 0);
        assert_eq!(stats.restorations, 0);
    }

    #[test]
    fn test_safety_monitor_with_threshold() {
        let monitor = SafetyMonitor::new_with_threshold(50.0);
        let stats = monitor.get_stats();
        assert_eq!(stats.checkpoints_stored, 0);
    }

    #[test]
    fn test_warm_up_period_no_spike_detection() {
        // Use a low threshold to test warm-up logic
        let mut monitor = SafetyMonitor::new_with_threshold(100.0);
        let params = [100u64; 5];

        // During warm-up (first 20 observations), even huge spikes should NOT trigger
        // because we don't have enough data to establish a reliable baseline
        for i in 0..MIN_OBSERVATIONS_FOR_SAFETY - 1 {
            // Huge spike that would normally trigger
            let result = monitor.check_and_protect(0, 0, 1000.0, &params);
            assert!(
                result.is_none(),
                "Should not trigger during warm-up on iteration {}",
                i + 1
            );
        }

        // After warm-up, spikes should be detected
        // First need to establish a baseline with stable loss
        for _ in 0..10 {
            monitor.check_and_protect(0, 0, 50.0, &params);
        }

        // Now trigger a spike (300% increase, above 100% threshold)
        // Note: We use 500% default, so use a custom monitor with 100% for this test
        let result = monitor.check_and_protect(0, 0, 500.0, &params);
        assert!(result.is_some(), "Should detect spike after warm-up");
    }

    #[test]
    fn test_check_and_protect_no_spike() {
        let mut monitor = SafetyMonitor::new_with_threshold(100.0);
        let params = [100u64; 5];

        // First call establishes baseline (but during warm-up)
        let result = monitor.check_and_protect(0, 0, 50.0, &params);
        assert!(result.is_none());

        // Continue past warm-up with stable loss
        for _ in 0..25 {
            let _ = monitor.check_and_protect(0, 0, 50.0, &params);
        }

        // Second call with similar loss - no spike
        let result = monitor.check_and_protect(0, 0, 55.0, &params);
        assert!(result.is_none());

        // Stats should show history
        let stats = monitor.get_stats();
        assert!(stats.loss_history_size > 20);
    }

    #[test]
    fn test_check_and_protect_spike_detection() {
        // Use 100% threshold for easier testing
        let mut monitor = SafetyMonitor::new_with_threshold(100.0);
        let params1 = [100u64; 5];
        let params2 = [200u64; 5];

        // Establish baseline past warm-up period with stable loss
        for _ in 0..MIN_OBSERVATIONS_FOR_SAFETY + 10 {
            let _ = monitor.check_and_protect(0, 0, 100.0, &params1);
        }

        // Create checkpoint by continuing to 10th update
        for _ in 0..10 {
            let _ = monitor.check_and_protect(0, 0, 100.0, &params1);
        }

        // Now trigger a spike (300% increase, well above 100% threshold)
        let result = monitor.check_and_protect(0, 0, 400.0, &params2);
        assert!(
            result.is_some(),
            "Should detect spike and return checkpoint params"
        );
        assert_eq!(result.unwrap(), params1);

        // Stats should show restoration
        let stats = monitor.get_stats();
        assert_eq!(stats.restorations, 1);
    }

    #[test]
    fn test_minimum_baseline_prevents_spurious_triggers() {
        // Use 100% threshold
        let mut monitor = SafetyMonitor::new_with_threshold(100.0);
        let params = [100u64; 5];

        // Pass warm-up with tiny baseline
        for _ in 0..MIN_OBSERVATIONS_FOR_SAFETY + 10 {
            let _ = monitor.check_and_protect(0, 0, 1.0, &params);
        }

        // Even a 1000% increase from tiny baseline should not trigger
        // because baseline is below MIN_BASELINE_FOR_SAFETY (50.0)
        let result = monitor.check_and_protect(0, 0, 50.0, &params);
        assert!(
            result.is_none(),
            "Should not trigger when baseline is below minimum threshold"
        );
    }

    #[test]
    fn test_checkpoint_creation_interval() {
        let mut monitor = SafetyMonitor::new();
        let params = [100u64; 5];

        // First 9 updates shouldn't create visible checkpoint in stats
        // (checkpoint created on 10th update)
        for _ in 0..9 {
            monitor.check_and_protect(0, 0, 50.0, &params);
        }

        // 10th update creates checkpoint
        monitor.check_and_protect(0, 0, 50.0, &params);
        let stats = monitor.get_stats();
        assert_eq!(stats.checkpoints_stored, 1);

        // 20th update creates another checkpoint (replaces)
        for _ in 0..10 {
            monitor.check_and_protect(0, 0, 50.0, &params);
        }
        let stats = monitor.get_stats();
        assert_eq!(stats.checkpoints_stored, 1); // Still 1, just updated
    }

    #[test]
    fn test_multiple_cpu_class_combinations() {
        let mut monitor = SafetyMonitor::new();

        // Create checkpoints for different (cpu, class) combinations
        for cpu in 0..2 {
            for class in 0..4 {
                let params = [cpu as u64 * 10 + class as u64; 5];
                for _ in 0..10 {
                    monitor.check_and_protect(cpu, class, 50.0, &params);
                }
            }
        }

        let stats = monitor.get_stats();
        assert_eq!(stats.checkpoints_stored, 8); // 2 CPUs * 4 classes
    }

    #[test]
    fn test_loss_history_ewma() {
        let mut monitor = SafetyMonitor::new();
        let params = [100u64; 5];

        // Add several loss values
        monitor.check_and_protect(0, 0, 100.0, &params);
        monitor.check_and_protect(0, 0, 200.0, &params);
        monitor.check_and_protect(0, 0, 100.0, &params);

        let stats = monitor.get_stats();
        assert_eq!(stats.loss_history_size, 3);

        // History should use EWMA smoothing
        // First: 100.0
        // Second: 0.3*200 + 0.7*100 = 130
        // Third: 0.3*100 + 0.7*130 = 121
    }

    #[test]
    fn test_spike_on_different_cpu_class() {
        let mut monitor = SafetyMonitor::new_with_threshold(100.0);
        let params1 = [100u64; 5];

        // Create checkpoint on CPU 0, class 0
        for _ in 0..MIN_OBSERVATIONS_FOR_SAFETY + 10 {
            monitor.check_and_protect(0, 0, 50.0, &params1);
        }

        // Spike on CPU 0, class 0
        let result = monitor.check_and_protect(0, 0, 200.0, &params1);
        assert!(result.is_some());

        // Reset and check different class - should have no checkpoint
        let params2 = [200u64; 5];
        let result = monitor.check_and_protect(0, 1, 200.0, &params2);
        // No checkpoint for class 1 yet, so returns None
        assert!(result.is_none());
    }
}
