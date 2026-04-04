//! Parameter Bounds Debugging for scx_descent
//!
//! Tracks when PIE controller parameters hit their bounds (min or max) during
//! clamping. This helps identify which bounds are constraining the scheduler
//! and may need widening for better performance under load.
//!
//! Key metrics tracked:
//! - Bound hit count per (CPU, class, parameter, bound_type)
//! - Percentage of time spent at each bound
//! - Correlation with latency error (to identify problematic bounds)

use std::collections::HashMap;

use crate::profiles::{ParamBounds, DESCENT_CLASS_MAX, PARAM_COUNT};

/// Types of bound hits
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundType {
    Min,
    Max,
}

/// Per-parameter bound tracking statistics
#[derive(Debug, Clone, Copy, Default)]
pub struct BoundStats {
    /// Number of times this parameter hit the minimum bound
    pub min_hits: u64,
    /// Number of times this parameter hit the maximum bound
    pub max_hits: u64,
    /// Number of updates where parameter was within bounds (not clamped)
    pub within_bounds: u64,
    /// Total number of updates for this parameter
    pub total_updates: u64,
    /// Accumulated latency error when at min bound (helps identify if min is too high)
    pub latency_error_at_min: i64,
    /// Accumulated latency error when at max bound (helps identify if max is too low)
    pub latency_error_at_max: i64,
    /// Last value when clamped (for debugging)
    pub last_clamped_value: u64,
    /// Last target bound when clamped
    pub last_bound_value: u64,
}

impl BoundStats {
    /// Calculate percentage of time spent at minimum bound (0-100)
    pub fn min_hit_percent(&self) -> f64 {
        if self.total_updates == 0 {
            return 0.0;
        }
        (self.min_hits as f64 / self.total_updates as f64) * 100.0
    }

    /// Calculate percentage of time spent at maximum bound (0-100)
    pub fn max_hit_percent(&self) -> f64 {
        if self.total_updates == 0 {
            return 0.0;
        }
        (self.max_hits as f64 / self.total_updates as f64) * 100.0
    }

    /// Calculate percentage of time spent at any bound (0-100)
    pub fn total_bound_hit_percent(&self) -> f64 {
        if self.total_updates == 0 {
            return 0.0;
        }
        ((self.min_hits + self.max_hits) as f64 / self.total_updates as f64) * 100.0
    }

    /// Check if this bound is being hit frequently (>80% of updates)
    pub fn is_constrained(&self) -> bool {
        self.total_bound_hit_percent() > 80.0
    }

    /// Get the average latency error when at bounds
    pub fn avg_latency_error_at_bounds(&self) -> i64 {
        let total_bound_hits = self.min_hits + self.max_hits;
        if total_bound_hits == 0 {
            return 0;
        }
        (self.latency_error_at_min + self.latency_error_at_max) / total_bound_hits as i64
    }
}

/// Parameter names for human-readable output
pub const PARAM_NAMES: [&str; PARAM_COUNT] = [
    "latency_weight",
    "base_slice_ns",
    "vruntime_scale",
    "preemption_priority",
    "migration_cost",
];

/// Class names for human-readable output
pub const CLASS_NAMES: [&str; DESCENT_CLASS_MAX] =
    ["LATENCY_CRITICAL", "NORMAL", "HOG", "BACKGROUND"];

/// Key for identifying a specific (cpu, class, param) combination
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BoundKey {
    pub cpu: u32,
    pub class: u32,
    pub param_idx: usize,
}

/// Bound debugger - tracks parameter clamping events
pub struct BoundDebugger {
    /// Statistics per (cpu, class, param)
    stats: HashMap<BoundKey, BoundStats>,
    /// Reference to bounds configuration
    bounds: ParamBounds,
    /// Whether debugging is enabled
    enabled: bool,
    /// Threshold for considering a bound "constrained" (%)
    constraint_threshold: f64,
}

impl BoundDebugger {
    /// Create a new bound debugger
    pub fn new(bounds: ParamBounds, enabled: bool) -> Self {
        Self {
            stats: HashMap::new(),
            bounds,
            enabled,
            constraint_threshold: 80.0,
        }
    }

    /// Enable or disable debugging at runtime
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Set the constraint threshold percentage
    pub fn set_constraint_threshold(&mut self, threshold: f64) {
        self.constraint_threshold = threshold;
    }

    /// Record a parameter update event (even if not clamped)
    ///
    /// Call this for every parameter update to track total update counts
    pub fn record_update(&mut self, cpu: u32, class: u32, param_idx: usize) {
        if !self.enabled || param_idx >= PARAM_COUNT || class as usize >= DESCENT_CLASS_MAX {
            return;
        }

        let key = BoundKey {
            cpu,
            class,
            param_idx,
        };

        let stats = self.stats.entry(key).or_insert_with(BoundStats::default);
        stats.total_updates += 1;
    }

    /// Record a parameter clamping event
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class
    /// * `param_idx` - Parameter index (0-4)
    /// * `original_value` - The value before clamping
    /// * `clamped_value` - The value after clamping
    /// * `latency_error_ns` - Current latency error (target - current, positive = need more resources)
    pub fn record_clamp(
        &mut self,
        cpu: u32,
        class: u32,
        param_idx: usize,
        original_value: u64,
        clamped_value: u64,
        latency_error_ns: i64,
    ) {
        if !self.enabled || param_idx >= PARAM_COUNT || class as usize >= DESCENT_CLASS_MAX {
            return;
        }

        let key = BoundKey {
            cpu,
            class,
            param_idx,
        };

        let (min_bound, max_bound) = self.bounds[class as usize][param_idx];

        let stats = self.stats.entry(key).or_insert_with(BoundStats::default);
        // Note: total_updates is already incremented in record_update() before this call

        if clamped_value == min_bound && original_value < min_bound {
            // Hit minimum bound
            stats.min_hits += 1;
            stats.latency_error_at_min += latency_error_ns;
            stats.last_clamped_value = original_value;
            stats.last_bound_value = min_bound;
        } else if clamped_value == max_bound && original_value > max_bound {
            // Hit maximum bound
            stats.max_hits += 1;
            stats.latency_error_at_max += latency_error_ns;
            stats.last_clamped_value = original_value;
            stats.last_bound_value = max_bound;
        } else {
            // Within bounds
            stats.within_bounds += 1;
        }
    }

    /// Get statistics for a specific (cpu, class, param)
    pub fn get_stats(&self, cpu: u32, class: u32, param_idx: usize) -> Option<&BoundStats> {
        let key = BoundKey {
            cpu,
            class,
            param_idx,
        };
        self.stats.get(&key)
    }

    /// Get mutable statistics for a specific (cpu, class, param)
    fn get_stats_mut(&mut self, cpu: u32, class: u32, param_idx: usize) -> &mut BoundStats {
        let key = BoundKey {
            cpu,
            class,
            param_idx,
        };
        self.stats.entry(key).or_insert_with(BoundStats::default)
    }

    /// Get all stats for a specific class across all CPUs and params
    pub fn get_class_stats(&self, class: u32) -> Vec<(BoundKey, &BoundStats)> {
        self.stats
            .iter()
            .filter(|(key, _)| key.class == class)
            .map(|(key, stats)| (*key, stats))
            .collect()
    }

    /// Get all stats for a specific parameter across all CPUs and classes
    pub fn get_param_stats(&self, param_idx: usize) -> Vec<(BoundKey, &BoundStats)> {
        self.stats
            .iter()
            .filter(|(key, _)| key.param_idx == param_idx)
            .map(|(key, stats)| (*key, stats))
            .collect()
    }

    /// Get all constrained parameters (hitting bounds > threshold %)
    pub fn get_constrained_params(&self) -> Vec<(BoundKey, &BoundStats, BoundType)> {
        let mut constrained = Vec::new();

        for (key, stats) in &self.stats {
            if stats.total_updates < 10 {
                // Need minimum samples
                continue;
            }

            if stats.min_hit_percent() > self.constraint_threshold {
                constrained.push((*key, stats, BoundType::Min));
            }

            if stats.max_hit_percent() > self.constraint_threshold {
                constrained.push((*key, stats, BoundType::Max));
            }
        }

        // Sort by hit percentage (highest first)
        constrained.sort_by(|a, b| {
            let a_pct = match a.2 {
                BoundType::Min => a.1.min_hit_percent(),
                BoundType::Max => a.1.max_hit_percent(),
            };
            let b_pct = match b.2 {
                BoundType::Min => b.1.min_hit_percent(),
                BoundType::Max => b.1.max_hit_percent(),
            };
            b_pct.partial_cmp(&a_pct).unwrap()
        });

        constrained
    }

    /// Get global summary statistics
    pub fn get_summary(&self) -> BoundSummary {
        let mut total_updates = 0u64;
        let mut total_min_hits = 0u64;
        let mut total_max_hits = 0u64;
        let mut constrained_count = 0usize;

        for (_, stats) in &self.stats {
            total_updates += stats.total_updates;
            total_min_hits += stats.min_hits;
            total_max_hits += stats.max_hits;

            if stats.is_constrained() {
                constrained_count += 1;
            }
        }

        let total_bound_hits = total_min_hits + total_max_hits;
        let overall_constraint_pct = if total_updates > 0 {
            (total_bound_hits as f64 / total_updates as f64) * 100.0
        } else {
            0.0
        };

        BoundSummary {
            total_tracked_params: self.stats.len(),
            constrained_params: constrained_count,
            total_updates,
            total_min_hits,
            total_max_hits,
            overall_constraint_percentage: overall_constraint_pct,
        }
    }

    /// Format a debug report for output
    pub fn format_report(&self, nr_cpus: usize) -> String {
        let mut report = String::new();
        let summary = self.get_summary();

        // Count active vs constrained parameters
        let mut active_params = 0usize;
        let mut constrained_params = 0usize;
        let mut freely_operating = 0usize;

        for (_, stats) in &self.stats {
            if stats.total_updates > 0 {
                active_params += 1;
                if stats.is_constrained() {
                    constrained_params += 1;
                } else {
                    freely_operating += 1;
                }
            }
        }

        report.push_str("\n=== Parameter Bounds Debug Report ===\n");
        report.push_str(&format!(
            "Summary: {} parameters active, {} constrained, {} freely operating\n",
            active_params, constrained_params, freely_operating
        ));
        report.push_str(&format!(
            "Overall constraint rate: {:.1}% (min: {} hits, max: {} hits, within: {})\n",
            summary.overall_constraint_percentage,
            summary.total_min_hits,
            summary.total_max_hits,
            summary
                .total_updates
                .saturating_sub(summary.total_min_hits + summary.total_max_hits)
        ));

        // Only show constrained section if there are constrained parameters
        let constrained = self.get_constrained_params();
        if !constrained.is_empty() {
            report.push_str("\nConstrained Parameters (>80% at bounds):\n");
            report.push_str(
                "  CPU  Class                Param              Bound    Hits    %   LatencyErr\n",
            );

            for (key, stats, bound_type) in constrained.iter().take(20) {
                // Limit output
                let class_name = CLASS_NAMES[key.class as usize];
                let param_name = PARAM_NAMES[key.param_idx];
                let (bound_name, hit_pct, error) = match bound_type {
                    BoundType::Min => ("MIN", stats.min_hit_percent(), stats.latency_error_at_min),
                    BoundType::Max => ("MAX", stats.max_hit_percent(), stats.latency_error_at_max),
                };
                let error_avg = if error != 0 {
                    error
                        / match bound_type {
                            BoundType::Min => stats.min_hits as i64,
                            BoundType::Max => stats.max_hits as i64,
                        }
                } else {
                    0
                };

                report.push_str(&format!(
                    "  {:3}  {:20} {:20} {:5}  {:5}  {:5.1}%  {:+7}µs\n",
                    key.cpu,
                    class_name,
                    param_name,
                    bound_name,
                    match bound_type {
                        BoundType::Min => stats.min_hits,
                        BoundType::Max => stats.max_hits,
                    },
                    hit_pct,
                    error_avg / 1000 // Convert to µs
                ));
            }
        } else if active_params > 0 {
            report.push_str(
                "\n✓ No constrained parameters - all parameters operating within bounds!\n",
            );
        } else {
            report.push_str("\n⚠ No parameter updates recorded - scheduler may not be receiving latency metrics\n");
        }

        // Show per-class summary for active parameters
        report.push_str("\nPer-Class Parameter Activity:\n");
        report.push_str(
            "  Class                Param              Updates  Min%    Max%   Within%  Status\n",
        );

        for class in 0..DESCENT_CLASS_MAX as u32 {
            for param_idx in 0..PARAM_COUNT {
                let mut class_updates = 0u64;
                let mut class_min_hits = 0u64;
                let mut class_max_hits = 0u64;

                for cpu in 0..nr_cpus as u32 {
                    if let Some(stats) = self.get_stats(cpu, class, param_idx) {
                        class_updates += stats.total_updates;
                        class_min_hits += stats.min_hits;
                        class_max_hits += stats.max_hits;
                    }
                }

                if class_updates > 0 {
                    let min_pct = (class_min_hits as f64 / class_updates as f64) * 100.0;
                    let max_pct = (class_max_hits as f64 / class_updates as f64) * 100.0;
                    let within_pct = 100.0 - min_pct - max_pct;

                    let status = if (min_pct + max_pct) > 80.0 {
                        "CONSTRAINED"
                    } else if (min_pct + max_pct) > 50.0 {
                        "TIGHT"
                    } else {
                        "OK"
                    };

                    report.push_str(&format!(
                        "  {:20} {:20} {:7}  {:5.1}%  {:5.1}%  {:5.1}%  {}\n",
                        CLASS_NAMES[class as usize],
                        PARAM_NAMES[param_idx],
                        class_updates,
                        min_pct,
                        max_pct,
                        within_pct,
                        status
                    ));
                }
            }
        }

        report
    }

    /// Reset all statistics
    pub fn reset(&mut self) {
        self.stats.clear();
    }
}

/// Global summary of bound statistics
#[derive(Debug, Clone, Copy)]
pub struct BoundSummary {
    /// Total number of tracked parameters
    pub total_tracked_params: usize,
    /// Number of parameters considered constrained
    pub constrained_params: usize,
    /// Total number of updates across all parameters
    pub total_updates: u64,
    /// Total hits at minimum bound
    pub total_min_hits: u64,
    /// Total hits at maximum bound
    pub total_max_hits: u64,
    /// Overall percentage of updates that hit any bound
    pub overall_constraint_percentage: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_bounds() -> ParamBounds {
        [
            // LATENCY_CRITICAL
            [
                (500_000, 2_000_000),    // latency_weight
                (1_000_000, 10_000_000), // base_slice_ns
                (1024, 2048),            // vruntime_scale
                (100, 100),              // preemption_priority
                (50_000, 500_000),       // migration_cost
            ],
            // NORMAL
            [
                (500_000, 2_000_000),
                (1_000_000, 10_000_000),
                (1024, 2048),
                (100, 100),
                (50_000, 500_000),
            ],
            // HOG
            [
                (500_000, 2_000_000),
                (1_000_000, 10_000_000),
                (1024, 2048),
                (100, 100),
                (50_000, 500_000),
            ],
            // BACKGROUND
            [
                (500_000, 2_000_000),
                (1_000_000, 10_000_000),
                (1024, 2048),
                (100, 100),
                (50_000, 500_000),
            ],
        ]
    }

    #[test]
    fn test_record_min_hit() {
        let bounds = test_bounds();
        let mut debugger = BoundDebugger::new(bounds, true);

        // Record a value below minimum
        debugger.record_clamp(0, 1, 1, 500_000, 1_000_000, -100_000);

        let stats = debugger.get_stats(0, 1, 1).unwrap();
        assert_eq!(stats.min_hits, 1);
        assert_eq!(stats.max_hits, 0);
        assert_eq!(stats.within_bounds, 0);
        assert_eq!(stats.total_updates, 1);
        assert_eq!(stats.last_clamped_value, 500_000);
        assert_eq!(stats.last_bound_value, 1_000_000);
    }

    #[test]
    fn test_record_max_hit() {
        let bounds = test_bounds();
        let mut debugger = BoundDebugger::new(bounds, true);

        // Record a value above maximum
        debugger.record_clamp(0, 1, 1, 15_000_000, 10_000_000, 500_000);

        let stats = debugger.get_stats(0, 1, 1).unwrap();
        assert_eq!(stats.min_hits, 0);
        assert_eq!(stats.max_hits, 1);
        assert_eq!(stats.within_bounds, 0);
        assert_eq!(stats.total_updates, 1);
        assert_eq!(stats.last_clamped_value, 15_000_000);
        assert_eq!(stats.last_bound_value, 10_000_000);
    }

    #[test]
    fn test_record_within_bounds() {
        let bounds = test_bounds();
        let mut debugger = BoundDebugger::new(bounds, true);

        // Record a value within bounds (no clamping needed)
        debugger.record_clamp(0, 1, 1, 5_000_000, 5_000_000, 0);

        let stats = debugger.get_stats(0, 1, 1).unwrap();
        assert_eq!(stats.min_hits, 0);
        assert_eq!(stats.max_hits, 0);
        assert_eq!(stats.within_bounds, 1);
        assert_eq!(stats.total_updates, 1);
    }

    #[test]
    fn test_disabled_debugger() {
        let bounds = test_bounds();
        let mut debugger = BoundDebugger::new(bounds, false);

        // Should not record when disabled
        debugger.record_clamp(0, 1, 1, 500_000, 1_000_000, -100_000);

        assert!(debugger.get_stats(0, 1, 1).is_none());
    }

    #[test]
    fn test_constrained_detection() {
        let bounds = test_bounds();
        let mut debugger = BoundDebugger::new(bounds, true);

        // Record 90% min hits out of 10 updates
        for _ in 0..9 {
            debugger.record_clamp(0, 1, 1, 500_000, 1_000_000, -100_000);
        }
        // 1 within bounds
        debugger.record_clamp(0, 1, 1, 5_000_000, 5_000_000, 0);

        let constrained = debugger.get_constrained_params();
        assert!(!constrained.is_empty());

        // Check the first constrained param is our min-bound slice
        let (key, _, bound_type) = constrained[0];
        assert_eq!(key.cpu, 0);
        assert_eq!(key.class, 1);
        assert_eq!(key.param_idx, 1);
        assert_eq!(bound_type, BoundType::Min);
    }

    #[test]
    fn test_summary() {
        let bounds = test_bounds();
        let mut debugger = BoundDebugger::new(bounds, true);

        // Record some hits
        for _ in 0..5 {
            debugger.record_clamp(0, 0, 1, 500_000, 1_000_000, -100_000); // min hit
            debugger.record_clamp(0, 1, 1, 15_000_000, 10_000_000, 500_000); // max hit
        }

        let summary = debugger.get_summary();
        assert_eq!(summary.total_tracked_params, 2);
        assert_eq!(summary.total_min_hits, 5);
        assert_eq!(summary.total_max_hits, 5);
        assert!(summary.overall_constraint_percentage > 0.0);
    }

    #[test]
    fn test_param_names_length() {
        assert_eq!(PARAM_NAMES.len(), PARAM_COUNT);
    }

    #[test]
    fn test_class_names_length() {
        assert_eq!(CLASS_NAMES.len(), DESCENT_CLASS_MAX);
    }
}
