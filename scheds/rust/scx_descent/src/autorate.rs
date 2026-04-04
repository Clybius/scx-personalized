//! CAKE Autorate Controller for scx_descent
//!
//! Implements coarse parameter adaptation based on system-wide load across
//! four task classes (LATENCY_CRITICAL, NORMAL, HOG, BACKGROUND).
//!
//! Key concepts:
//! - Per-class operation: Tracks load system-wide, NOT per-CPU
//! - Linear interpolation: Simple and efficient between min/baseline/max params
//! - Four states: STEADY, LOAD_HIGH, LOAD_LOW, BUFFERBLOAT
//! - Refractory periods: Prevent oscillation after adjustments
//!
//! The Autorate controller complements the PIE controller by providing
//! coarse-grained adaptation based on load conditions, while PIE handles
//! fine-grained latency-based optimization.

// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_descent authors
//
// CAKE Autorate controller for coarse parameter adaptation
// based on system-wide load per class.

use log::debug;
use std::collections::HashMap;
use std::time::Instant;

use crate::profiles::{DESCENT_CLASS_MAX, PARAM_COUNT};

/// CAKE Autorate state machine states
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AutorateState {
    Steady,
    LoadHigh,
    LoadLow,
    Bufferbloat,
}

impl AutorateState {
    /// Get human-readable name for the state
    pub fn name(&self) -> &'static str {
        match self {
            AutorateState::Steady => "STEADY",
            AutorateState::LoadHigh => "LOAD_HIGH",
            AutorateState::LoadLow => "LOAD_LOW",
            AutorateState::Bufferbloat => "BUFFERBLOAT",
        }
    }
}

/// Direction of last adjustment (for refractory period)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdjustmentDirection {
    Up,   // Toward max
    Down, // Toward min
    None,
}

/// Per-class autorate state (NOT per-CPU)
#[derive(Debug, Clone, Copy)]
pub struct AutorateClassState {
    /// Current rate (0.0=min, 0.5=baseline, 1.0=max)
    pub current_rate: f64,
    /// Current load percentage (0.0-1.0)
    pub load_percent: f64,
    /// Current autorate state
    pub state: AutorateState,
    /// Time of last adjustment
    pub last_adjustment_time: Instant,
    /// Direction of last adjustment
    pub last_adjustment_dir: AdjustmentDirection,
    /// Smoothed latency EWMA in nanoseconds
    pub latency_ewma_ns: u64,
    /// Previous latency measurement
    pub prev_latency_ns: u64,
}

impl AutorateClassState {
    /// Create new autorate state for a class
    fn new() -> Self {
        Self {
            current_rate: 0.5, // Start at baseline
            load_percent: 0.0, // 0.0 until real data arrives (was 0.5)
            state: AutorateState::Steady,
            last_adjustment_time: Instant::now(),
            last_adjustment_dir: AdjustmentDirection::None,
            latency_ewma_ns: 0,
            prev_latency_ns: 0,
        }
    }
}

/// Autorate configuration (from Profile)
#[derive(Debug, Clone)]
pub struct AutorateConfig {
    /// Enable autorate controller
    pub enabled: bool,
    /// Minimum parameters per class: [class][param]
    pub min_params: [[u64; PARAM_COUNT]; DESCENT_CLASS_MAX],
    /// Baseline parameters per class: [class][param]
    pub baseline_params: [[u64; PARAM_COUNT]; DESCENT_CLASS_MAX],
    /// Maximum parameters per class: [class][param]
    pub max_params: [[u64; PARAM_COUNT]; DESCENT_CLASS_MAX],
    /// High load threshold (e.g., 0.75)
    pub high_load_threshold: f64,
    /// Low load threshold (e.g., 0.25)
    pub low_load_threshold: f64,
    /// Ramp up rate multiplier (e.g., 1.08 for gaming, 8%)
    pub ramp_up_rate: f64,
    /// Ramp down rate multiplier (e.g., 0.75 for 25% down)
    pub ramp_down_rate: f64,
    /// Decay rate toward baseline (e.g., 0.99 = 1% toward baseline)
    pub decay_rate: f64,
    /// Refractory period after upward adjustment in ms
    pub adjust_up_refractory_ms: u64,
    /// Refractory period after downward adjustment in ms
    pub adjust_down_refractory_ms: u64,
    /// Bufferbloat threshold (e.g., 1.5 = 1.5x target latency)
    pub bufferbloat_threshold: f64,
}

/// Statistics for autorate controller monitoring
#[derive(Debug, Clone, Copy)]
pub struct AutorateStats {
    /// Total number of class states
    pub total_classes: usize,
    /// Number of upward adjustments
    pub adjustments_up: u64,
    /// Number of downward adjustments
    pub adjustments_down: u64,
    /// Number of blocked adjustments (refractory)
    pub adjustments_blocked: u64,
    /// Current average rate across all classes
    pub avg_rate: f64,
}

/// Main autorate controller
pub struct AutorateController {
    /// Per-class states (NOT per-CPU)
    states: HashMap<u32, AutorateClassState>,
    /// Controller configuration
    config: AutorateConfig,
    /// Statistics
    stats: AutorateStats,
}

impl AutorateController {
    /// Create new controller
    ///
    /// * `_nr_cpus` - Number of CPUs (used for initial load estimate, but states are per-class)
    /// * `config` - Autorate configuration
    pub fn new(_nr_cpus: usize, config: &AutorateConfig) -> Self {
        let mut states = HashMap::new();

        // Initialize states for all classes (per-class, NOT per-CPU)
        for class in 0..DESCENT_CLASS_MAX as u32 {
            states.insert(class, AutorateClassState::new());
        }

        debug!(
            "[AUTORATE-INIT] Created controller with {} classes",
            states.len()
        );

        Self {
            states,
            config: config.clone(),
            stats: AutorateStats {
                total_classes: DESCENT_CLASS_MAX,
                adjustments_up: 0,
                adjustments_down: 0,
                adjustments_blocked: 0,
                avg_rate: 0.5,
            },
        }
    }

    /// Main update - called once per class per interval
    ///
    /// # Arguments
    /// * `class` - Class ID (0-3)
    /// * `latency_ns` - Average latency across all CPUs for this class
    /// * `target_latency_ns` - Target for this class
    /// * `load_percent` - Aggregate load % across all CPUs (0.0-1.0)
    ///
    /// # Returns
    /// Tuple of (interpolated_params, current_state)
    pub fn update(
        &mut self,
        class: u32,
        latency_ns: u64,
        target_latency_ns: u64,
        load_percent: f64,
    ) -> ([u64; PARAM_COUNT], AutorateState) {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            // Invalid class - return baseline
            return (
                self.config.baseline_params[class.min(3) as usize],
                AutorateState::Steady,
            );
        }

        // First, compute the new state (only needs immutable borrow of self)
        let new_state = self.determine_state(latency_ns, target_latency_ns, load_percent);

        // CRITICAL FIX: Check cross-class protection BEFORE getting mutable state
        // If LATENCY_CRITICAL is in BUFFERBLOAT, cap HOG (Class 2) rate at baseline
        let effective_max_rate = if class == 2 {
            // Check if LATENCY_CRITICAL (Class 0) is under pressure
            if let Some(lc_state) = self.states.get(&0) {
                if lc_state.state == AutorateState::Bufferbloat {
                    0.5 // Cap at baseline to protect audio/latency tasks
                } else {
                    1.0 // Normal max rate
                }
            } else {
                1.0
            }
        } else {
            1.0
        };

        // ALWAYS update the state with current metrics (for monitoring/display)
        // even if autorate interpolation is disabled
        let state = self
            .states
            .entry(class)
            .or_insert_with(AutorateClassState::new);

        // CRITICAL FIX: Capture prev_state BEFORE updating state.state
        let prev_state = state.state;

        // Update metrics
        state.prev_latency_ns = state.latency_ewma_ns;
        state.latency_ewma_ns = latency_ns;
        state.load_percent = load_percent;
        state.state = new_state;

        // If autorate is disabled, return baseline params but still track state
        if !self.config.enabled {
            return (
                self.config.baseline_params[class_idx],
                new_state, // Return actual state, not Steady
            );
        }

        // Extract config values for rate adjustment
        let ramp_up_rate = self.config.ramp_up_rate;
        let ramp_down_rate = self.config.ramp_down_rate;
        let decay_rate = self.config.decay_rate;
        let adjust_up_refractory_ms = self.config.adjust_up_refractory_ms;
        let adjust_down_refractory_ms = self.config.adjust_down_refractory_ms;

        // Check refractory period if state change requires adjustment
        let requires_adjustment = new_state != AutorateState::Steady || prev_state != new_state;
        let adjustment_allowed = if requires_adjustment {
            let elapsed_ms = state.last_adjustment_time.elapsed().as_millis() as u64;
            let would_adjust_up = new_state == AutorateState::LoadHigh;
            let would_adjust_down = new_state == AutorateState::Bufferbloat
                || (new_state == AutorateState::LoadLow && state.current_rate > 0.5);

            match state.last_adjustment_dir {
                AdjustmentDirection::Up => {
                    if would_adjust_up {
                        elapsed_ms >= adjust_up_refractory_ms
                    } else {
                        elapsed_ms >= adjust_down_refractory_ms
                    }
                }
                AdjustmentDirection::Down => {
                    if would_adjust_down {
                        elapsed_ms >= adjust_down_refractory_ms
                    } else {
                        elapsed_ms >= adjust_up_refractory_ms
                    }
                }
                AdjustmentDirection::None => {
                    // No previous adjustment - this is the first one, allow it
                    true
                }
            }
        } else {
            true
        };

        let current_rate = state.current_rate;

        let adjustment_dir = if new_state == AutorateState::LoadHigh {
            AdjustmentDirection::Up
        } else if new_state == AutorateState::Bufferbloat {
            AdjustmentDirection::Down
        } else if new_state == AutorateState::LoadLow && current_rate > 0.5 {
            AdjustmentDirection::Down
        } else if new_state == AutorateState::LoadLow && current_rate < 0.5 {
            AdjustmentDirection::Up
        } else {
            AdjustmentDirection::None
        };

        // Update rate if adjustment is allowed
        if adjustment_allowed {
            // Calculate new rate inline to avoid borrow issues
            let new_rate = match new_state {
                AutorateState::Bufferbloat => current_rate * ramp_down_rate,
                AutorateState::LoadHigh => {
                    // Apply effective max rate (may be capped for HOG when LATENCY_CRITICAL under pressure)
                    (current_rate * ramp_up_rate).min(effective_max_rate)
                }
                AutorateState::LoadLow => {
                    if current_rate > 0.5 {
                        current_rate * decay_rate
                    } else {
                        let diff = 0.5 - current_rate;
                        current_rate + diff * (1.0 - decay_rate)
                    }
                }
                AutorateState::Steady => current_rate,
            };

            // Update stats if there was an actual adjustment
            if (new_rate - current_rate).abs() > f64::EPSILON {
                match adjustment_dir {
                    AdjustmentDirection::Up => self.stats.adjustments_up += 1,
                    AdjustmentDirection::Down => self.stats.adjustments_down += 1,
                    AdjustmentDirection::None => {}
                }
                state.last_adjustment_time = Instant::now();
                state.last_adjustment_dir = adjustment_dir;
            }

            state.current_rate = new_rate.clamp(0.0, 1.0);
        } else {
            self.stats.adjustments_blocked += 1;
            debug!(
                "[AUTORATE] Class {}: adjustment blocked by refractory period (dir={:?})",
                class, adjustment_dir
            );
        }

        let current_rate = state.current_rate;

        // Interpolate parameters based on current rate
        let params = self.interpolate_params(current_rate, class);

        debug!(
            "[AUTORATE] Class {}: state={} rate={:.3} load={:.2} lat={}µs target={}µs params=[{} {} {} {} {}]",
            class,
            new_state.name(),
            current_rate,
            load_percent,
            latency_ns / 1000,
            target_latency_ns / 1000,
            params[0], params[1], params[2], params[3], params[4]
        );

        (params, new_state)
    }

    /// Determine state based on latency and load
    fn determine_state(
        &self,
        latency_ns: u64,
        target_latency_ns: u64,
        load_percent: f64,
    ) -> AutorateState {
        // Avoid division by zero
        if target_latency_ns == 0 {
            // Fall back to load-based detection
            if load_percent > self.config.high_load_threshold {
                return AutorateState::LoadHigh;
            }
            if load_percent < self.config.low_load_threshold {
                return AutorateState::LoadLow;
            }
            return AutorateState::Steady;
        }

        let latency_ratio = latency_ns as f64 / target_latency_ns as f64;

        // Priority 1: Bufferbloat (emergency)
        if latency_ratio > self.config.bufferbloat_threshold {
            return AutorateState::Bufferbloat;
        }

        // Priority 2: High load (opportunity)
        // Only ramp up if latency is not exceeding target
        if load_percent >= self.config.high_load_threshold && latency_ratio <= 1.0 {
            return AutorateState::LoadHigh;
        }

        // Priority 3: Low load (conserve)
        if load_percent <= self.config.low_load_threshold {
            return AutorateState::LoadLow;
        }

        AutorateState::Steady
    }

    /// Linear interpolation between min/baseline/max
    ///
    /// rate 0.0 -> min_params
    /// rate 0.5 -> baseline_params
    /// rate 1.0 -> max_params
    fn interpolate_params(&self, rate: f64, class: u32) -> [u64; PARAM_COUNT] {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return [0; PARAM_COUNT];
        }

        let min = &self.config.min_params[class_idx];
        let baseline = &self.config.baseline_params[class_idx];
        let max = &self.config.max_params[class_idx];

        let mut result = [0u64; PARAM_COUNT];
        for i in 0..PARAM_COUNT {
            result[i] = if rate <= 0.5 {
                // Interpolate min -> baseline
                let t = rate * 2.0; // 0.0-0.5 -> 0.0-1.0
                min[i] + ((baseline[i] - min[i]) as f64 * t).round() as u64
            } else {
                // Interpolate baseline -> max
                let t = (rate - 0.5) * 2.0; // 0.5-1.0 -> 0.0-1.0
                baseline[i] + ((max[i] - baseline[i]) as f64 * t).round() as u64
            };
        }
        result
    }

    /// Get current state for debugging
    pub fn get_class_state(&self, class: u32) -> Option<&AutorateClassState> {
        self.states.get(&class)
    }

    /// Get controller statistics
    pub fn get_stats(&self) -> AutorateStats {
        let mut total_rate = 0.0;
        let mut count = 0;
        for state in self.states.values() {
            total_rate += state.current_rate;
            count += 1;
        }

        let mut stats = self.stats;
        stats.avg_rate = if count > 0 {
            total_rate / count as f64
        } else {
            0.5
        };
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::Profile;

    // Test configuration with known values for predictable tests
    fn test_config() -> AutorateConfig {
        AutorateConfig {
            enabled: true,
            min_params: [
                [100, 200, 300, 400, 500],                // Class 0
                [1000, 2000, 3000, 4000, 5000],           // Class 1
                [10000, 20000, 30000, 40000, 50000],      // Class 2
                [100000, 200000, 300000, 400000, 500000], // Class 3
            ],
            baseline_params: [
                [200, 400, 600, 800, 1000],                // Class 0 (2x min)
                [2000, 4000, 6000, 8000, 10000],           // Class 1
                [20000, 40000, 60000, 80000, 100000],      // Class 2
                [200000, 400000, 600000, 800000, 1000000], // Class 3
            ],
            max_params: [
                [300, 600, 900, 1200, 1500],                // Class 0 (3x min)
                [3000, 6000, 9000, 12000, 15000],           // Class 1
                [30000, 60000, 90000, 120000, 150000],      // Class 2
                [300000, 600000, 900000, 1200000, 1500000], // Class 3
            ],
            high_load_threshold: 0.75,
            low_load_threshold: 0.25,
            ramp_up_rate: 1.1,    // 10% up
            ramp_down_rate: 0.75, // 25% down
            decay_rate: 0.99,     // 1% decay
            adjust_up_refractory_ms: 50,
            adjust_down_refractory_ms: 20,
            bufferbloat_threshold: 1.5,
        }
    }

    #[test]
    fn test_state_detection_bufferbloat() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // Test bufferbloat detection when latency > 1.5x target
        let state = controller.determine_state(2000, 1000, 0.5);
        assert_eq!(
            state,
            AutorateState::Bufferbloat,
            "Should detect bufferbloat when 2x target"
        );

        // Just at threshold
        let state2 = controller.determine_state(1501, 1000, 0.5);
        assert_eq!(
            state2,
            AutorateState::Bufferbloat,
            "Should detect bufferbloat just above 1.5x"
        );

        // Below threshold
        let state3 = controller.determine_state(1400, 1000, 0.5);
        assert_ne!(
            state3,
            AutorateState::Bufferbloat,
            "Should not detect bufferbloat below threshold"
        );
    }

    #[test]
    fn test_state_detection_load_high() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // High load (80%) with normal latency
        let state = controller.determine_state(900, 1000, 0.80);
        assert_eq!(state, AutorateState::LoadHigh, "Should detect high load");

        // At threshold
        let state2 = controller.determine_state(900, 1000, 0.75);
        assert_eq!(
            state2,
            AutorateState::LoadHigh,
            "Should detect high load at threshold"
        );

        // High load but with high latency (should be bufferbloat)
        let state3 = controller.determine_state(2000, 1000, 0.80);
        assert_eq!(
            state3,
            AutorateState::Bufferbloat,
            "Bufferbloat takes priority over high load"
        );
    }

    #[test]
    fn test_state_detection_load_low() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // Low load (20%) with normal latency
        let state = controller.determine_state(900, 1000, 0.20);
        assert_eq!(state, AutorateState::LoadLow, "Should detect low load");

        // At threshold
        let state2 = controller.determine_state(900, 1000, 0.25);
        assert_eq!(
            state2,
            AutorateState::LoadLow,
            "Should detect low load at threshold"
        );

        // Below threshold
        let state3 = controller.determine_state(900, 1000, 0.30);
        assert_eq!(
            state3,
            AutorateState::Steady,
            "Should be steady between thresholds"
        );
    }

    #[test]
    fn test_state_detection_steady() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // Normal load and latency
        let state = controller.determine_state(1000, 1000, 0.50);
        assert_eq!(
            state,
            AutorateState::Steady,
            "Should be steady with normal conditions"
        );

        // Slightly above target but not bufferbloat
        let state2 = controller.determine_state(1200, 1000, 0.50);
        assert_eq!(
            state2,
            AutorateState::Steady,
            "Should be steady when slightly above target"
        );
    }

    #[test]
    fn test_state_priority_bufferbloat_over_high_load() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // Both high load AND bufferbloat - bufferbloat wins
        let state = controller.determine_state(2000, 1000, 0.90);
        assert_eq!(
            state,
            AutorateState::Bufferbloat,
            "Bufferbloat should take priority"
        );
    }

    #[test]
    fn test_interpolate_params_min() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // rate=0.0 should return min_params
        let params = controller.interpolate_params(0.0, 0);
        assert_eq!(
            params, config.min_params[0],
            "rate=0.0 should equal min_params"
        );

        let params1 = controller.interpolate_params(0.0, 1);
        assert_eq!(
            params1, config.min_params[1],
            "rate=0.0 should equal min_params for class 1"
        );
    }

    #[test]
    fn test_interpolate_params_baseline() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // rate=0.5 should return baseline_params
        let params = controller.interpolate_params(0.5, 0);
        assert_eq!(
            params, config.baseline_params[0],
            "rate=0.5 should equal baseline_params"
        );

        let params1 = controller.interpolate_params(0.5, 2);
        assert_eq!(
            params1, config.baseline_params[2],
            "rate=0.5 should equal baseline_params for class 2"
        );
    }

    #[test]
    fn test_interpolate_params_max() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // rate=1.0 should return max_params
        let params = controller.interpolate_params(1.0, 0);
        assert_eq!(
            params, config.max_params[0],
            "rate=1.0 should equal max_params"
        );

        let params1 = controller.interpolate_params(1.0, 3);
        assert_eq!(
            params1, config.max_params[3],
            "rate=1.0 should equal max_params for class 3"
        );
    }

    #[test]
    fn test_linear_interpolation_midpoint() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // rate=0.25 should be halfway between min and baseline
        let params = controller.interpolate_params(0.25, 0);
        let expected = [
            150, // 100 + (200-100)*0.5 = 150
            300, // 200 + (400-200)*0.5 = 300
            450, // 300 + (600-300)*0.5 = 450
            600, // 400 + (800-400)*0.5 = 600
            750, // 500 + (1000-500)*0.5 = 750
        ];
        assert_eq!(
            params, expected,
            "rate=0.25 should be halfway between min and baseline"
        );
    }

    #[test]
    fn test_linear_interpolation_upper_half() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // rate=0.75 should be halfway between baseline and max
        let params = controller.interpolate_params(0.75, 0);
        let expected = [
            250,  // 200 + (300-200)*0.5 = 250
            500,  // 400 + (600-400)*0.5 = 500
            750,  // 600 + (900-600)*0.5 = 750
            1000, // 800 + (1200-800)*0.5 = 1000
            1250, // 1000 + (1500-1000)*0.5 = 1250
        ];
        assert_eq!(
            params, expected,
            "rate=0.75 should be halfway between baseline and max"
        );
    }

    #[test]
    fn test_controller_creation() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // Should have 4 class states
        let stats = controller.get_stats();
        assert_eq!(stats.total_classes, DESCENT_CLASS_MAX);

        // All classes should start at baseline
        for class in 0..DESCENT_CLASS_MAX as u32 {
            let state = controller.get_class_state(class).unwrap();
            assert_eq!(
                state.current_rate, 0.5,
                "Class {} should start at baseline rate",
                class
            );
            assert_eq!(
                state.state,
                AutorateState::Steady,
                "Class {} should start in steady state",
                class
            );
        }
    }

    #[test]
    fn test_controller_update() {
        let config = test_config();
        let mut controller = AutorateController::new(4, &config);

        // Update with high load (should trigger LoadHigh)
        let (params, state) = controller.update(0, 500, 1000, 0.90);

        assert_eq!(
            state,
            AutorateState::LoadHigh,
            "High load should trigger LoadHigh"
        );

        // Params should be between baseline and max (since rate increased from 0.5)
        let baseline = config.baseline_params[0];
        let max = config.max_params[0];
        for i in 0..PARAM_COUNT {
            assert!(
                params[i] >= baseline[i] && params[i] <= max[i],
                "Param {} should be between baseline and max: {} in [{}, {}]",
                i,
                params[i],
                baseline[i],
                max[i]
            );
        }
    }

    #[test]
    fn test_controller_update_bufferbloat() {
        let config = test_config();
        let mut controller = AutorateController::new(4, &config);

        // Update with bufferbloat conditions
        let (params, state) = controller.update(1, 2000, 1000, 0.50);

        assert_eq!(
            state,
            AutorateState::Bufferbloat,
            "High latency ratio should trigger bufferbloat"
        );

        // Params should be between min and baseline (since rate decreased from 0.5)
        let min = config.min_params[1];
        let baseline = config.baseline_params[1];
        for i in 0..PARAM_COUNT {
            assert!(
                params[i] >= min[i] && params[i] <= baseline[i],
                "Param {} should be between min and baseline: {} in [{}, {}]",
                i,
                params[i],
                min[i],
                baseline[i]
            );
        }
    }

    #[test]
    fn test_stats_tracking() {
        let config = test_config();
        let mut controller = AutorateController::new(4, &config);

        // Initial stats
        let stats_before = controller.get_stats();
        assert_eq!(stats_before.adjustments_up, 0);
        assert_eq!(stats_before.adjustments_down, 0);

        // Trigger upward adjustment
        controller.update(0, 500, 1000, 0.90);
        let stats_after_up = controller.get_stats();
        assert!(
            stats_after_up.adjustments_up >= 1,
            "Should track upward adjustment"
        );

        // Trigger downward adjustment
        // Wait for refractory
        std::thread::sleep(std::time::Duration::from_millis(60));
        controller.update(0, 2000, 1000, 0.50);
        let stats_after_down = controller.get_stats();
        assert!(
            stats_after_down.adjustments_down >= 1,
            "Should track downward adjustment"
        );
    }

    #[test]
    fn test_disabled_controller() {
        let mut config = test_config();
        config.enabled = false;
        let mut controller = AutorateController::new(4, &config);

        // Update with latency at target (no bufferbloat) and high load
        // latency=1000, target=1000 (ratio=1.0, no bufferbloat)
        // load=0.90 (> 0.75 high_load_threshold)
        let (params, state) = controller.update(0, 1000, 1000, 0.90);

        // With load=0.90 (90%), should be LoadHigh state even when disabled
        assert_eq!(
            state,
            AutorateState::LoadHigh,
            "Disabled controller should still calculate and return actual state for monitoring"
        );
        assert_eq!(
            params, config.baseline_params[0],
            "Disabled controller should return baseline params"
        );

        // Verify state is tracked even when disabled
        let class_state = controller.get_class_state(0).unwrap();
        assert_eq!(class_state.load_percent, 0.90);
        assert_eq!(class_state.state, AutorateState::LoadHigh);
    }

    #[test]
    fn test_invalid_class() {
        let config = test_config();
        let mut controller = AutorateController::new(4, &config);

        // Invalid class should return baseline
        let (params, state) = controller.update(99, 500, 1000, 0.50);

        assert_eq!(
            state,
            AutorateState::Steady,
            "Invalid class should return steady"
        );
        assert_eq!(
            params, config.baseline_params[3],
            "Invalid class should clamp to max class baseline"
        );
    }

    #[test]
    fn test_zero_target_latency() {
        let config = test_config();
        let controller = AutorateController::new(4, &config);

        // Zero target should not panic, should fall back to load-based
        let state = controller.determine_state(1000, 0, 0.90);
        assert_eq!(
            state,
            AutorateState::LoadHigh,
            "Should fall back to load-based detection"
        );

        let state2 = controller.determine_state(1000, 0, 0.10);
        assert_eq!(
            state2,
            AutorateState::LoadLow,
            "Should fall back to load-based detection"
        );

        let state3 = controller.determine_state(1000, 0, 0.50);
        assert_eq!(
            state3,
            AutorateState::Steady,
            "Should fall back to load-based detection"
        );
    }

    #[test]
    fn test_ewma_latency_tracking() {
        let config = test_config();
        let mut controller = AutorateController::new(4, &config);

        // Initial update
        controller.update(0, 1000, 1000, 0.50);
        let state1 = controller.get_class_state(0).unwrap();
        assert_eq!(state1.latency_ewma_ns, 1000);
        assert_eq!(state1.prev_latency_ns, 0);

        // Second update
        controller.update(0, 2000, 1000, 0.50);
        let state2 = controller.get_class_state(0).unwrap();
        assert_eq!(state2.latency_ewma_ns, 2000);
        assert_eq!(state2.prev_latency_ns, 1000);
    }

    #[test]
    fn test_load_percent_tracking() {
        let config = test_config();
        let mut controller = AutorateController::new(4, &config);

        controller.update(0, 1000, 1000, 0.75);
        let state = controller.get_class_state(0).unwrap();
        assert_eq!(state.load_percent, 0.75);
    }
}
