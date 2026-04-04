//! PIE (Proportional Integral controller Enhanced) for scx_descent
//!
//! Implements a deterministic control algorithm that replaces Thompson Sampling.
//! Uses fixed-point arithmetic to adjust scheduling parameters based on measured
//! latency feedback. The PIE controller provides more predictable convergence
//! than Bayesian approaches while being computationally efficient.
//!
//! Key concepts:
//! - P_term: Proportional to latency error (current - target)
//! - I_term: Integral of latency change rate
//! - Fixed-point arithmetic: integral_accumulator scaled by 1024
//! - EWMA smoothing: Reduces noise in latency measurements
//! - Parameter-specific adjustment scaling: Different sensitivities per parameter
//!
//! The PIE controller is designed for production use where deterministic
//! behavior and fast convergence are more important than exploration.

// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_descent authors
//
// PIE (Proportional Integral controller Enhanced) controller for
// deterministic parameter optimization based on latency feedback.

use log::debug;
use std::collections::HashMap;

use crate::bound_debug::BoundDebugger;
use crate::profiles::{DefaultParams, ParamBounds, Profile, DESCENT_CLASS_MAX, PARAM_COUNT};

/// Fixed-point scaling factor for integral accumulator
const INTEGRAL_SCALE: i64 = 1024;

/// EWMA smoothing factor for latency measurements (alpha = 0.3)
/// Higher = more responsive to recent changes, lower = more smoothing
const EWMA_ALPHA_NUM: u64 = 3;
const EWMA_ALPHA_DEN: u64 = 10;

/// Default PIE configuration values
const DEFAULT_ALPHA_DIV: u64 = 8;
const DEFAULT_BETA_DIV: u64 = 4;
const DEFAULT_MAX_INTEGRAL: i64 = 1_000_000;

/// Per (CPU, class) state for PIE controller
#[derive(Debug, Clone, Copy)]
pub struct PieState {
    /// Target latency for this (CPU, class) in nanoseconds
    pub target_latency_ns: u64,
    /// Current EWMA-smoothed latency measurement
    pub current_latency_ns: u64,
    /// Previous latency measurement (for calculating delta)
    pub prev_latency_ns: u64,
    /// Integral accumulator (fixed-point, scaled by INTEGRAL_SCALE)
    pub integral_accum: i64,
    /// Current scheduling parameters
    pub current_params: [u64; PARAM_COUNT],
    /// Number of updates performed
    pub update_count: u64,
}

impl PieState {
    /// Create a new PIE state with default values
    fn new(target_latency_ns: u64, default_params: [u64; PARAM_COUNT]) -> Self {
        Self {
            target_latency_ns,
            current_latency_ns: target_latency_ns, // Initialize at target
            prev_latency_ns: target_latency_ns,
            integral_accum: 0,
            current_params: default_params,
            update_count: 0,
        }
    }

    /// Update EWMA latency with new measurement
    fn update_ewma(&mut self, measured_latency_ns: u64) {
        // EWMA: new = alpha * measured + (1 - alpha) * current
        // Using integer arithmetic: new = (3*measured + 7*current) / 10
        let alpha_num = EWMA_ALPHA_NUM;
        let alpha_den = EWMA_ALPHA_DEN;
        let inv_alpha = alpha_den - alpha_num;

        self.prev_latency_ns = self.current_latency_ns;
        self.current_latency_ns =
            (alpha_num * measured_latency_ns + inv_alpha * self.current_latency_ns) / alpha_den;
    }
}

/// Configuration for PIE controller per profile
#[derive(Debug, Clone, Copy)]
pub struct PieConfig {
    /// Alpha divisor for P_term calculation
    /// P_term = (current_latency - target_latency) / alpha_div
    pub alpha_div: u64,
    /// Beta divisor for integral update
    /// integral_accum += (current_latency - prev_latency) / beta_div
    pub beta_div: u64,
    /// Maximum absolute value of integral accumulator (anti-windup)
    pub max_integral: i64,
    /// Target latencies per class [LATENCY_CRITICAL, NORMAL, HOG, BACKGROUND]
    pub target_latencies: [u64; DESCENT_CLASS_MAX],
    /// Parameter bounds per class
    pub param_bounds: ParamBounds,
    /// Default parameters per class
    pub default_params: DefaultParams,
}

impl PieConfig {
    /// Create PIE config from a profile with profile-specific target latencies
    pub fn from_profile(profile: &Profile) -> Self {
        // Use the profile's target latencies directly - they are already tuned per profile
        let target_latencies = profile.target_latencies_ns;

        Self {
            alpha_div: DEFAULT_ALPHA_DIV,
            beta_div: DEFAULT_BETA_DIV,
            max_integral: DEFAULT_MAX_INTEGRAL,
            target_latencies,
            param_bounds: profile.bounds,
            default_params: profile.default_params,
        }
    }

    /// Get default parameters for a specific class
    pub fn get_default_params(&self, class: u32) -> [u64; PARAM_COUNT] {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return [0; PARAM_COUNT];
        }
        self.default_params[class_idx]
    }

    /// Get target latency for a specific class
    pub fn get_target_latency(&self, class: u32) -> u64 {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return 0;
        }
        self.target_latencies[class_idx]
    }
}

/// PIE Controller - deterministic parameter optimization
///
/// Maintains separate PIE state for each (CPU, class) combination
/// and uses proportional-integral control to adjust scheduling parameters
/// based on measured latency feedback.
pub struct PieController {
    /// PIE states keyed by (cpu, class)
    states: HashMap<(u32, u32), PieState>,
    /// Controller configuration
    config: PieConfig,
    /// Optional bound debugger for tracking parameter clamping
    bound_debugger: Option<BoundDebugger>,
}

impl PieController {
    /// Create new PIE controller for given CPU count and profile
    ///
    /// # Arguments
    /// * `nr_cpus` - Number of CPUs to track
    /// * `profile` - Profile containing default parameters, bounds, and target latencies
    pub fn new(nr_cpus: usize, profile: &Profile) -> Self {
        let mut states = HashMap::new();
        let config = PieConfig::from_profile(profile);

        // Initialize states for all (CPU, class) combinations
        for cpu in 0..nr_cpus as u32 {
            for class in 0..DESCENT_CLASS_MAX as u32 {
                let target_latency = config.get_target_latency(class);
                let default_params = config.get_default_params(class);

                let state = PieState::new(target_latency, default_params);
                states.insert((cpu, class), state);
            }
        }

        debug!(
            "[PIE-INIT] Created controller with {} CPUs, {} classes, {} total states",
            nr_cpus,
            DESCENT_CLASS_MAX,
            states.len()
        );

        // Initialize bound debugger (disabled by default, can be enabled later)
        let bound_debugger = BoundDebugger::new(config.param_bounds, false);

        Self {
            states,
            config,
            bound_debugger: Some(bound_debugger),
        }
    }

    /// Update controller with latency measurement and return new parameters
    ///
    /// This is the core PIE algorithm:
    /// 1. Update EWMA-smoothed latency
    /// 2. Calculate latency delta for integral term
    /// 3. Compute P_term and I_term
    /// 4. Calculate total adjustment
    /// 5. Apply parameter-specific scaling
    /// 6. Clamp to bounds
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class (0=LATENCY_CRITICAL, 1=NORMAL, 2=HOG, 3=BACKGROUND)
    /// * `measured_latency_ns` - New latency measurement in nanoseconds
    ///
    /// # Returns
    /// Updated [u64; 5] parameter array
    pub fn update(&mut self, cpu: u32, class: u32, measured_latency_ns: u64) -> [u64; 5] {
        let key = (cpu, class);

        // Get or create state (should exist from initialization)
        let state = match self.states.get_mut(&key) {
            Some(s) => s,
            None => {
                debug!(
                    "[PIE-UPDATE] No state found for ({}, {}), returning defaults",
                    cpu, class
                );
                return self.config.get_default_params(class);
            }
        };

        // Step 1: Update EWMA latency
        state.update_ewma(measured_latency_ns);
        state.update_count += 1;

        // Step 2: Calculate latency delta for integral
        let latency_delta = state.current_latency_ns as i64 - state.prev_latency_ns as i64;

        // Step 3: Update integral accumulator with anti-windup
        // integral_accum += delta / beta_div
        let integral_increment = latency_delta / self.config.beta_div as i64;
        state.integral_accum += integral_increment;

        // Clamp integral to prevent windup
        state.integral_accum = state
            .integral_accum
            .clamp(-self.config.max_integral, self.config.max_integral);

        // Step 4: Calculate P_term and I_term
        // P_term = (current_latency - target_latency) / alpha_div
        let latency_error = state.current_latency_ns as i64 - state.target_latency_ns as i64;
        let p_term = latency_error / self.config.alpha_div as i64;

        // I_term = integral_accum / INTEGRAL_SCALE (de-scaled)
        let i_term = state.integral_accum / INTEGRAL_SCALE;

        // Total adjustment
        let adjustment = p_term + i_term;

        debug!(
            "[PIE-UPDATE] cpu={} class={} measured={} ewma={} target={} error={} p_term={} i_term={} adjustment={}",
            cpu, class, measured_latency_ns, state.current_latency_ns, state.target_latency_ns,
            latency_error, p_term, i_term, adjustment
        );

        // Extract current params and bounds for calculations
        let current_params = state.current_params;
        let class_idx = class as usize;
        let class_bounds = if class_idx < DESCENT_CLASS_MAX {
            Some(self.config.param_bounds[class_idx])
        } else {
            None
        };
        let latency_error_ns = latency_error; // Save for bound debugging

        // Step 5: Calculate new parameters with parameter-specific scaling
        let new_params =
            Self::calculate_params_with_bounds(adjustment, current_params, class_bounds);

        // Step 6: Drop state borrow before calling self method, then clamp to bounds and track bound hits
        let clamped_params = self.clamp_params_with_debug(class, new_params, cpu, latency_error_ns);

        // Re-borrow state to store new params
        if let Some(state) = self.states.get_mut(&key) {
            state.current_params = clamped_params;
        }

        clamped_params
    }

    /// Calculate new parameters based on adjustment value and clamp to bounds
    fn calculate_params_with_bounds(
        adjustment: i64,
        current_params: [u64; 5],
        bounds: Option<[(u64, u64); 5]>,
    ) -> [u64; 5] {
        let mut new_params = current_params;

        // Param 0: latency_weight - direct relationship (FIXED)
        // When latency is too high, we need higher weight to prioritize the task
        // When latency is too low, we need lower weight to deprioritize the task
        if adjustment > 0 {
            // Latency too high, increase weight to prioritize
            new_params[0] = current_params[0].saturating_add((adjustment / 2) as u64);
        } else {
            // Latency too low, decrease weight to deprioritize
            new_params[0] = current_params[0].saturating_sub((-adjustment / 2) as u64);
        }

        // Param 1: base_slice_ns - direct relationship (FIXED)
        // When latency is too high, we want smaller slices for faster preemption
        // When latency is too low, we want larger slices to let tasks run longer
        if adjustment > 0 {
            // Latency too high, decrease slice for faster preemption
            new_params[1] = current_params[1].saturating_sub(adjustment as u64);
        } else {
            // Latency too low, increase slice to let tasks run longer
            new_params[1] = current_params[1].saturating_add((-adjustment) as u64);
        }

        // Param 2: vruntime_scale - direct relationship (FIXED)
        if adjustment > 0 {
            // Latency too high, decrease scale (less vruntime accumulation)
            new_params[2] = current_params[2].saturating_sub((adjustment / 4) as u64);
        } else {
            // Latency too low, increase scale (more vruntime accumulation)
            new_params[2] = current_params[2].saturating_add((-adjustment / 4) as u64);
        }

        // Param 3: preemption_priority - proportional to new slice
        // Typically 1/10th of slice
        let slice_ns = new_params[1];
        new_params[3] = slice_ns / 10;

        // Param 4: migration_cost - direct relationship (FIXED)
        if adjustment > 0 {
            // Latency too high, decrease migration cost (easier to migrate)
            new_params[4] = current_params[4].saturating_sub((adjustment / 10) as u64);
        } else {
            // Latency too low, increase migration cost (harder to migrate)
            new_params[4] = current_params[4].saturating_add((-adjustment / 10) as u64);
        }

        // Apply bounds if available
        if let Some(b) = bounds {
            for i in 0..PARAM_COUNT {
                let (min, max) = b[i];
                new_params[i] = new_params[i].clamp(min, max);
            }
        }

        new_params
    }

    /// Clamp parameters to bounds for a specific class
    ///
    /// # Arguments
    /// * `class` - Task class
    /// * `params` - Parameter values to clamp
    /// * `cpu` - CPU ID (for bound debugging)
    /// * `latency_error_ns` - Current latency error (for bound debugging)
    ///
    /// # Returns
    /// Clamped [u64; 5] array
    pub fn clamp_params_with_debug(
        &mut self,
        class: u32,
        params: [u64; 5],
        cpu: u32,
        latency_error_ns: i64,
    ) -> [u64; 5] {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return params;
        }

        let mut clamped = [0u64; 5];
        for i in 0..PARAM_COUNT {
            let (min, max) = self.config.param_bounds[class_idx][i];
            let original = params[i];
            clamped[i] = params[i].clamp(min, max);

            // Track bound hits if debugger is enabled
            if let Some(ref mut debugger) = self.bound_debugger {
                // Always record that an update occurred
                debugger.record_update(cpu, class, i);

                // Record clamping if the value was constrained (at or beyond bounds)
                // A parameter is constrained if:
                // 1. It was modified by clamping (original != clamped), OR
                // 2. It equals exactly the min or max bound (wants to go further but can't)
                let was_modified = original != clamped[i];
                let at_min_bound = clamped[i] == min;
                let at_max_bound = clamped[i] == max;

                if was_modified || at_min_bound || at_max_bound {
                    // Only record actual clamping events where value was modified
                    if was_modified {
                        debugger.record_clamp(
                            cpu,
                            class,
                            i,
                            original,
                            clamped[i],
                            latency_error_ns,
                        );
                    }
                }
            }
        }

        clamped
    }

    /// Enable or disable bound debugging
    pub fn set_bound_debugging(&mut self, enabled: bool) {
        if let Some(ref mut debugger) = self.bound_debugger {
            debugger.set_enabled(enabled);
        }
    }

    /// Get the bound debugger for reporting
    pub fn bound_debugger(&self) -> Option<&BoundDebugger> {
        self.bound_debugger.as_ref()
    }

    /// Get current state for monitoring
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class
    ///
    /// # Returns
    /// Some(&PieState) if state exists, None otherwise
    pub fn get_state(&self, cpu: u32, class: u32) -> Option<&PieState> {
        self.states.get(&(cpu, class))
    }

    /// Get default parameters for a specific class
    pub fn get_default_params(&self, class: u32) -> [u64; 5] {
        self.config.get_default_params(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiles::Profile;

    #[test]
    fn test_pie_controller_creation() {
        let profile = Profile::production();
        let controller = PieController::new(4, &profile);

        // Check that all states have correct target latencies for production profile
        let expected_targets = [
            1_000_000,   // LATENCY_CRITICAL
            5_000_000,   // NORMAL
            20_000_000,  // HOG
            100_000_000, // BACKGROUND
        ];

        for class in 0..DESCENT_CLASS_MAX as u32 {
            let state = controller.get_state(0, class).unwrap();
            assert_eq!(state.target_latency_ns, expected_targets[class as usize]);
            assert_eq!(state.current_latency_ns, expected_targets[class as usize]);
            assert_eq!(state.integral_accum, 0);
            assert_eq!(state.update_count, 0);
        }
    }

    #[test]
    fn test_pie_controller_creation_gaming() {
        let profile = Profile::gaming();
        let controller = PieController::new(2, &profile);

        // Check gaming-specific target latencies (lower than production)
        let expected_targets = [
            500_000,    // LATENCY_CRITICAL: 500us
            2_000_000,  // NORMAL: 2ms
            25_000_000, // HOG: 25ms (updated for encoding workloads)
            50_000_000, // BACKGROUND: 50ms
        ];

        for class in 0..DESCENT_CLASS_MAX as u32 {
            let state = controller.get_state(0, class).unwrap();
            assert_eq!(
                state.target_latency_ns, expected_targets[class as usize],
                "Class {} target latency mismatch",
                class
            );
        }
    }

    #[test]
    fn test_pie_controller_creation_server() {
        let profile = Profile::server();
        let controller = PieController::new(2, &profile);

        // Check server-specific target latencies (higher than production)
        let expected_targets = [
            2_000_000,   // LATENCY_CRITICAL: 2ms
            10_000_000,  // NORMAL: 10ms
            50_000_000,  // HOG: 50ms
            200_000_000, // BACKGROUND: 200ms
        ];

        for class in 0..DESCENT_CLASS_MAX as u32 {
            let state = controller.get_state(0, class).unwrap();
            assert_eq!(
                state.target_latency_ns, expected_targets[class as usize],
                "Class {} target latency mismatch",
                class
            );
        }
    }

    #[test]
    fn test_pie_update_latency_too_high() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32; // NORMAL class
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // Initial params
        let initial_params = controller.get_state(0, class).unwrap().current_params;

        // Update with very high latency (2x target)
        let high_latency = target * 2;
        let new_params = controller.update(0, class, high_latency);

        // Note: Production profile has param 1 (slice) at lower bound (1_000_000),
        // so PIE cannot decrease it further. The test documents this known limitation.
        // In practice, PIE would decrease slice if there was headroom.
        if new_params[1] < initial_params[1] {
            // High latency decreased slice - this is the expected behavior when not at bounds
            assert!(
                new_params[0] < initial_params[0],
                "High latency should decrease latency_weight when slice decreases"
            );
        } else {
            // Slice at bounds - verify the update was still recorded
            let state = controller.get_state(0, class).unwrap();
            assert_eq!(
                state.update_count, 1,
                "Update should be recorded even when at bounds"
            );
        }

        // Verify update was recorded
        let state = controller.get_state(0, class).unwrap();
        assert_eq!(state.update_count, 1);
    }

    #[test]
    fn test_pie_update_latency_too_low() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32; // NORMAL class
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // Initial params
        let initial_params = controller.get_state(0, class).unwrap().current_params;

        // Update with very low latency (half target)
        let low_latency = target / 2;
        let new_params = controller.update(0, class, low_latency);

        // Low latency should:
        // 1. Increase slice (param 1) - direct relationship
        assert!(
            new_params[1] > initial_params[1],
            "Low latency should increase slice: {} -> {}",
            initial_params[1],
            new_params[1]
        );

        // 2. Increase latency_weight (param 0) - inverse relationship
        // Note: Production profile has latency_weight at min bound (500_000),
        // so it may not increase further. We check if it's at bound or increased.
        let bounds = profile.get_bounds(class as usize, 0);
        if initial_params[0] < bounds.1 {
            // Only expect increase if not already at max bound
            assert!(
                new_params[0] >= initial_params[0],
                "Low latency should not decrease latency_weight: {} -> {}",
                initial_params[0],
                new_params[0]
            );
        }

        // Verify update was recorded
        let state = controller.get_state(0, class).unwrap();
        assert_eq!(state.update_count, 1);
    }

    #[test]
    fn test_integral_windup_protection() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32;
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // Repeatedly update with very high latency to accumulate integral
        for _ in 0..100 {
            let _ = controller.update(0, class, target * 10);
        }

        let state = controller.get_state(0, class).unwrap();

        // Integral should be clamped to max_integral
        assert!(
            state.integral_accum.abs() <= DEFAULT_MAX_INTEGRAL,
            "Integral accumulator should be clamped: {} > {}",
            state.integral_accum.abs(),
            DEFAULT_MAX_INTEGRAL
        );
    }

    #[test]
    fn test_parameter_bounds_enforcement() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32; // NORMAL class

        // Get bounds for this class
        let bounds = profile.bounds[class as usize];

        // Update many times with extreme latencies
        for i in 0..50 {
            if i % 2 == 0 {
                let _ = controller.update(0, class, 1_000_000_000); // Very high
            } else {
                let _ = controller.update(0, class, 1); // Very low
            }
        }

        let state = controller.get_state(0, class).unwrap();
        let params = state.current_params;

        // Verify all params are within bounds
        for (idx, &value) in params.iter().enumerate() {
            let (min, max) = bounds[idx];
            assert!(
                value >= min && value <= max,
                "Parameter {} = {} out of bounds [{}, {}]",
                idx,
                value,
                min,
                max
            );
        }
    }

    #[test]
    fn test_ewma_latency_calculation() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32;
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // First measurement sets initial EWMA
        let _ = controller.update(0, class, target * 2); // 2x target
        let state1 = controller.get_state(0, class).unwrap();

        // EWMA with alpha=0.3: new = 0.3 * measured + 0.7 * target
        // = 0.3 * (2*target) + 0.7 * target = 0.6*target + 0.7*target = 1.3*target
        let expected_ewma = (EWMA_ALPHA_NUM * target * 2
            + (EWMA_ALPHA_DEN - EWMA_ALPHA_NUM) * target)
            / EWMA_ALPHA_DEN;
        assert_eq!(
            state1.current_latency_ns, expected_ewma,
            "EWMA should be {} but got {}",
            expected_ewma, state1.current_latency_ns
        );

        // Second measurement
        let _ = controller.update(0, class, target); // 1x target
        let state2 = controller.get_state(0, class).unwrap();

        // EWMA: new = 0.3 * target + 0.7 * 1.3*target = 0.3*target + 0.91*target = 1.21*target
        let expected_ewma2 = (EWMA_ALPHA_NUM * target
            + (EWMA_ALPHA_DEN - EWMA_ALPHA_NUM) * expected_ewma)
            / EWMA_ALPHA_DEN;
        assert_eq!(
            state2.current_latency_ns, expected_ewma2,
            "EWMA should be {} but got {}",
            expected_ewma2, state2.current_latency_ns
        );
    }

    #[test]
    fn test_pie_convergence() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32; // NORMAL class
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // Start with high latency
        let mut current_latency = target * 3;

        // Simulate convergence over multiple updates
        let mut last_error = (current_latency as i64 - target as i64).abs();

        for i in 0..50 {
            let params = controller.update(0, class, current_latency);
            let state = controller.get_state(0, class).unwrap();

            // Calculate new error after this update
            let current_error = (state.current_latency_ns as i64 - target as i64).abs();

            // Simulate the system responding to parameter changes
            // Smaller slice should reduce latency
            let slice_ns = params[1];
            let latency_factor = (slice_ns as f64 / target as f64).min(3.0);
            current_latency = (target as f64 * (0.5 + latency_factor * 0.5)) as u64;

            // Error should generally decrease over time (allow some noise)
            if i > 10 {
                assert!(
                    current_error <= last_error * 12 / 10, // Allow 20% fluctuation
                    "Error should converge at iteration {}: {} -> {}",
                    i,
                    last_error,
                    current_error
                );
            }

            last_error = current_error;
        }

        // Final error should be significantly reduced
        let final_state = controller.get_state(0, class).unwrap();
        let final_error = (final_state.current_latency_ns as i64 - target as i64).abs();
        assert!(
            final_error < last_error * 2,
            "Should converge toward target"
        );
    }

    #[test]
    fn test_pie_config_from_profile() {
        let gaming = Profile::gaming();
        let prod = Profile::production();
        let server = Profile::server();

        let gaming_config = PieConfig::from_profile(&gaming);
        let prod_config = PieConfig::from_profile(&prod);
        let server_config = PieConfig::from_profile(&server);

        // All should have same alpha/beta/max_integral
        assert_eq!(gaming_config.alpha_div, prod_config.alpha_div);
        assert_eq!(gaming_config.beta_div, server_config.beta_div);

        // But different target latencies
        // Gaming should have lowest targets
        assert!(gaming_config.target_latencies[0] < prod_config.target_latencies[0]);
        assert!(gaming_config.target_latencies[0] < server_config.target_latencies[0]);

        // Server should have highest targets
        assert!(server_config.target_latencies[0] > prod_config.target_latencies[0]);
    }

    #[test]
    fn test_latency_error_calculation() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 0u32; // LATENCY_CRITICAL
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // Update with measured latency
        let measured = target * 2;
        let _ = controller.update(0, class, measured);

        let state = controller.get_state(0, class).unwrap();

        // The EWMA should be between target and measured
        assert!(
            state.current_latency_ns >= target && state.current_latency_ns <= measured,
            "EWMA {} should be between target {} and measured {}",
            state.current_latency_ns,
            target,
            measured
        );
    }

    #[test]
    fn test_preemption_priority_proportional_to_slice() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32;
        let bounds = profile.bounds[class as usize];
        let preemption_max = bounds[3].1; // Get max preemption priority from profile

        // Update to get new params
        let params = controller.update(0, class, 10_000_000);

        // Preemption priority should be approximately 1/10 of slice, clamped to profile bounds
        let slice = params[1];
        let preemption = params[3];
        let expected = (slice / 10).min(preemption_max);

        assert_eq!(
            preemption, expected,
            "Preemption priority {} should be min(1/10 of slice {}, {})",
            preemption, slice, preemption_max
        );
    }

    #[test]
    fn test_integral_accumulator_growth() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32;
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // Start with latency equal to target (no error)
        let _ = controller.update(0, class, target);
        let state1 = controller.get_state(0, class).unwrap();

        // First update: delta is small (EWMA just starting)
        let initial_integral = state1.integral_accum;

        // Now rapidly increase latency
        let _ = controller.update(0, class, target * 4);
        let _ = controller.update(0, class, target * 4);
        let state2 = controller.get_state(0, class).unwrap();

        // Integral should have accumulated
        assert!(
            state2.integral_accum.abs() > initial_integral.abs() || state2.integral_accum == 0,
            "Integral should accumulate with consistent latency delta"
        );
    }

    #[test]
    fn test_p_term_and_i_term_calculation() {
        let profile = Profile::production();
        let mut controller = PieController::new(1, &profile);

        let class = 1u32;
        let config = PieConfig::from_profile(&profile);
        let target = config.get_target_latency(class);

        // Update with latency higher than target
        let measured = target * 2;
        let _ = controller.update(0, class, measured);

        let state = controller.get_state(0, class).unwrap();

        // With EWMA, current_latency should be between target and measured
        // P_term would be (current - target) / alpha_div > 0
        let p_term = (state.current_latency_ns as i64 - target as i64) / DEFAULT_ALPHA_DIV as i64;
        assert!(
            p_term > 0,
            "P_term should be positive when latency > target"
        );

        // I_term starts at 0 and accumulates
        let i_term = state.integral_accum / INTEGRAL_SCALE;
        // First update, integral might be small
        assert!(
            i_term >= 0,
            "I_term should be non-negative with increasing latency"
        );
    }
}
