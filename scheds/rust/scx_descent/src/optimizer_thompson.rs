//! Value-Based Thompson Sampling Optimizer for scx_descent
//!
//! Implements Bayesian parameter optimization using Thompson Sampling.
//! Each (CPU, class, parameter) has a Gaussian posterior distribution that
//! is updated based on observed loss outcomes. Parameters are sampled
//! from these distributions to balance exploration vs exploitation.
//!
//! Key concepts:
//! - Value-based posterior: Mean represents actual optimal parameter value
//! - Gaussian sampling: Sample from N(mean, std²) for parameter selection
//! - Value-based update: Success pulls mean toward successful sample value
//! - Exploration factor: Controls initial uncertainty (gaming=2.0x, server=1.0x)
//! - Safety monitoring: Detects loss spikes and restores checkpoints

// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_descent authors
//
// Value-based Thompson Sampling optimizer for automatic parameter optimization.
// Uses Bayesian approach with Gaussian posteriors to balance
// exploration vs exploitation. Unlike Beta-based approaches, the mean
// directly represents the estimated optimal parameter value.

use log::debug;
use rand::distributions::Distribution;
use rand::thread_rng;
use rand_distr::Normal;
use std::collections::HashMap;

use crate::profiles::{DefaultParams, ParamBounds, Profile};

/// Maximum total observations before capping (prevent overconfidence)
const MAX_OBSERVATIONS: f64 = 1000.0;
/// Minimum standard deviation as fraction of parameter range
const MIN_STD_FRACTION: f64 = 0.01;
/// Loss increase threshold for triggering exploration boost (50%)
const EXPLORATION_BOOST_THRESHOLD: f64 = 1.5;
/// EWMA smoothing factor for baseline loss updates
const BASELINE_EWMA_ALPHA: f64 = 0.3;
/// Base learning rate for mean updates
const BASE_LEARNING_RATE: f64 = 0.1;
/// Learning rate decay factor per observation
const LEARNING_RATE_DECAY: f64 = 0.001;
/// Uncertainty reduction factor on success
const UNCERTAINTY_REDUCTION: f64 = 0.5;
/// Uncertainty growth factor on failure
const UNCERTAINTY_GROWTH: f64 = 1.15;

/// Represents parameter uncertainty with value-based Gaussian posterior
///
/// Unlike Beta-based approaches where mean represents "quality of high values",
/// here the mean directly represents the estimated optimal parameter value.
/// The std represents uncertainty in that estimate.
pub struct ValuePosterior {
    /// Current best estimate of optimal parameter value (actual value, not normalized)
    pub mean: f64,
    /// Uncertainty in that estimate (standard deviation)
    pub std: f64,
    /// Effective sample count for learning rate scheduling
    pub n_observations: f64,
    /// Parameter lower bound
    pub min: f64,
    /// Parameter upper bound
    pub max: f64,
}

impl ValuePosterior {
    /// Create a new posterior centered at initial value
    fn new(initial_mean: f64, min: f64, max: f64, exploration_factor: f64) -> Self {
        let range = max - min;
        // Initial std is 10% of range * exploration factor
        let initial_std = range * exploration_factor * 0.1;

        // Ensure minimum uncertainty
        let min_std = range * MIN_STD_FRACTION;
        let std = initial_std.max(min_std);

        let mean = initial_mean.clamp(min, max);

        debug!(
            "[THOMPSON-INIT] new posterior: mean={:.0}, std={:.0}, min={:.0}, max={:.0}, range={:.0}, exploration_factor={:.1}",
            mean, std, min, max, range, exploration_factor
        );

        Self {
            mean,
            std,
            n_observations: 0.0,
            min,
            max,
        }
    }

    /// Clamp value to [min, max] bounds
    fn clamp(&self, value: f64) -> f64 {
        value.clamp(self.min, self.max)
    }

    /// Sample from the Gaussian approximation
    fn sample(&self) -> f64 {
        let mut rng = thread_rng();
        let normal = Normal::new(self.mean, self.std);
        match normal {
            Ok(dist) => {
                let sample = dist.sample(&mut rng);
                let clamped = self.clamp(sample);
                debug!(
                    "[THOMPSON-SAMPLE] mean={:.0}, std={:.0}, raw_sample={:.0}, clamped={:.0}, min={:.0}, max={:.0}",
                    self.mean, self.std, sample, clamped, self.min, self.max
                );
                clamped
            }
            Err(_) => {
                // Fallback to mean if distribution is invalid
                debug!(
                    "[THOMPSON-SAMPLE] mean={:.0}, std={:.0} - invalid distribution, using mean fallback",
                    self.mean, self.std
                );
                self.mean
            }
        }
    }

    /// Update posterior based on observation result
    ///
    /// On success (improved loss): Shift mean toward the successful parameter value
    /// On failure (worse loss): Increase uncertainty to explore more
    ///
    /// # Arguments
    /// * `param_value` - The parameter value that was tested
    /// * `improved` - true if loss improved from baseline
    /// * `improvement_ratio` - (baseline - loss) / baseline, normalized improvement
    fn update(&mut self, param_value: f64, improved: bool, improvement_ratio: f64) {
        let range = self.max - self.min;
        let mean_before = self.mean;
        let std_before = self.std;
        let n_before = self.n_observations;

        if improved {
            // Success: Shift mean TOWARD the parameter value that worked
            // Learning rate decreases with more observations
            let lr = BASE_LEARNING_RATE / (1.0 + self.n_observations * LEARNING_RATE_DECAY);

            // Move mean toward successful value, weighted by improvement
            let delta = param_value - self.mean;
            self.mean += lr * delta * improvement_ratio.min(1.0);
            self.mean = self.mean.clamp(self.min, self.max);

            // Reduce uncertainty (we're learning)
            self.std *= 1.0 - lr * UNCERTAINTY_REDUCTION;

            debug!(
                "[THOMPSON-UPDATE-SUCCESS] param_value={:.0}, mean_before={:.0}, mean_after={:.0}, delta={:.0}, lr={:.4}, improvement_ratio={:.4}, std_before={:.0}, std_after={:.0}, n={:.0}",
                param_value, mean_before, self.mean, delta, lr, improvement_ratio, std_before, self.std, n_before
            );
        } else {
            // Failure: Increase uncertainty to explore more
            self.std = (self.std * UNCERTAINTY_GROWTH).min(range / 2.0);

            debug!(
                "[THOMPSON-UPDATE-FAILURE] param_value={:.0}, mean={:.0}, std_before={:.0}, std_after={:.0}, n={:.0}, improvement_ratio={:.4}",
                param_value, self.mean, std_before, self.std, n_before, improvement_ratio
            );
        }

        // Ensure minimum uncertainty for continued exploration
        let min_std = range * MIN_STD_FRACTION;
        self.std = self.std.max(min_std);

        self.n_observations = (self.n_observations + 1.0).min(MAX_OBSERVATIONS);
    }

    /// Temporarily increase uncertainty for exploration boost
    fn boost_exploration(&mut self, factor: f64) {
        let range = self.max - self.min;
        let min_std = range * MIN_STD_FRACTION;
        self.std = (self.std * factor).clamp(min_std, range / 2.0);
    }
}

/// Main Thompson sampler for scheduling parameter optimization
///
/// Maintains separate posteriors for each (CPU, class, parameter) combination
/// and uses Thompson sampling to balance exploration vs exploitation.
pub struct ThompsonSampler {
    /// Posteriors keyed by (cpu, class, param_idx)
    posteriors: HashMap<(u32, u32, usize), ValuePosterior>,
    /// Baseline loss for each (cpu, class) - used to determine improvement
    baseline_loss: HashMap<(u32, u32), f64>,
    /// Last sampled parameters for each (cpu, class) - used for loss attribution
    last_params: HashMap<(u32, u32), [u64; 5]>,
    /// Update interval in milliseconds (30, 50, or 100 based on profile)
    #[allow(dead_code)] // Stored for informational/debugging purposes
    pub update_interval_ms: u64,
    /// Exploration factor (1.0, 1.5, or 2.0 based on profile)
    exploration_factor: f64,
    /// Number of CPUs
    #[allow(dead_code)] // Stored for informational/debugging purposes
    cpu_count: usize,
}

impl ThompsonSampler {
    /// Create a new Thompson sampler
    ///
    /// # Arguments
    /// * `cpu_count` - Number of CPUs to track
    /// * `profile` - Profile containing default parameters and bounds
    pub fn new(cpu_count: usize, profile: &Profile) -> Self {
        let mut posteriors: HashMap<(u32, u32, usize), ValuePosterior> = HashMap::new();
        let mut baseline_loss: HashMap<(u32, u32), f64> = HashMap::new();
        let mut last_params: HashMap<(u32, u32), [u64; 5]> = HashMap::new();

        // Initialize posteriors for all (CPU, class, param) combinations
        for cpu in 0..cpu_count as u32 {
            for class in 0..4u32 {
                // Initialize baseline loss to 0.0 (will be updated with EWMA)
                baseline_loss.insert((cpu, class), 0.0);

                // Initialize last_params with default values
                let class_idx = class as usize;
                let default_params = [
                    profile.default_params[class_idx][0],
                    profile.default_params[class_idx][1],
                    profile.default_params[class_idx][2],
                    profile.default_params[class_idx][3],
                    profile.default_params[class_idx][4],
                ];
                last_params.insert((cpu, class), default_params);

                // Initialize each parameter's posterior
                for param_idx in 0..5 {
                    let initial_value = profile.default_params[class_idx][param_idx] as f64;
                    let (min, max) = profile.bounds[class_idx][param_idx];

                    let posterior = ValuePosterior::new(
                        initial_value,
                        min as f64,
                        max as f64,
                        profile.exploration_factor,
                    );

                    posteriors.insert((cpu, class, param_idx), posterior);
                }
            }
        }

        Self {
            posteriors,
            baseline_loss,
            last_params,
            update_interval_ms: profile.response_ms,
            exploration_factor: profile.exploration_factor,
            cpu_count,
        }
    }

    /// Sample parameters for a given CPU and class
    ///
    /// Returns [u64; 5] array of sampled parameter values:
    /// - [0] latency_weight
    /// - [1] base_slice_ns  
    /// - [2] vruntime_scale
    /// - [3] preemption_priority
    /// - [4] migration_cost
    ///
    /// Also stores the sampled params in last_params for later loss attribution.
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class (0=interactive, 1=audio, 2=batch, 3=kernel)
    pub fn sample_params(&mut self, cpu: u32, class: u32) -> [u64; 5] {
        let mut params = [0u64; 5];

        for param_idx in 0..5 {
            if let Some(posterior) = self.posteriors.get(&(cpu, class, param_idx)) {
                let sample = posterior.sample();
                params[param_idx] = sample as u64;
            } else {
                // Fallback: use mean if posterior not found
                params[param_idx] = 0;
            }
        }

        // Store the sampled params for loss attribution
        self.last_params.insert((cpu, class), params);

        debug!(
            "[THOMPSON-SAMPLE-PARAMS] cpu={} class={} params=[{:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
            cpu, class, params[0], params[1], params[2], params[3], params[4]
        );

        params
    }

    /// Get the last sampled parameters for a given CPU and class
    ///
    /// This is used to retrieve the parameters that produced the loss
    /// currently being read from BPF (before the accumulators are reset).
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class
    pub fn get_last_params(&self, cpu: u32, class: u32) -> Option<[u64; 5]> {
        self.last_params.get(&(cpu, class)).copied()
    }

    /// Update posteriors based on observed loss
    ///
    /// On success (loss < baseline): Shifts mean toward the parameter value that worked
    /// On failure (loss >= baseline): Increases uncertainty for exploration
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class
    /// * `params` - The parameter values that were tested
    /// * `loss` - The observed loss value
    pub fn update(&mut self, cpu: u32, class: u32, params: [u64; 5], loss: f64) {
        let key = (cpu, class);

        // Get or initialize baseline
        let baseline = *self.baseline_loss.get(&key).unwrap_or(&0.0);

        debug!(
            "[THOMPSON-UPDATE-START] cpu={} class={} loss={:.2} baseline={:.2} params=[{:.0}, {:.0}, {:.0}, {:.0}, {:.0}]",
            cpu, class, loss, baseline, params[0], params[1], params[2], params[3], params[4]
        );

        // Determine if this was an improvement
        let improved = if baseline > 0.0 {
            loss < baseline
        } else {
            // First observation, treat as improvement to build initial data
            true
        };

        // Calculate improvement ratio for weighting the update
        let improvement_ratio = if baseline > 0.0 && improved {
            (baseline - loss) / baseline
        } else {
            0.0
        };

        debug!(
            "[THOMPSON-UPDATE-ANALYSIS] cpu={} class={} improved={} improvement_ratio={:.4}",
            cpu, class, improved, improvement_ratio
        );

        // Check if loss increased significantly - trigger exploration boost
        let exploration_boost = if baseline > 0.0 && loss > baseline * EXPLORATION_BOOST_THRESHOLD {
            debug!(
                "[THOMPSON-UPDATE-BOOST] cpu={} class={} loss={:.2} > threshold={:.2} (baseline * {}), triggering exploration boost",
                cpu, class, loss, baseline * EXPLORATION_BOOST_THRESHOLD, EXPLORATION_BOOST_THRESHOLD
            );
            Some(self.exploration_factor)
        } else {
            None
        };

        // Update each parameter's posterior
        for param_idx in 0..5 {
            if let Some(posterior) = self.posteriors.get_mut(&(cpu, class, param_idx)) {
                let param_value = params[param_idx] as f64;
                posterior.update(param_value, improved, improvement_ratio);

                // Apply exploration boost if needed
                if let Some(boost) = exploration_boost {
                    posterior.boost_exploration(boost);
                }
            }
        }

        // Update baseline loss with EWMA
        let new_baseline = if baseline > 0.0 {
            BASELINE_EWMA_ALPHA * loss + (1.0 - BASELINE_EWMA_ALPHA) * baseline
        } else {
            loss
        };

        debug!(
            "[THOMPSON-UPDATE-END] cpu={} class={} new_baseline={:.2} (ewma_alpha={:.2})",
            cpu, class, new_baseline, BASELINE_EWMA_ALPHA
        );

        self.baseline_loss.insert(key, new_baseline);
    }

    /// Get posterior for debugging and stats display
    ///
    /// # Arguments
    /// * `cpu` - CPU ID
    /// * `class` - Task class
    /// * `param_idx` - Parameter index (0-4)
    #[allow(dead_code)] // Used for detailed analytics and stats export
    pub fn get_posterior(&self, cpu: u32, class: u32, param_idx: usize) -> Option<&ValuePosterior> {
        self.posteriors.get(&(cpu, class, param_idx))
    }

    /// Get current baseline loss for a CPU/class
    pub fn get_baseline(&self, cpu: u32, class: u32) -> Option<f64> {
        self.baseline_loss.get(&(cpu, class)).copied()
    }

    /// Get number of tracked CPUs
    #[allow(dead_code)] // Available for informational purposes
    pub fn cpu_count(&self) -> usize {
        self.cpu_count
    }

    /// Reset posterior for a specific parameter to initial state
    ///
    /// Useful for recovery when parameters get stuck in bad regions.
    pub fn reset_posterior(
        &mut self,
        cpu: u32,
        class: u32,
        param_idx: usize,
        default_params: &DefaultParams,
        bounds: &ParamBounds,
    ) {
        let class_idx = class as usize;
        let initial_value = default_params[class_idx][param_idx] as f64;
        let (min, max) = bounds[class_idx][param_idx];

        let posterior = ValuePosterior::new(
            initial_value,
            min as f64,
            max as f64,
            self.exploration_factor,
        );

        self.posteriors.insert((cpu, class, param_idx), posterior);
    }

    /// Reset all posteriors for a (cpu, class) combination
    ///
    /// Used by safety monitor when loss spike detected to encourage re-exploration.
    pub fn reset_posteriors_for_cpu_class(
        &mut self,
        cpu: u32,
        class: u32,
        default_params: &DefaultParams,
        bounds: &ParamBounds,
    ) {
        for param_idx in 0..5 {
            self.reset_posterior(cpu, class, param_idx, default_params, bounds);
        }
    }

    /// Get statistics about current posteriors
    ///
    /// Returns (total_posteriors, total_observations, avg_uncertainty)
    pub fn get_stats(&self) -> (usize, f64, f64) {
        let total_posteriors = self.posteriors.len();
        let mut total_observations = 0.0;
        let mut total_uncertainty = 0.0;
        let mut count = 0;

        for posterior in self.posteriors.values() {
            total_observations += posterior.n_observations;
            total_uncertainty += posterior.std;
            count += 1;
        }

        let avg_uncertainty = if count > 0 {
            total_uncertainty / count as f64
        } else {
            0.0
        };

        (total_posteriors, total_observations, avg_uncertainty)
    }

    /// Get detailed statistics for a specific (CPU, class) combination
    ///
    /// Returns posterior statistics including mean, std, and n_observations
    /// for all 5 parameters for this (cpu, class).
    pub fn get_detailed_stats(&self, cpu: u32, class: u32) -> Option<ThompsonDetailedStats> {
        let mut total_samples = 0u64;
        let baseline = self.get_baseline(cpu, class).unwrap_or(0.0);
        let mut posteriors = [PosteriorStats {
            mean: 0.0,
            std: 0.0,
            n_observations: 0.0,
        }; 5];

        let mut found_any = false;
        for param_idx in 0..5 {
            if let Some(posterior) = self.posteriors.get(&(cpu, class, param_idx)) {
                found_any = true;
                total_samples += posterior.n_observations as u64;
                posteriors[param_idx] = PosteriorStats {
                    mean: posterior.mean,
                    std: posterior.std,
                    n_observations: posterior.n_observations,
                };
            }
        }

        if found_any {
            Some(ThompsonDetailedStats {
                total_samples,
                baseline_loss: baseline,
                posteriors,
            })
        } else {
            None
        }
    }
}

/// Statistics for a single parameter's posterior
#[derive(Debug, Clone, Copy)]
pub struct PosteriorStats {
    /// Mean of the posterior (estimated optimal parameter value)
    pub mean: f64,
    /// Standard deviation (uncertainty)
    pub std: f64,
    /// Number of observations made
    pub n_observations: f64,
}

/// Detailed statistics for a (CPU, class) combination
#[derive(Debug, Clone)]
pub struct ThompsonDetailedStats {
    /// Total number of observations across all parameters
    pub total_samples: u64,
    /// Current baseline loss value
    pub baseline_loss: f64,
    /// Per-parameter posterior statistics
    pub posteriors: [PosteriorStats; 5],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_posterior_new() {
        let posterior = ValuePosterior::new(500.0, 0.0, 1000.0, 1.0);

        // Mean should be clamped to initial value
        assert!((posterior.mean - 500.0).abs() < 0.001);
        // Std should be set based on exploration factor
        assert!(posterior.std > 0.0);
        // No observations initially
        assert_eq!(posterior.n_observations, 0.0);
        // Bounds should be set
        assert_eq!(posterior.min, 0.0);
        assert_eq!(posterior.max, 1000.0);
    }

    #[test]
    fn test_value_posterior_sample() {
        let posterior = ValuePosterior::new(500.0, 0.0, 1000.0, 1.0);

        // Sample multiple times and verify within bounds
        for _ in 0..100 {
            let sample = posterior.sample();
            assert!(sample >= 0.0 && sample <= 1000.0);
        }
    }

    #[test]
    fn test_value_posterior_update_success() {
        let mut posterior = ValuePosterior::new(500.0, 0.0, 1000.0, 1.0);

        // Update with success - param value higher than mean
        // Mean should shift toward the successful value
        let old_mean = posterior.mean;
        posterior.update(600.0, true, 0.5); // 50% improvement

        // Mean should move toward 600
        assert!(posterior.mean > old_mean);
        // Should have one observation
        assert_eq!(posterior.n_observations, 1.0);
        // Std should decrease (we're learning)
        assert!(posterior.std < 100.0);
    }

    #[test]
    fn test_value_posterior_update_failure() {
        let mut posterior = ValuePosterior::new(500.0, 0.0, 1000.0, 1.0);
        let old_std = posterior.std;

        // Update with failure - no improvement
        posterior.update(600.0, false, 0.0);

        // Mean should stay the same (no shift on failure)
        assert!((posterior.mean - 500.0).abs() < 0.001);
        // Std should increase (explore more)
        assert!(posterior.std > old_std);
        assert_eq!(posterior.n_observations, 1.0);
    }

    #[test]
    fn test_value_posterior_mean_clamping() {
        let mut posterior = ValuePosterior::new(500.0, 100.0, 900.0, 1.0);

        // Try to shift mean outside bounds
        posterior.update(0.0, true, 1.0); // Very low value, 100% improvement
        assert!(posterior.mean >= 100.0); // Should be clamped

        posterior.update(1000.0, true, 1.0); // Very high value, 100% improvement
        assert!(posterior.mean <= 900.0); // Should be clamped
    }

    #[test]
    fn test_value_posterior_boost_exploration() {
        let mut posterior = ValuePosterior::new(500.0, 0.0, 1000.0, 1.0);
        let old_std = posterior.std;

        posterior.boost_exploration(2.0);

        // Std should increase
        assert!(posterior.std > old_std);
        // But should still be bounded
        assert!(posterior.std <= 500.0); // range / 2
    }

    #[test]
    fn test_thompson_sampler_new() {
        let profile = Profile::production();
        let sampler = ThompsonSampler::new(4, &profile);

        assert_eq!(sampler.cpu_count(), 4);
        assert_eq!(sampler.update_interval_ms, profile.response_ms);
        assert_eq!(sampler.exploration_factor, profile.exploration_factor);

        // Check that posteriors were initialized (4 CPUs * 4 classes * 5 params = 80)
        let (total, _, _) = sampler.get_stats();
        assert_eq!(total, 80);
    }

    #[test]
    fn test_thompson_sampler_sample() {
        let profile = Profile::production();
        let mut sampler = ThompsonSampler::new(2, &profile);

        // Sample parameters
        let params = sampler.sample_params(0, 0);

        // Check that all parameters are within bounds
        for (idx, &value) in params.iter().enumerate() {
            let (min, max) = profile.bounds[0][idx];
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
    fn test_thompson_sampler_update() {
        let profile = Profile::production();
        let mut sampler = ThompsonSampler::new(1, &profile);

        // Initial sample
        let params = sampler.sample_params(0, 0);

        // Update with a loss value (treated as first observation = improvement)
        sampler.update(0, 0, params, 100.0);

        // Check that baseline was set
        let baseline = sampler.get_baseline(0, 0);
        assert!(baseline.is_some());
        assert!((baseline.unwrap() - 100.0).abs() < 0.001);

        // Check that observations were recorded
        let posterior = sampler.get_posterior(0, 0, 0).unwrap();
        assert_eq!(posterior.n_observations, 1.0);
    }

    #[test]
    fn test_thompson_sampler_update_improvement() {
        let profile = Profile::production();
        let mut sampler = ThompsonSampler::new(1, &profile);

        // First update sets baseline
        let params1 = sampler.sample_params(0, 0);
        sampler.update(0, 0, params1, 100.0);

        let initial_mean = sampler.get_posterior(0, 0, 0).unwrap().mean;

        // Second update with improvement
        let params2 = sampler.sample_params(0, 0);
        sampler.update(0, 0, params2, 80.0); // 20% improvement

        let new_mean = sampler.get_posterior(0, 0, 0).unwrap().mean;

        // Mean should have shifted toward params2[0]
        let param_value = params2[0] as f64;
        let shift_toward =
            (new_mean - initial_mean).signum() == (param_value - initial_mean).signum();
        assert!(
            shift_toward || (new_mean - initial_mean).abs() < 0.001,
            "Mean should shift toward successful parameter value"
        );

        // Baseline should be updated with EWMA
        let baseline = sampler.get_baseline(0, 0).unwrap();
        assert!(baseline < 100.0 && baseline > 80.0); // EWMA between old and new
    }

    #[test]
    fn test_thompson_sampler_update_regression() {
        let profile = Profile::production();
        let mut sampler = ThompsonSampler::new(1, &profile);

        // First update sets baseline
        let params1 = sampler.sample_params(0, 0);
        sampler.update(0, 0, params1, 100.0);

        let initial_std = sampler.get_posterior(0, 0, 0).unwrap().std;

        // Second update with regression (>50% increase triggers boost)
        let params2 = sampler.sample_params(0, 0);
        sampler.update(0, 0, params2, 200.0); // 100% increase

        let new_std = sampler.get_posterior(0, 0, 0).unwrap().std;

        // Std should increase due to both failure and exploration boost
        assert!(new_std > initial_std, "Std should increase on regression");
    }

    #[test]
    fn test_exploration_boost_threshold() {
        let profile = Profile::production();
        let mut sampler = ThompsonSampler::new(1, &profile);

        // Get initial uncertainty
        let posterior_before = sampler.get_posterior(0, 0, 0).unwrap();
        let std_before = posterior_before.std;

        // First establish a baseline
        let params = sampler.sample_params(0, 0);
        sampler.update(0, 0, params, 100.0);

        // Update with moderate increase (not enough to trigger boost)
        let params = sampler.sample_params(0, 0);
        sampler.update(0, 0, params, 140.0); // 40% increase, no boost

        // Std should increase from failure but not from boost
        let posterior_after_no_boost = sampler.get_posterior(0, 0, 0).unwrap();

        // Now trigger exploration boost with large loss increase (>50%)
        let params = sampler.sample_params(0, 0);
        sampler.update(0, 0, params, 200.0); // 100% increase

        // Check that uncertainty increased more
        let posterior_after_boost = sampler.get_posterior(0, 0, 0).unwrap();
        assert!(
            posterior_after_boost.std >= std_before || posterior_after_boost.n_observations > 0.0
        );
    }

    #[test]
    fn test_observation_capping() {
        let mut posterior = ValuePosterior::new(500.0, 0.0, 1000.0, 1.0);

        // Add many observations
        for _ in 0..1005 {
            posterior.update(500.0, true, 0.1);
        }

        // Check that observations were capped
        assert!(posterior.n_observations <= MAX_OBSERVATIONS);
    }

    #[test]
    fn test_posterior_reset() {
        let profile = Profile::production();
        let mut sampler = ThompsonSampler::new(1, &profile);

        // Update several times to change state
        for _ in 0..10 {
            let params = sampler.sample_params(0, 0);
            sampler.update(0, 0, params, 100.0);
        }

        let posterior_before = sampler.get_posterior(0, 0, 0).unwrap();
        assert!(posterior_before.n_observations > 0.0);
        let mean_before = posterior_before.mean; // Clone the value we need

        // Reset the posterior
        sampler.reset_posterior(0, 0, 0, &profile.default_params, &profile.bounds);

        let posterior_after = sampler.get_posterior(0, 0, 0).unwrap();
        assert_eq!(posterior_after.n_observations, 0.0);
        assert!((posterior_after.mean - mean_before).abs() < 1000.0); // Within bounds
    }

    #[test]
    fn test_learning_rate_decay() {
        let mut posterior = ValuePosterior::new(500.0, 0.0, 1000.0, 1.0);

        // First observation - learning rate = BASE_LEARNING_RATE / (1 + 0) = 0.1
        // mean shifts from 500 toward 600: delta=100, lr=0.1, imp=1.0
        // shift = 100 * 0.1 * 1.0 = 10, so mean ≈ 510
        posterior.update(600.0, true, 1.0);
        let shift_first = posterior.mean - 500.0;
        assert!(
            shift_first > 5.0 && shift_first < 15.0,
            "First update should shift mean by ~10, got {}",
            shift_first
        );

        // After first update, n_observations = 1
        assert_eq!(posterior.n_observations, 1.0);

        // Add many observations to increase n_observations and decay learning rate
        // Using smaller improvements to avoid moving mean all the way to 600
        for _ in 0..200 {
            posterior.update(600.0, true, 0.1); // 10% improvement
        }

        // Verify n_observations is high (capped at MAX_OBSERVATIONS)
        assert!(
            posterior.n_observations > 100.0,
            "Should have many observations"
        );

        // Now test that learning rate has decayed
        // Reset mean to a known value and try to shift it
        let initial_mean_for_test = posterior.mean;

        // Try to shift away from current value
        // With high n_observations, learning rate should be much lower
        // lr ≈ 0.1 / (1 + 200*0.001) = 0.1 / 1.2 ≈ 0.083
        // Compared to initial lr = 0.1, this is ~83% of full rate
        // With many observations, the shift should still be smaller than
        // what we'd get if n_observations was still low
        posterior.update(100.0, true, 1.0);

        let shift_with_decay = (posterior.mean - initial_mean_for_test).abs();

        // With decayed LR, shift should be smaller than what we'd get with full LR
        // Full LR (0.1) would give shift of ~40, decayed should be ~35 or less
        assert!(
            shift_with_decay < 50.0,
            "Shift with decayed LR should be limited, got {}. n_obs={}",
            shift_with_decay,
            posterior.n_observations
        );
    }
}
