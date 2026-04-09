// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 OpenCode Assistant
//
// PID Tail-Latency-Target Based Controller for scx_autoland
//
// This controller uses control theory (PID) to adapt scheduling parameters
// based on measured P99 latencies per task class, with LAVD-inspired
// latency-criticality scoring for workload characterization.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::bpf_intf::{autoland_task_class, class_criticality_metrics, class_latency_stat};

/// Number of histogram buckets for P99 calculation (must match BPF)
const LATENCY_HISTOGRAM_BUCKETS: usize = 16;

/// PID controller coefficients per class (Kp, Ki, Kd)
/// - LC: Aggressive, fast response (high P, low I to avoid overshoot)
/// - Normal: Balanced response
/// - Hog: Slow, stable response (low P, higher I for steady state)
const PID_PARAMS: [(f64, f64, f64); 3] = [
    (1.0, 0.05, 0.3), // LC: aggressive
    (0.5, 0.1, 0.2),  // Normal: balanced
    (0.3, 0.15, 0.1), // Hog: slow and stable
];

/// Output limits as multiplicative factors (min, max)
const OUTPUT_LIMITS: [(f64, f64); 3] = [
    (0.5, 2.0), // LC: can vary more
    (0.6, 1.8), // Normal: moderate range
    (0.8, 1.5), // Hog: more stable
];

/// Default P99 latency targets per class (in microseconds)
const DEFAULT_LC_TARGET_US: u64 = 500; // 500μs for latency-critical
const DEFAULT_NORMAL_TARGET_US: u64 = 2000; // 2ms for normal
const DEFAULT_HOG_MAX_US: u64 = 50000; // 50ms max for hogs

/// Minimum and maximum acceptable latency multipliers
const MIN_ACCEPTABLE_MULTIPLIER: f64 = 0.2;
const MAX_ACCEPTABLE_MULTIPLIER: f64 = 2.0;

/// EWMA alpha for criticality score smoothing
const CRITICALITY_ALPHA: f32 = 0.3;

/// PID controller state for a single control loop
#[derive(Clone, Copy, Debug)]
pub struct PidController {
    /// Proportional gain
    kp: f64,
    /// Integral gain
    ki: f64,
    /// Derivative gain
    kd: f64,
    /// Accumulated integral error (for anti-windup)
    integral_error: f64,
    /// Previous error for derivative calculation
    prev_error: f64,
    /// Output limits
    min_output: f64,
    max_output: f64,
    /// Integral windup limit
    integral_limit: f64,
}

impl PidController {
    /// Create a new PID controller with specified gains and limits
    pub fn new(kp: f64, ki: f64, kd: f64, min_output: f64, max_output: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            integral_error: 0.0,
            prev_error: 0.0,
            min_output,
            max_output,
            integral_limit: (max_output - min_output) / ki.max(0.01), // Anti-windup
        }
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.integral_error = 0.0;
        self.prev_error = 0.0;
    }

    /// Compute PID output based on current error and time delta
    ///
    /// # Arguments
    /// * `error` - Normalized error (measured - target) / target
    /// * `dt_secs` - Time since last update in seconds
    ///
    /// # Returns
    /// Control output in range [min_output, max_output]
    pub fn compute(&mut self, error: f64, dt_secs: f64) -> f64 {
        let dt = dt_secs.max(0.001); // Minimum 1ms to prevent division issues

        // Proportional term
        let p = self.kp * error;

        // Integral term with anti-windup
        self.integral_error += error * dt;
        // Clamp integral to prevent windup
        self.integral_error = self
            .integral_error
            .clamp(-self.integral_limit, self.integral_limit);
        let i = self.ki * self.integral_error;

        // Derivative term (with filtering)
        let derivative = (error - self.prev_error) / dt;
        let d = self.kd * derivative;
        self.prev_error = error;

        // Combine terms
        let output = p + i + d;

        // Clamp to output limits
        output.clamp(self.min_output, self.max_output)
    }

    /// Get current gains for debugging
    pub fn get_gains(&self) -> (f64, f64, f64) {
        (self.kp, self.ki, self.kd)
    }
}

/// EMA Histogram for P99 calculation with natural decay
/// Replaces cumulative histogram with exponentially-weighted moving average
///
/// The EMA approach ensures old latency samples decay exponentially over time,
/// allowing the controller to rebound after load conditions change.
#[derive(Clone, Debug)]
pub struct EmaHistogram {
    /// EMA-smoothed bucket counts (not cumulative - represents "recent" activity)
    ema_buckets: [f64; LATENCY_HISTOGRAM_BUCKETS],
    /// Previous raw histogram from BPF for delta calculation
    prev_raw_histogram: [u64; LATENCY_HISTOGRAM_BUCKETS],
    /// EMA alpha (smoothing factor, higher = faster decay of old data)
    alpha: f64,
    /// Minimum samples required for valid P99
    min_samples: f64,
}

impl EmaHistogram {
    /// Create new EMA histogram with specified alpha
    ///
    /// # Arguments
    /// * `alpha` - Smoothing factor (0.0-1.0), higher = more responsive, faster decay
    ///
    /// Recommended values:
    /// - 0.1-0.2: Slow decay, more stable (good for consistent workloads)
    /// - 0.3-0.5: Fast decay, more responsive (good for bursty workloads)
    pub fn new(alpha: f64) -> Self {
        Self {
            ema_buckets: [0.0; LATENCY_HISTOGRAM_BUCKETS],
            prev_raw_histogram: [0; LATENCY_HISTOGRAM_BUCKETS],
            alpha: alpha.clamp(0.01, 0.99),
            min_samples: 10.0, // Need at least 10 samples for valid P99
        }
    }

    /// Update EMA histogram with new BPF data
    ///
    /// Uses differential update: ema = ema * (1-alpha) + delta * alpha
    /// This naturally "forgets" old data over time, solving the rebound problem.
    pub fn update(&mut self, new_raw_histogram: &[u64; LATENCY_HISTOGRAM_BUCKETS]) {
        for i in 0..LATENCY_HISTOGRAM_BUCKETS {
            // Calculate delta (new observations since last update)
            let delta = if new_raw_histogram[i] > self.prev_raw_histogram[i] {
                (new_raw_histogram[i] - self.prev_raw_histogram[i]) as f64
            } else {
                // BPF may have reset, or no new samples
                0.0
            };

            // EMA update: blend old EMA with new delta
            // Old data decays: ema_buckets[i] *= (1.0 - alpha)
            // New data added: delta * alpha
            self.ema_buckets[i] = self.ema_buckets[i] * (1.0 - self.alpha) + delta * self.alpha;
        }

        self.prev_raw_histogram = *new_raw_histogram;
    }

    /// Calculate P99 from EMA histogram
    ///
    /// # Returns
    /// * `Some(p99_ns)` - P99 latency in nanoseconds if enough samples
    /// * `None` - Not enough data for reliable P99
    pub fn calculate_p99(&self) -> Option<u64> {
        let total: f64 = self.ema_buckets.iter().sum();

        if total < self.min_samples {
            return None; // Not enough data
        }

        // P99 position (99th percentile)
        let p99_threshold = total * 0.99;
        let mut cumulative = 0.0;

        for (idx, &count) in self.ema_buckets.iter().enumerate() {
            cumulative += count;
            if cumulative >= p99_threshold {
                // Bucket 0: 0-1us, Bucket 1: 1-2us, Bucket 2: 2-4us, etc.
                // Return upper bound of bucket in nanoseconds
                let upper_bound_us = if idx == 0 { 1 } else { 1u64 << idx };
                return Some(upper_bound_us * 1000);
            }
        }

        // Default to max bucket
        Some((1u64 << (LATENCY_HISTOGRAM_BUCKETS - 1)) * 1000)
    }

    /// Get total EMA weight (for debugging)
    pub fn total_weight(&self) -> f64 {
        self.ema_buckets.iter().sum()
    }

    /// Reset histogram
    pub fn reset(&mut self) {
        self.ema_buckets = [0.0; LATENCY_HISTOGRAM_BUCKETS];
        self.prev_raw_histogram = [0; LATENCY_HISTOGRAM_BUCKETS];
    }

    /// Get the EMA alpha value
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

/// EMA-based PID controller
/// Uses exponential smoothing for integral term, providing natural decay
///
/// The key difference from standard PID:
/// - Standard PID: integral_error += error * dt (accumulates forever)
/// - EMA-PID: ema_error = ema_error * (1-alpha) + error * alpha (exponentially decays)
///
/// This means after load ends, old errors naturally fade away, allowing rebound.
#[derive(Clone, Copy, Debug)]
pub struct EmaPidController {
    /// Proportional gain
    kp: f64,
    /// Integral gain (determines EMA alpha for error smoothing)
    ki: f64,
    /// Derivative gain
    kd: f64,
    /// EMA-smoothed error (this replaces integral_error)
    /// Automatically decays old errors exponentially
    ema_error: f64,
    /// Previous error for derivative calculation
    prev_error: f64,
    /// Output limits
    min_output: f64,
    max_output: f64,
}

impl EmaPidController {
    /// Create a new EMA-PID controller
    ///
    /// # Arguments
    /// * `kp` - Proportional gain (immediate response)
    /// * `ki` - Integral gain (also determines EMA alpha: alpha = ki * dt)
    /// * `kd` - Derivative gain (dampens oscillations)
    /// * `min_output` - Minimum controller output
    /// * `max_output` - Maximum controller output
    ///
    /// # EMA Decay Behavior
    /// Higher ki = faster alpha = shorter memory of old errors
    /// After N control intervals, old error contribution = (1-alpha)^N
    pub fn new(kp: f64, ki: f64, kd: f64, min_output: f64, max_output: f64) -> Self {
        Self {
            kp,
            ki,
            kd,
            ema_error: 0.0,
            prev_error: 0.0,
            min_output,
            max_output,
        }
    }

    /// Reset controller state
    pub fn reset(&mut self) {
        self.ema_error = 0.0;
        self.prev_error = 0.0;
    }

    /// Compute EMA-PID output
    ///
    /// The integral term is replaced with an EMA of past errors:
    /// ema_error = ema_error * (1-alpha) + error * alpha
    ///
    /// This means:
    /// - After 1 time constant (1/alpha samples), old error contribution decays to 37%
    /// - After 3 time constants, old error decays to 5%
    /// - The controller "forgets" old load conditions naturally
    ///
    /// # Arguments
    /// * `error` - Normalized error (measured - target) / target
    /// * `dt_secs` - Time since last update in seconds
    ///
    /// # Returns
    /// Control output in range [min_output, max_output]
    pub fn compute(&mut self, error: f64, dt_secs: f64) -> f64 {
        let dt = dt_secs.max(0.001); // Minimum 1ms

        // Calculate EMA alpha based on ki and dt
        // Higher ki = faster response = shorter memory
        let alpha = (self.ki * dt).clamp(0.01, 0.95);

        // EMA update of error (this is the "integral" term)
        // Old errors exponentially decay: ema_error *= (1-alpha)
        // New error added: error * alpha
        self.ema_error = self.ema_error * (1.0 - alpha) + error * alpha;

        // Derivative term
        let derivative = (error - self.prev_error) / dt;
        self.prev_error = error;

        // Combine terms
        // P: immediate response to current error
        // I: EMA of past errors (naturally decays)
        // D: rate of change (dampens oscillations)
        let p = self.kp * error;
        let i = self.ki * self.ema_error; // Scaled EMA error
        let d = self.kd * derivative;

        let output = p + i + d;

        // Clamp to output limits
        output.clamp(self.min_output, self.max_output)
    }

    /// Get current EMA error value (for debugging)
    pub fn ema_error(&self) -> f64 {
        self.ema_error
    }

    /// Get current gains
    pub fn get_gains(&self) -> (f64, f64, f64) {
        (self.kp, self.ki, self.kd)
    }
}

/// Per-class latency target with acceptable bounds
#[derive(Clone, Copy, Debug)]
pub struct ClassLatencyTarget {
    /// Target P99 latency in nanoseconds
    pub target_p99_ns: u64,
    /// Upper bound of acceptable latency
    pub max_acceptable_ns: u64,
    /// Lower bound (prevents over-optimization)
    pub min_acceptable_ns: u64,
}

impl ClassLatencyTarget {
    /// Create a new latency target from microseconds
    pub fn from_micros(target_us: u64) -> Self {
        let target_ns = target_us * 1000;
        Self {
            target_p99_ns: target_ns,
            max_acceptable_ns: (target_ns as f64 * MAX_ACCEPTABLE_MULTIPLIER) as u64,
            min_acceptable_ns: (target_ns as f64 * MIN_ACCEPTABLE_MULTIPLIER) as u64,
        }
    }

    /// Create a bounded target (for hog class with max limit)
    pub fn bounded_max(max_us: u64) -> Self {
        let max_ns = max_us * 1000;
        Self {
            target_p99_ns: max_ns,
            max_acceptable_ns: max_ns,
            min_acceptable_ns: max_ns / 10,
        }
    }
}

impl Default for ClassLatencyTarget {
    fn default() -> Self {
        Self::from_micros(DEFAULT_LC_TARGET_US)
    }
}

/// LAVD-style latency criticality score [0, 1024]
#[derive(Clone, Copy, Debug)]
pub struct LatencyCriticalityScore {
    /// Current score [0, 1024]
    pub score: u32,
    /// EWMA-smoothed score for stability
    pub ewma_score: f32,
    /// Wakeup frequency (wakeups per 100ms window)
    pub wakeup_freq: f32,
    /// Runtime coefficient of variation (measure of burstiness)
    pub runtime_cv: f32,
    /// Last update timestamp
    pub last_update: Instant,
}

impl LatencyCriticalityScore {
    pub fn new() -> Self {
        Self {
            score: 0,
            ewma_score: 0.0,
            wakeup_freq: 0.0,
            runtime_cv: 0.0,
            last_update: Instant::now(),
        }
    }

    /// Update score based on wakeup frequency and runtime variance
    ///
    /// # Arguments
    /// * `wakeup_count` - Number of wakeups in measurement window
    /// * `runtime_ns` - Average runtime per scheduling cycle
    /// * `runtime_variance` - Variance of runtime samples
    pub fn update(&mut self, wakeup_count: u32, runtime_ns: u64, runtime_variance: f64) {
        // Wakeup frequency component (0.0 to 1.0)
        // High wakeup frequency indicates interactive behavior
        let wakeup_component = ((wakeup_count as f32) / 64.0).min(1.0);
        self.wakeup_freq = wakeup_count as f32;

        // Runtime variation component (0.0 to 1.0)
        // High CV indicates bursty behavior (latency-sensitive)
        let mean_runtime = runtime_ns as f64;
        let cv = if mean_runtime > 0.0 {
            (runtime_variance.sqrt() / mean_runtime).min(1.0)
        } else {
            0.0
        };
        self.runtime_cv = cv as f32;
        let variation_component = cv as f32;

        // Combined score with weighting (70% wakeup, 30% variation)
        let new_score = ((wakeup_component * 0.7 + variation_component * 0.3) * 1024.0) as u32;

        // EWMA update for smooth transitions
        self.ewma_score =
            CRITICALITY_ALPHA * new_score as f32 + (1.0 - CRITICALITY_ALPHA) * self.ewma_score;
        self.score = self.ewma_score as u32;
        self.last_update = Instant::now();
    }

    /// Get normalized score [0.0, 1.0] for weighting calculations
    pub fn normalized(&self) -> f64 {
        self.score as f64 / 1024.0
    }
}

impl Default for LatencyCriticalityScore {
    fn default() -> Self {
        Self::new()
    }
}

/// Task class enum matching BPF
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum TaskClass {
    Normal = 0,
    Hog = 1,
    LatencyCritical = 2,
}

impl TaskClass {
    pub fn from_u32(class: u32) -> Self {
        match class {
            1 => TaskClass::Hog,
            2 => TaskClass::LatencyCritical,
            _ => TaskClass::Normal,
        }
    }

    pub fn as_usize(self) -> usize {
        self as usize
    }
}

/// Per-class controller state with EMA-based tracking
#[derive(Clone, Debug)]
pub struct ClassControllerState {
    /// Latency target for this class
    pub target: ClassLatencyTarget,
    /// EMA-PID controller instance (replaces standard PID)
    pub pid: EmaPidController,
    /// EMA histogram for P99 calculation (replaces cumulative tracking)
    pub ema_histogram: EmaHistogram,
    /// Criticality score
    pub criticality: LatencyCriticalityScore,
    /// Measured P99 latency (last update)
    pub measured_p99_ns: Arc<AtomicU64>,
    /// Current slice size
    pub current_slice_ns: Arc<AtomicU64>,
    /// Base slice size (before adjustment)
    pub base_slice_ns: u64,
}

impl ClassControllerState {
    pub fn new(class: TaskClass, target: ClassLatencyTarget, base_slice_us: u64) -> Self {
        let idx = class.as_usize();
        let (kp, ki, kd) = PID_PARAMS[idx];
        let (min_out, max_out) = OUTPUT_LIMITS[idx];

        // EMA alpha for histogram: determines how fast old latency data decays
        // Higher = more responsive to changes, faster rebound after load ends
        let ema_alpha = 0.2; // After 10 intervals, old data is ~11% of original

        Self {
            target,
            pid: EmaPidController::new(kp, ki, kd, min_out, max_out),
            ema_histogram: EmaHistogram::new(ema_alpha),
            criticality: LatencyCriticalityScore::new(),
            measured_p99_ns: Arc::new(AtomicU64::new(0)),
            current_slice_ns: Arc::new(AtomicU64::new(base_slice_us * 1000)),
            base_slice_ns: base_slice_us * 1000,
        }
    }

    /// Update criticality score from BPF metrics
    pub fn update_criticality(&mut self, metrics: &class_criticality_metrics) {
        let sample_count = metrics.sample_count.max(1) as u64;
        let avg_runtime = metrics.total_runtime_ns / sample_count;

        // Calculate variance from sum of squares: Var = E[X^2] - E[X]^2
        let mean_sq = metrics.runtime_squared_sum / sample_count;
        let sq_mean = avg_runtime * avg_runtime;
        let variance = if mean_sq > sq_mean {
            (mean_sq - sq_mean) as f64
        } else {
            0.0
        };

        // Wakeup count scaled to 100ms window
        let wakeup_count = metrics.total_wakeup_count as u32;

        self.criticality.update(wakeup_count, avg_runtime, variance);
    }

    /// Update EMA histogram with new BPF data
    /// Call this before calculate_p99_ema()
    pub fn update_histogram(&mut self, stat: &class_latency_stat) {
        self.ema_histogram.update(&stat.histogram);
    }

    /// Calculate P99 from EMA histogram
    /// This uses exponentially-decayed data, so old samples naturally fade
    pub fn calculate_p99_ema(&self) -> Option<u64> {
        self.ema_histogram.calculate_p99()
    }

    /// Legacy P99 calculation from raw histogram (kept for compatibility)
    #[allow(dead_code)]
    pub fn calculate_p99(&self, stat: &class_latency_stat) -> u64 {
        let total_count: u64 = stat.histogram.iter().sum();

        if total_count == 0 {
            return 0;
        }

        // P99 position
        let p99_count = (total_count * 99) / 100;
        let mut cumulative = 0u64;

        // Find the bucket containing P99
        for (idx, &count) in stat.histogram.iter().enumerate() {
            cumulative += count;
            if cumulative >= p99_count {
                // Return upper bound of this bucket
                let upper_bound_us = if idx == 0 { 1 } else { 1u64 << idx };
                return upper_bound_us * 1000; // Convert to nanoseconds
            }
        }

        // Fallback: return max observed
        stat.max_ns
    }

    /// Compute control action and return new slice value
    pub fn compute_control(&mut self, measured_p99_ns: u64, dt_secs: f64) -> u64 {
        // Store measurement
        self.measured_p99_ns
            .store(measured_p99_ns, Ordering::Relaxed);

        if measured_p99_ns == 0 {
            // No data yet, maintain base slice
            return self.base_slice_ns;
        }

        // Calculate normalized error
        let target = self.target.target_p99_ns as f64;
        let measured = measured_p99_ns as f64;
        let error = (measured - target) / target;

        // Apply criticality weighting (more aggressive for high-criticality tasks)
        let criticality_weight = 1.0 + self.criticality.normalized() * 0.5;
        let weighted_error = error * criticality_weight;

        // Compute PID output (this is an adjustment factor)
        // Positive error (too slow) -> negative output (reduce slice)
        // Negative error (too fast) -> positive output (can increase slice)
        let output = self.pid.compute(weighted_error, dt_secs);

        // Convert output to slice adjustment
        // Output range is typically [0.5, 2.0] representing multiplicative factors
        let current_slice = self.current_slice_ns.load(Ordering::Relaxed);
        let base_slice = self.base_slice_ns;

        // Map PID output to slice multiplier
        // output < 1.0 -> reduce slice (latencies are too high)
        // output > 1.0 -> can increase slice (latencies are good)
        let multiplier = 2.0 - output; // Invert: high output means reduce slice

        let new_slice = (base_slice as f64 * multiplier) as u64;

        // Clamp to valid range
        let min_slice = (base_slice as f64 * OUTPUT_LIMITS[self.as_class_idx()].0) as u64;
        let max_slice = (base_slice as f64 * OUTPUT_LIMITS[self.as_class_idx()].1) as u64;
        let clamped = new_slice.clamp(min_slice, max_slice);

        self.current_slice_ns.store(clamped, Ordering::Relaxed);

        clamped
    }

    fn as_class_idx(&self) -> usize {
        // Determine class from base slice (this is a heuristic)
        // In practice, this should be passed explicitly
        if self.base_slice_ns <= 500_000 {
            0 // LC
        } else if self.base_slice_ns <= 2_000_000 {
            1 // Normal
        } else {
            2 // Hog
        }
    }

    /// Get current slice in microseconds (for BPF)
    pub fn get_slice_us(&self) -> u64 {
        self.current_slice_ns.load(Ordering::Relaxed) / 1000
    }
}

/// Main PID Tail-Latency-Target Based Controller
pub struct PidTailLatencyController {
    /// Per-class controller states
    class_states: [ClassControllerState; 3],
    /// Control interval (seconds between adjustments)
    control_interval_secs: f64,
    /// Last update timestamp
    last_update: Instant,
    /// Preemption threshold [0, 1024]
    preempt_threshold: Arc<AtomicU64>,
    /// Whether controller is enabled
    is_enabled: bool,
    /// Minimum slice in nanoseconds (absolute floor)
    min_slice_ns: u64,
    /// Maximum slice in nanoseconds (absolute ceiling)
    max_slice_ns: u64,
}

impl PidTailLatencyController {
    /// Create a new controller with default targets
    pub fn new(
        lc_base_slice_us: u64,
        normal_base_slice_us: u64,
        hog_base_slice_us: u64,
        enabled: bool,
    ) -> Self {
        let lc_target = ClassLatencyTarget::from_micros(DEFAULT_LC_TARGET_US);
        let normal_target = ClassLatencyTarget::from_micros(DEFAULT_NORMAL_TARGET_US);
        let hog_target = ClassLatencyTarget::bounded_max(DEFAULT_HOG_MAX_US);

        Self {
            class_states: [
                ClassControllerState::new(TaskClass::Normal, normal_target, normal_base_slice_us),
                ClassControllerState::new(TaskClass::Hog, hog_target, hog_base_slice_us),
                ClassControllerState::new(TaskClass::LatencyCritical, lc_target, lc_base_slice_us),
            ],
            control_interval_secs: 0.5, // 500ms default
            last_update: Instant::now(),
            preempt_threshold: Arc::new(AtomicU64::new(128)),
            is_enabled: enabled,
            min_slice_ns: 100_000,     // 100us minimum
            max_slice_ns: 100_000_000, // 100ms maximum
        }
    }

    /// Create with custom targets
    pub fn with_targets(
        mut self,
        lc_target_us: u64,
        normal_target_us: u64,
        hog_max_us: u64,
    ) -> Self {
        self.class_states[2].target = ClassLatencyTarget::from_micros(lc_target_us);
        self.class_states[0].target = ClassLatencyTarget::from_micros(normal_target_us);
        self.class_states[1].target = ClassLatencyTarget::bounded_max(hog_max_us);
        self
    }

    /// Create with custom PID gains for a specific class
    pub fn with_pid_gains(mut self, class: TaskClass, kp: f64, ki: f64, kd: f64) -> Self {
        let idx = class.as_usize();
        let (min_out, max_out) = OUTPUT_LIMITS[idx];
        self.class_states[idx].pid = EmaPidController::new(kp, ki, kd, min_out, max_out);
        self
    }

    /// Set control interval
    pub fn with_interval(mut self, interval_secs: f64) -> Self {
        self.control_interval_secs = interval_secs;
        self
    }

    /// Run one control iteration
    ///
    /// # Arguments
    /// * `latency_stats` - Array of 3 class_latency_stat from BPF
    /// * `criticality_stats` - Array of 3 class_criticality_metrics from BPF
    ///
    /// # Returns
    /// Tuple of (lc_slice_us, normal_slice_us, hog_slice_us, preempt_threshold)
    pub fn update(
        &mut self,
        latency_stats: &[class_latency_stat; 3],
        criticality_stats: &[class_criticality_metrics; 3],
    ) -> (u64, u64, u64, u32) {
        if !self.is_enabled {
            return self.get_current_values();
        }

        let now = Instant::now();
        let dt = now.duration_since(self.last_update).as_secs_f64();

        if dt < self.control_interval_secs {
            // Not enough time elapsed, return current values
            return self.get_current_values();
        }

        let mut results = [0u64; 3];

        // Process each class independently
        for idx in 0..3 {
            // Update criticality from BPF metrics
            self.class_states[idx].update_criticality(&criticality_stats[idx]);

            // Update EMA histogram with new BPF data
            // This naturally decays old samples, enabling rebound after load ends
            self.class_states[idx].update_histogram(&latency_stats[idx]);

            // Calculate P99 from EMA histogram (not cumulative)
            // Old samples have exponentially decayed, allowing natural rebound
            let p99_ns = match self.class_states[idx].calculate_p99_ema() {
                Some(p99) => p99,
                None => {
                    // Not enough data, maintain current slice
                    results[idx] = self.class_states[idx].get_slice_us();
                    continue;
                }
            };

            // Compute control action using EMA-PID
            // The EMA integral automatically decays when load ends
            let new_slice_ns = self.class_states[idx].compute_control(p99_ns, dt);

            // Apply absolute limits
            let clamped = new_slice_ns.clamp(self.min_slice_ns, self.max_slice_ns);
            self.class_states[idx]
                .current_slice_ns
                .store(clamped, Ordering::Relaxed);

            results[idx] = clamped / 1000; // Convert to microseconds
        }

        // Adjust preemption threshold based on LC latency
        let preempt = self.adjust_preemption_threshold(
            self.class_states[2].measured_p99_ns.load(Ordering::Relaxed),
            self.class_states[2].target.target_p99_ns,
        );
        self.preempt_threshold
            .store(preempt as u64, Ordering::Relaxed);

        self.last_update = now;

        // Map results: index 2 = LC, 0 = Normal, 1 = Hog
        (results[2], results[0], results[1], preempt)
    }

    /// Adjust preemption threshold based on LC latency vs target
    fn adjust_preemption_threshold(&self, lc_latency_ns: u64, lc_target_ns: u64) -> u32 {
        let current = self.preempt_threshold.load(Ordering::Relaxed) as u32;

        if lc_latency_ns == 0 {
            return current;
        }

        let latency_ratio = lc_latency_ns as f64 / lc_target_ns as f64;

        if latency_ratio > 2.0 {
            // LC is struggling: decrease threshold (easier preemption)
            current.saturating_sub(32).max(64)
        } else if latency_ratio > 1.5 {
            // LC is stressed: moderate decrease
            current.saturating_sub(16).max(64)
        } else if latency_ratio < 0.5 {
            // LC is doing great: increase threshold (reduce overhead)
            current.saturating_add(16).min(256)
        } else {
            // Within target: maintain current
            current
        }
    }

    fn get_current_values(&self) -> (u64, u64, u64, u32) {
        (
            self.class_states[2].get_slice_us(), // LC
            self.class_states[0].get_slice_us(), // Normal
            self.class_states[1].get_slice_us(), // Hog
            self.preempt_threshold.load(Ordering::Relaxed) as u32,
        )
    }

    /// Get current state for debugging/telemetry
    pub fn get_telemetry(&self) -> Vec<(TaskClass, u64, u64, u32, f64)> {
        vec![
            (
                TaskClass::Normal,
                self.class_states[0].target.target_p99_ns / 1000,
                self.class_states[0].measured_p99_ns.load(Ordering::Relaxed) / 1000,
                self.class_states[0].criticality.score,
                self.class_states[0].criticality.normalized(),
            ),
            (
                TaskClass::Hog,
                self.class_states[1].target.target_p99_ns / 1000,
                self.class_states[1].measured_p99_ns.load(Ordering::Relaxed) / 1000,
                self.class_states[1].criticality.score,
                self.class_states[1].criticality.normalized(),
            ),
            (
                TaskClass::LatencyCritical,
                self.class_states[2].target.target_p99_ns / 1000,
                self.class_states[2].measured_p99_ns.load(Ordering::Relaxed) / 1000,
                self.class_states[2].criticality.score,
                self.class_states[2].criticality.normalized(),
            ),
        ]
    }

    pub fn is_enabled(&self) -> bool {
        self.is_enabled
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.is_enabled = enabled;
    }

    /// Reset all PID controllers
    pub fn reset(&mut self) {
        for state in &mut self.class_states {
            state.pid.reset();
        }
        self.last_update = Instant::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_controller_basic() {
        let mut pid = PidController::new(1.0, 0.1, 0.2, 0.5, 2.0);

        // Reset to ensure clean state
        pid.reset();

        // Test with positive error (measured > target = too slow)
        // Error is positive, so we want output < 1.0 to reduce slice
        let output1 = pid.compute(0.5, 0.1); // 50% overshoot
                                             // With positive error, PID output should be > 1.0 (indicating need to reduce)
                                             // But remember we invert in the class controller: multiplier = 2.0 - output
                                             // So output1 > 1.0 means we need to reduce the slice
        assert!(
            output1 > 1.0,
            "Should produce high output for positive error (reduce slice)"
        );
        assert!(output1 <= 2.0, "Should be clamped to max");

        // Test with negative error (measured < target = faster than needed)
        // Error is negative, so we want output > 1.0 to increase slice
        let output2 = pid.compute(-0.3, 0.1); // 30% undershoot
                                              // With negative error, PID output should be < 1.0 (indicating can increase)
        assert!(
            output2 < 1.0,
            "Should produce low output for negative error (can increase slice)"
        );
        assert!(output2 >= 0.5, "Should be clamped to min");
    }

    #[test]
    fn test_pid_anti_windup() {
        let mut pid = PidController::new(0.5, 1.0, 0.1, 0.5, 1.5);

        // Apply large error multiple times to test anti-windup
        for _ in 0..100 {
            let _ = pid.compute(10.0, 0.1); // Huge error
        }

        // Should still be clamped to max_output despite large integral
        let output = pid.compute(10.0, 0.1);
        assert_eq!(output, 1.5, "Should be clamped to max_output");
    }

    #[test]
    fn test_class_latency_target() {
        let target = ClassLatencyTarget::from_micros(500);
        assert_eq!(target.target_p99_ns, 500_000);
        assert_eq!(target.min_acceptable_ns, 100_000); // 20% of 500us
        assert_eq!(target.max_acceptable_ns, 1_000_000); // 200% of 500us
    }

    #[test]
    fn test_criticality_score() {
        let mut score = LatencyCriticalityScore::new();

        // Update with high wakeup frequency (50 wakeups), low variation
        score.update(50, 10_000, 100.0); // 50 wakeups, 10us avg, low variance

        // Wakeup component: (50/64).min(1.0) = 0.78
        // Variation component: low
        // Combined with 70% weighting: 0.78 * 0.7 = 0.546
        // Score: 0.546 * 1024 = 559
        // With EWMA: 0.3 * 559 + 0.7 * 0 = 167

        // Score should be updated (EWMA smooths initial values)
        assert!(
            score.score > 0,
            "Score should be updated after first sample"
        );
        assert!(score.ewma_score > 0.0, "EWMA should be populated");
        assert!(
            score.wakeup_freq == 50.0,
            "Wakeup frequency should be recorded"
        );

        // Update again to build up EWMA
        score.update(50, 10_000, 100.0);
        // After second update, EWMA will be higher
        assert!(
            score.score > 100,
            "Score should increase with repeated high-wakeup samples"
        );
    }

    #[test]
    fn test_p99_calculation() {
        let state = ClassControllerState::new(
            TaskClass::Normal,
            ClassLatencyTarget::from_micros(2000),
            2000,
        );

        // Create histogram with known distribution
        let mut stat = class_latency_stat {
            sum_ns: 0,
            count: 0,
            sum_squares_ns: 0,
            min_ns: 0,
            max_ns: 0,
            histogram: [0; 16],
        };

        // Fill histogram: 100 samples at bucket 5 (16-32us)
        stat.histogram[5] = 100;

        let p99 = state.calculate_p99(&stat);
        assert!(p99 > 0, "P99 should be calculated");
    }

    #[test]
    fn test_controller_disabled() {
        let mut controller = PidTailLatencyController::new(500, 2000, 4000, false);

        let stats = [class_latency_stat {
            sum_ns: 0,
            count: 0,
            sum_squares_ns: 0,
            min_ns: 0,
            max_ns: 0,
            histogram: [0; 16],
        }; 3];

        let crit = [class_criticality_metrics {
            total_wakeup_count: 0,
            total_runtime_ns: 0,
            runtime_squared_sum: 0,
            sample_count: 0,
        }; 3];

        let (lc, normal, hog, _) = controller.update(&stats, &crit);

        // When disabled, should return base values
        assert_eq!(lc, 500);
        assert_eq!(normal, 2000);
        assert_eq!(hog, 4000);
    }

    #[test]
    fn test_preemption_adjustment() {
        let controller = PidTailLatencyController::new(500, 2000, 4000, true);

        // Test when LC is struggling (> 2x target)
        let preempt1 = controller.adjust_preemption_threshold(1_000_000, 500_000);
        assert!(
            preempt1 < 128,
            "Should decrease threshold when LC struggles"
        );

        // Test when LC is doing great (< 0.5x target)
        let preempt2 = controller.adjust_preemption_threshold(200_000, 500_000);
        assert!(preempt2 > 128, "Should increase threshold when LC is good");

        // Test within target
        let preempt3 = controller.adjust_preemption_threshold(500_000, 500_000);
        assert_eq!(preempt3, 128, "Should maintain when on target");
    }
}
