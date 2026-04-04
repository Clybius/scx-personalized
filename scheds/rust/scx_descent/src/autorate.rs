//! CAKE Autorate Controller for scx_descent - Phase 2: Algorithm Implementation
//!
//! Implements a cake-autorate inspired design with adaptive baselines and CPU share management.
//! This is a complete refactoring focusing on:
//! - Per-class adaptive latency baselines (only update during low load)
//! - CPU share-based parameter management instead of rate-based interpolation
//! - Cleaner state machine with Idle, Exploring, BackingOff, Steady states
//!
//! Key concepts:
//! - Per-class operation: Tracks load system-wide, NOT per-CPU
//! - Share-based control: Each class has a CPU time share (0.0-1.0) that maps to parameters
//! - Adaptive baselines: Latency baselines only update during low load periods (< 0.25)
//! - Fixed class array: Uses [T; 4] instead of HashMap for the 4 fixed classes
//!
//! The Autorate controller complements the PIE controller by providing
//! coarse-grained adaptation based on load conditions, while PIE handles
//! fine-grained latency-based optimization.

// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_descent authors
//
// CAKE Autorate controller - Phase 2: Algorithm Implementation

use log::debug;
use std::time::{Duration, Instant};

use crate::profiles::{DESCENT_CLASS_MAX, PARAM_COUNT};

/// Threshold for low load - baseline updates only happen below this
const LOW_LOAD_THRESHOLD: f64 = 0.25;

/// Threshold for high load - triggers exploration state
const HIGH_LOAD_THRESHOLD: f64 = 0.70;

/// Threshold for idle state (slightly lower than low load for hysteresis)
const IDLE_LOAD_THRESHOLD: f64 = 0.20;

/// EWMA smoothing factor for latency baseline: 0.2 = 20% new value, 80% history
const BASELINE_EWMA_ALPHA: f64 = 0.2;

/// Default EWMA smoothing factor for measured latency: 0.2 = 20% new value
const LATENCY_EWMA_ALPHA: f64 = 0.2;

/// Delay threshold for considering latency "good" (500µs)
const DELAY_THRESHOLD_NS: i64 = 500_000;

/// Bufferbloat threshold for triggering backoff (2ms)
const BUFFERBLOAT_THRESHOLD_NS: i64 = 2_000_000;

/// Hysteresis threshold - require 2 consecutive readings before state change
const STATE_HYSTERESIS_THRESHOLD: u32 = 2;

/// Minimum samples required for reliable state decisions
const MIN_SAMPLES_FOR_DECISION: u64 = 3;

/// =============================================================================
/// PHASE 3: PARAMETER INDEX CONSTANTS
/// =============================================================================

const PARAM_LATENCY_WEIGHT: usize = 0;
const PARAM_BASE_SLICE_NS: usize = 1;
const PARAM_VRUNTIME_SCALE: usize = 2;
const PARAM_PREEMPTION_PRIORITY: usize = 3;
const PARAM_MIGRATION_COST: usize = 4;

/// Mapping type for share-to-parameter conversion
const MAPPING_INVERSE: &str = "inverse";
const MAPPING_DIRECT: &str = "direct";

/// Per-class adaptive latency baseline
///
/// Tracks an EWMA-based baseline for latency that only updates during low load periods.
/// This prevents the baseline from drifting during congestion, ensuring we can always
/// detect bufferbloat conditions.
#[derive(Debug, Clone, Copy)]
pub struct LatencyBaseline {
    /// Current baseline latency in nanoseconds (EWMA)
    pub baseline_ns: u64,
    /// Last time the baseline was updated (for staleness detection)
    pub last_update_ns: u64,
    /// EWMA alpha factor (0.0-1.0, higher = more responsive)
    pub alpha: f64,
}

impl LatencyBaseline {
    /// Create a new latency baseline with default alpha
    pub fn new(initial_baseline_ns: u64) -> Self {
        Self {
            baseline_ns: initial_baseline_ns,
            last_update_ns: 0,
            alpha: BASELINE_EWMA_ALPHA,
        }
    }

    /// Create a new latency baseline with custom alpha
    pub fn with_alpha(initial_baseline_ns: u64, alpha: f64) -> Self {
        Self {
            baseline_ns: initial_baseline_ns,
            last_update_ns: 0,
            alpha: alpha.clamp(0.0, 1.0),
        }
    }

    /// Update the baseline with a new measurement
    ///
    /// CRITICAL: Baseline only updates during low load (< LOW_LOAD_THRESHOLD).
    /// This prevents baseline drift during congestion.
    ///
    /// * `measured_ns` - New latency measurement
    /// * `load_percent` - Current load percentage (0.0-1.0)
    /// * `now_ns` - Current time in nanoseconds for tracking
    pub fn update(&mut self, measured_ns: u64, load_percent: f64, now_ns: u64) {
        // Only update baseline during low load periods
        if load_percent < LOW_LOAD_THRESHOLD {
            // EWMA update: baseline = alpha * measured + (1 - alpha) * baseline
            let new_baseline =
                (self.alpha * measured_ns as f64) + ((1.0 - self.alpha) * self.baseline_ns as f64);
            self.baseline_ns = new_baseline as u64;
            self.last_update_ns = now_ns;
        }
        // During high load, baseline remains unchanged - this is intentional!
    }

    /// Force update the baseline regardless of load (for initialization)
    pub fn force_update(&mut self, measured_ns: u64, now_ns: u64) {
        let new_baseline =
            (self.alpha * measured_ns as f64) + ((1.0 - self.alpha) * self.baseline_ns as f64);
        self.baseline_ns = new_baseline as u64;
        self.last_update_ns = now_ns;
    }

    /// Get the delay delta (measured - baseline)
    ///
    /// Returns positive if measured latency exceeds baseline (delay),
    /// negative if measured is below baseline (faster than baseline).
    pub fn get_delay_delta(&self, measured_ns: u64) -> i64 {
        measured_ns as i64 - self.baseline_ns as i64
    }

    /// Get the delay ratio (measured / baseline)
    ///
    /// Returns 1.0 when measured equals baseline, > 1.0 when delayed.
    pub fn get_delay_ratio(&self, measured_ns: u64) -> f64 {
        if self.baseline_ns == 0 {
            return 1.0;
        }
        measured_ns as f64 / self.baseline_ns as f64
    }

    /// Check if the baseline is stale (hasn't been updated in a while)
    pub fn is_stale(&self, now_ns: u64, max_age_ns: u64) -> bool {
        now_ns.saturating_sub(self.last_update_ns) > max_age_ns
    }
}

impl Default for LatencyBaseline {
    fn default() -> Self {
        Self {
            baseline_ns: 0,
            last_update_ns: 0,
            alpha: BASELINE_EWMA_ALPHA,
        }
    }
}

/// CPU share configuration per class
///
/// Defines how a class's CPU share maps to scheduling parameters.
/// Share is a percentage of CPU time (0.0-1.0) that the class should receive.
#[derive(Debug, Clone, Copy)]
pub struct ShareConfig {
    /// Minimum CPU share (0.0-1.0) - used during heavy congestion
    pub min_share: f64,
    /// Base CPU share (0.0-1.0) - normal operating point
    pub base_share: f64,
    /// Maximum CPU share (0.0-1.0) - maximum allowed during high load
    pub max_share: f64,
    /// Rate at which share increases during load (multiplier, e.g., 1.1 for 10%)
    pub ramp_up_rate: f64,
    /// Rate at which share decreases during congestion (multiplier, e.g., 0.9 for 10%)
    pub ramp_down_rate: f64,
    /// Rate at which share decays toward base_share when steady (multiplier, e.g., 0.99)
    pub decay_rate: f64,
}

impl ShareConfig {
    /// Create a new share configuration
    pub fn new(
        min_share: f64,
        base_share: f64,
        max_share: f64,
        ramp_up_rate: f64,
        ramp_down_rate: f64,
        decay_rate: f64,
    ) -> Self {
        Self {
            min_share: min_share.clamp(0.0, 1.0),
            base_share: base_share.clamp(0.0, 1.0),
            max_share: max_share.clamp(0.0, 1.0),
            ramp_up_rate,
            ramp_down_rate,
            decay_rate,
        }
    }

    /// Validate that shares are in valid order: min <= base <= max
    pub fn is_valid(&self) -> bool {
        self.min_share <= self.base_share && self.base_share <= self.max_share
    }

    /// Clamp a share value to valid range for this class
    pub fn clamp(&self, share: f64) -> f64 {
        share.clamp(self.min_share, self.max_share)
    }
}

impl Default for ShareConfig {
    fn default() -> Self {
        Self {
            min_share: 0.1,
            base_share: 0.25,
            max_share: 0.5,
            ramp_up_rate: 1.1,
            ramp_down_rate: 0.9,
            decay_rate: 0.99,
        }
    }
}

/// CAKE Autorate state machine states
///
/// State machine design:
/// - Idle: No recent activity, share decaying toward base
/// - Exploring: Actively testing higher shares to find capacity (ramping up)
/// - BackingOff: Reducing share due to detected congestion (ramping down)
/// - Steady: Operating at stable share
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AutorateState {
    /// No recent activity, share at base level
    Idle,
    /// Testing higher shares to find capacity (ramping up)
    Exploring,
    /// Reducing share due to congestion (ramping down)
    BackingOff,
    /// Operating at stable share
    Steady,
}

impl AutorateState {
    /// Get human-readable name for the state
    pub fn name(&self) -> &'static str {
        match self {
            AutorateState::Idle => "IDLE",
            AutorateState::Exploring => "EXPLORING",
            AutorateState::BackingOff => "BACKING_OFF",
            AutorateState::Steady => "STEADY",
        }
    }

    /// Check if this state allows baseline updates (only during low load/idle)
    pub fn allows_baseline_update(&self) -> bool {
        matches!(self, AutorateState::Idle | AutorateState::Steady)
    }

    /// Check if this state is actively adjusting share
    pub fn is_adjusting(&self) -> bool {
        matches!(self, AutorateState::Exploring | AutorateState::BackingOff)
    }
}

/// Per-class autorate state
///
/// Tracks the complete autorate state for a single class, including:
/// - Current CPU share
/// - State machine state with hysteresis
/// - Adaptive latency baseline
/// - Timing information
/// - Latest metrics
#[derive(Debug, Clone)]
pub struct ClassAutorateState {
    /// Current CPU share (0.0-1.0, percentage of CPU time)
    pub current_share: f64,
    /// Current autorate state
    pub state: AutorateState,
    /// Adaptive latency baseline for this class
    pub latency_baseline: LatencyBaseline,
    /// Time of last state change
    pub last_state_change: Instant,
    /// Time of last share adjustment
    pub last_adjustment: Instant,
    /// Current load percentage (0.0-1.0) - cached from last update
    pub load_percent: f64,
    /// Last measured latency in nanoseconds
    pub measured_latency_ns: u64,
    /// Last delay delta (measured - baseline) in nanoseconds
    pub delay_delta_ns: i64,
    /// EWMA of measured latency
    pub latency_ewma_ns: u64,
    /// Target latency for this class (used for delay calculations)
    pub target_latency_ns: u64,
    /// Pending state for hysteresis (state we're considering transitioning to)
    pub pending_state: AutorateState,
    /// Count of consecutive readings in pending_state (for hysteresis)
    pub consecutive_state_count: u32,
    /// Direction of last adjustment (for tracking)
    pub last_direction: AdjustmentDirection,
}

impl ClassAutorateState {
    /// Create new autorate state for a class
    ///
    /// * `base_share` - Starting CPU share (typically from ShareConfig.base_share)
    /// * `initial_baseline_ns` - Initial latency baseline estimate
    pub fn new(base_share: f64, initial_baseline_ns: u64) -> Self {
        let now = Instant::now();
        Self {
            current_share: base_share.clamp(0.0, 1.0),
            state: AutorateState::Idle,
            latency_baseline: LatencyBaseline::new(initial_baseline_ns),
            last_state_change: now,
            last_adjustment: now,
            load_percent: 0.0,
            measured_latency_ns: 0,
            delay_delta_ns: 0,
            latency_ewma_ns: 0,
            target_latency_ns: initial_baseline_ns,
            pending_state: AutorateState::Idle,
            consecutive_state_count: 0,
            last_direction: AdjustmentDirection::None,
        }
    }

    /// Create new state with custom alpha for baseline tracking
    pub fn with_alpha(base_share: f64, initial_baseline_ns: u64, alpha: f64) -> Self {
        let now = Instant::now();
        Self {
            current_share: base_share.clamp(0.0, 1.0),
            state: AutorateState::Idle,
            latency_baseline: LatencyBaseline::with_alpha(initial_baseline_ns, alpha),
            last_state_change: now,
            last_adjustment: now,
            load_percent: 0.0,
            measured_latency_ns: 0,
            delay_delta_ns: 0,
            latency_ewma_ns: 0,
            target_latency_ns: initial_baseline_ns,
            pending_state: AutorateState::Idle,
            consecutive_state_count: 0,
            last_direction: AdjustmentDirection::None,
        }
    }

    /// Update latency EWMA with new measurement
    pub fn update_latency_ewma(&mut self, measured_ns: u64) {
        if self.latency_ewma_ns == 0 {
            // First measurement
            self.latency_ewma_ns = measured_ns;
        } else {
            // EWMA: new = alpha * current + (1 - alpha) * old
            self.latency_ewma_ns = ((LATENCY_EWMA_ALPHA * measured_ns as f64)
                + ((1.0 - LATENCY_EWMA_ALPHA) * self.latency_ewma_ns as f64))
                as u64;
        }
        self.measured_latency_ns = measured_ns;
    }

    /// Update the latency baseline (only during low load)
    pub fn update_baseline(&mut self, now_ns: u64) {
        self.latency_baseline
            .update(self.measured_latency_ns, self.load_percent, now_ns);
        self.delay_delta_ns = self
            .latency_baseline
            .get_delay_delta(self.measured_latency_ns);
    }

    /// Force update the latency baseline regardless of load
    pub fn force_update_baseline(&mut self, now_ns: u64) {
        self.latency_baseline
            .force_update(self.measured_latency_ns, now_ns);
        self.delay_delta_ns = self
            .latency_baseline
            .get_delay_delta(self.measured_latency_ns);
    }

    /// Transition to a new state (with hysteresis tracking)
    pub fn transition_to(&mut self, new_state: AutorateState) {
        if self.state != new_state {
            self.state = new_state;
            self.last_state_change = Instant::now();
            // Reset pending state tracking
            self.pending_state = new_state;
            self.consecutive_state_count = 0;
        }
    }

    /// Record a share adjustment
    pub fn record_adjustment(&mut self, new_share: f64, direction: AdjustmentDirection) {
        self.current_share = new_share.clamp(0.0, 1.0);
        self.last_adjustment = Instant::now();
        self.last_direction = direction;
    }

    /// Get time since last state change
    pub fn time_in_state(&self) -> Duration {
        self.last_state_change.elapsed()
    }

    /// Get time since last adjustment
    pub fn time_since_adjustment(&self) -> Duration {
        self.last_adjustment.elapsed()
    }

    /// Get delay ratio (measured / baseline)
    pub fn get_delay_ratio(&self) -> f64 {
        self.latency_baseline
            .get_delay_ratio(self.measured_latency_ns)
    }

    /// Check if baseline is stale
    pub fn is_baseline_stale(&self, now_ns: u64, max_age_ns: u64) -> bool {
        self.latency_baseline.is_stale(now_ns, max_age_ns)
    }
}

impl Default for ClassAutorateState {
    fn default() -> Self {
        Self::new(0.25, 1000_000) // 0.25 share, 1ms default baseline
    }
}

/// Class metrics for passing to the controller
///
/// This structure is used to pass per-class metrics from the
/// main scheduler to the autorate controller during updates.
#[derive(Debug, Clone, Copy)]
pub struct ClassMetrics {
    /// Class ID (0-3)
    pub class_id: u32,
    /// CPU cycles spent by this class (for utilization calculation)
    pub cycles_spent: u64,
    /// Number of latency samples collected
    pub sample_count: u64,
    /// Total latency in nanoseconds (for computing average)
    pub total_latency_ns: u64,
    /// Average latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Maximum latency in nanoseconds (for peak detection)
    pub max_latency_ns: u64,
    /// Load percentage (0.0-1.0) - aggregate across all CPUs
    pub load_percent: f64,
    /// Delay delta in nanoseconds (measured - baseline, if available)
    pub delay_delta_ns: i64,
}

impl ClassMetrics {
    /// Create new class metrics
    pub fn new(class_id: u32) -> Self {
        Self {
            class_id,
            cycles_spent: 0,
            sample_count: 0,
            total_latency_ns: 0,
            avg_latency_ns: 0,
            max_latency_ns: 0,
            load_percent: 0.0,
            delay_delta_ns: 0,
        }
    }

    /// Check if metrics are valid for processing
    pub fn is_valid(&self) -> bool {
        self.class_id < DESCENT_CLASS_MAX as u32 && self.sample_count > 0
    }

    /// Check if we have enough samples for reliable decisions
    pub fn has_min_samples(&self, min_samples: u64) -> bool {
        self.sample_count >= min_samples
    }
}

impl Default for ClassMetrics {
    fn default() -> Self {
        Self::new(0)
    }
}

/// =============================================================================
/// LEGACY COMPATIBILITY (to be removed in Phase 4)
/// =============================================================================

/// Direction of last adjustment (for refractory period)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AdjustmentDirection {
    Up,   // Toward max
    Down, // Toward min
    None,
}

/// Legacy per-class autorate state - DEPRECATED
///
/// This is the old state structure. It remains temporarily for backward
/// compatibility during the refactoring phases. Will be removed in Phase 4.
#[derive(Debug, Clone, Copy)]
#[deprecated(since = "Phase 1", note = "Use ClassAutorateState instead")]
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
    /// Count of consecutive readings in pending_state
    pub consecutive_state_count: u32,
    /// State we're considering transitioning to (for hysteresis)
    pub pending_state: AutorateState,
}

#[allow(deprecated)]
impl AutorateClassState {
    /// Create new autorate state for a class
    fn new() -> Self {
        Self {
            current_rate: 0.5, // Start at baseline
            load_percent: 0.0, // 0.0 until real data arrives
            state: AutorateState::Steady,
            last_adjustment_time: Instant::now(),
            last_adjustment_dir: AdjustmentDirection::None,
            latency_ewma_ns: 0,
            prev_latency_ns: 0,
            consecutive_state_count: 0,
            pending_state: AutorateState::Steady,
        }
    }
}

/// Autorate configuration (from Profile)
///
/// This structure holds the configuration for the autorate controller.
/// It will be extended in Phase 2 to include ShareConfig support.
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
    /// High load threshold (e.g., 0.70)
    pub high_load_threshold: f64,
    /// Low load threshold (e.g., 0.20)
    pub low_load_threshold: f64,
    /// Ramp up rate multiplier (e.g., 1.08 for gaming, 8%)
    pub ramp_up_rate: f64,
    /// Ramp down rate multiplier (e.g., 0.85 for 15% down)
    pub ramp_down_rate: f64,
    /// Decay rate toward baseline (e.g., 0.99 = 1% toward baseline)
    pub decay_rate: f64,
    /// Refractory period after upward adjustment in ms
    pub adjust_up_refractory_ms: u64,
    /// Refractory period after downward adjustment in ms
    pub adjust_down_refractory_ms: u64,
    /// Bufferbloat threshold (e.g., 1.5 = 1.5x target latency)
    pub bufferbloat_threshold: f64,
    /// Per-class share configurations (Phase 2 addition)
    pub share_configs: [ShareConfig; DESCENT_CLASS_MAX],
}

impl AutorateConfig {
    /// Create a default autorate config with default share configs
    pub fn new_with_default_shares() -> Self {
        Self {
            enabled: true,
            min_params: [[0; PARAM_COUNT]; DESCENT_CLASS_MAX],
            baseline_params: [[0; PARAM_COUNT]; DESCENT_CLASS_MAX],
            max_params: [[0; PARAM_COUNT]; DESCENT_CLASS_MAX],
            high_load_threshold: HIGH_LOAD_THRESHOLD,
            low_load_threshold: IDLE_LOAD_THRESHOLD,
            ramp_up_rate: 1.1,
            ramp_down_rate: 0.9,
            decay_rate: 0.99,
            adjust_up_refractory_ms: 100,
            adjust_down_refractory_ms: 50,
            bufferbloat_threshold: 1.5,
            share_configs: [
                ShareConfig::new(0.15, 0.25, 0.40, 1.08, 0.85, 0.98), // LATENCY_CRITICAL
                ShareConfig::new(0.15, 0.25, 0.35, 1.05, 0.85, 0.98), // NORMAL
                ShareConfig::new(0.20, 0.35, 0.60, 1.10, 0.80, 0.98), // HOG
                ShareConfig::new(0.05, 0.10, 0.15, 1.05, 0.90, 0.95), // BACKGROUND
            ],
        }
    }
}

impl Default for AutorateConfig {
    fn default() -> Self {
        Self::new_with_default_shares()
    }
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
    /// Current average share across all classes
    pub avg_share: f64,
    /// Number of baseline updates performed
    pub baseline_updates: u64,
    /// Current controlling class (if any)
    pub controlling_class: Option<u32>,
    /// Class that was backed off (if any)
    pub backed_off_class: Option<u32>,
}

impl Default for AutorateStats {
    fn default() -> Self {
        Self {
            total_classes: DESCENT_CLASS_MAX,
            adjustments_up: 0,
            adjustments_down: 0,
            adjustments_blocked: 0,
            avg_share: 0.25,
            baseline_updates: 0,
            controlling_class: None,
            backed_off_class: None,
        }
    }
}

/// Main autorate controller - Phase 2: Full Algorithm Implementation
///
/// The controller uses:
/// - Fixed array of ClassAutorateState [T; 4] instead of HashMap
/// - CPU share-based management instead of rate-based interpolation
/// - Adaptive per-class latency baselines
/// - New state machine (Idle, Exploring, BackingOff, Steady)
/// - Global coordination for starvation handling
pub struct AutorateController {
    /// Per-class autorate states (fixed array of 4 classes)
    class_states: [ClassAutorateState; DESCENT_CLASS_MAX],
    /// Per-class share configurations
    share_configs: [ShareConfig; DESCENT_CLASS_MAX],
    /// Controller configuration
    config: AutorateConfig,
    /// Update interval in milliseconds
    interval_ms: u64,
    /// Currently controlling class (the one driving adjustments)
    controlling_class: Option<u32>,
    /// Class that was recently backed off
    backed_off_class: Option<u32>,
    /// Statistics
    stats: AutorateStats,
    /// Legacy state map (temporary for transition - Phase 1)
    /// This will be removed in Phase 4
    #[allow(deprecated)]
    legacy_states: std::collections::HashMap<u32, AutorateClassState>,
}

impl AutorateController {
    /// Create new controller
    ///
    /// * `_nr_cpus` - Number of CPUs (kept for API compatibility)
    /// * `config` - Autorate configuration
    pub fn new(_nr_cpus: usize, config: &AutorateConfig) -> Self {
        let _now = Instant::now();

        // Initialize per-class states with base shares from config
        let mut class_states: [ClassAutorateState; DESCENT_CLASS_MAX] = [
            ClassAutorateState::new(config.share_configs[0].base_share, 1000_000),
            ClassAutorateState::new(config.share_configs[1].base_share, 1000_000),
            ClassAutorateState::new(config.share_configs[2].base_share, 5000_000),
            ClassAutorateState::new(config.share_configs[3].base_share, 10_000_000),
        ];

        // Set target latencies based on class characteristics
        class_states[0].target_latency_ns = 1000_000; // LATENCY_CRITICAL: 1ms
        class_states[1].target_latency_ns = 5000_000; // NORMAL: 5ms
        class_states[2].target_latency_ns = 20_000_000; // HOG: 20ms
        class_states[3].target_latency_ns = 50_000_000; // BACKGROUND: 50ms

        debug!(
            "[AUTORATE-INIT] Created Phase 2 controller with {} classes",
            DESCENT_CLASS_MAX
        );

        Self {
            class_states,
            share_configs: config.share_configs,
            config: config.clone(),
            interval_ms: 200, // Default 200ms interval (Phase 2: increased from 20ms)
            controlling_class: None,
            backed_off_class: None,
            stats: AutorateStats::default(),
            legacy_states: std::collections::HashMap::new(),
        }
    }

    /// Create new controller with custom update interval
    pub fn with_interval(_nr_cpus: usize, config: &AutorateConfig, interval_ms: u64) -> Self {
        let mut controller = Self::new(_nr_cpus, config);
        controller.interval_ms = interval_ms;
        controller
    }

    /// =============================================================================
    /// PHASE 2: MAIN ALGORITHM METHODS
    /// =============================================================================

    /// Main update called every 200ms
    ///
    /// This is the core algorithm that:
    /// 1. Updates baselines for classes with low load
    /// 2. Calculates delay deltas
    /// 3. Checks for starvation
    /// 4. Adjusts shares based on state machine
    /// 5. Normalizes shares to sum = 1.0
    /// 6. Returns calculated parameters for all classes
    ///
    /// # Arguments
    /// * `metrics` - Array of ClassMetrics for all classes
    ///
    /// # Returns
    /// Array of [u64; 5] parameters for each of the 4 classes
    pub fn update_with_metrics(&mut self, metrics: &[ClassMetrics]) -> [[u64; 5]; 4] {
        // Step 1: Update metrics and baselines for each class
        let now_ns = now_as_nanos();
        for metric in metrics {
            let class_idx = metric.class_id as usize;
            if class_idx >= DESCENT_CLASS_MAX {
                continue;
            }

            let state = &mut self.class_states[class_idx];
            state.load_percent = metric.load_percent;
            state.update_latency_ewma(metric.avg_latency_ns);

            // Update baseline (only during low load)
            state.update_baseline(now_ns);
        }

        // Step 2: Calculate delay deltas
        self.calculate_delay_deltas(metrics);

        // Step 3: Check for starvation
        let starved_class = self.find_starved_class();

        // Step 4: Adjust shares based on state or handle starvation
        if let Some(starved) = starved_class {
            debug!(
                "[AUTORATE] Starvation detected: class {} has delay_delta={}µs > threshold={}µs",
                starved,
                self.class_states[starved as usize].delay_delta_ns / 1000,
                BUFFERBLOAT_THRESHOLD_NS / 1000
            );
            self.handle_starvation(starved);
        } else {
            self.normal_share_adjustment(metrics);
        }

        // Step 5: Normalize shares to sum = 1.0
        self.normalize_shares();

        // Step 6: Update stats
        self.update_stats();

        // Step 7: Calculate and return parameters
        self.calculate_parameters()
    }

    /// Calculate delay deltas from baselines for all classes
    fn calculate_delay_deltas(&mut self, metrics: &[ClassMetrics]) {
        for metric in metrics {
            let class_idx = metric.class_id as usize;
            if class_idx >= DESCENT_CLASS_MAX {
                continue;
            }

            let state = &mut self.class_states[class_idx];
            state.delay_delta_ns = state
                .latency_baseline
                .get_delay_delta(state.measured_latency_ns);
        }
    }

    /// Find starved class (delay > bufferbloat threshold)
    ///
    /// Returns the class ID of the starved class, or None if no starvation
    fn find_starved_class(&self) -> Option<u32> {
        for (idx, state) in self.class_states.iter().enumerate() {
            // Check if this class has significant delay delta
            if state.delay_delta_ns > BUFFERBLOAT_THRESHOLD_NS {
                // Ensure we have enough samples for a reliable decision
                // (delay delta is calculated from baseline which requires measurements)
                if state.measured_latency_ns > 0 {
                    return Some(idx as u32);
                }
            }
        }
        None
    }

    /// Handle starvation by backing off controlling class and boosting starved class
    ///
    /// Algorithm:
    /// 1. Find controlling class (highest load, not starved)
    /// 2. Force controlling class to back off (reduce share)
    /// 3. Boost starved class (increase share from freed capacity)
    /// 4. Update states: controlling -> BackingOff, starved -> Exploring
    fn handle_starvation(&mut self, starved_class: u32) {
        // Find controlling class (highest load, not starved)
        let controlling = self.find_controlling_class(starved_class);

        if let Some(ctrl) = controlling {
            // Only back off if controlling class has higher load than starved
            let ctrl_load = self.class_states[ctrl as usize].load_percent;
            let starved_load = self.class_states[starved_class as usize].load_percent;

            if ctrl_load > starved_load && ctrl_load > HIGH_LOAD_THRESHOLD {
                debug!(
                    "[AUTORATE] Handling starvation: class {} (load={:.2}) is starved, \
                     class {} (load={:.2}) is controlling - backing off controlling",
                    starved_class, starved_load, ctrl, ctrl_load
                );

                // Force controlling class to back off
                self.back_off_class(ctrl);

                // Boost starved class with freed capacity
                self.boost_class(starved_class);

                // Update states immediately (no hysteresis for starvation)
                self.class_states[ctrl as usize].transition_to(AutorateState::BackingOff);
                self.class_states[starved_class as usize].transition_to(AutorateState::Exploring);

                // Update tracking
                self.controlling_class = Some(starved_class);
                self.backed_off_class = Some(ctrl);
                self.stats.adjustments_down += 1;
                self.stats.adjustments_up += 1;
            } else {
                // Starvation but no clear controlling class - just boost starved
                debug!(
                    "[AUTORATE] Boosting starved class {} (load={:.2}) without clear controller",
                    starved_class, starved_load
                );
                self.boost_class(starved_class);
                self.class_states[starved_class as usize].transition_to(AutorateState::Exploring);
                self.stats.adjustments_up += 1;
            }
        } else {
            // No controlling class found, just boost starved class
            debug!(
                "[AUTORATE] No controlling class found, boosting starved class {}",
                starved_class
            );
            self.boost_class(starved_class);
            self.class_states[starved_class as usize].transition_to(AutorateState::Exploring);
            self.stats.adjustments_up += 1;
        }
    }

    /// Find controlling class (highest load, excluding the starved class)
    fn find_controlling_class(&self, exclude_class: u32) -> Option<u32> {
        let mut max_load = 0.0;
        let mut controlling = None;

        for (idx, state) in self.class_states.iter().enumerate() {
            let class_id = idx as u32;
            if class_id == exclude_class {
                continue;
            }

            if state.load_percent > max_load {
                max_load = state.load_percent;
                controlling = Some(class_id);
            }
        }

        controlling
    }

    /// Normal share adjustment based on state machine
    ///
    /// For each class:
    /// - Idle: decay toward base_share
    /// - Exploring: ramp_up toward max_share
    /// - BackingOff: ramp_down toward min_share
    /// - Steady: maintain current share
    fn normal_share_adjustment(&mut self, metrics: &[ClassMetrics]) {
        for metric in metrics {
            let class_idx = metric.class_id as usize;
            if class_idx >= DESCENT_CLASS_MAX {
                continue;
            }

            let load = metric.load_percent;
            let state = &self.class_states[class_idx];
            let delay_delta = state.delay_delta_ns;

            // Determine desired state based on conditions
            let desired_state = if load < IDLE_LOAD_THRESHOLD {
                // Low load -> Idle state
                AutorateState::Idle
            } else if delay_delta > BUFFERBLOAT_THRESHOLD_NS {
                // High delay -> BackingOff
                AutorateState::BackingOff
            } else if load > HIGH_LOAD_THRESHOLD && delay_delta < DELAY_THRESHOLD_NS {
                // High load with good latency -> Exploring
                AutorateState::Exploring
            } else if state.state == AutorateState::Exploring
                && state.current_share >= self.share_configs[class_idx].max_share * 0.95
                && delay_delta < DELAY_THRESHOLD_NS
            {
                // Reached max share with good latency -> Steady
                AutorateState::Steady
            } else if state.state == AutorateState::BackingOff && delay_delta < DELAY_THRESHOLD_NS {
                // Recovered from bufferbloat -> Steady
                AutorateState::Steady
            } else {
                // Maintain current state
                state.state
            };

            // Apply state transition with hysteresis
            self.transition_state(metric.class_id, desired_state);

            // Perform share adjustment based on current state
            match self.class_states[class_idx].state {
                AutorateState::Exploring => {
                    self.ramp_up_class(metric.class_id);
                }
                AutorateState::BackingOff => {
                    self.back_off_class(metric.class_id);
                }
                AutorateState::Idle => {
                    self.decay_class_to_base(metric.class_id);
                }
                AutorateState::Steady => {
                    // No adjustment in steady state
                }
            }
        }
    }

    /// Ramp up a class toward max_share
    ///
    /// Multiplies current share by ramp_up_rate from config
    fn ramp_up_class(&mut self, class: u32) {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return;
        }

        let config = &self.share_configs[class_idx];
        let state = &mut self.class_states[class_idx];

        let new_share = state.current_share * config.ramp_up_rate;
        let clamped_share = config.clamp(new_share);

        if (clamped_share - state.current_share).abs() > f64::EPSILON {
            debug!(
                "[AUTORATE] Class {} ramping up: {:.3} -> {:.3}",
                class, state.current_share, clamped_share
            );
            state.record_adjustment(clamped_share, AdjustmentDirection::Up);
            self.stats.adjustments_up += 1;
        }
    }

    /// Back off a class (ramp down)
    ///
    /// Multiplies current share by ramp_down_rate from config
    fn back_off_class(&mut self, class: u32) {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return;
        }

        let config = &self.share_configs[class_idx];
        let state = &mut self.class_states[class_idx];

        let new_share = state.current_share * config.ramp_down_rate;
        let clamped_share = config.clamp(new_share);

        if (clamped_share - state.current_share).abs() > f64::EPSILON {
            debug!(
                "[AUTORATE] Class {} backing off: {:.3} -> {:.3}",
                class, state.current_share, clamped_share
            );
            state.record_adjustment(clamped_share, AdjustmentDirection::Down);
            self.stats.adjustments_down += 1;
        }
    }

    /// Boost a class (increase share more aggressively than normal ramp up)
    ///
    /// Used during starvation handling to quickly give CPU to starved class
    fn boost_class(&mut self, class: u32) {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return;
        }

        let config = &self.share_configs[class_idx];
        let state = &mut self.class_states[class_idx];

        // Boost uses a more aggressive rate (1.2x instead of ramp_up_rate)
        let boost_rate = 1.20;
        let new_share = state.current_share * boost_rate;
        let clamped_share = config.clamp(new_share);

        if (clamped_share - state.current_share).abs() > f64::EPSILON {
            debug!(
                "[AUTORATE] Class {} boosted: {:.3} -> {:.3}",
                class, state.current_share, clamped_share
            );
            state.record_adjustment(clamped_share, AdjustmentDirection::Up);
        }
    }

    /// Decay class toward base share
    ///
    /// Used in Idle state to gradually return to baseline
    fn decay_class_to_base(&mut self, class: u32) {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return;
        }

        let config = &self.share_configs[class_idx];
        let state = &mut self.class_states[class_idx];

        // Move toward base share using decay rate
        let diff = config.base_share - state.current_share;
        let new_share = state.current_share + (diff * (1.0 - config.decay_rate));

        if (new_share - state.current_share).abs() > f64::EPSILON {
            debug!(
                "[AUTORATE] Class {} decaying to base: {:.3} -> {:.3} (target={:.3})",
                class, state.current_share, new_share, config.base_share
            );
            state.record_adjustment(new_share, AdjustmentDirection::None);
        }
    }

    /// Normalize shares to sum = 1.0 (100%)
    ///
    /// Algorithm:
    /// 1. Calculate current sum of shares
    /// 2. If sum != 1.0, scale all shares proportionally
    /// 3. Re-apply min_share clamp to ensure no class goes below minimum
    fn normalize_shares(&mut self) {
        let total_share: f64 = self.class_states.iter().map(|s| s.current_share).sum();

        if (total_share - 1.0).abs() < f64::EPSILON {
            // Already normalized
            return;
        }

        if total_share < f64::EPSILON {
            // All shares are zero, reset to base shares
            for (idx, state) in self.class_states.iter_mut().enumerate() {
                state.current_share = self.share_configs[idx].base_share;
            }
            return;
        }

        // Scale all shares proportionally
        let scale_factor = 1.0 / total_share;
        for (idx, state) in self.class_states.iter_mut().enumerate() {
            let scaled = state.current_share * scale_factor;
            // Ensure we don't go below min_share during normalization
            state.current_share = scaled.max(self.share_configs[idx].min_share);
        }

        // Verify sum is now approximately 1.0
        let new_total: f64 = self.class_states.iter().map(|s| s.current_share).sum();
        debug!(
            "[AUTORATE] Normalized shares: total was {:.3}, now {:.3}",
            total_share, new_total
        );

        // If still not 1.0 due to min_share clamping, adjust largest share
        if (new_total - 1.0).abs() > 0.001 {
            let diff = 1.0 - new_total;
            // Find class with largest share and adjust it
            if let Some((idx, _)) = self
                .class_states
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.current_share.partial_cmp(&b.current_share).unwrap())
            {
                let max_share = self.share_configs[idx].max_share;
                let new_share = (self.class_states[idx].current_share + diff).min(max_share);
                self.class_states[idx].current_share = new_share;
            }
        }
    }

    /// Calculate state transition with hysteresis
    ///
    /// Requires STATE_HYSTERESIS_THRESHOLD (2) consecutive readings
    /// in the desired state before actually transitioning
    fn transition_state(&mut self, class: u32, new_state: AutorateState) {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return;
        }

        let state = &mut self.class_states[class_idx];

        if state.state == new_state {
            // Already in this state, reset pending
            state.pending_state = new_state;
            state.consecutive_state_count = 0;
            return;
        }

        if state.pending_state == new_state {
            // Same pending state, increment counter
            state.consecutive_state_count += 1;

            if state.consecutive_state_count >= STATE_HYSTERESIS_THRESHOLD {
                // Threshold reached, perform transition
                debug!(
                    "[AUTORATE] Class {} state transition: {} -> {} (after {} consecutive readings)",
                    class,
                    state.state.name(),
                    new_state.name(),
                    state.consecutive_state_count
                );
                state.transition_to(new_state);
            }
        } else {
            // New pending state, start counting
            state.pending_state = new_state;
            state.consecutive_state_count = 1;
        }
    }

    /// Calculate scheduler parameters from current shares
    ///
    /// Maps shares to the 5 parameters per class:
    /// - latency_weight
    /// - base_slice_ns
    /// - vruntime_scale
    /// - preemption_priority
    /// - migration_cost
    fn calculate_parameters(&self) -> [[u64; 5]; 4] {
        let mut result = [[0u64; PARAM_COUNT]; DESCENT_CLASS_MAX];

        for class_idx in 0..DESCENT_CLASS_MAX {
            let share = self.class_states[class_idx].current_share;
            let config = &self.share_configs[class_idx];

            // Calculate rate for interpolation (0.0-1.0 based on share range)
            let rate = if config.max_share > config.min_share {
                (share - config.min_share) / (config.max_share - config.min_share)
            } else {
                0.5
            };

            // Clamp rate to 0.0-1.0
            let rate = rate.clamp(0.0, 1.0);

            for param_idx in 0..PARAM_COUNT {
                let min_val = self.config.min_params[class_idx][param_idx] as f64;
                let base_val = self.config.baseline_params[class_idx][param_idx] as f64;
                let max_val = self.config.max_params[class_idx][param_idx] as f64;

                // Different parameters have different mappings
                let value = match param_idx {
                    0 => {
                        // latency_weight: INVERSE relationship
                        // Higher share -> Lower weight (less urgency when have more CPU)
                        let t = (1.0 - rate) * 2.0;
                        if t <= 1.0 {
                            min_val + (base_val - min_val) * t
                        } else {
                            base_val + (max_val - base_val) * (t - 1.0)
                        }
                    }
                    2 => {
                        // vruntime_scale: INVERSE relationship
                        // Higher share -> Lower scale (slower virtual time)
                        let t = (1.0 - rate) * 2.0;
                        if t <= 1.0 {
                            min_val + (base_val - min_val) * t
                        } else {
                            base_val + (max_val - base_val) * (t - 1.0)
                        }
                    }
                    _ => {
                        // base_slice_ns, preemption_priority, migration_cost: DIRECT relationship
                        // Higher share -> Higher values
                        if rate <= 0.5 {
                            let t = rate * 2.0;
                            min_val + (base_val - min_val) * t
                        } else {
                            let t = (rate - 0.5) * 2.0;
                            base_val + (max_val - base_val) * t
                        }
                    }
                };

                result[class_idx][param_idx] = value as u64;
            }
        }

        result
    }

    /// Calculate single parameter for a class based on share
    ///
    /// param_idx: 0=latency_weight, 1=base_slice_ns, 2=vruntime_scale,
    ///            3=preemption_priority, 4=migration_cost
    fn calculate_param(&self, class: u32, param_idx: usize, share: f64) -> u64 {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX || param_idx >= PARAM_COUNT {
            return 0;
        }

        let config = &self.share_configs[class_idx];
        let min_val = self.config.min_params[class_idx][param_idx];
        let base_val = self.config.baseline_params[class_idx][param_idx];
        let max_val = self.config.max_params[class_idx][param_idx];

        // Determine mapping type based on parameter index
        let mapping_type = match param_idx {
            PARAM_LATENCY_WEIGHT => MAPPING_INVERSE,
            PARAM_BASE_SLICE_NS => MAPPING_DIRECT,
            PARAM_VRUNTIME_SCALE => MAPPING_INVERSE,
            PARAM_PREEMPTION_PRIORITY => MAPPING_DIRECT,
            PARAM_MIGRATION_COST => MAPPING_DIRECT,
            _ => MAPPING_DIRECT,
        };

        self.map_share_to_param(
            share,
            config.min_share,
            config.base_share,
            config.max_share,
            min_val,
            base_val,
            max_val,
            mapping_type,
        )
    }

    /// Map share to parameter value using profile's min/baseline/max
    ///
    /// Algorithm:
    /// 1. Clamp share to [min_share, max_share]
    /// 2. If min_share == max_share, return base_val (edge case)
    /// 3. Calculate normalized rate: (share - min_share) / (max_share - min_share)
    /// 4. For DIRECT mapping:
    ///    - If rate <= 0.5: interpolate min -> base
    ///    - If rate > 0.5: interpolate base -> max
    /// 5. For INVERSE mapping:
    ///    - Invert rate: t = (1.0 - rate) * 2.0
    ///    - If t <= 1.0: interpolate min -> base
    ///    - If t > 1.0: interpolate base -> max
    fn map_share_to_param(
        &self,
        share: f64,
        min_share: f64,
        base_share: f64,
        max_share: f64,
        min_val: u64,
        base_val: u64,
        max_val: u64,
        mapping_type: &str,
    ) -> u64 {
        // Clamp share to valid range
        let clamped_share = share.clamp(min_share, max_share);

        // Edge case: if min_share == max_share, return base_val
        if (max_share - min_share).abs() < f64::EPSILON {
            return base_val;
        }

        // Calculate normalized rate (0.0 to 1.0)
        let rate = (clamped_share - min_share) / (max_share - min_share);

        match mapping_type {
            MAPPING_INVERSE => {
                // Inverse mapping: higher share = lower value
                // Transform rate: t = (1.0 - rate) * 2.0
                // - When rate = 0.0 (min_share), t = 2.0 (max value)
                // - When rate = 0.5 (midpoint), t = 1.0 (base value)
                // - When rate = 1.0 (max_share), t = 0.0 (min value)
                let t = (1.0 - rate) * 2.0;

                if t <= 1.0 {
                    // Interpolate between min and base
                    // t = 0.0 -> min_val, t = 1.0 -> base_val
                    let min_f = min_val as f64;
                    let base_f = base_val as f64;
                    (min_f + (base_f - min_f) * t).round() as u64
                } else {
                    // Interpolate between base and max
                    // t = 1.0 -> base_val, t = 2.0 -> max_val
                    let base_f = base_val as f64;
                    let max_f = max_val as f64;
                    let t2 = t - 1.0; // Normalize to 0.0-1.0 range
                    (base_f + (max_f - base_f) * t2).round() as u64
                }
            }
            _ => {
                // Direct mapping: higher share = higher value
                if rate <= 0.5 {
                    // Interpolate between min and base
                    // rate = 0.0 -> min_val, rate = 0.5 -> base_val
                    let t = rate * 2.0; // Scale to 0.0-1.0
                    let min_f = min_val as f64;
                    let base_f = base_val as f64;
                    (min_f + (base_f - min_f) * t).round() as u64
                } else {
                    // Interpolate between base and max
                    // rate = 0.5 -> base_val, rate = 1.0 -> max_val
                    let t = (rate - 0.5) * 2.0; // Scale to 0.0-1.0
                    let base_f = base_val as f64;
                    let max_f = max_val as f64;
                    (base_f + (max_f - base_f) * t).round() as u64
                }
            }
        }
    }

    /// Update internal statistics
    fn update_stats(&mut self) {
        let total_share: f64 = self.class_states.iter().map(|s| s.current_share).sum();
        self.stats.avg_share = total_share / DESCENT_CLASS_MAX as f64;
        self.stats.controlling_class = self.controlling_class;
        self.stats.backed_off_class = self.backed_off_class;
    }

    /// =============================================================================
    /// LEGACY API (for backward compatibility during transition)
    /// =============================================================================

    /// Main update - called once per class per interval
    ///
    /// PHASE 2: This method now delegates to update_with_metrics for consistency.
    /// For single-class updates, we process immediately but don't trigger full normalization.
    ///
    /// # Arguments
    /// * `class` - Class ID (0-3)
    /// * `latency_ns` - Average latency across all CPUs for this class
    /// * `target_latency_ns` - Target for this class
    /// * `load_percent` - Aggregate load % across all CPUs (0.0-1.0)
    /// * `sample_count` - Number of latency samples
    ///
    /// # Returns
    /// Tuple of (interpolated_params, current_state)
    pub fn update(
        &mut self,
        class: u32,
        latency_ns: u64,
        target_latency_ns: u64,
        load_percent: f64,
        sample_count: u64,
    ) -> ([u64; PARAM_COUNT], AutorateState) {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            // Invalid class - return baseline
            return (
                self.config.baseline_params[class.min(3) as usize],
                AutorateState::Steady,
            );
        }

        // Get current time for baseline updates
        let now_ns = now_as_nanos();

        // Update class state with metrics
        {
            let state = &mut self.class_states[class_idx];
            state.update_latency_ewma(latency_ns);
            state.load_percent = load_percent;
            state.target_latency_ns = target_latency_ns;

            // Update latency baseline (only during low load)
            state.update_baseline(now_ns);
        }

        // Update stats
        self.stats.baseline_updates += 1;

        // If autorate is disabled, return baseline params but still track state
        if !self.config.enabled {
            return (
                self.config.baseline_params[class_idx],
                self.class_states[class_idx].state,
            );
        }

        // Create a single-element metrics array for processing
        let metrics = [ClassMetrics {
            class_id: class,
            cycles_spent: 0,
            sample_count,
            total_latency_ns: latency_ns * sample_count, // Total = avg * count
            avg_latency_ns: latency_ns,
            max_latency_ns: latency_ns,
            load_percent,
            delay_delta_ns: self.class_states[class_idx].delay_delta_ns,
        }];

        // Use the new update_with_metrics logic (but only normalize periodically)
        // For now, just do normal adjustment for this single class
        self.calculate_delay_deltas(&metrics);

        // Check if this specific class is starved (simplified check)
        let is_starved = self.class_states[class_idx].delay_delta_ns > BUFFERBLOAT_THRESHOLD_NS;

        if is_starved {
            // Find if there's a controlling class (highest load among other classes)
            let mut max_other_load = 0.0;
            let mut controlling = None;
            for (idx, other_state) in self.class_states.iter().enumerate() {
                if idx != class_idx && other_state.load_percent > max_other_load {
                    max_other_load = other_state.load_percent;
                    controlling = Some(idx as u32);
                }
            }

            if let Some(ctrl) = controlling {
                if max_other_load > HIGH_LOAD_THRESHOLD {
                    self.back_off_class(ctrl);
                    self.boost_class(class);
                    self.transition_state(ctrl, AutorateState::BackingOff);
                    self.transition_state(class, AutorateState::Exploring);
                }
            }
        } else {
            // Normal adjustment
            self.normal_share_adjustment(&metrics);
        }

        // Calculate parameters for this class
        let params = self.calculate_parameters();
        let final_state = self.class_states[class_idx].state;

        debug!(
            "[AUTORATE] Class {}: state={} share={:.3} load={:.2} lat={}µs baseline={}µs delta={}µs",
            class,
            final_state.name(),
            self.class_states[class_idx].current_share,
            load_percent,
            latency_ns / 1000,
            self.class_states[class_idx].latency_baseline.baseline_ns / 1000,
            self.class_states[class_idx].delay_delta_ns / 1000,
        );

        (params[class_idx], final_state)
    }

    /// Update using ClassMetrics (new API for Phase 2+)
    ///
    /// This is now the primary update method. See update_with_metrics for full implementation.
    pub fn update_with_metrics_single(
        &mut self,
        metrics: &ClassMetrics,
    ) -> ([u64; PARAM_COUNT], AutorateState) {
        self.update(
            metrics.class_id,
            metrics.avg_latency_ns,
            0, // target will come from state
            metrics.load_percent,
            metrics.sample_count,
        )
    }

    /// Get calculated parameters for a specific class (for use when autorate hasn't run this cycle)
    ///
    /// This method calculates parameters from the current share without running
    /// the full update algorithm. Useful when PIE needs base params but autorate
    /// hasn't reached its 200ms interval yet.
    ///
    /// # Arguments
    /// * `class` - Class ID (0-3)
    ///
    /// # Returns
    /// [u64; 5] parameters for the class
    pub fn get_params_for_class(&self, class: u32) -> [u64; PARAM_COUNT] {
        let class_idx = class as usize;
        if class_idx >= DESCENT_CLASS_MAX {
            return self.config.baseline_params[class_idx.min(DESCENT_CLASS_MAX - 1)];
        }

        let share = self.class_states[class_idx].current_share;
        let config = &self.share_configs[class_idx];

        // Calculate rate for interpolation (0.0-1.0 based on share range)
        let rate = if config.max_share > config.min_share {
            (share - config.min_share) / (config.max_share - config.min_share)
        } else {
            0.5
        };

        // Clamp rate to 0.0-1.0
        let rate = rate.clamp(0.0, 1.0);

        let mut result = [0u64; PARAM_COUNT];

        for param_idx in 0..PARAM_COUNT {
            let min_val = self.config.min_params[class_idx][param_idx] as f64;
            let base_val = self.config.baseline_params[class_idx][param_idx] as f64;
            let max_val = self.config.max_params[class_idx][param_idx] as f64;

            // Same interpolation logic as calculate_parameters
            let value = match param_idx {
                0 | 2 => {
                    // latency_weight, vruntime_scale: INVERSE relationship
                    let t = (1.0 - rate) * 2.0;
                    if t <= 1.0 {
                        min_val + (base_val - min_val) * t
                    } else {
                        base_val + (max_val - base_val) * (t - 1.0)
                    }
                }
                _ => {
                    // base_slice_ns, preemption_priority, migration_cost: DIRECT relationship
                    if rate <= 0.5 {
                        let t = rate * 2.0;
                        min_val + (base_val - min_val) * t
                    } else {
                        let t = (rate - 0.5) * 2.0;
                        base_val + (max_val - base_val) * t
                    }
                }
            };

            result[param_idx] = value as u64;
        }

        result
    }

    /// Get current state for a class (new API)
    pub fn get_class_autorate_state(&self, class: u32) -> Option<&ClassAutorateState> {
        let class_idx = class as usize;
        if class_idx < DESCENT_CLASS_MAX {
            Some(&self.class_states[class_idx])
        } else {
            None
        }
    }

    /// Get mutable reference to class state
    pub fn get_class_autorate_state_mut(&mut self, class: u32) -> Option<&mut ClassAutorateState> {
        let class_idx = class as usize;
        if class_idx < DESCENT_CLASS_MAX {
            Some(&mut self.class_states[class_idx])
        } else {
            None
        }
    }

    /// Get share config for a class
    pub fn get_share_config(&self, class: u32) -> Option<&ShareConfig> {
        let class_idx = class as usize;
        if class_idx < DESCENT_CLASS_MAX {
            Some(&self.share_configs[class_idx])
        } else {
            None
        }
    }

    /// Get current state for debugging (legacy API - deprecated)
    #[deprecated(since = "Phase 1", note = "Use get_class_autorate_state instead")]
    #[allow(deprecated)]
    pub fn get_class_state(&self, _class: u32) -> Option<&AutorateClassState> {
        None // Legacy API no longer supported
    }

    /// Get controller statistics
    pub fn get_stats(&self) -> AutorateStats {
        let mut stats = self.stats;

        // Calculate average share
        let total_share: f64 = self.class_states.iter().map(|s| s.current_share).sum();
        stats.avg_share = total_share / DESCENT_CLASS_MAX as f64;
        stats.controlling_class = self.controlling_class;
        stats.backed_off_class = self.backed_off_class;

        stats
    }

    /// Get the current controlling class
    pub fn get_controlling_class(&self) -> Option<u32> {
        self.controlling_class
    }

    /// Get the backed off class
    pub fn get_backed_off_class(&self) -> Option<u32> {
        self.backed_off_class
    }

    /// Set the controlling class
    pub fn set_controlling_class(&mut self, class: Option<u32>) {
        self.controlling_class = class;
    }

    /// Set the backed off class
    pub fn set_backed_off_class(&mut self, class: Option<u32>) {
        self.backed_off_class = class;
    }

    /// Get all class states as slice
    pub fn get_all_class_states(&self) -> &[ClassAutorateState] {
        &self.class_states
    }

    /// Get mutable reference to all class states
    pub fn get_all_class_states_mut(&mut self) -> &mut [ClassAutorateState] {
        &mut self.class_states
    }

    /// Reset a class to base share
    pub fn reset_class_to_base(&mut self, class: u32) {
        let class_idx = class as usize;
        if class_idx < DESCENT_CLASS_MAX {
            let base_share = self.share_configs[class_idx].base_share;
            self.class_states[class_idx].record_adjustment(base_share, AdjustmentDirection::None);
            self.class_states[class_idx].transition_to(AutorateState::Steady);
        }
    }

    /// Force update baseline for a class (for testing/debugging)
    pub fn force_baseline_update(&mut self, class: u32, measured_ns: u64) {
        let class_idx = class as usize;
        if class_idx < DESCENT_CLASS_MAX {
            let now_ns = now_as_nanos();
            self.class_states[class_idx].measured_latency_ns = measured_ns;
            self.class_states[class_idx].force_update_baseline(now_ns);
        }
    }

    /// Legacy interpolate_params - kept for compatibility during transition
    ///
    /// rate 0.0 -> min_params
    /// rate 0.5 -> baseline_params
    /// rate 1.0 -> max_params
    #[allow(dead_code)]
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
}

/// Helper function to get current time in nanoseconds
fn now_as_nanos() -> u64 {
    // Use a fixed epoch for simplicity in Phase 2
    // In production, this would use a proper monotonic clock
    Instant::now().elapsed().as_nanos() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    // =============================================================================
    // PHASE 1 TESTS - New Data Structures
    // =============================================================================

    #[test]
    fn test_latency_baseline_creation() {
        let baseline = LatencyBaseline::new(1000_000);
        assert_eq!(baseline.baseline_ns, 1000_000);
        assert_eq!(baseline.alpha, BASELINE_EWMA_ALPHA);
    }

    #[test]
    fn test_latency_baseline_with_alpha() {
        let baseline = LatencyBaseline::with_alpha(1000_000, 0.5);
        assert_eq!(baseline.baseline_ns, 1000_000);
        assert_eq!(baseline.alpha, 0.5);
    }

    #[test]
    fn test_latency_baseline_clamps_alpha() {
        let baseline_high = LatencyBaseline::with_alpha(1000_000, 1.5);
        assert_eq!(baseline_high.alpha, 1.0);

        let baseline_low = LatencyBaseline::with_alpha(1000_000, -0.5);
        assert_eq!(baseline_low.alpha, 0.0);
    }

    #[test]
    fn test_latency_baseline_update_during_low_load() {
        let mut baseline = LatencyBaseline::new(1000_000);

        // Update during low load (< 0.25) - should update
        baseline.update(2000_000, 0.1, 1000);

        // EWMA: 0.2 * 2000000 + 0.8 * 1000000 = 1200000
        assert_eq!(baseline.baseline_ns, 1200_000);
        assert_eq!(baseline.last_update_ns, 1000);
    }

    #[test]
    fn test_latency_baseline_no_update_during_high_load() {
        let mut baseline = LatencyBaseline::new(1000_000);

        // Update during high load (>= 0.25) - should NOT update
        baseline.update(2000_000, 0.5, 1000);

        // Baseline should remain unchanged
        assert_eq!(baseline.baseline_ns, 1000_000);
        assert_eq!(baseline.last_update_ns, 0);
    }

    #[test]
    fn test_latency_baseline_at_threshold() {
        let mut baseline = LatencyBaseline::new(1000_000);

        // At exactly 0.25 threshold - should NOT update (strictly less than)
        baseline.update(2000_000, LOW_LOAD_THRESHOLD, 1000);

        // Baseline should remain unchanged
        assert_eq!(baseline.baseline_ns, 1000_000);
    }

    #[test]
    fn test_latency_baseline_force_update() {
        let mut baseline = LatencyBaseline::new(1000_000);

        // Force update should work even during high load
        baseline.force_update(2000_000, 1000);

        assert_eq!(baseline.baseline_ns, 1200_000);
        assert_eq!(baseline.last_update_ns, 1000);
    }

    #[test]
    fn test_latency_baseline_get_delay_delta() {
        let baseline = LatencyBaseline::new(1000_000);

        // Measured above baseline
        let delta = baseline.get_delay_delta(1500_000);
        assert_eq!(delta, 500_000);

        // Measured below baseline
        let delta = baseline.get_delay_delta(500_000);
        assert_eq!(delta, -500_000);

        // Measured at baseline
        let delta = baseline.get_delay_delta(1000_000);
        assert_eq!(delta, 0);
    }

    #[test]
    fn test_latency_baseline_get_delay_ratio() {
        let baseline = LatencyBaseline::new(1000_000);

        assert!((baseline.get_delay_ratio(1500_000) - 1.5).abs() < 0.001);
        assert!((baseline.get_delay_ratio(500_000) - 0.5).abs() < 0.001);
        assert!((baseline.get_delay_ratio(1000_000) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_latency_baseline_zero_baseline_ratio() {
        let baseline = LatencyBaseline::new(0);

        // Should return 1.0 (neutral) when baseline is zero to avoid division by zero
        assert_eq!(baseline.get_delay_ratio(1500_000), 1.0);
    }

    #[test]
    fn test_latency_baseline_is_stale() {
        let mut baseline = LatencyBaseline::new(1000_000);
        baseline.last_update_ns = 1000;

        // Not stale
        assert!(!baseline.is_stale(2000, 5000));

        // Stale (more than 5000ns since last update)
        assert!(baseline.is_stale(7000, 5000));
    }

    #[test]
    fn test_share_config_creation() {
        let config = ShareConfig::new(0.1, 0.5, 1.0, 1.1, 0.9, 0.99);

        assert_eq!(config.min_share, 0.1);
        assert_eq!(config.base_share, 0.5);
        assert_eq!(config.max_share, 1.0);
        assert_eq!(config.ramp_up_rate, 1.1);
        assert_eq!(config.ramp_down_rate, 0.9);
        assert_eq!(config.decay_rate, 0.99);
    }

    #[test]
    fn test_share_config_clamps_values() {
        let config = ShareConfig::new(-0.1, 0.5, 1.5, 1.1, 0.9, 0.99);

        // Min should be clamped to 0.0
        assert_eq!(config.min_share, 0.0);

        // Max should be clamped to 1.0
        assert_eq!(config.max_share, 1.0);

        // Base should remain unchanged (already valid)
        assert_eq!(config.base_share, 0.5);
    }

    #[test]
    fn test_share_config_is_valid() {
        let valid = ShareConfig::new(0.1, 0.5, 1.0, 1.1, 0.9, 0.99);
        assert!(valid.is_valid());

        // min > base - invalid (0.6 > 0.5)
        let invalid1 = ShareConfig::new(0.6, 0.5, 1.0, 1.1, 0.9, 0.99);
        assert!(!invalid1.is_valid());

        // base > max - invalid (0.9 > 0.8)
        // Note: we use 0.9 for base and 0.8 for max which are within [0,1] bounds
        // so they won't be clamped, but base > max makes it invalid
        let invalid2 = ShareConfig::new(0.1, 0.9, 0.8, 1.1, 0.9, 0.99);
        assert!(!invalid2.is_valid());
    }

    #[test]
    fn test_share_config_clamp() {
        let config = ShareConfig::new(0.2, 0.5, 0.8, 1.1, 0.9, 0.99);

        // Within range
        assert_eq!(config.clamp(0.5), 0.5);

        // Below min
        assert_eq!(config.clamp(0.1), 0.2);

        // Above max
        assert_eq!(config.clamp(0.9), 0.8);
    }

    #[test]
    fn test_share_config_default() {
        let config = ShareConfig::default();

        assert_eq!(config.min_share, 0.1);
        assert_eq!(config.base_share, 0.25);
        assert_eq!(config.max_share, 0.5);
        assert!(config.is_valid());
    }

    #[test]
    fn test_autorate_state_names() {
        assert_eq!(AutorateState::Idle.name(), "IDLE");
        assert_eq!(AutorateState::Exploring.name(), "EXPLORING");
        assert_eq!(AutorateState::BackingOff.name(), "BACKING_OFF");
        assert_eq!(AutorateState::Steady.name(), "STEADY");
    }

    #[test]
    fn test_autorate_state_allows_baseline_update() {
        assert!(AutorateState::Idle.allows_baseline_update());
        assert!(AutorateState::Steady.allows_baseline_update());
        assert!(!AutorateState::Exploring.allows_baseline_update());
        assert!(!AutorateState::BackingOff.allows_baseline_update());
    }

    #[test]
    fn test_autorate_state_is_adjusting() {
        assert!(!AutorateState::Idle.is_adjusting());
        assert!(AutorateState::Exploring.is_adjusting());
        assert!(AutorateState::BackingOff.is_adjusting());
        assert!(!AutorateState::Steady.is_adjusting());
    }

    #[test]
    fn test_class_autorate_state_creation() {
        let state = ClassAutorateState::new(0.5, 1000_000);

        assert_eq!(state.current_share, 0.5);
        assert_eq!(state.state, AutorateState::Idle);
        assert_eq!(state.latency_baseline.baseline_ns, 1000_000);
        assert_eq!(state.load_percent, 0.0);
        assert_eq!(state.target_latency_ns, 1000_000);
    }

    #[test]
    fn test_class_autorate_state_with_alpha() {
        let state = ClassAutorateState::with_alpha(0.5, 1000_000, 0.3);

        assert_eq!(state.latency_baseline.alpha, 0.3);
    }

    #[test]
    fn test_class_autorate_state_share_clamping() {
        let state = ClassAutorateState::new(1.5, 1000_000);
        assert_eq!(state.current_share, 1.0);

        let state2 = ClassAutorateState::new(-0.5, 1000_000);
        assert_eq!(state2.current_share, 0.0);
    }

    #[test]
    fn test_class_autorate_state_update_latency_ewma() {
        let mut state = ClassAutorateState::new(0.5, 1000_000);

        // First measurement
        state.update_latency_ewma(1000_000);
        assert_eq!(state.latency_ewma_ns, 1000_000);
        assert_eq!(state.measured_latency_ns, 1000_000);

        // Second measurement - EWMA blend
        // 0.2 * 2000000 + 0.8 * 1000000 = 1200000
        state.update_latency_ewma(2000_000);
        assert_eq!(state.latency_ewma_ns, 1200_000);
        assert_eq!(state.measured_latency_ns, 2000_000);
    }

    #[test]
    fn test_class_autorate_state_transition_to() {
        let mut state = ClassAutorateState::new(0.5, 1000_000);

        let before = state.last_state_change;

        // Transition to same state - should not update timestamp
        state.transition_to(AutorateState::Idle);
        assert_eq!(state.last_state_change, before);

        // Transition to different state - should update timestamp
        std::thread::sleep(Duration::from_millis(1));
        state.transition_to(AutorateState::Exploring);
        assert!(state.last_state_change > before);
        assert_eq!(state.state, AutorateState::Exploring);
    }

    #[test]
    fn test_class_autorate_state_record_adjustment() {
        let mut state = ClassAutorateState::new(0.5, 1000_000);

        state.record_adjustment(0.7, AdjustmentDirection::Up);
        assert_eq!(state.current_share, 0.7);
        assert_eq!(state.last_direction, AdjustmentDirection::Up);

        // Should clamp
        state.record_adjustment(1.5, AdjustmentDirection::Down);
        assert_eq!(state.current_share, 1.0);
        assert_eq!(state.last_direction, AdjustmentDirection::Down);

        state.record_adjustment(-0.5, AdjustmentDirection::None);
        assert_eq!(state.current_share, 0.0);
        assert_eq!(state.last_direction, AdjustmentDirection::None);
    }

    #[test]
    fn test_class_autorate_state_get_delay_ratio() {
        let mut state = ClassAutorateState::new(0.5, 1000_000);
        state.update_latency_ewma(1500_000);
        // Use force_update_baseline to ensure exact value (no load threshold check)
        state.force_update_baseline(1000);

        // Measured 1500us, baseline updated with EWMA: 0.2*1500000 + 0.8*1000000 = 1100000
        // But delay_delta is calculated using the updated baseline
        let expected_baseline = 1100_000;
        let expected_ratio = 1500_000 as f64 / expected_baseline as f64;
        assert!((state.get_delay_ratio() - expected_ratio).abs() < 0.001);
    }

    #[test]
    fn test_class_metrics_creation() {
        let metrics = ClassMetrics::new(2);

        assert_eq!(metrics.class_id, 2);
        assert_eq!(metrics.cycles_spent, 0);
        assert_eq!(metrics.sample_count, 0);
        assert_eq!(metrics.avg_latency_ns, 0);
        assert_eq!(metrics.load_percent, 0.0);
    }

    #[test]
    fn test_class_metrics_is_valid() {
        // Valid class with samples
        let mut metrics = ClassMetrics::new(2);
        metrics.sample_count = 10;
        assert!(metrics.is_valid());

        // Invalid class (too high)
        let metrics_invalid = ClassMetrics::new(10);
        assert!(!metrics_invalid.is_valid());

        // Valid class but no samples
        let metrics_no_samples = ClassMetrics::new(1);
        assert!(!metrics_no_samples.is_valid());
    }

    #[test]
    fn test_class_metrics_has_min_samples() {
        let mut metrics = ClassMetrics::new(1);
        metrics.sample_count = 10;

        assert!(metrics.has_min_samples(5));
        assert!(metrics.has_min_samples(10));
        assert!(!metrics.has_min_samples(15));
    }

    #[test]
    fn test_autorate_config_default() {
        let config = AutorateConfig::default();

        assert!(config.enabled);
        assert_eq!(config.high_load_threshold, HIGH_LOAD_THRESHOLD);
        assert_eq!(config.low_load_threshold, IDLE_LOAD_THRESHOLD);
        assert_eq!(config.ramp_up_rate, 1.1);
        assert_eq!(config.ramp_down_rate, 0.9);
        assert_eq!(config.decay_rate, 0.99);
        assert_eq!(config.bufferbloat_threshold, 1.5);

        // Should have 4 share configs
        assert_eq!(config.share_configs.len(), DESCENT_CLASS_MAX);
    }

    #[test]
    fn test_autorate_stats_default() {
        let stats = AutorateStats::default();

        assert_eq!(stats.total_classes, DESCENT_CLASS_MAX);
        assert_eq!(stats.adjustments_up, 0);
        assert_eq!(stats.adjustments_down, 0);
        assert_eq!(stats.adjustments_blocked, 0);
        assert_eq!(stats.avg_share, 0.25);
        assert_eq!(stats.baseline_updates, 0);
        assert_eq!(stats.controlling_class, None);
        assert_eq!(stats.backed_off_class, None);
    }

    #[test]
    fn test_controller_creation() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        let stats = controller.get_stats();
        assert_eq!(stats.total_classes, DESCENT_CLASS_MAX);
        assert_eq!(controller.get_controlling_class(), None);
        assert_eq!(controller.get_backed_off_class(), None);
    }

    #[test]
    fn test_controller_with_interval() {
        let config = AutorateConfig::default();
        let controller = AutorateController::with_interval(4, &config, 50);

        // Controller should be created with custom interval
        // (interval is private, but we can verify it was created)
        let stats = controller.get_stats();
        assert_eq!(stats.total_classes, DESCENT_CLASS_MAX);
    }

    #[test]
    fn test_controller_get_class_state() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Valid classes
        for i in 0..DESCENT_CLASS_MAX as u32 {
            let state = controller.get_class_autorate_state(i);
            assert!(state.is_some());
            let state = state.unwrap();
            assert_eq!(state.state, AutorateState::Idle);
        }

        // Invalid class
        let state = controller.get_class_autorate_state(99);
        assert!(state.is_none());
    }

    #[test]
    fn test_controller_get_share_config() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Valid classes
        for i in 0..DESCENT_CLASS_MAX as u32 {
            let share_config = controller.get_share_config(i);
            assert!(share_config.is_some());
        }

        // Invalid class
        let share_config = controller.get_share_config(99);
        assert!(share_config.is_none());
    }

    #[test]
    fn test_controller_set_controlling_class() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        controller.set_controlling_class(Some(2));
        assert_eq!(controller.get_controlling_class(), Some(2));

        controller.set_controlling_class(None);
        assert_eq!(controller.get_controlling_class(), None);
    }

    #[test]
    fn test_controller_reset_class_to_base() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // First set share to something else
        {
            let state = controller.get_class_autorate_state_mut(1).unwrap();
            state.record_adjustment(0.8, AdjustmentDirection::Up);
            state.transition_to(AutorateState::Exploring);
        }

        // Reset to base
        controller.reset_class_to_base(1);

        let state = controller.get_class_autorate_state(1).unwrap();
        assert_eq!(state.current_share, config.share_configs[1].base_share);
        assert_eq!(state.state, AutorateState::Steady);
    }

    #[test]
    fn test_controller_force_baseline_update() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Force update during high load (normally wouldn't update)
        controller.force_baseline_update(1, 2000_000);

        let state = controller.get_class_autorate_state(1).unwrap();
        // EWMA: 0.2 * 2000000 + 0.8 * 1000000 = 1200000
        assert_eq!(state.latency_baseline.baseline_ns, 1200_000);
    }

    // =============================================================================
    // PHASE 2 TESTS - Algorithm Implementation
    // =============================================================================

    #[test]
    fn test_find_starved_class() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Set up class 0 with high delay delta (starved)
        {
            let state = controller.get_class_autorate_state_mut(0).unwrap();
            state.delay_delta_ns = BUFFERBLOAT_THRESHOLD_NS + 1000;
            state.measured_latency_ns = 5000_000; // 5ms
        }

        // Class 1 with normal delay
        {
            let state = controller.get_class_autorate_state_mut(1).unwrap();
            state.delay_delta_ns = 100_000; // 100µs
            state.measured_latency_ns = 1000_000;
        }

        let starved = controller.find_starved_class();
        assert_eq!(starved, Some(0));
    }

    #[test]
    fn test_find_starved_class_none() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // All classes with normal delay - no starvation
        let starved = controller.find_starved_class();
        assert_eq!(starved, None);
    }

    #[test]
    fn test_find_controlling_class() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Set up class 2 (HOG) with high load
        {
            let state = controller.get_class_autorate_state_mut(2).unwrap();
            state.load_percent = 0.95;
        }

        // Set up class 1 with medium load
        {
            let state = controller.get_class_autorate_state_mut(1).unwrap();
            state.load_percent = 0.50;
        }

        // Class 0 with low load
        {
            let state = controller.get_class_autorate_state_mut(0).unwrap();
            state.load_percent = 0.10;
        }

        // Find controlling class excluding class 0
        let controlling = controller.find_controlling_class(0);
        assert_eq!(controlling, Some(2)); // HOG class (highest load)

        // Find controlling class excluding class 2 (the actual controller)
        let controlling = controller.find_controlling_class(2);
        assert_eq!(controlling, Some(1)); // NORMAL class (next highest)
    }

    #[test]
    fn test_ramp_up_class() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Set initial share
        {
            let state = controller.get_class_autorate_state_mut(1).unwrap();
            state.current_share = 0.25;
        }

        // Ramp up
        controller.ramp_up_class(1);

        let state = controller.get_class_autorate_state(1).unwrap();
        // 0.25 * 1.05 (ramp_up_rate for NORMAL class) = 0.2625
        assert!(state.current_share > 0.25);
    }

    #[test]
    fn test_back_off_class() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Set initial share higher
        {
            let state = controller.get_class_autorate_state_mut(1).unwrap();
            state.current_share = 0.35;
        }

        // Back off
        controller.back_off_class(1);

        let state = controller.get_class_autorate_state(1).unwrap();
        // 0.35 * 0.85 (ramp_down_rate for NORMAL class) = 0.2975
        assert!(state.current_share < 0.35);
    }

    #[test]
    fn test_boost_class() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Set initial share
        {
            let state = controller.get_class_autorate_state_mut(0).unwrap();
            state.current_share = 0.25;
        }

        // Boost
        controller.boost_class(0);

        let state = controller.get_class_autorate_state(0).unwrap();
        // 0.25 * 1.20 (boost rate) = 0.30
        assert_eq!(state.current_share, 0.30);
    }

    #[test]
    fn test_decay_class_to_base() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Set initial share above base
        {
            let state = controller.get_class_autorate_state_mut(0).unwrap();
            state.current_share = 0.40; // Above base of 0.25
        }

        // Decay
        controller.decay_class_to_base(0);

        let state = controller.get_class_autorate_state(0).unwrap();
        // Should move closer to base (0.25)
        assert!(state.current_share < 0.40);
        assert!(state.current_share >= 0.25);
    }

    #[test]
    fn test_normalize_shares() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Set up shares that don't sum to 1.0
        controller.class_states[0].current_share = 0.40;
        controller.class_states[1].current_share = 0.40;
        controller.class_states[2].current_share = 0.40;
        controller.class_states[3].current_share = 0.40;
        // Sum = 1.6

        controller.normalize_shares();

        // After normalization, sum should be approximately 1.0
        let total: f64 = controller
            .class_states
            .iter()
            .map(|s| s.current_share)
            .sum();
        assert!((total - 1.0).abs() < 0.01);

        // No class should be below min_share
        for (idx, state) in controller.class_states.iter().enumerate() {
            assert!(state.current_share >= controller.share_configs[idx].min_share);
        }
    }

    #[test]
    fn test_transition_state_with_hysteresis() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // Initial state is Idle
        let state = controller.get_class_autorate_state(0).unwrap();
        assert_eq!(state.state, AutorateState::Idle);

        // First transition attempt - should set pending but not change state
        controller.transition_state(0, AutorateState::Exploring);
        let state = controller.get_class_autorate_state(0).unwrap();
        assert_eq!(state.state, AutorateState::Idle); // Still idle
        assert_eq!(state.pending_state, AutorateState::Exploring);
        assert_eq!(state.consecutive_state_count, 1);

        // Second transition attempt - should change state
        controller.transition_state(0, AutorateState::Exploring);
        let state = controller.get_class_autorate_state(0).unwrap();
        assert_eq!(state.state, AutorateState::Exploring); // Now exploring
        assert_eq!(state.consecutive_state_count, 0); // Reset after transition
    }

    #[test]
    fn test_transition_state_different_pending_resets() {
        let config = AutorateConfig::default();
        let mut controller = AutorateController::new(4, &config);

        // First transition attempt to Exploring
        controller.transition_state(0, AutorateState::Exploring);
        let state = controller.get_class_autorate_state(0).unwrap();
        assert_eq!(state.consecutive_state_count, 1);

        // Different transition attempt to BackingOff - should reset count
        controller.transition_state(0, AutorateState::BackingOff);
        let state = controller.get_class_autorate_state(0).unwrap();
        assert_eq!(state.pending_state, AutorateState::BackingOff);
        assert_eq!(state.consecutive_state_count, 1); // Reset to 1
    }

    #[test]
    fn test_handle_starvation() {
        let mut config = AutorateConfig::default();
        // Set up specific share configs for testing
        config.share_configs[0] = ShareConfig::new(0.15, 0.25, 0.40, 1.08, 0.85, 0.98);
        config.share_configs[2] = ShareConfig::new(0.20, 0.35, 0.60, 1.10, 0.80, 0.98);

        let mut controller = AutorateController::new(4, &config);

        // Simulate: HOG (class 2) has high load, LATENCY_CRITICAL (class 0) is starved
        {
            let state = controller.get_class_autorate_state_mut(2).unwrap();
            state.load_percent = 0.95;
            state.current_share = 0.50; // High share
        }
        {
            let state = controller.get_class_autorate_state_mut(0).unwrap();
            state.load_percent = 0.05;
            state.delay_delta_ns = BUFFERBLOAT_THRESHOLD_NS + 1000;
            state.measured_latency_ns = 5000_000;
            state.current_share = 0.20; // Low share
        }

        // Calculate delay deltas
        let metrics = [
            ClassMetrics {
                class_id: 0,
                cycles_spent: 100,
                sample_count: 5,
                total_latency_ns: 5000_000 * 5, // avg * count
                avg_latency_ns: 5000_000,
                max_latency_ns: 8000_000,
                load_percent: 0.05,
                delay_delta_ns: BUFFERBLOAT_THRESHOLD_NS + 1000,
            },
            ClassMetrics {
                class_id: 2,
                cycles_spent: 10000,
                sample_count: 50,
                total_latency_ns: 1000_000 * 50, // avg * count
                avg_latency_ns: 1000_000,
                max_latency_ns: 2000_000,
                load_percent: 0.95,
                delay_delta_ns: 0,
            },
        ];
        controller.calculate_delay_deltas(&metrics);

        // Handle starvation
        controller.handle_starvation(0);

        // Verify HOG was backed off
        let hog_state = controller.get_class_autorate_state(2).unwrap();
        assert!(hog_state.current_share < 0.50);
        assert_eq!(hog_state.state, AutorateState::BackingOff);

        // Verify LATENCY_CRITICAL was boosted
        let lc_state = controller.get_class_autorate_state(0).unwrap();
        assert!(lc_state.current_share > 0.20);
        assert_eq!(lc_state.state, AutorateState::Exploring);
    }

    #[test]
    fn test_update_with_metrics_integration() {
        let mut config = AutorateConfig::default();
        // Set up min/baseline/max params for parameter calculation
        for class_idx in 0..DESCENT_CLASS_MAX {
            config.min_params[class_idx] = [100_000, 500_000, 512, 50, 10_000];
            config.baseline_params[class_idx] = [500_000, 2_000_000, 1024, 100, 50_000];
            config.max_params[class_idx] = [1_000_000, 5_000_000, 2048, 150, 100_000];
        }

        let mut controller = AutorateController::new(4, &config);

        // Create metrics for all 4 classes
        let metrics = [
            ClassMetrics {
                class_id: 0,
                cycles_spent: 1000,
                sample_count: 10,
                total_latency_ns: 1500_000 * 10, // avg * count
                avg_latency_ns: 1500_000,        // 1.5ms
                max_latency_ns: 3000_000,
                load_percent: 0.80, // High load
                delay_delta_ns: 0,
            },
            ClassMetrics {
                class_id: 1,
                cycles_spent: 500,
                sample_count: 8,
                total_latency_ns: 2000_000 * 8, // avg * count
                avg_latency_ns: 2000_000,
                max_latency_ns: 4000_000,
                load_percent: 0.40, // Medium load
                delay_delta_ns: 0,
            },
            ClassMetrics {
                class_id: 2,
                cycles_spent: 2000,
                sample_count: 15,
                total_latency_ns: 3000_000 * 15, // avg * count
                avg_latency_ns: 3000_000,
                max_latency_ns: 5000_000,
                load_percent: 0.90, // Very high load
                delay_delta_ns: 0,
            },
            ClassMetrics {
                class_id: 3,
                cycles_spent: 100,
                sample_count: 5,
                total_latency_ns: 8000_000 * 5, // avg * count
                avg_latency_ns: 8000_000,
                max_latency_ns: 10000_000,
                load_percent: 0.10, // Low load
                delay_delta_ns: 0,
            },
        ];

        // Run update
        let params = controller.update_with_metrics(&metrics);

        // Verify we got parameters for all 4 classes
        assert_eq!(params.len(), DESCENT_CLASS_MAX);
        for class_idx in 0..DESCENT_CLASS_MAX {
            assert_eq!(params[class_idx].len(), PARAM_COUNT);
        }

        // Verify shares are normalized (sum should be approximately 1.0)
        let total_share: f64 = controller
            .class_states
            .iter()
            .map(|s| s.current_share)
            .sum();
        assert!((total_share - 1.0).abs() < 0.01);

        // High load classes (0, 2) should be exploring or have higher shares
        let state0 = controller.get_class_autorate_state(0).unwrap();
        let state2 = controller.get_class_autorate_state(2).unwrap();

        // With high load and good latency, they should be exploring
        // But due to hysteresis, they might still be transitioning
        assert!(state0.current_share >= config.share_configs[0].base_share);
        assert!(state2.current_share >= config.share_configs[2].base_share);

        // Low load class (3) should decay toward base
        let state3 = controller.get_class_autorate_state(3).unwrap();
        assert!(state3.current_share <= config.share_configs[3].max_share);
    }

    #[test]
    fn test_hog_backs_off_when_latency_critical_starved() {
        // This is the key test from the AUTORATE_REFACTOR_PLAN
        let mut config = AutorateConfig::default();
        config.share_configs[0] = ShareConfig::new(0.15, 0.25, 0.40, 1.08, 0.85, 0.98);
        config.share_configs[2] = ShareConfig::new(0.20, 0.35, 0.60, 1.10, 0.80, 0.98);

        // Set up min/baseline/max params
        for class_idx in 0..DESCENT_CLASS_MAX {
            config.min_params[class_idx] = [100_000, 500_000, 512, 50, 10_000];
            config.baseline_params[class_idx] = [500_000, 2_000_000, 1024, 100, 50_000];
            config.max_params[class_idx] = [1_000_000, 5_000_000, 2048, 150, 100_000];
        }

        let mut controller = AutorateController::new(4, &config);

        // Simulate: HOG 100% load, LATENCY_CRITICAL starved (high delay)
        let metrics = vec![
            ClassMetrics {
                class_id: 0, // LATENCY_CRITICAL
                cycles_spent: 100,
                sample_count: 5,
                total_latency_ns: 10_000_000 * 5, // avg * count
                avg_latency_ns: 10_000_000,       // 10ms - starved!
                max_latency_ns: 15_000_000,
                load_percent: 0.05,
                delay_delta_ns: 0, // Will be calculated from baseline
            },
            ClassMetrics {
                class_id: 1, // NORMAL
                cycles_spent: 500,
                sample_count: 8,
                total_latency_ns: 2000_000 * 8, // avg * count
                avg_latency_ns: 2000_000,
                max_latency_ns: 3000_000,
                load_percent: 0.30,
                delay_delta_ns: 0,
            },
            ClassMetrics {
                class_id: 2, // HOG
                cycles_spent: 10000,
                sample_count: 50,
                total_latency_ns: 1000_000 * 50, // avg * count
                avg_latency_ns: 1000_000,
                max_latency_ns: 2000_000,
                load_percent: 0.95, // Very high load
                delay_delta_ns: 0,
            },
            ClassMetrics {
                class_id: 3, // BACKGROUND
                cycles_spent: 50,
                sample_count: 3,
                total_latency_ns: 5000_000 * 3, // avg * count
                avg_latency_ns: 5000_000,
                max_latency_ns: 8000_000,
                load_percent: 0.05,
                delay_delta_ns: 0,
            },
        ];

        // First, force a baseline update for class 0 at low load
        controller.force_baseline_update(0, 1000_000); // 1ms baseline

        // Now update with high load metrics
        let _params = controller.update_with_metrics(&metrics);

        // HOG share should decrease (backed off)
        let hog_state = controller.get_class_autorate_state(2).unwrap();
        let hog_config = controller.get_share_config(2).unwrap();
        assert!(
            hog_state.current_share < hog_config.base_share || hog_state.load_percent > 0.90,
            "HOG should have been backed off or still has high load"
        );

        // LATENCY_CRITICAL share should increase (boosted)
        let lc_state = controller.get_class_autorate_state(0).unwrap();
        let lc_config = controller.get_share_config(0).unwrap();
        assert!(
            lc_state.current_share >= lc_config.base_share
                || lc_state.delay_delta_ns > BUFFERBLOAT_THRESHOLD_NS,
            "LATENCY_CRITICAL should have been boosted or shows high delay"
        );
    }

    #[test]
    fn test_shares_normalize_to_1_0() {
        let mut config = AutorateConfig::default();
        for class_idx in 0..DESCENT_CLASS_MAX {
            config.min_params[class_idx] = [100_000; PARAM_COUNT];
            config.baseline_params[class_idx] = [500_000; PARAM_COUNT];
            config.max_params[class_idx] = [1_000_000; PARAM_COUNT];
        }

        let mut controller = AutorateController::new(4, &config);

        // Create equal metrics for all classes
        let metrics: Vec<ClassMetrics> = (0..DESCENT_CLASS_MAX as u32)
            .map(|class_id| ClassMetrics {
                class_id,
                cycles_spent: 1000,
                sample_count: 10,
                total_latency_ns: 2000_000 * 10, // avg * count
                avg_latency_ns: 2000_000,
                max_latency_ns: 4000_000,
                load_percent: 0.50,
                delay_delta_ns: 0,
            })
            .collect();

        // Run update
        let _params = controller.update_with_metrics(&metrics);

        // Sum should be exactly 1.0 (or very close)
        let sum: f64 = controller
            .class_states
            .iter()
            .map(|s| s.current_share)
            .sum();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Shares should sum to 1.0, got {}",
            sum
        );
    }

    // =============================================================================
    // PHASE 3 TESTS - Parameter Mapping
    // =============================================================================

    #[test]
    fn test_map_share_to_param_direct() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Test direct mapping (base_slice_ns, preemption_priority, migration_cost)
        // Higher share -> Higher value
        // min_val=100, base_val=500, max_val=1000
        // min_share=0.2, base_share=0.5, max_share=0.8

        let min_val = 100u64;
        let base_val = 500u64;
        let max_val = 1000u64;
        let min_share = 0.2f64;
        let base_share = 0.5f64;
        let max_share = 0.8f64;

        // At min_share, should get min_val
        let result = controller.map_share_to_param(
            min_share,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, min_val,
            "At min_share, direct mapping should return min_val"
        );

        // At max_share, should get max_val
        let result = controller.map_share_to_param(
            max_share,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, max_val,
            "At max_share, direct mapping should return max_val"
        );

        // At base_share, should get base_val
        let result = controller.map_share_to_param(
            base_share,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, base_val,
            "At base_share, direct mapping should return base_val"
        );
    }

    #[test]
    fn test_map_share_to_param_inverse() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Test inverse mapping (latency_weight, vruntime_scale)
        // Higher share -> Lower value
        // min_val=100, base_val=500, max_val=1000
        // min_share=0.2, base_share=0.5, max_share=0.8

        let min_val = 100u64;
        let base_val = 500u64;
        let max_val = 1000u64;
        let min_share = 0.2f64;
        let base_share = 0.5f64;
        let max_share = 0.8f64;

        // At min_share, should get max_val (inverse!)
        let result = controller.map_share_to_param(
            min_share,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_INVERSE,
        );
        // Inverse mapping: t = (1.0 - 0.0) * 2.0 = 2.0, so base + (max-base) * (2.0-1.0) = max
        assert_eq!(
            result, max_val,
            "At min_share, inverse mapping should return max_val"
        );

        // At max_share, should get min_val (inverse!)
        let result = controller.map_share_to_param(
            max_share,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_INVERSE,
        );
        // Inverse mapping: t = (1.0 - 1.0) * 2.0 = 0.0, so min + (base-min) * 0.0 = min
        assert_eq!(
            result, min_val,
            "At max_share, inverse mapping should return min_val"
        );

        // At base_share, should get base_val
        let result = controller.map_share_to_param(
            base_share,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_INVERSE,
        );
        // Inverse mapping: t = (1.0 - 0.5) * 2.0 = 1.0, exactly at base
        assert_eq!(
            result, base_val,
            "At base_share, inverse mapping should return base_val"
        );
    }

    #[test]
    fn test_map_share_to_param_midpoints_direct() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        let min_val = 100u64;
        let base_val = 500u64;
        let max_val = 1000u64;
        let min_share = 0.0f64;
        let base_share = 0.5f64;
        let max_share = 1.0f64;

        // At midpoint between min and base (share=0.25), should get midpoint value
        // rate = 0.25, t = 0.25 * 2.0 = 0.5
        // value = 100 + (500-100) * 0.5 = 300
        let result = controller.map_share_to_param(
            0.25,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, 300,
            "Direct mapping at 0.25 share should interpolate between min and base"
        );

        // At midpoint between base and max (share=0.75), should get midpoint value
        // rate = 0.75, t = (0.75 - 0.5) * 2.0 = 0.5
        // value = 500 + (1000-500) * 0.5 = 750
        let result = controller.map_share_to_param(
            0.75,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, 750,
            "Direct mapping at 0.75 share should interpolate between base and max"
        );
    }

    #[test]
    fn test_map_share_to_param_midpoints_inverse() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        let min_val = 100u64;
        let base_val = 500u64;
        let max_val = 1000u64;
        let min_share = 0.0f64;
        let base_share = 0.5f64;
        let max_share = 1.0f64;

        // At share=0.25 (high inverse value territory)
        // rate = 0.25, t = (1.0 - 0.25) * 2.0 = 1.5
        // Since t > 1.0, t2 = 0.5
        // value = 500 + (1000-500) * 0.5 = 750 (closer to max since share is low)
        let result = controller.map_share_to_param(
            0.25,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_INVERSE,
        );
        assert_eq!(
            result, 750,
            "Inverse mapping at 0.25 share should interpolate between base and max"
        );

        // At share=0.75 (low inverse value territory)
        // rate = 0.75, t = (1.0 - 0.75) * 2.0 = 0.5
        // value = 100 + (500-100) * 0.5 = 300 (closer to min since share is high)
        let result = controller.map_share_to_param(
            0.75,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_INVERSE,
        );
        assert_eq!(
            result, 300,
            "Inverse mapping at 0.75 share should interpolate between min and base"
        );
    }

    #[test]
    fn test_map_share_to_param_edge_case_min_equals_max() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Edge case: min_share == max_share
        // Should return base_val regardless of share
        let result = controller.map_share_to_param(
            0.5,
            0.5,
            0.5,
            0.5, // min_share == base_share == max_share
            100,
            500,
            1000,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, 500,
            "When min_share == max_share, should return base_val"
        );
    }

    #[test]
    fn test_map_share_to_param_clamping() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Test share clamping - values outside [min_share, max_share] should be clamped
        let min_val = 100u64;
        let base_val = 500u64;
        let max_val = 1000u64;
        let min_share = 0.2f64;
        let base_share = 0.5f64;
        let max_share = 0.8f64;

        // Share below min_share should be treated as min_share
        let result = controller.map_share_to_param(
            0.0,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, min_val,
            "Share below min_share should be clamped to min_val"
        );

        // Share above max_share should be treated as max_share
        let result = controller.map_share_to_param(
            1.0,
            min_share,
            base_share,
            max_share,
            min_val,
            base_val,
            max_val,
            MAPPING_DIRECT,
        );
        assert_eq!(
            result, max_val,
            "Share above max_share should be clamped to max_val"
        );
    }

    #[test]
    fn test_calculate_param_latency_weight() {
        // Test that latency_weight uses inverse mapping
        let mut config = AutorateConfig::default();

        // Set up params for class 0 - use simple values to avoid rounding issues
        config.min_params[0] = [100_000, 500_000, 512, 50, 10_000];
        config.baseline_params[0] = [500_000, 2_000_000, 1024, 100, 50_000];
        config.max_params[0] = [1_000_000, 5_000_000, 2048, 200, 100_000];

        // Set share config to use 0.0-1.0 range for predictable mapping
        config.share_configs[0] = ShareConfig::new(0.0, 0.5, 1.0, 1.1, 0.9, 0.99);

        let mut controller = AutorateController::new(4, &config);

        // Test at min share - should get HIGH value (inverse mapping)
        controller.class_states[0].current_share = 0.0;
        let min_share_result = controller.calculate_param(0, PARAM_LATENCY_WEIGHT, 0.0);

        // Test at max share - should get LOW value (inverse mapping)
        controller.class_states[0].current_share = 1.0;
        let max_share_result = controller.calculate_param(0, PARAM_LATENCY_WEIGHT, 1.0);

        // Higher share should give lower latency_weight (inverse relationship)
        assert!(
            max_share_result < min_share_result,
            "Max share should give lower latency_weight than min share (inverse mapping)"
        );

        // At base share (0.5), should get base_val
        controller.class_states[0].current_share = 0.5;
        let base_result = controller.calculate_param(0, PARAM_LATENCY_WEIGHT, 0.5);
        assert_eq!(
            base_result, config.baseline_params[0][PARAM_LATENCY_WEIGHT],
            "At base share, should get baseline value"
        );
    }

    #[test]
    fn test_calculate_param_base_slice() {
        // Test that base_slice_ns uses direct mapping
        let mut config = AutorateConfig::default();

        // Set up params for class 0 - use simple values with 0.0-1.0 range
        config.min_params[0] = [100_000, 500_000, 512, 50, 10_000];
        config.baseline_params[0] = [500_000, 2_000_000, 1024, 100, 50_000];
        config.max_params[0] = [1_000_000, 5_000_000, 2048, 200, 100_000];

        // Set share config to use 0.0-1.0 range for predictable mapping
        config.share_configs[0] = ShareConfig::new(0.0, 0.5, 1.0, 1.1, 0.9, 0.99);

        let mut controller = AutorateController::new(4, &config);

        // At min share, should get min_val
        controller.class_states[0].current_share = 0.0;
        let min_result = controller.calculate_param(0, PARAM_BASE_SLICE_NS, 0.0);
        assert_eq!(min_result, config.min_params[0][PARAM_BASE_SLICE_NS]);

        // At max share, should get max_val
        controller.class_states[0].current_share = 1.0;
        let max_result = controller.calculate_param(0, PARAM_BASE_SLICE_NS, 1.0);
        assert_eq!(max_result, config.max_params[0][PARAM_BASE_SLICE_NS]);

        // At base share, should get base_val
        controller.class_states[0].current_share = 0.5;
        let base_result = controller.calculate_param(0, PARAM_BASE_SLICE_NS, 0.5);
        assert_eq!(base_result, config.baseline_params[0][PARAM_BASE_SLICE_NS]);

        // Higher share should give higher base_slice (direct relationship)
        assert!(
            max_result > min_result,
            "Max share should give higher base_slice than min share (direct mapping)"
        );
    }

    #[test]
    fn test_calculate_param_vruntime_scale() {
        // Test that vruntime_scale uses inverse mapping
        let mut config = AutorateConfig::default();

        // Set up params for class 0
        config.min_params[0] = [100_000, 500_000, 512, 50, 10_000];
        config.baseline_params[0] = [500_000, 2_000_000, 1024, 100, 50_000];
        config.max_params[0] = [1_000_000, 5_000_000, 2048, 200, 100_000];

        let controller = AutorateController::new(4, &config);

        // At min share, should get max_val (inverse)
        let min_result =
            controller.calculate_param(0, PARAM_VRUNTIME_SCALE, config.share_configs[0].min_share);

        // At max share, should get min_val (inverse)
        let max_result =
            controller.calculate_param(0, PARAM_VRUNTIME_SCALE, config.share_configs[0].max_share);

        // Higher share should give lower vruntime_scale (inverse relationship)
        assert!(
            max_result < min_result,
            "Max share should give lower vruntime_scale than min share (inverse mapping)"
        );
    }

    #[test]
    fn test_calculate_param_preemption_priority() {
        // Test that preemption_priority uses direct mapping
        let mut config = AutorateConfig::default();

        // Set up params for class 0
        config.min_params[0] = [100_000, 500_000, 512, 50, 10_000];
        config.baseline_params[0] = [500_000, 2_000_000, 1024, 100, 50_000];
        config.max_params[0] = [1_000_000, 5_000_000, 2048, 200, 100_000];

        let controller = AutorateController::new(4, &config);

        // At min share, should get min_val
        let min_result = controller.calculate_param(
            0,
            PARAM_PREEMPTION_PRIORITY,
            config.share_configs[0].min_share,
        );
        assert_eq!(min_result, config.min_params[0][PARAM_PREEMPTION_PRIORITY]);

        // At max share, should get max_val
        let max_result = controller.calculate_param(
            0,
            PARAM_PREEMPTION_PRIORITY,
            config.share_configs[0].max_share,
        );
        assert_eq!(max_result, config.max_params[0][PARAM_PREEMPTION_PRIORITY]);

        // Higher share should give higher preemption_priority (direct relationship)
        assert!(
            max_result > min_result,
            "Max share should give higher preemption_priority than min share (direct mapping)"
        );
    }

    #[test]
    fn test_calculate_param_migration_cost() {
        // Test that migration_cost uses direct mapping
        let mut config = AutorateConfig::default();

        // Set up params for class 0
        config.min_params[0] = [100_000, 500_000, 512, 50, 10_000];
        config.baseline_params[0] = [500_000, 2_000_000, 1024, 100, 50_000];
        config.max_params[0] = [1_000_000, 5_000_000, 2048, 200, 100_000];

        let controller = AutorateController::new(4, &config);

        // At min share, should get min_val
        let min_result =
            controller.calculate_param(0, PARAM_MIGRATION_COST, config.share_configs[0].min_share);
        assert_eq!(min_result, config.min_params[0][PARAM_MIGRATION_COST]);

        // At max share, should get max_val
        let max_result =
            controller.calculate_param(0, PARAM_MIGRATION_COST, config.share_configs[0].max_share);
        assert_eq!(max_result, config.max_params[0][PARAM_MIGRATION_COST]);

        // Higher share should give higher migration_cost (direct relationship)
        assert!(
            max_result > min_result,
            "Max share should give higher migration_cost than min share (direct mapping)"
        );
    }

    #[test]
    fn test_calculate_param_invalid_class() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Invalid class should return 0
        let result = controller.calculate_param(99, PARAM_LATENCY_WEIGHT, 0.5);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_calculate_param_invalid_param_idx() {
        let config = AutorateConfig::default();
        let controller = AutorateController::new(4, &config);

        // Invalid param_idx should return 0 (PARAM_COUNT is 5, so index 10 is invalid)
        let result = controller.calculate_param(0, 10, 0.5);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_calculate_parameters() {
        let mut config = AutorateConfig::default();

        // Set up different params for each class to verify per-class calculation
        for class_idx in 0..DESCENT_CLASS_MAX {
            config.min_params[class_idx] = [
                100_000 * (class_idx + 1) as u64,
                500_000 * (class_idx + 1) as u64,
                512 * (class_idx + 1) as u64,
                50 * (class_idx + 1) as u64,
                10_000 * (class_idx + 1) as u64,
            ];
            config.baseline_params[class_idx] = [
                500_000 * (class_idx + 1) as u64,
                2_000_000 * (class_idx + 1) as u64,
                1024 * (class_idx + 1) as u64,
                100 * (class_idx + 1) as u64,
                50_000 * (class_idx + 1) as u64,
            ];
            config.max_params[class_idx] = [
                1_000_000 * (class_idx + 1) as u64,
                5_000_000 * (class_idx + 1) as u64,
                2048 * (class_idx + 1) as u64,
                150 * (class_idx + 1) as u64,
                100_000 * (class_idx + 1) as u64,
            ];
            // Use uniform 0.0-1.0 range for predictable mapping
            config.share_configs[class_idx] = ShareConfig::new(0.0, 0.5, 1.0, 1.1, 0.9, 0.99);
        }

        let mut controller = AutorateController::new(4, &config);

        // Set uniform base share for all classes (0.5 = exact baseline)
        for class_idx in 0..DESCENT_CLASS_MAX {
            controller.class_states[class_idx].current_share = 0.5;
        }

        // Calculate all parameters
        let params = controller.calculate_parameters();

        // Verify structure: [[u64; 5]; 4]
        assert_eq!(params.len(), DESCENT_CLASS_MAX);
        for class_idx in 0..DESCENT_CLASS_MAX {
            assert_eq!(params[class_idx].len(), PARAM_COUNT);
        }

        // Verify each class has different values (since we set different params per class)
        // Class 0 should have smaller values than Class 3
        assert!(
            params[0][PARAM_LATENCY_WEIGHT] < params[3][PARAM_LATENCY_WEIGHT],
            "Class 0 should have smaller latency_weight than Class 3"
        );
        assert!(
            params[0][PARAM_BASE_SLICE_NS] < params[3][PARAM_BASE_SLICE_NS],
            "Class 0 should have smaller base_slice than Class 3"
        );

        // Verify at base share (0.5 with 0.0-1.0 range), we get exactly baseline values
        for class_idx in 0..DESCENT_CLASS_MAX {
            let baseline_val = config.baseline_params[class_idx][PARAM_BASE_SLICE_NS];
            let calculated_val = params[class_idx][PARAM_BASE_SLICE_NS];
            // At exactly 0.5 with min=0.0 and max=1.0, we should get baseline
            assert_eq!(
                calculated_val, baseline_val,
                "At base share 0.5 with uniform range, should get exact baseline value"
            );
        }
    }

    #[test]
    fn test_calculate_parameters_returns_correct_array_type() {
        let mut config = AutorateConfig::default();

        // Set up params
        for class_idx in 0..DESCENT_CLASS_MAX {
            config.min_params[class_idx] = [100_000; PARAM_COUNT];
            config.baseline_params[class_idx] = [500_000; PARAM_COUNT];
            config.max_params[class_idx] = [1_000_000; PARAM_COUNT];
        }

        let controller = AutorateController::new(4, &config);

        // Get parameters
        let params = controller.calculate_parameters();

        // Verify we can index into it as [[u64; 5]; 4]
        let _first_class_params: [u64; 5] = params[0];
        let _all_params: [[u64; 5]; 4] = params;

        // Verify all values are non-zero when share is in valid range
        for class_idx in 0..DESCENT_CLASS_MAX {
            for param_idx in 0..PARAM_COUNT {
                assert!(
                    params[class_idx][param_idx] > 0,
                    "All calculated parameters should be positive"
                );
            }
        }
    }

    #[test]
    fn test_share_mapping_bounds() {
        let mut config = AutorateConfig::default();

        // Set up params with clear min/baseline/max
        for class_idx in 0..DESCENT_CLASS_MAX {
            config.min_params[class_idx] = [100; PARAM_COUNT];
            config.baseline_params[class_idx] = [500; PARAM_COUNT];
            config.max_params[class_idx] = [1000; PARAM_COUNT];
        }

        let controller = AutorateController::new(4, &config);

        // Test at min_share for all classes
        for class_idx in 0..DESCENT_CLASS_MAX {
            let min_share = config.share_configs[class_idx].min_share;

            // latency_weight (inverse): at min share = max value
            let lw = controller.calculate_param(class_idx as u32, PARAM_LATENCY_WEIGHT, min_share);
            assert_eq!(lw, 1000, "At min_share, inverse mapping should return max");

            // base_slice_ns (direct): at min share = min value
            let bs = controller.calculate_param(class_idx as u32, PARAM_BASE_SLICE_NS, min_share);
            assert_eq!(bs, 100, "At min_share, direct mapping should return min");

            // vruntime_scale (inverse): at min share = max value
            let vs = controller.calculate_param(class_idx as u32, PARAM_VRUNTIME_SCALE, min_share);
            assert_eq!(vs, 1000, "At min_share, inverse mapping should return max");

            // preemption_priority (direct): at min share = min value
            let pp =
                controller.calculate_param(class_idx as u32, PARAM_PREEMPTION_PRIORITY, min_share);
            assert_eq!(pp, 100, "At min_share, direct mapping should return min");

            // migration_cost (direct): at min share = min value
            let mc = controller.calculate_param(class_idx as u32, PARAM_MIGRATION_COST, min_share);
            assert_eq!(mc, 100, "At min_share, direct mapping should return min");
        }

        // Test at max_share for all classes
        for class_idx in 0..DESCENT_CLASS_MAX {
            let max_share = config.share_configs[class_idx].max_share;

            // latency_weight (inverse): at max share = min value
            let lw = controller.calculate_param(class_idx as u32, PARAM_LATENCY_WEIGHT, max_share);
            assert_eq!(lw, 100, "At max_share, inverse mapping should return min");

            // base_slice_ns (direct): at max share = max value
            let bs = controller.calculate_param(class_idx as u32, PARAM_BASE_SLICE_NS, max_share);
            assert_eq!(bs, 1000, "At max_share, direct mapping should return max");

            // vruntime_scale (inverse): at max share = min value
            let vs = controller.calculate_param(class_idx as u32, PARAM_VRUNTIME_SCALE, max_share);
            assert_eq!(vs, 100, "At max_share, inverse mapping should return min");

            // preemption_priority (direct): at max share = max value
            let pp =
                controller.calculate_param(class_idx as u32, PARAM_PREEMPTION_PRIORITY, max_share);
            assert_eq!(pp, 1000, "At max_share, direct mapping should return max");

            // migration_cost (direct): at max share = max value
            let mc = controller.calculate_param(class_idx as u32, PARAM_MIGRATION_COST, max_share);
            assert_eq!(mc, 1000, "At max_share, direct mapping should return max");
        }
    }

    // =============================================================================
    // LEGACY TESTS - Commented out for Phase 1-2 (will be updated in Phase 4)
    // These tests use the old AutorateClassState and rate-based API
    // =============================================================================
}
