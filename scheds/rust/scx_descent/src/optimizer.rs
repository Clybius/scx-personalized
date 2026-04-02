// Phase 2: Adam optimizer implementation
// Gradient descent optimization loop for adaptive scheduling

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

use log::{debug, warn};

use crate::profiles::LossWeights;

/// Configuration for Adam optimizer
#[derive(Clone, Debug)]
pub struct AdamConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub base_learning_rate: f64,
    pub lr_adaptation_enabled: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            base_learning_rate: 0.001,
            lr_adaptation_enabled: true,
        }
    }
}

/// State for a single (CPU, class) pair's Adam optimizer
pub struct AdamState {
    /// Current parameter values (5 parameters)
    pub params: [f64; 5],

    /// First moments (momentum)
    m: [f64; 5],

    /// Second moments (variance)
    v: [f64; 5],

    /// Timestep (t)
    t: u64,

    /// Parameter bounds (min, max)
    bounds: [(f64, f64); 5],

    /// Loss history for debugging
    pub loss_history: VecDeque<f64>,

    /// Parameter history
    pub param_history: VecDeque<[f64; 5]>,

    /// Current learning rate (may be adapted)
    current_lr: f64,

    /// NEW: Profile-specific loss weights
    loss_weights: LossWeights,
}

/// Raw loss values from BPF
pub struct RawLosses {
    pub latency_sum: u64,
    pub sample_count: u32,
    pub deadline_misses: u64,
    pub cpu_time_ns: u64,
    pub target_share_ns: u64,
}

impl AdamState {
    pub fn new(initial_params: [u64; 5], weights: LossWeights) -> Self {
        let params = [
            initial_params[0] as f64,
            initial_params[1] as f64,
            initial_params[2] as f64,
            initial_params[3] as f64,
            initial_params[4] as f64,
        ];

        Self {
            params,
            m: [0.0; 5],
            v: [0.0; 5],
            t: 0,
            bounds: [
                (0.0, 10e6),     // latency_weight: 0-10ms
                (100e3, 50e6),   // base_slice_ns: 100us-50ms
                (512.0, 2048.0), // vruntime_scale: 0.5-2.0 (*1024)
                (0.0, 100.0),    // preemption_priority: 0-100
                (0.0, 10e6),     // migration_cost: 0-10ms
            ],
            loss_history: VecDeque::with_capacity(100),
            param_history: VecDeque::with_capacity(100),
            current_lr: 0.001,
            loss_weights: weights,
        }
    }

    /// NEW: Phase 3 - Compute weighted loss from raw measurements
    pub fn compute_weighted_loss(&self, raw_losses: &RawLosses) -> f64 {
        let latency_loss = if raw_losses.sample_count > 0 {
            (raw_losses.latency_sum as f64) / (raw_losses.sample_count as f64)
        } else {
            0.0
        };

        let throughput_loss = (raw_losses.deadline_misses * 1_000_000) as f64; // Scale up

        let fairness_loss = if raw_losses.target_share_ns > 0 {
            let actual = raw_losses.cpu_time_ns as f64;
            let target = raw_losses.target_share_ns as f64;
            ((actual - target).abs() / target) * 1_000_000.0 // Scale up
        } else {
            0.0
        };

        self.loss_weights.latency * latency_loss
            + self.loss_weights.throughput * throughput_loss
            + self.loss_weights.fairness * fairness_loss
    }

    /// Compute gradient from loss measurements using central difference
    pub fn compute_gradient(&self, loss_plus: f64, loss_minus: f64, epsilon: f64) -> f64 {
        // Central difference: (L(θ+ε) - L(θ-ε)) / 2ε
        let grad = (loss_plus - loss_minus) / (2.0 * epsilon);
        debug!(
            "Computing gradient: loss_plus={}, loss_minus={}, epsilon={}, grad={}",
            loss_plus, loss_minus, epsilon, grad
        );
        grad
    }

    /// Adam update for a single parameter
    pub fn update_parameter(&mut self, param_idx: usize, gradient: f64, config: &AdamConfig) {
        self.t += 1;

        // Update biased first moment estimate
        self.m[param_idx] = config.beta1 * self.m[param_idx] + (1.0 - config.beta1) * gradient;

        // Update biased second raw moment estimate
        self.v[param_idx] =
            config.beta2 * self.v[param_idx] + (1.0 - config.beta2) * gradient * gradient;

        // Compute bias-corrected first moment
        let m_hat = self.m[param_idx] / (1.0 - config.beta1.powi(self.t as i32));

        // Compute bias-corrected second moment
        let v_hat = self.v[param_idx] / (1.0 - config.beta2.powi(self.t as i32));

        // Compute update
        let update = self.current_lr * m_hat / (v_hat.sqrt() + config.epsilon);

        // Apply update
        self.params[param_idx] -= update;

        // Clamp to bounds
        let (min, max) = self.bounds[param_idx];
        self.params[param_idx] = self.params[param_idx].clamp(min, max);

        debug!(
            "Updated param[{}]: value={}, grad={}, update={}",
            param_idx, self.params[param_idx], gradient, update
        );
    }

    /// Get parameters as fixed-point u64 for BPF
    pub fn get_params_fixed(&self) -> [u64; 5] {
        [
            self.params[0] as u64,
            self.params[1] as u64,
            self.params[2] as u64,
            self.params[3] as u64,
            self.params[4] as u64,
        ]
    }

    /// Reduce learning rate (called when oscillation detected)
    pub fn reduce_learning_rate(&mut self) {
        self.current_lr *= 0.5;
        warn!("Reduced learning rate to {}", self.current_lr);
    }

    /// Reset Adam state
    pub fn reset(&mut self) {
        self.t = 0;
        self.m = [0.0; 5];
        self.v = [0.0; 5];
        self.current_lr = 0.001;
    }
}

/// Oscillation detector for safety
pub struct OscillationDetector {
    history: VecDeque<f64>,
    threshold: f64,
}

impl OscillationDetector {
    pub fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(20),
            threshold: 0.6, // 60% sign changes = oscillation
        }
    }

    pub fn add(&mut self, value: f64) -> bool {
        self.history.push_back(value);
        if self.history.len() > 20 {
            self.history.pop_front();
        }

        self.detect()
    }

    fn detect(&self) -> bool {
        if self.history.len() < 10 {
            return false;
        }

        let mut sign_changes = 0;
        let values: Vec<_> = self.history.iter().collect();

        for i in 1..values.len() {
            let prev = *values[i - 1];
            let curr = *values[i];

            if (prev > 0.0 && curr < 0.0) || (prev < 0.0 && curr > 0.0) {
                sign_changes += 1;
            }
        }

        let ratio = sign_changes as f64 / (values.len() - 1) as f64;
        ratio > self.threshold
    }
}

/// Checkpoint for rollback
pub struct Checkpoint {
    params: [f64; 5],
    loss: f64,
    timestamp: Instant,
}

/// Main optimizer struct
pub struct DescentOptimizer {
    /// Per-CPU, per-class optimizer state
    states: HashMap<(u32, u32), AdamState>, // (cpu, class) -> state

    /// Hyperparameters
    config: AdamConfig,

    /// Safety monitoring
    oscillation_detectors: HashMap<(u32, u32), OscillationDetector>,
    checkpoints: HashMap<(u32, u32), Checkpoint>,
}

impl DescentOptimizer {
    pub fn new() -> Self {
        Self {
            states: HashMap::new(),
            config: AdamConfig::default(),
            oscillation_detectors: HashMap::new(),
            checkpoints: HashMap::new(),
        }
    }

    pub fn with_config(config: AdamConfig) -> Self {
        Self {
            states: HashMap::new(),
            config,
            oscillation_detectors: HashMap::new(),
            checkpoints: HashMap::new(),
        }
    }

    /// Initialize optimizer state for a CPU/class with loss weights
    pub fn init_state(
        &mut self,
        cpu: u32,
        class: u32,
        initial_params: [u64; 5],
        weights: LossWeights,
    ) {
        let state = AdamState::new(initial_params, weights);
        self.states.insert((cpu, class), state);
        self.oscillation_detectors
            .insert((cpu, class), OscillationDetector::new());
        debug!(
            "Initialized optimizer state for CPU {} class {} with params {:?}",
            cpu, class, initial_params
        );
    }

    /// NEW: Phase 3 - Initialize with weights from profile
    pub fn init_state_with_weights(
        &mut self,
        cpu: u32,
        class: u32,
        initial_params: [u64; 5],
        weights: LossWeights,
    ) {
        self.init_state(cpu, class, initial_params, weights);
    }

    /// Process a gradient event from BPF
    pub fn process_gradient_event(&mut self, event: &GradientEvent) -> Option<[u64; 5]> {
        let key = (event.cpu_id as u32, event.class_id);

        let state = self.states.get_mut(&key)?;
        let detector = self.oscillation_detectors.get_mut(&key)?;

        // Compute gradient for the perturbed parameter
        let gradient = state.compute_gradient(
            event.loss_plus as f64,
            event.loss_minus as f64,
            event.epsilon as f64,
        );

        // Update the parameter
        state.update_parameter(event.param_idx as usize, gradient, &self.config);

        // Track history
        let avg_loss = (event.loss_plus + event.loss_minus) as f64 / 2.0;
        state.loss_history.push_back(avg_loss);
        if state.loss_history.len() > 100 {
            state.loss_history.pop_front();
        }

        let params_copy = state.params;
        state.param_history.push_back(params_copy);
        if state.param_history.len() > 100 {
            state.param_history.pop_front();
        }

        // Check for oscillation
        /*
        let is_oscillating = detector.add(gradient);
        if is_oscillating && self.config.lr_adaptation_enabled {
            warn!(
                "Oscillation detected on CPU {} class {}, reducing LR",
                event.cpu_id, event.class_id
            );
            state.reduce_learning_rate();
        }
        */

        // Return updated parameters
        Some(state.get_params_fixed())
    }

    /// Create checkpoint for rollback
    pub fn create_checkpoint(&mut self, cpu: u32, class: u32) {
        if let Some(state) = self.states.get(&(cpu, class)) {
            let checkpoint = Checkpoint {
                params: state.params,
                loss: *state.loss_history.back().unwrap_or(&0.0),
                timestamp: Instant::now(),
            };
            self.checkpoints.insert((cpu, class), checkpoint);
        }
    }

    /// Rollback to checkpoint
    pub fn rollback(&mut self, cpu: u32, class: u32) -> Option<[u64; 5]> {
        let checkpoint = self.checkpoints.get(&(cpu, class))?;
        let state = self.states.get_mut(&(cpu, class))?;

        state.params = checkpoint.params;
        state.reset();

        warn!("Rolled back CPU {} class {} to checkpoint", cpu, class);

        Some(state.get_params_fixed())
    }

    /// Get current parameters for a CPU/class
    pub fn get_params(&self, cpu: u32, class: u32) -> Option<[u64; 5]> {
        self.states.get(&(cpu, class)).map(|s| s.get_params_fixed())
    }

    /// NEW: Phase 3 - Get state reference for monitoring
    pub fn get_state(&self, cpu: u32, class: u32) -> Option<&AdamState> {
        self.states.get(&(cpu, class))
    }

    /// NEW: Phase 3 - Get checkpoint loss for rollback decision
    pub fn get_checkpoint_loss(&self, cpu: u32, class: u32) -> Option<f64> {
        self.checkpoints.get(&(cpu, class)).map(|c| c.loss)
    }

    /// Get all states (for debugging)
    pub fn get_states(&self) -> &HashMap<(u32, u32), AdamState> {
        &self.states
    }
}

/// Event structure matching BPF ring buffer
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GradientEvent {
    pub cpu_id: i32,
    pub class_id: u32,
    pub param_idx: i32,
    pub loss_plus: u64,
    pub loss_minus: u64,
    pub epsilon: u64,
    pub timestamp: u64,
}
