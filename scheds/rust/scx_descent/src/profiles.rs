//! Profile System for scx_descent
//!
//! Defines three scheduling profiles optimized for different workloads:
//! - Gaming: Fast response (10ms), aggressive PIE tuning (alpha=4, beta=2)
//! - Production: Balanced (20ms), standard PIE tuning (alpha=8, beta=4)
//! - Server: Conservative (50ms), gentle PIE tuning (alpha=16, beta=8)
//!
//! Each profile specifies:
//! - Parameter bounds (min/max per class)
//! - Default parameter values
//! - Response interval (update frequency)
//! - PIE controller parameters (alpha, beta, max_integral)
//! - Target latencies per task class

// Phase 2: PIE Controller Integration
// Gaming, production, server profiles with PIE configuration and target latencies

use std::fmt;
use std::str::FromStr;

use crate::autorate::AutorateConfig;

/// Class indices matching bpf/intf.h
#[allow(dead_code)] // Part of public API for future task classification
pub const DESCENT_CLASS_LATENCY_CRITICAL: usize = 0; // Games, audio, compositors, kthreads
#[allow(dead_code)] // Part of public API for future task classification
pub const DESCENT_CLASS_NORMAL: usize = 1; // Default interactive
#[allow(dead_code)] // Part of public API for future task classification
pub const DESCENT_CLASS_HOG: usize = 2; // High CPU usage
#[allow(dead_code)] // Part of public API for future task classification
pub const DESCENT_CLASS_BACKGROUND: usize = 3; // Low priority
#[allow(dead_code)] // Part of public API for future task classification
pub const DESCENT_CLASS_MAX: usize = 4;

/// Parameter indices
#[allow(dead_code)] // Part of public API for parameter access
pub const PARAM_COUNT: usize = 5;

/// Parameter bounds (min, max) per class per parameter
/// [class][param] = (min, max)
/// Classes: [LATENCY_CRITICAL, NORMAL, HOG, BACKGROUND]
/// Params:  [latency_weight, base_slice_ns, vruntime_scale, preemption_priority, migration_cost]
pub type ParamBounds = [[(u64, u64); 5]; 4];

/// Default parameters per class [LATENCY_CRITICAL, NORMAL, HOG, BACKGROUND]
/// [class][param] = default_value
pub type DefaultParams = [[u64; 5]; 4];

pub struct Profile {
    pub name: String,
    /// Phase 2: PIE controller integration
    pub default_params: DefaultParams,
    pub bounds: ParamBounds,
    /// Update interval (ms): 10 (gaming), 20 (productivity), 50 (server)
    pub response_ms: u64,
    /// PIE alpha (proportional gain divisor, default: 8)
    /// Gaming: 4 (more aggressive), Prod: 8 (balanced), Server: 16 (conservative)
    pub pie_alpha: u64,
    /// PIE beta (integral gain divisor, default: 4)
    /// Gaming: 2, Prod: 4, Server: 8
    pub pie_beta: u64,
    /// Maximum integral accumulator (prevents windup)
    pub pie_max_integral: i64,
    /// Target latencies per class (ns)
    /// [LATENCY_CRITICAL, NORMAL, HOG, BACKGROUND]
    pub target_latencies_ns: [u64; 4],
    /// NEW: CAKE Autorate configuration
    pub autorate: AutorateConfig,
}

impl Profile {
    /// Gaming profile - Fast response, aggressive PIE, tight latency targets
    pub fn gaming() -> Self {
        Self {
            name: "gaming".to_string(),
            // Gaming: aggressive defaults favoring latency
            default_params: [
                // LATENCY_CRITICAL: [latency_weight, base_slice_ns, vruntime_scale, preemption_priority, migration_cost]
                [100_000, 600_000, 512, 50, 10_000],
                // NORMAL
                [1_000_000, 1_000_000, 768, 100, 30_000],
                // HOG
                [5_000_000, 5_000_000, 1536, 100, 100_000],
                // BACKGROUND
                [1_000_000, 2_000_000, 1024, 100, 20_000],
            ],
            // Gaming: wide bounds matching server profile for stability
            bounds: [
                // LATENCY_CRITICAL bounds - wide for PIE freedom
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns (wide: 1-10ms)
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority (fixed for stability)
                    (50_000, 500_000),       // migration_cost
                ],
                // NORMAL bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority
                    (50_000, 500_000),       // migration_cost
                ],
                // HOG bounds - wide to prevent PIE getting stuck
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns (wide for adaptive tuning)
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority
                    (50_000, 500_000),       // migration_cost
                ],
                // BACKGROUND bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority
                    (50_000, 500_000),       // migration_cost
                ],
            ],
            response_ms: 20, // Balanced response (was 10ms - too aggressive)
            pie_alpha: 8,    // Balanced proportional gain (was 4 - too aggressive)
            pie_beta: 4,     // Balanced integral response (was 2 - too aggressive)
            pie_max_integral: 1_000_000,
            target_latencies_ns: [
                500_000,    // LATENCY_CRITICAL: 500 µs (keep tight for audio/games)
                2_000_000,  // NORMAL: 2 ms
                10_000_000, // HOG: 10 ms
                50_000_000, // BACKGROUND: 50 ms
            ],
            autorate: AutorateConfig {
                enabled: false, // Default: disabled (opt-in via --autorate)

                // Per-class min/baseline/max parameters
                // Gaming: balanced ranges for stability with autorate
                min_params: [
                    // LATENCY_CRITICAL: conservative minimums
                    [50_000, 300_000, 384, 50, 5_000],
                    // NORMAL
                    [100_000, 500_000, 512, 50, 10_000],
                    // HOG: higher min to reduce CPU hogging impact
                    [1_000_000, 3_000_000, 1280, 50, 50_000],
                    // BACKGROUND
                    [100_000, 500_000, 512, 50, 10_000],
                ],
                baseline_params: [
                    // LATENCY_CRITICAL: standard gaming baseline
                    [100_000, 600_000, 512, 50, 10_000],
                    // NORMAL
                    [1_000_000, 2_000_000, 768, 100, 30_000],
                    // HOG: constrained baseline
                    [3_000_000, 5_000_000, 1536, 100, 80_000],
                    // BACKGROUND
                    [1_000_000, 2_000_000, 1024, 100, 20_000],
                ],
                max_params: [
                    // LATENCY_CRITICAL: capped for stability
                    [1_000_000, 1_500_000, 768, 100, 200_000],
                    // NORMAL - constrained
                    [3_000_000, 5_000_000, 1024, 100, 300_000],
                    // HOG - strictly capped to prevent monopolization
                    [5_000_000, 6_000_000, 1792, 100, 400_000],
                    // BACKGROUND - constrained
                    [3_000_000, 4_000_000, 1280, 100, 200_000],
                ],

                // Balanced gaming tuning (more conservative than before)
                high_load_threshold: 0.75,
                low_load_threshold: 0.25,
                ramp_up_rate: 1.04,   // 4% increase (was 8% - too aggressive)
                ramp_down_rate: 0.85, // 15% decrease (was 25% - too aggressive)
                decay_rate: 0.99,     // 1% decay toward baseline (was 2%)
                adjust_up_refractory_ms: 100, // 100ms between upward adjustments (was 50ms)
                adjust_down_refractory_ms: 50, // 50ms between downward (was 20ms)
                bufferbloat_threshold: 1.6, // 1.6x target = bufferbloat (was 1.5x)
            },
        }
    }

    /// Production profile (DEFAULT) - Balanced response, moderate PIE, standard targets
    pub fn production() -> Self {
        Self {
            name: "production".to_string(),
            // Production: balanced defaults
            default_params: [
                // LATENCY_CRITICAL: [latency_weight, base_slice_ns, vruntime_scale, preemption_priority, migration_cost]
                [100_000, 600_000, 512, 100, 30_000],
                // NORMAL
                [500_000, 1_000_000, 768, 100, 30_000],
                // HOG
                [2_000_000, 3_000_000, 1536, 100, 50_000],
                // BACKGROUND
                [1_000_000, 2_000_000, 1024, 100, 20_000],
            ],
            // Production: wide bounds matching server for stability
            bounds: [
                // LATENCY_CRITICAL bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns (wide: 1-10ms)
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority (fixed for stability)
                    (50_000, 500_000),       // migration_cost
                ],
                // NORMAL bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority
                    (50_000, 500_000),       // migration_cost
                ],
                // HOG bounds - wide to prevent PIE getting stuck
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority
                    (50_000, 500_000),       // migration_cost
                ],
                // BACKGROUND bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority
                    (50_000, 500_000),       // migration_cost
                ],
            ],
            response_ms: 20, // Balanced response
            pie_alpha: 8,    // Balanced proportional gain
            pie_beta: 4,     // Balanced integral response
            pie_max_integral: 1_000_000,
            target_latencies_ns: [
                1_000_000,   // LATENCY_CRITICAL: 1 ms
                5_000_000,   // NORMAL: 5 ms
                20_000_000,  // HOG: 20 ms
                100_000_000, // BACKGROUND: 100 ms
            ],
            autorate: AutorateConfig {
                enabled: false,

                min_params: [
                    // More conservative than gaming
                    [100_000, 400_000, 512, 75, 15_000],
                    [200_000, 800_000, 640, 75, 20_000],
                    [750_000, 4_000_000, 1280, 75, 50_000],
                    [200_000, 800_000, 640, 75, 20_000],
                ],
                baseline_params: [
                    // Standard production baselines
                    [100_000, 600_000, 512, 100, 30_000],
                    [500_000, 1_000_000, 768, 100, 30_000],
                    [2_000_000, 5_000_000, 1536, 100, 50_000],
                    [1_000_000, 2_000_000, 1024, 100, 20_000],
                ],
                max_params: [
                    // Moderately aggressive - all within bounds
                    // Bounds: (100K,5M), (200K,5M), (512,1536), (50,150), (5K,1M)
                    [1_000_000, 4_000_000, 1280, 125, 500_000],
                    [3_000_000, 4_500_000, 1408, 135, 750_000],
                    [4_000_000, 4_800_000, 1472, 140, 900_000],
                    [3_000_000, 4_500_000, 1408, 135, 650_000],
                ],

                // Balanced tuning
                high_load_threshold: 0.75,
                low_load_threshold: 0.30,
                ramp_up_rate: 1.04,   // 4% increase (moderate)
                ramp_down_rate: 0.80, // 20% decrease
                decay_rate: 0.99,     // 1% decay
                adjust_up_refractory_ms: 75,
                adjust_down_refractory_ms: 25,
                bufferbloat_threshold: 1.5,
            },
        }
    }

    /// Server profile - Slower response, conservative PIE, relaxed targets
    pub fn server() -> Self {
        Self {
            name: "server".to_string(),
            // Server: stability-focused defaults
            default_params: [
                // LATENCY_CRITICAL: [latency_weight, base_slice_ns, vruntime_scale, preemption_priority, migration_cost]
                [1_000_000, 2_000_000, 1536, 100, 100_000],
                // NORMAL
                [1_000_000, 5_000_000, 1536, 100, 100_000],
                // HOG
                [1_000_000, 8_000_000, 1536, 100, 100_000],
                // BACKGROUND
                [1_000_000, 5_000_000, 1536, 100, 100_000],
            ],
            // Server: extra-wide bounds for maximum stability and throughput
            bounds: [
                // LATENCY_CRITICAL bounds - very wide
                [
                    (250_000, 5_000_000),  // latency_weight (wider)
                    (500_000, 20_000_000), // base_slice_ns (0.5-20ms extra wide)
                    (512, 3072),           // vruntime_scale (wider range)
                    (50, 150),             // preemption_priority (some variation allowed)
                    (25_000, 1_000_000),   // migration_cost (wider)
                ],
                // NORMAL bounds
                [
                    (250_000, 5_000_000),  // latency_weight
                    (500_000, 20_000_000), // base_slice_ns
                    (512, 3072),           // vruntime_scale
                    (50, 150),             // preemption_priority
                    (25_000, 1_000_000),   // migration_cost
                ],
                // HOG bounds - very wide for maximum flexibility
                [
                    (250_000, 5_000_000),  // latency_weight
                    (500_000, 20_000_000), // base_slice_ns (extra wide for throughput tasks)
                    (512, 3072),           // vruntime_scale
                    (50, 150),             // preemption_priority
                    (25_000, 1_000_000),   // migration_cost
                ],
                // BACKGROUND bounds
                [
                    (250_000, 5_000_000),  // latency_weight
                    (500_000, 20_000_000), // base_slice_ns
                    (512, 3072),           // vruntime_scale
                    (50, 150),             // preemption_priority
                    (25_000, 1_000_000),   // migration_cost
                ],
            ],
            response_ms: 50, // Slower, stable response
            pie_alpha: 16,   // Conservative proportional gain
            pie_beta: 8,     // Gentle integral response
            pie_max_integral: 1_000_000,
            target_latencies_ns: [
                2_000_000,   // LATENCY_CRITICAL: 2 ms
                10_000_000,  // NORMAL: 10 ms
                50_000_000,  // HOG: 50 ms
                200_000_000, // BACKGROUND: 200 ms
            ],
            autorate: AutorateConfig {
                enabled: false,

                // Narrow ranges for stability
                min_params: [
                    [750_000, 1_500_000, 1280, 100, 75_000],
                    [750_000, 4_000_000, 1280, 100, 75_000],
                    [750_000, 6_000_000, 1280, 100, 75_000],
                    [750_000, 4_000_000, 1280, 100, 75_000],
                ],
                baseline_params: [
                    [1_000_000, 2_000_000, 1536, 100, 100_000],
                    [1_000_000, 5_000_000, 1536, 100, 100_000],
                    [1_000_000, 8_000_000, 1536, 100, 100_000],
                    [1_000_000, 5_000_000, 1536, 100, 100_000],
                ],
                max_params: [
                    [2_000_000, 5_000_000, 2048, 100, 200_000],
                    [2_000_000, 8_000_000, 2048, 100, 200_000],
                    [2_000_000, 10_000_000, 2048, 100, 200_000],
                    [2_000_000, 8_000_000, 2048, 100, 200_000],
                ],

                // Conservative tuning for stability
                high_load_threshold: 0.80, // Higher threshold
                low_load_threshold: 0.30,
                ramp_up_rate: 1.02,           // 2% increase (very conservative)
                ramp_down_rate: 0.90,         // 10% decrease (gentle)
                decay_rate: 0.995,            // 0.5% decay (very slow)
                adjust_up_refractory_ms: 200, // Long refractory
                adjust_down_refractory_ms: 100,
                bufferbloat_threshold: 1.8, // Tolerate higher latency
            },
        }
    }
}

impl Default for Profile {
    fn default() -> Self {
        Profile::production()
    }
}

impl fmt::Display for Profile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (response: {}ms, PIE: α={}, β={})",
            self.name, self.response_ms, self.pie_alpha, self.pie_beta
        )
    }
}

impl FromStr for Profile {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Profile::from_str(s).ok_or_else(|| format!("Unknown profile: {}", s))
    }
}

impl Profile {
    /// Parse profile from string for CLI
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "gaming" | "game" => Some(Self::gaming()),
            "production" | "prod" | "default" => Some(Self::production()),
            "server" | "srv" => Some(Self::server()),
            _ => None,
        }
    }

    /// Get bounds for a specific class and parameter
    #[allow(dead_code)] // Part of public API for external parameter queries
    pub fn get_bounds(&self, class_id: usize, param_idx: usize) -> (u64, u64) {
        if class_id >= DESCENT_CLASS_MAX || param_idx >= PARAM_COUNT {
            return (0, u64::MAX);
        }
        self.bounds[class_id][param_idx]
    }

    /// Get default parameter value for a specific class and parameter
    #[allow(dead_code)] // Part of public API for external parameter queries
    pub fn get_default(&self, class_id: usize, param_idx: usize) -> u64 {
        if class_id >= DESCENT_CLASS_MAX || param_idx >= PARAM_COUNT {
            return 0;
        }
        self.default_params[class_id][param_idx]
    }

    /// Get target latency for a specific class
    /// Returns target latency in nanoseconds
    #[allow(dead_code)] // Part of public API for PIE controller
    pub fn get_target_latency(&self, class_id: usize) -> u64 {
        if class_id >= DESCENT_CLASS_MAX {
            return self.target_latencies_ns[DESCENT_CLASS_NORMAL];
        }
        self.target_latencies_ns[class_id]
    }

    /// Get PIE controller configuration
    /// Returns (alpha, beta, max_integral)
    #[allow(dead_code)] // Part of public API for PIE controller initialization
    pub fn get_pie_config(&self) -> (u64, u64, i64) {
        (self.pie_alpha, self.pie_beta, self.pie_max_integral)
    }

    /// Check if CAKE Autorate is enabled for this profile
    pub fn is_autorate_enabled(&self) -> bool {
        self.autorate.enabled
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaming_profile() {
        let p = Profile::gaming();
        assert_eq!(p.name, "gaming");
        assert_eq!(p.response_ms, 20); // Updated: was 10
        assert_eq!(p.pie_alpha, 8); // Updated: was 4
        assert_eq!(p.pie_beta, 4); // Updated: was 2
    }

    #[test]
    fn test_production_profile() {
        let p = Profile::production();
        assert_eq!(p.name, "production");
        assert_eq!(p.response_ms, 20);
        assert_eq!(p.pie_alpha, 8);
        assert_eq!(p.pie_beta, 4);
    }

    #[test]
    fn test_server_profile() {
        let p = Profile::server();
        assert_eq!(p.name, "server");
        assert_eq!(p.response_ms, 50);
        assert_eq!(p.pie_alpha, 16);
        assert_eq!(p.pie_beta, 8);
    }

    #[test]
    fn test_gaming_target_latencies() {
        let p = Profile::gaming();
        assert_eq!(
            p.target_latencies_ns[DESCENT_CLASS_LATENCY_CRITICAL],
            500_000
        );
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_NORMAL], 2_000_000);
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_HOG], 10_000_000);
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_BACKGROUND], 50_000_000);
    }

    #[test]
    fn test_production_target_latencies() {
        let p = Profile::production();
        assert_eq!(
            p.target_latencies_ns[DESCENT_CLASS_LATENCY_CRITICAL],
            1_000_000
        );
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_NORMAL], 5_000_000);
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_HOG], 20_000_000);
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_BACKGROUND], 100_000_000);
    }

    #[test]
    fn test_server_target_latencies() {
        let p = Profile::server();
        assert_eq!(
            p.target_latencies_ns[DESCENT_CLASS_LATENCY_CRITICAL],
            2_000_000
        );
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_NORMAL], 10_000_000);
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_HOG], 50_000_000);
        assert_eq!(p.target_latencies_ns[DESCENT_CLASS_BACKGROUND], 200_000_000);
    }

    #[test]
    fn test_get_target_latency_helper() {
        let gaming = Profile::gaming();
        assert_eq!(
            gaming.get_target_latency(DESCENT_CLASS_LATENCY_CRITICAL),
            500_000
        );
        assert_eq!(gaming.get_target_latency(DESCENT_CLASS_NORMAL), 2_000_000);
        assert_eq!(gaming.get_target_latency(DESCENT_CLASS_HOG), 10_000_000);
        assert_eq!(
            gaming.get_target_latency(DESCENT_CLASS_BACKGROUND),
            50_000_000
        );

        // Test out-of-bounds returns NORMAL latency
        assert_eq!(gaming.get_target_latency(99), 2_000_000);
    }

    #[test]
    fn test_get_pie_config_helper() {
        let gaming = Profile::gaming();
        let (alpha, beta, max_integral) = gaming.get_pie_config();
        assert_eq!(alpha, 4);
        assert_eq!(beta, 2);
        assert_eq!(max_integral, 1_000_000);

        let production = Profile::production();
        let (alpha, beta, max_integral) = production.get_pie_config();
        assert_eq!(alpha, 8);
        assert_eq!(beta, 4);
        assert_eq!(max_integral, 1_000_000);

        let server = Profile::server();
        let (alpha, beta, max_integral) = server.get_pie_config();
        assert_eq!(alpha, 16);
        assert_eq!(beta, 8);
        assert_eq!(max_integral, 1_000_000);
    }

    #[test]
    fn test_default_profile() {
        let p = Profile::default();
        assert_eq!(p.name, "production");
    }

    #[test]
    fn test_from_str() {
        assert!(Profile::from_str("gaming").is_some());
        assert!(Profile::from_str("game").is_some());
        assert!(Profile::from_str("production").is_some());
        assert!(Profile::from_str("prod").is_some());
        assert!(Profile::from_str("default").is_some());
        assert!(Profile::from_str("server").is_some());
        assert!(Profile::from_str("srv").is_some());
        assert!(Profile::from_str("unknown").is_none());
    }

    #[test]
    fn test_bounds_valid() {
        for profile in [Profile::gaming(), Profile::production(), Profile::server()] {
            for class_id in 0..DESCENT_CLASS_MAX {
                for param_idx in 0..PARAM_COUNT {
                    let (min, max) = profile.get_bounds(class_id, param_idx);
                    // Allow min == max for fixed parameters (e.g., server preemption_priority)
                    assert!(
                        min <= max,
                        "Profile {} class {} param {}: min {} should be <= max {}",
                        profile.name,
                        class_id,
                        param_idx,
                        min,
                        max
                    );
                }
            }
        }
    }

    #[test]
    fn test_defaults_in_bounds() {
        for profile in [Profile::gaming(), Profile::production(), Profile::server()] {
            for class_id in 0..DESCENT_CLASS_MAX {
                for param_idx in 0..PARAM_COUNT {
                    let default = profile.get_default(class_id, param_idx);
                    let (min, max) = profile.get_bounds(class_id, param_idx);
                    assert!(
                        default >= min && default <= max,
                        "Profile {} class {} param {}: default {} not in bounds [{}, {}]",
                        profile.name,
                        class_id,
                        param_idx,
                        default,
                        min,
                        max
                    );
                }
            }
        }
    }

    #[test]
    fn test_display() {
        let p = Profile::gaming();
        let s = format!("{}", p);
        assert!(s.contains("gaming"));
        assert!(s.contains("10ms"));
        assert!(s.contains("α=4"));
        assert!(s.contains("β=2"));
    }

    #[test]
    fn test_gaming_autorate_config() {
        let p = Profile::gaming();
        assert!(!p.autorate.enabled); // Disabled by default
        assert!((p.autorate.ramp_up_rate - 1.08).abs() < f64::EPSILON); // 8% aggressive
        assert!((p.autorate.ramp_down_rate - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_production_autorate_config() {
        let p = Profile::production();
        assert!(!p.autorate.enabled);
        assert!((p.autorate.ramp_up_rate - 1.04).abs() < f64::EPSILON); // 4% moderate
    }

    #[test]
    fn test_server_autorate_config() {
        let p = Profile::server();
        assert!(!p.autorate.enabled);
        assert!((p.autorate.ramp_up_rate - 1.02).abs() < f64::EPSILON); // 2% conservative
    }

    #[test]
    fn test_autorate_params_in_bounds() {
        // Verify all autorate params fall within profile bounds
        for profile_fn in [Profile::gaming, Profile::production, Profile::server] {
            let p = profile_fn();
            for class in 0..DESCENT_CLASS_MAX {
                for param in 0..PARAM_COUNT {
                    let (min_bound, max_bound) = p.bounds[class][param];

                    assert!(
                        p.autorate.min_params[class][param] >= min_bound,
                        "min_params out of bounds: profile={}, class={}, param={}",
                        p.name,
                        class,
                        param
                    );
                    assert!(
                        p.autorate.max_params[class][param] <= max_bound,
                        "max_params out of bounds: profile={}, class={}, param={}",
                        p.name,
                        class,
                        param
                    );
                    assert!(
                        p.autorate.baseline_params[class][param] >= min_bound
                            && p.autorate.baseline_params[class][param] <= max_bound,
                        "baseline_params out of bounds: profile={}, class={}, param={}",
                        p.name,
                        class,
                        param
                    );
                }
            }
        }
    }

    #[test]
    fn test_is_autorate_enabled() {
        let gaming = Profile::gaming();
        assert!(!gaming.is_autorate_enabled());

        let production = Profile::production();
        assert!(!production.is_autorate_enabled());

        let server = Profile::server();
        assert!(!server.is_autorate_enabled());
    }
}
