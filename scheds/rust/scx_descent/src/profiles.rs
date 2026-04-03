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
            // Gaming: wider bounds for aggressive optimization
            bounds: [
                // LATENCY_CRITICAL bounds
                [
                    (10_000, 10_000_000),  // latency_weight
                    (100_000, 10_000_000), // base_slice_ns
                    (256, 2048),           // vruntime_scale
                    (10, 200),             // preemption_priority
                    (1_000, 10_000_000),   // migration_cost
                ],
                // NORMAL bounds
                [
                    (10_000, 10_000_000),  // latency_weight
                    (100_000, 10_000_000), // base_slice_ns
                    (256, 2048),           // vruntime_scale
                    (10, 200),             // preemption_priority
                    (1_000, 10_000_000),   // migration_cost
                ],
                // HOG bounds
                [
                    (10_000, 10_000_000),  // latency_weight
                    (100_000, 10_000_000), // base_slice_ns
                    (256, 2048),           // vruntime_scale
                    (10, 200),             // preemption_priority
                    (1_000, 10_000_000),   // migration_cost
                ],
                // BACKGROUND bounds
                [
                    (10_000, 10_000_000),  // latency_weight
                    (100_000, 10_000_000), // base_slice_ns
                    (256, 2048),           // vruntime_scale
                    (10, 200),             // preemption_priority
                    (1_000, 10_000_000),   // migration_cost
                ],
            ],
            response_ms: 10, // Fast response for gaming
            pie_alpha: 4,    // Aggressive proportional gain
            pie_beta: 2,     // Fast integral response
            pie_max_integral: 1_000_000,
            target_latencies_ns: [
                500_000,    // LATENCY_CRITICAL: 500 µs
                2_000_000,  // NORMAL: 2 ms
                10_000_000, // HOG: 10 ms
                50_000_000, // BACKGROUND: 50 ms
            ],
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
            // Production: standard ranges
            bounds: [
                // LATENCY_CRITICAL bounds
                [
                    (100_000, 5_000_000), // latency_weight
                    (200_000, 5_000_000), // base_slice_ns
                    (512, 1536),          // vruntime_scale
                    (50, 150),            // preemption_priority
                    (5_000, 1_000_000),   // migration_cost
                ],
                // NORMAL bounds
                [
                    (100_000, 5_000_000), // latency_weight
                    (200_000, 5_000_000), // base_slice_ns
                    (512, 1536),          // vruntime_scale
                    (50, 150),            // preemption_priority
                    (5_000, 1_000_000),   // migration_cost
                ],
                // HOG bounds
                [
                    (100_000, 5_000_000), // latency_weight
                    (200_000, 5_000_000), // base_slice_ns
                    (512, 1536),          // vruntime_scale
                    (50, 150),            // preemption_priority
                    (5_000, 1_000_000),   // migration_cost
                ],
                // BACKGROUND bounds
                [
                    (100_000, 5_000_000), // latency_weight
                    (200_000, 5_000_000), // base_slice_ns
                    (512, 1536),          // vruntime_scale
                    (50, 150),            // preemption_priority
                    (5_000, 1_000_000),   // migration_cost
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
            // Server: narrow bounds for stability
            bounds: [
                // LATENCY_CRITICAL bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority (fixed)
                    (50_000, 500_000),       // migration_cost
                ],
                // NORMAL bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority (fixed)
                    (50_000, 500_000),       // migration_cost
                ],
                // HOG bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority (fixed)
                    (50_000, 500_000),       // migration_cost
                ],
                // BACKGROUND bounds
                [
                    (500_000, 2_000_000),    // latency_weight
                    (1_000_000, 10_000_000), // base_slice_ns
                    (1024, 2048),            // vruntime_scale
                    (100, 100),              // preemption_priority (fixed)
                    (50_000, 500_000),       // migration_cost
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaming_profile() {
        let p = Profile::gaming();
        assert_eq!(p.name, "gaming");
        assert_eq!(p.response_ms, 10);
        assert_eq!(p.pie_alpha, 4);
        assert_eq!(p.pie_beta, 2);
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
}
