//! Profile System for scx_descent
//!
//! Defines three scheduling profiles optimized for different workloads:
//! - Gaming: Fast response (30ms), wide bounds, high exploration (2.0x)
//! - Production: Balanced (50ms), standard bounds, moderate exploration (1.5x)
//! - Server: Conservative (100ms), narrow bounds, low exploration (1.0x)
//!
//! Each profile specifies:
//! - Parameter bounds (min/max per class)
//! - Default parameter values
//! - Response interval (update frequency)
//! - Exploration factor (Thompson sampling uncertainty)

// Phase 3: Complete Profile System
// Gaming, production, server profiles with bounds, response speeds, and exploration factors

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

/// Parameter bounds for Thompson sampling (min, max) per class per parameter
/// [class][param] = (min, max)
/// Classes: [LATENCY_CRITICAL, NORMAL, HOG, BACKGROUND]
/// Params:  [latency_weight, base_slice_ns, vruntime_scale, preemption_priority, migration_cost]
pub type ParamBounds = [[(u64, u64); 5]; 4];

/// Default parameters per class [LATENCY_CRITICAL, NORMAL, HOG, BACKGROUND]
/// [class][param] = default_value
pub type DefaultParams = [[u64; 5]; 4];

pub struct Profile {
    pub name: String,
    /// Phase 3: Thompson sampler integration
    pub default_params: DefaultParams,
    pub bounds: ParamBounds,
    pub response_ms: u64,        // 30, 50, or 100
    pub exploration_factor: f64, // 1.0, 1.5, or 2.0
}

impl Profile {
    /// Gaming profile - Fast response, high exploration, wide bounds
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
            response_ms: 30,         // Fast response for gaming
            exploration_factor: 2.0, // High exploration
        }
    }

    /// Production profile (DEFAULT) - Balanced response, moderate exploration, standard bounds
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
            response_ms: 50,         // Balanced response
            exploration_factor: 1.5, // Moderate exploration
        }
    }

    /// Server profile - Slower response, conservative exploration, narrow bounds
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
            response_ms: 100,        // Slower, stable response
            exploration_factor: 1.0, // Conservative exploration
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
            "{} (response: {}ms, exploration: {:.1}x)",
            self.name, self.response_ms, self.exploration_factor
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaming_profile() {
        let p = Profile::gaming();
        assert_eq!(p.name, "gaming");
        assert_eq!(p.response_ms, 30);
        assert_eq!(p.exploration_factor, 2.0);
    }

    #[test]
    fn test_production_profile() {
        let p = Profile::production();
        assert_eq!(p.name, "production");
        assert_eq!(p.response_ms, 50);
        assert_eq!(p.exploration_factor, 1.5);
    }

    #[test]
    fn test_server_profile() {
        let p = Profile::server();
        assert_eq!(p.name, "server");
        assert_eq!(p.response_ms, 100);
        assert_eq!(p.exploration_factor, 1.0);
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
        assert!(s.contains("30ms"));
        assert!(s.contains("2.0x"));
    }
}
