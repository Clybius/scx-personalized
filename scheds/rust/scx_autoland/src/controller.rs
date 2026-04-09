// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 OpenCode Assistant
//
// Adaptive controller for scx_autoland - automatically tunes scheduling
// parameters based on task load ratios.
//
// UNIFIED LOAD-BASED FRAMEWORK:
// All classes use baseline * ratio as their preferred state, then adapt based
// on the ratio of tasks in each class (load-based, not latency-based).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Adaptive controller state
pub struct AdaptiveController {
    // Class ratios (from CLI) - stored for potential external adjustment
    #[allow(dead_code)]
    lc_ratio: f64,
    #[allow(dead_code)]
    normal_ratio: f64,
    #[allow(dead_code)]
    hog_ratio: f64,

    // Baseline slice values (fixed reference points) - stored for debugging
    #[allow(dead_code)]
    baseline_slice_us: u64,

    // Preferred slice values (baseline * ratio)
    lc_preferred_slice_us: u64,
    normal_preferred_slice_us: u64,
    hog_preferred_slice_us: u64,

    // Class load thresholds (ratio of total tasks)
    lc_load_threshold: f64,
    normal_load_threshold: f64,
    hog_load_threshold: f64,

    // Class-specific multipliers for min/max caps
    lc_min_multiplier: f64,
    normal_min_multiplier: f64,
    hog_max_multiplier: f64,

    // Current observations (EWMA alpha = 0.25)
    lc_latency_ewma: Arc<AtomicU64>,
    normal_latency_ewma: Arc<AtomicU64>,
    hog_latency_ewma: Arc<AtomicU64>,

    // Current slice values (in microseconds)
    current_lc_slice_us: Arc<AtomicU64>,
    current_normal_slice_us: Arc<AtomicU64>,
    current_hog_slice_us: Arc<AtomicU64>,

    // Preemption threshold [0, 1024]
    current_preempt_threshold: Arc<AtomicU64>,

    // Preemption parameters
    preempt_threshold_min: u32,
    preempt_threshold_max: u32,
    lc_latency_critical_threshold_us: u64,
    lc_latency_good_threshold_us: u64,

    // Tuning parameters
    adjustment_step: f64,
    min_slice_us: u64,
    max_slice_us: u64,
    is_enabled: bool,
}

impl AdaptiveController {
    pub fn new(
        baseline_slice_us: u64,
        lc_ratio: f64,
        normal_ratio: f64,
        hog_ratio: f64,
        enabled: bool,
    ) -> Self {
        // Calculate preferred slices (baseline * ratio)
        let lc_preferred = (baseline_slice_us as f64 * lc_ratio) as u64;
        let normal_preferred = (baseline_slice_us as f64 * normal_ratio) as u64;
        let hog_preferred = (baseline_slice_us as f64 * hog_ratio) as u64;

        Self {
            // Class ratios
            lc_ratio,
            normal_ratio,
            hog_ratio,

            // Baseline
            baseline_slice_us,

            // Preferred slices
            lc_preferred_slice_us: lc_preferred,
            normal_preferred_slice_us: normal_preferred,
            hog_preferred_slice_us: hog_preferred,

            // Load thresholds (configurable defaults)
            lc_load_threshold: 0.05, // 5% - gaming/audio tasks are typically few
            normal_load_threshold: 0.50, // 50% - normal tasks are the bulk
            hog_load_threshold: 0.10, // 10% - hog tasks indicate batch processing

            // Class-specific caps
            lc_min_multiplier: 0.5,     // Can decrease to 50% of preferred
            normal_min_multiplier: 0.5, // Can decrease to 50% of preferred
            hog_max_multiplier: 4.0,    // Can increase to 4x preferred

            // Latency observations
            lc_latency_ewma: Arc::new(AtomicU64::new(0)),
            normal_latency_ewma: Arc::new(AtomicU64::new(0)),
            hog_latency_ewma: Arc::new(AtomicU64::new(0)),

            // Current slice values (start at preferred)
            current_lc_slice_us: Arc::new(AtomicU64::new(lc_preferred)),
            current_normal_slice_us: Arc::new(AtomicU64::new(normal_preferred)),
            current_hog_slice_us: Arc::new(AtomicU64::new(hog_preferred)),
            current_preempt_threshold: Arc::new(AtomicU64::new(128)),

            // Preemption parameters
            preempt_threshold_min: 64,
            preempt_threshold_max: 256,
            lc_latency_critical_threshold_us: 1000,
            lc_latency_good_threshold_us: 500,

            // Tuning
            adjustment_step: 0.1,
            min_slice_us: 100,
            max_slice_us: 16000,
            is_enabled: enabled,
        }
    }

    /// Update latency observation for a class (called from stats polling)
    pub fn update_latency(&self, class: u32, latency_us: u64) {
        // Match BPF enum from intf.h:
        // AUTOLAND_CLASS_NORMAL = 0, AUTOLAND_CLASS_HOG = 1, AUTOLAND_CLASS_LATENCY_CRITICAL = 2
        let ewma_ref = match class {
            2 => &self.lc_latency_ewma,     // AUTOLAND_CLASS_LATENCY_CRITICAL
            0 => &self.normal_latency_ewma, // AUTOLAND_CLASS_NORMAL
            1 => &self.hog_latency_ewma,    // AUTOLAND_CLASS_HOG
            _ => &self.normal_latency_ewma, // Default to normal for unknown
        };

        // EWMA update: new = old * 0.75 + new * 0.25
        let old_val = ewma_ref.load(Ordering::Relaxed);
        let new_val = (old_val * 3 + latency_us) / 4;
        ewma_ref.store(new_val, Ordering::Relaxed);
    }

    /// Run one iteration of the adaptive controller
    /// Takes task counts for load-based adaptation
    /// Returns (new_lc_slice, new_normal_slice, new_hog_slice, new_preempt_threshold)
    pub fn update(
        &self,
        nr_lc_tasks: u64,
        nr_normal_tasks: u64,
        nr_hog_tasks: u64,
    ) -> (u64, u64, u64, u32) {
        if !self.is_enabled {
            return self.get_current_values();
        }

        let nr_total = nr_lc_tasks + nr_normal_tasks + nr_hog_tasks;
        if nr_total == 0 {
            return self.get_current_values();
        }

        // Calculate load ratios
        let lc_ratio = nr_lc_tasks as f64 / nr_total as f64;
        let normal_ratio = nr_normal_tasks as f64 / nr_total as f64;
        let hog_ratio = nr_hog_tasks as f64 / nr_total as f64;

        // Detect heavy load for each class
        let lc_heavy = lc_ratio > self.lc_load_threshold;
        let normal_heavy = normal_ratio > self.normal_load_threshold;
        let hog_heavy = hog_ratio > self.hog_load_threshold;

        // Adjust slices based on class-specific strategies
        let lc_slice =
            self.adjust_lc_slice(self.current_lc_slice_us.load(Ordering::Relaxed), lc_heavy);

        let normal_slice = self.adjust_normal_slice(
            self.current_normal_slice_us.load(Ordering::Relaxed),
            normal_heavy,
        );

        let hog_slice =
            self.adjust_hog_slice(self.current_hog_slice_us.load(Ordering::Relaxed), hog_heavy);

        // Adjust preemption considering all factors
        let preempt_threshold = self.adjust_preemption_unified(lc_heavy, hog_heavy);

        // Store updated values
        self.current_lc_slice_us.store(lc_slice, Ordering::Relaxed);
        self.current_normal_slice_us
            .store(normal_slice, Ordering::Relaxed);
        self.current_hog_slice_us
            .store(hog_slice, Ordering::Relaxed);
        self.current_preempt_threshold
            .store(preempt_threshold as u64, Ordering::Relaxed);

        (lc_slice, normal_slice, hog_slice, preempt_threshold)
    }

    /// LC slice adjustment: DECREASE on heavy load (more tasks need rapid scheduling)
    fn adjust_lc_slice(&self, current: u64, is_heavy_load: bool) -> u64 {
        let preferred = self.lc_preferred_slice_us;
        let min_slice =
            (preferred as f64 * self.lc_min_multiplier).max(self.min_slice_us as f64) as u64;

        let new_slice = if is_heavy_load {
            // Heavy LC load: decrease slice for more frequent scheduling
            let new_slice = (current as f64 * (1.0 - self.adjustment_step)) as u64;
            new_slice.max(min_slice)
        } else {
            // Light LC load: maintain preferred (never increase)
            preferred
        };

        // Clamp to valid range
        new_slice.clamp(self.min_slice_us, self.max_slice_us)
    }

    /// Normal slice adjustment: DECREASE on heavy load (fairer sharing)
    fn adjust_normal_slice(&self, current: u64, is_heavy_load: bool) -> u64 {
        let preferred = self.normal_preferred_slice_us;
        let min_slice =
            (preferred as f64 * self.normal_min_multiplier).max(self.min_slice_us as f64) as u64;

        let new_slice = if is_heavy_load {
            // Heavy Normal load: decrease slice for fairer sharing
            let new_slice = (current as f64 * (1.0 - self.adjustment_step)) as u64;
            new_slice.max(min_slice)
        } else {
            // Light Normal load: maintain preferred (never increase)
            preferred
        };

        // Clamp to valid range
        new_slice.clamp(self.min_slice_us, self.max_slice_us)
    }

    /// Hog slice adjustment: INCREASE on heavy load (throughput focus)
    fn adjust_hog_slice(&self, current: u64, is_heavy_load: bool) -> u64 {
        let preferred = self.hog_preferred_slice_us;
        let max_slice =
            (preferred as f64 * self.hog_max_multiplier).min(self.max_slice_us as f64) as u64;

        let new_slice = if is_heavy_load {
            // Heavy Hog load: increase slice for throughput
            let new_slice = (current as f64 * (1.0 + self.adjustment_step)) as u64;
            new_slice.min(max_slice)
        } else {
            // Light Hog load: decay toward preferred
            if current > preferred {
                let decay_rate = self.adjustment_step * 0.5;
                let new_slice = (current as f64 * (1.0 - decay_rate)) as u64;
                new_slice.max(preferred)
            } else {
                preferred
            }
        };

        // Clamp to valid range
        new_slice.clamp(self.min_slice_us, self.max_slice_us)
    }

    /// Unified preemption adjustment considering LC latency and class load
    fn adjust_preemption_unified(&self, lc_heavy: bool, hog_heavy: bool) -> u32 {
        let current = self.current_preempt_threshold.load(Ordering::Relaxed) as u32;
        let lc_latency = self.lc_latency_ewma.load(Ordering::Relaxed);

        // Priority 1: Protect LC tasks if struggling
        if lc_latency > self.lc_latency_critical_threshold_us {
            return current.saturating_sub(32).max(self.preempt_threshold_min);
        }

        // Priority 2: Adapt to load conditions
        if lc_heavy {
            // Many LC tasks: easier preemption for responsiveness
            current.saturating_sub(16).max(self.preempt_threshold_min)
        } else if hog_heavy && lc_latency < self.lc_latency_good_threshold_us {
            // Heavy hogs + good LC latency: harder preemption for throughput
            current.saturating_add(16).min(self.preempt_threshold_max)
        } else if lc_latency < self.lc_latency_good_threshold_us {
            // Light load + good LC latency: harder preemption (reduce overhead)
            current.saturating_add(8).min(self.preempt_threshold_max)
        } else {
            current
        }
    }

    fn get_current_values(&self) -> (u64, u64, u64, u32) {
        (
            self.current_lc_slice_us.load(Ordering::Relaxed),
            self.current_normal_slice_us.load(Ordering::Relaxed),
            self.current_hog_slice_us.load(Ordering::Relaxed),
            self.current_preempt_threshold.load(Ordering::Relaxed) as u32,
        )
    }

    pub fn is_enabled(&self) -> bool {
        self.is_enabled
    }

    pub fn get_current_slices(&self) -> (u64, u64, u64) {
        (
            self.current_lc_slice_us.load(Ordering::Relaxed),
            self.current_normal_slice_us.load(Ordering::Relaxed),
            self.current_hog_slice_us.load(Ordering::Relaxed),
        )
    }

    #[allow(dead_code)]
    pub fn set_enabled(&mut self, enabled: bool) {
        self.is_enabled = enabled;
    }

    /// Get current load thresholds (for debugging)
    #[allow(dead_code)]
    pub fn get_thresholds(&self) -> (f64, f64, f64) {
        (
            self.lc_load_threshold,
            self.normal_load_threshold,
            self.hog_load_threshold,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_disabled() {
        let controller = AdaptiveController::new(700, 0.5, 1.0, 2.0, false);
        let (lc, normal, hog, preempt) = controller.update(0, 0, 0);
        assert_eq!(lc, 350); // 700 * 0.5
        assert_eq!(normal, 700); // 700 * 1.0
        assert_eq!(hog, 1400); // 700 * 2.0
        assert_eq!(preempt, 128);
    }

    #[test]
    fn test_light_load_maintains_preferred() {
        let controller = AdaptiveController::new(700, 0.5, 1.0, 2.0, true);

        // Light load: 2% LC, 20% normal, 0% hog
        let (lc, normal, hog, _) = controller.update(2, 20, 0);

        // Should maintain preferred values
        assert_eq!(lc, 350, "LC should stay at preferred 350us");
        assert_eq!(normal, 700, "Normal should stay at preferred 700us");
        assert_eq!(hog, 1400, "Hog should stay at preferred 1400us");
    }

    #[test]
    fn test_heavy_lc_decreases_slice() {
        let controller = AdaptiveController::new(700, 0.5, 1.0, 2.0, true);

        // Heavy LC load: 6% > 5% threshold
        let (lc, _, _, _) = controller.update(60, 940, 0);

        // LC slice should decrease (capped at 50% of preferred = 175us)
        assert!(lc < 350, "LC slice should decrease under heavy load");
        assert!(lc >= 175, "LC slice should not go below 175us");
    }

    #[test]
    fn test_heavy_normal_decreases_slice() {
        let controller = AdaptiveController::new(700, 0.5, 1.0, 2.0, true);

        // Heavy normal load: 60% > 50% threshold
        let (_, normal, _, _) = controller.update(2, 600, 398);

        // Normal slice should decrease
        assert!(
            normal < 700,
            "Normal slice should decrease under heavy load"
        );
        assert!(normal >= 350, "Normal slice should not go below 350us");
    }

    #[test]
    fn test_heavy_hog_increases_slice() {
        let controller = AdaptiveController::new(700, 0.5, 1.0, 2.0, true);

        // Heavy hog load: 15% > 10% threshold
        let (_, _, hog, _) = controller.update(2, 100, 150);

        // Hog slice should increase
        assert!(hog > 1400, "Hog slice should increase under heavy load");
    }

    #[test]
    fn test_hog_slice_capped() {
        let controller = AdaptiveController::new(700, 0.5, 1.0, 2.0, true);

        // Run multiple iterations with heavy hog load
        let mut last_hog = 1400;
        for _ in 0..20 {
            let (_, _, hog, _) = controller.update(0, 100, 900);
            last_hog = hog;
        }

        // Should be capped at 4x preferred = 5600us
        assert!(last_hog <= 5600, "Hog slice should be capped at 5600us");
    }

    #[test]
    fn test_hog_decay_to_preferred() {
        let controller = AdaptiveController::new(700, 0.5, 1.0, 2.0, true);

        // First, build up hog slice with heavy load
        let (_, _, hog, _) = controller.update(0, 100, 900);
        assert!(hog > 1400);

        // Then switch to light load - should decay toward preferred
        let (_, _, hog2, _) = controller.update(0, 200, 0);

        // Should start decaying toward 1400us
        assert!(hog2 <= hog, "Hog slice should decay toward preferred");
    }
}
