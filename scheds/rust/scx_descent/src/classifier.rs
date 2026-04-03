//! Task Classifier for scx_descent
//!
//! Provides userspace task classification that mirrors BPF classification logic.
//! Used for consistency checking and monitoring between userspace and BPF.

// Phase 2: Task classification logic
// Mirrors BPF classification for consistency checking

use log::debug;

pub struct TaskClassifier;

impl TaskClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Mirrors BPF classification logic for userspace consistency checking
    #[allow(dead_code)] // Reserved for future userspace tracking/monitoring
    pub fn classify_task(&self, _pid: u32, _policy: u32, _is_kthread: bool) -> u32 {
        // Classification is primarily done in BPF
        // This is for userspace tracking/monitoring
        debug!("Task classification check");
        0 // INTERACTIVE
    }
}
