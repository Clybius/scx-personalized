// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 OpenCode Assistant
//
// Task classifier for scx_autoland - detects Steam games, SCX_TURBO tasks,
// desktop environment processes, and audio tasks.

use std::collections::HashSet;
use std::fs;

/// Known desktop environment process names
const DE_PROCESSES: &[&str] = &[
    "kwin",
    "kwin_wayland",
    "kwin_x11",
    "mutter",
    "gnome-shell",
    "gnome-session",
    "sway",
    "wayfire",
    "hyprland",
    "xfwm4",
    "openbox",
    "i3",
    "compositor",
    "weston",
];

/// Known audio process names
const AUDIO_PROCESSES: &[&str] = &[
    "pipewire",
    "pipewire-pulse",
    "pipewire-media-session",
    "pulseaudio",
    "jackd",
    "jackdbus",
    "alsa-sink",
    "alsa-source",
];

/// Task classifier for detecting special task types
pub struct TaskClassifier {
    de_processes: HashSet<String>,
    audio_processes: HashSet<String>,
}

impl TaskClassifier {
    pub fn new() -> Self {
        let de_processes = DE_PROCESSES.iter().map(|s| s.to_string()).collect();
        let audio_processes = AUDIO_PROCESSES.iter().map(|s| s.to_string()).collect();

        Self {
            de_processes,
            audio_processes,
        }
    }

    /// Check if a task has SCX_TURBO=1 environment variable
    pub fn check_scx_turbo(pid: u32) -> bool {
        Self::check_env_var(pid, "SCX_TURBO=1")
    }

    /// Check if a task is a Steam game
    pub fn check_steam_game(pid: u32) -> bool {
        // Check for SteamGameId or STEAM_GAME in environment
        if Self::check_env_var(pid, "SteamGameId=") {
            return true;
        }
        if Self::check_env_var(pid, "STEAM_GAME=") {
            return true;
        }
        false
    }

    /// Check if process is a known desktop environment task
    pub fn is_de_task(&self, comm: &str) -> bool {
        self.de_processes.contains(comm)
    }

    /// Check if process is a known audio task
    pub fn is_audio_task(&self, comm: &str) -> bool {
        self.audio_processes.contains(comm)
    }

    /// Check if command line contains .exe (likely Wine/Proton game)
    pub fn has_exe_in_cmdline(pid: u32) -> bool {
        let cmdline_path = format!("/proc/{}/cmdline", pid);
        if let Ok(content) = fs::read(&cmdline_path) {
            // cmdline is null-separated
            let content_str = String::from_utf8_lossy(&content);
            content_str.to_lowercase().contains(".exe")
        } else {
            false
        }
    }

    /// Generic environment variable checker
    fn check_env_var(pid: u32, var_prefix: &str) -> bool {
        let environ_path = format!("/proc/{}/environ", pid);

        match fs::read(&environ_path) {
            Ok(content) => {
                // environ is null-separated key=value pairs
                content
                    .split(|&b| b == 0)
                    .filter_map(|bytes| std::str::from_utf8(bytes).ok())
                    .any(|entry| entry.starts_with(var_prefix))
            }
            Err(_) => false,
        }
    }

    /// Classify a task based on all available heuristics
    /// Returns (is_scx_turbo, is_steam_game, is_de, is_audio)
    pub fn classify(&self, pid: u32, comm: &str) -> (bool, bool, bool, bool) {
        let is_scx_turbo = Self::check_scx_turbo(pid);
        let is_steam_game = Self::check_steam_game(pid) || Self::has_exe_in_cmdline(pid);
        let is_de = self.is_de_task(comm);
        let is_audio = self.is_audio_task(comm);

        (is_scx_turbo, is_steam_game, is_de, is_audio)
    }
}

impl Default for TaskClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_de_detection() {
        let classifier = TaskClassifier::new();
        assert!(classifier.is_de_task("kwin"));
        assert!(classifier.is_de_task("gnome-shell"));
        assert!(classifier.is_de_task("sway"));
        assert!(!classifier.is_de_task("firefox"));
    }

    #[test]
    fn test_audio_detection() {
        let classifier = TaskClassifier::new();
        assert!(classifier.is_audio_task("pipewire"));
        assert!(classifier.is_audio_task("pulseaudio"));
        assert!(!classifier.is_audio_task("chrome"));
    }
}
