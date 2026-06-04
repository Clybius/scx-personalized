// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_astro contributors

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use libbpf_rs::MapCore;

use crate::bpf_intf;

const SCX_ASTRO_ENV_PREFIX: &str = "SCX_ASTRO=";

fn parse_profile_from_env(environ: &[u8]) -> Option<u8> {
    for kv in environ.split(|&b| b == 0) {
        if let Ok(s) = std::str::from_utf8(kv) {
            if let Some(val) = s.strip_prefix(SCX_ASTRO_ENV_PREFIX) {
                return match val {
                    "interactive" => Some(bpf_intf::consts_ASTRO_PROFILE_INTERACTIVE as u8),
                    "normal" => Some(bpf_intf::consts_ASTRO_PROFILE_NORMAL as u8),
                    "compute" => Some(bpf_intf::consts_ASTRO_PROFILE_COMPUTE as u8),
                    "background" => Some(bpf_intf::consts_ASTRO_PROFILE_BACKGROUND as u8),
                    _ => None,
                };
            }
        }
    }
    None
}

fn scan_proc_environ() -> HashMap<i32, u8> {
    let mut overrides = HashMap::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return overrides;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if !name_str.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let pid: i32 = match name_str.parse() {
            Ok(p) => p,
            Err(_) => continue,
        };
        let path = format!("/proc/{}/environ", pid);
        if let Ok(environ) = std::fs::read(&path) {
            if let Some(profile) = parse_profile_from_env(&environ) {
                overrides.insert(pid, profile);
            }
        }
    }
    overrides
}

pub struct OverrideScanner {
    map: libbpf_rs::MapHandle,
    shutdown: Arc<AtomicBool>,
    interval: Duration,
    prev_overrides: HashMap<i32, u8>,
}

impl OverrideScanner {
    pub fn new(
        map: libbpf_rs::MapHandle,
        shutdown: Arc<AtomicBool>,
        interval: Duration,
    ) -> Self {
        Self {
            map,
            shutdown,
            interval,
            prev_overrides: HashMap::new(),
        }
    }

    pub fn run(&mut self) {
        while !self.shutdown.load(Ordering::Relaxed) {
            let overrides = scan_proc_environ();

            // Delete stale entries
            for (tgid, _) in &self.prev_overrides {
                if !overrides.contains_key(tgid) {
                    let _ = self.map.delete(&tgid.to_ne_bytes());
                }
            }

            // Insert/update current entries
            for (tgid, profile) in &overrides {
                let val = bpf_intf::astro_tgid_profile { profile: *profile };
                let val_bytes = unsafe {
                    std::slice::from_raw_parts(
                        &val as *const _ as *const u8,
                        std::mem::size_of::<bpf_intf::astro_tgid_profile>(),
                    )
                };
                if let Err(e) = self
                    .map
                    .update(&tgid.to_ne_bytes(), val_bytes, libbpf_rs::MapFlags::ANY)
                {
                    log::warn!("tgid_profile_map update failed for {}: {}", tgid, e);
                }
            }

            self.prev_overrides = overrides;
            std::thread::sleep(self.interval);
        }
    }
}
