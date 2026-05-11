// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 scx_astro contributors

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;
pub use bpf_intf::*;

mod stats;

use std::collections::HashMap;
use std::fs;
use std::io::Read;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use clap::CommandFactory;
use clap::Parser;
use clap_complete::generate;
use clap_complete::Shell;
use crossbeam::channel::RecvTimeoutError;
use libbpf_rs::MapCore;
use log::info;
use scx_stats::prelude::*;
use scx_utils::build_id;
use scx_utils::compat;
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::try_set_rlimit_infinity;
use scx_utils::uei_exited;
use scx_utils::uei_report;
use scx_utils::UserExitInfo;

use stats::Metrics;

const SCHEDULER_NAME: &str = "scx_astro";
#[allow(dead_code)]
const ENV_VAR_NAME: &str = "SCX_ASTRO";

fn full_version() -> String {
    build_id::full_version(env!("CARGO_PKG_VERSION"))
}

#[derive(Debug, Parser)]
#[command(name = SCHEDULER_NAME, version, disable_version_flag = true)]
struct Opts {
    /// Enable stats monitoring with the specified interval (seconds).
    #[clap(long)]
    stats: Option<f64>,

    /// Run in stats monitoring mode only (scheduler is not launched).
    #[clap(long)]
    monitor: Option<f64>,

    /// Enable BPF debug printk output.
    #[clap(short, long, action = clap::ArgAction::SetTrue)]
    debug: bool,

    /// Print version and exit.
    #[clap(short = 'V', long, action = clap::ArgAction::SetTrue)]
    version: bool,

    /// Disable adaptive runtime tuning.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    no_autotune: bool,

    /// Disable automatic /proc scanning for SCX_ASTRO environment overrides.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    no_autoscan: bool,

    /// Interval in milliseconds between /proc environment scans.
    #[clap(long, default_value = "2000")]
    scan_interval_ms: u64,

    /// Generate shell completions and exit.
    #[clap(long, value_name = "SHELL", hide = true)]
    completions: Option<Shell>,

    #[clap(flatten, next_help_heading = "Libbpf Options")]
    libbpf: LibbpfOpts,
}

struct Scheduler<'a> {
    skel: BpfSkel<'a>,
    _struct_ops: Option<libbpf_rs::Link>,
    stats_server: StatsServer<(), Metrics>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum AutoTuneMode {
    Balanced,
    Latency,
    Throughput,
}

impl AutoTuneMode {
    fn as_u64(self) -> u64 {
        match self {
            Self::Balanced => 0,
            Self::Latency => 1,
            Self::Throughput => 2,
        }
    }
    fn as_str(self) -> &'static str {
        match self {
            Self::Balanced => "balanced",
            Self::Latency => "latency",
            Self::Throughput => "throughput",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuntimeTunables {
    interactive_slice_us: u64,
    normal_slice_us: u64,
    compute_slice_us: u64,
    background_slice_us: u64,
}

impl Default for RuntimeTunables {
    fn default() -> Self {
        Self {
            interactive_slice_us: 150,
            normal_slice_us: 1000,
            compute_slice_us: 3000,
            background_slice_us: 500,
        }
    }
}

impl RuntimeTunables {
    fn target_for(mode: AutoTuneMode) -> Self {
        match mode {
            AutoTuneMode::Balanced => Self::default(),
            AutoTuneMode::Latency => Self {
                interactive_slice_us: 100,
                normal_slice_us: 800,
                compute_slice_us: 2000,
                background_slice_us: 300,
            },
            AutoTuneMode::Throughput => Self {
                interactive_slice_us: 200,
                normal_slice_us: 1200,
                compute_slice_us: 5000,
                background_slice_us: 800,
            },
        }
    }

    fn step_towards(&mut self, target: Self) -> bool {
        let mut changed = false;
        changed |= step_u64(
            &mut self.interactive_slice_us,
            target.interactive_slice_us,
            25,
        );
        changed |= step_u64(&mut self.normal_slice_us, target.normal_slice_us, 100);
        changed |= step_u64(&mut self.compute_slice_us, target.compute_slice_us, 250);
        changed |= step_u64(
            &mut self.background_slice_us,
            target.background_slice_us,
            50,
        );
        changed
    }
}

fn step_u64(value: &mut u64, target: u64, step: u64) -> bool {
    if *value == target {
        return false;
    }
    if *value < target {
        *value = (*value + step).min(target);
    } else {
        *value = value.saturating_sub(step).max(target);
    }
    true
}

#[derive(Debug)]
struct AutoTuner {
    tunables: RuntimeTunables,
    mode: AutoTuneMode,
    pending_mode: AutoTuneMode,
    pending_steps: u8,
    generation: u64,
    prev_metrics: Metrics,
}

impl AutoTuner {
    fn new(initial_metrics: Metrics) -> Self {
        Self {
            tunables: RuntimeTunables::default(),
            mode: AutoTuneMode::Balanced,
            pending_mode: AutoTuneMode::Balanced,
            pending_steps: 0,
            generation: 0,
            prev_metrics: initial_metrics,
        }
    }

    fn evaluate_mode(&self, _current: &Metrics, delta: &Metrics) -> AutoTuneMode {
        let total_disp = delta.interactive_dispatches
            + delta.waker_boost_dispatches
            + delta.normal_dispatches
            + delta.compute_dispatches
            + delta.background_dispatches;
        if total_disp < 3 {
            return self.mode;
        }
        let interactive_ratio = (delta.interactive_dispatches + delta.waker_boost_dispatches)
            as f64
            / total_disp as f64;
        let bg_ratio = delta.background_dispatches as f64 / total_disp as f64;
        let preempt_ratio = delta.preempts as f64 / total_disp.max(1) as f64;

        if interactive_ratio > 0.30 || preempt_ratio > 0.15 {
            AutoTuneMode::Latency
        } else if bg_ratio > 0.25 {
            AutoTuneMode::Throughput
        } else {
            AutoTuneMode::Balanced
        }
    }

    fn update(&mut self, current: &Metrics) -> Option<(AutoTuneMode, RuntimeTunables, u64)> {
        let delta = current.delta(&self.prev_metrics);
        self.prev_metrics = current.clone();

        let desired = self.evaluate_mode(current, &delta);
        let mut next = self.mode;

        if desired == self.mode {
            self.pending_mode = self.mode;
            self.pending_steps = 0;
        } else {
            if desired == self.pending_mode {
                self.pending_steps = self.pending_steps.saturating_add(1);
            } else {
                self.pending_mode = desired;
                self.pending_steps = 1;
            }
            if self.pending_steps >= 3 {
                next = desired;
                self.pending_mode = desired;
                self.pending_steps = 0;
            }
        }

        let target = RuntimeTunables::target_for(next);
        let mode_changed = next != self.mode;
        let tunables_changed = self.tunables.step_towards(target);

        if !mode_changed && !tunables_changed {
            return None;
        }

        self.mode = next;
        self.generation += 1;
        Some((self.mode, self.tunables, self.generation))
    }
}

impl<'a> Scheduler<'a> {
    fn init(
        opts: &'a Opts,
        open_object: &'a mut MaybeUninit<libbpf_rs::OpenObject>,
    ) -> Result<Self> {
        try_set_rlimit_infinity();

        let mut skel_builder = BpfSkelBuilder::default();
        skel_builder.obj_builder.debug(opts.debug);

        let open_opts = opts.libbpf.clone().into_bpf_open_opts();
        let mut skel = scx_ops_open!(skel_builder, open_object, astro_ops, open_opts)?;

        skel.struct_ops.astro_ops_mut().flags = *compat::SCX_OPS_ENQ_EXITING
            | *compat::SCX_OPS_ENQ_LAST
            | *compat::SCX_OPS_ENQ_MIGRATION_DISABLED
            | *compat::SCX_OPS_ALLOW_QUEUED_WAKEUP;

        let mut skel = scx_ops_load!(skel, astro_ops, uei)?;
        Self::write_tunables(
            &mut skel,
            RuntimeTunables::default(),
            AutoTuneMode::Balanced,
            0,
        );

        let struct_ops = scx_ops_attach!(skel, astro_ops)?;
        let stats_server = StatsServer::new(stats::server_data()).launch()?;

        Ok(Self {
            skel,
            _struct_ops: Some(struct_ops),
            stats_server,
        })
    }

    fn write_tunables(
        skel: &mut BpfSkel<'a>,
        tunables: RuntimeTunables,
        mode: AutoTuneMode,
        generation: u64,
    ) {
        let data = skel.maps.data_data.as_mut().unwrap();
        data.tune_interactive_slice_ns = tunables.interactive_slice_us * 1000;
        data.tune_normal_slice_ns = tunables.normal_slice_us * 1000;
        data.tune_compute_slice_ns = tunables.compute_slice_us * 1000;
        data.tune_background_slice_ns = tunables.background_slice_us * 1000;

        let bss_data = skel.maps.bss_data.as_mut().unwrap();
        bss_data.autotune_mode = mode.as_u64();
        bss_data.autotune_generation = generation;
    }

    fn get_metrics(&self) -> Metrics {
        let bss_data = self.skel.maps.bss_data.as_ref().unwrap();
        let data = self.skel.maps.data_data.as_ref().unwrap();
        let cpu_policy = self.read_cpu_state_agg();

        Metrics {
            nr_running: bss_data.nr_running,
            interactive_dispatches: bss_data.interactive_dispatches
                + cpu_policy.interactive_dispatches,
            waker_boost_dispatches: bss_data.waker_boost_dispatches
                + cpu_policy.waker_boost_dispatches,
            normal_dispatches: bss_data.normal_dispatches + cpu_policy.normal_dispatches,
            compute_dispatches: bss_data.compute_dispatches + cpu_policy.compute_dispatches,
            background_dispatches: bss_data.background_dispatches
                + cpu_policy.background_dispatches,
            preempts: bss_data.preempts + cpu_policy.preempts,
            kicks_idle: bss_data.kicks_idle + cpu_policy.kicks_idle,
            kicks_preempt: bss_data.kicks_preempt + cpu_policy.kicks_preempt,
            profile_transitions: bss_data.profile_transitions + cpu_policy.profile_transitions,
            waker_boosts: bss_data.waker_boosts + cpu_policy.waker_boosts,
            srpt_short_tasks: bss_data.srpt_short_tasks + cpu_policy.srpt_short_tasks,
            budget_refills: bss_data.budget_refills + cpu_policy.budget_refills,
            budget_exhaustions: bss_data.budget_exhaustions + cpu_policy.budget_exhaustions,
            contained_starvation_rounds: cpu_policy.contained_starvation_rounds,
            shared_starvation_rounds: cpu_policy.shared_starvation_rounds,
            high_priority_burst_rounds: cpu_policy.high_priority_burst_rounds,
            autotune_mode: bss_data.autotune_mode,
            autotune_generation: bss_data.autotune_generation,
            tune_interactive_slice_us: data.tune_interactive_slice_ns / 1000,
            tune_normal_slice_us: data.tune_normal_slice_ns / 1000,
            tune_compute_slice_us: data.tune_compute_slice_ns / 1000,
            tune_background_slice_us: data.tune_background_slice_ns / 1000,
        }
    }

    fn read_cpu_state_agg(&self) -> stats::Metrics {
        let key = 0u32.to_ne_bytes();
        let mut agg = stats::Metrics::default();
        let percpu_vals: Vec<Vec<u8>> = match self
            .skel
            .maps
            .cpu_state
            .lookup_percpu(&key, libbpf_rs::MapFlags::ANY)
        {
            Ok(Some(vals)) => vals,
            _ => return agg,
        };
        for cpu_val in percpu_vals.iter() {
            if cpu_val.len() < std::mem::size_of::<bpf_intf::astro_cpu_state>() {
                continue;
            }
            let state = unsafe {
                std::ptr::read_unaligned(cpu_val.as_ptr() as *const bpf_intf::astro_cpu_state)
            };
            agg.interactive_dispatches = agg
                .interactive_dispatches
                .saturating_add(state.interactive_dispatches);
            agg.waker_boost_dispatches = agg
                .waker_boost_dispatches
                .saturating_add(state.waker_boost_dispatches);
            agg.normal_dispatches = agg
                .normal_dispatches
                .saturating_add(state.normal_dispatches);
            agg.compute_dispatches = agg
                .compute_dispatches
                .saturating_add(state.compute_dispatches);
            agg.background_dispatches = agg
                .background_dispatches
                .saturating_add(state.background_dispatches);
            agg.preempts = agg.preempts.saturating_add(state.preempts);
            agg.kicks_idle = agg.kicks_idle.saturating_add(state.kicks_idle);
            agg.kicks_preempt = agg.kicks_preempt.saturating_add(state.kicks_preempt);
            agg.profile_transitions = agg
                .profile_transitions
                .saturating_add(state.profile_transitions);
            agg.waker_boosts = agg.waker_boosts.saturating_add(state.waker_boosts);
            agg.srpt_short_tasks = agg.srpt_short_tasks.saturating_add(state.srpt_short_tasks);
            agg.budget_refills = agg.budget_refills.saturating_add(state.budget_refills);
            agg.budget_exhaustions = agg
                .budget_exhaustions
                .saturating_add(state.budget_exhaustions);
            agg.contained_starvation_rounds = agg
                .contained_starvation_rounds
                .max(state.contained_starvation_rounds);
            agg.shared_starvation_rounds = agg
                .shared_starvation_rounds
                .max(state.shared_starvation_rounds);
            agg.high_priority_burst_rounds = agg
                .high_priority_burst_rounds
                .max(state.high_priority_burst_rounds);
        }
        agg
    }

    fn exited(&self) -> bool {
        uei_exited!(&self.skel, uei)
    }

    fn run(&mut self, shutdown: Arc<AtomicBool>, autotune_enabled: bool) -> Result<UserExitInfo> {
        let (res_ch, req_ch) = self.stats_server.channels();
        let mut autotuner = autotune_enabled.then(|| AutoTuner::new(self.get_metrics()));
        let mut next_tune_at = Instant::now() + Duration::from_secs(1);

        while !shutdown.load(Ordering::Relaxed) && !self.exited() {
            match req_ch.recv_timeout(Duration::from_millis(250)) {
                Ok(()) => res_ch.send(self.get_metrics())?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(e) => Err(e)?,
            }

            if let Some(autotuner) = autotuner.as_mut() {
                if Instant::now() >= next_tune_at {
                    let current = self.get_metrics();
                    if let Some((mode, tunables, generation)) = autotuner.update(&current) {
                        Self::write_tunables(&mut self.skel, tunables, mode, generation);
                        info!(
                            "autotune={} gen={} int_slice={}us norm_slice={}us comp_slice={}us bg_slice={}us",
                            mode.as_str(),
                            generation,
                            tunables.interactive_slice_us,
                            tunables.normal_slice_us,
                            tunables.compute_slice_us,
                            tunables.background_slice_us,
                        );
                    }
                    next_tune_at = Instant::now() + Duration::from_secs(1);
                }
            }
        }

        let _ = self._struct_ops.take();
        uei_report!(&self.skel, uei)
    }
}

#[allow(dead_code)]
fn parse_profile_from_environ(environ: &str) -> Option<u8> {
    for item in environ.split('\0') {
        if let Some(val) = item.strip_prefix(ENV_VAR_NAME) {
            let val = val.strip_prefix("=")?;
            let profile = match val {
                "interactive" => bpf_intf::consts_ASTRO_PROFILE_INTERACTIVE as u8,
                "normal" => bpf_intf::consts_ASTRO_PROFILE_NORMAL as u8,
                "compute" => bpf_intf::consts_ASTRO_PROFILE_COMPUTE as u8,
                "background" => bpf_intf::consts_ASTRO_PROFILE_BACKGROUND as u8,
                _ => continue,
            };
            return Some(profile);
        }
    }
    None
}

#[allow(dead_code)]
fn scan_proc_environ(skel: &mut BpfSkel, interval_ms: u64, shutdown: Arc<AtomicBool>) {
    let mut known: HashMap<i32, u8> = HashMap::new();
    let interval = Duration::from_millis(interval_ms);

    while !shutdown.load(Ordering::Relaxed) {
        let mut current_pids: Vec<i32> = Vec::new();

        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.flatten() {
                let name = entry.file_name();
                let name_str = name.to_string_lossy();
                let pid: i32 = match name_str.parse() {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                let path = format!("/proc/{}/environ", pid);
                let mut file = match fs::File::open(&path) {
                    Ok(f) => f,
                    Err(_) => continue,
                };
                let mut buf = Vec::new();
                if file.read_to_end(&mut buf).is_err() {
                    continue;
                }
                let environ = String::from_utf8_lossy(&buf);
                if let Some(profile) = parse_profile_from_environ(&environ) {
                    current_pids.push(pid);
                    let old = known.get(&pid).copied();
                    if old != Some(profile) {
                        let key = pid.to_ne_bytes();
                        let val = bpf_intf::astro_tgid_profile { profile };
                        let val_bytes = unsafe {
                            std::slice::from_raw_parts(
                                &val as *const _ as *const u8,
                                std::mem::size_of::<bpf_intf::astro_tgid_profile>(),
                            )
                        };
                        if let Err(e) = skel.maps.tgid_profile_map.update(
                            &key,
                            val_bytes,
                            libbpf_rs::MapFlags::ANY,
                        ) {
                            log::debug!("failed to update tgid_profile_map for {}: {}", pid, e);
                        } else {
                            known.insert(pid, profile);
                            log::debug!("set profile={} for pid={}", profile, pid);
                        }
                    }
                }
            }
        }

        // Remove stale entries
        known.retain(|pid, _| current_pids.contains(pid));
        // Note: we don't delete from BPF map on exit to keep it simple; stale entries
        // are overwritten on reuse. A production version could use a cleanup pass.

        std::thread::sleep(interval);
    }
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    if let Some(shell) = opts.completions {
        generate(
            shell,
            &mut Opts::command(),
            SCHEDULER_NAME,
            &mut std::io::stdout(),
        );
        return Ok(());
    }

    let monitor_only = opts.monitor.is_some();

    if opts.version {
        println!("{} {}", SCHEDULER_NAME, full_version());
        return Ok(());
    }

    if !monitor_only {
        simplelog::SimpleLogger::init(
            if opts.debug {
                simplelog::LevelFilter::Debug
            } else {
                simplelog::LevelFilter::Info
            },
            simplelog::Config::default(),
        )?;
        info!("{} {}", SCHEDULER_NAME, full_version());
        info!("Starting {} scheduler", SCHEDULER_NAME);
    }

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })?;

    if let Some(intv) = opts.monitor.or(opts.stats) {
        let monitor_shutdown = shutdown.clone();
        let jh = std::thread::spawn(move || {
            if let Err(err) = stats::monitor(Duration::from_secs_f64(intv), monitor_shutdown) {
                log::warn!("stats monitor thread finished with error: {err}");
            }
        });
        if monitor_only {
            let _ = jh.join();
            return Ok(());
        }
    }

    let mut open_object = MaybeUninit::<libbpf_rs::OpenObject>::uninit();
    let mut sched = Scheduler::init(&opts, &mut open_object)?;

    // TODO: Spawn auto-scan thread for /proc environ scanning.
    // This requires extracting map FDs from the loaded skeleton and using
    // libbpf_rs::MapHandle in a separate thread, as BpfSkel is not Send.
    // For now, explicit profile overrides can be set via the tgid_profile_map
    // using bpftool or a small standalone helper.

    sched.run(shutdown, !opts.no_autotune)?;
    info!("Scheduler exited");
    Ok(())
}
