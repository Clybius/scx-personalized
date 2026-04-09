// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2025 Andrea Righi <arighi@nvidia.com>

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;
pub use bpf_intf::*;

mod controller;
mod pid_tail_latency_controller;
mod stats;
mod task_classifier;

use pid_tail_latency_controller::PidTailLatencyController;
use task_classifier::TaskClassifier;

use libbpf_rs::MapCore;

use std::collections::HashSet;
use std::ffi::{c_int, c_ulong};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use crossbeam::channel::RecvTimeoutError;
use libbpf_rs::OpenObject;
use libbpf_rs::ProgramInput;
use log::{debug, info, warn};
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
use scx_utils::CoreType;
use scx_utils::Topology;
use scx_utils::UserExitInfo;
use scx_utils::NR_CPU_IDS;
use stats::Metrics;

const SCHEDULER_NAME: &str = "scx_autoland";

#[derive(Debug, clap::Parser)]
#[command(
    name = "scx_autoland",
    version,
    disable_version_flag = true,
    about = "Adaptive scheduler with automatic task classification for latency-critical workloads."
)]
struct Opts {
    /// Exit debug dump buffer length. 0 indicates default.
    #[clap(long, default_value = "0")]
    exit_dump_len: u32,

    /// Base time slice for normal tasks in microseconds.
    /// Latency-critical and Hog slices are calculated as ratios of this base.
    #[clap(short = 's', long, default_value = "2000")]
    slice_us: u64,

    /// Maximum time slice lag in microseconds.
    ///
    /// A positive value can help to enhance the responsiveness of interactive tasks, but it can
    /// also make performance more "spikey".
    #[clap(short = 'l', long, default_value = "40000")]
    slice_us_lag: u64,

    /// CPU busy threshold.
    ///
    /// Specifies the CPU utilization percentage (0-100%) at which the scheduler considers the
    /// system to be busy.
    #[clap(short = 'c', long, default_value = "75")]
    cpu_busy_thresh: u64,

    /// Polling time (ms) to refresh the CPU utilization.
    ///
    /// This interval determines how often the scheduler refreshes the CPU utilization that is
    /// compared with the CPU busy threshold (option -c) to decide if the system is busy or not.
    ///
    /// Value is clamped to the range [10 .. 1000].
    ///
    /// 0 = disabled.
    #[clap(short = 'p', long, default_value = "250")]
    polling_ms: u64,

    /// Specifies a list of CPUs to prioritize.
    ///
    /// Accepts a comma-separated list of CPUs or ranges (i.e., 0-3,12-15) or the following special
    /// keywords:
    ///
    /// "turbo" = automatically detect and prioritize the CPUs with the highest max frequency,
    /// "performance" = automatically detect and prioritize the fastest CPUs,
    /// "powersave" = automatically detect and prioritize the slowest CPUs,
    /// "all" = all CPUs assigned to the primary domain.
    ///
    /// By default "all" CPUs are used.
    #[clap(short = 'm', long)]
    primary_domain: Option<String>,

    /// Enable preferred idle CPU scanning.
    ///
    /// With this option enabled, the scheduler will prioritize assigning tasks to higher-ranked
    /// cores before considering lower-ranked ones.
    #[clap(short = 'P', long, action = clap::ArgAction::SetTrue)]
    preferred_idle_scan: bool,

    /// Enable stats monitoring with the specified interval.
    #[clap(long)]
    stats: Option<f64>,

    /// Run in stats monitoring mode with the specified interval. Scheduler
    /// is not launched.
    #[clap(long)]
    monitor: Option<f64>,

    /// Enable verbose output, including libbpf details.
    #[clap(short = 'v', long, action = clap::ArgAction::SetTrue)]
    verbose: bool,

    /// Print scheduler version and exit.
    #[clap(short = 'V', long, action = clap::ArgAction::SetTrue)]
    version: bool,

    /// Per-class slice ratios as comma-separated triple (LC,Normal,Hog).
    /// LC and Hog slices are calculated as: base_slice * ratio
    #[clap(long, default_value = "0.25,1.0,2.0")]
    slice_ratios: String,

    /// Disable adaptive controller (use fixed parameters).
    #[clap(long, action = clap::ArgAction::SetTrue)]
    no_adaptive: bool,

    /// Preemption threshold [0, 1024]. Lower values allow more preemption.
    #[clap(long, default_value = "128")]
    preempt_threshold: u32,

    /// Enable verbose output, including libbpf details.
    #[clap(flatten)]
    libbpf: LibbpfOpts,

    /// Print stats help and exit.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    help_stats: bool,
}

#[derive(PartialEq)]
enum Powermode {
    Turbo,
    Performance,
    Powersave,
    Any,
}

/*
 * TODO: this code is shared between scx_bpfland, scx_flash and scx_cosmos; consder to move it to
 * scx_utils.
 */
fn get_primary_cpus(mode: Powermode) -> std::io::Result<Vec<usize>> {
    let cpus: Vec<usize> = Topology::new()
        .unwrap()
        .all_cores
        .values()
        .flat_map(|core| &core.cpus)
        .filter_map(|(cpu_id, cpu)| match (&mode, &cpu.core_type) {
            // Turbo mode: prioritize CPUs with the highest max frequency
            (Powermode::Turbo, CoreType::Big { turbo: true }) |
            // Performance mode: add all the Big CPUs (either Turbo or non-Turbo)
            (Powermode::Performance, CoreType::Big { .. }) |
            // Powersave mode: add all the Little CPUs
            (Powermode::Powersave, CoreType::Little) => Some(*cpu_id),
            (Powermode::Any, ..) => Some(*cpu_id),
            _ => None,
        })
        .collect();

    Ok(cpus)
}

pub fn parse_cpu_list(optarg: &str) -> Result<Vec<usize>, String> {
    let mut cpus = Vec::new();
    let mut seen = HashSet::new();

    // Handle special keywords
    if let Some(mode) = match optarg {
        "powersave" => Some(Powermode::Powersave),
        "performance" => Some(Powermode::Performance),
        "turbo" => Some(Powermode::Turbo),
        "all" => Some(Powermode::Any),
        _ => None,
    } {
        return get_primary_cpus(mode).map_err(|e| e.to_string());
    }

    // Validate input characters
    if optarg
        .chars()
        .any(|c| !c.is_ascii_digit() && c != '-' && c != ',' && !c.is_whitespace())
    {
        return Err("Invalid character in CPU list".to_string());
    }

    // Replace all whitespace with tab (or just trim later)
    let cleaned = optarg.replace(' ', "\t");

    for token in cleaned.split(',') {
        let token = token.trim_matches(|c: char| c.is_whitespace());

        if token.is_empty() {
            continue;
        }

        if let Some((start_str, end_str)) = token.split_once('-') {
            let start = start_str
                .trim()
                .parse::<usize>()
                .map_err(|_| "Invalid range start")?;
            let end = end_str
                .trim()
                .parse::<usize>()
                .map_err(|_| "Invalid range end")?;

            if start > end {
                return Err(format!("Invalid CPU range: {}-{}", start, end));
            }

            for i in start..=end {
                if cpus.len() >= *NR_CPU_IDS {
                    return Err(format!("Too many CPUs specified (max {})", *NR_CPU_IDS));
                }
                if seen.insert(i) {
                    cpus.push(i);
                }
            }
        } else {
            let cpu = token
                .parse::<usize>()
                .map_err(|_| format!("Invalid CPU: {}", token))?;
            if cpus.len() >= *NR_CPU_IDS {
                return Err(format!("Too many CPUs specified (max {})", *NR_CPU_IDS));
            }
            if seen.insert(cpu) {
                cpus.push(cpu);
            }
        }
    }

    Ok(cpus)
}

#[derive(Debug, Clone, Copy)]
struct CpuTimes {
    user: u64,
    nice: u64,
    total: u64,
}

struct Scheduler<'a> {
    skel: BpfSkel<'a>,
    opts: &'a Opts,
    struct_ops: Option<libbpf_rs::Link>,
    stats_server: StatsServer<(), Metrics>,
    controller: PidTailLatencyController,
    classifier: TaskClassifier,
    detected_turbo_tasks: HashSet<i32>,
    detected_game_tasks: HashSet<i32>,
}

impl<'a> Scheduler<'a> {
    fn init(opts: &'a Opts, open_object: &'a mut MaybeUninit<OpenObject>) -> Result<Self> {
        try_set_rlimit_infinity();

        // Initialize CPU topology.
        let topo = Topology::new().unwrap();

        // Check host topology to determine if we need to enable SMT capabilities.
        let smt_enabled = topo.smt_enabled;

        info!(
            "{} {} {}",
            SCHEDULER_NAME,
            build_id::full_version(env!("CARGO_PKG_VERSION")),
            if smt_enabled { "SMT on" } else { "SMT off" }
        );

        // Print command line.
        info!(
            "scheduler options: {}",
            std::env::args().collect::<Vec<_>>().join(" ")
        );

        // Initialize BPF connector.
        let mut skel_builder = BpfSkelBuilder::default();
        skel_builder.obj_builder.debug(opts.verbose);
        let open_opts = opts.libbpf.clone().into_bpf_open_opts();
        let mut skel = scx_ops_open!(skel_builder, open_object, autoland_ops, open_opts)?;

        skel.struct_ops.autoland_ops_mut().exit_dump_len = opts.exit_dump_len;

        // Override default BPF scheduling parameters.
        let rodata = skel.maps.rodata_data.as_mut().unwrap();
        rodata.slice_ns = opts.slice_us * 1000;
        rodata.slice_lag = opts.slice_us_lag * 1000;
        rodata.smt_enabled = smt_enabled;

        // Normalize CPU busy threshold in the range [0 .. 1024].
        rodata.busy_threshold = opts.cpu_busy_thresh * 1024 / 100;

        // Parse slice ratios and calculate per-class slices
        let ratios: Vec<f64> = opts
            .slice_ratios
            .split(',')
            .map(|s| s.trim().parse::<f64>().expect("Invalid slice ratio format"))
            .collect();

        if ratios.len() != 3 {
            panic!("--slice-ratios must be exactly 3 comma-separated values (LC,Normal,Hog)");
        }

        let slice_lc = (opts.slice_us as f64 * ratios[0]) as u64;
        let slice_normal = (opts.slice_us as f64 * ratios[1]) as u64;
        let slice_hog = (opts.slice_us as f64 * ratios[2]) as u64;

        info!(
            "Slice configuration: base={}us, ratios=[{},{},{}]",
            opts.slice_us, ratios[0], ratios[1], ratios[2]
        );
        info!(
            "Per-class slices: LC={}us, Normal={}us, Hog={}us",
            slice_lc, slice_normal, slice_hog
        );

        // Set per-class slice durations.
        rodata.slice_ns_lc = slice_lc * 1000;
        rodata.slice_ns_normal = slice_normal * 1000;
        rodata.slice_ns_hog = slice_hog * 1000;
        rodata.preempt_threshold = opts.preempt_threshold;

        // Define the primary scheduling domain.
        let primary_cpus = if let Some(ref domain) = opts.primary_domain {
            match parse_cpu_list(domain) {
                Ok(cpus) => cpus,
                Err(e) => bail!("Error parsing primary domain: {}", e),
            }
        } else {
            (0..*NR_CPU_IDS).collect()
        };
        if primary_cpus.len() < *NR_CPU_IDS {
            info!("Primary CPUs: {:?}", primary_cpus);
            rodata.primary_all = false;
        } else {
            rodata.primary_all = true;
        }

        // Generate the list of available CPUs sorted by capacity in descending order.
        let mut cpus: Vec<_> = topo.all_cpus.values().collect();
        cpus.sort_by_key(|cpu| std::cmp::Reverse(cpu.cpu_capacity));
        for (i, cpu) in cpus.iter().enumerate() {
            rodata.cpu_capacity[cpu.id] = cpu.cpu_capacity as c_ulong;
            rodata.preferred_cpus[i] = cpu.id as u64;
        }
        if opts.preferred_idle_scan {
            info!(
                "Preferred CPUs: {:?}",
                &rodata.preferred_cpus[0..cpus.len()]
            );
        }
        rodata.preferred_idle_scan = opts.preferred_idle_scan;

        // Set scheduler flags.
        skel.struct_ops.autoland_ops_mut().flags = *compat::SCX_OPS_ENQ_EXITING
            | *compat::SCX_OPS_ENQ_LAST
            | *compat::SCX_OPS_ENQ_MIGRATION_DISABLED
            | *compat::SCX_OPS_ALLOW_QUEUED_WAKEUP;
        info!(
            "scheduler flags: {:#x}",
            skel.struct_ops.autoland_ops_mut().flags
        );

        // Load the BPF program for validation.
        let mut skel = scx_ops_load!(skel, autoland_ops, uei)?;

        // Initialize SMT domains.
        if smt_enabled {
            Self::init_smt_domains(&mut skel, &topo)?;
        }

        // Enable primary scheduling domain, if defined.
        if primary_cpus.len() < *NR_CPU_IDS {
            for cpu in primary_cpus {
                if let Err(err) = Self::enable_primary_cpu(&mut skel, cpu as i32) {
                    bail!("failed to add CPU {} to primary domain: error {}", cpu, err);
                }
            }
        }

        // Attach the scheduler.
        let struct_ops = Some(scx_ops_attach!(skel, autoland_ops)?);
        let stats_server = StatsServer::new(stats::server_data()).launch()?;

        // Initialize PID tail-latency-target based controller
        // This controller uses P99 latency measurements and PID control theory
        let controller =
            PidTailLatencyController::new(slice_lc, slice_normal, slice_hog, !opts.no_adaptive);
        let classifier = TaskClassifier::new();

        info!(
            "Adaptive controller: {}",
            if !opts.no_adaptive {
                "enabled"
            } else {
                "disabled"
            }
        );
        info!(
            "Per-class slices: LC={}us, Normal={}us, Hog={}us",
            slice_lc, slice_normal, slice_hog
        );
        info!("Preemption threshold: {}", opts.preempt_threshold);

        Ok(Self {
            skel,
            opts,
            struct_ops,
            stats_server,
            controller,
            classifier,
            detected_turbo_tasks: HashSet::new(),
            detected_game_tasks: HashSet::new(),
        })
    }

    fn enable_sibling_cpu(
        skel: &mut BpfSkel<'_>,
        cpu: usize,
        sibling_cpu: usize,
    ) -> Result<(), u32> {
        let prog = &mut skel.progs.enable_sibling_cpu;
        let mut args = domain_arg {
            cpu_id: cpu as c_int,
            sibling_cpu_id: sibling_cpu as c_int,
        };
        let input = ProgramInput {
            context_in: Some(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut args as *mut _ as *mut u8,
                    std::mem::size_of_val(&args),
                )
            }),
            ..Default::default()
        };
        let out = prog.test_run(input).unwrap();
        if out.return_value != 0 {
            return Err(out.return_value);
        }

        Ok(())
    }

    fn enable_primary_cpu(skel: &mut BpfSkel<'_>, cpu: i32) -> Result<(), u32> {
        let prog = &mut skel.progs.enable_primary_cpu;
        let mut args = cpu_arg {
            cpu_id: cpu as c_int,
        };
        let input = ProgramInput {
            context_in: Some(unsafe {
                std::slice::from_raw_parts_mut(
                    &mut args as *mut _ as *mut u8,
                    std::mem::size_of_val(&args),
                )
            }),
            ..Default::default()
        };
        let out = prog.test_run(input).unwrap();
        if out.return_value != 0 {
            return Err(out.return_value);
        }

        Ok(())
    }

    fn scan_and_classify_tasks(&mut self) {
        use std::fs;

        // Track which PIDs we see in this scan
        let mut current_turbo_tasks = HashSet::new();
        let mut current_game_tasks = HashSet::new();

        // Scan /proc for running processes
        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(Result::ok) {
                let filename = entry.file_name();
                let name_str = filename.to_string_lossy();

                // Check if it's a PID (numeric)
                if let Ok(pid) = name_str.parse::<i32>() {
                    // Read comm
                    let comm_path = format!("/proc/{}/comm", pid);
                    if let Ok(comm) = fs::read_to_string(&comm_path) {
                        let comm = comm.trim();

                        // Classify the task
                        let (is_scx_turbo, is_steam_game, is_de, is_audio) =
                            self.classifier.classify(pid as u32, comm);

                        // Determine class priority
                        let class_hint = if is_scx_turbo || is_steam_game {
                            // Track detected tasks
                            if is_scx_turbo {
                                current_turbo_tasks.insert(pid);
                            } else {
                                current_game_tasks.insert(pid);
                            }
                            2u32 // LATENCY_CRITICAL
                        } else if is_de || is_audio {
                            0u32 // NORMAL (default)
                        } else {
                            continue; // Don't override heuristics for unknown tasks
                        };

                        // Update BPF task hint
                        let hints_map = &mut self.skel.maps.task_hints;
                        let key = (pid as u32).to_le_bytes();
                        let value = class_hint.to_le_bytes();
                        if let Err(e) = hints_map.update(&key, &value, libbpf_rs::MapFlags::ANY) {
                            if self.opts.verbose {
                                debug!("Failed to update task hint for {}: {}", pid, e);
                            }
                        }
                    }
                }
            }
        }

        // Log newly detected turbo tasks
        for &pid in current_turbo_tasks.difference(&self.detected_turbo_tasks) {
            let comm_path = format!("/proc/{}/comm", pid);
            if let Ok(comm) = fs::read_to_string(&comm_path) {
                info!("Detected turbo task: PID={} comm={}", pid, comm.trim());
            }
        }

        // Log newly detected game tasks
        for &pid in current_game_tasks.difference(&self.detected_game_tasks) {
            let comm_path = format!("/proc/{}/comm", pid);
            if let Ok(comm) = fs::read_to_string(&comm_path) {
                info!("Detected Steam game: PID={} comm={}", pid, comm.trim());
            }
        }

        // Log removed turbo tasks
        for &pid in self.detected_turbo_tasks.difference(&current_turbo_tasks) {
            info!("Turbo task ended: PID={}", pid);
        }

        // Log removed game tasks
        for &pid in self.detected_game_tasks.difference(&current_game_tasks) {
            info!("Steam game ended: PID={}", pid);
        }

        // Update tracked state
        self.detected_turbo_tasks = current_turbo_tasks;
        self.detected_game_tasks = current_game_tasks;
    }

    fn init_smt_domains(skel: &mut BpfSkel<'_>, topo: &Topology) -> Result<(), std::io::Error> {
        let smt_siblings = topo.sibling_cpus();

        info!("SMT sibling CPUs: {:?}", smt_siblings);
        for (cpu, sibling_cpu) in smt_siblings.iter().enumerate() {
            Self::enable_sibling_cpu(skel, cpu, *sibling_cpu as usize).unwrap();
        }

        Ok(())
    }

    fn get_metrics(&mut self) -> Metrics {
        let bss_data = self.skel.maps.bss_data.as_ref().unwrap();
        let (lc_slice, normal_slice, hog_slice) = (
            self.skel.maps.rodata_data.as_ref().unwrap().slice_ns_lc / 1000,
            self.skel.maps.rodata_data.as_ref().unwrap().slice_ns_normal / 1000,
            self.skel.maps.rodata_data.as_ref().unwrap().slice_ns_hog / 1000,
        );

        Metrics {
            nr_local_dispatch: bss_data.nr_local_dispatch,
            nr_remote_dispatch: bss_data.nr_remote_dispatch,
            nr_keep_running: bss_data.nr_keep_running,
            nr_preemptions: 0,
            nr_lc_tasks: bss_data.nr_lc_tasks,
            nr_normal_tasks: bss_data.nr_normal_tasks,
            nr_hog_tasks: bss_data.nr_hog_tasks,
            lc_slice_us: lc_slice,
            normal_slice_us: normal_slice,
            hog_slice_us: hog_slice,
        }
    }

    pub fn exited(&mut self) -> bool {
        uei_exited!(&self.skel, uei)
    }

    fn compute_user_cpu_pct(prev: &CpuTimes, curr: &CpuTimes) -> Option<u64> {
        // Evaluate total user CPU time as user + nice.
        let user_diff = (curr.user + curr.nice).saturating_sub(prev.user + prev.nice);
        let total_diff = curr.total.saturating_sub(prev.total);

        if total_diff > 0 {
            let user_ratio = user_diff as f64 / total_diff as f64;
            Some((user_ratio * 1024.0).round() as u64)
        } else {
            None
        }
    }

    fn read_cpu_times() -> Option<CpuTimes> {
        let file = File::open("/proc/stat").ok()?;
        let reader = BufReader::new(file);

        for line in reader.lines() {
            let line = line.ok()?;
            if line.starts_with("cpu ") {
                let fields: Vec<&str> = line.split_whitespace().collect();
                if fields.len() < 5 {
                    return None;
                }

                let user: u64 = fields[1].parse().ok()?;
                let nice: u64 = fields[2].parse().ok()?;

                // Sum the first 8 fields as total time, including idle, system, etc.
                let total: u64 = fields
                    .iter()
                    .skip(1)
                    .take(8)
                    .filter_map(|v| v.parse::<u64>().ok())
                    .sum();

                return Some(CpuTimes { user, nice, total });
            }
        }

        None
    }

    fn run(&mut self, shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
        let (res_ch, req_ch) = self.stats_server.channels();

        // Periodically evaluate user CPU utilization from user-space and update a global variable
        // in BPF.
        //
        // The BPF scheduler can use this value to determine when the system is busy or idle.
        let polling_time = Duration::from_millis(self.opts.polling_ms).min(Duration::from_secs(1));
        let mut prev_cputime = Self::read_cpu_times().expect("Failed to read initial CPU stats");
        let mut last_update = Instant::now();

        // Adaptive controller update interval (1 second)
        let mut last_controller_update = Instant::now();
        let controller_interval = Duration::from_secs(1);

        // Task classification scan interval (5 seconds)
        let mut last_task_scan = Instant::now();
        let task_scan_interval = Duration::from_secs(5);

        while !shutdown.load(Ordering::Relaxed) && !self.exited() {
            // Scan and classify tasks periodically
            if last_task_scan.elapsed() >= task_scan_interval {
                self.scan_and_classify_tasks();
                last_task_scan = Instant::now();
            }
            // Update CPU utilization.
            if !polling_time.is_zero() && last_update.elapsed() >= polling_time {
                if let Some(curr_cputime) = Self::read_cpu_times() {
                    Self::compute_user_cpu_pct(&prev_cputime, &curr_cputime)
                        .map(|util| self.skel.maps.bss_data.as_mut().unwrap().cpu_util = util);
                    prev_cputime = curr_cputime;
                }
                last_update = Instant::now();
            }

            /*
             * PID Tail-Latency-Target Based Controller.
             * Adjusts slices based on measured P99 latencies using PID control theory.
             */
            if self.controller.is_enabled()
                && last_controller_update.elapsed() >= controller_interval
            {
                // Read latency statistics from BPF maps
                let mut latency_stats = [crate::bpf_intf::class_latency_stat {
                    sum_ns: 0,
                    count: 0,
                    sum_squares_ns: 0,
                    min_ns: 0,
                    max_ns: 0,
                    histogram: [0; 16],
                }; 3];

                let mut criticality_stats = [crate::bpf_intf::class_criticality_metrics {
                    total_wakeup_count: 0,
                    total_runtime_ns: 0,
                    runtime_squared_sum: 0,
                    sample_count: 0,
                }; 3];

                // Read from BPF maps in a separate scope to release borrows before updating rodata
                {
                    let latency_map = &self.skel.maps.class_latency_stats;
                    let criticality_map = &self.skel.maps.class_criticality_stats;

                    for class in 0u32..3 {
                        let key = class.to_le_bytes();

                        // Read latency stats
                        if let Ok(Some(data)) = latency_map.lookup(&key, libbpf_rs::MapFlags::ANY) {
                            // Parse the binary data into the struct
                            // The struct is: sum_ns(8), count(8), sum_squares_ns(8), min_ns(8), max_ns(8), histogram(128)
                            if data.len() >= 168 {
                                // 8 * 5 + 16 * 8 = 40 + 128 = 168
                                latency_stats[class as usize].sum_ns = u64::from_le_bytes([
                                    data[0], data[1], data[2], data[3], data[4], data[5], data[6],
                                    data[7],
                                ]);
                                latency_stats[class as usize].count = u64::from_le_bytes([
                                    data[8], data[9], data[10], data[11], data[12], data[13],
                                    data[14], data[15],
                                ]);
                                latency_stats[class as usize].sum_squares_ns =
                                    u64::from_le_bytes([
                                        data[16], data[17], data[18], data[19], data[20], data[21],
                                        data[22], data[23],
                                    ]);
                                latency_stats[class as usize].min_ns = u64::from_le_bytes([
                                    data[24], data[25], data[26], data[27], data[28], data[29],
                                    data[30], data[31],
                                ]);
                                latency_stats[class as usize].max_ns = u64::from_le_bytes([
                                    data[32], data[33], data[34], data[35], data[36], data[37],
                                    data[38], data[39],
                                ]);
                                // Read histogram (16 u64 values starting at offset 40)
                                for i in 0..16 {
                                    let offset = 40 + i * 8;
                                    latency_stats[class as usize].histogram[i] =
                                        u64::from_le_bytes([
                                            data[offset],
                                            data[offset + 1],
                                            data[offset + 2],
                                            data[offset + 3],
                                            data[offset + 4],
                                            data[offset + 5],
                                            data[offset + 6],
                                            data[offset + 7],
                                        ]);
                                }
                            }
                        }

                        // Read criticality stats
                        if let Ok(Some(data)) =
                            criticality_map.lookup(&key, libbpf_rs::MapFlags::ANY)
                        {
                            // The struct is: total_wakeup_count(8), total_runtime_ns(8), runtime_squared_sum(8), sample_count(4)
                            if data.len() >= 28 {
                                // 8 + 8 + 8 + 4 = 28
                                criticality_stats[class as usize].total_wakeup_count =
                                    u64::from_le_bytes([
                                        data[0], data[1], data[2], data[3], data[4], data[5],
                                        data[6], data[7],
                                    ]);
                                criticality_stats[class as usize].total_runtime_ns =
                                    u64::from_le_bytes([
                                        data[8], data[9], data[10], data[11], data[12], data[13],
                                        data[14], data[15],
                                    ]);
                                criticality_stats[class as usize].runtime_squared_sum =
                                    u64::from_le_bytes([
                                        data[16], data[17], data[18], data[19], data[20], data[21],
                                        data[22], data[23],
                                    ]);
                                criticality_stats[class as usize].sample_count =
                                    u32::from_le_bytes([data[24], data[25], data[26], data[27]]);
                            }
                        }
                    }
                }

                // Run controller update
                let (lc_slice, normal_slice, hog_slice, preempt_threshold) =
                    self.controller.update(&latency_stats, &criticality_stats);
                last_controller_update = Instant::now();

                // NOTE: BPF slice_ns_lc, slice_ns_normal, slice_ns_hog, and preempt_threshold
                // are const volatile, meaning they can only be set at load time, not runtime.
                // The controller computes optimal values, but we cannot update BPF with them.
                // TODO: Add mutable BPF variables for runtime slice adjustment.

                // Log with latency information for debugging
                let bss_data = self.skel.maps.bss_data.as_ref().unwrap();
                let nr_lc = bss_data.nr_lc_tasks;
                let nr_normal = bss_data.nr_normal_tasks;
                let nr_hog = bss_data.nr_hog_tasks;
                let nr_total = nr_lc + nr_normal + nr_hog;

                // Only log PID controller updates if --stats flag is provided
                if self.opts.stats.is_some() {
                    if nr_total > 0 {
                        let lc_pct = (nr_lc as f64 / nr_total as f64 * 100.0) as u32;
                        let normal_pct = (nr_normal as f64 / nr_total as f64 * 100.0) as u32;
                        let hog_pct = (nr_hog as f64 / nr_total as f64 * 100.0) as u32;

                        info!("PID Controller: LC={}us({}%) Normal={}us({}%) Hog={}us({}%) Preempt={} Tasks=[{}/{}/{}]",
                              lc_slice, lc_pct, normal_slice, normal_pct, hog_slice, hog_pct,
                              preempt_threshold, nr_lc, nr_normal, nr_hog);
                    } else {
                        info!(
                            "PID Controller: LC={}us Normal={}us Hog={}us Preempt={} (no tasks)",
                            lc_slice, normal_slice, hog_slice, preempt_threshold
                        );
                    }
                }
            }

            // Update statistics and check for exit condition.
            let timeout = if polling_time.is_zero() {
                Duration::from_secs(1)
            } else {
                polling_time
            };
            match req_ch.recv_timeout(timeout) {
                Ok(()) => res_ch.send(self.get_metrics())?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(e) => Err(e)?,
            }
        }

        let _ = self.struct_ops.take();
        uei_report!(&self.skel, uei)
    }
}

impl Drop for Scheduler<'_> {
    fn drop(&mut self) {
        info!("Unregister {SCHEDULER_NAME} scheduler");
    }
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    if opts.version {
        println!(
            "{} {}",
            SCHEDULER_NAME,
            build_id::full_version(env!("CARGO_PKG_VERSION"))
        );
        return Ok(());
    }

    if opts.help_stats {
        stats::server_data().describe_meta(&mut std::io::stdout(), None)?;
        return Ok(());
    }

    let loglevel = simplelog::LevelFilter::Info;

    let mut lcfg = simplelog::ConfigBuilder::new();
    lcfg.set_time_offset_to_local()
        .expect("Failed to set local time offset")
        .set_time_level(simplelog::LevelFilter::Error)
        .set_location_level(simplelog::LevelFilter::Off)
        .set_target_level(simplelog::LevelFilter::Off)
        .set_thread_level(simplelog::LevelFilter::Off);
    simplelog::TermLogger::init(
        loglevel,
        lcfg.build(),
        simplelog::TerminalMode::Stderr,
        simplelog::ColorChoice::Auto,
    )?;

    let shutdown = Arc::new(AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    ctrlc::set_handler(move || {
        shutdown_clone.store(true, Ordering::Relaxed);
    })
    .context("Error setting Ctrl-C handler")?;

    if let Some(intv) = opts.monitor.or(opts.stats) {
        let shutdown_copy = shutdown.clone();
        let jh = std::thread::spawn(move || {
            match stats::monitor(Duration::from_secs_f64(intv), shutdown_copy) {
                Ok(_) => {
                    debug!("stats monitor thread finished successfully")
                }
                Err(error_object) => {
                    warn!(
                        "stats monitor thread finished because of an error {}",
                        error_object
                    )
                }
            }
        });
        if opts.monitor.is_some() {
            let _ = jh.join();
            return Ok(());
        }
    }

    let mut open_object = MaybeUninit::uninit();
    loop {
        let mut sched = Scheduler::init(&opts, &mut open_object)?;
        if !sched.run(shutdown.clone())?.should_restart() {
            break;
        }
    }

    Ok(())
}
