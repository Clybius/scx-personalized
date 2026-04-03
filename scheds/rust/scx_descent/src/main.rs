// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2024 Andrea Righi <arighi@nvidia.com>

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;
pub use bpf_intf::*;

mod classifier;
mod optimizer_pie;
mod profiles;
mod safety;
mod stats;

use std::collections::HashSet;
use std::ffi::c_int;
use std::fmt::Write;
use std::fs;
use std::mem::MaybeUninit;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use classifier::TaskClassifier;
use crossbeam::channel::RecvTimeoutError;
use libbpf_rs::MapCore;
use libbpf_rs::MapFlags;
use libbpf_rs::OpenObject;
use libbpf_rs::ProgramInput;
use log::{debug, info, warn};
use optimizer_pie::PieController;
use profiles::Profile;
use safety::SafetyMonitor;
use scx_stats::prelude::*;
use scx_utils::autopower::{fetch_power_profile, PowerProfile};
use scx_utils::build_id;
use scx_utils::compat;
use scx_utils::libbpf_clap_opts::LibbpfOpts;
use scx_utils::pm::{cpu_idle_resume_latency_supported, update_cpu_idle_resume_latency};
use scx_utils::scx_ops_attach;
use scx_utils::scx_ops_load;
use scx_utils::scx_ops_open;
use scx_utils::try_set_rlimit_infinity;
use scx_utils::uei_exited;
use scx_utils::uei_report;
use scx_utils::CoreType;
use scx_utils::Cpumask;
use scx_utils::Topology;
use scx_utils::UserExitInfo;
use scx_utils::NR_CPU_IDS;
use stats::Metrics;

const SCHEDULER_NAME: &str = "scx_descent";

#[derive(PartialEq)]
enum Powermode {
    Turbo,
    Performance,
    Powersave,
    Any,
}

fn get_primary_cpus(mode: Powermode) -> std::io::Result<Vec<usize>> {
    let topo = Topology::new().unwrap();

    let cpus: Vec<usize> = topo
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

// Convert an array of CPUs to the corresponding cpumask of any arbitrary size.
fn cpus_to_cpumask(cpus: &Vec<usize>) -> String {
    if cpus.is_empty() {
        return String::from("none");
    }

    // Determine the maximum CPU ID to create a sufficiently large byte vector.
    let max_cpu_id = *cpus.iter().max().unwrap();

    // Create a byte vector with enough bytes to cover all CPU IDs.
    let mut bitmask = vec![0u8; (max_cpu_id + 1 + 7) / 8];

    // Set the appropriate bits for each CPU ID.
    for cpu_id in cpus {
        let byte_index = cpu_id / 8;
        let bit_index = cpu_id % 8;
        bitmask[byte_index] |= 1 << bit_index;
    }

    // Convert the byte vector to a hexadecimal string.
    let hex_str: String = bitmask.iter().rev().fold(String::new(), |mut f, byte| {
        let _ = write!(&mut f, "{:02x}", byte);
        f
    });

    format!("0x{}", hex_str)
}

#[derive(Debug, clap::Parser)]
#[command(
    name = "scx_descent",
    version,
    disable_version_flag = true,
    about = "A PIE controller-based scheduler that automatically optimizes scheduling parameters for different workload classes.",
    long_about = r#"
scx_descent is a scheduler that uses a PIE (Proportional Integral controller Enhanced) to automatically
optimize scheduling parameters for different workload classes (latency_critical, normal, hog, background).

It operates using an earliest deadline first (EDF) policy with per-class tunable parameters:

1. latency_weight (θ₁): Virtual deadline offset
2. base_slice_ns (θ₂): Preferred time slice
3. vruntime_scale (θ₃): Vruntime multiplier
4. preemption_priority (θ₄): Urgency threshold
5. migration_cost (θ₅): Cross-CPU migration penalty

These parameters are optimized via PIE controller to minimize latency error from target.

Key features:
- Automatic task classification into workload classes
- Per-class parameter optimization via PIE controller
- Three optimization profiles: gaming, productivity, server
- Safety mechanisms to prevent parameter oscillation
- Deterministic control without random exploration
"#
)]
struct Opts {
    /// Exit debug dump buffer length. 0 indicates default.
    #[clap(long, default_value = "0")]
    exit_dump_len: u32,

    /// Maximum scheduling slice duration in microseconds.
    #[clap(short = 's', long, default_value = "700")]
    slice_us: u64,

    /// Maximum runtime budget that a task can accumulate while sleeping (in microseconds).
    #[clap(short = 'l', long, default_value = "20000")]
    slice_us_lag: u64,

    /// Throttle the running CPUs by periodically injecting idle cycles.
    #[clap(short = 't', long, default_value = "0")]
    throttle_us: u64,

    /// Set CPU idle QoS resume latency in microseconds (-1 = disabled).
    #[clap(short = 'I', long, allow_hyphen_values = true, default_value = "32")]
    idle_resume_us: i64,

    /// Enable tickless mode.
    #[clap(short = 'T', long, action = clap::ArgAction::SetTrue)]
    tickless: bool,

    /// Enable round-robin scheduling.
    #[clap(short = 'R', long, action = clap::ArgAction::SetTrue)]
    rr_sched: bool,

    /// Specifies the initial set of CPUs as a bitmask in hex (e.g., 0xff).
    #[clap(short = 'm', long, default_value = "auto")]
    primary_domain: String,

    /// Disable SMT awareness.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    disable_smt: bool,

    /// Disable NUMA rebalancing.
    #[clap(long, action = clap::ArgAction::SetTrue)]
    disable_numa: bool,

    /// Enable CPU frequency control (only with schedutil governor).
    #[clap(short = 'f', long, action = clap::ArgAction::SetTrue)]
    cpufreq: bool,

    /// Profile selection (gaming, productivity, server)
    #[clap(short = 'p', long, default_value = "productivity")]
    profile: String,

    /// Enable PIE controller debug output every N ms
    #[clap(long, value_name = "N")]
    debug_pie: Option<u64>,

    /// Update interval for parameter sync (ms)
    #[clap(long, default_value = "50")]
    update_interval_ms: u64,

    /// Audio cgroup path for automatic classification
    #[clap(long)]
    audio_cgroup: Option<String>,

    /// Enable stats monitoring with the specified interval.
    #[clap(long)]
    stats: Option<f64>,

    /// Run in stats monitoring mode with the specified interval. Scheduler
    /// is not launched.
    #[clap(long)]
    monitor: Option<f64>,

    /// Enable BPF debugging via /sys/kernel/tracing/trace_pipe.
    #[clap(short = 'd', long, action = clap::ArgAction::SetTrue)]
    debug: bool,

    /// Enable verbose output, including libbpf details.
    #[clap(short = 'v', long, action = clap::ArgAction::SetTrue)]
    verbose: bool,

    /// Print scheduler version and exit.
    #[clap(short = 'V', long, action = clap::ArgAction::SetTrue)]
    version: bool,

    /// Show descriptions for statistics.
    #[clap(long)]
    help_stats: bool,

    #[clap(flatten, next_help_heading = "Libbpf Options")]
    pub libbpf: LibbpfOpts,
}

// Shared counters for metrics
#[allow(dead_code)] // Reserved for future metrics implementation
static GRADIENT_UPDATES: AtomicU64 = AtomicU64::new(0);
#[allow(dead_code)] // Reserved for future metrics implementation
static OSCILLATIONS: AtomicU64 = AtomicU64::new(0);

/// Matches `struct class_loss_accumulator` from BPF (descent.bpf.h)
/// Layout: 4 x u64 + 1 x u32 = 32 bytes (with 4 bytes implicit padding)
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ClassLossAccumulator {
    pub latency_loss_sum: u64, // Sum of squared wakeup latencies
    pub deadline_misses: u64,  // Count of scheduling deadline misses
    pub cpu_time_ns: u64,      // Total CPU time consumed
    pub target_share_ns: u64,  // Expected fair share
    pub sample_count: u32,     // Number of samples in this window
                               // Implicit 4 bytes padding to align to 8-byte boundary
}

/// Size of class_loss_accumulator in bytes (matches BPF struct size)
/// BPF struct: 4 x u64 (32 bytes) + 1 x u32 (4 bytes) + 4 bytes padding = 40 bytes
const CLASS_LOSS_ACCUMULATOR_SIZE: usize = 40;

/// Number of task classes
const DESCENT_CLASS_MAX: usize = 4;

/// Offset to class_loss array within cpu_descent_ctx
/// class_params[4] = 4 * (5 * 8 bytes) = 160 bytes
const CLASS_LOSS_OFFSET: usize = 160;

/// Latency metrics structure for PIE controller
#[derive(Debug, Default)]
pub struct LatencyMetrics {
    pub total_latency_ns: u64,
    pub max_latency_ns: u64,
    pub sample_count: u64,
}

/// Helper to safely extract u64 from native-endian bytes
fn read_u64(bytes: &[u8]) -> u64 {
    if bytes.len() >= 8 {
        u64::from_ne_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    } else {
        0
    }
}

struct Scheduler<'a> {
    skel: BpfSkel<'a>,
    struct_ops: Option<libbpf_rs::Link>,
    opts: &'a Opts,
    topo: Topology,
    power_profile: PowerProfile,
    stats_server: StatsServer<(), Metrics>,
    user_restart: bool,
    pie: PieController,
    _classifier: TaskClassifier,
    safety: SafetyMonitor,
    profile: Profile,
    nr_cpus: usize,
}

impl<'a> Scheduler<'a> {
    fn init(opts: &'a Opts, open_object: &'a mut MaybeUninit<OpenObject>) -> Result<Self> {
        try_set_rlimit_infinity();

        // Initialize CPU topology.
        let topo = Topology::new().unwrap();

        // Check host topology to determine if we need to enable SMT capabilities.
        let smt_enabled = !opts.disable_smt && topo.smt_enabled;

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

        // Load profile configuration using inherent method
        let profile = Profile::from_str(&opts.profile).unwrap_or_else(|| Profile::default());
        info!("Using profile: {}", profile);

        let nr_cpus = topo.all_cpus.len();

        if opts.idle_resume_us >= 0 {
            if !cpu_idle_resume_latency_supported() {
                warn!("idle resume latency not supported");
            } else {
                info!("Setting idle QoS to {} us", opts.idle_resume_us);
                for cpu in topo.all_cpus.values() {
                    update_cpu_idle_resume_latency(
                        cpu.id,
                        opts.idle_resume_us.try_into().unwrap(),
                    )?;
                }
            }
        }

        // Determine the amount of non-empty NUMA nodes in the system.
        let nr_nodes = topo
            .nodes
            .values()
            .filter(|node| !node.all_cpus.is_empty())
            .count();
        info!("NUMA nodes: {}", nr_nodes);

        // Automatically disable NUMA optimizations when running on non-NUMA systems.
        let numa_disabled = opts.disable_numa || nr_nodes == 1;
        if numa_disabled {
            info!("Disabling NUMA optimizations");
        }

        // Determine the primary scheduling domain.
        let power_profile = Self::power_profile();
        let domain =
            Self::resolve_energy_domain(&opts.primary_domain, power_profile).map_err(|err| {
                anyhow!(
                    "failed to resolve primary domain '{}': {}",
                    &opts.primary_domain,
                    err
                )
            })?;

        // Initialize BPF connector.
        let mut skel_builder = BpfSkelBuilder::default();
        skel_builder.obj_builder.debug(opts.verbose);
        let open_opts = opts.libbpf.clone().into_bpf_open_opts();
        let mut skel = scx_ops_open!(skel_builder, open_object, descent_ops, open_opts)?;

        skel.struct_ops.descent_ops_mut().exit_dump_len = opts.exit_dump_len;

        // Override default BPF scheduling parameters.
        let rodata = skel.maps.rodata_data.as_mut().unwrap();
        rodata.debug = opts.debug;
        rodata.smt_enabled = smt_enabled;
        rodata.numa_disabled = numa_disabled;
        rodata.rr_sched = opts.rr_sched;
        rodata.tickless_sched = opts.tickless;
        rodata.slice_max = opts.slice_us * 1000;
        rodata.slice_lag = opts.slice_us_lag * 1000;
        rodata.throttle_ns = opts.throttle_us * 1000;
        rodata.primary_all = domain.weight() == *NR_CPU_IDS;

        // Set scheduler flags.
        skel.struct_ops.descent_ops_mut().flags = *compat::SCX_OPS_ENQ_EXITING
            | *compat::SCX_OPS_ENQ_LAST
            | *compat::SCX_OPS_ENQ_MIGRATION_DISABLED
            | *compat::SCX_OPS_ALLOW_QUEUED_WAKEUP
            | if numa_disabled {
                0
            } else {
                *compat::SCX_OPS_BUILTIN_IDLE_PER_NODE
            };
        info!(
            "scheduler flags: {:#x}",
            skel.struct_ops.descent_ops_mut().flags
        );

        // Load the BPF program for validation.
        let mut skel = scx_ops_load!(skel, descent_ops, uei)?;

        // Initialize the primary scheduling domain and the preferred domain.
        Self::init_energy_domain(&mut skel, &domain).map_err(|err| {
            anyhow!(
                "failed to initialize primary domain 0x{:x}: {}",
                domain,
                err
            )
        })?;

        if let Err(err) = Self::init_cpufreq_perf(&mut skel, &opts.primary_domain, opts.cpufreq) {
            bail!(
                "failed to initialize cpufreq performance level: error {}",
                err
            );
        }

        // Initialize SMT domains.
        if smt_enabled {
            Self::init_smt_domains(&mut skel, &topo)?;
        }

        // Attach the scheduler.
        let struct_ops = Some(scx_ops_attach!(skel, descent_ops)?);
        let stats_server = StatsServer::new(stats::server_data()).launch()?;

        // Initialize Phase 4 components - PIE Controller
        let pie = PieController::new(nr_cpus, &profile);
        let classifier = TaskClassifier::new();
        let safety = SafetyMonitor::new();

        Ok(Self {
            skel,
            struct_ops,
            opts,
            topo,
            power_profile,
            stats_server,
            user_restart: false,
            pie,
            _classifier: classifier,
            safety,
            profile,
            nr_cpus,
        })
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

    fn epp_to_cpumask(profile: Powermode) -> Result<Cpumask> {
        let mut cpus = get_primary_cpus(profile).unwrap_or_default();
        if cpus.is_empty() {
            cpus = get_primary_cpus(Powermode::Any).unwrap_or_default();
        }
        Cpumask::from_str(&cpus_to_cpumask(&cpus))
    }

    fn resolve_energy_domain(primary_domain: &str, power_profile: PowerProfile) -> Result<Cpumask> {
        let domain = match primary_domain {
            "powersave" => Self::epp_to_cpumask(Powermode::Powersave)?,
            "performance" => Self::epp_to_cpumask(Powermode::Performance)?,
            "turbo" => Self::epp_to_cpumask(Powermode::Turbo)?,
            "auto" => match power_profile {
                PowerProfile::Powersave => Self::epp_to_cpumask(Powermode::Powersave)?,
                PowerProfile::Balanced { power: true } => {
                    Self::epp_to_cpumask(Powermode::Powersave)?
                }
                PowerProfile::Balanced { power: false }
                | PowerProfile::Performance
                | PowerProfile::Unknown => Self::epp_to_cpumask(Powermode::Any)?,
            },
            "all" => Self::epp_to_cpumask(Powermode::Any)?,
            &_ => Cpumask::from_str(primary_domain)?,
        };

        Ok(domain)
    }

    fn init_energy_domain(skel: &mut BpfSkel<'_>, domain: &Cpumask) -> Result<()> {
        info!("primary CPU domain = 0x{:x}", domain);

        // Clear the primary domain by passing a negative CPU id.
        if let Err(err) = Self::enable_primary_cpu(skel, -1) {
            bail!("failed to reset primary domain: error {}", err);
        }

        // Update primary scheduling domain.
        for cpu in 0..*NR_CPU_IDS {
            if domain.test_cpu(cpu) {
                if let Err(err) = Self::enable_primary_cpu(skel, cpu as i32) {
                    bail!("failed to add CPU {} to primary domain: error {}", cpu, err);
                }
            }
        }

        Ok(())
    }

    // Update hint for the cpufreq governor.
    fn init_cpufreq_perf(
        skel: &mut BpfSkel<'_>,
        primary_domain: &String,
        auto: bool,
    ) -> Result<()> {
        // If we are using the powersave profile always scale the CPU frequency to the minimum,
        // otherwise use the maximum, unless automatic frequency scaling is enabled.
        let perf_lvl: i64 = match primary_domain.as_str() {
            "powersave" => 0,
            _ if auto => -1,
            _ => 1024,
        };
        info!(
            "cpufreq performance level: {}",
            match perf_lvl {
                1024 => "max".into(),
                0 => "min".into(),
                n if n < 0 => "auto".into(),
                _ => perf_lvl.to_string(),
            }
        );
        skel.maps.bss_data.as_mut().unwrap().cpufreq_perf_lvl = perf_lvl;

        Ok(())
    }

    fn power_profile() -> PowerProfile {
        let profile = fetch_power_profile(true);
        if profile == PowerProfile::Unknown {
            fetch_power_profile(false)
        } else {
            profile
        }
    }

    fn refresh_sched_domain(&mut self) -> bool {
        if self.power_profile != PowerProfile::Unknown {
            let power_profile = Self::power_profile();
            if power_profile != self.power_profile {
                self.power_profile = power_profile;

                if self.opts.primary_domain == "auto" {
                    return true;
                }
                if let Err(err) = Self::init_cpufreq_perf(
                    &mut self.skel,
                    &self.opts.primary_domain,
                    self.opts.cpufreq,
                ) {
                    warn!("failed to refresh cpufreq performance level: error {}", err);
                }
            }
        }

        false
    }

    fn enable_sibling_cpu(
        skel: &mut BpfSkel<'_>,
        lvl: usize,
        cpu: usize,
        sibling_cpu: usize,
    ) -> Result<(), u32> {
        let prog = &mut skel.progs.enable_sibling_cpu;
        let mut args = domain_arg {
            lvl_id: lvl as c_int,
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

    fn init_smt_domains(skel: &mut BpfSkel<'_>, topo: &Topology) -> Result<(), std::io::Error> {
        let smt_siblings = topo.sibling_cpus();

        info!("SMT sibling CPUs: {:?}", smt_siblings);
        for (cpu, sibling_cpu) in smt_siblings.iter().enumerate() {
            Self::enable_sibling_cpu(skel, 0, cpu, *sibling_cpu as usize).unwrap();
        }

        Ok(())
    }

    fn get_metrics(&self) -> Metrics {
        let bss_data = self.skel.maps.bss_data.as_ref().unwrap();

        // Get PIE stats
        let pie_stats = self.pie.get_stats();

        // Calculate average latency across all states
        let mut total_latency = 0u64;
        let mut latency_count = 0u64;
        let mut total_integral = 0i64;
        let mut total_error = 0i64;

        for cpu in 0..self.nr_cpus as u32 {
            for class in 0..4u32 {
                if let Some(state) = self.pie.get_state(cpu, class) {
                    total_latency += state.current_latency_ns;
                    latency_count += 1;
                    total_integral += state.integral_accum;

                    let error = state.current_latency_ns as i64 - state.target_latency_ns as i64;
                    total_error += error;
                }
            }
        }

        let avg_latency = if latency_count > 0 {
            total_latency / latency_count
        } else {
            0
        };
        let avg_error = if latency_count > 0 {
            total_error / latency_count as i64
        } else {
            0
        };

        Metrics {
            nr_running: bss_data.nr_running,
            nr_cpus: bss_data.nr_online_cpus,
            nr_kthread_dispatches: bss_data.nr_kthread_dispatches,
            nr_direct_dispatches: bss_data.nr_direct_dispatches,
            nr_shared_dispatches: bss_data.nr_shared_dispatches,

            // PIE controller metrics
            nr_tasks_latency_critical: 0, // TODO: read from BPF if available
            nr_tasks_normal: 0,
            nr_tasks_hog: 0,
            nr_tasks_background: 0,
            pie_updates: pie_stats.total_updates,
            pie_avg_latency_us: avg_latency / 1000, // Convert ns to µs
            pie_target_latency_us: 0,               // TODO: get from profile
            pie_integral: total_integral / 1024,    // De-scale
            pie_latency_error_us: avg_error / 1000, // Convert to µs
        }
    }

    /// Detect audio daemon TGIDs by scanning comm names
    fn detect_audio_daemons(&self) -> Vec<u32> {
        let mut audio_tgids = HashSet::new();

        const AUDIO_COMMS: &[&str] = &[
            "pipewire",
            "wireplumber",
            "pipewire-pulse",
            "pulseaudio",
            "jackd",
            "jackdbus",
        ];

        // Scan /proc for matching comm names
        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();
                if let Ok(pid) = pid_str.parse::<u32>() {
                    if let Ok(comm) = fs::read_to_string(format!("/proc/{}/comm", pid)) {
                        let comm = comm.trim();
                        if AUDIO_COMMS.iter().any(|&ac| comm.contains(ac)) {
                            // Get TGID from status
                            if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", pid))
                            {
                                for line in status.lines() {
                                    if line.starts_with("Tgid:") {
                                        if let Some(tgid_str) = line.split_whitespace().nth(1) {
                                            if let Ok(tgid) = tgid_str.parse::<u32>() {
                                                audio_tgids.insert(tgid);
                                            }
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        audio_tgids.into_iter().collect()
    }

    /// Detect game process via Steam envvar or Wine exe
    fn detect_game_process(&self) -> Option<(u32, u32, u8)> {
        // Scan /proc for game indicators
        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();
                if let Ok(pid) = pid_str.parse::<u32>() {
                    // Check for Steam environment
                    if let Ok(environ) = fs::read(format!("/proc/{}/environ", pid)) {
                        let has_steam = environ
                            .split(|&b| b == 0)
                            .filter_map(|kv| std::str::from_utf8(kv).ok())
                            .any(|s| s.starts_with("SteamGameId=") || s.starts_with("STEAM_GAME="));

                        if has_steam {
                            // Get TGID and PPID from status
                            if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", pid))
                            {
                                let mut tgid = pid;
                                let mut ppid = 0u32;
                                for line in status.lines() {
                                    if line.starts_with("Tgid:") {
                                        if let Some(t) = line.split_whitespace().nth(1) {
                                            tgid = t.parse().unwrap_or(pid);
                                        }
                                    }
                                    if line.starts_with("PPid:") {
                                        if let Some(p) = line.split_whitespace().nth(1) {
                                            ppid = p.parse().unwrap_or(0);
                                        }
                                    }
                                }
                                return Some((tgid, ppid, 100)); // 100 = Steam confidence
                            }
                        }
                    }

                    // Check for Wine exe
                    if let Ok(cmdline) = fs::read(format!("/proc/{}/cmdline", pid)) {
                        let has_exe = cmdline
                            .split(|&b| b == 0)
                            .filter_map(|arg| std::str::from_utf8(arg).ok())
                            .any(|s| s.to_lowercase().ends_with(".exe"));

                        if has_exe {
                            if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", pid))
                            {
                                let mut tgid = pid;
                                let mut ppid = 0u32;
                                for line in status.lines() {
                                    if line.starts_with("Tgid:") {
                                        if let Some(t) = line.split_whitespace().nth(1) {
                                            tgid = t.parse().unwrap_or(pid);
                                        }
                                    }
                                    if line.starts_with("PPid:") {
                                        if let Some(p) = line.split_whitespace().nth(1) {
                                            ppid = p.parse().unwrap_or(0);
                                        }
                                    }
                                }
                                return Some((tgid, ppid, 90)); // 90 = Wine confidence
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Detect system state (GAMING, COMPILATION, IDLE)
    fn detect_sched_state(&self, game_tgid: u32) -> u32 {
        // GAMING: game detected
        if game_tgid != 0 {
            return 2; // GAMING
        }

        // COMPILATION: ≥2 compilers with high CPU usage
        const COMPILE_COMMS: &[&str] = &[
            "cc1", "rustc", "clang", "clang++", "ld", "ld.lld", "ninja", "cmake", "as", "gcc",
            "g++",
        ];

        let mut compile_count = 0;
        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();
                if let Ok(pid) = pid_str.parse::<u32>() {
                    if let Ok(comm) = fs::read_to_string(format!("/proc/{}/comm", pid)) {
                        let comm = comm.trim();
                        if COMPILE_COMMS.iter().any(|&c| comm.contains(c)) {
                            compile_count += 1;
                            if compile_count >= 2 {
                                return 1; // COMPILATION
                            }
                        }
                    }
                }
            }
        }

        0 // IDLE
    }

    /// Update BPF BSS variables for game detection
    fn update_bpf_game_state(&mut self, game_tgid: u32, game_ppid: u32, game_confidence: u8) {
        if let Some(bss_data) = self.skel.maps.bss_data.as_mut() {
            bss_data.game_tgid = game_tgid;
            bss_data.game_ppid = game_ppid;
            bss_data.game_confidence = game_confidence;
        }
    }

    /// Update BPF audio TGIDs
    fn update_bpf_audio_tgids(&mut self, audio_tgids: &[u32]) {
        if let Some(bss_data) = self.skel.maps.bss_data.as_mut() {
            let nr_audio = audio_tgids.len().min(16);
            bss_data.nr_audio_tgids = nr_audio as u32;
            for (i, &tgid) in audio_tgids.iter().take(16).enumerate() {
                bss_data.audio_tgids[i] = tgid;
            }
        }
    }

    /// Update BPF sched state
    fn update_bpf_sched_state(&mut self, state: u32) {
        if let Some(bss_data) = self.skel.maps.bss_data.as_mut() {
            bss_data.sched_state = state;
        }
    }

    pub fn exited(&mut self) -> bool {
        uei_exited!(&self.skel, uei)
    }

    /// Read accumulated loss from BPF for a CPU/class using safe struct parsing
    fn read_loss_from_bpf(&self, cpu: i32, class: u32) -> f64 {
        if class as usize >= DESCENT_CLASS_MAX {
            return 0.0;
        }

        // Use libbpf-rs to lookup per-CPU element
        let key: u32 = 0;

        match self
            .skel
            .maps
            .cpu_descent_ctx_stor
            .lookup_percpu(&key.to_ne_bytes(), MapFlags::ANY)
        {
            Ok(Some(values)) => {
                // values is Vec<Vec<u8>> where each element is data for a CPU
                // Get the specific CPU's data
                let cpu_idx = cpu as usize;
                if cpu_idx >= values.len() {
                    return 0.0;
                }
                let data = &values[cpu_idx];

                // Calculate offset to this class's class_loss[class_id]
                let class_offset =
                    CLASS_LOSS_OFFSET + (class as usize * CLASS_LOSS_ACCUMULATOR_SIZE);

                // Ensure we have enough data
                if data.len() < class_offset + CLASS_LOSS_ACCUMULATOR_SIZE {
                    return 0.0;
                }

                // Parse the ClassLossAccumulator fields using safe byte conversion
                let accumulator_data =
                    &data[class_offset..class_offset + CLASS_LOSS_ACCUMULATOR_SIZE];

                // Helper to safely extract u64 from native-endian bytes
                fn read_u64(bytes: &[u8]) -> u64 {
                    if bytes.len() >= 8 {
                        u64::from_ne_bytes([
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6],
                            bytes[7],
                        ])
                    } else {
                        0
                    }
                }

                // Helper to safely extract u32 from native-endian bytes
                fn read_u32(bytes: &[u8]) -> u32 {
                    if bytes.len() >= 4 {
                        u32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                    } else {
                        0
                    }
                }

                let accum = ClassLossAccumulator {
                    latency_loss_sum: read_u64(&accumulator_data[0..8]),
                    deadline_misses: read_u64(&accumulator_data[8..16]),
                    cpu_time_ns: read_u64(&accumulator_data[16..24]),
                    target_share_ns: read_u64(&accumulator_data[24..32]),
                    sample_count: read_u32(&accumulator_data[32..36]), // sample_count is at offset 32-36 (after 4 u64 fields)
                };

                // Compute composite loss (same formula as BPF)
                if accum.sample_count == 0 {
                    return 0.0;
                }

                let avg_latency_loss = accum.latency_loss_sum / accum.sample_count as u64;
                let deadline_penalty = accum.deadline_misses * 10; // 10ms per miss

                (avg_latency_loss + deadline_penalty) as f64
            }
            Ok(None) => 0.0,
            Err(e) => {
                eprintln!("Error reading loss from BPF: {:?}", e);
                0.0
            }
        }
    }

    /// Read latency metrics from BPF for a CPU/class for PIE controller
    fn read_latency_metrics(&self, cpu: i32, class: u32) -> LatencyMetrics {
        if class as usize >= DESCENT_CLASS_MAX {
            return LatencyMetrics::default();
        }

        let key: u32 = 0;

        match self
            .skel
            .maps
            .cpu_descent_ctx_stor
            .lookup_percpu(&key.to_ne_bytes(), MapFlags::ANY)
        {
            Ok(Some(values)) => {
                let cpu_idx = cpu as usize;
                if cpu_idx >= values.len() {
                    return LatencyMetrics::default();
                }
                let data = &values[cpu_idx];

                // Calculate offset to class_latency[class_id]
                // class_params[4] = 4 * (5 * 8 bytes) = 160 bytes
                // class_latency starts after class_params
                let class_offset = 160 + (class as usize * 24); // 24 bytes per latency_accumulator

                if data.len() < class_offset + 24 {
                    return LatencyMetrics::default();
                }

                let accum_data = &data[class_offset..class_offset + 24];

                LatencyMetrics {
                    total_latency_ns: read_u64(&accum_data[0..8]),
                    max_latency_ns: read_u64(&accum_data[8..16]),
                    sample_count: read_u64(&accum_data[16..24]),
                }
            }
            Ok(None) => LatencyMetrics::default(),
            Err(_) => LatencyMetrics::default(),
        }
    }

    /// Update BPF class parameters using syscall program
    fn update_bpf_params(&mut self, cpu: i32, class: u32, params: [u64; 5]) {
        let prog = &mut self.skel.progs.update_class_params;

        let mut args = descent_params_update {
            cpu_id: cpu,
            class_id: class,
            latency_weight: params[0],
            base_slice_ns: params[1],
            vruntime_scale: params[2],
            preemption_priority: params[3],
            migration_cost: params[4],
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

        match prog.test_run(input) {
            Ok(out) => {
                if out.return_value != 0 {
                    warn!(
                        "Failed to update BPF params for CPU {} class {}: {}",
                        cpu, class, out.return_value
                    );
                } else {
                    debug!(
                        "Updated BPF params for CPU {} class {:?}: {:?}",
                        cpu, class, params
                    );
                }
            }
            Err(e) => {
                warn!("Error updating BPF params: {}", e);
            }
        }
    }

    fn run(&mut self, shutdown: Arc<AtomicBool>) -> Result<UserExitInfo> {
        let (res_ch, req_ch) = self.stats_server.channels();

        // Phase 4: PIE Controller update interval from profile
        let update_interval = Duration::from_millis(self.profile.response_ms);
        let mut last_update = Instant::now();

        // Detection state
        let mut detection_counter: u64 = 0;
        let mut current_game: Option<(u32, u32, u8)>;
        let mut current_audio_tgids: Vec<u32>;
        let mut current_state: u32 = 0;

        // Perform initial detection
        current_audio_tgids = self.detect_audio_daemons();
        info!(
            "Detected {} audio daemon(s): {:?}",
            current_audio_tgids.len(),
            current_audio_tgids
        );
        self.update_bpf_audio_tgids(&current_audio_tgids);

        // Initial parameter sync: write default PIE params to BPF
        let nr_cpus = self.nr_cpus;
        for cpu in 0..nr_cpus {
            for class in 0..4u32 {
                let params = self.pie.get_default_params(class);
                self.update_bpf_params(cpu as i32, class, params);
            }
        }
        info!("Initial parameters synced to BPF");

        // Main loop
        while !shutdown.load(Ordering::Relaxed) && !self.exited() {
            if self.refresh_sched_domain() {
                self.user_restart = true;
                break;
            }

            // Periodic detection (every ~5 seconds)
            detection_counter += 1;
            if detection_counter % 500 == 0 {
                // Re-detect audio daemons periodically
                current_audio_tgids = self.detect_audio_daemons();
                self.update_bpf_audio_tgids(&current_audio_tgids);

                // Detect game process
                current_game = self.detect_game_process();
                let game_tgid = current_game.map(|(tgid, _, _)| tgid).unwrap_or(0);
                let game_ppid = current_game.map(|(_, ppid, _)| ppid).unwrap_or(0);
                let game_confidence = current_game.map(|(_, _, conf)| conf).unwrap_or(0);

                // Detect system state
                let new_state = self.detect_sched_state(game_tgid);

                // Update BPF state if changed
                if new_state != current_state || game_tgid != 0 {
                    current_state = new_state;
                    self.update_bpf_game_state(game_tgid, game_ppid, game_confidence);
                    self.update_bpf_sched_state(current_state);

                    /*
                    let state_str = match current_state {
                        2 => "GAMING",
                        1 => "COMPILATION",
                        _ => "IDLE",
                    };

                    if game_tgid != 0 {
                        info!(
                            "State: {} (game TGID={}, PPID={}, confidence={})",
                            state_str, game_tgid, game_ppid, game_confidence
                        );
                    } else {
                        debug!("State: {} (no game detected)", state_str);
                    }*/
                }
            }

            // Phase 4: PIE Controller update cycle
            if last_update.elapsed() >= update_interval {
                // PIE update cycle for all CPUs and classes
                for cpu in 0..self.nr_cpus {
                    for class in 0..4u32 {
                        // Step 1: Read latency metrics from BPF
                        let metrics = self.read_latency_metrics(cpu as i32, class);

                        if metrics.sample_count > 0 {
                            // Calculate average latency
                            let avg_latency_ns = metrics.total_latency_ns / metrics.sample_count;

                            // Step 2: Run PIE controller to get new parameters
                            let new_params = self.pie.update(cpu as u32, class, avg_latency_ns);

                            // Step 3: Write new params to BPF
                            self.update_bpf_params(cpu as i32, class, new_params);
                        }
                    }
                }

                // Optional: Debug output
                if self.opts.debug_pie.is_some() {
                    self.output_pie_debug();
                }

                last_update = Instant::now();
            }

            // Handle stats
            match req_ch.recv_timeout(Duration::from_millis(10)) {
                Ok(()) => res_ch.send(self.get_metrics())?,
                Err(RecvTimeoutError::Timeout) => {}
                Err(e) => Err(e)?,
            }
        }

        let _ = self.struct_ops.take();
        uei_report!(&self.skel, uei)
    }

    /// Output PIE controller debug info
    fn output_pie_debug(&self) {
        let interval_ms = self.opts.debug_pie.unwrap_or(1000);
        static mut LAST_OUTPUT: u64 = 0;

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        unsafe {
            if now - LAST_OUTPUT < interval_ms {
                return;
            }
            LAST_OUTPUT = now;
        }

        eprintln!("\n=== PIE Controller State at {:?} ===", Instant::now());

        // Show stats for representative CPUs
        for cpu in 0..self.nr_cpus.min(4) {
            eprintln!("\nCPU {}:", cpu);

            for class in 0..4u32 {
                if let Some(state) = self.pie.get_state(cpu as u32, class) {
                    let class_name = match class {
                        0 => "LATENCY_CRITICAL",
                        1 => "NORMAL",
                        2 => "HOG",
                        3 => "BACKGROUND",
                        _ => "UNKNOWN",
                    };

                    let error_us =
                        (state.current_latency_ns as i64 - state.target_latency_ns as i64) / 1000;

                    eprintln!(
                        "  Class {} ({}): target={}µs current={}µs error={}µs integral={} updates={}",
                        class, class_name,
                        state.target_latency_ns / 1000,
                        state.current_latency_ns / 1000,
                        error_us,
                        state.integral_accum / 1024,
                        state.update_count
                    );

                    eprintln!(
                        "    Params: lw={} slice={} scale={} preempt={} migrate={}",
                        state.current_params[0],
                        state.current_params[1],
                        state.current_params[2],
                        state.current_params[3],
                        state.current_params[4]
                    );
                }
            }
        }

        let stats = self.pie.get_stats();
        eprintln!(
            "\nGlobal: {} states, {} updates",
            stats.total_states, stats.total_updates
        );
    }
}

impl Drop for Scheduler<'_> {
    fn drop(&mut self) {
        info!("Unregister {SCHEDULER_NAME} scheduler");

        // Restore default CPU idle QoS resume latency.
        if self.opts.idle_resume_us >= 0 {
            if cpu_idle_resume_latency_supported() {
                for cpu in self.topo.all_cpus.values() {
                    update_cpu_idle_resume_latency(cpu.id, cpu.pm_qos_resume_latency_us as i32)
                        .unwrap();
                }
            }
        }
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
            if sched.user_restart {
                continue;
            }
            break;
        }
    }

    Ok(())
}
