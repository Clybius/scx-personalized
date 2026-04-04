// SPDX-License-Identifier: GPL-2.0
//
// Copyright (c) 2024 Andrea Righi <arighi@nvidia.com>

// This software may be used and distributed according to the terms of the
// GNU General Public License version 2.

mod bpf_skel;
pub use bpf_skel::*;
pub mod bpf_intf;
pub use bpf_intf::*;

mod autorate;
mod classifier;
mod optimizer_pie;
mod profiles;
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
use autorate::{AutorateController, AutorateState};
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

SCX_DESCENT_TURBO Environment Variable:
  Processes with SCX_DESCENT_TURBO=1 receive the highest scheduling priority:
  - Automatically classified as LATENCY_CRITICAL
  - Receive half the normal time slice for faster scheduling
  - Get earlier deadlines (higher priority within their class)
  - Non-turbo tasks on SMT siblings are migrated away or deprioritized
  
  Usage: SCX_DESCENT_TURBO=1 ./your_benchmark
  
  This is detected automatically by scanning /proc every 5 seconds.
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
    #[clap(short = 'I', long, allow_hyphen_values = true, default_value = "-1")]
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

    /// Profile selection: gaming, production, server
    ///
    /// Profile defaults:
    ///   gaming:       response=20ms, α=8, β=4,  lat_crit=500µs,  normal=2ms,  hog=10ms,  bg=50ms
    ///   production:   response=20ms, α=8, β=4,  lat_crit=1ms,    normal=5ms,  hog=20ms,  bg=100ms  
    ///   server:       response=50ms, α=16, β=8, lat_crit=2ms,    normal=10ms, hog=50ms,  bg=200ms
    #[clap(short = 'p', long, default_value = "productivity")]
    profile: String,

    /// Enable PIE controller debug output every N ms
    #[clap(long, value_name = "N")]
    debug_pie: Option<u64>,

    /// Enable CAKE Autorate for adaptive parameter tuning
    #[clap(long, action = clap::ArgAction::SetTrue)]
    autorate: bool,

    /// Enable autorate debug output every N ms
    #[clap(long, value_name = "N")]
    debug_autorate: Option<u64>,

    /// Override profile response interval (ms). Lower = faster updates, higher = more stable.
    /// Profile defaults: gaming=20, production=20, server=50
    #[clap(long, value_name = "MS")]
    response_ms: Option<u64>,

    /// Override PIE alpha (proportional gain divisor). Higher = more conservative.
    /// Profile defaults: gaming=8, production=8, server=16
    #[clap(long, value_name = "N")]
    pie_alpha: Option<u64>,

    /// Override PIE beta (integral gain divisor). Higher = slower integral response.
    /// Profile defaults: gaming=4, production=4, server=8
    #[clap(long, value_name = "N")]
    pie_beta: Option<u64>,

    /// Override target latency for LATENCY_CRITICAL class (µs). Audio/games/compositors.
    /// Profile defaults: gaming=500, production=1000, server=2000
    #[clap(long, value_name = "MICROSECONDS")]
    target_latency_critical: Option<u64>,

    /// Override target latency for NORMAL class (µs). Default interactive tasks.
    /// Profile defaults: gaming=2000, production=5000, server=10000
    #[clap(long, value_name = "MICROSECONDS")]
    target_latency_normal: Option<u64>,

    /// Override target latency for HOG class (µs). High CPU usage tasks.
    /// Profile defaults: gaming=10000, production=20000, server=50000
    #[clap(long, value_name = "MICROSECONDS")]
    target_latency_hog: Option<u64>,

    /// Override target latency for BACKGROUND class (µs). Low priority tasks.
    /// Profile defaults: gaming=50000, production=100000, server=200000
    #[clap(long, value_name = "MICROSECONDS")]
    target_latency_background: Option<u64>,

    /// Autorate: high load threshold (0.0-1.0). Load above this triggers ramp up.
    /// Profile defaults: gaming=0.75, production=0.75, server=0.80
    #[clap(long, value_name = "FRACTION")]
    autorate_high_load: Option<f64>,

    /// Autorate: low load threshold (0.0-1.0). Load below this triggers ramp down.
    /// Profile defaults: gaming=0.25, production=0.25, server=0.30
    #[clap(long, value_name = "FRACTION")]
    autorate_low_load: Option<f64>,

    /// Autorate: ramp up rate multiplier (e.g., 1.04 = 4% increase).
    /// Profile defaults: gaming=1.04, production=1.08, server=1.02
    #[clap(long, value_name = "RATE")]
    autorate_ramp_up: Option<f64>,

    /// Autorate: ramp down rate multiplier (e.g., 0.85 = 15% decrease).
    /// Profile defaults: gaming=0.85, production=0.85, server=0.90
    #[clap(long, value_name = "RATE")]
    autorate_ramp_down: Option<f64>,

    /// Autorate: decay rate toward baseline (e.g., 0.99 = 1% per interval).
    /// Profile defaults: gaming=0.99, production=0.99, server=0.995
    #[clap(long, value_name = "RATE")]
    autorate_decay: Option<f64>,

    /// Show detailed profile parameter defaults and exit
    #[clap(long, action = clap::ArgAction::SetTrue, help_heading = "Help")]
    help_profiles: bool,

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

/// Number of task classes
const DESCENT_CLASS_MAX: usize = 4;

/// Print detailed profile help showing all default values
fn print_profile_help() {
    use crate::profiles::Profile;

    println!("scx_descent Profile Parameter Defaults");
    println!("======================================");
    println!();

    let gaming = Profile::gaming();
    let prod = Profile::production();
    let server = Profile::server();

    // Core PIE Parameters
    println!("Core PIE Parameters:");
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "Parameter", "Gaming", "Production", "Server"
    );
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "response_ms (update interval)", gaming.response_ms, prod.response_ms, server.response_ms
    );
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "pie_alpha (proportional)", gaming.pie_alpha, prod.pie_alpha, server.pie_alpha
    );
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "pie_beta (integral)", gaming.pie_beta, prod.pie_beta, server.pie_beta
    );
    println!();

    // Target Latencies
    println!("Target Latencies (microseconds):");
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "Class", "Gaming", "Production", "Server"
    );
    let classes = ["LATENCY_CRITICAL", "NORMAL", "HOG", "BACKGROUND"];
    for (i, class) in classes.iter().enumerate() {
        let g_lat = gaming.target_latencies_ns[i] / 1000;
        let p_lat = prod.target_latencies_ns[i] / 1000;
        let s_lat = server.target_latencies_ns[i] / 1000;
        println!("  {:30} {:>12} {:>15} {:>15}", class, g_lat, p_lat, s_lat);
    }
    println!();

    // Autorate Parameters (only for gaming/production - server has enabled=false)
    println!("CAKE Autorate Parameters:");
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "Parameter", "Gaming", "Production", "Server"
    );
    println!(
        "  {:30} {:>12.2} {:>15.2} {:>15.2}",
        "high_load_threshold",
        gaming.autorate.high_load_threshold,
        prod.autorate.high_load_threshold,
        server.autorate.high_load_threshold
    );
    println!(
        "  {:30} {:>12.2} {:>15.2} {:>15.2}",
        "low_load_threshold",
        gaming.autorate.low_load_threshold,
        prod.autorate.low_load_threshold,
        server.autorate.low_load_threshold
    );
    println!(
        "  {:30} {:>12.2} {:>15.2} {:>15.2}",
        "ramp_up_rate",
        gaming.autorate.ramp_up_rate,
        prod.autorate.ramp_up_rate,
        server.autorate.ramp_up_rate
    );
    println!(
        "  {:30} {:>12.2} {:>15.2} {:>15.2}",
        "ramp_down_rate",
        gaming.autorate.ramp_down_rate,
        prod.autorate.ramp_down_rate,
        server.autorate.ramp_down_rate
    );
    println!(
        "  {:30} {:>12.3} {:>15.3} {:>15.3}",
        "decay_rate",
        gaming.autorate.decay_rate,
        prod.autorate.decay_rate,
        server.autorate.decay_rate
    );
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "refractory_up_ms",
        gaming.autorate.adjust_up_refractory_ms,
        prod.autorate.adjust_up_refractory_ms,
        server.autorate.adjust_up_refractory_ms
    );
    println!(
        "  {:30} {:>12} {:>15} {:>15}",
        "refractory_down_ms",
        gaming.autorate.adjust_down_refractory_ms,
        prod.autorate.adjust_down_refractory_ms,
        server.autorate.adjust_down_refractory_ms
    );
    println!(
        "  {:30} {:>12.1} {:>15.1} {:>15.1}",
        "bufferbloat_threshold",
        gaming.autorate.bufferbloat_threshold,
        prod.autorate.bufferbloat_threshold,
        server.autorate.bufferbloat_threshold
    );
    println!();

    // Profile descriptions
    println!("Profile Descriptions:");
    println!("  gaming:       Optimized for low-latency audio, games, and compositors.");
    println!("                Tight latency targets (500µs for critical), moderate response.");
    println!("  production:   Balanced for general desktop use. Good for mixed workloads.");
    println!("                Moderate latency targets with balanced PIE tuning.");
    println!("  server:       Conservative, throughput-focused. Best for background tasks.");
    println!("                High latency tolerance, gentle parameter adjustments.");
    println!();

    println!("Use --profile <name> to select a base profile, then override specific");
    println!("parameters with --response-ms, --pie-alpha, --target-latency-*, etc.");
}

/// Latency metrics structure for PIE controller
#[derive(Debug, Default)]
pub struct LatencyMetrics {
    pub total_latency_ns: u64,
    pub max_latency_ns: u64,
    pub sample_count: u64,
}

/// Load metrics structure
#[derive(Debug, Default)]
pub struct LoadMetrics {
    pub cycles_spent: u64,
    pub sample_count: u64,
    pub last_update_ns: u64,
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
    autorate: Option<AutorateController>, // NEW: None if --autorate not set
    _classifier: TaskClassifier,
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
        let mut profile = Profile::from_str(&opts.profile).unwrap_or_else(|| Profile::default());

        // Apply CLI overrides to profile parameters
        if let Some(response_ms) = opts.response_ms {
            profile.response_ms = response_ms;
            info!("Override: response_ms = {}ms", response_ms);
        }
        if let Some(alpha) = opts.pie_alpha {
            profile.pie_alpha = alpha;
            info!("Override: pie_alpha = {}", alpha);
        }
        if let Some(beta) = opts.pie_beta {
            profile.pie_beta = beta;
            info!("Override: pie_beta = {}", beta);
        }

        // Apply target latency overrides (convert µs to ns)
        if let Some(latency_us) = opts.target_latency_critical {
            profile.target_latencies_ns[0] = latency_us * 1000;
            info!("Override: target_latency_critical = {}µs", latency_us);
        }
        if let Some(latency_us) = opts.target_latency_normal {
            profile.target_latencies_ns[1] = latency_us * 1000;
            info!("Override: target_latency_normal = {}µs", latency_us);
        }
        if let Some(latency_us) = opts.target_latency_hog {
            profile.target_latencies_ns[2] = latency_us * 1000;
            info!("Override: target_latency_hog = {}µs", latency_us);
        }
        if let Some(latency_us) = opts.target_latency_background {
            profile.target_latencies_ns[3] = latency_us * 1000;
            info!("Override: target_latency_background = {}µs", latency_us);
        }

        // Apply autorate overrides if autorate is enabled
        if opts.autorate {
            profile.autorate.enabled = true; // Enable autorate in profile config
            if let Some(threshold) = opts.autorate_high_load {
                profile.autorate.high_load_threshold = threshold;
                info!("Override: autorate_high_load = {:.2}", threshold);
            }
            if let Some(threshold) = opts.autorate_low_load {
                profile.autorate.low_load_threshold = threshold;
                info!("Override: autorate_low_load = {:.2}", threshold);
            }
            if let Some(rate) = opts.autorate_ramp_up {
                profile.autorate.ramp_up_rate = rate;
                info!("Override: autorate_ramp_up = {:.2}", rate);
            }
            if let Some(rate) = opts.autorate_ramp_down {
                profile.autorate.ramp_down_rate = rate;
                info!("Override: autorate_ramp_down = {:.2}", rate);
            }
            if let Some(rate) = opts.autorate_decay {
                profile.autorate.decay_rate = rate;
                info!("Override: autorate_decay = {:.3}", rate);
            }
        }

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

        // Initialize Phase 4 components - PIE Controller and Autorate
        let pie = PieController::new(nr_cpus, &profile);

        // Initialize autorate if enabled
        let autorate = if opts.autorate {
            info!("CAKE Autorate enabled");
            Some(AutorateController::new(nr_cpus, &profile.autorate))
        } else {
            None
        };

        let classifier = TaskClassifier::new();

        Ok(Self {
            skel,
            struct_ops,
            opts,
            topo,
            power_profile,
            stats_server,
            user_restart: false,
            pie,
            autorate,
            _classifier: classifier,
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

        // Calculate average latency across all states
        let mut total_latency = 0u64;
        let mut latency_count = 0u64;
        let mut total_integral = 0i64;
        let mut total_error = 0i64;
        let mut total_updates = 0u64;

        for cpu in 0..self.nr_cpus as u32 {
            for class in 0..4u32 {
                if let Some(state) = self.pie.get_state(cpu, class) {
                    total_latency += state.current_latency_ns;
                    latency_count += 1;
                    total_integral += state.integral_accum;
                    total_updates += state.update_count;

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

        // Get autorate stats if enabled
        let (autorate_enabled, autorate_state, autorate_rate) =
            if let Some(ref autorate) = self.autorate {
                // Get state from class 0 as representative
                if let Some(state) = autorate.get_class_state(0) {
                    (
                        1u64,
                        state.state as u64,
                        (state.current_rate * 100.0) as u64,
                    )
                } else {
                    (1u64, 0, 50) // Default to steady, mid-rate
                }
            } else {
                (0u64, 0, 0)
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
            pie_updates: total_updates,
            pie_avg_latency_us: avg_latency / 1000, // Convert ns to µs
            pie_target_latency_us: 0,               // TODO: get from profile
            pie_integral: total_integral / 1024,    // De-scale
            pie_latency_error_us: avg_error / 1000, // Convert to µs
            // Autorate metrics
            autorate_enabled,
            autorate_state,
            autorate_rate_percent: autorate_rate,
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

    /// Detect input-related kworker threads by scanning /proc
    ///
    /// Input kworkers are kernel worker threads that handle input device events.
    /// They typically have names like "kworker/0:1-events" or similar.
    fn detect_input_kworkers(&self) -> Vec<u32> {
        let mut input_tgids = HashSet::new();

        // Input-related patterns in kernel thread names
        const INPUT_PATTERNS: &[&str] = &[
            // ksoftirqd threads - handle deferred interrupts including input
            "ksoftirqd/",
            // HID (Human Interface Device) workers
            "hid-",
            // USB input workers
            "usbhid",
            // Input event handlers
            "input_",
            // IRQ workers for input devices
            "irq/",
        ];

        // Scan /proc for kernel threads
        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();
                if let Ok(pid) = pid_str.parse::<u32>() {
                    // Check if this is a kernel thread (parent is kthreadd, PID 2)
                    if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", pid)) {
                        let mut ppid: Option<u32> = None;
                        let mut tgid: Option<u32> = None;

                        for line in status.lines() {
                            if line.starts_with("PPid:") {
                                ppid = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
                            }
                            if line.starts_with("Tgid:") {
                                tgid = line.split_whitespace().nth(1).and_then(|s| s.parse().ok());
                            }
                        }

                        // Check if parent is kthreadd (PID 2) - this is a kernel thread
                        if ppid == Some(2) {
                            if let Ok(comm) = fs::read_to_string(format!("/proc/{}/comm", pid)) {
                                let comm = comm.trim();

                                // Check for input-related patterns
                                if INPUT_PATTERNS
                                    .iter()
                                    .any(|&pattern| comm.starts_with(pattern))
                                {
                                    if let Some(tgid) = tgid {
                                        input_tgids.insert(tgid);
                                        debug!("Detected input kworker: {} (TGID: {})", comm, tgid);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        input_tgids.into_iter().collect()
    }

    /// Detect ksoftirqd threads (handle deferred interrupts)
    ///
    /// ksoftirqd threads are critical for input latency as they handle
    /// the bottom half of interrupt processing for input devices.
    fn detect_ksoftirqd_threads(&self) -> Vec<u32> {
        let mut ksoftirqd_tgids = HashSet::new();

        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();
                if let Ok(pid) = pid_str.parse::<u32>() {
                    if let Ok(comm) = fs::read_to_string(format!("/proc/{}/comm", pid)) {
                        let comm = comm.trim();
                        if comm.starts_with("ksoftirqd/") {
                            // Get TGID
                            if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", pid))
                            {
                                for line in status.lines() {
                                    if line.starts_with("Tgid:") {
                                        if let Some(tgid_str) = line.split_whitespace().nth(1) {
                                            if let Ok(tgid) = tgid_str.parse::<u32>() {
                                                ksoftirqd_tgids.insert(tgid);
                                                debug!(
                                                    "Detected ksoftirqd: {} (TGID: {})",
                                                    comm, tgid
                                                );
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

        ksoftirqd_tgids.into_iter().collect()
    }

    /// Detect processes with SCX_DESCENT_TURBO=1 environment variable
    ///
    /// Scans /proc for processes with the SCX_DESCENT_TURBO environment variable
    /// set to a non-empty, non-zero value. Returns a list of TGIDs that should
    /// receive turbo scheduling priority.
    fn detect_turbo_processes(&self) -> Vec<u32> {
        let mut turbo_tgids = Vec::new();

        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();

                // Only process numeric entries (PIDs)
                if let Ok(pid) = pid_str.parse::<u32>() {
                    // Check /proc/PID/environ for SCX_DESCENT_TURBO
                    if let Ok(environ) = fs::read(format!("/proc/{}/environ", pid)) {
                        let is_turbo = environ
                            .split(|&b| b == 0) // Environment vars are null-delimited
                            .filter_map(|kv| std::str::from_utf8(kv).ok())
                            .any(|s| {
                                // Check for SCX_DESCENT_TURBO= with non-empty, non-zero value
                                if let Some(value) = s.strip_prefix("SCX_DESCENT_TURBO=") {
                                    !value.is_empty() && value != "0"
                                } else {
                                    false
                                }
                            });

                        if is_turbo {
                            // Get TGID from /proc/PID/status
                            if let Some(tgid) = Self::get_tgid_from_status(pid) {
                                // Avoid duplicates
                                if !turbo_tgids.contains(&tgid) {
                                    turbo_tgids.push(tgid);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Limit to 16 entries to match BPF array size
        turbo_tgids.truncate(16);
        turbo_tgids
    }

    /// Helper to read TGID from /proc/PID/status
    ///
    /// The TGID (Thread Group ID) is the PID of the thread group leader.
    /// All threads in a process share the same TGID.
    fn get_tgid_from_status(pid: u32) -> Option<u32> {
        if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", pid)) {
            for line in status.lines() {
                if line.starts_with("Tgid:") {
                    return line.split_whitespace().nth(1).and_then(|t| t.parse().ok());
                }
            }
        }
        None
    }

    /// Detect Desktop Environment component TGIDs by scanning comm names
    ///
    /// DE components are promoted to LATENCY_CRITICAL during non-GAMING states
    /// to ensure desktop responsiveness (UI, panel, notifications, etc.)
    fn detect_de_components(&self) -> Vec<u32> {
        let mut de_tgids = HashSet::new();

        const DE_COMMS: &[&str] = &[
            // GNOME
            "gnome-shell",
            "gnome-panel",
            // KDE Plasma
            "plasmashell",
            "kwin_wayland",
            "kwin_x11",
            "plasma-desktop",
            // Sway
            "sway",
            "swaybar",
            // Hyprland
            "Hyprland",
            // XFCE
            "xfce4-panel",
            "xfwm4",
            "xfdesktop",
            // LXQt
            "lxqt-panel",
            "pcmanfm-qt",
            // MATE
            "marco",
            "mate-panel",
            // Cinnamon
            "cinnamon",
            "muffin",
            // i3/sway family
            "i3",
            "i3bar",
            // Wayfire
            "wayfire",
            // Weston
            "weston",
            // Gamescope (Steam Deck UI)
            "gamescope",
            // Budgie
            "budgie-panel",
            "budgie-wm",
            // Deepin
            "dde-desktop",
            "dde-panel",
            // Pantheon (elementary)
            "gala",
            "wingpanel",
            // Common file managers (for desktop icons)
            "nautilus", // GNOME Files
            "dolphin",  // KDE Files
            "thunar",   // XFCE Files
            "pcmanfm",  // LXDE/LXQt Files
            "caja",     // MATE Files
            "nemo",     // Cinnamon Files
        ];

        // Scan /proc for matching comm names
        if let Ok(entries) = fs::read_dir("/proc") {
            for entry in entries.filter_map(|e| e.ok()) {
                let file_name = entry.file_name();
                let pid_str = file_name.to_string_lossy();
                if let Ok(pid) = pid_str.parse::<u32>() {
                    if let Ok(comm) = fs::read_to_string(format!("/proc/{}/comm", pid)) {
                        let comm = comm.trim();
                        if DE_COMMS.iter().any(|&de| comm == de) {
                            // Get TGID from status
                            if let Ok(status) = fs::read_to_string(format!("/proc/{}/status", pid))
                            {
                                for line in status.lines() {
                                    if line.starts_with("Tgid:") {
                                        if let Some(tgid_str) = line.split_whitespace().nth(1) {
                                            if let Ok(tgid) = tgid_str.parse::<u32>() {
                                                de_tgids.insert(tgid);
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

        de_tgids.into_iter().collect()
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

    /// Update BPF turbo process tracking
    ///
    /// Writes the list of turbo TGIDs to the BPF BSS section, enabling
    /// the BPF scheduler to identify and prioritize turbo tasks.
    fn update_bpf_turbo_tgids(&mut self, turbo_tgids: &[u32]) {
        if let Some(bss_data) = self.skel.maps.bss_data.as_mut() {
            let nr_turbo = turbo_tgids.len().min(16);
            bss_data.nr_turbo_tgids = nr_turbo as u32;

            for (i, &tgid) in turbo_tgids.iter().take(16).enumerate() {
                bss_data.turbo_tgids[i] = tgid;
            }

            // Clear remaining slots
            for i in nr_turbo..16 {
                bss_data.turbo_tgids[i] = 0;
            }
        }
    }

    /// Update BPF Desktop Environment component tracking
    ///
    /// Writes the list of DE TGIDs to the BPF BSS section, enabling
    /// the BPF scheduler to identify and prioritize DE tasks during non-GAMING states.
    fn update_bpf_de_tgids(&mut self, de_tgids: &[u32]) {
        if let Some(bss_data) = self.skel.maps.bss_data.as_mut() {
            let nr_de = de_tgids.len().min(16);
            bss_data.nr_de_tgids = nr_de as u32;
            bss_data.de_detected = if nr_de > 0 { 1 } else { 0 };

            for (i, &tgid) in de_tgids.iter().take(16).enumerate() {
                bss_data.de_tgids[i] = tgid;
            }

            // Clear remaining slots
            for i in nr_de..16 {
                bss_data.de_tgids[i] = 0;
            }
        }
    }

    /// Update BPF input kworker tracking
    ///
    /// Writes the list of input kworker TGIDs to the BPF BSS section,
    /// enabling the BPF scheduler to identify and prioritize input tasks.
    fn update_bpf_input_kworkers(&mut self, input_tgids: &[u32]) {
        if let Some(bss_data) = self.skel.maps.bss_data.as_mut() {
            let nr_input = input_tgids.len().min(16);
            bss_data.nr_input_kworker_tgids = nr_input as u32;

            for (i, &tgid) in input_tgids.iter().take(16).enumerate() {
                bss_data.input_kworker_tgids[i] = tgid;
            }

            // Clear remaining slots
            for i in nr_input..16 {
                bss_data.input_kworker_tgids[i] = 0;
            }

            if nr_input > 0 {
                debug!(
                    "Updated BPF with {} input kworker(s): {:?}",
                    nr_input, input_tgids
                );
            }
        }
    }

    /// Update BPF ksoftirqd tracking
    fn update_bpf_ksoftirqd(&mut self, ksoftirqd_tgids: &[u32]) {
        if let Some(bss_data) = self.skel.maps.bss_data.as_mut() {
            let nr_ksoftirqd = ksoftirqd_tgids.len().min(16);
            bss_data.nr_ksoftirqd_tgids = nr_ksoftirqd as u32;

            for (i, &tgid) in ksoftirqd_tgids.iter().take(16).enumerate() {
                bss_data.ksoftirqd_tgids[i] = tgid;
            }

            // Clear remaining slots
            for i in nr_ksoftirqd..16 {
                bss_data.ksoftirqd_tgids[i] = 0;
            }

            if nr_ksoftirqd > 0 {
                debug!(
                    "Updated BPF with {} ksoftirqd thread(s): {:?}",
                    nr_ksoftirqd, ksoftirqd_tgids
                );
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

    /// Read load metrics from BPF for a CPU/class
    fn read_load_metrics(&self, cpu: i32, class: u32) -> LoadMetrics {
        if class as usize >= DESCENT_CLASS_MAX {
            return LoadMetrics::default();
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
                    return LoadMetrics::default();
                }
                let data = &values[cpu_idx];

                // Calculate offset to class_load[class_id]
                // class_params[4] = 160 bytes
                // class_latency[4] = 96 bytes (24 bytes each)
                // class_load starts after = 256 bytes
                let class_offset = 256 + (class as usize * 24); // 24 bytes per load_accumulator

                if data.len() < class_offset + 24 {
                    return LoadMetrics::default();
                }

                let load_data = &data[class_offset..class_offset + 24];

                LoadMetrics {
                    cycles_spent: read_u64(&load_data[0..8]),
                    sample_count: read_u64(&load_data[8..16]),
                    last_update_ns: read_u64(&load_data[16..24]),
                }
            }
            Ok(None) => LoadMetrics::default(),
            Err(_) => LoadMetrics::default(),
        }
    }

    /// Reset load accumulators for a class across all CPUs via BPF syscall
    fn reset_load_accumulators(&mut self, class: u32) {
        let prog = &mut self.skel.progs.reset_load_accumulators;

        let mut args = reset_load_args {
            cpu_id: -1, // All CPUs
            class_id: class,
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
                        "Failed to reset load accumulators for class {}: {}",
                        class, out.return_value
                    );
                } else {
                    debug!("Reset load accumulators for class {}", class);
                }
            }
            Err(e) => {
                warn!("Error resetting load accumulators: {}", e);
            }
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
        let mut last_turbo_tgids: Vec<u32> = Vec::new();

        // Perform initial detection
        current_audio_tgids = self.detect_audio_daemons();
        info!(
            "Detected {} audio daemon(s): {:?}",
            current_audio_tgids.len(),
            current_audio_tgids
        );
        self.update_bpf_audio_tgids(&current_audio_tgids);

        // Initial DE detection
        let de_tgids = self.detect_de_components();
        if !de_tgids.is_empty() {
            info!(
                "Detected {} DE component(s): {:?}",
                de_tgids.len(),
                de_tgids
            );
        }
        self.update_bpf_de_tgids(&de_tgids);

        // Initial input detection (NEW)
        let input_kworker_tgids = self.detect_input_kworkers();
        if !input_kworker_tgids.is_empty() {
            info!(
                "Detected {} input kworker(s): {:?}",
                input_kworker_tgids.len(),
                input_kworker_tgids
            );
        } else {
            info!("No input kworkers detected (may appear later)");
        }
        self.update_bpf_input_kworkers(&input_kworker_tgids);

        // Initial ksoftirqd detection (NEW)
        let ksoftirqd_tgids = self.detect_ksoftirqd_threads();
        if !ksoftirqd_tgids.is_empty() {
            info!(
                "Detected {} ksoftirqd thread(s): {:?}",
                ksoftirqd_tgids.len(),
                ksoftirqd_tgids
            );
        }
        self.update_bpf_ksoftirqd(&ksoftirqd_tgids);

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

                // Detect DE components
                let de_tgids = self.detect_de_components();
                if !de_tgids.is_empty() {
                    debug!(
                        "Detected {} DE component(s): {:?}",
                        de_tgids.len(),
                        de_tgids
                    );
                }
                self.update_bpf_de_tgids(&de_tgids);

                // Detect input kworkers (NEW)
                let input_kworker_tgids = self.detect_input_kworkers();
                if !input_kworker_tgids.is_empty() {
                    debug!(
                        "Detected {} input kworker(s): {:?}",
                        input_kworker_tgids.len(),
                        input_kworker_tgids
                    );
                }
                self.update_bpf_input_kworkers(&input_kworker_tgids);

                // Detect ksoftirqd threads (NEW)
                let ksoftirqd_tgids = self.detect_ksoftirqd_threads();
                if !ksoftirqd_tgids.is_empty() {
                    debug!(
                        "Detected {} ksoftirqd thread(s): {:?}",
                        ksoftirqd_tgids.len(),
                        ksoftirqd_tgids
                    );
                }
                self.update_bpf_ksoftirqd(&ksoftirqd_tgids);

                // Detect turbo processes
                let turbo_tgids = self.detect_turbo_processes();

                // Log changes in turbo mode status
                if turbo_tgids != last_turbo_tgids {
                    if !turbo_tgids.is_empty() {
                        info!(
                            "Turbo mode active for {} process(es): {:?}",
                            turbo_tgids.len(),
                            turbo_tgids
                        );
                    } else if !last_turbo_tgids.is_empty() {
                        info!("Turbo mode deactivated");
                    }
                    last_turbo_tgids = turbo_tgids.clone();
                }

                self.update_bpf_turbo_tgids(&turbo_tgids);

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
                if self.opts.autorate {
                    // TWO-STAGE: Autorate + PIE
                    self.update_autorate_and_pie();
                } else {
                    // PIE ONLY (original behavior)
                    self.update_pie_only();
                }

                // Optional: Debug output
                if self.opts.debug_pie.is_some() {
                    self.output_pie_debug();
                }

                // Optional: Autorate debug output
                if self.opts.debug_autorate.is_some() {
                    self.output_autorate_debug();
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

        // Calculate global stats manually
        let mut total_states = 0usize;
        let mut total_updates = 0u64;
        for cpu in 0..self.nr_cpus {
            for class in 0..4u32 {
                if let Some(state) = self.pie.get_state(cpu as u32, class) {
                    total_states += 1;
                    total_updates += state.update_count;
                }
            }
        }
        eprintln!(
            "\nGlobal: {} states, {} updates",
            total_states, total_updates
        );
    }

    /// Stage 1: Autorate (per-class) + Stage 2: PIE (per-CPU)
    fn update_autorate_and_pie(&mut self) {
        // ─────────────────────────────────────────────────────────
        // STAGE 1: Aggregate metrics and run Autorate per-class
        // ─────────────────────────────────────────────────────────

        let mut class_metrics: [(u64, u64, u64); 4] = [(0, 0, 0); 4];
        // (total_latency_ns, latency_count, total_cycles)

        // Aggregate across all CPUs per class
        for cpu in 0..self.nr_cpus {
            for class in 0..4u32 {
                let latency = self.read_latency_metrics(cpu as i32, class);
                let load = self.read_load_metrics(cpu as i32, class);

                class_metrics[class as usize].0 += latency.total_latency_ns;
                class_metrics[class as usize].1 += latency.sample_count;
                class_metrics[class as usize].2 += load.cycles_spent;
            }
        }

        // Run Autorate per-class
        let mut class_base_params: [[u64; 5]; 4] = [[0; 5]; 4];
        let mut class_states: [Option<AutorateState>; 4] = [None; 4];

        if let Some(ref mut autorate) = self.autorate {
            for class in 0..4u32 {
                let (total_latency, total_count, total_cycles) = class_metrics[class as usize];

                // Process if we have either latency data OR load data
                if total_count > 0 || total_cycles > 0 {
                    // Calculate avg latency if we have samples, otherwise use target as default
                    let avg_latency_ns = if total_count > 0 {
                        total_latency / total_count
                    } else {
                        0 // Will be replaced with target below
                    };

                    // Calculate load % - using per-CPU capacity for meaningful values
                    // load = (total_cycles / nr_cpus) / interval_ns
                    // This gives average per-CPU utilization (0.0-1.0)
                    // where 0.0 = 0% load and 1.0 = 100% load (full CPU utilization)
                    let interval_ns = self.profile.response_ms * 1_000_000;

                    let load_percent = if interval_ns > 0 && self.nr_cpus > 0 {
                        // Average cycles per CPU
                        let avg_cycles_per_cpu = total_cycles / (self.nr_cpus as u64);

                        // Compare against interval (single CPU capacity)
                        // This gives a proper ratio between 0.0 (0% load) and 1.0 (100% load)
                        let raw_load = (avg_cycles_per_cpu as f64) / (interval_ns as f64);

                        // Clamp to valid percentage range (0% - 100%)
                        raw_load.clamp(0.0, 1.0)
                    } else {
                        0.0
                    };

                    let target_latency = self.profile.target_latencies_ns[class as usize];

                    // Use target latency as fallback if no measured latency
                    let effective_latency_ns = if avg_latency_ns > 0 {
                        avg_latency_ns
                    } else {
                        target_latency
                    };

                    // Run Autorate
                    let (base_params, state) =
                        autorate.update(class, effective_latency_ns, target_latency, load_percent);

                    class_base_params[class as usize] = base_params;
                    class_states[class as usize] = Some(state);
                } else {
                    // No data - use baseline
                    class_base_params[class as usize] =
                        self.profile.autorate.baseline_params[class as usize];
                }
            }
        }

        // Reset load accumulators for all classes after reading
        for class in 0..4u32 {
            self.reset_load_accumulators(class);
        }

        // ─────────────────────────────────────────────────────────
        // STAGE 2: PIE fine-tuning per-(CPU, class)
        // ─────────────────────────────────────────────────────────

        for cpu in 0..self.nr_cpus {
            for class in 0..4u32 {
                let latency = self.read_latency_metrics(cpu as i32, class);

                if latency.sample_count > 0 {
                    let local_avg_latency_ns = latency.total_latency_ns / latency.sample_count;
                    let base_params = class_base_params[class as usize];

                    // PIE fine-tunes base_params for this specific CPU
                    let final_params = self.pie.update_with_base_params(
                        cpu as u32,
                        class,
                        local_avg_latency_ns,
                        base_params,
                    );

                    // Write to BPF
                    self.update_bpf_params(cpu as i32, class, final_params);
                }
            }
        }
    }

    /// PIE-only update (when --autorate not specified)
    fn update_pie_only(&mut self) {
        for cpu in 0..self.nr_cpus {
            for class in 0..4u32 {
                let metrics = self.read_latency_metrics(cpu as i32, class);

                if metrics.sample_count > 0 {
                    let avg_latency_ns = metrics.total_latency_ns / metrics.sample_count;
                    let new_params = self.pie.update(cpu as u32, class, avg_latency_ns);
                    self.update_bpf_params(cpu as i32, class, new_params);
                }
            }
        }
    }

    /// Output Autorate controller debug info
    fn output_autorate_debug(&self) {
        let interval_ms = self.opts.debug_autorate.unwrap_or(1000);
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

        eprintln!("\n=== CAKE Autorate State at {:?} ===", Instant::now());

        if let Some(ref autorate) = self.autorate {
            let stats = autorate.get_stats();
            eprintln!(
                "Global: {} classes, {} up, {} down, {} blocked, avg_rate={:.2}",
                stats.total_classes,
                stats.adjustments_up,
                stats.adjustments_down,
                stats.adjustments_blocked,
                stats.avg_rate
            );

            for class in 0..4u32 {
                if let Some(state) = autorate.get_class_state(class) {
                    let class_name = match class {
                        0 => "LATENCY_CRITICAL",
                        1 => "NORMAL",
                        2 => "HOG",
                        3 => "BACKGROUND",
                        _ => "UNKNOWN",
                    };

                    eprintln!(
                        "  Class {} ({}): state={} rate={:.3} load={:.4} ({:.2}%)",
                        class,
                        class_name,
                        state.state.name(),
                        state.current_rate,
                        state.load_percent,
                        state.load_percent * 100.0
                    );
                }
            }

            // Show raw load metrics from BPF for diagnosis
            eprintln!("\n  Raw load metrics from BPF:");
            for class in 0..4u32 {
                let mut total_cycles = 0u64;
                let mut total_samples = 0u64;
                for cpu in 0..self.nr_cpus.min(8) {
                    let load = self.read_load_metrics(cpu as i32, class);
                    total_cycles += load.cycles_spent;
                    total_samples += load.sample_count;
                }
                let class_name = match class {
                    0 => "LATENCY_CRITICAL",
                    1 => "NORMAL",
                    2 => "HOG",
                    3 => "BACKGROUND",
                    _ => "UNKNOWN",
                };
                eprintln!(
                    "    Class {} ({}): cycles={} samples={}",
                    class, class_name, total_cycles, total_samples
                );
            }
        } else {
            eprintln!("  Autorate not enabled");
        }
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

    if opts.help_profiles {
        print_profile_help();
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
